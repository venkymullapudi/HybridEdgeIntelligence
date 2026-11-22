"""Design review question 64: concurrency invariants and outage visibility.

A seeded stress run of overlapping requests (single-flight, cache, streaming,
shadow, cancellation, a flaky cloud) checking invariants, plus a check that total
outages show up in metrics.
"""

import asyncio
import random

import anthropic
import httpx
import pytest

from hybrid_edge.backends.base import ChatMessage, EdgeBackendError
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.orchestrator import ServiceUnavailable
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


class FlakyClaude(FakeClaude):
    def __init__(self, r):
        super().__init__()
        inner = self.beta.messages.create

        async def create(**kw):
            await asyncio.sleep(r.uniform(0, 0.01))
            if r.random() < 0.2:
                raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
            return await inner(**kw)

        self.beta.messages.create = create


@pytest.mark.parametrize("seed", range(3))
async def test_concurrent_mixed_traffic_keeps_invariants(seed):
    r = random.Random(seed)
    s = Settings.from_dict({"edge_backend": "mock", "emit_emf": False, "edge_deadline_s": 1, "shadow_rate": 0.3,
                            "cloud_breaker_failures": 1000, "cloud_max_escalations_per_min": 0,
                            "cloud_max_escalations_per_key_per_min": 0})
    o = build_orchestrator(s, backend=MockBackend(slow_s=0.01), escalator=ClaudeEscalator(s, client=FlakyClaude(r)),
                           records=MemorySink(s))
    prompts = ["hard q", "easy q", "slow hard q", "unsure q", "write an essay", "slow easy"]
    problems, done, cancelled = [], [0], [0]

    async def one(i):
        thr = r.choice([None, 0.3, 0.7, 0.95])
        route = r.choice(["auto"] * 6 + ["edge", "cloud"])
        kw = dict(min_confidence=thr, caller=r.choice(["a", "b"]), route=route, use_cache=r.random() < .8,
                  request_id=f"r{i}")
        if r.random() < .3:
            async def sink(e):
                return None
            kw["on_event"] = sink
        try:
            res = await o.infer([ChatMessage("user", r.choice(prompts))], **kw)
        except ServiceUnavailable:
            done[0] += 1
            return
        except asyncio.CancelledError:
            cancelled[0] += 1
            raise
        except Exception as e:  # noqa: BLE001 - any other exception is a bug
            problems.append(f"{type(e).__name__}: {e}")
            return
        done[0] += 1
        eff = o.gate.threshold if thr is None else thr
        if res.request_id != f"r{i}":
            problems.append(f"request id {res.request_id} != r{i}")
        if res.source == "edge" and not res.degraded and route == "auto" and (res.confidence or 0) < eff:
            problems.append(f"edge answer at {res.confidence:.2f} below its caller's threshold {eff}")

    tasks = []
    for i in range(150):
        tasks.append(asyncio.ensure_future(one(i)))
        if r.random() < .5:
            await asyncio.sleep(0)
        if r.random() < .1:
            tasks[r.randrange(len(tasks))].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await o.drain()

    assert problems == []
    assert o._inflight == {} and not o._background
    total = o.metrics.counters.get("requests_total", 0)
    # Every completed request is counted; a cancelled caller is counted only if the
    # shared work it joined still finished.
    assert done[0] <= total <= done[0] + cancelled[0]


async def test_total_outage_is_visible_in_metrics():
    class Down:
        name = "down"

        async def generate_stream(self, *a, **k):
            raise EdgeBackendError("gpu down")

        async def ready(self):
            return False

        async def aclose(self):
            return None

    s = Settings.from_dict({"edge_backend": "mock", "cloud_provider": "none", "emit_emf": False})
    o = build_orchestrator(s, backend=Down(), escalator=ClaudeEscalator(s), records=MemorySink(s))
    for i in range(30):
        with pytest.raises(ServiceUnavailable):
            await o.infer([ChatMessage("user", f"q{i}")])
    c = o.metrics.counters
    assert c["requests_total"] == 30 and c["degraded_total"] == 30 and c["served_unavailable_total"] == 30


def test_outage_reaches_the_emf_alarm_inputs(capsys):
    import json
    from hybrid_edge.telemetry import Metrics
    Metrics(Settings.from_dict({"region": "eu-west-1"})).observe("unavailable", False, True, None, 12.0)
    doc = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert doc["Requests"] == 1 and doc["Degraded"] == 1 and doc["Source"] == "unavailable"
