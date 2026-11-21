"""Design review questions 15-17."""

import asyncio

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.calibrate import choose_threshold, fit
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.gate.confidence import Calibrator
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def user(text):
    return [ChatMessage("user", text)]


class SlowClaude(FakeClaude):
    def __init__(self, delay):
        super().__init__()
        inner = self.beta.messages.create

        async def create(**kw):
            await asyncio.sleep(delay)
            return await inner(**kw)

        self.beta.messages.create = create


# --- Q15: a disconnecting leader must not fail coalesced waiters ----------------
async def test_leader_cancellation_does_not_cancel_waiters(settings):
    fake = SlowClaude(0.1)
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    leader = asyncio.ensure_future(o.infer(user("hard question")))
    await asyncio.sleep(0.01)
    waiters = [asyncio.ensure_future(o.infer(user("hard question"))) for _ in range(3)]
    await asyncio.sleep(0.01)
    leader.cancel()  # client of the first request hung up
    results = await asyncio.gather(*waiters)
    assert all(r.answer == "Cloud answer." for r in results)
    assert len(fake.calls) == 1
    # The shared work finished and was cached even though its first caller left.
    r = await o.infer(user("hard question"))
    assert r.source == "cache"


async def test_failed_shared_work_propagates_to_all_waiters(settings):
    settings.cloud_provider = "none"

    class Broken:
        name = "broken"

        async def generate_stream(self, *a, **k):
            await asyncio.sleep(0.05)
            from hybrid_edge.backends.base import EdgeBackendError
            raise EdgeBackendError("down")

        async def ready(self):
            return False

        async def aclose(self):
            return None

    o = build_orchestrator(settings, backend=Broken(), escalator=ClaudeEscalator(settings),
                           records=MemorySink(settings))
    res = await asyncio.gather(*[o.infer(user("q")) for _ in range(3)], return_exceptions=True)
    assert all(type(r).__name__ == "ServiceUnavailable" for r in res)
    assert o._inflight == {}


# --- Q16: one deadline over the whole escalation ------------------------------
async def test_cloud_deadline_degrades_and_counts_as_failure(settings):
    settings.cloud_deadline_s = 0.05
    fake = SlowClaude(1.0)
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer(user("hard question"))
    assert r.degraded and r.source == "edge"
    assert any("deadline" in x for x in r.reasons)
    assert o.escalator.breaker.failures == 1


# --- Q17: calibration must weight shadow samples ------------------------------
def test_threshold_selection_uses_sample_weights():
    cal = Calibrator(weights={"mean_prob": 10.0}, bias=-5.0)
    confident = [({"mean_prob": 0.9 + i * 0.005}, 1, 10.0) for i in range(9)] + [({"mean_prob": 0.92}, 0, 10.0)]
    shaky = [({"mean_prob": 0.2 + i * 0.01}, 0, 1.0) for i in range(10)]
    rows = confident + shaky
    _, weighted = choose_threshold(cal, rows, 0.9)
    _, unweighted = choose_threshold(cal, [(x, y) for x, y, _ in rows], 0.9)
    # 10 shadow rows stand for 100 confident requests: the edge serves ~91% of
    # traffic, not the 50% the raw sample suggests.
    assert abs(weighted["edge_coverage"] - 100 / 110) < 1e-9
    assert abs(unweighted["edge_coverage"] - 0.5) < 1e-9


async def test_shadow_weight_values(settings):
    settings.shadow_rate = 0.25
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    import random
    random.seed(0)
    for i in range(40):
        await o.infer(user(f"easy {i}"))
    await o.infer(user("hard question"))
    await o.drain()
    shadow = [r for r in o.records.records if r["shadow"]]
    escalated = [r for r in o.records.records if r["escalated"]]
    assert shadow and all(r["sample_weight"] == 4.0 for r in shadow)
    assert escalated and all(r["sample_weight"] == 1.0 for r in escalated)


def test_fit_accepts_weighted_and_unweighted_rows():
    rows = [({"mean_prob": 0.9}, 1), ({"mean_prob": 0.2}, 0)] * 10
    a = fit(rows, epochs=50)
    b = fit([(x, y, 1.0) for x, y in rows], epochs=50)
    assert a.weights == b.weights
