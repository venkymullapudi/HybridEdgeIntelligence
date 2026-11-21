"""Design review questions 38-39: streaming heartbeats and time to first token."""

import asyncio

from fastapi.testclient import TestClient

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink
from .test_streaming import parse_sse


class ThinkingClaude(FakeClaude):
    """Pauses before any text, like adaptive thinking does."""

    def __init__(self, delay, **kw):
        super().__init__(**kw)
        inner = self.beta.messages.create

        async def create(**k):
            await asyncio.sleep(delay)
            return await inner(**k)

        self.beta.messages.create = create


def client(settings, fake):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    return TestClient(create_app(settings, o)), o


def test_idle_stream_sends_heartbeats(settings):
    settings.sse_heartbeat_s = 0.05
    c, _ = client(settings, ThinkingClaude(0.3, text="Final answer."))
    with c:
        body = c.post("/v1/infer/stream", json={"prompt": "hard question"}).text
    assert body.count(": ping\n\n") >= 2
    events = parse_sse("\n\n".join(b for b in body.split("\n\n") if b.strip() and not b.startswith(":")))
    assert events[-1][0] == "done" and events[-1][1]["answer"] == "Final answer."


def test_first_token_latency_reported_and_measured(settings):
    c, o = client(settings, ThinkingClaude(0.1, text="Streamed."))
    with c:
        done = parse_sse(c.post("/v1/infer/stream", json={"prompt": "hard question"}).text)[-1][1]
        plain = c.post("/v1/infer", json={"prompt": "hard question", "use_cache": False}).json()
        metrics = c.get("/metrics").text
    assert done["cloud_first_token_ms"] >= 100
    assert done["cloud_first_token_ms"] <= done["cloud_latency_ms"] + 1
    assert plain["cloud_first_token_ms"] is None  # only streamed escalations have one
    assert "hybrid_edge_cloud_first_token_ms_count" in metrics and 'le="+Inf"} 1' in metrics
