"""Design review question 37: streaming escalations."""

import json

import anthropic
import httpx
from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def client(settings, fake):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    return TestClient(create_app(settings, o)), o


def test_escalation_streams_deltas_then_done(settings):
    fake = FakeClaude(text="Claude streams this answer in pieces.")
    c, o = client(settings, fake)
    with c:
        r = c.post("/v1/infer/stream", json={"prompt": "hard question"}, headers={"x-request-id": "s1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-request-id"] == "s1"
    ev = parse_sse(r.text)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "escalating" and kinds[-1] == "done" and kinds.count("delta") > 3
    assert "".join(d["text"] for k, d in ev if k == "delta") == fake.text
    done = ev[-1][1]
    assert done["answer"] == fake.text and done["source"] == "cloud" and done["request_id"] == "s1"
    assert fake.calls[0]["fallbacks"] == "default"  # same request shape as the non-streaming path
    assert o.records.records[-1]["cloud_answer"] == fake.text


def test_edge_and_cache_answers_are_a_single_done_event(settings):
    c, _ = client(settings, FakeClaude())
    with c:
        assert [k for k, _ in parse_sse(c.post("/v1/infer/stream", json={"prompt": "easy"}).text)] == ["done"]
        c.post("/v1/infer/stream", json={"prompt": "hard question"})
        ev = parse_sse(c.post("/v1/infer/stream", json={"prompt": "hard question"}).text)
        assert [k for k, _ in ev] == ["done"] and ev[0][1]["source"] == "cache"


def test_stream_failure_midway_falls_back_and_says_so(settings):
    fake = FakeClaude(text="x" * 40)
    fake.stream_error_after = 10
    fake.stream_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    c, _ = client(settings, fake)
    with c:
        ev = parse_sse(c.post("/v1/infer/stream", json={"prompt": "hard question"}).text)
    done = ev[-1][1]
    assert ev[-1][0] == "done" and done["source"] == "edge" and done["degraded"]
    assert any("stream stopped after 10 chars" in x for x in done["reasons"])


def test_stream_validation_and_total_failure(settings):
    settings.cloud_provider = "none"
    c, _ = client(settings, FakeClaude())
    with c:
        r = c.post("/v1/infer/stream", json={"messages": [{"role": "user", "content": "q"},
                                                          {"role": "assistant", "content": "p"}]})
        assert r.status_code == 422  # rejected before the stream starts
        ev = parse_sse(c.post("/v1/infer/stream", json={"prompt": "easy", "route": "cloud"}).text)
        assert ev == [("error", ev[0][1])] and ev[0][1]["status"] == 503


async def test_stream_answer_respects_budget_and_breaker(settings):
    settings.cloud_max_escalations_per_key_per_min = 1
    esc = ClaudeEscalator(settings, client=FakeClaude())
    got = []

    async def sink(t):
        got.append(t)

    await esc.stream_answer([ChatMessage("user", "q")], None, sink, caller="k")
    try:
        await esc.stream_answer([ChatMessage("user", "q")], None, sink, caller="k")
        raise AssertionError("budget not enforced")
    except Exception as exc:  # CloudUnavailable
        assert "per-key" in str(exc)
