"""Design review questions 35-36."""

import asyncio

from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, edge_messages
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


# --- Q35: edge models without a system role -----------------------------------
def test_edge_messages_merge():
    msgs = [ChatMessage("system", "Be brief."), ChatMessage("user", "q1"), ChatMessage("assistant", "a1"),
            ChatMessage("user", "q2")]
    assert edge_messages(msgs, "native") is msgs
    merged = edge_messages(msgs, "merge")
    assert [m.role for m in merged] == ["user", "assistant", "user"]
    assert merged[0].content == "Be brief.\n\nq1" and merged[2].content == "q2"
    assert edge_messages([ChatMessage("user", "q")], "merge")[0].content == "q"


async def test_merge_mode_reaches_edge_but_claude_keeps_system(settings):
    settings.edge_system_role = "merge"

    class Recording(MockBackend):
        async def generate_stream(self, messages, **kw):
            self.roles = [m.role for m in messages]
            return await super().generate_stream(messages, **kw)

    backend, fake = Recording(), FakeClaude()
    o = build_orchestrator(settings, backend=backend, escalator=ClaudeEscalator(settings, client=fake),
                           records=MemorySink(settings))
    await o.infer([ChatMessage("system", "Answer in French."), ChatMessage("user", "hard question")])
    assert backend.roles == ["user"]                     # no system role sent to the edge
    assert fake.calls[0]["system"][0]["text"] == "Answer in French."  # Claude still gets a real system prompt


# --- Q36: request IDs ---------------------------------------------------------
def _client(settings):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    return TestClient(create_app(settings, o)), o


def test_request_id_is_echoed_or_generated(settings):
    c, o = _client(settings)
    with c:
        r = c.post("/v1/infer", json={"prompt": "hard question"}, headers={"x-request-id": "trace-123"})
        assert r.headers["x-request-id"] == "trace-123" and r.json()["request_id"] == "trace-123"
        assert o.records.records[-1]["request_id"] == "trace-123"
        r = c.post("/v1/infer", json={"prompt": "hi"}, headers={"x-request-id": "bad id\nInjected: 1"})
        rid = r.headers["x-request-id"]
        assert rid != "bad id\nInjected: 1" and len(rid) == 32 and r.json()["request_id"] == rid
        r = c.post("/v1/infer", json={"messages": [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": "p"}]},
                   headers={"x-request-id": "err-1"})
        assert r.status_code == 422 and r.headers["x-request-id"] == "err-1"


async def test_coalesced_and_cached_requests_keep_their_own_id(settings):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    q = [ChatMessage("user", "hard question")]
    a, b = await asyncio.gather(o.infer(q, request_id="a"), o.infer(q, request_id="b"))
    c = await o.infer(q, request_id="c")
    assert (a.request_id, b.request_id, c.request_id) == ("a", "b", "c")
    assert c.source == "cache"
