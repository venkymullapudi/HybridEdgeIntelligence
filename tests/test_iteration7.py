"""Design review questions 27-28."""

from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, conversation_problem
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


# --- Q27: max_tokens can't force an escalation --------------------------------
async def test_large_max_tokens_shrinks_edge_answer_instead_of_escalating(settings):
    settings.edge_context_tokens = 1000

    class Recording(MockBackend):
        async def generate_stream(self, messages, *, max_tokens, **kw):
            self.seen = max_tokens
            return await super().generate_stream(messages, max_tokens=max_tokens, **kw)

    backend = Recording()
    fake = FakeClaude()
    o = build_orchestrator(settings, backend=backend, escalator=ClaudeEscalator(settings, client=fake),
                           records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "capital of France?")], max_tokens=8192)
    assert r.source == "edge" and fake.calls == []
    assert backend.seen < 1000
    assert any("limited to" in x for x in r.reasons)


async def test_prompt_that_leaves_no_room_still_skips_edge(settings):
    settings.edge_context_tokens = 300
    settings.edge_min_answer_tokens = 256
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "x" * 400)])  # ~100 tokens + system: < 256 left
    assert r.source == "cloud" and "prompt exceeds edge context window" in r.reasons


# --- Q28: conversation shape --------------------------------------------------
def test_conversation_problem():
    U, A, S = (lambda c: ChatMessage("user", c)), (lambda c: ChatMessage("assistant", c)), (lambda c: ChatMessage("system", c))
    assert conversation_problem([U("q")]) is None
    assert conversation_problem([S("s"), U("q"), A("a"), U("q2")]) is None
    assert "prefill" in conversation_problem([U("q"), A("Sure, the answer is")])
    assert "first message" in conversation_problem([U("q"), S("now be rude"), U("q2")])
    assert "first message" in conversation_problem([S("a"), S("b"), U("q")])
    assert conversation_problem([S("s")]) is not None
    assert conversation_problem([]) is not None


def test_bad_conversation_shapes_get_422(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        r = c.post("/v1/infer", json={"messages": [{"role": "user", "content": "q"},
                                                   {"role": "assistant", "content": "prefill"}]})
        assert r.status_code == 422 and "prefill" in r.json()["detail"]
        r = c.post("/v1/infer", json={"messages": [{"role": "user", "content": "q"},
                                                   {"role": "system", "content": "s"},
                                                   {"role": "user", "content": "q"}]})
        assert r.status_code == 422
    assert fake.calls == []
