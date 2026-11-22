"""Design review question 47: empty content never reaches Claude."""

from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, conversation_problem
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


async def test_empty_system_prompt_is_omitted_not_sent_empty(settings):
    for caching in (True, False):
        settings.cloud_prompt_caching = caching
        fake = FakeClaude()
        await ClaudeEscalator(settings, client=fake).answer([ChatMessage("system", "  "), ChatMessage("user", "q")])
        assert "system" not in fake.calls[0]


async def test_empty_default_system_prompt_is_omitted(settings):
    settings.system_prompt = ""
    fake = FakeClaude()
    await ClaudeEscalator(settings, client=fake).answer([ChatMessage("user", "q")])
    assert "system" not in fake.calls[0]


def test_empty_messages_are_a_conversation_problem():
    assert "empty" in conversation_problem([ChatMessage("user", "   ")])
    assert "message 2 (assistant)" in conversation_problem(
        [ChatMessage("system", ""), ChatMessage("user", "q"), ChatMessage("assistant", ""), ChatMessage("user", "q2")])
    assert conversation_problem([ChatMessage("system", ""), ChatMessage("user", "q")]) is None  # empty system is fine


def test_empty_prompt_gets_422_before_any_model(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        assert c.post("/v1/infer", json={"prompt": ""}).status_code == 422
        assert c.post("/v1/infer/stream", json={"prompt": " \n"}).status_code == 422
    assert fake.calls == []
