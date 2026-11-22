"""Design review question 67: a client's max_tokens bounds Claude too."""

import json

from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


async def test_client_max_tokens_reaches_claude_with_thinking_headroom(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    await o.infer([ChatMessage("user", "hard question")], max_tokens=200)
    assert fake.calls[-1]["max_tokens"] == 200 + 8000


async def test_without_a_client_limit_claude_gets_the_configured_cap(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    await o.infer([ChatMessage("user", "hard question")])
    assert fake.calls[-1]["max_tokens"] == settings.cloud_max_tokens


async def test_cloud_max_tokens_is_still_the_ceiling(settings):
    settings.cloud_max_tokens = 5000
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    await o.infer([ChatMessage("user", "hard question")], max_tokens=4000)
    assert fake.calls[-1]["max_tokens"] == 5000


def test_streaming_escalations_honour_it_too(settings):
    settings.cloud_thinking_headroom_tokens = 1000
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        c.post("/v1/infer/stream", json={"prompt": "hard question", "max_tokens": 64})
    assert fake.calls[-1]["max_tokens"] == 64 + 1000
