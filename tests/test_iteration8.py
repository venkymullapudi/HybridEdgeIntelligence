"""Design review question 29: empty or truncated Claude answers."""

import pytest

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.orchestrator import ServiceUnavailable
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def user(t):
    return [ChatMessage("user", t)]


async def test_empty_cloud_answer_falls_back_to_edge_and_is_not_cached(settings):
    fake = FakeClaude(text="", stop_reason="max_tokens")  # thinking used the whole budget
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer(user("hard question"))
    assert r.source == "edge" and r.degraded and r.answer
    assert any("no answer text" in x for x in r.reasons)
    await o.infer(user("hard question"))
    assert len(fake.calls) == 2  # nothing was cached
    assert o.escalator.breaker.state == "closed"  # not an outage


async def test_empty_cloud_answer_with_no_edge_answer_is_503(settings):
    fake = FakeClaude(text="  ")
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with pytest.raises(ServiceUnavailable):
        await o.infer(user("easy"), route="cloud")


async def test_truncated_cloud_answer_is_flagged_and_not_cached(settings):
    fake = FakeClaude(text="The first half of an answ", stop_reason="max_tokens")
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer(user("hard question"))
    assert r.source == "cloud" and any("truncated at cloud_max_tokens" in x for x in r.reasons)
    await o.infer(user("hard question"))
    assert len(fake.calls) == 2


async def test_refusal_with_empty_text_still_reports_refusal(settings):
    fake = FakeClaude(text="", stop_reason="refusal")
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer(user("hard question"))
    assert r.refused and r.source == "cloud"
