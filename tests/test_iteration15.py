"""Design review question 42: coalescing must respect each caller's threshold."""

import asyncio

import anthropic
import httpx

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink

Q = [ChatMessage("user", "slow hard question")]  # mock: low confidence (~0.32), takes slow_s


def make(settings, fake):
    return build_orchestrator(settings, backend=MockBackend(slow_s=0.1),
                              escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))


async def test_stricter_waiter_does_not_get_leaders_weaker_answer(settings):
    fake = FakeClaude()
    o = make(settings, fake)
    leader = asyncio.ensure_future(o.infer(Q, min_confidence=0.3))   # accepts the shaky edge answer
    await asyncio.sleep(0.01)
    waiter = asyncio.ensure_future(o.infer(Q))                       # default threshold 0.7
    a, b = await asyncio.gather(leader, waiter)
    assert a.source == "edge" and a.confidence < 0.7
    assert b.source == "cloud" and b.answer == "Cloud answer."      # escalated on its own terms
    assert len(fake.calls) == 1


async def test_looser_waiter_shares_leaders_answer(settings):
    fake = FakeClaude()
    o = make(settings, fake)
    leader = asyncio.ensure_future(o.infer(Q))                       # escalates
    await asyncio.sleep(0.01)
    waiter = asyncio.ensure_future(o.infer(Q, min_confidence=0.3))
    a, b = await asyncio.gather(leader, waiter)
    assert a.source == "cloud" and b.source == "cache" and len(fake.calls) == 1


async def test_shared_degraded_answer_stays_marked_degraded(settings):
    settings.cloud_breaker_failures = 100
    fake = FakeClaude(error=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")))
    o = make(settings, fake)
    leader = asyncio.ensure_future(o.infer(Q))
    await asyncio.sleep(0.01)
    waiter = asyncio.ensure_future(o.infer(Q))
    a, b = await asyncio.gather(leader, waiter)
    assert a.degraded and b.degraded                                  # not silently "healthy"
    assert any("cloud unavailable" in r for r in b.reasons)
    assert len(fake.calls) == 1                                       # no retry storm into a failing cloud
