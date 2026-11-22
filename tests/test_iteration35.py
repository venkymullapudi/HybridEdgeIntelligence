"""Design review question 68: a client-chosen length isn't a truncation."""

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink

Q = [ChatMessage("user", "What is the capital of France? Answer briefly")]  # mock: long, confident answer


def make(settings):
    fake = FakeClaude()
    return build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake),
                              records=MemorySink(settings)), fake


async def test_answer_stopping_at_client_max_tokens_is_not_penalised(settings):
    o, fake = make(settings)
    full = await o.infer(Q, use_cache=False)
    short = await o.infer(Q, max_tokens=4, use_cache=False)
    assert short.source == "edge" and fake.calls == []
    assert short.signals["truncated"] is False
    assert abs(short.confidence - full.confidence) < 0.05  # no 0.15 truncation penalty
    assert "edge answer truncated" not in short.reasons


async def test_our_own_limits_still_count_as_truncation(settings):
    settings.edge_max_tokens = 4  # the operator's default limit, not the client's
    o, _ = make(settings)
    r = await o.infer(Q, use_cache=False)
    assert r.signals["truncated"] is True and "edge answer truncated" in r.reasons


async def test_context_clamped_client_limit_still_counts_as_truncation(settings):
    settings.edge_context_tokens = 1000
    o, _ = make(settings)
    seen = []
    real = o._edge

    async def spy(messages, max_tokens, threshold, route, caller, client_capped=False):
        seen.append((max_tokens, client_capped))
        return await real(messages, max_tokens, threshold, route, caller, client_capped)

    o._edge = spy
    await o.infer([ChatMessage("user", "x" * 2400)], max_tokens=2000, use_cache=False)  # ~600 tokens of prompt
    await o.infer(Q, max_tokens=50, use_cache=False)
    (clamped, capped_1), (asked, capped_2) = seen
    assert clamped < 2000 and capped_1 is False   # we shrank it: a cut-off is ours, penalise it
    assert asked == 50 and capped_2 is True       # the client's own limit: no penalty
