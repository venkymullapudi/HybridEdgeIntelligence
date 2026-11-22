"""Design review question 46: prompt caching on escalations."""

from types import SimpleNamespace

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator, estimate_cost

from .conftest import FakeClaude

S, U, A = (lambda c: ChatMessage("system", c)), (lambda c: ChatMessage("user", c)), (lambda c: ChatMessage("assistant", c))


async def test_single_turn_caches_only_the_system_prompt(settings):
    fake = FakeClaude()
    await ClaudeEscalator(settings, client=fake).answer([S("You are a tax assistant."), U("q")])
    call = fake.calls[0]
    assert call["system"] == [{"type": "text", "text": "You are a tax assistant.",
                               "cache_control": {"type": "ephemeral"}}]
    assert "cache_control" not in call  # a unique final question isn't worth a write premium


async def test_multi_turn_also_caches_the_history(settings):
    fake = FakeClaude()
    await ClaudeEscalator(settings, client=fake).answer([S("s"), U("q1"), A("a1"), U("q2")])
    assert fake.calls[0]["cache_control"] == {"type": "ephemeral"}


async def test_caching_can_be_turned_off(settings):
    settings.cloud_prompt_caching = False
    fake = FakeClaude()
    await ClaudeEscalator(settings, client=fake).answer([S("s"), U("q1"), A("a1"), U("q2")])
    assert fake.calls[0]["system"] == "s" and "cache_control" not in fake.calls[0]


def test_cost_prices_cache_reads_and_writes(settings):
    # 1k uncached in, 100 out, 9k read from cache, 2k written.
    got = estimate_cost(settings, 1000, 100, cache_read_tokens=9000, cache_write_tokens=2000)
    want = (1000 * 4.0 + 100 * 20.0 + 9000 * 0.20 + 2000 * 5.0) / 1e6
    assert abs(got - want) < 1e-12
    # Reading 9k from cache is ~20x cheaper than sending it uncached.
    assert 9000 * settings.cloud_price_cache_read_per_mtok * 20 == 9000 * settings.cloud_price_in_per_mtok


async def test_cache_usage_reaches_cost_and_response(settings):
    fake = FakeClaude()
    inner = fake.beta.messages.create

    async def create(**kw):
        resp = await inner(**kw)
        resp.usage = SimpleNamespace(input_tokens=50, output_tokens=20, cache_read_input_tokens=8000,
                                     cache_creation_input_tokens=0, iterations=None)
        return resp

    fake.beta.messages.create = create
    ans = await ClaudeEscalator(settings, client=fake).answer([S("s"), U("q")])
    assert ans.cache_read_tokens == 8000
    assert abs(ans.cost_usd - estimate_cost(settings, 50, 20, 8000, 0)) < 1e-12
