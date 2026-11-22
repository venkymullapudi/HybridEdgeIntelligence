import anthropic
import httpx
import pytest

from hybrid_edge.backends.base import ChatMessage, EdgeBackendError
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.orchestrator import ServiceUnavailable
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def user(text):
    return [ChatMessage("user", text)]


async def test_confident_prompt_served_at_edge(orch, fake_claude):
    res = await orch.infer(user("What is the capital of France?"))
    assert res.source == "edge"
    assert not res.escalated
    assert res.confidence >= res.threshold
    assert fake_claude.calls == []


async def test_low_confidence_escalates_to_claude(orch, fake_claude):
    res = await orch.infer(user("Prove the Riemann hypothesis, this is hard"))
    assert res.source == "cloud" and res.escalated
    assert res.answer == "Cloud answer."
    assert res.confidence < res.threshold
    call = fake_claude.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert res.usage["cloud_output_tokens"] == 22
    rec = orch.records.records[-1]
    assert rec["escalated"] and rec["edge_answer"] and rec["cloud_answer"] == "Cloud answer."
    assert set(rec["features"]) >= {"mean_prob", "min_window_prob", "entropy_conf"}


async def test_hedged_answer_escalates(orch):
    res = await orch.infer(user("I am unsure what is the answer"))
    assert res.escalated and "edge answer hedges" in res.reasons


async def test_cloud_failure_degrades_to_edge_answer(settings):
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude(error=err)),
                           records=MemorySink(settings))
    res = await o.infer(user("this is a hard legal question"))
    assert res.source == "edge" and res.escalated and res.degraded
    assert any("cloud unavailable" in r for r in res.reasons)


async def test_circuit_breaker_opens_after_failures(settings):
    settings.cloud_breaker_failures = 2
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    fake = FakeClaude(error=err)
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    for _ in range(3):
        await o.infer(user("hard question"), use_cache=False)
    assert len(fake.calls) == 2  # third request short-circuited
    assert o.escalator.breaker.open


async def test_escalation_budget(settings):
    settings.cloud_max_escalations_per_min = 1
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r1 = await o.infer(user("hard one"), use_cache=False)
    r2 = await o.infer(user("hard two"), use_cache=False)
    assert r1.source == "cloud"
    assert r2.degraded and any("budget" in r for r in r2.reasons)


async def test_route_overrides(orch, fake_claude):
    r = await orch.infer(user("hard question"), route="edge")
    assert r.source == "edge" and not r.escalated
    r = await orch.infer(user("easy question"), route="cloud")
    assert r.source == "cloud" and "client requested cloud" in r.reasons


async def test_min_confidence_override(orch):
    r = await orch.infer(user("easy question"), min_confidence=0.999)
    assert r.escalated


async def test_cache_hit_for_escalated_answer(orch, fake_claude):
    await orch.infer(user("hard question"))
    r = await orch.infer(user("  hard question\n"))
    assert r.source == "cache"
    assert len(fake_claude.calls) == 1


async def test_refusal_is_reported_and_not_cached(settings):
    fake = FakeClaude(text="", stop_reason="refusal")
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer(user("hard question"))
    assert r.refused and r.answer
    await o.infer(user("hard question"))
    assert len(fake.calls) == 2


async def test_long_prompt_skips_edge(orch, fake_claude):
    orch.s.edge_context_tokens = 100
    r = await orch.infer(user("x" * 4000))
    assert r.source == "cloud" and "prompt exceeds edge context window" in r.reasons
    assert r.confidence is None


async def test_edge_down_escalates_and_both_down_raises(settings):
    class Broken:
        name = "broken"

        async def generate(self, *a, **k):
            raise EdgeBackendError("gpu on fire")

        async def generate_stream(self, *a, **k):
            raise EdgeBackendError("gpu on fire")

        async def ready(self):
            return False

        async def aclose(self):
            return None

    fake = FakeClaude()
    o = build_orchestrator(settings, backend=Broken(), escalator=ClaudeEscalator(settings, client=fake),
                           records=MemorySink(settings))
    r = await o.infer(user("anything"))
    assert r.source == "cloud"

    settings.cloud_provider = "none"
    o2 = build_orchestrator(settings, backend=Broken(), escalator=ClaudeEscalator(settings),
                            records=MemorySink(settings))
    with pytest.raises(ServiceUnavailable):
        await o2.infer(user("anything"))


async def test_consistency_sampling_lowers_score_on_disagreement(settings, fake_claude):
    settings.gate_consistency_samples = 3
    settings.gate_consistency_band = 1.0  # always in the gray zone
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake_claude),
                           records=MemorySink(settings))
    r = await o.infer(user("hard"))
    assert r.signals["consistency_measured"] and r.signals["consistency"] < 1.0


async def test_bedrock_model_prefix(settings):
    settings.cloud_provider = "bedrock"
    fake = FakeClaude()
    esc = ClaudeEscalator(settings, client=fake)
    await esc.answer(user("hi"))
    assert fake.calls[0]["model"] == "anthropic.claude-opus-5-5"
    assert "fallbacks" not in fake.calls[0]  # handled by client-side middleware on Bedrock


async def test_edge_draft_included_when_enabled(settings):
    settings.cloud_include_edge_draft = True
    fake = FakeClaude()
    esc = ClaudeEscalator(settings, client=fake)
    await esc.answer([ChatMessage("system", "sys"), ChatMessage("user", "q")], edge_draft="draft!")
    call = fake.calls[0]
    assert call["system"][0]["text"] == "sys"
    assert "draft!" in call["messages"][-1]["content"]
