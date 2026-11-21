"""Tests for the v0.2 design-review fixes (docs/DESIGN_REVIEW.md)."""

import asyncio
import json
import time

import anthropic
import httpx

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.backends.openai_compat import StreamAccumulator
from hybrid_edge.cloud.claude import CircuitBreaker, ClaudeEscalator, estimate_cost
from hybrid_edge.orchestrator import caller_id
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def user(text):
    return [ChatMessage("user", text)]


def make(settings, fake=None, backend=None):
    fake = fake or FakeClaude()
    o = build_orchestrator(settings, backend=backend, escalator=ClaudeEscalator(settings, client=fake),
                           records=MemorySink(settings))
    return o, fake


# --- Q6: circuit breaker half-open ---------------------------------------------
def test_breaker_half_open_allows_exactly_one_probe():
    br = CircuitBreaker(failures=1, cooldown_s=0.0)
    br.failure()
    assert br.state == "half_open"
    assert br.allow()        # the probe
    assert not br.allow()    # everyone else waits
    assert br.open
    br.failure()             # probe failed -> re-open
    assert br.opened_at is not None
    br2 = CircuitBreaker(failures=1, cooldown_s=0.0)
    br2.failure()
    assert br2.allow()
    br2.success()
    assert br2.state == "closed" and br2.allow() and br2.allow()


def test_breaker_stays_open_during_cooldown():
    br = CircuitBreaker(failures=2, cooldown_s=60)
    br.failure()
    assert br.allow()
    br.failure()
    assert br.state == "open" and not br.allow()


# --- Q5: single-flight ---------------------------------------------------------
async def test_concurrent_identical_requests_make_one_cloud_call(settings):
    class SlowClaude(FakeClaude):
        def __init__(self):
            super().__init__()
            inner = self.beta.messages.create

            async def create(**kw):
                await asyncio.sleep(0.05)
                return await inner(**kw)

            self.beta.messages.create = create

    o, fake = make(settings, SlowClaude())
    results = await asyncio.gather(*[o.infer(user("hard question")) for _ in range(10)])
    assert len(fake.calls) == 1
    assert sum(r.source == "cloud" for r in results) == 1
    assert all(r.answer == "Cloud answer." for r in results)
    assert any("coalesced" in r.reasons[0] for r in results if r.source == "cache")


# --- Q8: tenant-scoped cache ---------------------------------------------------
async def test_cache_is_scoped_per_caller(settings):
    o, fake = make(settings)
    await o.infer(user("hard question"), caller="tenant-a")
    r = await o.infer(user("hard question"), caller="tenant-b")
    assert r.source == "cloud" and len(fake.calls) == 2
    r = await o.infer(user("hard question"), caller="tenant-a")
    assert r.source == "cache"


async def test_cache_can_be_shared_when_enabled(settings):
    settings.cache_shared_across_keys = True
    o, fake = make(settings)
    await o.infer(user("hard question"), caller="tenant-a")
    r = await o.infer(user("hard question"), caller="tenant-b")
    assert r.source == "cache" and len(fake.calls) == 1


def test_caller_id_is_not_the_key():
    assert caller_id(None) == "anon"
    cid = caller_id("s3cret")
    assert "s3cret" not in cid and cid == caller_id("s3cret") and len(cid) == 16


# --- Q7: per-key budget and client policy --------------------------------------
async def test_per_key_budget_isolates_noisy_client(settings):
    settings.cloud_max_escalations_per_key_per_min = 1
    o, fake = make(settings)
    await o.infer(user("hard one"), caller="noisy", use_cache=False)
    r = await o.infer(user("hard two"), caller="noisy", use_cache=False)
    assert r.degraded and any("per-key" in x for x in r.reasons)
    r = await o.infer(user("hard three"), caller="quiet", use_cache=False)
    assert r.source == "cloud"


async def test_client_overrides_can_be_disabled(settings):
    settings.allow_client_overrides = False
    o, fake = make(settings)
    r = await o.infer(user("easy question"), route="cloud", min_confidence=0.99)
    assert r.source == "edge" and "client overrides disabled by policy" in r.reasons
    assert fake.calls == []


async def test_min_confidence_is_capped(settings):
    settings.max_client_min_confidence = 0.8
    o, _ = make(settings)
    r = await o.infer(user("easy question"), min_confidence=1.0)
    assert r.threshold == 0.8 and r.source == "edge"


# --- Q3: deadline and early abort ----------------------------------------------
async def test_slow_edge_escalates_at_deadline(settings):
    settings.edge_deadline_s = 0.05
    o, fake = make(settings, backend=MockBackend(slow_s=1.0))
    t0 = time.perf_counter()
    r = await o.infer(user("slow but easy"))
    assert time.perf_counter() - t0 < 0.5
    assert r.source == "cloud" and any("deadline" in x for x in r.reasons)


async def test_early_abort_stops_doomed_generation(settings):
    o, fake = make(settings)
    r = await o.infer(user("write an essay"))
    assert r.source == "cloud"
    assert r.signals["aborted"]
    assert r.usage["edge_completion_tokens"] < 50  # stopped well before the 200-token answer
    assert any("aborted early" in x for x in r.reasons)


async def test_no_early_abort_when_cloud_cannot_take_it(settings):
    settings.cloud_provider = "none"
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings), records=MemorySink(settings))
    r = await o.infer(user("write an essay"))
    assert not r.signals["aborted"] and r.degraded
    assert r.usage["edge_completion_tokens"] > 100


# --- Q4: gray-zone sampling ----------------------------------------------------
async def test_consistency_samples_skipped_outside_gray_zone(settings):
    settings.gate_consistency_samples = 3
    backend = MockBackend()
    o, _ = make(settings, backend=backend)
    r = await o.infer(user("what is the capital of france"))  # clear accept
    assert not r.signals["consistency_measured"] and backend.calls == 1


# --- Q1: shadow escalation -----------------------------------------------------
async def test_shadow_escalation_records_reference_for_confident_answer(settings):
    settings.shadow_rate = 1.0
    o, fake = make(settings)
    r = await o.infer(user("what is the capital of france"))
    assert r.source == "edge" and not r.escalated  # user got the edge answer
    await o.drain()
    assert len(fake.calls) == 1
    shadow = [x for x in o.records.records if x["shadow"]]
    assert shadow and shadow[0]["cloud_answer"] == "Cloud answer." and shadow[0]["edge_answer"]


async def test_shadow_respects_its_own_budget(settings):
    settings.shadow_rate = 1.0
    settings.shadow_max_per_min = 1
    o, fake = make(settings)
    for q in ("easy a", "easy b", "easy c"):
        await o.infer(user(q))
    await o.drain()
    assert len(fake.calls) == 1


# --- Q11: cost accounting ------------------------------------------------------
async def test_cost_and_avoided_cost(settings):
    o, _ = make(settings)
    r = await o.infer(user("hard question"))
    assert r.usage["cloud_cost_usd"] == round(estimate_cost(settings, 11, 22), 6)
    r = await o.infer(user("easy question"))
    assert r.usage["avoided_cost_usd"] > 0
    m = o.metrics.prometheus()
    assert "hybrid_edge_cloud_cost_usd_total" in m and "hybrid_edge_avoided_cost_usd_total" in m


# --- streaming parser ----------------------------------------------------------
def test_stream_accumulator_parses_vllm_sse():
    acc = StreamAccumulator()
    lines = [
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Par"}, "logprobs": {"content": [
            {"token": "Par", "logprob": -0.1, "top_logprobs": [{"token": "Par", "logprob": -0.1}]}]}}]}),
        '',
        'data: ' + json.dumps({"choices": [{"delta": {"content": "is"}, "finish_reason": "stop", "logprobs": {
            "content": [{"token": "is", "logprob": -0.2, "top_logprobs": []}]}}]}),
        'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}),
        'data: [DONE]',
    ]
    done = False
    for line in lines:
        done = acc.feed_sse_line(line)
    g = acc.result(1.0)
    assert done and g.text == "Paris" and len(g.tokens) == 2 and g.prompt_tokens == 9 and g.finish_reason == "stop"


async def test_openai_backend_streams_and_aborts():
    from hybrid_edge.backends.openai_compat import OpenAICompatBackend

    def chunk(tok, lp):
        return "data: " + json.dumps({"choices": [{"delta": {"content": tok + " "}, "logprobs": {"content": [
            {"token": tok, "logprob": lp, "top_logprobs": []}]}}]}) + "\n\n"

    body = "".join(chunk(f"t{i}", -3.0) for i in range(100)) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    b = OpenAICompatBackend("http://edge/v1", "m")
    b._client = httpx.AsyncClient(base_url="http://edge/v1", transport=httpx.MockTransport(handler))
    g = await b.generate_stream(user("x"), max_tokens=100, temperature=0.2,
                                should_abort=lambda toks: len(toks) >= 20)
    assert g.finish_reason == "aborted" and len(g.tokens) == 20
    await b.aclose()


async def test_rate_limit_trips_breaker(settings):
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
                                   body=None)
    settings.cloud_breaker_failures = 1
    o, fake = make(settings, FakeClaude(error=err))
    r = await o.infer(user("hard question"), use_cache=False)
    assert r.degraded and o.escalator.breaker.state == "open"
