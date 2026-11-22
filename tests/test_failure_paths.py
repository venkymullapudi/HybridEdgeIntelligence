"""Design review question 69: failure paths that no test had ever executed."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, EdgeBackendError
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.backends.openai_compat import OpenAICompatBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.orchestrator import ServiceUnavailable
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink

OK = {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop", "logprobs": {"content": [
    {"token": "Paris", "logprob": -0.01, "top_logprobs": [{"token": "Paris", "logprob": -0.01}]}]}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


def http_backend(handler):
    b = OpenAICompatBackend("http://edge/v1", "m")
    b._client = httpx.AsyncClient(base_url="http://edge/v1", transport=httpx.MockTransport(handler))
    return b


def refused(req):
    raise httpx.ConnectError("connection refused", request=req)


# --- real HTTP failures from the edge server, both transports -------------------
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("handler,needle", [
    (lambda req: httpx.Response(503, text="model loading"), "HTTP 503"),
    (lambda req: httpx.Response(500, text="CUDA error: out of memory"), "HTTP 500"),
    (refused, "unreachable"),
])
async def test_edge_http_failures_become_edge_errors(stream, handler, needle):
    b = http_backend(handler)
    with pytest.raises(EdgeBackendError, match=needle):
        if stream:
            await b.generate_stream([ChatMessage("user", "q")], max_tokens=8, temperature=0.2)
        else:
            await b.generate([ChatMessage("user", "q")], max_tokens=8, temperature=0.2)
    await b.aclose()


@pytest.mark.parametrize("stream", [True, False])
async def test_edge_outage_escalates_to_claude(settings, stream):
    settings.edge_stream = stream
    fake = FakeClaude()
    o = build_orchestrator(settings, backend=http_backend(lambda req: httpx.Response(503, text="loading")),
                           escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "capital of France?")])
    assert r.source == "cloud" and any("HTTP 503" in x for x in r.reasons) and len(fake.calls) == 1
    assert o.metrics.counters["edge_errors_total"] == 1


# --- the non-streaming edge path (HEI_EDGE_STREAM=false) ------------------------
async def test_non_streaming_edge_path_serves_and_gates(settings):
    settings.edge_stream = False
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=OK)

    o = build_orchestrator(settings, backend=http_backend(handler), escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "capital of France?")])
    assert r.source == "edge" and r.answer == "Paris" and r.confidence > 0.9
    assert "stream" not in bodies[0] and bodies[0]["logprobs"] is True


# --- route=edge with the edge down: 503, never a silent escalation ---------------
async def test_forced_edge_route_with_edge_down_is_unavailable(settings):
    fake = FakeClaude()
    o = build_orchestrator(settings, backend=http_backend(refused), escalator=ClaudeEscalator(settings, client=fake),
                           records=MemorySink(settings))
    with pytest.raises(ServiceUnavailable, match="unreachable"):
        await o.infer([ChatMessage("user", "q")], route="edge")
    assert fake.calls == []  # the client asked for the edge only
    assert o.metrics.counters["served_unavailable_total"] == 1


# --- /v1/infer answers 503 (with request id) when both tiers are down -----------
def test_plain_endpoint_returns_503_with_request_id(settings):
    settings.cloud_provider = "none"
    o = build_orchestrator(settings, backend=http_backend(refused), escalator=ClaudeEscalator(settings),
                           records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        r = c.post("/v1/infer", json={"prompt": "q"}, headers={"x-request-id": "out-1"})
    assert r.status_code == 503 and r.headers["x-request-id"] == "out-1"
    assert "unreachable" in r.json()["detail"]


# --- consistency sampling failing doesn't lose the primary answer --------------
async def test_failed_consistency_sampling_keeps_the_primary_decision(settings):
    settings.gate_consistency_samples = 3
    settings.gate_consistency_band = 1.0

    class SamplesFail(MockBackend):
        async def generate(self, *a, **k):
            raise EdgeBackendError("sampler overloaded")

    o = build_orchestrator(settings, backend=SamplesFail(), escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "capital of France?")])
    assert r.source == "edge" and r.signals["consistency_measured"] is False


# --- refusal fallback is reported ----------------------------------------------
async def test_fallback_model_is_reported(settings):
    fake = FakeClaude()
    inner = fake.beta.messages.create

    async def create(**kw):
        resp = await inner(**kw)
        resp.content = [SimpleNamespace(type="fallback"), SimpleNamespace(type="text", text="Rescued answer.")]
        resp.model = "claude-opus-4-8"
        return resp

    fake.beta.messages.create = create
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "hard question")])
    assert r.answer == "Rescued answer." and r.model == "claude-opus-4-8"
    assert "served by fallback model claude-opus-4-8" in r.reasons
