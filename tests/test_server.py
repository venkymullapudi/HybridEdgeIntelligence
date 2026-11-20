from fastapi.testclient import TestClient

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


def make_client(settings):
    orch = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                              records=MemorySink(settings))
    return TestClient(create_app(settings, orch))


def test_infer_endpoint_edge_and_cloud(settings):
    with make_client(settings) as c:
        r = c.post("/v1/infer", json={"prompt": "capital of France?"})
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "edge" and body["signals"] is None

        r = c.post("/v1/infer", json={"messages": [{"role": "user", "content": "hard proof"}], "include_signals": True})
        body = r.json()
        assert body["source"] == "cloud" and body["signals"]["mean_prob"] < 0.6


def test_validation_and_health(settings):
    with make_client(settings) as c:
        assert c.post("/v1/infer", json={}).status_code == 422
        assert c.post("/v1/infer", json={"prompt": "x", "route": "moon"}).status_code == 422
        assert c.get("/healthz").json()["status"] == "ok"
        assert c.get("/readyz").json()["status"] == "ready"
        c.post("/v1/infer", json={"prompt": "hello"})
        m = c.get("/metrics").text
        assert "hybrid_edge_requests_total" in m and "hybrid_edge_escalation_rate" in m


def test_api_key_auth(settings):
    settings.api_keys = ["s3cret"]
    with make_client(settings) as c:
        assert c.post("/v1/infer", json={"prompt": "hi"}).status_code == 401
        assert c.post("/v1/infer", json={"prompt": "hi"}, headers={"x-api-key": "s3cret"}).status_code == 200
        assert c.post("/v1/infer", json={"prompt": "hi"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
