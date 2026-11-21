"""Design review questions 21-23."""

import json

import pytest
from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, Generation
from hybrid_edge.calibrate import main as calibrate_main
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.gate.confidence import Calibrator
from hybrid_edge.server import build_orchestrator, create_app
from hybrid_edge.telemetry import Metrics

from .conftest import FakeClaude, MemorySink
from .test_misc import _synthetic


# --- Q21: calibration bound to the edge model ---------------------------------
def _write_records(path, model, n=200):
    with open(path, "a") as fh:
        for feats, y in _synthetic(n):
            fh.write(json.dumps({"edge_model": model, "features": feats, "label": y}) + "\n")


def test_calibration_records_edge_model_and_rejects_mixed(tmp_path, capsys):
    rec = tmp_path / "r.jsonl"
    _write_records(rec, "model-a")
    out = tmp_path / "cal.json"
    calibrate_main([str(rec), "-o", str(out)])
    assert json.loads(out.read_text())["edge_model"] == "model-a"

    _write_records(rec, "model-b")
    with pytest.raises(SystemExit, match="mix edge models"):
        calibrate_main([str(rec), "-o", str(out)])
    calibrate_main([str(rec), "-o", str(out), "--edge-model", "model-b"])
    assert json.loads(out.read_text())["edge_model"] == "model-b"


def test_startup_refuses_mismatched_calibration(tmp_path, settings):
    path = tmp_path / "cal.json"
    Calibrator(weights={"mean_prob": 1.0}, bias=0.0, threshold=0.5, edge_model="old-model").dump(str(path))
    settings.gate_calibration_path = str(path)
    with pytest.raises(ValueError, match="old-model"):
        build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()))
    settings.gate_allow_calibration_mismatch = True
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()))
    assert o.gate.threshold == 0.5

    settings.gate_allow_calibration_mismatch = False
    Calibrator(weights={"mean_prob": 1.0}, bias=0.0, threshold=0.5,
               edge_model=settings.edge_model).dump(str(path))
    build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()))


# --- Q22: /metrics behind the same auth ---------------------------------------
def test_metrics_requires_api_key_when_configured(settings):
    settings.api_keys = ["k"]
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"x-api-key": "k"}).status_code == 200
        assert c.get("/healthz").status_code == 200  # health checks stay open
        assert c.get("/readyz").status_code == 200


# --- Q23: missing logprobs are loud; EMF is well-formed -------------------------
async def test_missing_logprobs_counted(settings):
    class NoLogprobs:
        name = "nolp"

        async def generate_stream(self, *a, **k):
            return Generation(text="an answer", tokens=[], completion_tokens=2)

        async def ready(self):
            return True

        async def aclose(self):
            return None

    o = build_orchestrator(settings, backend=NoLogprobs(), escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    for q in ("a", "b"):
        r = await o.infer([ChatMessage("user", q)])
        assert "no logprobs from edge model" in r.reasons
    assert o.metrics.counters["edge_no_logprobs_total"] == 2


def test_emf_document_is_valid(capsys):
    s = Settings.from_dict({"region": "eu-west-1", "emit_emf": True})
    Metrics(s).observe("cloud", True, False, 0.4, 1234.5, cloud_tokens=10, cloud_cost=0.01, avoided_cost=0.0)
    doc = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    spec = doc["_aws"]["CloudWatchMetrics"][0]
    assert isinstance(doc["_aws"]["Timestamp"], int) and spec["Namespace"] == "HybridEdge"
    for m in spec["Metrics"]:  # every declared metric must have a numeric value
        assert isinstance(doc[m["Name"]], (int, float)), m["Name"]
    for dims in spec["Dimensions"]:  # every dimension must be a string field
        assert all(isinstance(doc[d], str) for d in dims)
    # The Terraform alarm reads Escalations and Requests by Region.
    assert ["Region"] in spec["Dimensions"] and doc["Region"] == "eu-west-1"
    assert doc["Escalations"] == 1 and doc["Requests"] == 1
