"""Design review question 66: calibrations are bound to the gate settings they were fit under."""

import json

import pytest

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.calibrate import main as calibrate_main
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.gate.confidence import Calibrator
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink
from .test_misc import _synthetic


async def test_records_carry_the_signal_config(settings):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    await o.infer([ChatMessage("user", "hard question")])
    assert o.records.records[-1]["signal_config"] == settings.signal_config()
    assert set(settings.signal_config()) >= {"edge_top_logprobs", "gate_min_window", "edge_temperature"}


def _write(path, signal_config, n=200, mode="a"):
    with open(path, mode) as fh:
        for feats, y in _synthetic(n):
            fh.write(json.dumps({"edge_model": "m", "signal_config": signal_config, "features": feats, "label": y}) + "\n")


def test_calibration_stores_config_and_refuses_mixed_sets(tmp_path):
    a = Settings.from_dict({}).signal_config()
    b = {**a, "edge_top_logprobs": 10}
    rec, out = tmp_path / "r.jsonl", tmp_path / "c.json"
    _write(rec, a)
    calibrate_main([str(rec), "-o", str(out)])
    assert json.loads(out.read_text())["signal_config"] == a

    _write(rec, b)
    with pytest.raises(SystemExit, match="different gate settings"):
        calibrate_main([str(rec), "-o", str(out)])
    calibrate_main([str(rec), "-o", str(out), "--any-signal-config"])
    assert json.loads(out.read_text())["signal_config"] is None


def test_startup_refuses_changed_gate_settings(tmp_path):
    s = Settings.from_dict({"edge_backend": "mock", "edge_model": "m"})
    path = tmp_path / "cal.json"
    Calibrator({"mean_prob": 1.0}, 0.0, 0.5, edge_model="m", signal_config=s.signal_config()).dump(str(path))
    s.gate_calibration_path = str(path)
    build_orchestrator(s, escalator=ClaudeEscalator(s, client=FakeClaude()))  # same settings: fine

    s.edge_top_logprobs = 10
    with pytest.raises(ValueError, match=r"edge_top_logprobs: 5 -> 10"):
        build_orchestrator(s, escalator=ClaudeEscalator(s, client=FakeClaude()))
    s.gate_allow_calibration_mismatch = True
    build_orchestrator(s, escalator=ClaudeEscalator(s, client=FakeClaude()))


def test_legacy_calibrations_without_a_config_still_load(tmp_path):
    s = Settings.from_dict({"edge_backend": "mock", "edge_model": "m", "edge_top_logprobs": 10})
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"weights": {"mean_prob": 1.0}, "bias": 0.0, "threshold": 0.5, "edge_model": "m"}))
    s.gate_calibration_path = str(path)
    o = build_orchestrator(s, escalator=ClaudeEscalator(s, client=FakeClaude()))
    assert o.gate.threshold == 0.5
