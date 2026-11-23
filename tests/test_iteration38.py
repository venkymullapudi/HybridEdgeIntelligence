"""Design review question 77: never write a calibration that turns the edge off."""

import json
import random

import pytest

from hybrid_edge.calibrate import main as calibrate_main

from .test_misc import _synthetic


def write(path, labels):
    with open(path, "w") as fh:
        for (feats, _), y in zip(_synthetic(len(labels)), labels):
            fh.write(json.dumps({"features": feats, "label": y}) + "\n")


def test_uninformative_labels_are_refused_not_written(tmp_path):
    rng = random.Random(7)
    rec, out = tmp_path / "r.jsonl", tmp_path / "cal.json"
    write(rec, [int(rng.random() < .5) for _ in range(300)])  # labels unrelated to features
    with pytest.raises(SystemExit, match=r"refusing to write .*escalation rate 99"):
        calibrate_main([str(rec), "-o", str(out)])
    assert not out.exists()


def test_override_writes_it_anyway(tmp_path):
    rng = random.Random(7)
    rec, out = tmp_path / "r.jsonl", tmp_path / "cal.json"
    write(rec, [int(rng.random() < .5) for _ in range(300)])
    calibrate_main([str(rec), "-o", str(out), "--max-escalation-rate", "1.0"])
    assert out.exists()


def test_reasonable_trade_offs_are_not_second_guessed(tmp_path):
    rec, out = tmp_path / "r.jsonl", tmp_path / "cal.json"
    with open(rec, "w") as fh:
        for feats, y in _synthetic(300):
            fh.write(json.dumps({"features": feats, "label": y}) + "\n")
    calibrate_main([str(rec), "-o", str(out)])  # ~65% escalation is a legitimate operator choice
    assert out.exists()


def test_stricter_limit_is_enforced(tmp_path):
    rec, out = tmp_path / "r.jsonl", tmp_path / "cal.json"
    with open(rec, "w") as fh:
        for feats, y in _synthetic(300):
            fh.write(json.dumps({"features": feats, "label": y}) + "\n")
    with pytest.raises(SystemExit, match="refusing to write"):
        calibrate_main([str(rec), "-o", str(out), "--max-escalation-rate", "0.3"])
