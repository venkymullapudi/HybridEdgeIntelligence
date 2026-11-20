import math

from hybrid_edge.backends.base import Generation, TokenInfo
from hybrid_edge.config import GateWeights
from hybrid_edge.gate.confidence import Calibrator, ConfidenceGate
from hybrid_edge.gate.signals import (
    Signals,
    compute_signals,
    entropy_confidence,
    is_hedged,
    mean_prob,
    min_window_prob,
    self_consistency,
)


def toks(probs, top=None):
    return [TokenInfo(token=f"t{i}", logprob=math.log(p), top=top or {f"t{i}": math.log(p)}) for i, p in enumerate(probs)]


def test_mean_prob_is_geometric_mean():
    assert abs(mean_prob(toks([0.5, 0.5])) - 0.5) < 1e-9
    assert mean_prob([]) == 0.0


def test_min_window_catches_local_dip():
    t = toks([0.99] * 20 + [0.1, 0.1, 0.1] + [0.99] * 20)
    assert mean_prob(t) > 0.8
    assert min_window_prob(t, 3) < 0.11


def test_entropy_confidence_peaked_vs_flat():
    peaked = [TokenInfo("a", math.log(0.97), {"a": math.log(0.97), "b": math.log(0.01), "c": math.log(0.01)})]
    flat = [TokenInfo("a", math.log(0.25), {"a": math.log(0.25), "b": math.log(0.25), "c": math.log(0.25), "d": math.log(0.25)})]
    assert entropy_confidence(peaked) > 0.8
    assert entropy_confidence(flat) < 0.05


def test_hedge_detection():
    assert is_hedged("I'm not sure, but maybe Paris.")
    assert is_hedged("I don't know.")
    assert not is_hedged("Paris is the capital of France.")


def test_self_consistency():
    assert self_consistency("Paris", ["Paris", "paris."]) == 1.0
    assert self_consistency("option A", ["option B"]) < 1.0
    assert self_consistency("x", []) == 1.0


def test_gate_heuristic_decisions():
    gate = ConfidenceGate(threshold=0.7)
    confident = Signals(mean_prob=0.95, min_window_prob=0.9, entropy_conf=0.9)
    shaky = Signals(mean_prob=0.4, min_window_prob=0.2, entropy_conf=0.3)
    assert not gate.decide(confident).escalate
    d = gate.decide(shaky)
    assert d.escalate and any("threshold" in r for r in d.reasons)


def test_gate_hedge_and_truncation_penalties():
    gate = ConfidenceGate(threshold=0.7, hedge_penalty=0.3)
    s = Signals(mean_prob=0.9, min_window_prob=0.9, entropy_conf=0.9, hedge=True)
    d = gate.decide(s)
    assert d.escalate and "edge answer hedges" in d.reasons


def test_consistency_only_weighted_when_measured():
    gate = ConfidenceGate(threshold=0.7, weights=GateWeights(consistency=0.5))
    base = Signals(mean_prob=0.9, min_window_prob=0.9, entropy_conf=0.9, consistency=0.0)
    assert gate.score(base) > 0.85
    base.consistency_measured = True
    assert gate.score(base) < 0.7


def test_calibrated_gate_uses_calibrator_threshold():
    cal = Calibrator(weights={"mean_prob": 10.0}, bias=-5.0, threshold=0.6)
    gate = ConfidenceGate(threshold=0.9, calibrator=cal)
    assert gate.threshold == 0.6
    d = gate.decide(Signals(mean_prob=0.9))
    assert d.method == "calibrated" and not d.escalate


def test_compute_signals_without_logprobs_is_neutral():
    g = Generation(text="hello", tokens=[], finish_reason="length", completion_tokens=3)
    s = compute_signals(g, [], 8)
    assert not s.has_logprobs and s.truncated and s.mean_prob == 0.5
