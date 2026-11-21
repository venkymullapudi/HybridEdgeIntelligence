"""The confidence gate: turns signals into a score and an escalate/serve decision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..config import GateWeights
from .signals import Signals


@dataclass
class Calibrator:
    """Logistic model P(edge answer is acceptable | features), fit offline."""

    weights: Dict[str, float]
    bias: float
    threshold: Optional[float] = None

    def predict(self, features: Dict[str, float]) -> float:
        z = self.bias + sum(w * features.get(k, 0.0) for k, w in self.weights.items())
        return _sigmoid(z)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(weights=raw["weights"], bias=float(raw["bias"]), threshold=raw.get("threshold"))

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"weights": self.weights, "bias": self.bias, "threshold": self.threshold}, fh, indent=2)


@dataclass
class GateDecision:
    score: float
    threshold: float
    escalate: bool
    reasons: List[str] = field(default_factory=list)
    method: str = "heuristic"


class ConfidenceGate:
    def __init__(
        self,
        threshold: float,
        weights: Optional[GateWeights] = None,
        calibrator: Optional[Calibrator] = None,
        hedge_penalty: float = 0.25,
        truncation_penalty: float = 0.15,
    ):
        self.weights = weights or GateWeights()
        self.calibrator = calibrator
        self.hedge_penalty = hedge_penalty
        self.truncation_penalty = truncation_penalty
        self.threshold = calibrator.threshold if (calibrator and calibrator.threshold is not None) else threshold

    def score(self, s: Signals) -> float:
        if self.calibrator is not None:
            return self.calibrator.predict(s.vector())
        parts = [
            (self.weights.mean_prob, s.mean_prob),
            (self.weights.min_window_prob, s.min_window_prob),
            (self.weights.entropy, s.entropy_conf),
        ]
        if s.consistency_measured:
            parts.append((self.weights.consistency, s.consistency))
        total_w = sum(w for w, _ in parts) or 1.0
        score = sum(w * v for w, v in parts) / total_w
        if s.hedge:
            score -= self.hedge_penalty
        if s.truncated:
            score -= self.truncation_penalty
        return max(0.0, min(1.0, score))

    def decide(self, s: Signals, threshold: Optional[float] = None) -> GateDecision:
        thr = self.threshold if threshold is None else threshold
        sc = self.score(s)
        reasons: List[str] = []
        if sc < thr:
            reasons.append(f"score {sc:.3f} < threshold {thr:.3f}")
        if s.hedge:
            reasons.append("edge answer hedges")
        if s.truncated:
            reasons.append("edge answer truncated")
        if s.aborted:
            reasons.append("edge generation aborted early (confidence collapsed)")
        if not s.has_logprobs:
            reasons.append("no logprobs from edge model")
        if s.consistency_measured and s.consistency < 0.5:
            reasons.append(f"samples disagree (consistency {s.consistency:.2f})")
        return GateDecision(
            score=sc,
            threshold=thr,
            escalate=sc < thr or s.aborted,
            reasons=reasons,
            method="calibrated" if self.calibrator else "heuristic",
        )


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
