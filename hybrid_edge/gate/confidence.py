"""The confidence gate: turns signals into a score and an escalate/serve decision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import GateWeights
from .signals import Signals


@dataclass
class Calibrator:
    """Logistic model P(edge answer is acceptable | features), fit offline."""

    weights: Dict[str, float]
    bias: float
    threshold: Optional[float] = None
    # The edge model whose signals this was fit on. Logprob scales differ between
    # models, so a calibration must never silently outlive a model swap.
    edge_model: Optional[str] = None
    # The settings the features were computed under (Settings.signal_config()).
    signal_config: Optional[Dict[str, Any]] = None

    def predict(self, features: Dict[str, float]) -> float:
        z = self.bias + sum(w * features.get(k, 0.0) for k, w in self.weights.items())
        return _sigmoid(z)

    @classmethod
    def load(cls, path: str, s3_client: Any = None) -> "Calibrator":
        """Load from a local path or an ``s3://bucket/key`` URI (how deployed edges get it)."""
        raw = json.loads(read_text(path, s3_client))
        return cls(weights=raw["weights"], bias=float(raw["bias"]), threshold=raw.get("threshold"),
                   edge_model=raw.get("edge_model"), signal_config=raw.get("signal_config"))

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"weights": self.weights, "bias": self.bias, "threshold": self.threshold,
                       "edge_model": self.edge_model, "signal_config": self.signal_config}, fh, indent=2)


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
        score = self._raw_score(s)
        # Fail safe: NaN would compare False against the threshold and min()/max() would
        # clamp it to full confidence, serving garbage from the edge.
        return score if math.isfinite(score) else 0.0

    def _raw_score(self, s: Signals) -> float:
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
        if not math.isfinite(score):
            return 0.0  # check before clamping: min(1.0, nan) is 1.0
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


def read_text(path: str, s3_client: Any = None) -> str:
    if path.startswith("s3://"):
        bucket, _, key = path[5:].partition("/")
        if not bucket or not key:
            raise ValueError(f"bad S3 URI {path!r}; expected s3://bucket/key")
        if s3_client is None:
            import boto3  # optional dependency, in the image via the [aws] extra

            s3_client = boto3.client("s3")
        return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)
