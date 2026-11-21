"""Runtime configuration.

Every field can be set from an environment variable named ``HEI_<FIELD>`` (upper
case), or from a YAML file pointed to by ``HEI_CONFIG``. Environment variables win.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GateWeights:
    """Weights of the heuristic (uncalibrated) confidence score. Normalised at use."""

    mean_prob: float = 0.35
    min_window_prob: float = 0.25
    entropy: float = 0.20
    consistency: float = 0.20


@dataclass
class Settings:
    # --- identity ---------------------------------------------------------
    region: str = "local"
    edge_id: str = "edge-local"

    # --- edge (open-weight) model ----------------------------------------
    # "openai" = any OpenAI-compatible server with logprobs (vLLM, llama.cpp, SGLang, Ollama)
    # "mock"   = deterministic fake, for tests and local demos
    edge_backend: str = "openai"
    edge_base_url: str = "http://127.0.0.1:8000/v1"
    edge_model: str = "Qwen/Qwen2.5-7B-Instruct"
    edge_api_key: str = "EMPTY"
    edge_max_tokens: int = 512
    edge_temperature: float = 0.2
    edge_top_logprobs: int = 5
    edge_timeout_s: float = 30.0
    edge_context_tokens: int = 8192
    # Below this much room left in the edge context, skip the edge and escalate.
    edge_min_answer_tokens: int = 256
    # Escalate instead of waiting when the edge model is slower than this (0 = off).
    edge_deadline_s: float = 10.0
    # Stream from the edge model so the gate can abort a doomed answer early.
    edge_stream: bool = True

    # --- confidence gate --------------------------------------------------
    gate_threshold: float = 0.70
    # Extra samples drawn for self-consistency. 0 disables (cheapest).
    gate_consistency_samples: int = 0
    gate_consistency_temperature: float = 0.8
    # Draw consistency samples only when |score - threshold| < band (gray zone).
    gate_consistency_band: float = 0.15
    gate_min_window: int = 8
    # Early abort: stop streaming once a window of tokens falls below this probability
    # (geometric mean) after gate_abort_min_tokens tokens. 0 disables.
    gate_abort_window_prob: float = 0.15
    gate_abort_min_tokens: int = 16
    gate_weights: GateWeights = field(default_factory=GateWeights)
    # Optional JSON produced by `hybrid-edge-calibrate`; replaces the heuristic.
    gate_calibration_path: Optional[str] = None
    # Refuse to start when the calibration was fit on a different edge model.
    gate_allow_calibration_mismatch: bool = False
    gate_hedge_penalty: float = 0.25
    gate_truncation_penalty: float = 0.15

    # --- cloud LLM (escalation target) -----------------------------------
    # "anthropic" = Claude API, "bedrock" = Claude on Amazon Bedrock, "none" = never escalate
    cloud_provider: str = "anthropic"
    cloud_model: str = "claude-opus-5-5"
    cloud_effort: str = "medium"
    cloud_max_tokens: int = 16000
    cloud_timeout_s: float = 120.0
    # Total budget for one escalation, SDK retries included. Keep below the load
    # balancer's idle timeout (180 s in the Terraform module).
    cloud_deadline_s: float = 90.0
    cloud_aws_region: Optional[str] = None
    cloud_refusal_fallbacks: bool = True
    cloud_include_edge_draft: bool = False
    # Cost guard: max escalations per minute per edge node (token bucket). 0 = unlimited.
    cloud_max_escalations_per_min: int = 600
    # Per API key, so one client can't drain the node budget. 0 = unlimited.
    cloud_max_escalations_per_key_per_min: int = 120
    cloud_breaker_failures: int = 5
    cloud_breaker_cooldown_s: float = 30.0
    # USD per million tokens, for cost / avoided-cost metrics (defaults: claude-opus-5-5).
    cloud_price_in_per_mtok: float = 4.0
    cloud_price_out_per_mtok: float = 20.0

    # Shadow escalation: fraction of edge-served requests also sent to Claude in the
    # background, so calibration sees reference answers for *confident* edge answers too.
    shadow_rate: float = 0.0
    shadow_max_per_min: int = 30

    # Client controls. When disabled, `route` and `min_confidence` in requests are ignored.
    allow_client_overrides: bool = True
    max_client_min_confidence: float = 0.95

    system_prompt: str = "You are a helpful, accurate assistant. Answer concisely."

    # --- caching ---------------------------------------------------------
    cache_size: int = 10000
    cache_ttl_s: float = 3600.0
    # Off: cache entries are scoped to the caller's API key (no cross-tenant answers).
    cache_shared_across_keys: bool = False

    # --- telemetry -------------------------------------------------------
    metrics_namespace: str = "HybridEdge"
    emit_emf: bool = True
    # "none", "stdout", "file", "firehose"
    record_sink: str = "none"
    record_path: str = "data/escalations.jsonl"
    record_firehose_stream: Optional[str] = None
    # Store prompts/answers in records (needed for distillation; disable for sensitive traffic).
    record_include_text: bool = True
    # Fraction of confident (non-escalated) requests also recorded, for gate calibration.
    record_sample_rate_confident: float = 0.02
    record_queue_size: int = 10000
    record_flush_interval_s: float = 1.0

    # --- server ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8080
    api_keys: List[str] = field(default_factory=list)
    # Request limits (reject with 413): stop oversized prompts from burning memory or
    # escalating straight to Claude because they overflow the edge context.
    max_request_chars: int = 100000
    max_messages: int = 200

    @classmethod
    def load(cls, overrides: Optional[Dict[str, Any]] = None) -> "Settings":
        data: Dict[str, Any] = {}
        path = os.environ.get("HEI_CONFIG")
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data.update(yaml.safe_load(fh) or {})
        for f in dataclasses.fields(cls):
            env = os.environ.get("HEI_" + f.name.upper())
            if env is not None:
                data[f.name] = env
        if overrides:
            data.update(overrides)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        s = cls()
        types = {f.name: f for f in dataclasses.fields(cls)}
        for key, raw in data.items():
            if key not in types:
                raise ValueError(f"unknown setting: {key}")
            setattr(s, key, _coerce(getattr(s, key), raw, key))
        s.validate()
        return s

    def validate(self) -> None:
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be in [0, 1]")
        if self.edge_backend not in ("openai", "mock"):
            raise ValueError("edge_backend must be 'openai' or 'mock'")
        if self.cloud_provider not in ("anthropic", "bedrock", "none"):
            raise ValueError("cloud_provider must be 'anthropic', 'bedrock' or 'none'")
        if self.record_sink not in ("none", "stdout", "file", "firehose"):
            raise ValueError("record_sink must be none|stdout|file|firehose")
        if self.record_sink == "firehose" and not self.record_firehose_stream:
            raise ValueError("record_firehose_stream is required for the firehose sink")
        for name in ("shadow_rate", "record_sample_rate_confident", "max_client_min_confidence",
                     "gate_abort_window_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cloud_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("cloud_effort must be low|medium|high|xhigh|max")


def _coerce(default: Any, raw: Any, key: str) -> Any:
    if isinstance(default, GateWeights):
        if isinstance(raw, GateWeights):
            return raw
        if isinstance(raw, str):
            raw = yaml.safe_load(raw)
        return GateWeights(**raw)
    if not isinstance(raw, str):
        return raw
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if default is None:
        if key.endswith("_path") or key.endswith("_stream") or key.endswith("_region"):
            return raw or None
    return raw
