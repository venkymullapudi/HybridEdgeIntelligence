"""Edge-first inference with confidence-gated escalation to Claude."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .backends.base import ChatMessage, EdgeBackend, EdgeBackendError, Generation, estimate_tokens
from .cache import TTLCache, cache_key
from .cloud.claude import ClaudeEscalator, CloudAnswer, CloudUnavailable
from .config import Settings
from .gate.confidence import ConfidenceGate, GateDecision
from .gate.signals import Signals, compute_signals
from .telemetry import Metrics, RecordSink

log = logging.getLogger("hybrid_edge")

REFUSAL_TEXT = "I can't help with that request."


class ServiceUnavailable(RuntimeError):
    """Neither the edge model nor the cloud could answer."""


@dataclass
class InferenceResult:
    request_id: str
    answer: str
    source: str  # "edge" | "cloud" | "cache"
    escalated: bool
    degraded: bool
    confidence: Optional[float]
    threshold: float
    reasons: List[str]
    region: str
    edge_id: str
    model: str
    latency_ms: float
    edge_latency_ms: Optional[float] = None
    cloud_latency_ms: Optional[float] = None
    signals: Optional[Dict[str, Any]] = None
    usage: Dict[str, int] = field(default_factory=dict)
    refused: bool = False


class HybridOrchestrator:
    def __init__(
        self,
        settings: Settings,
        backend: EdgeBackend,
        gate: ConfidenceGate,
        escalator: ClaudeEscalator,
        metrics: Metrics,
        records: RecordSink,
    ):
        self.s = settings
        self.backend = backend
        self.gate = gate
        self.escalator = escalator
        self.metrics = metrics
        self.records = records
        self.cache: TTLCache[InferenceResult] = TTLCache(settings.cache_size, settings.cache_ttl_s)

    async def infer(
        self,
        messages: List[ChatMessage],
        *,
        route: str = "auto",  # "auto" | "edge" | "cloud"
        min_confidence: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex
        if not any(m.role != "system" for m in messages):
            messages = messages + [ChatMessage("user", "")]
        if not any(m.role == "system" for m in messages):
            messages = [ChatMessage("system", self.s.system_prompt)] + messages
        threshold = self.gate.threshold if min_confidence is None else min_confidence
        key = cache_key(messages, self.s.edge_model, max_tokens or self.s.edge_max_tokens)

        if use_cache and route == "auto":
            hit = self.cache.get(key)
            if hit is not None and (hit.source == "cloud" or (hit.confidence or 0) >= threshold):
                return self._finish_cached(hit, rid, t0)

        reasons: List[str] = []
        primary: Optional[Generation] = None
        signals: Optional[Signals] = None
        decision: Optional[GateDecision] = None
        edge_latency: Optional[float] = None

        # ---- pre-gate routing -----------------------------------------------
        skip_edge = False
        if route == "cloud":
            skip_edge = True
            reasons.append("client requested cloud")
        elif estimate_tokens(messages) + (max_tokens or self.s.edge_max_tokens) > self.s.edge_context_tokens:
            skip_edge = True
            reasons.append("prompt exceeds edge context window")

        # ---- edge -----------------------------------------------------------
        if not skip_edge:
            try:
                primary, samples = await self._edge_generate(messages, max_tokens)
                edge_latency = primary.latency_ms
                signals = compute_signals(primary, samples, self.s.gate_min_window)
                decision = self.gate.decide(signals, threshold)
                reasons.extend(decision.reasons)
            except EdgeBackendError as exc:
                log.warning("edge backend failed: %s", exc)
                self.metrics.incr("edge_errors_total")
                reasons.append(f"edge error: {exc}")
                if route == "edge":
                    raise ServiceUnavailable(str(exc)) from exc

        must_escalate = skip_edge or primary is None or (decision is not None and decision.escalate)
        if route == "edge":
            must_escalate = False

        # ---- serve from edge ------------------------------------------------
        if not must_escalate and primary is not None:
            result = self._result(rid, t0, primary.text, "edge", False, False, decision, threshold, reasons,
                                  signals, edge_latency, None, self.s.edge_model,
                                  {"edge_prompt_tokens": primary.prompt_tokens,
                                   "edge_completion_tokens": primary.completion_tokens})
            self._after(result, messages, primary, None, key, cacheable=decision is not None)
            return result

        # ---- escalate -------------------------------------------------------
        try:
            cloud = await self.escalator.answer(messages, primary.text if primary else None)
        except CloudUnavailable as exc:
            reasons.append(f"cloud unavailable: {exc}")
            self.metrics.incr("cloud_errors_total")
            if primary is None:
                raise ServiceUnavailable("; ".join(reasons)) from exc
            result = self._result(rid, t0, primary.text, "edge", True, True, decision, threshold, reasons,
                                  signals, edge_latency, None, self.s.edge_model,
                                  {"edge_prompt_tokens": primary.prompt_tokens,
                                   "edge_completion_tokens": primary.completion_tokens})
            self._after(result, messages, primary, None, key, cacheable=False)
            return result

        if cloud.refused:
            reasons.append("cloud model declined the request")
        usage = {"cloud_input_tokens": cloud.input_tokens, "cloud_output_tokens": cloud.output_tokens}
        if primary is not None:
            usage.update({"edge_prompt_tokens": primary.prompt_tokens,
                          "edge_completion_tokens": primary.completion_tokens})
        if cloud.fallback_used:
            reasons.append(f"served by fallback model {cloud.model}")
        result = self._result(rid, t0, cloud.text or (REFUSAL_TEXT if cloud.refused else ""), "cloud", True, False,
                              decision, threshold, reasons, signals, edge_latency, cloud.latency_ms, cloud.model, usage)
        result.refused = cloud.refused
        self._after(result, messages, primary, cloud, key, cacheable=not cloud.refused and cloud.stop_reason != "max_tokens")
        return result

    async def _edge_generate(self, messages: List[ChatMessage], max_tokens: Optional[int]):
        mt = max_tokens or self.s.edge_max_tokens
        primary_task = self.backend.generate(
            messages, max_tokens=mt, temperature=self.s.edge_temperature, n=1, top_logprobs=self.s.edge_top_logprobs
        )
        k = self.s.gate_consistency_samples
        if k > 0:
            sample_task = self.backend.generate(
                messages, max_tokens=mt, temperature=self.s.gate_consistency_temperature, n=k,
                top_logprobs=self.s.edge_top_logprobs,
            )
            prim, samples = await asyncio.gather(primary_task, sample_task)
            return prim[0], samples
        prim = await primary_task
        return prim[0], []

    def _result(self, rid, t0, answer, source, escalated, degraded, decision, threshold, reasons, signals,
                edge_latency, cloud_latency, model, usage) -> InferenceResult:
        return InferenceResult(
            request_id=rid,
            answer=answer,
            source=source,
            escalated=escalated,
            degraded=degraded,
            confidence=decision.score if decision else None,
            threshold=threshold,
            reasons=reasons,
            region=self.s.region,
            edge_id=self.s.edge_id,
            model=model,
            latency_ms=(time.perf_counter() - t0) * 1000,
            edge_latency_ms=edge_latency,
            cloud_latency_ms=cloud_latency,
            signals=signals.as_dict() if signals else None,
            usage=usage,
        )

    def _finish_cached(self, hit: InferenceResult, rid: str, t0: float) -> InferenceResult:
        res = InferenceResult(**{**hit.__dict__})
        res.request_id = rid
        res.source = "cache"
        res.escalated = False
        res.degraded = False
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.edge_latency_ms = None
        res.cloud_latency_ms = None
        res.usage = {}
        res.reasons = ["cache hit (originally served by %s)" % hit.source]
        self.metrics.observe("cache", False, False, hit.confidence, res.latency_ms)
        return res

    def _after(self, result: InferenceResult, messages: List[ChatMessage], primary: Optional[Generation],
               cloud: Optional[CloudAnswer], key: str, cacheable: bool) -> None:
        cloud_tokens = (cloud.input_tokens + cloud.output_tokens) if cloud else 0
        self.metrics.observe(result.source, result.escalated, result.degraded, result.confidence,
                             result.latency_ms, cloud_tokens)
        if cacheable:
            self.cache.put(key, result)
        if self.records.should_record(result.escalated):
            self.records.submit({
                "ts": time.time(),
                "request_id": result.request_id,
                "region": self.s.region,
                "edge_id": self.s.edge_id,
                "edge_model": self.s.edge_model,
                "cloud_model": cloud.model if cloud else None,
                "escalated": result.escalated,
                "degraded": result.degraded,
                "confidence": result.confidence,
                "threshold": result.threshold,
                "reasons": result.reasons,
                "signals": result.signals,
                "features": None if result.signals is None else _features(result.signals),
                "messages": [m.__dict__ for m in messages],
                "edge_answer": primary.text if primary else None,
                "cloud_answer": cloud.text if cloud else None,
                "cloud_refused": cloud.refused if cloud else None,
                "usage": result.usage,
            })


def _features(signals: Dict[str, Any]) -> Dict[str, float]:
    s = Signals(**signals)
    return s.vector()
