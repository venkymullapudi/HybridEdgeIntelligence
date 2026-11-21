"""Edge-first inference with confidence-gated escalation to Claude."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .backends.base import (ChatMessage, EdgeBackend, EdgeBackendError, Generation, TokenInfo, edge_messages,
                            estimate_tokens)
from .cache import TTLCache, cache_key
from .cloud.claude import ClaudeEscalator, CloudAnswer, CloudUnavailable, estimate_cost
from .config import Settings
from .gate.confidence import ConfidenceGate, GateDecision
from .gate.signals import Signals, compute_signals
from .telemetry import Metrics, RecordSink

log = logging.getLogger("hybrid_edge")

REFUSAL_TEXT = "I can't help with that request."
ANON = "anon"


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
    usage: Dict[str, float] = field(default_factory=dict)
    refused: bool = False


def result_size(r: "InferenceResult") -> int:
    """Approximate memory held by a cached result: the answer dominates."""
    return 2048 + 4 * len(r.answer) + sum(4 * len(x) for x in r.reasons)


def caller_id(api_key: Optional[str]) -> str:
    """Stable, non-reversible identifier for budgets, cache scoping and records."""
    if not api_key:
        return ANON
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


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
        self.cache: TTLCache[InferenceResult] = TTLCache(settings.cache_size, settings.cache_ttl_s,
                                                         max_bytes=settings.cache_max_bytes, sizeof=result_size)
        self._inflight: Dict[str, "asyncio.Future[InferenceResult]"] = {}
        self._background: Set["asyncio.Task[Any]"] = set()
        self._warned_no_logprobs = False

    # ------------------------------------------------------------------ public
    async def infer(
        self,
        messages: List[ChatMessage],
        *,
        route: str = "auto",  # "auto" | "edge" | "cloud"
        min_confidence: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        caller: str = ANON,
        request_id: Optional[str] = None,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        rid = request_id or uuid.uuid4().hex
        if not any(m.role != "system" for m in messages):
            messages = messages + [ChatMessage("user", "")]
        if not any(m.role == "system" for m in messages):
            messages = [ChatMessage("system", self.s.system_prompt)] + messages

        policy_notes: List[str] = []
        if not self.s.allow_client_overrides:
            if route != "auto" or min_confidence is not None:
                policy_notes.append("client overrides disabled by policy")
            route, min_confidence = "auto", None
        elif min_confidence is not None and min_confidence > self.s.max_client_min_confidence:
            policy_notes.append(f"min_confidence capped at {self.s.max_client_min_confidence}")
            min_confidence = self.s.max_client_min_confidence
        threshold = self.gate.threshold if min_confidence is None else min_confidence

        scope = "shared" if self.s.cache_shared_across_keys else caller
        key = cache_key(messages, self.s.edge_model, max_tokens or self.s.edge_max_tokens, scope)
        cacheable_route = use_cache and route == "auto"

        if cacheable_route:
            hit = self.cache.get(key)
            if hit is not None and (hit.source == "cloud" or (hit.confidence or 0) >= threshold):
                return self._copy_shared(hit, t0, "cache hit (originally served by %s)" % hit.source, rid)
            # Single-flight: identical concurrent requests share one computation. It runs
            # as its own task, so a caller disconnecting (cancellation) never cancels the
            # work other callers are waiting on.
            pending = self._inflight.get(key)
            if pending is not None:
                res = await asyncio.shield(pending)
                return self._copy_shared(res, t0, "coalesced with an identical in-flight request", rid)
            task = asyncio.ensure_future(
                self._infer(messages, route, threshold, max_tokens, caller, key, t0, policy_notes, rid))
            self._inflight[key] = task
            task.add_done_callback(lambda t, k=key: self._settle(k, t))
            return await asyncio.shield(task)
        return await self._infer(messages, route, threshold, max_tokens, caller, key, t0, policy_notes, rid)

    def _settle(self, key: str, task: "asyncio.Future[InferenceResult]") -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()  # mark retrieved: a failure nobody awaited shouldn't log noise

    async def drain(self) -> None:
        """Wait for background (shadow) work; used on shutdown and in tests."""
        while self._background or self._inflight:
            await asyncio.gather(*list(self._background), *list(self._inflight.values()), return_exceptions=True)

    # ---------------------------------------------------------------- internal
    async def _infer(self, messages, route, threshold, max_tokens, caller, key, t0, policy_notes,
                     rid) -> InferenceResult:
        reasons: List[str] = list(policy_notes)
        primary: Optional[Generation] = None
        signals: Optional[Signals] = None
        decision: Optional[GateDecision] = None
        edge_latency: Optional[float] = None

        # ---- pre-gate routing -----------------------------------------------
        skip_edge = False
        if route == "cloud":
            skip_edge = True
            reasons.append("client requested cloud")
        # The edge answer is sized to fit what's left of the context window, so a large
        # requested max_tokens can't by itself force a (billed) escalation. Only a
        # prompt that leaves no room for a useful answer skips the edge.
        edge_max_tokens = max_tokens or self.s.edge_max_tokens
        room = self.s.edge_context_tokens - estimate_tokens(messages)
        if not skip_edge and room < self.s.edge_min_answer_tokens:
            skip_edge = True
            reasons.append("prompt exceeds edge context window")
        elif not skip_edge and edge_max_tokens > room:
            edge_max_tokens = room
            reasons.append(f"edge answer limited to {room} tokens by its context window")

        # ---- edge -----------------------------------------------------------
        if not skip_edge:
            try:
                primary, signals, decision = await self._edge(messages, edge_max_tokens, threshold, route, caller)
                edge_latency = primary.latency_ms
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
            usage = self._edge_usage(primary)
            usage["avoided_cost_usd"] = estimate_cost(self.s, primary.prompt_tokens, primary.completion_tokens)
            result = self._result(rid, t0, primary.text, "edge", False, False, decision, threshold, reasons,
                                  signals, edge_latency, None, self.s.edge_model, usage)
            self._after(result, messages, primary, None, key, caller, cacheable=decision is not None)
            self._maybe_shadow(result, messages, primary, caller)
            return result

        # ---- escalate -------------------------------------------------------
        try:
            cloud = await self.escalator.answer(messages, primary.text if primary else None, caller=caller)
        except CloudUnavailable as exc:
            reasons.append(f"cloud unavailable: {exc}")
            self.metrics.incr("cloud_errors_total")
            if primary is None:
                raise ServiceUnavailable("; ".join(reasons)) from exc
            result = self._result(rid, t0, primary.text, "edge", True, True, decision, threshold, reasons,
                                  signals, edge_latency, None, self.s.edge_model, self._edge_usage(primary))
            self._after(result, messages, primary, None, key, caller, cacheable=False)
            return result

        if cloud.refused:
            reasons.append("cloud model declined the request")
        if cloud.stop_reason == "max_tokens":
            reasons.append(f"cloud answer truncated at cloud_max_tokens ({self.s.cloud_max_tokens})")
        if cloud.fallback_used:
            reasons.append(f"served by fallback model {cloud.model}")
        usage: Dict[str, float] = {"cloud_input_tokens": cloud.input_tokens,
                                   "cloud_output_tokens": cloud.output_tokens,
                                   "cloud_cost_usd": round(cloud.cost_usd, 6)}
        if primary is not None:
            usage.update(self._edge_usage(primary))
        result = self._result(rid, t0, cloud.text or (REFUSAL_TEXT if cloud.refused else ""), "cloud", True, False,
                              decision, threshold, reasons, signals, edge_latency, cloud.latency_ms, cloud.model, usage)
        result.refused = cloud.refused
        self._after(result, messages, primary, cloud, key, caller,
                    cacheable=not cloud.refused and cloud.stop_reason != "max_tokens")
        return result

    async def _edge(self, messages, max_tokens, threshold, route, caller) -> Tuple[Generation, Signals, GateDecision]:
        messages = edge_messages(messages, self.s.edge_system_role)
        mt = max_tokens
        # Only abort early when the answer can actually be escalated; otherwise a
        # complete (if shaky) edge answer beats a truncated one.
        abort = self._abort_check() if route == "auto" and self.escalator.can_escalate(caller) else None

        async def run() -> Generation:
            if self.s.edge_stream:
                return await self.backend.generate_stream(
                    messages, max_tokens=mt, temperature=self.s.edge_temperature,
                    top_logprobs=self.s.edge_top_logprobs, should_abort=abort)
            gens = await self.backend.generate(messages, max_tokens=mt, temperature=self.s.edge_temperature,
                                               n=1, top_logprobs=self.s.edge_top_logprobs)
            return gens[0]

        t0 = time.perf_counter()
        if self.s.edge_deadline_s > 0 and route != "edge":
            try:
                primary = await asyncio.wait_for(run(), timeout=self.s.edge_deadline_s)
            except asyncio.TimeoutError as exc:
                self.metrics.incr("edge_deadline_exceeded_total")
                raise EdgeBackendError(f"edge deadline of {self.s.edge_deadline_s}s exceeded") from exc
        else:
            primary = await run()
        primary.latency_ms = (time.perf_counter() - t0) * 1000
        if primary.finish_reason == "aborted":
            self.metrics.incr("edge_aborted_total")

        signals = compute_signals(primary, [], self.s.gate_min_window)
        if not signals.has_logprobs and primary.completion_tokens > 0:
            # Without logprobs every answer scores a neutral 0.5 and escalates: a silent
            # cost explosion. Count it so the alarm (and /metrics) can catch it.
            self.metrics.incr("edge_no_logprobs_total")
            if not self._warned_no_logprobs:
                self._warned_no_logprobs = True
                log.error("edge model returned no logprobs; the gate cannot score answers and "
                          "will escalate them. Check the server supports logprobs/top_logprobs.")
        decision = self.gate.decide(signals, threshold)

        # Gray-zone self-consistency: extra samples only when they could flip the decision.
        k = self.s.gate_consistency_samples
        if k > 0 and not signals.aborted and abs(decision.score - threshold) < self.s.gate_consistency_band:
            try:
                samples = await self.backend.generate(
                    messages, max_tokens=mt, temperature=self.s.gate_consistency_temperature, n=k,
                    top_logprobs=self.s.edge_top_logprobs)
                signals = compute_signals(primary, samples, self.s.gate_min_window)
                decision = self.gate.decide(signals, threshold)
                self.metrics.incr("consistency_sampled_total")
            except EdgeBackendError as exc:
                log.warning("consistency sampling failed: %s", exc)
        return primary, signals, decision

    def _abort_check(self):
        prob = self.s.gate_abort_window_prob
        if prob <= 0:
            return None
        w = max(1, self.s.gate_min_window)
        min_tokens = max(self.s.gate_abort_min_tokens, w)
        log_floor = math.log(prob)

        def check(tokens: List[TokenInfo]) -> bool:
            if len(tokens) < min_tokens:
                return False
            recent = tokens[-w:]
            return sum(t.logprob for t in recent) / len(recent) < log_floor

        return check

    def _maybe_shadow(self, result: InferenceResult, messages, primary: Generation, caller: str) -> None:
        if self.s.shadow_rate <= 0 or not self.escalator.enabled or random.random() >= self.s.shadow_rate:
            return

        async def shadow() -> None:
            try:
                cloud = await self.escalator.answer(messages, None, caller=caller, shadow=True)
            except CloudUnavailable as exc:
                log.debug("shadow skipped: %s", exc)
                return
            self.metrics.incr("shadow_total")
            self.metrics.incr("cloud_cost_usd_total", cloud.cost_usd)
            self.records.submit(self._record(result, messages, primary, cloud, caller, shadow=True))

        task = asyncio.ensure_future(shadow())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @staticmethod
    def _edge_usage(primary: Generation) -> Dict[str, float]:
        return {"edge_prompt_tokens": primary.prompt_tokens, "edge_completion_tokens": primary.completion_tokens}

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

    def _copy_shared(self, hit: InferenceResult, t0: float, reason: str, rid: str) -> InferenceResult:
        res = replace(hit, request_id=rid, source="cache", escalated=False, degraded=False,
                      latency_ms=(time.perf_counter() - t0) * 1000, edge_latency_ms=None, cloud_latency_ms=None,
                      reasons=[reason])
        avoided = hit.usage.get("cloud_cost_usd") or hit.usage.get("avoided_cost_usd") or 0.0
        res.usage = {"avoided_cost_usd": avoided}
        self.metrics.observe("cache", False, False, hit.confidence, res.latency_ms, avoided_cost=avoided)
        return res

    def _after(self, result: InferenceResult, messages: List[ChatMessage], primary: Optional[Generation],
               cloud: Optional[CloudAnswer], key: str, caller: str, cacheable: bool) -> None:
        self.metrics.observe(
            result.source, result.escalated, result.degraded, result.confidence, result.latency_ms,
            cloud_tokens=(cloud.input_tokens + cloud.output_tokens) if cloud else 0,
            cloud_cost=cloud.cost_usd if cloud else 0.0,
            avoided_cost=float(result.usage.get("avoided_cost_usd", 0.0)),
        )
        if cacheable:
            self.cache.put(key, result)
        if self.records.should_record(result.escalated):
            self.records.submit(self._record(result, messages, primary, cloud, caller, shadow=False))

    def _record(self, result, messages, primary, cloud, caller, shadow) -> Dict[str, Any]:
        return {
            "ts": time.time(),
            "request_id": result.request_id,
            "region": self.s.region,
            "edge_id": self.s.edge_id,
            "caller": caller,
            "edge_model": self.s.edge_model,
            "cloud_model": cloud.model if cloud else None,
            "escalated": result.escalated,
            "shadow": shadow,
            # Inverse-probability weight for calibration: shadow records stand for
            # 1/shadow_rate confident requests; every escalation is recorded.
            "sample_weight": (1.0 / self.s.shadow_rate) if shadow and self.s.shadow_rate > 0 else 1.0,
            "degraded": result.degraded,
            "confidence": result.confidence,
            "threshold": result.threshold,
            "reasons": result.reasons,
            "signals": result.signals,
            "features": None if result.signals is None else Signals(**result.signals).vector(),
            "messages": [m.__dict__ for m in messages],
            "edge_answer": primary.text if primary else None,
            "cloud_answer": cloud.text if cloud else None,
            "cloud_refused": cloud.refused if cloud else None,
            "usage": result.usage,
        }
