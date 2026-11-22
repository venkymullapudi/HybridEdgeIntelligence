"""Escalation target: Claude, via the Claude API or Amazon Bedrock.

Only reached when the confidence gate rejects the edge answer (or for a sampled
shadow call), so every call here is a deliberate spend. Includes:
  * refusal fallbacks (server-side on the Claude API, client-side middleware on Bedrock)
  * a circuit breaker with a real half-open state, so a cloud outage degrades
    to edge answers instead of timeouts
  * token-bucket budgets per node, per API key, and for shadow traffic
  * per-call cost estimates
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import anthropic

from ..backends.base import ChatMessage
from ..config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Bedrock has no server-side `fallbacks`; the client-side middleware needs an explicit model.
BEDROCK_FALLBACK_MODEL = "anthropic.claude-opus-4-8"


class CloudUnavailable(RuntimeError):
    """Cloud escalation was skipped or failed; caller should degrade gracefully.

    ``reason`` is a stable, low-cardinality label for metrics: our own throttling
    (budget, key_budget, shadow_budget, breaker, disabled) reads very differently
    on-call from Claude failing (error, deadline, empty, rejected)."""

    def __init__(self, message: str, reason: str = "error"):
        super().__init__(message)
        self.reason = reason


@dataclass
class CloudAnswer:
    text: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    refused: bool = False
    fallback_used: bool = False


class TokenBucket:
    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.last = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def available(self) -> bool:
        if self.capacity <= 0:
            return True  # unlimited
        self._refill()
        return self.tokens >= 1.0

    def take(self) -> bool:
        if self.capacity <= 0:
            return True
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def refund(self) -> None:
        if self.capacity > 0:
            self.tokens = min(self.capacity, self.tokens + 1.0)


class CircuitBreaker:
    """closed -> (N failures) -> open -> (cooldown) -> half-open: one probe -> closed | open."""

    def __init__(self, failures: int, cooldown_s: float):
        self.threshold = failures
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: Optional[float] = None
        self.probe_in_flight = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    @property
    def open(self) -> bool:
        """True when a request would be rejected right now."""
        st = self.state
        return st == "open" or (st == "half_open" and self.probe_in_flight)

    def allow(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self.probe_in_flight:
            self.probe_in_flight = True
            return True
        return False

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.probe_in_flight = False

    def failure(self) -> None:
        self.failures += 1
        self.probe_in_flight = False
        if self.opened_at is not None or self.failures >= self.threshold:
            self.opened_at = time.monotonic()  # (re-)open

    def release(self) -> None:
        """A probe ended without reaching the cloud (e.g. budget denied after allow())."""
        self.probe_in_flight = False


class KeyedBuckets:
    """Bounded LRU of per-caller token buckets."""

    def __init__(self, per_minute: int, max_keys: int = 10000):
        self.per_minute = per_minute
        self.max_keys = max_keys
        self._buckets: "OrderedDict[str, TokenBucket]" = OrderedDict()

    def get(self, key: str) -> TokenBucket:
        b = self._buckets.get(key)
        if b is None:
            b = TokenBucket(self.per_minute)
            self._buckets[key] = b
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return b


def build_claude_client(settings: Settings) -> Any:
    if settings.cloud_provider == "none":
        return None
    if settings.cloud_provider == "bedrock":
        middleware = []
        if settings.cloud_refusal_fallbacks:
            middleware.append(anthropic.BetaRefusalFallbackMiddleware([{"model": BEDROCK_FALLBACK_MODEL}]))
        return anthropic.AsyncAnthropicBedrockMantle(
            aws_region=settings.cloud_aws_region or settings.region,
            timeout=settings.cloud_timeout_s,
            middleware=middleware,
        )
    # Credentials resolve from ANTHROPIC_API_KEY (injected from Secrets Manager on AWS).
    return anthropic.AsyncAnthropic(timeout=settings.cloud_timeout_s, max_retries=2)


def model_id(settings: Settings) -> str:
    m = settings.cloud_model
    if settings.cloud_provider == "bedrock" and not m.startswith("anthropic."):
        m = "anthropic." + m
    return m


def fallback_kwargs(settings: Settings) -> Dict[str, Any]:
    if settings.cloud_refusal_fallbacks and settings.cloud_provider == "anthropic":
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return {}


def estimate_cost(settings: Settings, input_tokens: int, output_tokens: int,
                  cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """USD for one call. ``input_tokens`` excludes cached tokens, which bill separately."""
    return (input_tokens * settings.cloud_price_in_per_mtok
            + output_tokens * settings.cloud_price_out_per_mtok
            + cache_read_tokens * settings.cloud_price_cache_read_per_mtok
            + cache_write_tokens * settings.cloud_price_cache_write_per_mtok) / 1e6


class ClaudeEscalator:
    def __init__(self, settings: Settings, client: Any = None):
        self.s = settings
        self.provider = settings.cloud_provider
        self.model = model_id(settings)
        self.client = client if client is not None else build_claude_client(settings)
        self.bucket = TokenBucket(settings.cloud_max_escalations_per_min)
        self.key_buckets = KeyedBuckets(settings.cloud_max_escalations_per_key_per_min)
        self.shadow_bucket = TokenBucket(settings.shadow_max_per_min)
        self.breaker = CircuitBreaker(settings.cloud_breaker_failures, settings.cloud_breaker_cooldown_s)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def can_escalate(self, caller: str = "anon") -> bool:
        """Cheap pre-check (consumes nothing): would an escalation be attempted right now?"""
        return (self.enabled and not self.breaker.open and self.bucket.available()
                and self.key_buckets.get(caller).available())

    def _request(self, messages: List[ChatMessage], edge_draft: Optional[str]) -> Dict[str, Any]:
        system = self.s.system_prompt
        convo: List[Dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system = m.content  # caller-supplied system prompt wins
            else:
                convo.append({"role": m.role, "content": m.content})
        if edge_draft and self.s.cloud_include_edge_draft and convo and convo[-1]["role"] == "user":
            convo[-1] = {
                "role": "user",
                "content": (
                    convo[-1]["content"]
                    + "\n\n<draft_from_smaller_model>\n"
                    + edge_draft
                    + "\n</draft_from_smaller_model>\n"
                    + "A smaller model drafted the answer above with low confidence. "
                    "Use it only if it is correct; otherwise answer from scratch. "
                    "Reply with the final answer only."
                ),
            }
        req: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.s.cloud_max_tokens,
            "system": system,
            "messages": convo,
            "output_config": {"effort": self.s.cloud_effort},
        }
        if not system.strip():
            # The API rejects empty text blocks; no system prompt is the faithful translation.
            del req["system"]
        elif self.s.cloud_prompt_caching:
            # The system prompt is the prefix shared across a tenant's escalations: mark
            # its end. (Below the model's minimum cacheable size it silently isn't cached.)
            req["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if self.s.cloud_prompt_caching:
            if sum(1 for m in convo if m["role"] == "user") > 1:
                # Multi-turn: the next turn re-sends this history, so let the automatic
                # breakpoint follow the conversation. Single-turn requests end in a unique
                # question; caching that tail would be a write premium never read back.
                req["cache_control"] = {"type": "ephemeral"}
        req.update(fallback_kwargs(self.s))
        return req

    def _admit(self, caller: str, shadow: bool) -> None:
        """Breaker and budget checks; raises CloudUnavailable, consumes a token on success."""
        if not self.enabled:
            raise CloudUnavailable("cloud escalation disabled", "disabled")
        if shadow:
            # Shadow traffic is optional: never probe a half-open breaker with it.
            if self.breaker.state != "closed":
                raise CloudUnavailable("circuit breaker not closed", "breaker")
            if not self.shadow_bucket.take():
                raise CloudUnavailable("shadow budget exhausted", "shadow_budget")
            return
        if not self.breaker.allow():
            raise CloudUnavailable("circuit breaker open", "breaker")
        key_bucket = self.key_buckets.get(caller)
        if not key_bucket.take():
            self.breaker.release()
            raise CloudUnavailable("per-key escalation budget exhausted", "key_budget")
        if not self.bucket.take():
            key_bucket.refund()
            self.breaker.release()
            raise CloudUnavailable("escalation budget exhausted", "budget")

    async def _guard(self, awaitable: Awaitable[Any]) -> Any:
        """Run one cloud call under the deadline, mapping failures for the breaker."""
        try:
            # One deadline over the SDK's own retries, kept below the ALB idle timeout so
            # the user gets a (degraded) answer instead of a 504 while we keep paying.
            resp = await asyncio.wait_for(awaitable, timeout=self.s.cloud_deadline_s)
        except asyncio.TimeoutError as exc:
            self.breaker.failure()
            raise CloudUnavailable(f"cloud deadline of {self.s.cloud_deadline_s}s exceeded", "deadline") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            # A bad or missing key is a config mistake, not an outage: label it so on-call
            # doesn't wait one out. Still trip the breaker; retrying won't fix it.
            self.breaker.failure()
            raise CloudUnavailable(f"cloud rejected credentials: {type(exc).__name__}: {exc}", "auth") from exc
        except anthropic.BadRequestError as exc:
            # A malformed request is our bug, not a cloud outage: don't trip the breaker.
            self.breaker.release()
            raise CloudUnavailable(f"cloud rejected request: {exc}", "rejected") from exc
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            self.breaker.failure()
            raise CloudUnavailable(f"cloud error: {type(exc).__name__}: {exc}") from exc
        except BaseException:
            self.breaker.release()  # e.g. cancelled: don't leave a probe stuck in flight
            raise
        self.breaker.success()
        return resp

    def _to_answer(self, resp: Any, t0: float) -> CloudAnswer:
        latency_ms = (time.perf_counter() - t0) * 1000
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        fallback_used = any(getattr(b, "type", None) == "fallback" for b in resp.content)
        iterations = getattr(resp.usage, "iterations", None) or []
        fallback_used = fallback_used or any(getattr(i, "type", None) == "fallback_message" for i in iterations)
        refused = resp.stop_reason == "refusal"
        if not refused and not text.strip():
            # e.g. adaptive thinking used the whole max_tokens budget. An empty string is
            # not an answer: let the caller fall back to the edge answer (or 503).
            raise CloudUnavailable(f"cloud returned no answer text (stop_reason={resp.stop_reason})", "empty")
        in_tok = int(getattr(resp.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(resp.usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(resp.usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(resp.usage, "cache_creation_input_tokens", 0) or 0)
        return CloudAnswer(
            text=text,
            model=getattr(resp, "model", self.model),
            stop_reason=resp.stop_reason or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd=estimate_cost(self.s, in_tok, out_tok, cache_read, cache_write),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            refused=refused,
            fallback_used=fallback_used,
        )

    async def answer(self, messages: List[ChatMessage], edge_draft: Optional[str] = None,
                     caller: str = "anon", shadow: bool = False) -> CloudAnswer:
        self._admit(caller, shadow)
        req = self._request(messages, None if shadow else edge_draft)
        t0 = time.perf_counter()
        resp = await self._guard(self.client.beta.messages.create(**req))
        return self._to_answer(resp, t0)

    async def stream_answer(self, messages: List[ChatMessage], edge_draft: Optional[str],
                            on_text: Callable[[str], Awaitable[None]], caller: str = "anon") -> CloudAnswer:
        """Like answer(), but forwards text to ``on_text`` as Claude writes it.

        The returned CloudAnswer (from the final message) is authoritative: with
        server-side refusal fallbacks, text streamed before a mid-stream switch to the
        fallback model can differ from the final answer."""
        self._admit(caller, shadow=False)
        req = self._request(messages, edge_draft)
        t0 = time.perf_counter()

        async def consume() -> Any:
            async with self.client.beta.messages.stream(**req) as stream:
                async for text in stream.text_stream:
                    if text:
                        await on_text(text)
                return await stream.get_final_message()

        resp = await self._guard(consume())
        return self._to_answer(resp, t0)

    async def aclose(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            await self.client.close()
