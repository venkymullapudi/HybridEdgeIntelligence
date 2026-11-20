"""Escalation target: Claude, via the Claude API or Amazon Bedrock.

Only reached when the confidence gate rejects the edge answer, so every call
here is a deliberate spend. Includes:
  * refusal fallbacks (server-side on the Claude API, client-side middleware on Bedrock)
  * a circuit breaker, so a cloud outage degrades to edge answers instead of timeouts
  * a token-bucket budget, so a misconfigured gate can't run up an unbounded bill
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic

from ..backends.base import ChatMessage
from ..config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Bedrock has no server-side `fallbacks`; the client-side middleware needs an explicit model.
BEDROCK_FALLBACK_MODEL = "anthropic.claude-opus-4-8"


class CloudUnavailable(RuntimeError):
    """Cloud escalation was skipped or failed; caller should degrade gracefully."""


@dataclass
class CloudAnswer:
    text: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    refused: bool = False
    fallback_used: bool = False


class TokenBucket:
    def __init__(self, per_minute: int):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.last = time.monotonic()

    def take(self) -> bool:
        if self.capacity <= 0:
            return True  # unlimited
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class CircuitBreaker:
    def __init__(self, failures: int, cooldown_s: float):
        self.threshold = failures
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: Optional[float] = None

    @property
    def open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            return False  # half-open: let one request probe
        return True

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


class ClaudeEscalator:
    def __init__(self, settings: Settings, client: Any = None):
        self.s = settings
        self.provider = settings.cloud_provider
        self.model = settings.cloud_model
        if self.provider == "bedrock" and not self.model.startswith("anthropic."):
            self.model = "anthropic." + self.model
        self.client = client if client is not None else self._build_client()
        self.bucket = TokenBucket(settings.cloud_max_escalations_per_min)
        self.breaker = CircuitBreaker(settings.cloud_breaker_failures, settings.cloud_breaker_cooldown_s)
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def _build_client(self) -> Any:
        if self.provider == "none":
            return None
        if self.provider == "bedrock":
            middleware = []
            if self.s.cloud_refusal_fallbacks:
                middleware.append(anthropic.BetaRefusalFallbackMiddleware([{"model": BEDROCK_FALLBACK_MODEL}]))
            return anthropic.AsyncAnthropicBedrockMantle(
                aws_region=self.s.cloud_aws_region or self.s.region,
                timeout=self.s.cloud_timeout_s,
                middleware=middleware,
            )
        # Credentials resolve from ANTHROPIC_API_KEY (injected from Secrets Manager on AWS).
        return anthropic.AsyncAnthropic(timeout=self.s.cloud_timeout_s, max_retries=2)

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
        if self.s.cloud_refusal_fallbacks and self.provider == "anthropic":
            req["betas"] = [FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    async def answer(self, messages: List[ChatMessage], edge_draft: Optional[str] = None) -> CloudAnswer:
        if not self.enabled:
            raise CloudUnavailable("cloud escalation disabled")
        if self.breaker.open:
            raise CloudUnavailable("circuit breaker open")
        async with self._lock:
            allowed = self.bucket.take()
        if not allowed:
            raise CloudUnavailable("escalation budget exhausted")

        req = self._request(messages, edge_draft)
        t0 = time.perf_counter()
        try:
            resp = await self.client.beta.messages.create(**req)
        except anthropic.BadRequestError as exc:
            # A malformed request is our bug, not a cloud outage: don't trip the breaker.
            raise CloudUnavailable(f"cloud rejected request: {exc}") from exc
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            self.breaker.failure()
            raise CloudUnavailable(f"cloud error: {type(exc).__name__}: {exc}") from exc
        self.breaker.success()
        latency_ms = (time.perf_counter() - t0) * 1000

        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        fallback_used = any(getattr(b, "type", None) == "fallback" for b in resp.content)
        iterations = getattr(resp.usage, "iterations", None) or []
        fallback_used = fallback_used or any(getattr(i, "type", None) == "fallback_message" for i in iterations)
        refused = resp.stop_reason == "refusal"
        return CloudAnswer(
            text=text,
            model=getattr(resp, "model", self.model),
            stop_reason=resp.stop_reason or "",
            input_tokens=int(getattr(resp.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(resp.usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
            refused=refused,
            fallback_used=fallback_used,
        )

    async def aclose(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            await self.client.close()
