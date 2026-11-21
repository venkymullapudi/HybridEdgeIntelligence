"""Deterministic mock edge model, for tests and laptop demos (no GPU needed).

Behaviour is driven by the prompt so the whole pipeline can be exercised:
  * "hard", "prove", "legal", "diagnos", ... -> low-confidence tokens, samples disagree
  * "unsure"                                 -> hedged answer
  * "essay"                                  -> long, low-confidence answer (early-abort path)
  * "slow"                                   -> sleeps ``slow_s`` first (deadline path)
  * anything else                            -> high-confidence answer
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import List, Optional

from .base import AbortCheck, ChatMessage, Generation, TokenInfo, estimate_tokens, last_user

HARD_MARKERS = ("hard", "prove", "legal", "diagnos", "derivative of", "theorem", "essay")


class MockBackend:
    name = "mock"

    def __init__(self, slow_s: float = 1.0):
        self.slow_s = slow_s
        self.calls = 0

    def _sample(self, prompt: str, i: int, max_tokens: int, top_logprobs: int) -> Generation:
        hard = any(m in prompt for m in HARD_MARKERS)
        hedge = "unsure" in prompt
        if hedge:
            text = "I'm not sure, but it might be 42."
        elif "essay" in prompt:
            text = " ".join(f"word{k}" for k in range(200))
        elif hard:
            text = f"The answer is probably option {'ABCD'[i % 4]}."
        else:
            text = f"Edge answer: {prompt[:60]}".strip()
        words = text.split(" ")[:max_tokens]
        tokens = []
        for j, w in enumerate(words):
            seed = int(hashlib.sha256(f"{prompt}|{i}|{j}".encode()).hexdigest()[:8], 16)
            jitter = (seed % 100) / 1000.0
            p = (0.05 + jitter) if "essay" in prompt else ((0.35 + jitter) if hard or hedge else (0.97 - jitter / 10))
            lp = math.log(p)
            rest = (1 - p) / 4
            top = {w: lp}
            for k in range(1, min(top_logprobs, 5)):
                top[f"<alt{k}>"] = math.log(max(rest, 1e-6))
            tokens.append(TokenInfo(token=w, logprob=lp, top=top))
        return Generation(
            text=" ".join(words),
            tokens=tokens,
            finish_reason="length" if len(text.split(" ")) > max_tokens else "stop",
            completion_tokens=len(tokens),
            latency_ms=5.0,
        )

    async def _maybe_sleep(self, prompt: str) -> None:
        if "slow" in prompt:
            await asyncio.sleep(self.slow_s)

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        top_logprobs: int = 5,
    ) -> List[Generation]:
        self.calls += 1
        prompt = (last_user(messages) or "").lower()
        await self._maybe_sleep(prompt)
        out = []
        for i in range(n):
            g = self._sample(prompt, i, max_tokens, top_logprobs)
            g.prompt_tokens = estimate_tokens(messages)
            out.append(g)
        return out

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_logprobs: int = 5,
        should_abort: Optional[AbortCheck] = None,
    ) -> Generation:
        self.calls += 1
        prompt = (last_user(messages) or "").lower()
        await self._maybe_sleep(prompt)
        full = self._sample(prompt, 0, max_tokens, top_logprobs)
        full.prompt_tokens = estimate_tokens(messages)
        if should_abort is None:
            return full
        for k in range(1, len(full.tokens) + 1):
            if should_abort(full.tokens[:k]):
                full.tokens = full.tokens[:k]
                full.text = " ".join(t.token for t in full.tokens)
                full.completion_tokens = k
                full.finish_reason = "aborted"
                break
        return full

    async def ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
