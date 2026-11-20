"""Deterministic mock edge model, for tests and laptop demos (no GPU needed).

Behaviour is driven by the prompt so the whole pipeline can be exercised:
  * prompts containing "hard", "prove", "legal", "diagnos" -> low-confidence tokens
  * prompts containing "unsure"                              -> hedged answer
  * anything else                                           -> high-confidence answer
"""

from __future__ import annotations

import hashlib
import math
from typing import List

from .base import ChatMessage, Generation, TokenInfo, estimate_tokens, last_user

HARD_MARKERS = ("hard", "prove", "legal", "diagnos", "derivative of", "theorem")


class MockBackend:
    name = "mock"

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        top_logprobs: int = 5,
    ) -> List[Generation]:
        prompt = (last_user(messages) or "").lower()
        hard = any(m in prompt for m in HARD_MARKERS)
        hedge = "unsure" in prompt
        out: List[Generation] = []
        for i in range(n):
            if hedge:
                text = "I'm not sure, but it might be 42."
            elif hard:
                # Different samples disagree on hard prompts.
                text = f"The answer is probably option {'ABCD'[i % 4]}."
            else:
                text = f"Edge answer: {prompt[:60]}".strip()
            words = text.split(" ")[:max_tokens]
            tokens = []
            for j, w in enumerate(words):
                seed = int(hashlib.sha256(f"{prompt}|{i}|{j}".encode()).hexdigest()[:8], 16)
                jitter = (seed % 100) / 1000.0
                p = (0.35 + jitter) if hard or hedge else (0.97 - jitter / 10)
                lp = math.log(p)
                rest = (1 - p) / 4
                top = {w: lp}
                for k in range(1, min(top_logprobs, 5)):
                    top[f"<alt{k}>"] = math.log(max(rest, 1e-6))
                tokens.append(TokenInfo(token=w, logprob=lp, top=top))
            out.append(
                Generation(
                    text=" ".join(words),
                    tokens=tokens,
                    finish_reason="stop",
                    prompt_tokens=estimate_tokens(messages),
                    completion_tokens=len(tokens),
                    latency_ms=5.0,
                )
            )
        return out

    async def ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
