"""Backend for any OpenAI-compatible server that returns token logprobs.

Works with vLLM (recommended on AWS GPU edges), SGLang, llama.cpp server and
recent Ollama builds. The server hosts the open-weight model; this module only
speaks HTTP to it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx

from .base import ChatMessage, EdgeBackendError, Generation, TokenInfo


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 30.0):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        top_logprobs: int = 5,
    ) -> List[Generation]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise EdgeBackendError(f"edge model unreachable: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            raise EdgeBackendError(f"edge model HTTP {resp.status_code}: {resp.text[:300]}")
        return parse_chat_completion(resp.json(), latency_ms)

    async def ready(self) -> bool:
        try:
            resp = await self._client.get("/models", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_chat_completion(payload: Dict[str, Any], latency_ms: float = 0.0) -> List[Generation]:
    usage = payload.get("usage") or {}
    gens: List[Generation] = []
    for choice in payload.get("choices", []):
        content = ((choice.get("message") or {}).get("content")) or ""
        tokens: List[TokenInfo] = []
        lp = choice.get("logprobs") or {}
        for item in lp.get("content") or []:
            top = {t["token"]: float(t["logprob"]) for t in (item.get("top_logprobs") or [])}
            tokens.append(TokenInfo(token=item.get("token", ""), logprob=float(item.get("logprob", 0.0)), top=top))
        gens.append(
            Generation(
                text=content,
                tokens=tokens,
                finish_reason=choice.get("finish_reason") or "stop",
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", len(tokens))),
                latency_ms=latency_ms,
            )
        )
    if not gens:
        raise EdgeBackendError("edge model returned no choices")
    return gens
