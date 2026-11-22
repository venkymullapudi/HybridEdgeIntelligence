"""Backend for any OpenAI-compatible server that returns token logprobs.

Works with vLLM (recommended on AWS GPU edges), SGLang, llama.cpp server and
recent Ollama builds. The server hosts the open-weight model; this module only
speaks HTTP to it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import AbortCheck, ChatMessage, EdgeBackendError, Generation, TokenInfo


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 30.0):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
        )

    def _body(self, messages: List[ChatMessage], max_tokens: int, temperature: float, n: int,
              top_logprobs: int) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        top_logprobs: int = 5,
    ) -> List[Generation]:
        body = self._body(messages, max_tokens, temperature, n, top_logprobs)
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise EdgeBackendError(f"edge model unreachable: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            raise EdgeBackendError(f"edge model HTTP {resp.status_code}: {resp.text[:300]}")
        return parse_chat_completion(resp.json(), latency_ms)

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_logprobs: int = 5,
        should_abort: Optional[AbortCheck] = None,
    ) -> Generation:
        body = self._body(messages, max_tokens, temperature, 1, top_logprobs)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        acc = StreamAccumulator()
        t0 = time.perf_counter()
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise EdgeBackendError(f"edge model HTTP {resp.status_code}: {text[:300]}")
                async for line in resp.aiter_lines():
                    if acc.feed_sse_line(line):
                        break
                    if should_abort is not None and acc.new_tokens and should_abort(acc.tokens):
                        # Leaving the context manager closes the connection; vLLM then
                        # aborts the request and frees the GPU slot.
                        acc.finish_reason = "aborted"
                        break
        except httpx.HTTPError as exc:
            raise EdgeBackendError(f"edge model unreachable: {exc}") from exc
        return acc.result((time.perf_counter() - t0) * 1000)

    async def ready(self) -> bool:
        """The server is up *and* serves the configured model.

        A reachable server with a different model loaded (a typo'd model id, a
        half-finished swap) fails every generation with "model not found"; if
        readiness only checked reachability, the edge would look healthy while
        escalating everything."""
        try:
            resp = await self._client.get("/models", timeout=3.0)
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        try:
            served = {m.get("id") for m in resp.json().get("data", [])}
        except (ValueError, AttributeError):
            return False
        return self.model in served

    async def aclose(self) -> None:
        await self._client.aclose()


class StreamAccumulator:
    """Builds a Generation from OpenAI-style SSE chunks."""

    def __init__(self) -> None:
        self.text: List[str] = []
        self.tokens: List[TokenInfo] = []
        self.finish_reason = "stop"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.new_tokens = False

    def feed_sse_line(self, line: str) -> bool:
        """Consume one SSE line. Returns True at end of stream."""
        self.new_tokens = False
        line = line.strip()
        if not line.startswith("data:"):
            return False
        data = line[5:].strip()
        if data == "[DONE]":
            return True
        chunk = json.loads(data)
        usage = chunk.get("usage") or {}
        if usage:
            self.prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.completion_tokens = int(usage.get("completion_tokens", 0))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text.append(delta["content"])
            for item in ((choice.get("logprobs") or {}).get("content") or []):
                top = {t["token"]: float(t["logprob"]) for t in (item.get("top_logprobs") or [])}
                self.tokens.append(TokenInfo(token=item.get("token", ""), logprob=float(item.get("logprob", 0.0)), top=top))
                self.new_tokens = True
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return False

    def result(self, latency_ms: float) -> Generation:
        return Generation(
            text="".join(self.text),
            tokens=self.tokens,
            finish_reason=self.finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens or len(self.tokens),
            latency_ms=latency_ms,
        )


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
