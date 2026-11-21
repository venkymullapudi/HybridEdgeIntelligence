"""Edge model backend interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol


@dataclass
class TokenInfo:
    token: str
    logprob: float
    # Top-k alternatives at this position (token -> logprob), including the chosen one.
    top: Dict[str, float] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    tokens: List[TokenInfo]
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


AbortCheck = Callable[[List[TokenInfo]], bool]


class EdgeBackend(Protocol):
    name: str

    async def generate(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        n: int = 1,
        top_logprobs: int = 5,
    ) -> List[Generation]:
        """Return ``n`` generations with per-token logprobs."""
        ...

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_logprobs: int = 5,
        should_abort: Optional[AbortCheck] = None,
    ) -> Generation:
        """Stream one generation; stop early (finish_reason="aborted") when
        ``should_abort(tokens_so_far)`` returns True."""
        ...

    async def ready(self) -> bool:
        ...

    async def aclose(self) -> None:
        ...


class EdgeBackendError(RuntimeError):
    pass


def estimate_tokens(messages: List[ChatMessage]) -> int:
    """Cheap, tokenizer-free estimate (~4 chars/token) used for routing decisions only."""
    return sum(len(m.content) for m in messages) // 4 + 4 * len(messages)


def last_user(messages: List[ChatMessage]) -> Optional[str]:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return None


def conversation_problem(messages: List[ChatMessage]) -> Optional[str]:
    """Why a conversation can't be served, or None.

    Both tiers need the same shape: at most one system message, first; then turns
    ending with the user. A trailing assistant turn is a prefill, which current
    Claude models reject, and stray system turns break many edge chat templates."""
    if not messages:
        return "no messages"
    for i, m in enumerate(messages):
        if m.role == "system" and i != 0:
            return "a system message is only allowed as the first message"
    turns = messages[1:] if messages[0].role == "system" else messages
    if not turns:
        return "the conversation needs at least one user message"
    if turns[-1].role != "user":
        return "the last message must be from the user (assistant prefill is not supported)"
    return None
