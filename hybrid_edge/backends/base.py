"""Edge model backend interface."""

from __future__ import annotations

import re
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
    """Cheap, tokenizer-free estimate used for routing decisions only.

    ~4 characters per token holds for English and code, but CJK and many other
    scripts run close to one token per character. Counting those at 4 chars/token
    underestimated long non-Latin prompts ~4x, so the edge was asked for more tokens
    than its context held, failed, and the request escalated at full cost."""
    total = 0
    for m in messages:
        ascii_chars = len(m.content.encode("ascii", "ignore"))  # C-speed count of ASCII chars
        total += ascii_chars // 4 + (len(m.content) - ascii_chars) + 4
    return total


def last_user(messages: List[ChatMessage]) -> Optional[str]:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return None


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


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
    for i, m in enumerate(messages):
        if _LONE_SURROGATE.search(m.content):
            # Valid JSON (\ud800) but not encodable as UTF-8: it would crash cache keys and
            # records, and could be used to make every upstream call fail on purpose.
            return f"message {i} ({m.role}) contains invalid Unicode (a lone surrogate)"
        if m.role != "system" and not m.content.strip():
            # Claude rejects empty content: catch it here as a 422, not as a degraded answer.
            return f"message {i} ({m.role}) is empty"
    if turns[-1].role != "user":
        return "the last message must be from the user (assistant prefill is not supported)"
    return None


def edge_messages(messages: List[ChatMessage], system_role: str) -> List[ChatMessage]:
    """Adapt a conversation to the edge model's chat template.

    ``native`` sends the system message as is. ``merge`` folds it into the first user
    turn, for models whose templates reject a system role (e.g. Gemma), which would
    otherwise fail every edge call and escalate everything."""
    if system_role != "merge" or not messages or messages[0].role != "system":
        return messages
    system, rest = messages[0], list(messages[1:])
    for i, m in enumerate(rest):
        if m.role == "user":
            rest[i] = ChatMessage("user", f"{system.content}\n\n{m.content}" if system.content else m.content)
            return rest
    return rest
