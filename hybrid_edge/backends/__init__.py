from __future__ import annotations

from ..config import Settings
from .base import ChatMessage, EdgeBackend, EdgeBackendError, Generation, TokenInfo
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend


def build_backend(settings: Settings) -> EdgeBackend:
    if settings.edge_backend == "mock":
        return MockBackend()
    return OpenAICompatBackend(
        base_url=settings.edge_base_url,
        model=settings.edge_model,
        api_key=settings.edge_api_key,
        timeout_s=settings.edge_timeout_s,
    )


__all__ = [
    "ChatMessage",
    "EdgeBackend",
    "EdgeBackendError",
    "Generation",
    "TokenInfo",
    "MockBackend",
    "OpenAICompatBackend",
    "build_backend",
]
