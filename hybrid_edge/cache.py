"""Small in-process TTL + LRU cache for answers (escalated ones are the valuable hits)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

from .backends.base import ChatMessage

V = TypeVar("V")


class TTLCache(Generic[V]):
    """TTL + LRU, bounded by entry count *and* by approximate bytes.

    Entries vary from a few bytes to a long Claude answer, so a count limit alone
    doesn't bound memory."""

    def __init__(self, maxsize: int, ttl_s: float, max_bytes: int = 0,
                 sizeof: Optional[Callable[[V], int]] = None):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self.max_bytes = max_bytes
        self.sizeof = sizeof or (lambda v: 0)
        self.bytes = 0
        self._data: "OrderedDict[str, Tuple[float, V, int]]" = OrderedDict()

    def get(self, key: str) -> Optional[V]:
        item = self._data.get(key)
        if item is None:
            return None
        ts, val, size = item
        if time.monotonic() - ts > self.ttl_s:
            self._drop(key)
            return None
        self._data.move_to_end(key)
        return val

    def put(self, key: str, value: V) -> None:
        if self.maxsize <= 0:
            return
        size = self.sizeof(value)
        if self.max_bytes and size > self.max_bytes:
            return  # would evict everything else and still not fit
        if key in self._data:
            self._drop(key)
        self._data[key] = (time.monotonic(), value, size)
        self.bytes += size
        while len(self._data) > self.maxsize or (self.max_bytes and self.bytes > self.max_bytes):
            self._drop(next(iter(self._data)))

    def _drop(self, key: str) -> None:
        _, _, size = self._data.pop(key)
        self.bytes -= size

    def __len__(self) -> int:
        return len(self._data)


def cache_key(messages: List[ChatMessage], *extra: Any) -> str:
    payload = json.dumps(
        {"m": [[m.role, _normalise(m.content)] for m in messages], "x": list(extra)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    """Only differences that can't change the answer: line endings and outer whitespace.

    Inner whitespace is kept: indentation is meaning in code, YAML and tables, so
    prompts that differ only in spacing must not share a cached answer."""
    return text.replace("\r\n", "\n").strip()
