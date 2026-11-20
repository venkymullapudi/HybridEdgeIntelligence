"""Small in-process TTL + LRU cache for answers (escalated ones are the valuable hits)."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Generic, List, Optional, Tuple, TypeVar

from .backends.base import ChatMessage

V = TypeVar("V")


class TTLCache(Generic[V]):
    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: "OrderedDict[str, Tuple[float, V]]" = OrderedDict()

    def get(self, key: str) -> Optional[V]:
        item = self._data.get(key)
        if item is None:
            return None
        ts, val = item
        if time.monotonic() - ts > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return val

    def put(self, key: str, value: V) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def cache_key(messages: List[ChatMessage], *extra: Any) -> str:
    payload = json.dumps(
        {"m": [[m.role, " ".join(m.content.split())] for m in messages], "x": list(extra)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
