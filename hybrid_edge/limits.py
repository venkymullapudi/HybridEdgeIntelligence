"""Reject oversized request bodies before they are read.

`max_request_chars` is checked on parsed messages, but by then the framework has
already buffered and JSON-decoded the whole body. Neither uvicorn nor the ALB caps
body size, so without this a client could stream gigabytes into gateway memory.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict

Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class BodySizeLimit:
    """Pure ASGI middleware: 413 when Content-Length or the streamed body exceeds max_bytes."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send, status=400, detail="invalid Content-Length")
                return

        seen = 0
        started = rejected = False

        async def limited_receive() -> Message:
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:  # chunked or lying Content-Length
                    # Answer 413 ourselves and tell the app the client went away. Raising
                    # instead would be caught by the framework's body parser and turned
                    # into a 400.
                    rejected = True
                    if not started:
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if rejected:
                return  # we've already answered; drop whatever the app tries to send
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)

    async def _reject(self, send: Send, status: int = 413, detail: str = "request body too large") -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})
