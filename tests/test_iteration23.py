"""Design review question 53: oversized bodies are rejected before they're read."""

import json

from fastapi.testclient import TestClient

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.limits import BodySizeLimit
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


def test_declared_oversized_body_gets_413_without_reaching_handlers(settings):
    settings.max_body_bytes = 1000
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        r = c.post("/v1/infer", content=json.dumps({"prompt": "hard " * 400}),
                   headers={"content-type": "application/json"})
        assert r.status_code == 413 and r.json()["detail"] == "request body too large"
        assert c.post("/v1/infer", json={"prompt": "hard question"}).status_code == 200
    assert len(fake.calls) == 1  # only the small request reached a model


async def _drive(app, chunks, headers=()):
    sent, called = [], []

    async def inner(scope, receive, send):
        while True:
            m = await receive()
            if m["type"] == "http.disconnect":
                return  # like Starlette: ClientDisconnect, the handler never runs
            if not m.get("more_body"):
                break
        called.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    msgs = iter([{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)])

    async def receive():
        return next(msgs)

    async def send(m):
        sent.append(m)

    await BodySizeLimit(inner, max_bytes=100)({"type": "http", "headers": list(headers)}, receive, send)
    return sent[0]["status"], bool(called)


async def test_chunked_body_is_cut_off_while_streaming():
    status, called = await _drive(None, [b"x" * 60, b"x" * 60, b"x" * 60])  # no Content-Length
    assert status == 413 and not called


async def test_small_chunked_body_passes():
    status, called = await _drive(None, [b"x" * 40, b"x" * 40])
    assert status == 200 and called


async def test_lying_content_length_is_still_capped():
    status, called = await _drive(None, [b"x" * 80, b"x" * 80], headers=[(b"content-length", b"10")])
    assert status == 413 and not called


async def test_bad_content_length_is_400():
    status, _ = await _drive(None, [b"{}"], headers=[(b"content-length", b"lots")])
    assert status == 400


def test_default_cap_never_cuts_off_a_legal_request():
    s = Settings.from_dict({"max_request_chars": 1000})
    # Worst case: every char sent as a 6-byte \\uXXXX escape.
    body = json.dumps({"prompt": "中" * 1000}, ensure_ascii=True).encode()
    assert len(body) <= s.effective_max_body_bytes
    assert Settings.from_dict({"max_body_bytes": 5}).effective_max_body_bytes == 5
