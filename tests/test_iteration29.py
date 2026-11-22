"""Design review question 61: invalid Unicode is a 422, not a 500."""

import json

import pytest
from fastapi.testclient import TestClient

from hybrid_edge.backends.base import ChatMessage, conversation_problem
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator, create_app

from .conftest import FakeClaude, MemorySink


@pytest.mark.parametrize("raw", ['"\\ud800"', '"hi \\udfff there"', '"\\udc00\\ud800"'])
def test_lone_surrogates_are_a_conversation_problem(raw):
    assert "invalid Unicode" in conversation_problem([ChatMessage("user", json.loads(raw))])


def test_paired_surrogates_and_cjk_are_fine():
    emoji = json.loads('"\\ud83d\\ude00"')  # a correctly paired surrogate = one real character
    assert conversation_problem([ChatMessage("user", f"{emoji} 中文 é")]) is None


@pytest.mark.parametrize("path", ["/v1/infer", "/v1/infer/stream"])
@pytest.mark.parametrize("body", [
    '{"prompt": "hard question \\ud800"}',
    '{"messages": [{"role": "system", "content": "\\udfff"}, {"role": "user", "content": "q"}]}',
])
def test_endpoints_return_422_not_500(settings, path, body):
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    with TestClient(create_app(settings, o)) as c:
        r = c.post(path, content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and "invalid Unicode" in r.json()["detail"]
    assert fake.calls == []
