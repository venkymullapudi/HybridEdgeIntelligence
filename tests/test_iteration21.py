"""Design review question 49: bad credentials are labelled, not mistaken for an outage."""

import anthropic
import httpx
import pytest

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator, CloudUnavailable
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def status_error(cls, code):
    return cls("nope", response=httpx.Response(code, request=httpx.Request("POST", "https://x")), body=None)


@pytest.mark.parametrize("cls,code", [(anthropic.AuthenticationError, 401), (anthropic.PermissionDeniedError, 403)])
async def test_credential_errors_get_auth_reason_and_trip_breaker(settings, cls, code):
    settings.cloud_breaker_failures = 1
    esc = ClaudeEscalator(settings, client=FakeClaude(error=status_error(cls, code)))
    with pytest.raises(CloudUnavailable) as e:
        await esc.answer([ChatMessage("user", "q")])
    assert e.value.reason == "auth" and esc.breaker.state == "open"


async def test_auth_reason_reaches_metrics(settings):
    o = build_orchestrator(settings, escalator=ClaudeEscalator(
        settings, client=FakeClaude(error=status_error(anthropic.AuthenticationError, 401))),
        records=MemorySink(settings))
    r = await o.infer([ChatMessage("user", "hard question")])
    assert r.degraded and 'reason="auth"} 1.0' in o.metrics.prometheus()


async def test_server_errors_are_still_plain_errors(settings):
    esc = ClaudeEscalator(settings, client=FakeClaude(error=status_error(anthropic.InternalServerError, 500)))
    with pytest.raises(CloudUnavailable) as e:
        await esc.answer([ChatMessage("user", "q")])
    assert e.value.reason == "error"
