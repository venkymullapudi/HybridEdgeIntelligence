"""Design review question 60: persistent 400s and the quickstart's real API calls."""

import pathlib

import anthropic
import httpx

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator

from .conftest import FakeClaude, MemorySink


def bad_request(msg):
    return anthropic.BadRequestError(msg, response=httpx.Response(400, request=httpx.Request("POST", "https://x")),
                                     body={"type": "error", "error": {"type": "invalid_request_error", "message": msg}})


async def test_persistent_400_opens_the_breaker_instead_of_calling_forever(settings):
    settings.cloud_breaker_failures = 3
    fake = FakeClaude(error=bad_request("Your credit balance is too low to access the Anthropic API."))
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    for i in range(10):
        r = await o.infer([ChatMessage("user", f"hard {i}")], use_cache=False)
        assert r.degraded
    assert len(fake.calls) == 3  # then the breaker stops the calls
    assert o.escalator.breaker.state == "open"
    m = o.metrics.prometheus()
    assert 'reason="rejected"} 3.0' in m and 'reason="breaker"} 7.0' in m


async def test_one_off_400_is_reset_by_the_next_success(settings):
    settings.cloud_breaker_failures = 2
    fake = FakeClaude(error=bad_request("bad request"))
    esc = ClaudeEscalator(settings, client=fake)
    o = build_orchestrator(settings, escalator=esc, records=MemorySink(settings))
    await o.infer([ChatMessage("user", "hard 1")], use_cache=False)
    fake.error = None
    await o.infer([ChatMessage("user", "hard 2")], use_cache=False)
    assert esc.breaker.failures == 0 and esc.breaker.state == "closed"


def test_quickstart_does_not_escalate_with_ambient_credentials():
    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "HEI_EDGE_BACKEND=mock HEI_CLOUD_PROVIDER=none .venv/bin/hybrid-edge" in readme
    assert "finds credentials **on its own**" in readme
