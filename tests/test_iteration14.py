"""Design review questions 40-41."""

import anthropic
import httpx

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.server import build_orchestrator
from hybrid_edge.telemetry import RecordSink

from .conftest import FakeClaude, MemorySink


def user(t):
    return [ChatMessage("user", t)]


# --- Q40: no shadow spend that can't become calibration data ------------------
async def test_shadow_skipped_without_a_record_sink(settings, caplog):
    settings.shadow_rate = 1.0
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake),
                           records=RecordSink(settings))  # record_sink = "none"
    await o.infer(user("easy question"))
    await o.drain()
    assert fake.calls == []
    assert "shadow_rate=1.0 ignored" in caplog.text


async def test_shadow_skipped_without_record_text(settings):
    settings.shadow_rate = 1.0
    settings.record_include_text = False
    fake = FakeClaude()
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    await o.infer(user("easy question"))
    await o.drain()
    assert fake.calls == []


# --- Q41: why the cloud was unavailable, by reason ----------------------------
async def test_cloud_unavailable_is_counted_by_reason(settings):
    settings.cloud_max_escalations_per_key_per_min = 1
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    await o.infer(user("hard one"), use_cache=False)
    await o.infer(user("hard two"), use_cache=False)  # key budget

    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    settings.cloud_max_escalations_per_key_per_min = 0
    settings.cloud_breaker_failures = 1
    o2 = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude(error=err)),
                            records=MemorySink(settings))
    await o2.infer(user("hard a"), use_cache=False)  # error, opens breaker
    await o2.infer(user("hard b"), use_cache=False)  # breaker

    m1, m2 = o.metrics.prometheus(), o2.metrics.prometheus()
    assert 'hybrid_edge_cloud_unavailable_total{region="local",edge_id="edge-local",reason="key_budget"} 1.0' in m1
    assert 'reason="error"} 1.0' in m2 and 'reason="breaker"} 1.0' in m2
