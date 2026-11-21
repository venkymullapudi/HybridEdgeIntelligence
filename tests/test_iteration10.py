"""Design review question 34: memory bounds."""

import threading

from hybrid_edge.backends.base import ChatMessage
from hybrid_edge.cache import TTLCache
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.server import build_orchestrator
from hybrid_edge.telemetry import Metrics, RecordSink

from .conftest import FakeClaude, MemorySink


def test_cache_evicts_by_bytes():
    c = TTLCache(maxsize=1000, ttl_s=100, max_bytes=100, sizeof=len)
    for k in "abcde":
        c.put(k, "x" * 30)
    assert c.bytes <= 100 and len(c) == 3
    assert c.get("a") is None and c.get("e") is not None
    c.put("e", "y" * 10)           # replacing an entry frees its old size
    assert c.bytes == 70
    c.put("huge", "z" * 1000)      # bigger than the whole budget: not cached, nothing evicted
    assert c.get("huge") is None and len(c) == 3


async def test_orchestrator_cache_stays_within_byte_budget(settings):
    settings.cache_max_bytes = 50_000
    fake = FakeClaude(text="A" * 5_000)  # long Claude answers
    o = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=fake), records=MemorySink(settings))
    for i in range(40):
        await o.infer([ChatMessage("user", f"hard question {i}")])
    assert 0 < len(o.cache) < 40 and o.cache.bytes <= 50_000


def test_record_queue_drops_by_bytes_and_releases_on_write():
    gate = threading.Event()

    class Slow:
        def put_record_batch(self, DeliveryStreamName, Records):
            gate.wait(2)
            return {"FailedPutCount": 0, "RequestResponses": [{}] * len(Records)}

    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_queue_max_bytes": 5_000, "record_flush_interval_s": 0.01})
    m = Metrics(s)
    rs = RecordSink(s, firehose_client=Slow(), metrics=m)
    for i in range(20):
        rs.submit({"i": i, "pad": "p" * 1000})  # ~1 KB each against a 5 KB budget
    assert m.counters["records_dropped_total"] >= 10
    assert rs.queued_bytes <= 5_000
    gate.set()
    rs.flush()
    assert rs.queued_bytes == 0  # every byte accounted for once written
