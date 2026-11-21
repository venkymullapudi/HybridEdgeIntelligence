"""Design review questions 44-45: bytes vs characters."""

import json

from hybrid_edge.backends.base import ChatMessage, estimate_tokens
from hybrid_edge.backends.mock import MockBackend
from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.server import build_orchestrator
from hybrid_edge.telemetry import FIREHOSE_MAX_BATCH_BYTES, Metrics, RecordSink

from .conftest import FakeClaude, MemorySink


# --- Q44: Firehose limits are bytes ------------------------------------------
class SizeCheckingFirehose:
    def __init__(self):
        self.batches = []

    def put_record_batch(self, DeliveryStreamName, Records):
        total = sum(len(r["Data"]) for r in Records)
        assert total <= FIREHOSE_MAX_BATCH_BYTES, f"batch of {total} bytes would be rejected"
        self.batches.append(Records)
        return {"FailedPutCount": 0, "RequestResponses": [{}] * len(Records)}


def test_cjk_records_are_batched_within_firehose_byte_limit():
    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_flush_interval_s": 0.5, "record_queue_max_bytes": 64 * 1024 * 1024})
    fh = SizeCheckingFirehose()
    m = Metrics(s)
    rs = RecordSink(s, firehose_client=fh, metrics=m)
    text = "答" * 100_000  # 100k chars, ~300 KB of UTF-8
    for i in range(40):     # ~12 MB in total: must take several batches
        rs.submit({"i": i, "cloud_answer": text})
    rs.flush()
    assert m.counters["records_written_total"] == 40 and not m.counters.get("records_failed_total")
    assert len(fh.batches) >= 3
    assert json.loads(fh.batches[0][0]["Data"])["cloud_answer"] == text  # round-trips intact


def test_queue_budget_counts_utf8_bytes():
    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_queue_max_bytes": 10_000, "record_flush_interval_s": 5})

    class Never:
        def put_record_batch(self, **kw):
            raise AssertionError("not reached in this test")

    m = Metrics(s)
    rs = RecordSink(s, firehose_client=Never(), metrics=m)
    rs._ensure_worker = lambda: None  # keep everything queued
    rs.submit({"t": "é" * 2_000})      # ~4 KB of bytes, 2k chars
    rs.submit({"t": "é" * 2_000})
    rs.submit({"t": "é" * 2_000})      # would fit by characters, not by bytes
    assert m.counters["records_dropped_total"] == 1 and rs.queued_bytes <= 10_000


# --- Q45: token estimates for non-Latin scripts -------------------------------
def test_estimate_tokens_counts_cjk_near_one_per_char():
    en = estimate_tokens([ChatMessage("user", "word " * 400)])   # 2000 ASCII chars
    zh = estimate_tokens([ChatMessage("user", "中" * 2000)])     # 2000 CJK chars
    assert 490 <= en <= 520
    assert zh >= 2000
    mixed = estimate_tokens([ChatMessage("user", "abcd" * 10 + "日本" * 5)])
    assert mixed == 10 + 10 + 4


async def test_long_cjk_prompt_fits_edge_answer_to_real_room(settings):
    settings.edge_context_tokens = 4000

    class Recording(MockBackend):
        async def generate_stream(self, messages, *, max_tokens, **kw):
            self.seen = max_tokens
            return await super().generate_stream(messages, max_tokens=max_tokens, **kw)

    backend = Recording()
    o = build_orchestrator(settings, backend=backend, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                           records=MemorySink(settings))
    await o.infer([ChatMessage("user", "请总结" * 1000)], max_tokens=2000)  # ~3000 tokens of CJK
    # Old estimate (~750 tokens) would have asked for all 2000; the real room is ~1000.
    assert backend.seen < 1100
