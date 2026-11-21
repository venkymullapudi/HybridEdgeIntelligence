import json

from fastapi.testclient import TestClient

from hybrid_edge.cloud.claude import ClaudeEscalator
from hybrid_edge.config import Settings
from hybrid_edge.server import build_orchestrator, create_app
from hybrid_edge.telemetry import Metrics, RecordSink

from .conftest import FakeClaude, MemorySink


class FakeFirehose:
    def __init__(self, fail_first=0):
        self.batches = []
        self.fail_first = fail_first

    def put_record_batch(self, DeliveryStreamName, Records):
        self.batches.append([r["Data"] for r in Records])
        if self.fail_first:
            self.fail_first -= 1
            return {"FailedPutCount": 1,
                    "RequestResponses": [{"ErrorCode": "ServiceUnavailable"}] + [{}] * (len(Records) - 1)}
        return {"FailedPutCount": 0, "RequestResponses": [{}] * len(Records)}


def test_firehose_records_are_batched():
    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_flush_interval_s": 0.2})
    fh = FakeFirehose()
    m = Metrics(s)
    rs = RecordSink(s, firehose_client=fh, metrics=m)
    for i in range(120):
        rs.submit({"i": i})
    rs.flush()
    assert sum(len(b) for b in fh.batches) == 120
    assert len(fh.batches) < 10  # batched, not one call per record
    assert json.loads(fh.batches[0][0].decode()) == {"i": 0}
    assert m.counters["records_written_total"] == 120


def test_firehose_partial_failure_is_retried():
    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_flush_interval_s": 0.05})
    fh = FakeFirehose(fail_first=1)
    rs = RecordSink(s, firehose_client=fh)
    for i in range(3):
        rs.submit({"i": i})
    rs.flush()
    assert len(fh.batches[-1]) == 1  # only the rejected record was resent


def test_full_queue_drops_and_counts_instead_of_blocking():
    s = Settings.from_dict({"record_sink": "firehose", "record_firehose_stream": "s", "emit_emf": False,
                            "record_queue_size": 1, "record_flush_interval_s": 0.05})

    class Stuck(FakeFirehose):
        def put_record_batch(self, **kw):
            import time
            time.sleep(0.3)
            return super().put_record_batch(**kw)

    m = Metrics(s)
    rs = RecordSink(s, firehose_client=Stuck(), metrics=m)
    for i in range(50):
        rs.submit({"i": i})
    assert m.counters["records_dropped_total"] > 0
    rs.flush()


def test_file_sink_and_text_redaction(tmp_path):
    path = tmp_path / "r.jsonl"
    s = Settings.from_dict({"record_sink": "file", "record_path": str(path), "record_include_text": False,
                            "record_flush_interval_s": 0.05})
    rs = RecordSink(s)
    rs.submit({"request_id": "x", "messages": ["secret"], "edge_answer": "e", "cloud_answer": "c"})
    rs.flush()
    rec = json.loads(path.read_text().strip())
    assert rec == {"request_id": "x"}


def test_request_size_limits(settings):
    settings.max_request_chars = 50
    settings.max_messages = 2
    orch = build_orchestrator(settings, escalator=ClaudeEscalator(settings, client=FakeClaude()),
                              records=MemorySink(settings))
    with TestClient(create_app(settings, orch)) as c:
        assert c.post("/v1/infer", json={"prompt": "x" * 51}).status_code == 413
        msgs = [{"role": "user", "content": "a"}] * 3
        assert c.post("/v1/infer", json={"messages": msgs}).status_code == 413
        assert c.post("/v1/infer", json={"prompt": "hi"}).status_code == 200
