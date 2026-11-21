"""Metrics and escalation records.

* Metrics: CloudWatch Embedded Metric Format lines on stdout (ECS awslogs ships them,
  CloudWatch turns them into metrics with no agent), plus in-process counters for
  a Prometheus-style ``/metrics`` endpoint.
* Records: one JSON object per escalation (and a sample of confident requests),
  sent to Kinesis Firehose -> S3. These are the training/calibration data for the
  edge model: (prompt, edge answer, gate signals, Claude answer).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .config import Settings

log = logging.getLogger("hybrid_edge")


class Metrics:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self.counters: Dict[str, float] = defaultdict(float)
        self.latency_sum: Dict[str, float] = defaultdict(float)
        self.latency_count: Dict[str, int] = defaultdict(int)

    def observe(self, source: str, escalated: bool, degraded: bool, score: Optional[float],
                latency_ms: float, cloud_tokens: int = 0, cloud_cost: float = 0.0,
                avoided_cost: float = 0.0) -> None:
        with self._lock:
            self.counters["requests_total"] += 1
            self.counters[f"served_{source}_total"] += 1
            if escalated:
                self.counters["escalations_total"] += 1
            if degraded:
                self.counters["degraded_total"] += 1
            self.counters["cloud_tokens_total"] += cloud_tokens
            self.counters["cloud_cost_usd_total"] += cloud_cost
            self.counters["avoided_cost_usd_total"] += avoided_cost
            self.latency_sum[source] += latency_ms
            self.latency_count[source] += 1
        if self.s.emit_emf:
            self._emf(source, escalated, degraded, score, latency_ms, cloud_tokens, cloud_cost, avoided_cost)

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def _emf(self, source: str, escalated: bool, degraded: bool, score: Optional[float],
             latency_ms: float, cloud_tokens: int, cloud_cost: float, avoided_cost: float) -> None:
        metrics = [
            {"Name": "Requests", "Unit": "Count"},
            {"Name": "Escalations", "Unit": "Count"},
            {"Name": "Degraded", "Unit": "Count"},
            {"Name": "LatencyMs", "Unit": "Milliseconds"},
            {"Name": "CloudTokens", "Unit": "Count"},
            {"Name": "CloudCostUsd", "Unit": "None"},
            {"Name": "AvoidedCostUsd", "Unit": "None"},
        ]
        doc: Dict[str, Any] = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": self.s.metrics_namespace,
                    "Dimensions": [["Region"], ["Region", "Source"]],
                    "Metrics": metrics + ([{"Name": "Confidence", "Unit": "None"}] if score is not None else []),
                }],
            },
            "Region": self.s.region,
            "Source": source,
            "EdgeId": self.s.edge_id,
            "Requests": 1,
            "Escalations": 1 if escalated else 0,
            "Degraded": 1 if degraded else 0,
            "LatencyMs": round(latency_ms, 2),
            "CloudTokens": cloud_tokens,
            "CloudCostUsd": round(cloud_cost, 6),
            "AvoidedCostUsd": round(avoided_cost, 6),
        }
        if score is not None:
            doc["Confidence"] = round(score, 4)
        sys.stdout.write(json.dumps(doc) + "\n")
        sys.stdout.flush()

    def prometheus(self) -> str:
        lines: List[str] = []
        labels = f'region="{self.s.region}",edge_id="{self.s.edge_id}"'
        with self._lock:
            for k, v in sorted(self.counters.items()):
                lines.append(f"hybrid_edge_{k}{{{labels}}} {v}")
            for src, total in sorted(self.latency_sum.items()):
                cnt = self.latency_count[src]
                lines.append(f'hybrid_edge_latency_ms_sum{{{labels},source="{src}"}} {total:.3f}')
                lines.append(f'hybrid_edge_latency_ms_count{{{labels},source="{src}"}} {cnt}')
            req = self.counters.get("requests_total", 0.0)
            esc = self.counters.get("escalations_total", 0.0)
            lines.append(f"hybrid_edge_escalation_rate{{{labels}}} {(esc / req) if req else 0.0:.4f}")
        return "\n".join(lines) + "\n"


FIREHOSE_MAX_BATCH = 500            # PutRecordBatch limit (records)
FIREHOSE_MAX_BATCH_BYTES = 4 * 1024 * 1024
FIREHOSE_MAX_RECORD_BYTES = 1000 * 1024


class RecordSink:
    """Fire-and-forget writer; never blocks or fails a user request.

    Records go onto a bounded in-memory queue drained by one background thread
    that writes in batches (PutRecordBatch for Firehose). When the queue is full,
    new records are dropped and counted rather than slowing down serving.
    """

    def __init__(self, settings: Settings, firehose_client: Any = None, metrics: Optional[Metrics] = None):
        self.s = settings
        self.kind = settings.record_sink
        self.metrics = metrics
        self._firehose = firehose_client
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=max(1, settings.record_queue_size))
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        if self.kind == "firehose" and self._firehose is None:
            import boto3  # optional dependency

            self._firehose = boto3.client("firehose", region_name=settings.region if settings.region != "local" else None)
        if self.kind == "file":
            os.makedirs(os.path.dirname(os.path.abspath(settings.record_path)), exist_ok=True)

    def should_record(self, escalated: bool) -> bool:
        if self.kind == "none":
            return False
        return escalated or random.random() < self.s.record_sample_rate_confident

    def submit(self, record: Dict[str, Any]) -> None:
        if self.kind == "none":
            return
        if not self.s.record_include_text:
            record = {k: v for k, v in record.items() if k not in ("messages", "edge_answer", "cloud_answer")}
        line = json.dumps(record, ensure_ascii=False, default=str)
        if len(line.encode("utf-8")) > FIREHOSE_MAX_RECORD_BYTES:
            self._count("records_dropped_total")
            log.warning("dropping oversized record (%d bytes)", len(line))
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self._count("records_dropped_total")

    def flush(self, timeout_s: float = 5.0) -> None:
        """Stop the worker after writing everything queued (used on shutdown)."""
        with self._lock:
            t = self._thread
            self._thread = None
        if t is None:
            return
        try:
            self._queue.put(None, timeout=timeout_s)  # sentinel
        except queue.Full:
            return
        t.join(timeout_s)

    # ------------------------------------------------------------- internals
    def _count(self, name: str, n: float = 1.0) -> None:
        if self.metrics is not None:
            self.metrics.incr(name, n)

    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="record-sink", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch: List[str] = [first]
            size = len(first)
            stop = False
            deadline = time.monotonic() + self.s.record_flush_interval_s
            while len(batch) < FIREHOSE_MAX_BATCH and size < FIREHOSE_MAX_BATCH_BYTES - FIREHOSE_MAX_RECORD_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
                size += len(nxt)
            self._write_batch(batch)
            if stop:
                return

    def _write_batch(self, lines: List[str]) -> None:
        try:
            if self.kind == "stdout":
                sys.stdout.write("".join(line + "\n" for line in lines))
                sys.stdout.flush()
            elif self.kind == "file":
                with open(self.s.record_path, "a", encoding="utf-8") as fh:
                    fh.write("".join(line + "\n" for line in lines))
            elif self.kind == "firehose":
                self._put_firehose(lines)
            self._count("records_written_total", len(lines))
        except Exception:  # noqa: BLE001 - telemetry must never break serving
            log.exception("failed to write %d escalation records", len(lines))
            self._count("records_failed_total", len(lines))

    def _put_firehose(self, lines: List[str], attempts: int = 3) -> None:
        pending = lines
        for attempt in range(attempts):
            resp = self._firehose.put_record_batch(
                DeliveryStreamName=self.s.record_firehose_stream,
                Records=[{"Data": (line + "\n").encode("utf-8")} for line in pending],
            )
            if not resp.get("FailedPutCount"):
                return
            pending = [line for line, r in zip(pending, resp.get("RequestResponses", [])) if r.get("ErrorCode")]
            time.sleep(0.2 * (2 ** attempt))
        raise RuntimeError(f"firehose rejected {len(pending)} records after {attempts} attempts")
