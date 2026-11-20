"""Metrics and escalation records.

* Metrics: CloudWatch Embedded Metric Format lines on stdout (ECS awslogs ships them,
  CloudWatch turns them into metrics with no agent), plus in-process counters for
  a Prometheus-style ``/metrics`` endpoint.
* Records: one JSON object per escalation (and a sample of confident requests),
  sent to Kinesis Firehose -> S3. These are the training/calibration data for the
  edge model: (prompt, edge answer, gate signals, Claude answer).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
                latency_ms: float, cloud_tokens: int = 0) -> None:
        with self._lock:
            self.counters["requests_total"] += 1
            self.counters[f"served_{source}_total"] += 1
            if escalated:
                self.counters["escalations_total"] += 1
            if degraded:
                self.counters["degraded_total"] += 1
            self.counters["cloud_tokens_total"] += cloud_tokens
            self.latency_sum[source] += latency_ms
            self.latency_count[source] += 1
        if self.s.emit_emf:
            self._emf(source, escalated, degraded, score, latency_ms, cloud_tokens)

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def _emf(self, source: str, escalated: bool, degraded: bool, score: Optional[float],
             latency_ms: float, cloud_tokens: int) -> None:
        metrics = [
            {"Name": "Requests", "Unit": "Count"},
            {"Name": "Escalations", "Unit": "Count"},
            {"Name": "Degraded", "Unit": "Count"},
            {"Name": "LatencyMs", "Unit": "Milliseconds"},
            {"Name": "CloudTokens", "Unit": "Count"},
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


class RecordSink:
    """Fire-and-forget writer; never blocks or fails a user request."""

    def __init__(self, settings: Settings, firehose_client: Any = None):
        self.s = settings
        self.kind = settings.record_sink
        self._firehose = firehose_client
        self._file_lock = threading.Lock()
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
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._write, line)
        except RuntimeError:
            self._write(line)

    def _write(self, line: str) -> None:
        try:
            if self.kind == "stdout":
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            elif self.kind == "file":
                with self._file_lock, open(self.s.record_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            elif self.kind == "firehose":
                self._firehose.put_record(
                    DeliveryStreamName=self.s.record_firehose_stream,
                    Record={"Data": (line + "\n").encode("utf-8")},
                )
        except Exception:  # noqa: BLE001 - telemetry must never break serving
            log.exception("failed to write escalation record")
