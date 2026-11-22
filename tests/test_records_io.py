"""Design review question 65: the offline tools read what the deployment writes."""

import gzip
import json
import random
from types import SimpleNamespace

from hybrid_edge import label
from hybrid_edge.calibrate import main as calibrate_main
from hybrid_edge.records_io import ReadStats, iter_records

from .test_misc import _synthetic


def firehose_tree(root, rows_per_object=40, objects=4, region="eu-west-1"):
    """Mimic Firehose: GZIP objects with no .gz suffix under date partitions."""
    feats_and_labels = _synthetic(rows_per_object * objects, seed=3)
    for i in range(objects):
        d = root / "escalations" / f"region={region}" / "2026" / "11" / "22" / f"{i:02d}"
        d.mkdir(parents=True)
        lines = [json.dumps({"edge_model": "m", "features": f, "label": y, "request_id": f"{i}-{j}",
                             "messages": [{"role": "user", "content": "q"}], "edge_answer": "e", "cloud_answer": "c"})
                 for j, (f, y) in enumerate(feats_and_labels[i * rows_per_object:(i + 1) * rows_per_object])]
        with gzip.open(d / f"hybrid-edge-{region}-escalations-1-2026-11-22-{i:02d}-abc", "wt") as fh:
            fh.write("\n".join(lines) + "\n")
    return objects * rows_per_object


def test_reads_gzip_without_suffix_recursively(tmp_path):
    n = firehose_tree(tmp_path)
    stats = ReadStats()
    recs = list(iter_records([str(tmp_path)], stats))
    assert len(recs) == n and stats.files == 4 and stats.skipped == 0


def test_malformed_lines_are_skipped_and_counted(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text('{"a": 1}\n[1, 2]\nnot json\n\n{"b": 2}\n{"truncated": "mid-wri')
    stats = ReadStats()
    assert [r for r in iter_records([str(p)], stats)] == [{"a": 1}, {"b": 2}]
    assert stats.skipped == 3


def test_calibrate_runs_straight_on_a_synced_bucket(tmp_path, capsys):
    firehose_tree(tmp_path)
    (tmp_path / "local.jsonl").write_text('{"oops"\n')  # plus a broken plain file
    out = tmp_path / "cal.json"
    calibrate_main([str(tmp_path / "escalations"), str(tmp_path / "local.jsonl"), "-o", str(out)])
    assert json.loads(out.read_text())["edge_model"] == "m"
    assert "skipped 1 malformed" in capsys.readouterr().err


def test_label_reads_gzip_directories(tmp_path, monkeypatch):
    firehose_tree(tmp_path, rows_per_object=5, objects=2)
    for gz in (tmp_path / "escalations").rglob("*abc"):
        rows = [json.loads(l) for l in gzip.open(gz, "rt")]
        for r in rows:
            r.pop("label")
        with gzip.open(gz, "wt") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")

    class Judge:
        n = 0
        beta = None

        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        async def create(self, **kw):
            Judge.n += 1
            return SimpleNamespace(stop_reason="end_turn", content=[
                SimpleNamespace(type="text", text=json.dumps({"acceptable": random.random() < 2, "reason": "ok"}))])

        async def close(self):
            return None

    monkeypatch.setattr(label, "build_claude_client", lambda s: Judge())
    out = tmp_path / "labelled.jsonl"
    label.main([str(tmp_path / "escalations"), "-o", str(out), "--concurrency", "2"])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 10 and Judge.n == 10 and all(r["label"] == 1 for r in rows)
