"""Design review question 48: paid-for labels survive crashes."""

import json
from types import SimpleNamespace

import pytest

from hybrid_edge import label
from hybrid_edge.config import Settings


def rec(i):
    return {"request_id": str(i), "messages": [{"role": "user", "content": "q"}], "edge_answer": "e",
            "cloud_answer": "c"}


class Judge:
    def __init__(self, crash_after=None):
        self.n = 0
        self.crash_after = crash_after
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.n += 1
        if self.crash_after is not None and self.n > self.crash_after:
            raise KeyboardInterrupt  # operator hits Ctrl-C mid-run
        return SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="text", text=json.dumps({"acceptable": True, "reason": "ok"}))])

    async def close(self):
        return None


async def test_checkpoints_every_n_labels():
    records = [rec(i) for i in range(25)]
    writes = []
    await label.run(records, Settings.from_dict({}), Judge(), concurrency=1, limit=None,
                    checkpoint=lambda: writes.append(sum(1 for r in records if "label" in r)), checkpoint_every=10)
    assert writes == [10, 20]


def _input(tmp_path, n):
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(rec(i)) + "\n" for i in range(n)))
    return src


def test_crash_keeps_labels_already_paid_for(tmp_path, monkeypatch):
    src, out = _input(tmp_path, 30), tmp_path / "out.jsonl"
    monkeypatch.setattr(label, "build_claude_client", lambda s: Judge(crash_after=12))
    with pytest.raises(KeyboardInterrupt):
        label.main([str(src), "-o", str(out), "--concurrency", "1", "--checkpoint-every", "1000"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 30 and sum(1 for r in rows if r.get("label") == 1) == 12  # none lost
    assert not (tmp_path / "out.jsonl.tmp").exists()


def test_rerun_on_output_resumes_without_rejudging(tmp_path, monkeypatch):
    src, out = _input(tmp_path, 30), tmp_path / "out.jsonl"
    monkeypatch.setattr(label, "build_claude_client", lambda s: Judge(crash_after=12))
    with pytest.raises(KeyboardInterrupt):
        label.main([str(src), "-o", str(out), "--concurrency", "1"])
    judge = Judge()
    monkeypatch.setattr(label, "build_claude_client", lambda s: judge)
    label.main([str(out), "-o", str(out), "--concurrency", "1"])
    assert judge.n == 18  # only the unlabelled remainder was billed
    assert all(json.loads(line)["label"] == 1 for line in out.read_text().splitlines())
