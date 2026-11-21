import json
from types import SimpleNamespace

from hybrid_edge.config import Settings
from hybrid_edge.label import VERDICT_SCHEMA, eligible, judge_prompt, run


class JudgeClaude:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        v = self.verdicts.pop(0)
        if v == "refuse":
            return SimpleNamespace(stop_reason="refusal", content=[])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(v))])


def rec(i, **kw):
    base = {"request_id": str(i), "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
            "edge_answer": "e", "cloud_answer": "c"}
    base.update(kw)
    return base


def test_eligibility_and_prompt():
    assert eligible(rec(1))
    assert not eligible(rec(1, label=1))
    assert not eligible(rec(1, cloud_answer=None))
    assert not eligible(rec(1, cloud_refused=True))
    p = judge_prompt(rec(1))
    assert "<candidate_answer>\ne\n" in p and "[system]" not in p


async def test_run_labels_and_respects_limit():
    s = Settings.from_dict({"cloud_provider": "anthropic"})
    records = [rec(1), rec(2), rec(3), rec(4, label=0)]
    client = JudgeClaude([{"acceptable": True, "reason": "ok"}, "refuse"])
    stats = await run(records, s, client, concurrency=2, limit=2)
    assert stats == {"judged": 1, "skipped": 1, "errors": 0}
    assert sum(1 for r in records if r.get("label") == 1) == 1
    call = client.calls[0]
    assert call["output_config"]["format"]["schema"] == VERDICT_SCHEMA
    assert call["fallbacks"] == "default" and call["model"] == "claude-opus-5-5"
