"""Label escalation/shadow records with a Claude judge, for gate calibration.

For each record that has both an ``edge_answer`` and a ``cloud_answer`` and no
``label``, Claude decides whether the edge answer would have been acceptable,
using Claude's answer as the reference. The output is the same JSONL with
``label`` (1/0) and ``label_reason`` added, ready for ``hybrid-edge-calibrate``.

    hybrid-edge-label data/escalations.jsonl -o data/labelled.jsonl

Every judged record is a billed Claude call. ``--limit`` caps the spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

import anthropic

from .cloud.claude import build_claude_client, fallback_kwargs, model_id
from .config import Settings

JUDGE_SYSTEM = (
    "You grade answers from a small model against a reference answer from a stronger model. "
    "An answer is acceptable if a careful user would be satisfied with it: it is correct, "
    "answers what was asked, and does not omit anything essential the reference includes. "
    "Differences in wording, length or style do not matter. If the reference itself looks "
    "wrong, judge the small model's answer on its own merits."
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "acceptable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["acceptable", "reason"],
    "additionalProperties": False,
}


def judge_prompt(record: Dict[str, Any]) -> str:
    convo = "\n".join(f"[{m['role']}] {m['content']}" for m in record.get("messages") or [] if m["role"] != "system")
    return (
        f"<conversation>\n{convo}\n</conversation>\n\n"
        f"<reference_answer>\n{record['cloud_answer']}\n</reference_answer>\n\n"
        f"<candidate_answer>\n{record['edge_answer']}\n</candidate_answer>\n\n"
        "Is the candidate answer acceptable?"
    )


async def judge_one(client: Any, settings: Settings, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resp = await client.beta.messages.create(
        model=model_id(settings),
        max_tokens=4000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(record)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        **fallback_kwargs(settings),
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not text:
        return None
    verdict = json.loads(text)
    return {"label": int(bool(verdict["acceptable"])), "label_reason": verdict["reason"]}


def eligible(record: Dict[str, Any]) -> bool:
    return (record.get("label") is None and bool(record.get("edge_answer"))
            and bool(record.get("cloud_answer")) and not record.get("cloud_refused"))


async def run(records: List[Dict[str, Any]], settings: Settings, client: Any, concurrency: int,
              limit: Optional[int]) -> Dict[str, int]:
    sem = asyncio.Semaphore(concurrency)
    todo = [r for r in records if eligible(r)]
    if limit is not None:
        todo = todo[:limit]
    stats = {"judged": 0, "skipped": 0, "errors": 0}

    async def one(r: Dict[str, Any]) -> None:
        async with sem:
            try:
                out = await judge_one(client, settings, r)
            except (anthropic.APIError, json.JSONDecodeError, KeyError) as exc:
                print(f"judge error on {r.get('request_id')}: {exc}", file=sys.stderr)
                stats["errors"] += 1
                return
            if out is None:
                stats["skipped"] += 1
            else:
                r.update(out)
                stats["judged"] += 1

    await asyncio.gather(*(one(r) for r in todo))
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="max records to judge (cost cap)")
    args = ap.parse_args(argv)

    settings = Settings.load()
    if settings.cloud_provider == "none":
        raise SystemExit("set HEI_CLOUD_PROVIDER to anthropic or bedrock to run the judge")
    with open(args.records, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    async def go() -> Dict[str, int]:
        client = build_claude_client(settings)
        try:
            return await run(records, settings, client, args.concurrency, args.limit)
        finally:
            await client.close()

    stats = asyncio.run(go())
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    labelled = sum(1 for r in records if r.get("label") is not None)
    print(json.dumps({**stats, "total_records": len(records), "labelled_records": labelled}, indent=2))


if __name__ == "__main__":
    main()
