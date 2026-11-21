"""Fit the confidence gate from logged records.

Input: JSONL records (what the record sink writes) with ``features`` and a label.
Labels come from, in order of preference:
  * ``label`` field (1 = edge answer acceptable, 0 = not), e.g. from human review
    or an offline LLM judge comparing ``edge_answer`` to ``cloud_answer``;
  * ``--weak-labels``: agreement between edge and cloud answers (token Jaccard).

Output: a JSON calibrator for ``HEI_GATE_CALIBRATION_PATH`` with a threshold chosen
to hit a target precision (fraction of edge-served answers that are acceptable).

    hybrid-edge-calibrate data/escalations.jsonl -o calibration.json --target-precision 0.95
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .gate.confidence import Calibrator, _sigmoid
from .gate.signals import jaccard

FEATURES = ["mean_prob", "min_window_prob", "entropy_conf", "consistency", "hedge", "truncated", "log_len"]


Row = Tuple[Dict[str, float], int, float]  # features, label, sample weight


def _row(r: Sequence) -> Row:
    return (r[0], int(r[1]), float(r[2]) if len(r) > 2 else 1.0)


def load(path: str, weak_labels: bool, weak_threshold: float) -> List[Row]:
    """Rows carry ``sample_weight`` so shadow records (sampled at ``shadow_rate``)
    count for the confident traffic they represent."""
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            feats = r.get("features")
            if not feats:
                continue
            label: Optional[int] = r.get("label")
            if label is None and weak_labels and r.get("edge_answer") and r.get("cloud_answer"):
                label = int(jaccard(r["edge_answer"], r["cloud_answer"]) >= weak_threshold)
            if label is None:
                continue
            rows.append((feats, int(label), float(r.get("sample_weight") or 1.0)))
    return rows


def fit(rows: Sequence[Sequence], epochs: int = 500, lr: float = 0.5,
        l2: float = 1e-3) -> Calibrator:
    data = [_row(r) for r in rows]
    w = {f: 0.0 for f in FEATURES}
    b = 0.0
    n = sum(wt for _, _, wt in data)
    for _ in range(epochs):
        gw = {f: 0.0 for f in FEATURES}
        gb = 0.0
        for x, y, wt in data:
            p = _sigmoid(b + sum(w[f] * x.get(f, 0.0) for f in FEATURES))
            err = (p - y) * wt
            for f in FEATURES:
                gw[f] += err * x.get(f, 0.0)
            gb += err
        for f in FEATURES:
            w[f] -= lr * (gw[f] / n + l2 * w[f])
        b -= lr * gb / n
    return Calibrator(weights=w, bias=b)


def choose_threshold(cal: Calibrator, rows: Sequence[Sequence],
                     target_precision: float) -> Tuple[float, Dict[str, float]]:
    """Lowest threshold whose edge-served subset meets the target precision (max edge coverage).

    Precision and coverage are weighted estimates over the *traffic*, not the sample."""
    scored = sorted(((cal.predict(x), y, wt) for x, y, wt in (_row(r) for r in rows)), key=lambda t: -t[0])
    total = sum(wt for _, _, wt in scored) or 1.0
    best_thr, best_cov, best_prec = 1.0, 0.0, 1.0
    tp = seen = 0.0
    for score, y, wt in scored:
        tp += y * wt
        seen += wt
        prec = tp / seen
        if prec >= target_precision:
            best_thr, best_cov, best_prec = score, seen / total, prec
    return best_thr, {"edge_coverage": best_cov, "edge_precision": best_prec, "escalation_rate": 1 - best_cov}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records")
    ap.add_argument("-o", "--output", default="calibration.json")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--weak-labels", action="store_true")
    ap.add_argument("--weak-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    rows = load(args.records, args.weak_labels, args.weak_threshold)
    if len(rows) < 20 or len({y for _, y, _ in rows}) < 2:
        raise SystemExit(f"need >= 20 labelled rows with both classes, got {len(rows)}")
    cal = fit(rows)
    thr, stats = choose_threshold(cal, rows, args.target_precision)
    cal.threshold = round(thr, 4)
    cal.dump(args.output)
    ll = -sum(y * math.log(max(cal.predict(x), 1e-9)) + (1 - y) * math.log(max(1 - cal.predict(x), 1e-9))
              for x, y, _ in rows) / len(rows)
    print(json.dumps({"rows": len(rows), "log_loss": round(ll, 4), "threshold": cal.threshold, **stats}, indent=2))


if __name__ == "__main__":
    main()
