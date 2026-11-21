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


def edge_models(path: str) -> Dict[Optional[str], int]:
    counts: Dict[Optional[str], int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line).get("edge_model")
                counts[m] = counts.get(m, 0) + 1
    return counts


def load(path: str, weak_labels: bool, weak_threshold: float, edge_model: Optional[str] = None) -> List[Row]:
    """Rows carry ``sample_weight`` so shadow records (sampled at ``shadow_rate``)
    count for the confident traffic they represent."""
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if edge_model is not None and r.get("edge_model") != edge_model:
                continue
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


def sweep(cal: Calibrator, rows: Sequence[Sequence], step: float = 0.05) -> List[Dict[str, float]]:
    """Edge coverage and precision (traffic-weighted) if the gate used each threshold."""
    data = [(cal.predict(x), y, wt) for x, y, wt in (_row(r) for r in rows)]
    total = sum(wt for _, _, wt in data) or 1.0
    out: List[Dict[str, float]] = []
    steps = int(round(1 / step))
    for i in range(steps + 1):
        thr = round(i * step, 4)
        served = [(y, wt) for p, y, wt in data if p >= thr]
        w = sum(wt for _, wt in served)
        good = sum(y * wt for y, wt in served)
        out.append({"threshold": thr, "edge_coverage": round(w / total, 4),
                    "edge_precision": round(good / w, 4) if w else None, "escalation_rate": round(1 - w / total, 4)})
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records")
    ap.add_argument("-o", "--output", default="calibration.json")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--weak-labels", action="store_true")
    ap.add_argument("--weak-threshold", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true",
                    help="also print coverage/precision at each threshold, to choose --target-precision")
    ap.add_argument("--edge-model", default=None,
                    help="fit only on records from this edge model (required if records mix models)")
    args = ap.parse_args(argv)

    models = edge_models(args.records)
    edge_model = args.edge_model
    if edge_model is None:
        if len(models) > 1:
            raise SystemExit(f"records mix edge models {sorted(map(str, models))}; pick one with --edge-model")
        edge_model = next(iter(models), None)

    rows = load(args.records, args.weak_labels, args.weak_threshold, edge_model)
    if len(rows) < 20 or len({y for _, y, _ in rows}) < 2:
        raise SystemExit(f"need >= 20 labelled rows with both classes, got {len(rows)}")
    cal = fit(rows)
    thr, stats = choose_threshold(cal, rows, args.target_precision)
    cal.threshold = round(thr, 4)
    cal.edge_model = edge_model
    cal.dump(args.output)
    ll = -sum(y * math.log(max(cal.predict(x), 1e-9)) + (1 - y) * math.log(max(1 - cal.predict(x), 1e-9))
              for x, y, _ in rows) / len(rows)
    report = {"edge_model": edge_model, "rows": len(rows), "log_loss": round(ll, 4), "threshold": cal.threshold, **stats}
    if args.sweep:
        report["sweep"] = sweep(cal, rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
