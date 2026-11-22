"""Read escalation/shadow records the way the deployment writes them.

Firehose delivers many GZIP-compressed objects under
``escalations/region=<r>/YYYY/MM/DD/HH/``; the file sink writes plain JSONL. These
helpers take any mix of files, ``.gz`` files and directories (searched recursively),
so ``aws s3 sync s3://<bucket>/escalations/ data/`` followed by
``hybrid-edge-label data/ ...`` just works.

Malformed lines (a partial last line after a crash, a stray non-object) are skipped
and counted rather than aborting a long, possibly billed, run.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Sequence


class ReadStats:
    def __init__(self) -> None:
        self.files = 0
        self.records = 0
        self.skipped = 0

    def report(self) -> None:
        if self.skipped:
            print(f"warning: skipped {self.skipped} malformed record line(s) in {self.files} file(s)",
                  file=sys.stderr)


def record_files(paths: Sequence[str]) -> List[str]:
    """Expand directories recursively; keep files as given. Sorted for determinism."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                out.extend(os.path.join(root, n) for n in names if not n.startswith("."))
        else:
            out.append(p)
    return sorted(out)


def _open(path: str):
    with open(path, "rb") as probe:
        gz = probe.read(2) == b"\x1f\x8b"  # by content: Firehose objects have no .gz suffix
    return gzip.open(path, "rt", encoding="utf-8") if gz else open(path, "r", encoding="utf-8")


def iter_records(paths: Sequence[str], stats: ReadStats = None) -> Iterator[Dict[str, Any]]:
    stats = stats if stats is not None else ReadStats()
    for path in record_files(paths):
        stats.files += 1
        with _open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    stats.skipped += 1
                    continue
                if not isinstance(rec, dict):
                    stats.skipped += 1
                    continue
                stats.records += 1
                yield rec
