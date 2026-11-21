"""Uncertainty signals computed from an edge generation.

All signals are mapped to [0, 1] where 1 means "confident".
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from ..backends.base import Generation, TokenInfo

HEDGE_PATTERNS = [
    r"\bi(?:'| a)m not (?:sure|certain)\b",
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"\bi cannot (?:determine|be certain)\b",
    r"\bi can'?t (?:determine|be certain|verify)\b",
    r"\bnot enough information\b",
    r"\bi(?:'| a)m unable to\b",
    r"\bas an ai\b",
    r"\bit(?:'s| is) (?:unclear|hard to say)\b",
    r"\bi might be wrong\b",
]
_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Signals:
    mean_prob: float = 1.0
    min_window_prob: float = 1.0
    entropy_conf: float = 1.0
    consistency: float = 1.0
    consistency_measured: bool = False
    hedge: bool = False
    truncated: bool = False
    aborted: bool = False
    n_tokens: int = 0
    has_logprobs: bool = True

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def vector(self) -> Dict[str, float]:
        """Feature vector used by the calibrated (logistic) gate."""
        return {
            "mean_prob": self.mean_prob,
            "min_window_prob": self.min_window_prob,
            "entropy_conf": self.entropy_conf,
            "consistency": self.consistency,
            "hedge": 1.0 if self.hedge else 0.0,
            "truncated": 1.0 if self.truncated else 0.0,
            "log_len": math.log1p(self.n_tokens),
        }


def mean_prob(tokens: Sequence[TokenInfo]) -> float:
    if not tokens:
        return 0.0
    return math.exp(sum(t.logprob for t in tokens) / len(tokens))


def min_window_prob(tokens: Sequence[TokenInfo], window: int) -> float:
    """Geometric-mean probability of the least confident window of ``window`` tokens.

    A long, mostly-boilerplate answer can have a high mean while a single key span
    (a number, a name) is a guess; this catches that.
    """
    if not tokens:
        return 0.0
    w = max(1, min(window, len(tokens)))
    lps = [t.logprob for t in tokens]
    cur = sum(lps[:w])
    worst = cur
    for i in range(w, len(lps)):
        cur += lps[i] - lps[i - w]
        worst = min(worst, cur)
    return math.exp(worst / w)


def entropy_confidence(tokens: Sequence[TokenInfo]) -> float:
    """1 - mean normalised entropy of the next-token distribution (from top-k)."""
    vals: List[float] = []
    for t in tokens:
        if not t.top:
            continue
        probs = [math.exp(lp) for lp in t.top.values()]
        mass = sum(probs)
        residual = max(0.0, 1.0 - mass)
        dist = probs + ([residual] if residual > 1e-9 else [])
        total = sum(dist)
        if total <= 0:
            continue
        dist = [p / total for p in dist]
        h = -sum(p * math.log(p) for p in dist if p > 0)
        hmax = math.log(len(dist)) if len(dist) > 1 else 1.0
        vals.append(1.0 - min(1.0, h / hmax))
    if not vals:
        return 1.0
    return sum(vals) / len(vals)


def normalise_answer(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(normalise_answer(a)), set(normalise_answer(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def self_consistency(primary: str, samples: Sequence[str]) -> float:
    """Mean lexical agreement between the primary answer and extra samples."""
    if not samples:
        return 1.0
    return sum(jaccard(primary, s) for s in samples) / len(samples)


def is_hedged(text: str) -> bool:
    return bool(_HEDGE_RE.search(text))


def compute_signals(primary: Generation, samples: Sequence[Generation], window: int) -> Signals:
    toks = primary.tokens
    has_lp = bool(toks)
    return Signals(
        mean_prob=mean_prob(toks) if has_lp else 0.5,
        min_window_prob=min_window_prob(toks, window) if has_lp else 0.5,
        entropy_conf=entropy_confidence(toks) if has_lp else 0.5,
        consistency=self_consistency(primary.text, [s.text for s in samples]),
        consistency_measured=bool(samples),
        hedge=is_hedged(primary.text),
        truncated=primary.finish_reason == "length",
        aborted=primary.finish_reason == "aborted",
        n_tokens=primary.completion_tokens or len(toks),
        has_logprobs=has_lp,
    )
