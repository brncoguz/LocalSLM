"""Confidence intervals and model comparison.

Two ideas do most of the work here.

Bootstrap rather than a normal approximation: scores are bounded in [0, 1] and
often skewed, so a mean ± 1.96·SE can produce intervals extending past 1.0.
Resampling makes no distributional assumption and handles small n honestly.

Paired comparison by default: models are run on identical items, so comparing
them item-by-item removes the variance caused by some questions simply being
harder. At n=40 that routinely halves the interval width, which is the
difference between "A beats B" being sayable and not.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def overlaps(self, other: Interval) -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi, "n": self.n}


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over the mean."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return Interval(0.0, 0.0, 0.0, 0)
    if len(clean) == 1:
        # One observation carries no information about spread. Returning a
        # zero-width interval would imply certainty we don't have, so the
        # bounds stay at the value and `n=1` warns the caller.
        return Interval(clean[0], clean[0], clean[0], 1)

    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(resamples):
        means.append(statistics.fmean(rng.choices(clean, k=n)))
    means.sort()
    tail = (1 - confidence) / 2
    return Interval(
        mean=statistics.fmean(clean),
        lo=means[int(tail * resamples)],
        hi=means[min(resamples - 1, int((1 - tail) * resamples))],
        n=n,
    )


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str
    delta: float  # mean(a) - mean(b)
    lo: float
    hi: float
    n: int
    paired: bool
    significant: bool

    @property
    def verdict(self) -> str:
        """Plain English, quoted verbatim in reports so the caveat travels with
        the number instead of being dropped in summary."""
        if not self.significant:
            return (
                f"{self.a} and {self.b} are not distinguishable on this data "
                f"(difference {self.delta:+.3f}, 95% CI [{self.lo:+.3f}, {self.hi:+.3f}], n={self.n})"
            )
        better, worse = (self.a, self.b) if self.delta > 0 else (self.b, self.a)
        # Report the interval in the direction of the stated winner. Printing
        # a negative range beside "beats by" reads as a contradiction.
        lo, hi = (self.lo, self.hi) if self.delta > 0 else (-self.hi, -self.lo)
        return (
            f"{better} beats {worse} by {abs(self.delta):.3f} "
            f"(95% CI [{lo:+.3f}, {hi:+.3f}], n={self.n})"
        )

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "delta": self.delta,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            "paired": self.paired,
            "significant": self.significant,
            "verdict": self.verdict,
        }


def paired_bootstrap(
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    a_name: str = "a",
    b_name: str = "b",
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Comparison:
    """Compare two models on the items they both attempted.

    Inputs are {item_id: score}. Items only one model saw are dropped rather
    than filled in, because comparing different question sets is not a comparison.
    """
    shared = sorted(set(a_scores) & set(b_scores))
    if not shared:
        return Comparison(a_name, b_name, 0.0, 0.0, 0.0, 0, True, False)

    diffs = [a_scores[i] - b_scores[i] for i in shared]
    interval = bootstrap_ci(diffs, resamples, confidence, seed)
    # Significant when the interval for the difference excludes zero.
    significant = interval.lo > 0 or interval.hi < 0
    return Comparison(
        a=a_name,
        b=b_name,
        delta=interval.mean,
        lo=interval.lo,
        hi=interval.hi,
        n=len(shared),
        paired=True,
        significant=significant,
    )


def rank_with_ties(
    intervals: dict[str, Interval],
) -> list[tuple[str, Interval, bool]]:
    """Order by mean, flagging each entry that cannot be separated from the one
    above it. Lets the UI grey out distinctions the data doesn't support instead
    of printing a leaderboard that implies precision it hasn't earned."""
    ordered = sorted(intervals.items(), key=lambda kv: kv[1].mean, reverse=True)
    out = []
    for idx, (name, interval) in enumerate(ordered):
        tied = idx > 0 and interval.overlaps(ordered[idx - 1][1])
        out.append((name, interval, tied))
    return out
