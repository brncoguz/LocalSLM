"""Measuring how stable a model's output is.

Framed as a *measurement*, not a check. It is tempting to assume temperature 0
with a fixed seed is deterministic and write this as a confirmation, but on
Metal the reduction order in batched matmuls isn't guaranteed stable, so
identical inputs can still diverge. "Even at temperature 0, 6% of outputs
differed across 20 samples" is a more useful finding than the one you expected,
and it is the number that matters to anyone building an extraction pipeline.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from localslm.bench.graders.base import normalise


@dataclass(frozen=True)
class Stability:
    n: int
    distinct: int
    exact_duplicate_rate: float  # share of samples matching the modal output
    length_mean: float
    length_stdev: float
    score_mean: float | None = None
    score_stdev: float | None = None

    @property
    def fully_deterministic(self) -> bool:
        return self.distinct == 1

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "distinct": self.distinct,
            "exact_duplicate_rate": self.exact_duplicate_rate,
            "length_mean": self.length_mean,
            "length_stdev": self.length_stdev,
            "score_mean": self.score_mean,
            "score_stdev": self.score_stdev,
            "fully_deterministic": self.fully_deterministic,
        }


def stability(outputs: Sequence[str], scores: Sequence[float] | None = None) -> Stability:
    """Agreement across repeated samples of the same prompt.

    Comparison is on normalised text: a trailing newline is not a different
    answer, and counting it as one would overstate the instability.
    """
    texts = [normalise(o or "") for o in outputs]
    if not texts:
        return Stability(0, 0, 0.0, 0.0, 0.0)

    counts: dict[str, int] = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    modal = max(counts.values())
    lengths = [len(o or "") for o in outputs]

    score_mean = score_sd = None
    if scores:
        clean = [s for s in scores if s is not None]
        if clean:
            score_mean = statistics.fmean(clean)
            score_sd = statistics.stdev(clean) if len(clean) > 1 else 0.0

    return Stability(
        n=len(texts),
        distinct=len(counts),
        exact_duplicate_rate=modal / len(texts),
        length_mean=statistics.fmean(lengths),
        length_stdev=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        score_mean=score_mean,
        score_stdev=score_sd,
    )
