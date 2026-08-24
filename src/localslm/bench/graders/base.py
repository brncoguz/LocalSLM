"""The grader contract."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult


@dataclass
class Grade:
    score: float  # 0..1, comparable across graders
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    pred_label: str | None = None
    gold_label: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    judged: bool = False


@dataclass
class GradeContext:
    """Everything a grader may need beyond the item and the output.

    Kept deliberately small: a grader that needs the network or a second model
    is a judge, and judges are opt-in.
    """

    retry_used: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Grader(Protocol):
    key: str
    requires_judge: bool

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade: ...

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]: ...


def normalise(text: str) -> str:
    """Case-fold, strip punctuation and collapse whitespace.

    Declared here rather than per-grader so "what counts as the same answer" is
    one auditable rule instead of six slightly different ones.
    """
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    # Curly quotes are the point here, not a typo: models emit them and the
    # expected answers use straight ones.
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"[^\w\s.'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Interior periods survive so decimals stay intact ("3.5"), but trailing
    # ones are stripped: a model answering "Positive." is not wrong, and
    # scoring it wrong would systematically penalise anything that punctuates.
    return text.strip(".-' ")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class BaseGrader:
    """Default aggregate: accuracy plus the item count."""

    requires_judge = False

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        return {
            "n": len(grades),
            "accuracy": mean([g.score for g in grades]),
            "pass_rate": mean([1.0 if g.passed else 0.0 for g in grades]),
        }
