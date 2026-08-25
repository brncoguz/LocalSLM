"""Exact-match and numeric graders."""

from __future__ import annotations

import re
from collections.abc import Sequence

from localslm.bench.graders.base import BaseGrader, Grade, GradeContext, normalise
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult

# Trailing number, tolerating $ , % and thousands separators.
_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_number(text: str) -> float | None:
    """Take the *last* number in the output.

    Models restate the question before answering, so the first number is often
    an input. The last one is nearly always the answer.
    """
    matches = _NUMBER.findall(text.replace("**", ""))
    if not matches:
        return None
    raw = matches[-1].replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


class NumericGrader(BaseGrader):
    key = "numeric"

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        got = extract_number(result.text)
        want = float(item.expected) if item.expected is not None else None
        if got is None or want is None:
            # Labels are set even on failure so the failure gallery can show
            # what the model actually said next to what was expected.
            return Grade(
                0.0,
                False,
                pred_label=None if got is None else f"{got:g}",
                gold_label=None if want is None else f"{want:g}",
                detail={"got": got, "want": want, "reason": "no number"},
            )
        tolerance = (
            float(item.constraints[0].get("tolerance", 1e-6)) if item.constraints else 1e-6
        )
        ok = abs(got - want) <= tolerance
        return Grade(
            score=1.0 if ok else 0.0,
            passed=ok,
            pred_label=f"{got:g}",
            gold_label=f"{want:g}",
            detail={"got": got, "want": want},
        )


class ExactGrader(BaseGrader):
    key = "exact"

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        got = normalise(result.text)
        want = normalise(str(item.expected or ""))
        ok = got == want
        return Grade(
            score=1.0 if ok else 0.0,
            passed=ok,
            pred_label=got[:80],
            gold_label=want[:80],
            detail={"got": result.text[:200], "want": item.expected},
        )

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        """Adds macro-F1 when the items look like a classification task."""
        base = super().aggregate(grades, items)
        labels = {g.gold_label for g in grades if g.gold_label}
        if len(labels) < 2 or len(labels) > 20:
            return base
        f1s = []
        for label in labels:
            tp = sum(1 for g in grades if g.pred_label == label and g.gold_label == label)
            fp = sum(1 for g in grades if g.pred_label == label and g.gold_label != label)
            fn = sum(1 for g in grades if g.pred_label != label and g.gold_label == label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        base["macro_f1"] = sum(f1s) / len(f1s)
        return base
