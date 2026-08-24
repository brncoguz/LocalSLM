"""Programmatically verifiable instruction following.

Every constraint here is checkable in code, so this measures obedience without
any judge in the loop. Constraints live on the item as data:

    {"type": "max_words", "value": 20}
    {"type": "ends_with", "value": "BANANA"}
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from localslm.bench.graders.base import BaseGrader, Grade, GradeContext, mean, normalise
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


CHECKS = {
    "max_words": lambda t, v: len(_words(t)) <= int(v),
    "min_words": lambda t, v: len(_words(t)) >= int(v),
    "exact_lines": lambda t, v: (
        len([ln for ln in t.strip().splitlines() if ln.strip()]) == int(v)
    ),
    "max_chars": lambda t, v: len(t.strip()) <= int(v),
    "starts_with": lambda t, v: normalise(t).startswith(normalise(str(v))),
    "ends_with": lambda t, v: normalise(t.rstrip(".!")).endswith(normalise(str(v))),
    "contains": lambda t, v: normalise(str(v)) in normalise(t),
    "not_contains": lambda t, v: normalise(str(v)) not in normalise(t),
    "uppercase": lambda t, v: t.strip() == t.strip().upper(),
    "lowercase": lambda t, v: t.strip() == t.strip().lower(),
    "no_word_longer_than": lambda t, v: all(len(w) <= int(v) for w in _words(t)),
    "bullet_count": lambda t, v: len(re.findall(r"^\s*[-*•]\s+", t, re.M)) == int(v),
    "json_only": lambda t, v: t.strip().startswith(("{", "[")),
    "regex": lambda t, v: re.search(str(v), t) is not None,
}


class IFEvalGrader(BaseGrader):
    key = "ifeval"

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        text = result.text
        results, failed = [], []
        for constraint in item.constraints:
            kind = constraint.get("type")
            check = CHECKS.get(kind)
            if check is None:
                continue
            try:
                ok = bool(check(text, constraint.get("value")))
            except (ValueError, TypeError, re.error):
                ok = False
            results.append(ok)
            if not ok:
                failed.append(kind)

        if not results:
            return Grade(0.0, False, detail={"reason": "no checkable constraints"})
        score = sum(results) / len(results)
        return Grade(
            score=score,
            passed=all(results),
            metrics={"constraints_met": score, "all_met": 1.0 if all(results) else 0.0},
            detail={"failed": failed, "checked": len(results)},
        )

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        return {
            "n": len(grades),
            # Partial credit and strict obedience tell different stories: a model
            # can satisfy most constraints on every item and still never follow
            # an instruction completely.
            "accuracy": mean([g.score for g in grades]),
            "all_constraints_met": mean([g.metrics.get("all_met", 0.0) for g in grades]),
        }
