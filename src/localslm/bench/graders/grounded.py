"""Passage QA with deliberately unanswerable items.

The headline number is not accuracy but the hallucination rate: how often the
model invents an answer for a question its context cannot support. For anything
pointed at your own documents, that figure decides whether the model is usable
at all, and averaging it into overall accuracy hides it.
"""

from __future__ import annotations

from collections.abc import Sequence

from localslm.bench.graders.base import BaseGrader, Grade, GradeContext, mean, normalise
from localslm.bench.graders.contains import looks_like_abstention
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult


class GroundedGrader(BaseGrader):
    key = "grounded"

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        abstained = looks_like_abstention(result.text)

        if item.unanswerable:
            # Correct behaviour is to decline. Answering is a hallucination.
            return Grade(
                score=1.0 if abstained else 0.0,
                passed=abstained,
                metrics={
                    "unanswerable": 1.0,
                    "hallucinated": 0.0 if abstained else 1.0,
                    "abstained": 1.0 if abstained else 0.0,
                },
                detail={"text": result.text[:200]},
            )

        expected = item.expected if isinstance(item.expected, list) else [item.expected]
        wanted = [normalise(str(e)) for e in expected if e is not None]
        got = normalise(result.text)
        correct = any(w in got for w in wanted) if wanted else False
        return Grade(
            score=1.0 if correct else 0.0,
            passed=correct,
            metrics={
                "unanswerable": 0.0,
                "hallucinated": 0.0,
                # Declining a question the passage *does* answer is its own
                # failure mode, and a model can score well on hallucination
                # simply by refusing everything.
                "over_abstained": 1.0 if (abstained and not correct) else 0.0,
            },
            detail={"want": wanted, "text": result.text[:200]},
        )

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        answerable = [g for g in grades if not g.metrics.get("unanswerable")]
        unanswerable = [g for g in grades if g.metrics.get("unanswerable")]
        return {
            "n": len(grades),
            "accuracy": mean([g.score for g in grades]),
            "answerable_accuracy": mean([g.score for g in answerable]),
            "hallucination_rate": mean(
                [g.metrics.get("hallucinated", 0.0) for g in unanswerable]
            ),
            "over_abstention_rate": mean(
                [g.metrics.get("over_abstained", 0.0) for g in answerable]
            ),
            "n_unanswerable": len(unanswerable),
        }
