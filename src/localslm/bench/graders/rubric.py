"""Scoring open-ended answers against a written criterion.

Every other grader in this package is a function of the output alone. This one
is a function of the output *and another model's opinion of it*, which makes it
the only grader whose result can be wrong for reasons that have nothing to do
with the model being measured.

So it refuses to guess. A missing or unusable verdict scores nothing and says
why, rather than falling through to zero — a zero here would be read as "the
model answered badly", and the whole point is that we don't know that.
"""

from __future__ import annotations

from collections.abc import Sequence

from localslm.bench.graders.base import BaseGrader, Grade, GradeContext, mean
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult


class RubricGrader(BaseGrader):
    key = "rubric"
    # The runner reads this to decide whether to run a judge before grading.
    requires_judge = True

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        verdict = ctx.extra.get("verdict")

        if verdict is None:
            # No judge was configured for this run. The item was answered, but
            # nothing scored it.
            return Grade(
                score=0.0,
                passed=False,
                judged=False,
                metrics={"unjudged": 1.0},
                detail={
                    "why": "no judge configured — this item is unscored, not failed",
                    "text": result.text[:200],
                },
            )

        if not verdict.usable:
            return Grade(
                score=0.0,
                passed=False,
                judged=False,
                metrics={"unjudged": 1.0, "judge_parse_failure": 1.0},
                detail={
                    "why": "the judge produced nothing usable",
                    "judge_model": verdict.judge_model,
                    "judge_detail": verdict.detail,
                    "judge_raw": verdict.raw[:400],
                    "text": result.text[:200],
                },
            )

        return Grade(
            score=verdict.score,
            passed=bool(verdict.passed),
            judged=True,
            metrics={
                "unjudged": 0.0,
                "judge_parse_failure": 0.0,
                "rating": float(verdict.rating or 0),
            },
            detail={
                "judge_model": verdict.judge_model,
                "backend": verdict.backend,
                "rating": verdict.rating,
                "reason": verdict.rationale,
                "text": result.text[:200],
            },
        )

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        """Averages cover the judged items only.

        Including unjudged items would let a judge that failed half the time
        report a mediocre score for a model that might have been excellent —
        the same mistake as averaging a swapped request in as merely slow.
        `judged_rate` is what says how much of the suite the headline covers.
        """
        judged = [g for g in grades if g.judged]
        return {
            "n": len(grades),
            "n_judged": len(judged),
            "judged_rate": mean([1.0 if g.judged else 0.0 for g in grades]),
            "judge_parse_failure_rate": mean(
                [g.metrics.get("judge_parse_failure", 0.0) for g in grades]
            ),
            "accuracy": mean([g.score for g in judged]),
            "pass_rate": mean([1.0 if g.passed else 0.0 for g in judged]),
            "mean_rating": mean([g.metrics.get("rating", 0.0) for g in judged]),
        }
