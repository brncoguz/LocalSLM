"""Grader registry.

A grader turns one model output into a score. Items name their grader, so the
runner never grows task-specific branches.

`aggregate` on the protocol is load-bearing: macro-F1, hallucination rate on the
unanswerable subset, and valid@1-vs-valid@2 cannot be produced by averaging
per-item scores, and putting them here is what keeps that logic out of the runner.
"""

from __future__ import annotations

from localslm.bench.graders.base import Grade, GradeContext, Grader
from localslm.bench.graders.contains import ContainsGrader, RefusalGrader
from localslm.bench.graders.grounded import GroundedGrader
from localslm.bench.graders.ifeval import IFEvalGrader
from localslm.bench.graders.json_schema import JsonSchemaGrader
from localslm.bench.graders.numeric import ExactGrader, NumericGrader
from localslm.bench.graders.rubric import RubricGrader

REGISTRY: dict[str, Grader] = {
    g.key: g  # type: ignore[misc]
    for g in (
        JsonSchemaGrader(),
        NumericGrader(),
        ExactGrader(),
        IFEvalGrader(),
        ContainsGrader(),
        RefusalGrader(),
        GroundedGrader(),
        RubricGrader(),
    )
}


def get(key: str) -> Grader:
    if key not in REGISTRY:
        raise KeyError(f"unknown grader {key!r} (have: {', '.join(sorted(REGISTRY))})")
    return REGISTRY[key]


__all__ = ["REGISTRY", "Grade", "GradeContext", "Grader", "get"]
