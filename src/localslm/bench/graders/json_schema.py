"""Structured-output grading.

Reports a tier rather than one number, because "did it produce JSON at all",
"did it match the schema" and "were the values right" fail for different
reasons and have different fixes:

    valid_json@1  ->  schema_valid@1  ->  valid@2 (after one retry)  ->  field accuracy

The retry is the pattern this project cares about demonstrating: catch an
invalid output, reprompt once, then fail gracefully rather than looping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from localslm.bench.graders.base import BaseGrader, Grade, GradeContext, mean, normalise
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> tuple[Any | None, str | None]:
    """Return (parsed, error). Tolerates code fences and surrounding prose,
    because a model that wrapped correct JSON in ``` has a formatting problem,
    not a comprehension one, and the tiers should tell them apart."""
    candidate = text.strip()
    if m := _FENCE.search(candidate):
        candidate = m.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _types_match(value: Any, spec: str) -> bool:
    return {
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(spec, True)


def validate(obj: Any, schema: dict[str, Any]) -> list[str]:
    """Minimal JSON Schema check covering the subset Ollama's `format` accepts:
    type, required, properties. Avoids a dependency for three rules."""
    errors: list[str] = []
    if (want := schema.get("type")) and not _types_match(obj, want):
        return [f"root: expected {want}, got {type(obj).__name__}"]
    if not isinstance(obj, dict):
        return errors
    for key in schema.get("required", []):
        if key not in obj:
            errors.append(f"missing required field {key!r}")
    for key, spec in (schema.get("properties") or {}).items():
        if key in obj and (want := spec.get("type")) and not _types_match(obj[key], want):
            errors.append(f"{key}: expected {want}, got {type(obj[key]).__name__}")
    return errors


def _field_accuracy(got: dict[str, Any], want: dict[str, Any]) -> tuple[float, dict]:
    if not want:
        return 1.0, {}
    hits, diffs = 0, {}
    for key, expected in want.items():
        actual = got.get(key)
        same = (
            normalise(str(actual)) == normalise(str(expected))
            if not isinstance(expected, int | float | bool) or isinstance(expected, bool)
            else actual == expected
        )
        if same:
            hits += 1
        else:
            diffs[key] = {"got": actual, "want": expected}
    return hits / len(want), diffs


class JsonSchemaGrader(BaseGrader):
    key = "json_schema"

    def grade(self, item: TaskItem, result: GenResult, ctx: GradeContext) -> Grade:
        parsed, parse_error = extract_json(result.text)
        metrics = {
            "valid_json": 1.0 if parsed is not None else 0.0,
            "retry_used": 1.0 if ctx.retry_used else 0.0,
        }
        if parsed is None:
            return Grade(0.0, False, metrics, detail={"parse_error": parse_error})

        errors = validate(parsed, item.output_schema or {})
        metrics["schema_valid"] = 0.0 if errors else 1.0
        if errors:
            return Grade(0.0, False, metrics, detail={"schema_errors": errors})

        want = item.expected if isinstance(item.expected, dict) else {}
        accuracy, diffs = _field_accuracy(parsed, want)
        metrics["field_accuracy"] = accuracy
        metrics["exact_object"] = 1.0 if accuracy == 1.0 else 0.0
        return Grade(
            score=accuracy,
            passed=accuracy == 1.0,
            metrics=metrics,
            detail={"got": parsed, "field_diffs": diffs} if diffs else {"got": parsed},
        )

    def aggregate(
        self, grades: Sequence[Grade], items: Sequence[TaskItem]
    ) -> dict[str, float]:
        def m(key: str) -> float:
            return mean([g.metrics.get(key, 0.0) for g in grades])

        first_try = [g for g in grades if not g.metrics.get("retry_used")]
        return {
            "n": len(grades),
            "accuracy": mean([g.score for g in grades]),
            # valid@1 counts only outputs that never needed a retry; valid@2 is
            # the rate after the single reprompt. The gap is what the retry buys.
            "valid_json_at_1": mean([g.metrics.get("valid_json", 0.0) for g in first_try]),
            "valid_json_at_2": m("valid_json"),
            "schema_valid": m("schema_valid"),
            "field_accuracy": m("field_accuracy"),
            "exact_object": m("exact_object"),
            "retry_rate": m("retry_used"),
        }
