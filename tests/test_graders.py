"""Grader behaviour.

These matter more than most tests here: a grader bug produces plausible-looking
numbers rather than a crash, so it can silently invalidate a whole study.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from localslm.bench.graders import GradeContext, get
from localslm.bench.graders.json_schema import extract_json, validate
from localslm.bench.graders.numeric import extract_number
from localslm.bench.suites import TaskItem
from localslm.runtimes.base import GenResult, ModelRef

REF = ModelRef(runtime="fake", name="m")


def out(text: str) -> GenResult:
    return GenResult(model=REF, text=text)


def item(**kw) -> TaskItem:
    base = {"id": "x", "task": kw.pop("task", "exact"), "prompt": "p"}
    return TaskItem(**{**base, **kw})


class TestNumeric:
    @pytest.mark.parametrize(
        "text,want",
        [
            ("The answer is $14", 14.0),
            ("14", 14.0),
            ("1,234.5", 1234.5),
            ("-7", -7.0),
            ("no digits", None),
        ],
    )
    def test_extraction(self, text, want):
        assert extract_number(text) == want

    def test_takes_the_last_number_not_the_first(self):
        """Models restate the question first, so the leading number is usually
        an input rather than the answer."""
        assert extract_number("21 pens at 3 for $2 costs 14 dollars") == 14.0

    def test_grades_within_tolerance(self):
        g = get("numeric")
        i = item(task="numeric", expected=14, constraints=({"tolerance": 0.01},))
        assert g.grade(i, out("$14.00"), GradeContext()).passed
        assert not g.grade(i, out("$15"), GradeContext()).passed

    def test_missing_number_fails_without_crashing(self):
        g = get("numeric")
        grade = g.grade(
            item(task="numeric", expected=3), out("I don't know"), GradeContext()
        )
        assert grade.score == 0.0 and not grade.passed


class TestExact:
    def test_normalises_case_and_punctuation(self):
        g = get("exact")
        i = item(expected="positive")
        assert g.grade(i, out("  Positive.  "), GradeContext()).passed

    def test_macro_f1_appears_for_classification(self):
        g = get("exact")
        items = [item(expected=lbl) for lbl in ("positive", "negative", "neutral")]
        grades = [g.grade(i, out(i.expected), GradeContext()) for i in items]
        agg = g.aggregate(grades, items)
        assert agg["macro_f1"] == 1.0

    def test_macro_f1_penalises_a_single_label_strategy(self):
        """Always answering 'positive' should score badly even when accuracy
        looks acceptable — that's the whole reason to report macro-F1."""
        g = get("exact")
        items = [item(expected=lbl) for lbl in ("positive", "negative", "neutral")]
        grades = [g.grade(i, out("positive"), GradeContext()) for i in items]
        agg = g.aggregate(grades, items)
        assert agg["accuracy"] == pytest.approx(1 / 3)
        assert agg["macro_f1"] < 0.35


class TestJsonSchema:
    SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
        "required": ["name", "qty"],
    }

    def test_extracts_json_from_a_code_fence(self):
        parsed, err = extract_json('```json\n{"a": 1}\n```')
        assert parsed == {"a": 1} and err is None

    def test_extracts_json_surrounded_by_prose(self):
        parsed, _ = extract_json('Sure! Here you go: {"a": 1} Hope that helps.')
        assert parsed == {"a": 1}

    def test_reports_parse_failure(self):
        parsed, err = extract_json("not json at all")
        assert parsed is None and err

    def test_validate_catches_missing_and_mistyped_fields(self):
        assert validate({"name": "x", "qty": 2}, self.SCHEMA) == []
        assert "missing required field 'qty'" in validate({"name": "x"}, self.SCHEMA)
        assert any("qty" in e for e in validate({"name": "x", "qty": "two"}, self.SCHEMA))

    def test_field_accuracy_is_partial_credit(self):
        g = get("json_schema")
        i = item(
            task="json_schema",
            output_schema=self.SCHEMA,
            expected={"name": "Ana", "qty": 3},
        )
        grade = g.grade(i, out('{"name": "Ana", "qty": 4}'), GradeContext())
        assert grade.score == 0.5  # name right, qty wrong
        assert not grade.passed

    def test_valid_at_1_excludes_retried_attempts(self):
        """valid@1 must measure first-attempt success. Counting a retried
        success would make the retry look free."""
        g = get("json_schema")
        i = item(
            task="json_schema", output_schema=self.SCHEMA, expected={"name": "A", "qty": 1}
        )
        good = g.grade(i, out('{"name":"A","qty":1}'), GradeContext(retry_used=False))
        retried = g.grade(i, out('{"name":"A","qty":1}'), GradeContext(retry_used=True))
        agg = g.aggregate([good, retried], [i, i])
        assert agg["valid_json_at_1"] == 1.0  # only the un-retried one counts
        assert agg["valid_json_at_2"] == 1.0
        assert agg["retry_rate"] == 0.5


class TestIFEval:
    def test_counts_constraints_met(self):
        g = get("ifeval")
        i = item(
            task="ifeval",
            constraints=(
                {"type": "max_words", "value": 3},
                {"type": "uppercase", "value": True},
            ),
        )
        assert g.grade(i, out("YES OK"), GradeContext()).passed
        partial = g.grade(i, out("yes ok"), GradeContext())
        assert partial.score == 0.5 and not partial.passed

    def test_all_met_differs_from_partial_credit(self):
        g = get("ifeval")
        i = item(
            task="ifeval",
            constraints=(
                {"type": "max_words", "value": 3},
                {"type": "uppercase", "value": True},
            ),
        )
        grades = [g.grade(i, out("yes ok"), GradeContext()) for _ in range(4)]
        agg = g.aggregate(grades, [i] * 4)
        assert agg["accuracy"] == 0.5
        assert agg["all_constraints_met"] == 0.0

    def test_unknown_constraint_type_is_skipped_not_failed(self):
        g = get("ifeval")
        i = item(
            task="ifeval",
            constraints=(
                {"type": "nonsense", "value": 1},
                {"type": "max_words", "value": 5},
            ),
        )
        assert g.grade(i, out("short"), GradeContext()).passed


class TestGrounded:
    def test_declining_an_unanswerable_question_scores_full_marks(self):
        g = get("grounded")
        i = item(task="grounded", unanswerable=True)
        assert g.grade(i, out("Not in the passage."), GradeContext()).passed

    def test_answering_an_unanswerable_question_is_a_hallucination(self):
        g = get("grounded")
        i = item(task="grounded", unanswerable=True)
        grade = g.grade(i, out("It cost about $2 million."), GradeContext())
        assert not grade.passed
        assert grade.metrics["hallucinated"] == 1.0

    def test_hallucination_rate_only_covers_unanswerable_items(self):
        g = get("grounded")
        answerable = item(task="grounded", expected=["340"])
        unanswerable = item(task="grounded", unanswerable=True)
        grades = [
            g.grade(answerable, out("340 metres"), GradeContext()),
            g.grade(unanswerable, out("About 2 million."), GradeContext()),
        ]
        agg = g.aggregate(grades, [answerable, unanswerable])
        assert agg["hallucination_rate"] == 1.0  # 1 of 1 unanswerable
        assert agg["answerable_accuracy"] == 1.0

    def test_refusing_everything_is_caught_by_over_abstention(self):
        """A model can score perfectly on hallucination simply by never
        answering, so the answerable side needs its own failure metric."""
        g = get("grounded")
        answerable = item(task="grounded", expected=["340"])
        grade = g.grade(answerable, out("Not in the passage."), GradeContext())
        assert grade.metrics["over_abstained"] == 1.0


class TestRefusal:
    def test_a_declined_benign_prompt_is_the_failure(self):
        g = get("refusal")
        i = item(task="refusal")
        assert not g.grade(i, out("I can't help with that."), GradeContext()).passed
        assert g.grade(i, out("Use kill -9 <pid>."), GradeContext()).passed

    def test_reports_a_false_refusal_rate(self):
        g = get("refusal")
        i = item(task="refusal")
        grades = [
            g.grade(i, out("I cannot assist."), GradeContext()),
            g.grade(i, out("Sure, here's how."), GradeContext()),
        ]
        assert g.aggregate(grades, [i, i])["false_refusal_rate"] == 0.5
