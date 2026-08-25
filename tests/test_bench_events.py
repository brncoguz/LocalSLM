"""What a running benchmark tells the UI about itself.

The live view renders whatever the unit event carries, so a field quietly
dropped here does not raise anything — the panel simply shows a blank where the
question used to be, and a run becomes a progress bar again. That is the whole
reason this file exists: the failure is invisible at every layer except the one
nobody is looking at.

Also the only end-to-end exercise of `BenchRunner.run` in the suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import FakeRuntime, FakeSampler

from localslm.bench.runner import _brief
from localslm.bench.spec import BenchSpec
from localslm.bench.suites import Suite, TaskItem
from localslm.runtimes.base import Capabilities, ModelInfo, ModelRef
from localslm.telemetry.sampler import SampleWindow

SERVER = {
    "prompt_eval_count": 12,
    "prompt_eval_duration": 100_000_000,
    "eval_count": 4,
    "eval_duration": 200_000_000,
    "load_duration": 0,
}


class RunnerRuntime(FakeRuntime):
    """A fake with the two extra methods the runner reaches for."""

    async def show(self, name: str) -> ModelInfo:
        return ModelInfo(
            ref=ModelRef(runtime="fake", name=name),
            capabilities=Capabilities(completion=True, thinking=False),
        )


def clean_window() -> SampleWindow:
    return SampleWindow(
        n=8,
        duration_s=1.0,
        peak_rss_bytes=2_000_000_000,
        mean_rss_bytes=2e9,
        swap_delta_bytes=0,
        pressure_max="normal",
    )


async def run_events(conn, answer: str = "4"):
    from localslm.bench.runner import BenchRunner

    sampler = FakeSampler(clean_window())
    sampler.abort_latched = False
    sampler.reset_abort = lambda: None
    state = SimpleNamespace(
        conn=conn,
        runtime=RunnerRuntime([(0.1, "", answer)], SERVER),
        sampler=sampler,
        env_hash="test",
        cfg=SimpleNamespace(fingerprint=lambda: "cfg"),
    )
    spec = BenchSpec(
        models=["m:4b"],
        suites=[
            Suite(
                "math",
                [
                    TaskItem(
                        id="m1",
                        task="numeric",
                        prompt="What is 2+2?",
                        expected=4,
                    )
                ],
            )
        ],
        repeats=1,
        warmups=0,
        thinking_capable={"m:4b": False},
    )
    runner = BenchRunner(state, spec, "run-live")
    return [ev async for ev in runner.run()]


@pytest.mark.asyncio
async def test_a_unit_event_carries_the_work_not_just_a_counter(conn):
    """Prompt, answer and expected are what make a run watchable. Without them
    the live view is a progress bar with extra steps."""
    events = await run_events(conn, answer="4")
    unit = next(e.data for e in events if e.kind == "unit")

    assert unit["prompt"] == "What is 2+2?"
    assert unit["answer"] == "4"
    assert unit["expected"] == "4"
    assert unit["done"] == 1
    assert unit["total"] == 1
    assert unit["passed"] is True


@pytest.mark.asyncio
async def test_a_unit_event_says_whether_its_measurement_survived(conn):
    """The live tallies are built from these events, so a row excluded for
    swapping has to be visible as excluded — counting it as a failure would
    blame the model for the machine."""
    events = await run_events(conn)
    unit = next(e.data for e in events if e.kind == "unit")

    assert unit["excluded"] is False
    assert "elapsed_s" in unit


@pytest.mark.asyncio
async def test_a_wrong_answer_still_reports_what_was_expected(conn):
    """The expected value is only shown on failures, which is exactly when it
    has to be there."""
    events = await run_events(conn, answer="5")
    unit = next(e.data for e in events if e.kind == "unit")

    assert unit["passed"] is False
    assert unit["expected"] == "4"
    assert unit["answer"] == "5"


@pytest.mark.asyncio
async def test_the_run_ends_with_a_done_event_and_a_frozen_cell(conn):
    events = await run_events(conn)
    kinds = [e.kind for e in events]

    assert kinds[0] == "plan"
    assert "cell" in kinds
    assert kinds[-1] == "done"


class TestBriefExpected:
    """`expected` reaches the UI as a string whatever shape the suite used."""

    def test_a_scalar(self):
        assert _brief(4) == "4"

    def test_a_list_of_accepted_answers(self):
        assert _brief(["Paris", "paris"]) == "Paris, paris"

    def test_nothing_expected_is_empty_not_the_word_none(self):
        """`grounded_qa` items marked unanswerable have no expected value, and
        rendering the string "None" beside a correct refusal would read as a
        missing answer key."""
        assert _brief(None) == ""
