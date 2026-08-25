"""Pairing in the thinking endpoint.

This is where the project's headline claim comes from — "reasoning tripled
accuracy on arithmetic" is one cell of this response divided by another. The
comparison is only valid if every cell is scored on the same items.

That is not a hypothetical concern here. Long reasoning runs push a 16 GB
machine into swap, those requests get excluded, and each think level is left
holding a different subset. Averaging over whatever survived would compare
think=high on the questions it happened to finish against think=off on all of
them, and read the difference as a property of reasoning.
"""

from __future__ import annotations

import pytest

from localslm.api.routes.bench import run_thinking
from localslm.db import insert

RUN = "run-1"


def seed(conn, rows):
    """rows: (cell_key, think, item_id, score, excluded, thinking_tokens)."""
    insert(conn, "runs", {"id": RUN, "ts": 1.0, "suite": "math", "status": "done"})
    for idx, (cell, think, item_id, score, excluded, thinking) in enumerate(rows):
        request_id = f"req{idx}"
        insert(
            conn,
            "requests",
            {
                "id": request_id,
                "ts": 1.0 + idx,
                "kind": "bench",
                "run_id": RUN,
                "runtime": "fake",
                "model": "m:4b",
                "cell_key": cell,
                "item_id": item_id,
                "task": "numeric",
                "think": think,
                "thinking_tokens": thinking,
                "answer_tokens": 10,
                "excluded": int(excluded),
                "ttft_ms": 100.0,
                "ttfat_ms": 200.0,
                "total_ms": 300.0,
            },
        )
        insert(
            conn,
            "results",
            {
                "id": f"res{idx}",
                "run_id": RUN,
                "request_id": request_id,
                "cell_key": cell,
                "model": "m:4b",
                "item_id": item_id,
                "task": "numeric",
                "grader": "numeric",
                "score": score,
                "passed": int(bool(score)),
            },
        )


@pytest.fixture
def state(conn, monkeypatch):
    """Point the route's module-global state at a temporary database."""
    from types import SimpleNamespace

    import localslm.api.deps as deps

    monkeypatch.setattr(deps, "_state", SimpleNamespace(conn=conn), raising=False)
    return conn


@pytest.mark.asyncio
async def test_cells_are_compared_only_on_items_both_completed(state):
    """`off` finished three items, `high` swapped on one of them. Both cells
    must be scored on the two they share, not on what each managed."""
    seed(
        state,
        [
            ("off", "off", "i1", 0.0, False, 0),
            ("off", "off", "i2", 0.0, False, 0),
            ("off", "off", "i3", 0.0, False, 0),
            ("high", "high", "i1", 1.0, False, 900),
            ("high", "high", "i2", 1.0, False, 900),
            ("high", "high", "i3", 1.0, True, 900),  # swapped, excluded
        ],
    )

    body = await run_thinking(RUN)
    points = {p["cell_key"]: p for p in body["points"]}

    assert body["paired"] is True
    assert body["n_shared_items"] == 2
    assert body["n_dropped_items"] == 1
    assert points["off"]["n"] == 2
    assert points["high"]["n"] == 2
    # Without pairing, `off` would average over three items and `high` over two.
    assert points["off"]["accuracy"] == 0.0
    assert points["high"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_the_dropped_count_is_reported_not_just_applied(state):
    """Silently narrowing the comparison would be its own kind of dishonesty.
    The reader has to be able to see how much of the run went missing."""
    seed(
        state,
        [
            ("off", "off", "i1", 1.0, False, 0),
            ("off", "off", "i2", 1.0, False, 0),
            ("high", "high", "i1", 1.0, False, 900),
        ],
    )

    body = await run_thinking(RUN)

    assert body["n_shared_items"] == 1
    assert body["n_dropped_items"] == 1
    assert "1" in body["note"]


@pytest.mark.asyncio
async def test_n_attempted_preserves_what_the_cell_actually_ran(state):
    """`n` is what was compared; `n_attempted` is what was measured. Keeping
    both is what lets someone check the pairing rather than trust it."""
    seed(
        state,
        [
            ("off", "off", "i1", 1.0, False, 0),
            ("off", "off", "i2", 1.0, False, 0),
            ("high", "high", "i1", 1.0, False, 900),
        ],
    )

    points = {p["cell_key"]: p for p in (await run_thinking(RUN))["points"]}

    assert (points["off"]["n"], points["off"]["n_attempted"]) == (1, 2)


@pytest.mark.asyncio
async def test_no_shared_items_falls_back_to_unpaired_but_says_so(state):
    """The trap. With an empty intersection the endpoint reverts to exactly the
    invalid averaging it exists to prevent — `paired = [...] or entries`.

    That is a defensible choice (an empty chart helps nobody) but only because
    `paired: False` discloses it. A refactor that dropped the flag would
    republish the invalid comparison with nothing left to signal it, so the
    disclosure is pinned here rather than the fallback.
    """
    seed(
        state,
        [
            ("off", "off", "i1", 0.0, False, 0),
            ("high", "high", "i2", 1.0, False, 900),
        ],
    )

    body = await run_thinking(RUN)

    assert body["paired"] is False
    assert body["n_shared_items"] == 0
    assert len(body["points"]) == 2  # still drawn, from disjoint items


@pytest.mark.asyncio
async def test_excluded_requests_never_reach_the_comparison(state):
    """The join that makes the whole endpoint trustworthy."""
    seed(
        state,
        [
            ("off", "off", "i1", 1.0, False, 0),
            ("off", "off", "i2", 0.0, True, 0),  # excluded: must not count
        ],
    )

    points = (await run_thinking(RUN))["points"]

    assert points[0]["n"] == 1
    assert points[0]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_an_ungraded_score_counts_as_zero(state):
    """`e["score"] or 0` treats a NULL score as a failure rather than skipping
    it, so an ungraded row deflates accuracy instead of shrinking n.

    Pinned deliberately: for the graders in use a NULL score means the item was
    not answered, which is a zero. If a judge is ever added that leaves scores
    pending, this test is where that assumption becomes visible.
    """
    seed(
        state,
        [
            ("off", "off", "i1", 1.0, False, 0),
            ("off", "off", "i2", None, False, 0),
        ],
    )

    points = (await run_thinking(RUN))["points"]

    assert points[0]["n"] == 2
    assert points[0]["accuracy"] == 0.5


@pytest.mark.asyncio
async def test_a_run_with_no_graded_results_says_so_rather_than_dividing(state):
    """A run that aborted before grading anything has no comparison to make.
    It has to say that, not divide by an empty list."""
    insert(state, "runs", {"id": "empty", "ts": 1.0, "suite": "m", "status": "aborted"})

    body = await run_thinking("empty")

    assert body["points"] == []
    assert "No graded results" in body["note"]
