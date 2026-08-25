"""The frozen leaderboard must agree with the live inspector.

`_finalise` freezes a per-cell aggregate at the end of a run so that "the
leaderboard and the report quote identical numbers instead of each
re-averaging live" — its own docstring. Every analysis endpoint, meanwhile,
joins `requests.excluded = 0`, because a request measured while the machine was
swapping is not comparable to one that wasn't.

Those two rules only agree if the freeze also drops excluded requests. If it
doesn't, the divergence is invisible: both numbers look plausible, and it is
worst on the long reasoning runs that swap most, which are exactly the runs the
project's headline claims come from.
"""

from __future__ import annotations

from types import SimpleNamespace

from localslm.bench.graders.base import Grade
from localslm.bench.runner import BenchRunner
from localslm.bench.spec import BenchSpec, Cell, Unit
from localslm.bench.suites import Suite, TaskItem

RUN_ID = "run-under-test"


def item(n: int) -> TaskItem:
    return TaskItem(id=f"i{n}", task="numeric", prompt="p", expected=n)


def build(conn, scores_and_exclusions):
    """Record one graded unit per (score, excluded) pair, then freeze.

    Each unit gets a real `requests` row so the runner can read back whether
    the measurement survived, which is the whole point of the exercise.
    """
    from localslm.db import insert

    insert(conn, "runs", {"id": RUN_ID, "ts": 1.0, "suite": "math", "status": "running"})

    items = [item(n) for n in range(len(scores_and_exclusions))]
    spec = BenchSpec(models=["m:4b"], suites=[Suite("math", items)], repeats=1)
    cell = Cell(model="m:4b")
    runner = BenchRunner(SimpleNamespace(conn=conn), spec, RUN_ID)

    for idx, (score, excluded) in enumerate(scores_and_exclusions):
        request_id = f"req-{idx}"
        insert(
            conn,
            "requests",
            {
                "id": request_id,
                "ts": 1.0 + idx,
                "kind": "bench",
                "run_id": RUN_ID,
                "runtime": "fake",
                "model": "m:4b",
                "cell_key": cell.key,
                "item_id": items[idx].id,
                "task": "numeric",
                "excluded": int(excluded),
                "exclude_reason": "swapped" if excluded else None,
            },
        )
        unit = Unit(cell, items[idx], "math", 0, 0, False, 0)
        grade = Grade(score=score, passed=score >= 1.0)
        runner._record(unit, grade, request_id, None)

    runner._finalise(started=0.0, done=len(scores_and_exclusions), planned=0, aborted=None)
    return cell.key


def live(conn, cell_key: str) -> tuple[int, float]:
    """What the analysis endpoints see: graded results whose measurement was
    not excluded. Mirrors the join in `/api/runs/{id}/thinking`."""
    row = conn.execute(
        """SELECT COUNT(*) n, AVG(res.score) score
           FROM results res JOIN requests req ON req.id = res.request_id
           WHERE res.run_id = ? AND res.cell_key = ? AND req.excluded = 0""",
        (RUN_ID, cell_key),
    ).fetchone()
    return row["n"], row["score"]


def frozen(conn, cell_key: str):
    return conn.execute(
        "SELECT n, n_excluded, score_mean FROM cells WHERE run_id = ? AND cell_key = ?",
        (RUN_ID, cell_key),
    ).fetchone()


def test_frozen_aggregate_matches_the_live_query(conn):
    """Three clean items and two that swapped. The frozen row must describe the
    three, because that is what every chart and endpoint will describe."""
    cell_key = build(
        conn, [(1.0, False), (1.0, False), (1.0, False), (0.0, True), (0.0, True)]
    )

    live_n, live_score = live(conn, cell_key)
    row = frozen(conn, cell_key)

    assert (row["n"], row["score_mean"]) == (live_n, live_score)


def test_excluded_measurements_are_counted_not_silently_dropped(conn):
    """`n_excluded` is why the cell has fewer items than the run planned. Left
    at zero it reads as a complete cell, and the missing rows look like they
    were never attempted."""
    cell_key = build(conn, [(1.0, False), (0.0, True), (0.0, True)])

    row = frozen(conn, cell_key)

    assert row["n"] == 1
    assert row["n_excluded"] == 2


def test_results_rows_carry_the_exclusion_from_their_measurement(conn):
    """`results.excluded` exists so a grader-side consumer can apply the same
    rule without joining back to `requests` and possibly forgetting to."""
    build(conn, [(1.0, False), (0.0, True)])

    rows = dict(
        conn.execute(
            "SELECT res.request_id, res.excluded FROM results res WHERE res.run_id = ?",
            (RUN_ID,),
        ).fetchall()
    )

    assert rows == {"req-0": 0, "req-1": 1}


def test_a_clean_run_freezes_exactly_as_before(conn):
    """The fix must be a no-op when nothing was excluded, which is every run on
    a machine with headroom. Without this, the change is untestable as a
    regression."""
    cell_key = build(conn, [(1.0, False), (0.0, False), (1.0, False), (0.5, False)])

    row = frozen(conn, cell_key)
    live_n, live_score = live(conn, cell_key)

    assert row["n"] == 4
    assert row["n_excluded"] == 0
    assert row["score_mean"] == live_score
    assert live_n == 4


def test_a_fully_excluded_cell_does_not_divide_by_zero(conn):
    """A cell whose every measurement swapped has nothing to report. It must
    say so rather than crash the run that produced it or publish a 0.0 that
    reads as "this model scored zero"."""
    cell_key = build(conn, [(1.0, True), (1.0, True)])

    row = frozen(conn, cell_key)

    assert row is None or (row["n"] == 0 and row["score_mean"] is None)
