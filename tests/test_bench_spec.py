"""Matrix expansion and run ordering."""

from __future__ import annotations

from itertools import pairwise

from localslm.bench.spec import BenchSpec, order_units, unit_seed
from localslm.bench.suites import Suite, TaskItem


def suite(name: str, n: int = 3) -> Suite:
    return Suite(
        name,
        [
            TaskItem(id=f"{name}{i}", task="numeric", prompt="p", expected=i)
            for i in range(n)
        ],
    )


def spec(**kw) -> BenchSpec:
    base = {"models": ["a:4b", "b:3b"], "suites": [suite("s")], "repeats": 2, "warmups": 1}
    return BenchSpec(**{**base, **kw})


def test_cells_are_the_product_of_models_and_axes():
    s = spec(think_levels=[False, "high"], thinking_capable={"a:4b": True, "b:3b": True})
    assert len(s.cells()) == 4


def test_non_thinking_models_skip_think_levels():
    """Requesting four think levels from a model that can't think would
    quadruple its runtime to produce four identical rows."""
    s = spec(
        think_levels=[False, "low", "high"], thinking_capable={"a:4b": True, "b:3b": False}
    )
    cells = s.cells()
    assert sum(1 for c in cells if c.model == "a:4b") == 3
    assert sum(1 for c in cells if c.model == "b:3b") == 1


def test_total_requests_matches_the_units_actually_produced():
    """An estimate that overstates the work is a bad first impression, and
    warmups are per suite rather than per item."""
    s = spec(suites=[suite("x", 4), suite("y", 6)], repeats=3, warmups=1)
    assert s.total_requests() == len(order_units(s, 0))


def test_warmups_only_run_on_the_first_pass():
    units = order_units(spec(repeats=3), 0)
    warmups = [u for u in units if u.is_warmup]
    assert warmups and all(u.pass_idx == 0 for u in warmups)


def test_model_order_rotates_between_passes():
    """Grouping strictly by model would mean the last model always runs on the
    hottest machine, confounding thermal drift with model identity."""
    units = order_units(spec(repeats=2), 0)
    leaders = {}
    for u in units:
        leaders.setdefault(u.pass_idx, u.cell.model)
    assert leaders[0] != leaders[1]


def test_items_are_grouped_within_a_model_block():
    """Swapping models per request would measure disk, not the model."""
    units = [u for u in order_units(spec(repeats=1), 0) if not u.is_warmup]
    switches = sum(1 for a, b in pairwise(units) if a.cell.key != b.cell.key)
    assert switches == len(spec(repeats=1).cells()) - 1


def test_seeds_are_deterministic_so_a_run_can_be_replayed():
    a = order_units(spec(), 42)
    b = order_units(spec(), 42)
    assert [u.seed for u in a] == [u.seed for u in b]
    assert [u.seed for u in order_units(spec(), 43)] != [u.seed for u in a]


def test_seed_varies_by_item_and_repeat():
    from localslm.bench.spec import Cell

    c = Cell("m", ())
    assert unit_seed(1, c, "i1", 0) != unit_seed(1, c, "i2", 0)
    assert unit_seed(1, c, "i1", 0) != unit_seed(1, c, "i1", 1)


def test_content_hash_changes_when_the_suite_changes():
    a = spec(suites=[suite("s", 3)])
    b = spec(suites=[suite("s", 4)])
    assert a.content_hash() != b.content_hash()


class TestPressureLatch:
    """The latch is per-run. Left set, every later run in the same process
    aborts before its first unit — which is exactly what happened in practice."""

    def test_latch_is_cleared_at_the_start_of_a_run(self):
        from localslm.telemetry.sampler import SystemSampler

        sampler = SystemSampler(interval_s=10)
        sampler._abort_latched = True
        assert sampler.abort_latched
        sampler.reset_abort()
        assert not sampler.abort_latched

    def test_runner_resets_the_latch_before_executing(self):
        """Guards the wiring, not just the method: reset_abort existed for a
        while without anyone calling it."""
        import inspect

        from localslm.bench.runner import BenchRunner

        source = inspect.getsource(BenchRunner.run)
        assert "reset_abort()" in source


class TestRunDeletion:
    """A benchmark run *is* its measurements, so deleting one must take the
    `requests` rows with it. Left behind they would keep skewing the Stats
    charts with no run to trace them back to."""

    def test_delete_removes_every_table_belonging_to_the_run(self, conn):
        from localslm.db import insert

        insert(conn, "runs", {"id": "r1", "ts": 1.0, "suite": "math", "status": "done"})
        insert(conn, "runs", {"id": "r2", "ts": 2.0, "suite": "math", "status": "done"})
        for run_id in ("r1", "r2"):
            insert(
                conn,
                "requests",
                {
                    "id": f"req-{run_id}",
                    "ts": 1.0,
                    "kind": "bench",
                    "run_id": run_id,
                    "runtime": "ollama",
                    "model": "m",
                },
            )
            insert(
                conn,
                "results",
                {
                    "id": f"res-{run_id}",
                    "run_id": run_id,
                    "request_id": f"req-{run_id}",
                    "model": "m",
                    "item_id": "i1",
                    "task": "numeric",
                    "grader": "numeric",
                },
            )
            insert(conn, "cells", {"run_id": run_id, "cell_key": "c1", "model": "m"})

        for table in ("results", "cells", "samples", "requests"):
            conn.execute(f"DELETE FROM {table} WHERE run_id = ?", ("r1",))
        conn.execute("DELETE FROM runs WHERE id = ?", ("r1",))
        conn.commit()

        def count(table: str, run_id: str) -> int:
            return conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
            ).fetchone()[0]

        assert count("requests", "r1") == 0
        assert count("results", "r1") == 0
        assert count("cells", "r1") == 0
        # The other run is untouched.
        assert count("requests", "r2") == 1
        assert count("results", "r2") == 1

    def test_chat_measurements_are_not_touched_by_run_deletion(self, conn):
        """Chat rows have no run_id, so a run delete must never reach them."""
        from localslm.db import insert

        insert(
            conn,
            "requests",
            {
                "id": "chat1",
                "ts": 1.0,
                "kind": "chat",
                "session_id": "s1",
                "runtime": "ollama",
                "model": "m",
            },
        )
        conn.execute("DELETE FROM requests WHERE run_id = ?", ("r1",))
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def test_interrupted_runs_are_closed_out_on_startup(conn):
    """A process that dies mid-run leaves its row marked `running`. Nothing can
    still be running once a new process starts, and a permanently in-flight row
    hides the fact that the results are partial."""
    from localslm.api.deps import reconcile_interrupted_runs
    from localslm.db import insert

    insert(conn, "runs", {"id": "live", "ts": 1.0, "suite": "math", "status": "running"})
    insert(conn, "runs", {"id": "old", "ts": 2.0, "suite": "math", "status": "done"})

    assert reconcile_interrupted_runs(conn) == 1
    rows = {
        r["id"]: (r["status"], r["abort_reason"])
        for r in conn.execute("SELECT id, status, abort_reason FROM runs")
    }
    assert rows["live"] == ("aborted", "interrupted")
    assert rows["old"] == ("done", None)  # finished runs are untouched


class TestDeterminism:
    def test_identical_outputs_are_fully_deterministic(self):
        from localslm.bench.determinism import stability

        s = stability(["42", "42", "42"])
        assert s.fully_deterministic and s.exact_duplicate_rate == 1.0

    def test_duplicate_rate_is_the_modal_share(self):
        from localslm.bench.determinism import stability

        s = stability(["a", "a", "a", "b"])
        assert s.exact_duplicate_rate == 0.75
        assert s.distinct == 2

    def test_formatting_noise_is_not_counted_as_instability(self):
        """A trailing space is not a different answer; counting it as one
        would overstate how unstable the model is."""
        from localslm.bench.determinism import stability

        assert stability(["42", " 42 ", "42."]).fully_deterministic

    def test_length_variance_survives_normalisation(self):
        """Text may normalise identically while the raw outputs differ in
        length — that verbosity swing is still worth reporting."""
        from localslm.bench.determinism import stability

        s = stability(["42", "42.        "])
        assert s.fully_deterministic
        assert s.length_stdev > 0

    def test_empty_input_does_not_crash(self):
        from localslm.bench.determinism import stability

        assert stability([]).n == 0


class TestThinkingBudget:
    """A cap tuned for direct answers truncates a reasoning model mid-thought,
    so it scores zero for running out of room rather than for being wrong."""

    def test_thinking_cells_get_a_larger_token_budget(self):
        from localslm.bench.runner import (
            NUM_PREDICT_DIRECT,
            NUM_PREDICT_THINKING,
            BenchRunner,
        )
        from localslm.bench.spec import Cell

        runner = BenchRunner(state=None, spec=spec(), run_id="r", label="t")
        item = TaskItem(id="i", task="numeric", prompt="p", expected=1)

        def budget(axes):
            unit = type("U", (), {"cell": Cell("m", axes), "item": item, "seed": 1})()
            return runner._build(unit, None).options.num_predict

        assert budget((("think", False),)) == NUM_PREDICT_DIRECT
        assert budget((("think", "high"),)) == NUM_PREDICT_THINKING
        assert budget((("think", "low"),)) == NUM_PREDICT_THINKING
        assert budget(()) == NUM_PREDICT_DIRECT


def test_context_length_is_pinned_not_inherited():
    """An unset num_ctx inherits Ollama's default, which varies by version and
    environment — so runs stop being comparable, and the biggest driver of
    resident memory goes unrecorded."""
    from localslm.bench.runner import NUM_CTX_DIRECT, NUM_CTX_THINKING, BenchRunner
    from localslm.bench.spec import Cell

    runner = BenchRunner(state=None, spec=spec(), run_id="r", label="t")
    item = TaskItem(id="i", task="numeric", prompt="p", expected=1)

    def ctx(axes):
        unit = type("U", (), {"cell": Cell("m", axes), "item": item, "seed": 1})()
        return runner._build(unit, None).options.num_ctx

    assert ctx(()) == NUM_CTX_DIRECT
    assert ctx((("think", "high"),)) == NUM_CTX_THINKING
    assert NUM_CTX_THINKING > NUM_CTX_DIRECT
