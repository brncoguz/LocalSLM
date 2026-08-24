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
