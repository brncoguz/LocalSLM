"""The timing maths. This is where the project's credibility lives."""

from localslm.runtimes.base import GenResult, ModelRef, ServerTiming

REF = ModelRef(runtime="fake", name="m")


def test_server_timing_converts_ns_to_ms_and_derives_rates():
    st = ServerTiming.from_done(
        {
            "total_duration": 2_000_000_000,
            "load_duration": 300_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 200,
            "eval_duration": 1_000_000_000,
        }
    )
    assert st.total_ms == 2000
    assert st.load_ms == 300
    assert st.prefill_tps == 200.0  # 100 tokens / 0.5 s
    assert st.decode_tps == 200.0  # 200 tokens / 1.0 s


def test_rates_are_none_rather_than_dividing_by_zero():
    st = ServerTiming.from_done({"eval_count": 5})
    assert st.decode_tps is None
    assert st.prefill_tps is None


def test_itl_percentiles_and_jitter():
    r = GenResult(model=REF, inter_token_ms=[10, 10, 10, 10, 100])
    assert r.itl_p50_ms == 10
    assert r.itl_p95_ms == 100
    assert r.itl_stddev_ms > 30  # the outlier dominates, which is the point


def test_itl_undefined_for_a_single_token():
    r = GenResult(model=REF, inter_token_ms=[])
    assert r.itl_p50_ms is None
    assert r.itl_stddev_ms is None


def test_think_ratio_reports_share_of_thinking_chunks():
    r = GenResult(
        model=REF,
        thinking_chunks=90,
        answer_chunks=10,
        server=ServerTiming(output_tokens=100, eval_ms=1000),
    )
    assert r.think_ratio == 0.9


class TestBootstrap:
    """Confidence intervals and paired comparison."""

    def test_interval_brackets_the_mean(self):
        from localslm.bench.stats import bootstrap_ci

        ci = bootstrap_ci([1.0] * 8 + [0.0] * 2)
        assert ci.lo <= ci.mean <= ci.hi
        assert ci.n == 10

    def test_single_observation_reports_no_spread(self):
        """One sample carries no information about variance; a zero-width
        interval plus n=1 is honest, a fabricated range is not."""
        from localslm.bench.stats import bootstrap_ci

        ci = bootstrap_ci([0.7])
        assert (ci.lo, ci.mean, ci.hi, ci.n) == (0.7, 0.7, 0.7, 1)

    def test_identical_scores_give_a_zero_width_interval(self):
        from localslm.bench.stats import bootstrap_ci

        ci = bootstrap_ci([1.0] * 20)
        assert ci.lo == ci.hi == 1.0

    def test_paired_comparison_uses_only_shared_items(self):
        """Comparing models on different question sets is not a comparison."""
        from localslm.bench.stats import paired_bootstrap

        c = paired_bootstrap({"a": 1.0, "b": 1.0, "only_a": 1.0}, {"a": 0.0, "b": 0.0})
        assert c.n == 2

    def test_a_clear_win_is_significant(self):
        from localslm.bench.stats import paired_bootstrap

        a = {f"i{i}": 1.0 for i in range(30)}
        b = {f"i{i}": 0.0 for i in range(30)}
        c = paired_bootstrap(a, b, "A", "B")
        assert c.significant and c.delta == 1.0
        assert "A beats B" in c.verdict

    def test_a_small_gap_at_low_n_is_not_significant(self):
        """The point of the interval: a visible difference that the data
        cannot actually support must not be reported as a win."""
        from localslm.bench.stats import paired_bootstrap

        a = {f"i{i}": float(i % 2 == 0) for i in range(10)}
        b = {f"i{i}": float(i % 3 == 0) for i in range(10)}
        c = paired_bootstrap(a, b, "A", "B")
        assert not c.significant
        assert "not distinguishable" in c.verdict

    def test_ranking_flags_overlapping_neighbours(self):
        from localslm.bench.stats import Interval, rank_with_ties

        ranked = rank_with_ties(
            {
                "clear_winner": Interval(0.90, 0.85, 0.95, 20),
                "middle": Interval(0.60, 0.50, 0.70, 20),
                "close_to_middle": Interval(0.58, 0.48, 0.68, 20),
            }
        )
        names = [r[0] for r in ranked]
        assert names[0] == "clear_winner"
        assert ranked[0][2] is False
        assert ranked[2][2] is True  # overlaps the row above

    def test_verdict_reports_the_interval_in_the_winner_s_direction(self):
        """A negative range printed beside "beats by" reads as a contradiction."""
        from localslm.bench.stats import paired_bootstrap

        weak = {f"i{i}": 0.0 for i in range(20)}
        strong = {f"i{i}": 1.0 for i in range(20)}
        c = paired_bootstrap(weak, strong, "weak", "strong")
        assert c.delta < 0  # a is worse
        assert "strong beats weak" in c.verdict
        assert "[-" not in c.verdict.split("CI ")[1]
