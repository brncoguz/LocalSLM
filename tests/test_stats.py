"""What the dashboard's summary numbers actually mean.

These are pinning tests, not bug hunts. `_spread` feeds every chart and every
figure in the README, so its definitions need to be stated somewhere a reader
can check them — otherwise a refactor can change what "p95" means across the
whole project without a single test going red.
"""

from __future__ import annotations

import pytest

from localslm.api.routes.stats import _spread, summarise


class TestSpread:
    def test_reports_the_five_figures_and_the_count(self):
        s = _spread([10.0, 20.0, 30.0, 40.0])

        assert s["n"] == 4
        assert s["mean"] == 25.0
        assert s["p50"] == 25.0
        assert s["min"] == 10.0
        assert s["max"] == 40.0

    @pytest.mark.parametrize(
        "n, expected_p95",
        [
            (1, 1),
            (2, 2),
            (5, 5),
            (10, 10),
            (11, 11),  # last n at which p95 is still the maximum
            (12, 11),
            (20, 19),
            (100, 95),
        ],
    )
    def test_p95_is_the_maximum_below_twelve_samples(self, n, expected_p95):
        """`ordered[min(n-1, round(0.95 * (n-1)))]` — nearest-rank, no
        interpolation. At the sample sizes this project actually reaches, that
        makes the published "p95" the slowest single request observed. Worth
        knowing before quoting one as a tail latency.
        """
        assert _spread([float(v) for v in range(1, n + 1)])["p95"] == expected_p95

    def test_missing_values_are_dropped_not_counted_as_zero(self):
        """A NULL column means "not measured". Counting it as 0 would drag
        every mean toward zero in proportion to how much data was missing."""
        s = _spread([10.0, None, 30.0, None])

        assert s["n"] == 2
        assert s["mean"] == 20.0

    def test_nothing_to_describe_is_none_rather_than_a_zeroed_shape(self):
        """The UI keys off None to render an em dash. A dict of zeros would
        draw a real-looking chart of a model that was never measured."""
        assert _spread([]) is None
        assert _spread([None, None]) is None


class TestSummarise:
    def rows(self, *specs):
        """(decode_tps, cold, on_ac) triples, padded into the row shape."""
        return [
            {
                "decode_tps": tps,
                "prefill_tps": None,
                "ttfat_ms": 200.0,
                "ttft_ms": 100.0,
                "total_ms": 300.0,
                "peak_rss_bytes": 3_000_000_000,
                "model_vram_bytes": 1_500_000_000,
                "on_ac": on_ac,
                "cold": cold,
                "srv_load_ms": 2000.0 if cold else 150.0,
                "thinking_tokens": 0,
                "answer_tokens": 10,
                "ts": 1.0,
            }
            for tps, cold, on_ac in specs
        ]

    def test_cold_and_warm_are_summarised_separately(self):
        """Averaging a cold request's load time into the warm figure is the
        single easiest way to make a model look slower than it is."""
        s = summarise(self.rows((100.0, 1, 1), (200.0, 0, 1), (210.0, 0, 1)))

        assert s["cold"]["n"] == 1
        assert s["warm"]["n"] == 2
        assert s["cold"]["load_ms"]["mean"] == 2000.0
        assert s["warm"]["load_ms"]["mean"] == 150.0

    def test_the_power_split_travels_with_every_rollup(self):
        """The headline figures above it average both sources, so the split has
        to be present for the badge to be drawable at all."""
        s = summarise(self.rows((83.7, 0, 1), (74.8, 0, 0)))

        assert s["power"]["mixed"] is True

    def test_think_ratio_is_none_when_nothing_was_generated(self):
        rows = self.rows((100.0, 0, 1))
        rows[0]["answer_tokens"] = 0

        assert summarise(rows)["thinking"]["ratio"] is None

    def test_an_empty_model_summarises_without_dividing_by_zero(self):
        s = summarise([])

        assert s["n"] == 0
        assert s["decode_tps"] is None
        assert s["first_ts"] is None
