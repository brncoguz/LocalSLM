"""The battery/AC confound.

Apple Silicon throttles hard on battery: `llama3.2:3b` measured 74.8 tok/s
unplugged against 83.7 plugged in, ~100 requests either way. That 12% is wider
than most of the model-to-model gaps this project reports, so a rollup that
averages the two sources is not merely imprecise — it can invent a difference
that is really a charger. These tests pin the reporting rule, and in particular
pin that the rule is *report*, never *filter*.
"""

from __future__ import annotations

from localslm.api.routes.stats import power_split


def rows(*specs):
    """(on_ac, decode_tps) pairs as the aggregation layer sees them."""
    return [{"on_ac": on_ac, "decode_tps": tps, "ttfat_ms": 300.0} for on_ac, tps in specs]


def test_mixed_sources_are_flagged_and_reported_separately():
    split = power_split(rows((1, 83.7), (1, 83.7), (0, 74.8), (0, 74.8)))

    assert split["mixed"] is True
    assert split["ac"]["n"] == 2
    assert split["battery"]["n"] == 2
    assert split["ac"]["decode_tps"]["mean"] > split["battery"]["decode_tps"]["mean"]


def test_a_single_source_is_not_flagged():
    """Most runs never change power source, and those figures are clean.
    Badging them anyway would train people to ignore the badge."""
    assert power_split(rows((1, 83.7), (1, 84.1)))["mixed"] is False
    assert power_split(rows((0, 74.8), (0, 75.2)))["mixed"] is False


def test_battery_only_models_are_kept_not_dropped():
    """The tempting fix is to report AC only. Of the six models measured so
    far exactly one has rows on both sources, so that would delete most of the
    results table — which is why this is a split and not a filter."""
    split = power_split(rows((0, 74.8), (0, 75.2), (0, 74.1)))

    assert split["battery"]["n"] == 3
    assert split["battery"]["decode_tps"]["n"] == 3
    assert split["ac"]["n"] == 0
    assert split["ac"]["decode_tps"] is None


def test_empty_input_describes_nothing_rather_than_claiming_agreement():
    split = power_split([])

    assert split["mixed"] is False
    assert split["ac"]["decode_tps"] is None
    assert split["battery"]["decode_tps"] is None
