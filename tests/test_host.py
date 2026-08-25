"""Reading the power source.

Apple Silicon throttles ~12% on battery, which is wider than most of the
model-to-model differences this project reports. `power_state` is therefore
part of the measurement, not decoration — and its failure mode matters as much
as its success: an unreadable `pmset` that defaulted to AC would reintroduce
the confound invisibly, on every row, with nothing to show it had happened.
"""

from __future__ import annotations

import pytest

from localslm.host import PowerState, _parse_pmset, power_label

# Real `pmset -g batt` output, both sources.
ON_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=12345)\t100%; charged; 0:00 remaining present: true"
)
ON_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=12345)\t87%; discharging; 4:32 remaining present: true"
)


def test_reads_ac_power():
    state = _parse_pmset(ON_AC)

    assert state.on_ac is True
    assert state.battery_pct == 100


def test_reads_battery_power_with_its_charge():
    state = _parse_pmset(ON_BATTERY)

    assert state.on_ac is False
    assert state.battery_pct == 87


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty output"),
        pytest.param("pmset: command not found", id="error message"),
        pytest.param("Now drawing from 'Solar Panel'", id="unrecognised source"),
    ],
)
def test_unreadable_output_does_not_silently_claim_ac(text):
    """The dangerous default. Claiming AC on failure would mark every throttled
    battery measurement as unthrottled, and `power_split` would report a clean
    single-source rollup that is nothing of the sort."""
    assert _parse_pmset(text) is None


def test_unknown_is_distinct_from_battery_in_the_label():
    """The label becomes a benchmark cell axis, so folding "we don't know" into
    "battery" would merge two different conditions into one cell."""
    assert power_label(PowerState(on_ac=True, battery_pct=None)) == "ac"
    assert power_label(PowerState(on_ac=False, battery_pct=50)) == "battery"
    assert power_label(PowerState(on_ac=None, battery_pct=None)) == "unknown"
