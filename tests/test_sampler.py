"""Separating the Ollama supervisor from the runners it spawns.

`ollama serve` is a ~30-60 MB supervisor process actually named "ollama"; the
weights live in a `llama-server` child whose name is different but whose exe
path sits under the Ollama install prefix. Summing both into one figure is what
made `llama3.2:1b` report 5.2 GB of peak RAM. Both numbers are still wanted —
the total decides whether the machine is thrashing, the runner total is the
only one chargeable to a model — so what these test is that they stay apart.

Process shapes below are copied from a real M1 Pro with Ollama 0.32.9 running.
"""

from __future__ import annotations

import psutil
import pytest

from localslm.telemetry.sampler import (
    SystemSample,
    SystemSampler,
    _is_supervisor,
    _parse_swap_used,
    _pressure_from_free,
)

SUPERVISOR = ("ollama", "/opt/homebrew/bin/ollama", ["ollama", "serve"])
RUNNER = (
    "llama-server",
    "/opt/homebrew/cellar/ollama/0.32.9/libexec/lib/ollama/llama-server",
    [
        "llama-server",
        "--model",
        "/Users/me/.ollama/models/blobs/sha256-74701a",
        "-c",
        "4096",
    ],
)


class FakeProc:
    def __init__(self, name, exe, cmdline, rss):
        self.info = {
            "name": name,
            "exe": exe,
            "cmdline": cmdline,
            "memory_info": type("Mem", (), {"rss": rss})(),
        }


def test_supervisor_and_runner_are_told_apart():
    assert _is_supervisor(*SUPERVISOR) is True
    assert _is_supervisor(*RUNNER) is False


def test_a_runner_binary_named_ollama_is_not_mistaken_for_the_supervisor():
    """Ollama has shipped its runner as `ollama runner` before. Matching on the
    binary name alone would charge a whole model's memory to the supervisor
    bucket and report ~0 for the model."""
    assert (
        _is_supervisor("ollama", "/opt/homebrew/bin/ollama", ["ollama", "runner"]) is False
    )


@pytest.mark.parametrize(
    "procs, total, runner, count",
    [
        pytest.param(
            [FakeProc(*SUPERVISOR, 60_000_000), FakeProc(*RUNNER, 1_500_000_000)],
            1_560_000_000,
            1_500_000_000,
            1,
            id="one model resident",
        ),
        pytest.param(
            [
                FakeProc(*SUPERVISOR, 60_000_000),
                FakeProc(*RUNNER, 1_500_000_000),
                FakeProc(*RUNNER, 6_500_000_000),
            ],
            8_060_000_000,
            8_000_000_000,
            2,
            id="compare mode, two runners",
        ),
        pytest.param(
            [FakeProc(*SUPERVISOR, 30_000_000)],
            30_000_000,
            0,
            0,
            id="server up, nothing loaded",
        ),
    ],
)
def test_runner_total_excludes_the_supervisor(monkeypatch, procs, total, runner, count):
    monkeypatch.setattr(psutil, "process_iter", lambda _fields: iter(procs))

    assert SystemSampler()._runner_rss() == (total, runner, count)


def test_unrelated_processes_are_ignored(monkeypatch):
    other = FakeProc(
        "Google Chrome", "/Applications/Chrome.app/chrome", ["chrome"], 4_000_000_000
    )
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda _fields: iter([other, FakeProc(*RUNNER, 1_500_000_000)]),
    )

    assert SystemSampler()._runner_rss() == (1_500_000_000, 1_500_000_000, 1)


class TestSwapParsing:
    """`sysctl -n vm.swapusage` is the input to the exclusion rule.

    A parser that answers 0 to output it cannot read does not fail loudly — it
    silently switches the rule off, and measurements taken while the machine
    was thrashing get published as merely slow. So "unreadable" has to be a
    distinct answer from "no swap in use".
    """

    def test_reads_a_real_sysctl_line(self):
        line = "total = 8192.00M  used = 7147.31M  free = 1044.69M  (encrypted)"

        assert _parse_swap_used(line) == pytest.approx(7147.31 * 1024**2)

    @pytest.mark.parametrize("unit, factor", [("K", 1024), ("M", 1024**2), ("G", 1024**3)])
    def test_honours_each_unit_suffix(self, unit, factor):
        assert _parse_swap_used(f"used = 2.00{unit}") == 2 * factor

    def test_zero_swap_is_zero_not_unknown(self):
        """The common case on a healthy machine, and it must stay distinguishable
        from a parse failure."""
        assert _parse_swap_used("total = 0.00M  used = 0.00M  free = 0.00M") == 0

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("", id="empty output"),
            pytest.param("sysctl: unknown oid 'vm.swapusage'", id="error message"),
            pytest.param("total = 8192.00M  free = 1044.69M", id="no used field"),
            pytest.param("used = 7147.31Z", id="unrecognised unit"),
            pytest.param("used = notanumber M", id="non-numeric"),
        ],
    )
    def test_unreadable_output_is_unknown_rather_than_zero(self, text):
        assert _parse_swap_used(text) is None


class TestPressureThresholds:
    """These three bands decide whether a run aborts and whether a row is
    excluded. They are the only use of the free-memory percentage, so the
    boundaries are load-bearing and worth stating explicitly."""

    @pytest.mark.parametrize(
        "free_pct, expected",
        [
            (0.0, "critical"),
            (7.9, "critical"),
            (8.0, "warn"),
            (19.9, "warn"),
            (20.0, "normal"),
            (100.0, "normal"),
        ],
    )
    def test_bands(self, free_pct, expected):
        assert _pressure_from_free(free_pct) == expected


class TestWindow:
    """How a slice of the ring becomes one request's machine-state figures."""

    def sampler_with(self, samples):
        s = SystemSampler()
        for sample in samples:
            s._ring.append(sample)
        return s

    def sample(self, t, *, swap=0, pressure="normal", rss=1_000, runners=1):
        return SystemSample(
            t_mono=t,
            ollama_rss_bytes=rss,
            runner_rss_bytes=rss,
            runner_count=runners,
            free_bytes=8_000_000_000,
            swap_used_bytes=swap,
            pressure=pressure,
            cpu_pct=10.0,
        )

    def test_swap_delta_is_measured_across_the_window(self):
        s = self.sampler_with([self.sample(1.0, swap=100), self.sample(2.0, swap=180)])

        assert s.window(0.5, 2.5).swap_delta_bytes == 80

    def test_swap_freed_during_the_window_is_not_negative_growth(self):
        """Swap shrinking means the machine recovered, not that it swapped
        backwards. Only growth marks a row as incomparable."""
        s = self.sampler_with([self.sample(1.0, swap=500), self.sample(2.0, swap=200)])

        window = s.window(0.5, 2.5)
        assert window.swap_delta_bytes == 0
        assert window.swapped is False

    def test_pressure_is_the_worst_seen_not_the_last(self):
        """A request that briefly hit critical is suspect even if it ended
        calm — taking the last sample would let a spike disappear."""
        s = self.sampler_with(
            [
                self.sample(1.0, pressure="normal"),
                self.sample(2.0, pressure="critical"),
                self.sample(3.0, pressure="normal"),
            ]
        )

        assert s.window(0.5, 3.5).pressure_max == "critical"

    def test_boundaries_are_inclusive(self):
        s = self.sampler_with([self.sample(1.0), self.sample(2.0), self.sample(3.0)])

        assert s.window(1.0, 3.0).n == 3

    def test_a_request_faster_than_the_sample_interval_still_gets_evidence(self):
        """The 1B model answers in 0.2 s and the sampler ticks every 0.25 s, so
        a request can enclose no sample at all. Reporting "nothing happened"
        would exempt the fastest requests from the swap rule entirely."""
        s = self.sampler_with([self.sample(1.0, swap=100), self.sample(2.0, swap=400)])

        window = s.window(1.4, 1.6)
        assert window.n == 0  # nothing strictly inside
        assert window.swap_delta_bytes == 300  # but the bracket still tells us
        assert window.swapped is True

    def test_an_unreadable_swap_sample_poisons_the_delta(self):
        """Differencing two endpoints is meaningless if either is a guess."""
        s = self.sampler_with([self.sample(1.0, swap=None), self.sample(2.0, swap=400)])

        window = s.window(0.5, 2.5)
        assert window.swap_delta_bytes is None
        assert window.swap_known is False

    def test_no_samples_at_all_is_unknown_rather_than_clean(self):
        """An empty ring means the sampler was not running. Zero would claim
        the machine was observed and found idle."""
        window = SystemSampler().window(0.0, 1.0)

        assert window.swap_delta_bytes is None
        assert window.swap_known is False
