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

from localslm.telemetry.sampler import SystemSampler, _is_supervisor

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
