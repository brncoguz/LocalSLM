"""Continuous system sampling.

One background thread for the whole process, holding a ring buffer. Individual
requests take *slices* of that stream rather than starting their own probe —
partly because `powermetrics` cannot be spawned thousands of times, and partly
because a per-request sampler would disagree with the per-run timeseries about
what the machine was doing.

Deliberately conservative about energy: at a 4 Hz sample rate a two-second
request has ~8 samples and an unattributable baseline draw, so per-request
joules would be fiction. `SampleWindow.energy_j` stays None unless the window
is long enough to mean something.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import psutil

Pressure = Literal["normal", "warn", "critical"]

# Below this, an energy figure says more about sampling noise than the model.
MIN_ENERGY_WINDOW_S = 60.0


# `sysctl -n vm.swapusage` → "total = 8192.00M  used = 7147.31M  free = 1044.69M"
_SWAP_RE = re.compile(r"used\s*=\s*([\d.]+)([KMG])", re.IGNORECASE)
_UNIT = {"K": 1024, "M": 1024**2, "G": 1024**3}


def _parse_swap_used(text: str) -> int | None:
    """Swap bytes from `sysctl -n vm.swapusage`, or None if it can't be read.

    None rather than 0 is the whole point. A row is excluded from every average
    when swap grew during it, so a parser that answered 0 to unreadable output
    would not fail loudly — it would silently switch the exclusion rule off and
    let measurements taken during thrashing be published as merely slow.
    """
    if m := _SWAP_RE.search(text or ""):
        try:
            return int(float(m.group(1)) * _UNIT[m.group(2).upper()])
        except (ValueError, KeyError):
            return None
    return None


def _vm_stat() -> tuple[int | None, float]:
    """Return (swap_used_bytes | None, free_pct).

    Swap comes from sysctl rather than psutil because psutil's macOS numbers
    don't account for compressed memory, which is most of what macOS actually
    does under pressure. psutil is kept as a fallback anyway: it undercounts
    here, but an undercount is still evidence, and the alternative is throwing
    away every measurement on a machine whose `sysctl` output we can't read.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""

    swap_used = _parse_swap_used(out)
    if swap_used is None:
        try:
            swap_used = psutil.swap_memory().used
        except Exception:
            swap_used = None

    vm = psutil.virtual_memory()
    return swap_used, vm.available / vm.total * 100


def _pressure_from_free(free_pct: float) -> Pressure:
    # macOS `memory_pressure` needs a subprocess per call, which is too slow to
    # sample at 4 Hz. Free-percentage thresholds track it closely enough to
    # decide "abort this run", which is the only decision we make with it.
    if free_pct < 8:
        return "critical"
    if free_pct < 20:
        return "warn"
    return "normal"


def _bracketing(ring: list[SystemSample], t0: float, t1: float) -> list[SystemSample]:
    """The samples immediately either side of a range that contains none.

    Used for requests shorter than the sample interval. Returns at most two
    samples, in time order, or an empty list when the ring holds nothing.
    """
    before = [s for s in ring if s.t_mono < t0]
    after = [s for s in ring if s.t_mono > t1]
    return [s for s in (before[-1] if before else None, after[0] if after else None) if s]


def _is_supervisor(name: str, exe: str, cmdline: list[str]) -> bool:
    """`ollama serve` itself, as opposed to a runner child it spawned.

    The runner is a differently-named binary (`llama-server` today) living
    under the Ollama install prefix, so the exe path matches but the basename
    does not. Checking the subcommand as well keeps a future `ollama runner`
    child from being mistaken for the supervisor.
    """
    basename = exe.rsplit("/", 1)[-1] or name
    if basename not in ("ollama", "ollama.exe"):
        return False
    return "serve" in [a.lower() for a in cmdline[1:]]


@dataclass(frozen=True)
class SystemSample:
    t_mono: float
    # Every Ollama process including the supervisor: the machine-level figure,
    # and the right input to the swap/pressure decision.
    ollama_rss_bytes: int
    # Inference runners only. Attributable to a model when exactly one is up.
    runner_rss_bytes: int
    runner_count: int
    free_bytes: int
    # None when neither sysctl nor psutil could say. Distinct from 0.
    swap_used_bytes: int | None
    pressure: Pressure
    cpu_pct: float


@dataclass(frozen=True)
class SampleWindow:
    """What the machine did during one slice of wall-clock time."""

    n: int
    duration_s: float
    peak_rss_bytes: int
    mean_rss_bytes: float
    # None when swap could not be read across this window — "we don't know",
    # which `_persist` treats as a reason to exclude rather than as "clean".
    swap_delta_bytes: int | None
    pressure_max: Pressure
    # Runner processes only, so this can be charged to a model — but only when
    # `runner_count_max` is 1. With several runners up the peak is a sum again
    # and the caller must fall back to Ollama's own per-model accounting.
    peak_runner_rss_bytes: int = 0
    runner_count_max: int = 0
    # Stays None until a power-instrumented run fills it in. At 4 Hz sampling a
    # two-second request has ~8 samples plus an unattributable baseline draw, so
    # a per-request joule figure would describe noise, not the model. Energy is
    # only meaningful over sustained windows (see MIN_ENERGY_WINDOW_S).
    energy_j: float | None = None

    @property
    def swapped(self) -> bool:
        """A cell that swapped is not comparable to one that didn't, and is
        badged rather than quietly averaged in.

        False when swap is unknown — that case is handled separately, because
        "did not swap" and "we could not tell" must not collapse into the same
        answer here.
        """
        return self.swap_delta_bytes is not None and self.swap_delta_bytes > 0

    @property
    def swap_known(self) -> bool:
        return self.swap_delta_bytes is not None


class SystemSampler:
    def __init__(self, interval_s: float = 0.25, ring_seconds: int = 3600) -> None:
        self.interval_s = interval_s
        self._ring: deque[SystemSample] = deque(maxlen=int(ring_seconds / interval_s))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._abort_latched = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> SystemSampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- sampling ---------------------------------------------------------

    def _runner_rss(self) -> tuple[int, int, int]:
        """Return (total_rss, runner_rss, runner_count).

        `ollama serve` is a supervisor — ~30 MB idle, ~60 MB with a model up;
        the weights live in a `llama-server` (or MLX runner) child it spawns,
        which does not have "ollama" in its process *name*, only in its exe
        path. Matching on name alone silently reports 60 MB for a 9B model.

        Both figures are kept because they answer different questions. The
        total is what to compare against the 16 GB ceiling and what decides
        whether the machine is thrashing. The runner-only figure is the one
        that can be attributed to a model — but only when a single runner is
        resident, which is why the count comes back with it. Summing the two
        roles into one number is what made a 1B model read 5.2 GB.
        """
        total = 0
        runner = 0
        count = 0
        for proc in psutil.process_iter(["name", "exe", "cmdline", "memory_info"]):
            try:
                info = proc.info
                name = (info["name"] or "").lower()
                exe = (info["exe"] or "").lower()
                if "ollama" not in name and "ollama" not in exe:
                    continue
                mem = info["memory_info"]
                if not mem:
                    continue
                total += mem.rss
                if not _is_supervisor(name, exe, info["cmdline"] or []):
                    runner += mem.rss
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total, runner, count

    def _sample(self) -> SystemSample:
        rss, runner_rss, runner_count = self._runner_rss()
        swap, free_pct = _vm_stat()
        pressure = _pressure_from_free(free_pct)
        if pressure == "critical":
            self._abort_latched = True
        return SystemSample(
            t_mono=time.perf_counter(),
            ollama_rss_bytes=rss,
            runner_rss_bytes=runner_rss,
            runner_count=runner_count,
            free_bytes=int(psutil.virtual_memory().available),
            swap_used_bytes=swap,
            pressure=pressure,
            cpu_pct=psutil.cpu_percent(interval=None),
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = self._sample()
                with self._lock:
                    self._ring.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    # -- reading ----------------------------------------------------------

    @property
    def latest(self) -> SystemSample | None:
        with self._lock:
            return self._ring[-1] if self._ring else None

    @property
    def abort_latched(self) -> bool:
        """Set once memory pressure hit critical. Latched rather than
        instantaneous: a run that briefly thrashed is suspect throughout."""
        return self._abort_latched

    def reset_abort(self) -> None:
        self._abort_latched = False

    def window(self, t0: float, t1: float) -> SampleWindow:
        duration = max(t1 - t0, 0.0)
        with self._lock:
            ring = list(self._ring)
        rows = [s for s in ring if t0 <= s.t_mono <= t1]

        # A request faster than the sample interval can finish between two
        # ticks and enclose no sample at all — the 1B model answers in 0.2 s at
        # a 0.25 s interval. Falling back to the samples either side keeps the
        # machine-state evidence rather than reporting "nothing happened", which
        # would have quietly exempted the fastest requests from the swap rule.
        bracket = rows or _bracketing(ring, t0, t1)
        if not bracket:
            # No evidence at all: the sampler was not running. Unknown swap
            # rather than zero, so `_persist` excludes instead of publishing a
            # measurement taken in unobserved conditions.
            return SampleWindow(0, duration, 0, 0.0, None, "normal")

        rss = [s.ollama_rss_bytes for s in bracket]
        runner_rss = [s.runner_rss_bytes for s in bracket]
        swaps = [s.swap_used_bytes for s in bracket]
        order: dict[str, int] = {"normal": 0, "warn": 1, "critical": 2}
        worst = max(bracket, key=lambda s: order[s.pressure]).pressure
        # One unreadable sample poisons the delta: the difference between two
        # endpoints is meaningless if either is a guess.
        delta = (
            None if swaps[0] is None or swaps[-1] is None else max(0, swaps[-1] - swaps[0])
        )
        return SampleWindow(
            n=len(rows),
            duration_s=duration,
            peak_rss_bytes=max(rss),
            mean_rss_bytes=sum(rss) / len(rss),
            swap_delta_bytes=delta,
            pressure_max=worst,
            energy_j=None,
            peak_runner_rss_bytes=max(runner_rss),
            runner_count_max=max(s.runner_count for s in bracket),
        )

    def series(self, t0: float, t1: float, hz: float = 1.0) -> list[SystemSample]:
        """Downsample the ring for the per-run timeseries chart."""
        with self._lock:
            rows = [s for s in self._ring if t0 <= s.t_mono <= t1]
        if not rows:
            return []
        step = max(1, round((1 / hz) / self.interval_s))
        return rows[::step]
