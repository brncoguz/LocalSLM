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


def _vm_stat() -> tuple[int, int]:
    """Return (swap_used_bytes, free_pct).

    Swap comes from sysctl rather than psutil because psutil's macOS swap
    numbers don't account for compressed memory, which is most of what macOS
    actually does under pressure.
    """
    swap_used = 0
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2
        ).stdout
        if m := _SWAP_RE.search(out):
            swap_used = int(float(m.group(1)) * _UNIT[m.group(2).upper()])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
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


@dataclass(frozen=True)
class SystemSample:
    t_mono: float
    ollama_rss_bytes: int
    free_bytes: int
    swap_used_bytes: int
    pressure: Pressure
    cpu_pct: float


@dataclass(frozen=True)
class SampleWindow:
    """What the machine did during one slice of wall-clock time."""

    n: int
    duration_s: float
    peak_rss_bytes: int
    mean_rss_bytes: float
    swap_delta_bytes: int
    pressure_max: Pressure
    # Stays None until a power-instrumented run fills it in. At 4 Hz sampling a
    # two-second request has ~8 samples plus an unattributable baseline draw, so
    # a per-request joule figure would describe noise, not the model. Energy is
    # only meaningful over sustained windows (see MIN_ENERGY_WINDOW_S).
    energy_j: float | None = None

    @property
    def swapped(self) -> bool:
        """A cell that swapped is not comparable to one that didn't, and is
        badged rather than quietly averaged in."""
        return self.swap_delta_bytes > 0


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

    def _runner_rss(self) -> int:
        """Total resident bytes across every Ollama inference process.

        `ollama serve` is a ~10 MB supervisor; the weights live in a
        `llama-server` (or MLX runner) child it spawns, which does not have
        "ollama" in its process name. Matching on name alone silently reports
        10 MB for a 9B model. We match the executable path instead, and sum —
        Compare mode can have several runners resident at once, and the
        interesting question is always the total against the 16 GB ceiling.
        """
        total = 0
        for proc in psutil.process_iter(["name", "exe", "memory_info"]):
            try:
                info = proc.info
                name = (info["name"] or "").lower()
                exe = (info["exe"] or "").lower()
                if "ollama" not in name and "ollama" not in exe:
                    continue
                if mem := info["memory_info"]:
                    total += mem.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _sample(self) -> SystemSample:
        rss = self._runner_rss()
        swap, free_pct = _vm_stat()
        pressure = _pressure_from_free(free_pct)
        if pressure == "critical":
            self._abort_latched = True
        return SystemSample(
            t_mono=time.perf_counter(),
            ollama_rss_bytes=rss,
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
        with self._lock:
            rows = [s for s in self._ring if t0 <= s.t_mono <= t1]
        duration = max(t1 - t0, 0.0)
        if not rows:
            return SampleWindow(0, duration, 0, 0.0, 0, "normal")
        rss = [s.ollama_rss_bytes for s in rows]
        swaps = [s.swap_used_bytes for s in rows]
        order: dict[str, int] = {"normal": 0, "warn": 1, "critical": 2}
        worst = max(rows, key=lambda s: order[s.pressure]).pressure
        return SampleWindow(
            n=len(rows),
            duration_s=duration,
            peak_rss_bytes=max(rss),
            mean_rss_bytes=sum(rss) / len(rss),
            swap_delta_bytes=max(0, swaps[-1] - swaps[0]),
            pressure_max=worst,
            energy_j=None,
        )

    def series(self, t0: float, t1: float, hz: float = 1.0) -> list[SystemSample]:
        """Downsample the ring for the per-run timeseries chart."""
        with self._lock:
            rows = [s for s in self._ring if t0 <= s.t_mono <= t1]
        if not rows:
            return []
        step = max(1, round((1 / hz) / self.interval_s))
        return rows[::step]
