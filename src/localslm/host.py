"""Host facts and provenance.

Every benchmark row is only as trustworthy as what we know about the machine
that produced it. This module answers "what was this Mac doing at the time",
and produces a stable hash of the invariant parts so runs from an identical
environment collapse to one provenance row.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from localslm import __version__

# Ollama env vars that materially change measured performance. Recorded because
# a run taken with a different OLLAMA_NUM_PARALLEL is not comparable, and
# nothing in the numbers themselves would reveal that.
OLLAMA_ENV_KEYS = (
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_CONTEXT_LENGTH",
    "OLLAMA_HOST",
)


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _ollama_version() -> str | None:
    if not shutil.which("ollama"):
        return None
    try:
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip().split()[-1] if out.stdout else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class HostInfo:
    """Invariant facts about this machine. Hashed into `env_snapshots`."""

    chip: str | None
    cores: int
    ram_bytes: int
    macos: str
    python: str
    ollama_version: str | None
    localslm_version: str
    ollama_env: dict[str, str] = field(default_factory=dict)

    @property
    def ram_gb(self) -> float:
        """Binary GB — what the marketing sticker says (16, not 17.2)."""
        return round(self.ram_bytes / 1024**3, 1)

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@lru_cache(maxsize=1)
def host_info() -> HostInfo:
    ram = _sysctl("hw.memsize")
    cores = _sysctl("hw.ncpu")
    return HostInfo(
        chip=_sysctl("machdep.cpu.brand_string"),
        cores=int(cores) if cores else os.cpu_count() or 0,
        ram_bytes=int(ram) if ram else 0,
        macos=platform.mac_ver()[0] or platform.release(),
        python=platform.python_version(),
        ollama_version=_ollama_version(),
        localslm_version=__version__,
        ollama_env={k: v for k in OLLAMA_ENV_KEYS if (v := os.environ.get(k))},
    )


@dataclass(frozen=True)
class PowerState:
    """Whether we're on wall power. Apple Silicon throttles hard on battery,
    so a run that changed power source mid-flight is not internally comparable."""

    on_ac: bool
    battery_pct: int | None


def power_state() -> PowerState:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return PowerState(on_ac=True, battery_pct=None)
    on_ac = "AC Power" in out
    pct: int | None = None
    for token in out.replace(";", " ").split():
        if token.endswith("%") and token[:-1].isdigit():
            pct = int(token[:-1])
            break
    return PowerState(on_ac=on_ac, battery_pct=pct)


def disk_free_bytes(path: str = ".") -> int:
    return shutil.disk_usage(path).free
