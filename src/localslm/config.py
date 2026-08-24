"""Configuration loading.

Config is a plain nested dict reached by dotted path, not a pydantic model.
That is deliberate: presets need to overlay arbitrary keys at runtime
(`cfg.with_preset("quant_ladder")`), and a typed model would force every
experimental knob to be declared up front. Validation happens where a value
is consumed, not where it is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_MISSING = object()


class ConfigError(KeyError):
    pass


class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, path: str) -> Any:
        value = self.get(path, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"no config key {path!r}")
        return value

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-path keys replaced."""
        data = copy.deepcopy(self._data)
        for path, value in overrides.items():
            node = data
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data)

    def with_preset(self, name: str) -> Config:
        presets = self.get("presets", {})
        if name not in presets:
            known = ", ".join(sorted(presets)) or "none"
            raise ConfigError(f"unknown preset {name!r} (known: {known})")
        return self.with_overrides(presets[name])

    @property
    def presets(self) -> list[str]:
        return sorted(self.get("presets", {}))

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def fingerprint(self) -> str:
        """Stable hash of the resolved config, minus the preset catalogue.

        Written into every run row so a metric change can be attributed to a
        config change rather than guessed at.
        """
        data = copy.deepcopy(self._data)
        data.pop("presets", None)
        blob = yaml.safe_dump(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def path(*parts: str) -> Path:
    """Resolve a repo-relative path."""
    return REPO_ROOT.joinpath(*parts)


@lru_cache(maxsize=1)
def load_config() -> Config:
    with open(path("config", "app.yaml")) as fh:
        data = yaml.safe_load(fh)
    # A single env override, because the Ollama host is the one thing that
    # legitimately differs between machines.
    if host := os.environ.get("LOCALSLM_OLLAMA_HOST"):
        data.setdefault("ollama", {})["host"] = host
    if os.environ.get("LOCALSLM_ENABLE_POWERMETRICS", "").lower() in {"1", "true", "yes"}:
        data.setdefault("telemetry", {})["powermetrics"] = True
    return Config(data)
