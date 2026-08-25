"""Turning a benchmark request into an ordered list of work.

A run is the cartesian product of models x suites x think levels x repeats.
Expanding it here — rather than looping inside the runner — is what lets an
experiment be described declaratively and reproduced exactly: `Unit.id` and the
per-unit seed are both derived from the run seed, so the same run replays.

Ordering is the subtle part. See `order_units`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from localslm.bench.suites import Suite, TaskItem

ThinkAxis = bool | str


@dataclass(frozen=True)
class Cell:
    """One experimental condition: a model under a fixed set of settings."""

    model: str
    axes: tuple[tuple[str, Any], ...] = ()

    @property
    def key(self) -> str:
        blob = json.dumps([self.model, sorted(self.axes)], sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    @property
    def axes_dict(self) -> dict[str, Any]:
        return dict(self.axes)

    def label(self) -> str:
        if not self.axes:
            return self.model
        bits = ", ".join(f"{k}={v}" for k, v in self.axes)
        return f"{self.model} ({bits})"


@dataclass(frozen=True)
class Unit:
    """One request to make."""

    cell: Cell
    item: TaskItem
    suite: str
    rep: int
    pass_idx: int
    is_warmup: bool
    seed: int

    @property
    def id(self) -> str:
        return f"{self.cell.key}:{self.suite}:{self.item.id}:{self.rep}"


@dataclass
class BenchSpec:
    models: list[str]
    suites: list[Suite]
    think_levels: list[ThinkAxis] = field(default_factory=lambda: [False])
    temperatures: list[float] = field(default_factory=lambda: [0.0])
    repeats: int = 3
    warmups: int = 1
    seed: int = 0
    # Recorded, never set: nothing here can plug in a charger. When it is
    # supplied the runner stamps it onto the cell so a run measured unplugged
    # can never be silently compared against one that wasn't. Apple Silicon
    # throttles ~12% on battery, which is wider than most model differences.
    power_source: str | None = None
    # Models that can't think are still requested with think=None; the adapter
    # records "unsupported" rather than the runner branching on capability.
    thinking_capable: dict[str, bool] = field(default_factory=dict)

    def cells(self) -> list[Cell]:
        out = []
        for model in self.models:
            for think in self.think_levels:
                # Asking a non-thinking model for four think levels would
                # quadruple its runtime to produce four identical rows.
                if think is not False and not self.thinking_capable.get(model, True):
                    continue
                for temp in self.temperatures:
                    axes: list[tuple[str, Any]] = []
                    if len(self.think_levels) > 1 or think is not False:
                        axes.append(("think", think))
                    if len(self.temperatures) > 1:
                        axes.append(("temperature", temp))
                    if self.power_source:
                        axes.append(("power", self.power_source))
                    out.append(Cell(model=model, axes=tuple(axes)))
        # Dedupe: collapsing think levels above can produce repeats.
        seen, unique = set(), []
        for c in out:
            if c.key not in seen:
                seen.add(c.key)
                unique.append(c)
        return unique

    def total_requests(self) -> int:
        """Exact count, not an estimate — warmups are per suite, not per item,
        so multiplying them by the item count inflates the figure by ~2x."""
        items = sum(len(s) for s in self.suites)
        measured = len(self.cells()) * items * self.repeats
        warmups = len(self.cells()) * len(self.suites) * self.warmups
        return measured + warmups

    def content_hash(self) -> str:
        blob = json.dumps(
            {
                "models": sorted(self.models),
                "suites": {s.name: s.content_hash() for s in self.suites},
                "think": [str(t) for t in self.think_levels],
                "temperatures": self.temperatures,
                "power_source": self.power_source,
                "repeats": self.repeats,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def unit_seed(run_seed: int, cell: Cell, item_id: str, rep: int) -> int:
    blob = f"{run_seed}:{cell.key}:{item_id}:{rep}".encode()
    return int.from_bytes(hashlib.sha1(blob).digest()[:4], "big")


def order_units(spec: BenchSpec, run_seed: int) -> list[Unit]:
    """Model-blocked, item-interleaved, repeated in rotated passes.

    Three constraints pull against each other:

    * Grouping strictly by model means the last model always runs on the
      hottest machine, so thermal drift becomes indistinguishable from model
      identity.
    * Grouping strictly by item means swapping models on every request. On
      16 GB a 9B reload costs seconds, so you would be measuring disk.
    * Reloading is unavoidable at some point, so it should at least be spread
      evenly across models.

    The compromise: one block per model per pass (one load each), items
    interleaved inside the block, and the model order rotated between passes so
    every model sees both an early-cold and a late-hot pass. `pass_idx` is
    recorded on each row so drift can be regressed out afterwards rather than
    assumed away.
    """
    cells = spec.cells()
    units: list[Unit] = []

    for pass_idx in range(spec.repeats):
        # Rotate so a different cell leads each pass.
        rotated = cells[pass_idx % len(cells) :] + cells[: pass_idx % len(cells)]
        for cell in rotated:
            warm = pass_idx == 0
            for suite in spec.suites:
                for w in range(spec.warmups if warm else 0):
                    item = suite.items[w % len(suite.items)]
                    units.append(
                        Unit(
                            cell,
                            item,
                            suite.name,
                            -1,
                            pass_idx,
                            True,
                            unit_seed(run_seed, cell, item.id, -1),
                        )
                    )
                for item in suite.items:
                    units.append(
                        Unit(
                            cell,
                            item,
                            suite.name,
                            pass_idx,
                            pass_idx,
                            False,
                            unit_seed(run_seed, cell, item.id, pass_idx),
                        )
                    )
    return units


def summarise_plan(spec: BenchSpec, units: Sequence[Unit]) -> dict[str, Any]:
    real = [u for u in units if not u.is_warmup]
    return {
        "models": spec.models,
        "suites": [
            {"name": s.name, "items": len(s), "verified": s.verified} for s in spec.suites
        ],
        "cells": [
            {"key": c.key, "label": c.label(), "model": c.model} for c in spec.cells()
        ],
        "think_levels": [str(t) for t in spec.think_levels],
        "repeats": spec.repeats,
        "warmups": spec.warmups,
        "total_units": len(units),
        "measured_units": len(real),
        "suite_hash": spec.content_hash(),
    }
