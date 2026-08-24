"""Progress events emitted while a run executes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchEvent:
    kind: str  # plan|unit|cell|warning|aborted|done
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
