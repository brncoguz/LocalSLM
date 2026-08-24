"""Test suites.

A suite is a JSONL file: one item per line, so a diff shows exactly which test
changed rather than reflowing the whole file. Each item names the grader that
scores it, which is what keeps the runner free of task-specific branches.

`verified_by` records who vouched for the expected answer. Items drafted by a
model are usable for exploration but only human-verified ones should gate
anything — a model-written test scored by a model is a closed loop that looks
impressive and measures nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from localslm.config import path as repo_path

SUITES_DIR = repo_path("suites")


@dataclass(frozen=True)
class TaskItem:
    id: str
    task: str  # grader key
    prompt: str
    expected: Any = None
    context: str | None = None  # passage, haystack, or few-shot corpus
    output_schema: dict[str, Any] | None = None
    category: str = "general"
    lang: str = "en"
    difficulty: str = "medium"
    # Deliberately unanswerable from the given context. Answering anyway is a
    # hallucination, which is the number that decides local-RAG viability.
    unanswerable: bool = False
    constraints: tuple[dict[str, Any], ...] = ()
    verified_by: str = "llm"  # "human" once reviewed
    notes: str = ""
    tags: tuple[str, ...] = ()

    @property
    def is_verified(self) -> bool:
        return self.verified_by == "human"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskItem:
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        for key in ("constraints", "tags"):
            if key in clean and clean[key] is not None:
                clean[key] = tuple(clean[key])
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["constraints"] = list(self.constraints)
        out["tags"] = list(self.tags)
        return {k: v for k, v in out.items() if v not in (None, "", (), [])}


@dataclass
class Suite:
    name: str
    items: list[TaskItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def verified(self) -> int:
        return sum(1 for i in self.items if i.is_verified)

    @property
    def tasks(self) -> set[str]:
        return {i.task for i in self.items}

    def content_hash(self) -> str:
        """Written into every run so a score change can be attributed to a
        suite edit instead of guessed at."""
        blob = json.dumps(
            [i.to_dict() for i in self.items], sort_keys=True, ensure_ascii=False
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def suite_path(name: str) -> Path:
    return SUITES_DIR / f"{name}.jsonl"


def load(name: str) -> Suite:
    path = suite_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no suite {name!r} at {path}")
    items = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            items.append(TaskItem.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return Suite(name=name, items=items)


def save(suite: Suite) -> Path:
    path = suite_path(suite.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(i.to_dict(), ensure_ascii=False, sort_keys=True) for i in suite.items
        )
        + "\n"
    )
    return path


def available() -> list[str]:
    if not SUITES_DIR.exists():
        return []
    return sorted(p.stem for p in SUITES_DIR.glob("*.jsonl"))
