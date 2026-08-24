"""The runtime contract.

The benchmark layer talks only to these types, never to Ollama directly. That
is what makes "which runtime is fastest on Apple Silicon" a question we can
answer later by writing one file, rather than by rewriting the bench harness.

When a runtime lacks a capability the request field is ignored and the recorded
row says so (`think="unsupported"`, `constraint_mode="unsupported"`) rather than
silently succeeding. Bench code never asks "am I on Ollama?".
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ThinkLevel = Literal["low", "medium", "high"]

# Ollama reports a `load_duration` of ~230-290 ms even when the weights are
# already resident — that is handle setup, not loading. Genuine cold loads on
# this hardware run 1.5-6.6 s. A threshold anywhere near zero therefore marks
# every request cold. Residency from /api/ps is authoritative; this is only the
# fallback for when that lookup fails.
COLD_LOAD_MS_FALLBACK = 1000.0


@dataclass(frozen=True)
class ModelRef:
    runtime: str
    name: str
    digest: str | None = None
    quant: str | None = None
    params_b: float | None = None
    ctx_max: int | None = None

    @property
    def key(self) -> str:
        return f"{self.runtime}:{self.name}"


@dataclass(frozen=True)
class Capabilities:
    completion: bool = True
    thinking: bool = False
    tools: bool = False
    vision: bool = False
    schema: bool = True  # Ollama accepts a JSON Schema in `format` for any model
    think_levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelInfo:
    ref: ModelRef
    family: str | None = None
    arch: str | None = None
    size_bytes: int = 0
    capabilities: Capabilities = field(default_factory=Capabilities)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size_bytes: int
    size_vram_bytes: int
    expires_at: str | None = None


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class DecodeOptions:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    num_ctx: int | None = None
    num_predict: int | None = None

    def as_ollama(self) -> dict[str, Any]:
        out: dict[str, Any] = {"temperature": self.temperature}
        for key in ("top_p", "top_k", "seed", "num_ctx", "num_predict"):
            if (value := getattr(self, key)) is not None:
                out[key] = value
        return out

    def fingerprint(self) -> str:
        blob = json.dumps(self.as_ollama(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


@dataclass(frozen=True)
class GenRequest:
    model: ModelRef
    messages: tuple[Message, ...]
    schema: dict[str, Any] | None = None
    tools: tuple[dict[str, Any], ...] | None = None
    think: bool | ThinkLevel | None = None
    options: DecodeOptions = DecodeOptions()
    # Never left to the runtime default in a benchmark: keep_alive decides
    # whether the next request is cold, which is a measurement, not luck.
    keep_alive: str = "10m"
    timeout_s: float = 300.0

    def prompt_hash(self) -> str:
        blob = json.dumps(
            [[m.role, m.content] for m in self.messages], sort_keys=True
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Chunk:
    """One streamed piece. `t_mono` is stamped before any parsing so the
    timestamp measures the network/runtime, not our own JSON handling."""

    t_mono: float
    text: str = ""
    thinking: str = ""
    done: bool = False
    raw: dict[str, Any] | None = None

    @property
    def kind(self) -> Literal["thinking", "answer", "empty"]:
        if self.thinking:
            return "thinking"
        if self.text:
            return "answer"
        return "empty"


@dataclass(frozen=True)
class ServerTiming:
    """Ollama's own counters, in ms. Recorded alongside the client numbers and
    never merged with them — a persistent divergence is a real finding about
    our streaming path, and averaging would hide it."""

    total_ms: float = 0.0
    load_ms: float = 0.0
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    output_tokens: int = 0
    eval_ms: float = 0.0

    @property
    def prefill_tps(self) -> float | None:
        return self.prompt_tokens / (self.prompt_ms / 1000) if self.prompt_ms else None

    @property
    def decode_tps(self) -> float | None:
        return self.output_tokens / (self.eval_ms / 1000) if self.eval_ms else None

    @classmethod
    def from_done(cls, payload: dict[str, Any]) -> ServerTiming:
        ns = 1_000_000  # ollama reports nanoseconds
        return cls(
            total_ms=payload.get("total_duration", 0) / ns,
            load_ms=payload.get("load_duration", 0) / ns,
            prompt_tokens=payload.get("prompt_eval_count", 0) or 0,
            prompt_ms=payload.get("prompt_eval_duration", 0) / ns,
            output_tokens=payload.get("eval_count", 0) or 0,
            eval_ms=payload.get("eval_duration", 0) / ns,
        )


@dataclass
class GenResult:
    model: ModelRef
    text: str = ""
    thinking: str = ""
    finish_reason: str = "stop"

    # `ttft` is the first chunk of any kind; `ttfat` is the first *answer*
    # token. For a thinking model these differ by tens of seconds, and only
    # the second one describes what a user actually waits for.
    ttft_ms: float | None = None
    ttfat_ms: float | None = None
    total_ms: float = 0.0
    queue_ms: float = 0.0

    inter_token_ms: list[float] = field(default_factory=list)
    server: ServerTiming | None = None
    thinking_chunks: int = 0
    answer_chunks: int = 0
    cold: bool = False
    error: str | None = None
    raw_done: dict[str, Any] | None = None

    @property
    def itl_p50_ms(self) -> float | None:
        return statistics.median(self.inter_token_ms) if self.inter_token_ms else None

    @property
    def itl_p95_ms(self) -> float | None:
        if len(self.inter_token_ms) < 2:
            return None
        ordered = sorted(self.inter_token_ms)
        return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]

    @property
    def itl_stddev_ms(self) -> float | None:
        """Jitter. A model with the same mean tok/s but double the jitter feels
        visibly worse to stream, and no throughput number captures that."""
        if len(self.inter_token_ms) < 2:
            return None
        return statistics.stdev(self.inter_token_ms)

    @property
    def think_ratio(self) -> float | None:
        """Share of generated tokens spent thinking rather than answering."""
        if not self.server or not self.server.output_tokens:
            return None
        total = self.thinking_chunks + self.answer_chunks
        return self.thinking_chunks / total if total else None


class Runtime(Protocol):
    name: str

    async def health(self) -> dict[str, Any]: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def show(self, name: str) -> ModelInfo: ...
    async def loaded(self) -> list[LoadedModel]: ...
    async def warm(self, name: str, keep_alive: str = "10m") -> float: ...
    async def unload(self, name: str) -> None: ...
    async def delete(self, name: str) -> None: ...
    def pull(self, name: str) -> AsyncIterator[dict[str, Any]]: ...
    def stream(self, req: GenRequest) -> AsyncIterator[Chunk]: ...
    async def generate(self, req: GenRequest) -> GenResult: ...
