"""The one place a generation is timed, sampled and persisted.

Every metric's validity lives here. Chat, Compare and the benchmark runner all
go through this function, which is why a number under a chat bubble is the
same number the report aggregates — there is no second code path that could
compute it differently.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Any

from localslm.db import insert, new_id
from localslm.host import power_state
from localslm.runtimes.base import Chunk, GenRequest, GenResult, ServerTiming
from localslm.telemetry.sampler import SystemSampler

# Full outputs are kept so a grader bugfix can re-score without re-running
# inference, but a runaway generation shouldn't bloat the row.
MAX_STORED_CHARS = 100_000


@dataclass
class Provenance:
    """Where this generation came from. Denormalised onto the request row
    because these fields *are* the experiment."""

    kind: str = "chat"
    run_id: str | None = None
    session_id: str | None = None
    unit_id: str | None = None
    parent_id: str | None = None
    pass_idx: int = 0
    rep: int = 0
    cell_key: str | None = None
    task: str | None = None
    item_id: str | None = None
    prompt_ref: str | None = None
    excluded: bool = False
    exclude_reason: str | None = None
    concurrency: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class MeasuredSession:
    def __init__(self, runtime: Any, conn: Connection, sampler: SystemSampler) -> None:
        self.runtime = runtime
        self.conn = conn
        self.sampler = sampler

    async def stream(
        self, req: GenRequest, prov: Provenance | None = None
    ) -> AsyncIterator[tuple[Chunk | None, GenResult | None]]:
        """Yield (chunk, None) while streaming, then (None, result) once.

        Streaming and measuring are the same pass — the UI sees exactly the
        token timings that get written to the database.
        """
        prov = prov or Provenance()
        result = GenResult(model=req.model)
        text: list[str] = []
        thinking: list[str] = []
        t_start = time.perf_counter()
        t_prev: float | None = None

        try:
            async for chunk in self.runtime.stream(req):
                if chunk.done:
                    result.raw_done = chunk.raw
                    if chunk.raw:
                        result.server = ServerTiming.from_done(chunk.raw)
                        result.finish_reason = chunk.raw.get("done_reason") or "stop"
                    continue
                kind = chunk.kind
                if kind == "empty":
                    continue
                if result.ttft_ms is None:
                    result.ttft_ms = (chunk.t_mono - t_start) * 1000
                if kind == "thinking":
                    thinking.append(chunk.thinking)
                    result.thinking_chunks += 1
                else:
                    if result.ttfat_ms is None:
                        result.ttfat_ms = (chunk.t_mono - t_start) * 1000
                    text.append(chunk.text)
                    result.answer_chunks += 1
                if t_prev is not None:
                    result.inter_token_ms.append((chunk.t_mono - t_prev) * 1000)
                t_prev = chunk.t_mono
                yield chunk, None
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.finish_reason = "error"

        t_end = time.perf_counter()
        result.total_ms = (t_end - t_start) * 1000
        result.text = "".join(text)
        result.thinking = "".join(thinking)
        result.cold = bool(result.server and result.server.load_ms > 100)

        self._persist(req, result, prov, t_start, t_end)
        yield None, result

    async def generate(self, req: GenRequest, prov: Provenance | None = None) -> GenResult:
        result: GenResult | None = None
        async for _chunk, done in self.stream(req, prov):
            if done is not None:
                result = done
        assert result is not None
        return result

    # -- persistence ------------------------------------------------------

    def _persist(
        self,
        req: GenRequest,
        result: GenResult,
        prov: Provenance,
        t_start: float,
        t_end: float,
    ) -> None:
        window = self.sampler.window(t_start, t_end)
        srv = result.server
        power = power_state()
        opts = req.options

        # A window that swapped is not comparable to one that didn't. Mark it
        # here rather than letting it average in downstream looking merely slow.
        excluded = prov.excluded
        reason = prov.exclude_reason
        if window.swapped and not excluded:
            excluded, reason = True, "swapped"
        elif window.pressure_max == "critical" and not excluded:
            excluded, reason = True, "memory_pressure"

        row: dict[str, Any] = {
            "id": new_id(),
            "ts": time.time(),
            "kind": prov.kind,
            "run_id": prov.run_id,
            "session_id": prov.session_id,
            "unit_id": prov.unit_id,
            "parent_id": prov.parent_id,
            "pass_idx": prov.pass_idx,
            "runtime": req.model.runtime,
            "model": req.model.name,
            "model_digest": req.model.digest,
            "quant": req.model.quant,
            "params_b": req.model.params_b,
            "prompt_ref": prov.prompt_ref,
            "prompt_hash": req.prompt_hash(),
            "task": prov.task,
            "item_id": prov.item_id,
            "cell_key": prov.cell_key,
            "rep": prov.rep,
            "excluded": int(excluded),
            "exclude_reason": reason,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "top_k": opts.top_k,
            "seed": opts.seed,
            "num_ctx": opts.num_ctx,
            "num_predict": opts.num_predict,
            "think": _think_label(req.think),
            "constraint_mode": "schema" if req.schema else "none",
            "concurrency": prov.concurrency,
            "keep_alive": req.keep_alive,
            "ttft_ms": result.ttft_ms,
            "ttfat_ms": result.ttfat_ms,
            "total_ms": result.total_ms,
            "itl_p50_ms": result.itl_p50_ms,
            "itl_p95_ms": result.itl_p95_ms,
            "itl_stddev_ms": result.itl_stddev_ms,
            "srv_total_ms": srv.total_ms if srv else None,
            "srv_load_ms": srv.load_ms if srv else None,
            "prompt_tokens": srv.prompt_tokens if srv else None,
            "srv_prompt_ms": srv.prompt_ms if srv else None,
            "output_tokens": srv.output_tokens if srv else None,
            "srv_eval_ms": srv.eval_ms if srv else None,
            "prefill_tps": srv.prefill_tps if srv else None,
            "decode_tps": srv.decode_tps if srv else None,
            "thinking_tokens": result.thinking_chunks,
            "answer_tokens": result.answer_chunks,
            "cold": int(result.cold),
            "peak_rss_bytes": window.peak_rss_bytes or None,
            "swap_delta_bytes": window.swap_delta_bytes,
            "mem_pressure": window.pressure_max,
            "energy_j": window.energy_j,
            "on_ac": int(power.on_ac),
            "ok": int(result.error is None),
            "error": result.error,
            "finish_reason": result.finish_reason,
            "prompt_json": json.dumps(
                [[m.role, m.content] for m in req.messages], ensure_ascii=False
            )[:MAX_STORED_CHARS],
            "output_text": result.text[:MAX_STORED_CHARS],
            "thinking_text": result.thinking[:MAX_STORED_CHARS],
            "options_json": json.dumps(opts.as_ollama()),
            "timings_json": json.dumps(
                {"inter_token_ms": [round(x, 3) for x in result.inter_token_ms[:5000]]}
            ),
            "raw_json": json.dumps(result.raw_done) if result.raw_done else None,
        }
        insert(self.conn, "requests", row)
        result.raw_done = result.raw_done  # keep for the caller
        self._last_id = row["id"]

    @property
    def last_request_id(self) -> str | None:
        return getattr(self, "_last_id", None)


def _think_label(think: bool | str | None) -> str:
    if think is None:
        return "unset"
    if think is False:
        return "off"
    if think is True:
        return "on"
    return str(think)
