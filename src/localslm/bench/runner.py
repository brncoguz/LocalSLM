"""Executing a benchmark run.

The runner owns three things the rest of the system deliberately doesn't:
exclusive access to the inference engine, the retry-once-then-give-up policy for
structured output, and the decision to stop early when the machine is thrashing.

Everything it produces lands in the same `requests` table that chat writes to,
plus a graded row per item in `results` and a frozen per-cell aggregate in
`cells`. Freezing the aggregates at completion means the leaderboard and the
report quote identical numbers instead of each re-averaging live.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from localslm.bench.events import BenchEvent
from localslm.bench.graders import GradeContext
from localslm.bench.graders import get as get_grader
from localslm.bench.spec import BenchSpec, Unit, order_units, summarise_plan
from localslm.bench.stats import bootstrap_ci
from localslm.bench.suites import TaskItem
from localslm.db import insert, new_id
from localslm.db import write as db_write
from localslm.inference.session import MeasuredSession, Provenance
from localslm.runtimes.base import DecodeOptions, GenRequest, GenResult, Message, ModelRef

NUM_PREDICT_DIRECT = 2048
NUM_PREDICT_THINKING = 8192

# Pinned rather than left to Ollama's default, for two reasons. The default
# varies by version and by OLLAMA_CONTEXT_LENGTH, so an unset value makes runs
# incomparable across machines without recording why. And context length is the
# single biggest driver of resident memory — a 2 GB model at its advertised
# 131k context needs ~18 GB — so leaving it unset is also how a benchmark ends
# up swapping.
NUM_CTX_DIRECT = 4096
NUM_CTX_THINKING = 12288

RETRY_PROMPT = (
    "That response could not be parsed as valid JSON matching the required schema.\n"
    "Error: {error}\n"
    "Reply with only the JSON object, no prose and no code fences."
)


def _brief(expected: Any) -> str:
    """The expected answer as one short string, whatever shape it was."""
    if expected is None:
        return ""
    if isinstance(expected, list):
        return ", ".join(str(e) for e in expected)[:200]
    return str(expected)[:200]


class BenchAborted(RuntimeError):
    pass


class BenchRunner:
    def __init__(self, state: Any, spec: BenchSpec, run_id: str, label: str = "") -> None:
        self.state = state
        self.spec = spec
        self.run_id = run_id
        self.label = label
        self._abort = False
        self._graded: dict[str, list] = {}
        self._items: dict[str, list[TaskItem]] = {}

    def abort(self) -> None:
        self._abort = True

    # -- request construction ---------------------------------------------

    def _build(self, unit: Unit, info: Any) -> GenRequest:
        axes = unit.cell.axes_dict
        thinking = axes.get("think", False) not in (False, "off", None)
        item = unit.item
        messages = []
        if item.context:
            messages.append(
                Message(
                    "user",
                    f"{item.context}\n\n{item.prompt}",
                )
            )
        else:
            messages.append(Message("user", item.prompt))
        return GenRequest(
            model=ModelRef(
                runtime="ollama",
                name=unit.cell.model,
                quant=info.ref.quant if info else None,
                params_b=info.ref.params_b if info else None,
            ),
            messages=tuple(messages),
            think=axes.get("think", False),
            schema=item.output_schema if item.task == "json_schema" else None,
            options=DecodeOptions(
                temperature=float(axes.get("temperature", 0.0)),
                seed=unit.seed,
                # A reasoning model spends most of its budget before it starts
                # answering. A cap tuned for direct replies truncates it
                # mid-thought, so it scores zero for running out of room rather
                # than for being wrong — which measures the cap, not the model.
                num_predict=NUM_PREDICT_THINKING if thinking else NUM_PREDICT_DIRECT,
                num_ctx=NUM_CTX_THINKING if thinking else NUM_CTX_DIRECT,
            ),
            # Held long enough that the next unit in the same block is warm,
            # which is what makes model-blocked ordering worth doing.
            keep_alive="10m",
        )

    # -- execution ---------------------------------------------------------

    async def run(self) -> AsyncIterator[BenchEvent]:
        st = self.state
        # The pressure latch is deliberately sticky *within* a run — a run that
        # briefly thrashed is suspect throughout. It must not outlive the run,
        # though: left set, every later run in the same process aborts at unit 0.
        st.sampler.reset_abort()
        units = order_units(self.spec, self.spec.seed)
        plan = summarise_plan(self.spec, units)

        started = time.time()
        insert(
            st.conn,
            "runs",
            {
                "id": self.run_id,
                "ts": started,
                "suite": ",".join(s.name for s in self.spec.suites),
                "suite_hash": self.spec.content_hash(),
                "suite_json": json.dumps(plan),
                "preset": self.label or "custom",
                "label": self.label,
                "status": "running",
                "run_seed": self.spec.seed,
                "planned_units": len(units),
                "done_units": 0,
                "models_json": json.dumps(self.spec.models),
                "env_hash": st.env_hash,
                "config_hash": st.cfg.fingerprint(),
            },
        )
        yield BenchEvent("plan", {"run_id": self.run_id, **plan})

        session = MeasuredSession(st.runtime, st.conn, st.sampler)
        info_cache: dict[str, Any] = {}
        done = 0
        aborted_reason: str | None = None

        for unit in units:
            if self._abort:
                aborted_reason = "cancelled"
                break
            # Latched by the sampler the moment pressure hits critical. Better
            # to stop than to publish a run measured on a thrashing machine.
            if st.sampler.abort_latched:
                aborted_reason = "memory_pressure"
                break

            if unit.cell.model not in info_cache:
                try:
                    info_cache[unit.cell.model] = await st.runtime.show(unit.cell.model)
                except Exception as exc:
                    yield BenchEvent(
                        "warning",
                        {"message": f"{unit.cell.model} unavailable: {exc}", "fatal": True},
                    )
                    aborted_reason = "model_missing"
                    break

            req = self._build(unit, info_cache[unit.cell.model])
            prov = Provenance(
                kind="bench",
                run_id=self.run_id,
                unit_id=unit.id,
                pass_idx=unit.pass_idx,
                rep=max(unit.rep, 0),
                cell_key=unit.cell.key,
                task=unit.item.task,
                item_id=unit.item.id,
                excluded=unit.is_warmup,
                exclude_reason="warmup" if unit.is_warmup else None,
            )

            result = await session.generate(req, prov)
            request_id = session.last_request_id
            retry_used = False

            # Retry-once for structured output: catch the invalid response,
            # reprompt with the parse error, then accept whatever comes back.
            if unit.item.task == "json_schema" and not unit.is_warmup:
                grade_try = get_grader("json_schema").grade(
                    unit.item, result, GradeContext()
                )
                if not grade_try.metrics.get("schema_valid"):
                    retry_used = True
                    error = grade_try.detail.get("parse_error") or grade_try.detail.get(
                        "schema_errors"
                    )
                    retry_req = GenRequest(
                        model=req.model,
                        messages=(
                            *req.messages,
                            Message("assistant", result.text[:2000]),
                            Message("user", RETRY_PROMPT.format(error=error)),
                        ),
                        schema=req.schema,
                        think=req.think,
                        options=req.options,
                        keep_alive=req.keep_alive,
                    )
                    result = await session.generate(
                        retry_req,
                        Provenance(**{**prov.__dict__, "parent_id": request_id}),
                    )
                    request_id = session.last_request_id

            if not unit.is_warmup:
                grade = get_grader(unit.item.task).grade(
                    unit.item, result, GradeContext(retry_used=retry_used)
                )
                # Ran out of budget before producing an answer: the score is
                # real, but the reason is the cap, so record it separately.
                if result.ttfat_ms is None and result.thinking_chunks:
                    grade.metrics["truncated_while_thinking"] = 1.0
                    grade.detail["truncated"] = True
                excluded = self._record(unit, grade, request_id, result)
                done += 1
                yield BenchEvent(
                    "unit",
                    {
                        "done": done,
                        "total": plan["measured_units"],
                        "cell": unit.cell.label(),
                        "model": unit.cell.model,
                        "suite": unit.suite,
                        "item": unit.item.id,
                        "score": grade.score,
                        "passed": grade.passed,
                        "decode_tps": result.server.decode_tps if result.server else None,
                        "ttfat_ms": result.ttfat_ms,
                        "pass_idx": unit.pass_idx,
                        # The actual work, so a run is watchable rather than a
                        # counter. Truncated because a long run streams
                        # hundreds of these and nobody reads past a few lines.
                        "prompt": unit.item.prompt[:400],
                        "answer": result.text[:400],
                        "expected": _brief(unit.item.expected),
                        "thinking_tokens": result.thinking_chunks,
                        # A measurement thrown out for swapping still happened,
                        # and hiding it would make the live view disagree with
                        # the totals it is counting towards.
                        "excluded": excluded,
                        "elapsed_s": round(time.time() - started, 1),
                    },
                )

        cells = self._finalise(started, done, len(units), aborted_reason)
        for cell_summary in cells:
            yield BenchEvent("cell", cell_summary)

        if aborted_reason:
            yield BenchEvent(
                "aborted", {"run_id": self.run_id, "reason": aborted_reason, "done": done}
            )
        else:
            yield BenchEvent(
                "done",
                {
                    "run_id": self.run_id,
                    "done": done,
                    "elapsed_s": round(time.time() - started, 1),
                    "cells": cells,
                },
            )

    # -- persistence -------------------------------------------------------

    def _record(
        self, unit: Unit, grade: Any, request_id: str | None, result: GenResult
    ) -> bool:
        """Persist one graded unit. Returns whether its measurement survived."""
        # Whether the measurement survived is decided in `_persist`, after this
        # unit has already been generated and graded — a window that swapped or
        # hit critical pressure gets flagged there. Read it back off the row
        # rather than tracking it in memory, so `results`, `cells` and every
        # endpoint are all applying one stored fact instead of three copies of
        # it that can drift apart.
        excluded = self._was_excluded(request_id)
        insert(
            self.state.conn,
            "results",
            {
                "id": new_id(),
                "run_id": self.run_id,
                "request_id": request_id or "",
                "cell_key": unit.cell.key,
                "model": unit.cell.model,
                "item_id": unit.item.id,
                "task": unit.item.task,
                "category": unit.item.category,
                "lang": unit.item.lang,
                "difficulty": unit.item.difficulty,
                "rep": max(unit.rep, 0),
                "excluded": int(excluded),
                "grader": unit.item.task,
                "judged": int(grade.judged),
                "score": grade.score,
                "passed": int(grade.passed),
                "pred_label": grade.pred_label,
                "gold_label": grade.gold_label,
                "metrics_json": json.dumps(grade.metrics),
                "detail_json": json.dumps(grade.detail, default=str)[:20000],
            },
        )
        self._graded.setdefault(unit.cell.key, []).append((unit, grade, excluded))
        return excluded

    def _was_excluded(self, request_id: str | None) -> bool:
        """Did the measurement behind this grade survive?

        A missing row is treated as excluded: it means the request was never
        persisted, so there is no evidence the machine was in a fit state.
        """
        if not request_id:
            return True
        row = self.state.conn.execute(
            "SELECT excluded FROM requests WHERE id = ?", (request_id,)
        ).fetchone()
        return True if row is None else bool(row["excluded"])

    def _finalise(
        self, started: float, done: int, planned: int, aborted: str | None
    ) -> list[dict[str, Any]]:
        """Freeze per-cell aggregates so every consumer quotes the same numbers."""
        conn = self.state.conn
        summaries = []

        for cell_key, entries in self._graded.items():
            cell = entries[0][0].cell
            # The frozen aggregate has to describe the same rows the charts and
            # the analysis endpoints will, and those all join `excluded = 0`.
            # Freezing over everything graded instead would publish a
            # leaderboard number that cannot be reproduced in the inspector —
            # and would do it worst on the long reasoning runs that swap most,
            # which is where the project's headline claims come from.
            kept = [(u, g) for u, g, excluded in entries if not excluded]
            n_excluded = len(entries) - len(kept)

            if not kept:
                # Every measurement in this cell was thrown out. The cell is
                # still recorded, because "attempted and produced nothing
                # usable" is a result — dropping the row would make it look
                # like the cell was never run, and a 0.0 would read as the
                # model scoring zero.
                insert(
                    conn,
                    "cells",
                    {
                        "run_id": self.run_id,
                        "cell_key": cell_key,
                        "model": cell.model,
                        "axes_json": json.dumps(cell.axes_dict),
                        "n": 0,
                        "n_excluded": n_excluded,
                        "n_failed": 0,
                        "metrics_json": "{}",
                    },
                )
                summaries.append(
                    {
                        "cell_key": cell_key,
                        "label": cell.label(),
                        "model": cell.model,
                        "axes": cell.axes_dict,
                        "n": 0,
                        "n_excluded": n_excluded,
                        "score": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "pass_rate": None,
                        "tasks": {},
                    }
                )
                continue

            units = [u for u, _ in kept]
            grades = [g for _, g in kept]
            scores = [g.score for g in grades]
            ci = bootstrap_ci(scores)

            per_task: dict[str, dict[str, float]] = {}
            for task in {u.item.task for u in units}:
                pairs = [(u, g) for u, g in kept if u.item.task == task]
                per_task[task] = get_grader(task).aggregate(
                    [g for _, g in pairs], [u.item for u, _ in pairs]
                )

            row = {
                "run_id": self.run_id,
                "cell_key": cell_key,
                "model": cell.model,
                "axes_json": json.dumps(cell.axes_dict),
                "n": len(grades),
                "n_excluded": n_excluded,
                "n_failed": sum(1 for g in grades if not g.passed),
                "score_mean": ci.mean,
                "score_ci_lo": ci.lo,
                "score_ci_hi": ci.hi,
                "pass_rate": sum(1 for g in grades if g.passed) / len(grades),
                "metrics_json": json.dumps(per_task),
            }
            insert(conn, "cells", row)
            summaries.append(
                {
                    "cell_key": cell_key,
                    "label": cell.label(),
                    "model": cell.model,
                    "axes": cell.axes_dict,
                    "n": len(grades),
                    "n_excluded": n_excluded,
                    "score": ci.mean,
                    "ci_lo": ci.lo,
                    "ci_hi": ci.hi,
                    "pass_rate": row["pass_rate"],
                    "tasks": per_task,
                }
            )

        with db_write(conn):
            conn.execute(
                "UPDATE runs SET status = ?, finished_ts = ?, done_units = ?, "
                "abort_reason = ?, elapsed_s = ?, metrics_json = ? WHERE id = ?",
                (
                    "aborted" if aborted else "done",
                    time.time(),
                    done,
                    aborted,
                    round(time.time() - started, 1),
                    json.dumps(summaries),
                    self.run_id,
                ),
            )
        return summaries
