"""Benchmark suites and runs.

A run holds the inference engine exclusively for its whole duration, so chat is
refused with a 409 while one is active. Progress streams over SSE; the run
itself executes in a background task so a dropped connection doesn't kill it.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from localslm.api.deps import get_state
from localslm.api.sse import event, sse
from localslm.bench.agreement import agreement, judge_run, pending_rows, verdicts
from localslm.bench.determinism import stability
from localslm.bench.graders import REGISTRY as GRADERS
from localslm.bench.judge import build_judge
from localslm.bench.runner import BenchRunner
from localslm.bench.spec import BenchSpec
from localslm.bench.stats import Interval, paired_bootstrap, rank_with_ties
from localslm.bench.suites import Suite, TaskItem, available, load, save
from localslm.db import new_id, query
from localslm.db import write as db_write
from localslm.host import power_label
from localslm.inference.lock import LeaseBusy
from localslm.inference.session import MeasuredSession

router = APIRouter(tags=["bench"])

# Live runs, so the SSE endpoint can attach after the run has started.
_ACTIVE: dict[str, dict[str, Any]] = {}

# The same, for judging passes. Kept separate from `_ACTIVE` because a judging
# pass is not a run — it produces no `runs` row and nothing to abort halfway
# that would leave partial results, since verdicts are written as they arrive.
_JUDGING: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    models: list[str]
    suites: list[str]
    repeats: int = Field(default=1, ge=1, le=10)
    think_levels: list[str] = Field(default_factory=lambda: ["false"])
    # More than one temperature turns the run into a determinism study: the
    # same items, the same seed, sampled differently.
    temperatures: list[float] = Field(default_factory=lambda: [0.0])
    label: str = ""


@router.get("/suites")
async def list_suites() -> list[dict]:
    out = []
    for name in available():
        suite = load(name)
        out.append(
            {
                "name": name,
                "items": len(suite),
                "verified": suite.verified,
                "tasks": sorted(suite.tasks),
                "hash": suite.content_hash(),
                "categories": sorted({i.category for i in suite.items}),
                "unanswerable": sum(1 for i in suite.items if i.unanswerable),
            }
        )
    return out


@router.get("/suites/{name}/items")
async def suite_items(name: str) -> dict:
    try:
        suite = load(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"name": name, "items": [i.to_dict() for i in suite.items]}


class FromSessionRequest(BaseModel):
    name: str = ""
    task: str = "contains"
    include_answers: bool = True


@router.post("/suites/from-session/{session_id}")
async def suite_from_session(session_id: str, body: FromSessionRequest) -> dict:
    """Turn a conversation into a reusable suite.

    This is the payoff for chat and benchmarks sharing one table: the prompts
    you actually type are more representative of your workload than any public
    test set, and they are already stored.

    Items land as `verified_by: "llm"` — the assistant's own reply becomes the
    expected answer, which is a starting point, not a ground truth. Only
    human-verified items should ever gate anything.
    """
    st = get_state()
    rows = query(
        st.conn,
        "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY idx",
        (session_id,),
    )
    if not rows:
        raise HTTPException(404, "no such session, or it has no messages")

    if body.task not in GRADERS:
        raise HTTPException(422, f"unknown grader {body.task!r}")

    name = re.sub(r"[^a-z0-9_-]+", "-", (body.name or f"session-{session_id[:8]}").lower())
    if name in available():
        raise HTTPException(
            409, {"error": "exists", "message": f"suite {name!r} already exists"}
        )

    items: list[TaskItem] = []
    pending: str | None = None
    for row in rows:
        if row["role"] == "user":
            pending = row["content"]
        elif pending:
            items.append(
                TaskItem(
                    id=f"{name}-{len(items) + 1}",
                    task=body.task,
                    prompt=pending,
                    expected=(row["content"] or "").strip()[:500]
                    if body.include_answers
                    else None,
                    category="from_chat",
                    verified_by="llm",
                    notes=f"captured from chat session {session_id}",
                )
            )
            pending = None

    if not items:
        raise HTTPException(422, "no complete question/answer pairs in that session")

    suite = Suite(name=name, items=items)
    path = save(suite)
    return {
        "name": name,
        "items": len(items),
        "path": str(path),
        "verified": 0,
        "hash": suite.content_hash(),
    }


@router.post("/bench/runs")
async def start_run(body: RunRequest) -> dict:
    st = get_state()
    if not body.models or not body.suites:
        raise HTTPException(422, "pick at least one model and one suite")

    installed = {m.ref.name for m in await st.runtime.list_models()}
    if missing := [m for m in body.models if m not in installed]:
        raise HTTPException(404, {"error": "model_missing", "models": missing})

    capable = {}
    for name in body.models:
        info = await st.runtime.show(name)
        capable[name] = info.capabilities.thinking

    levels: list[bool | str] = [
        False if raw.lower() in ("false", "off") else raw.lower()
        for raw in body.think_levels
    ]
    try:
        suites = [load(n) for n in body.suites]
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    spec = BenchSpec(
        models=body.models,
        suites=suites,
        think_levels=levels,
        temperatures=body.temperatures or [0.0],
        repeats=body.repeats,
        warmups=1,
        thinking_capable=capable,
        # Stamped rather than checked: nothing here can plug in a charger. The
        # point is that a run measured on battery is never silently compared
        # against one on AC. A source change mid-run is caught downstream by
        # `power_split`, off the per-request `on_ac` column.
        power_source=power_label(),
    )
    # Starting a run that will abort on its first unit wastes the user's time
    # and leaves a confusing 0/N row behind. Refuse while the machine is
    # already thrashing, and say what to do about it.
    sample = st.sampler.latest
    if sample and sample.pressure == "critical":
        raise HTTPException(
            409,
            {
                "error": "memory_pressure",
                "free_bytes": sample.free_bytes,
                "message": (
                    "The machine is under critical memory pressure, so timings "
                    "would be meaningless. Unload a model or close something first."
                ),
            },
        )

    run_id = new_id()
    try:
        await st.lease.acquire_exclusive("bench", run_id, timeout_s=5)
    except LeaseBusy as busy:
        raise HTTPException(
            409, {"error": "engine_busy", "holder": busy.holder, "run_id": busy.run_id}
        ) from busy

    runner = BenchRunner(st, spec, run_id, body.label or "ui")
    queue: asyncio.Queue = asyncio.Queue()
    _ACTIVE[run_id] = {"runner": runner, "queue": queue, "finished": False}

    async def execute() -> None:
        try:
            async for ev in runner.run():
                await queue.put(ev.as_dict())
        except Exception as exc:  # surfaced to the stream, never swallowed
            await queue.put({"kind": "aborted", "data": {"reason": str(exc)}})
        finally:
            _ACTIVE[run_id]["finished"] = True
            await queue.put(None)
            await st.lease.release_exclusive()

    # Detached so closing the browser tab doesn't abort a 40-minute run, but
    # the handle is kept: asyncio holds only a weak reference, and a collected
    # task would stop the benchmark silently part-way through.
    _ACTIVE[run_id]["task"] = asyncio.create_task(execute())
    return {
        "run_id": run_id,
        "total_requests": spec.total_requests(),
        "cells": len(spec.cells()),
    }


@router.get("/bench/runs/{run_id}/stream")
async def stream_run(run_id: str) -> object:
    active = _ACTIVE.get(run_id)
    if not active:
        raise HTTPException(404, "no such active run")

    async def gen():
        queue: asyncio.Queue = active["queue"]
        while True:
            item = await queue.get()
            if item is None:
                break
            yield event(item["kind"], item["data"])

    return sse(gen())


@router.post("/bench/runs/{run_id}/abort")
async def abort_run(run_id: str) -> dict:
    active = _ACTIVE.get(run_id)
    if not active:
        raise HTTPException(404, "no such active run")
    active["runner"].abort()
    return {"aborting": run_id}


@router.get("/bench/active")
async def active_run() -> dict | None:
    for run_id, entry in _ACTIVE.items():
        if not entry["finished"]:
            return {"run_id": run_id}
    return None


@router.get("/runs")
async def list_runs(limit: int = 50) -> list[dict]:
    st = get_state()
    rows = query(
        st.conn,
        "SELECT id, ts, finished_ts, suite, suite_hash, preset, label, status, "
        "planned_units, done_units, models_json, elapsed_s, metrics_json "
        "FROM runs ORDER BY ts DESC LIMIT ?",
        (limit,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["models"] = json.loads(r["models_json"] or "[]")
        cells = json.loads(r["metrics_json"] or "[]")
        d["best_score"] = max((c["score"] for c in cells), default=None)
        d["n_cells"] = len(cells)
        d.pop("models_json"), d.pop("metrics_json")
        out.append(d)
    return out


@router.get("/runs/{run_id}")
async def run_detail(run_id: str) -> dict:
    st = get_state()
    rows = query(st.conn, "SELECT * FROM runs WHERE id = ?", (run_id,))
    if not rows:
        raise HTTPException(404, "no such run")
    run = dict(rows[0])
    cells = json.loads(run.pop("metrics_json") or "[]")
    run["plan"] = json.loads(run.pop("suite_json") or "{}")
    run["models"] = json.loads(run.pop("models_json") or "[]")

    # Rank with ties flagged so the UI can grey out distinctions the data
    # doesn't support, rather than printing a confident ordering.
    intervals = {
        c["cell_key"]: Interval(c["score"], c["ci_lo"], c["ci_hi"], c["n"]) for c in cells
    }
    tied = {key: is_tied for key, _, is_tied in rank_with_ties(intervals)}
    ranked = sorted(cells, key=lambda x: -x["score"])
    for c in ranked:
        c["tied_with_previous"] = tied.get(c["cell_key"], False)
    run["cells"] = ranked
    return run


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Delete a run and everything it produced.

    Unlike a chat session — where the transcript is disposable but the
    measurement it produced is the valuable part — a benchmark run *is* its
    measurements. Leaving its `requests` rows behind would orphan them: they
    would keep skewing the Stats charts with no run left to trace them back to.
    So this removes the run, its graded results, its frozen aggregates and its
    raw requests together, and reports the counts so the UI can say what went.
    """
    st = get_state()
    rows = query(st.conn, "SELECT status FROM runs WHERE id = ?", (run_id,))
    if not rows:
        raise HTTPException(404, "no such run")

    active = _ACTIVE.get(run_id)
    if rows[0]["status"] == "running" and active and not active["finished"]:
        raise HTTPException(
            409,
            {
                "error": "run_active",
                "message": "That run is still going. Stop it first, then delete it.",
            },
        )

    counted = {
        table: query(
            st.conn, f"SELECT COUNT(*) n FROM {table} WHERE run_id = ?", (run_id,)
        )[0]["n"]
        for table in ("requests", "results", "cells")
    }
    with db_write(st.conn):
        for table in ("results", "cells", "samples", "requests"):
            st.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        st.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    _ACTIVE.pop(run_id, None)
    return {"deleted": run_id, **counted}


@router.get("/runs/{run_id}/thinking")
async def run_thinking(run_id: str) -> dict:
    """What reasoning cost, and whether it bought anything.

    The question the whole project is pointed at. Each think level becomes a
    point of (thinking tokens spent, accuracy), so the curve shows directly
    whether more reasoning is buying correctness or just latency.

    `ttfat` is reported beside `ttft` because for these models they differ by
    tens of seconds, and only the second describes what a user waits for.
    """
    st = get_state()
    rows = query(
        st.conn,
        """SELECT res.cell_key, res.model, res.score, res.item_id,
                  req.think, req.thinking_tokens, req.answer_tokens,
                  req.ttft_ms, req.ttfat_ms, req.total_ms, req.output_tokens
           FROM results res
           JOIN requests req ON req.id = res.request_id
           WHERE res.run_id = ? AND req.excluded = 0""",
        (run_id,),
    )
    if not rows:
        return {"run_id": run_id, "points": [], "note": "No graded results for this run."}

    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["cell_key"], []).append(row)

    # Compare on the items every cell actually completed.
    #
    # Long reasoning runs push this machine into swap, and those requests are
    # excluded — correctly — which leaves each cell with a different subset of
    # items. Averaging over whatever survived would compare think=low on the
    # items it happened to finish against think=off on all of them, and read
    # the difference as a property of reasoning rather than of which questions
    # got dropped.
    per_cell_items = {
        key: {r["item_id"] for r in entries} for key, entries in grouped.items()
    }
    shared = set.intersection(*per_cell_items.values()) if per_cell_items else set()
    all_items = set().union(*per_cell_items.values()) if per_cell_items else set()

    def avg(entries: list, key: str) -> float | None:
        vals = [e[key] for e in entries if e[key] is not None]
        return sum(vals) / len(vals) if vals else None

    points = []
    for cell_key, entries in grouped.items():
        paired = [e for e in entries if e["item_id"] in shared] or entries
        n = len(paired)
        thinking = [e["thinking_tokens"] or 0 for e in paired]
        answering = [e["answer_tokens"] or 0 for e in paired]
        overthought = sum(
            1 for t, a in zip(thinking, answering, strict=True) if a and t > 3 * a
        )
        points.append(
            {
                "cell_key": cell_key,
                "model": paired[0]["model"],
                "think": paired[0]["think"],
                "n": n,
                "n_attempted": len(entries),
                "accuracy": sum(e["score"] or 0 for e in paired) / n,
                "thinking_tokens": sum(thinking) / n,
                "answer_tokens": sum(answering) / n,
                "think_ratio": (
                    sum(thinking) / (sum(thinking) + sum(answering))
                    if sum(thinking) + sum(answering)
                    else 0.0
                ),
                "overthink_rate": overthought / n,
                "ttft_ms": avg(paired, "ttft_ms"),
                "ttfat_ms": avg(paired, "ttfat_ms"),
                "total_ms": avg(paired, "total_ms"),
            }
        )

    points.sort(key=lambda p: (p["model"], p["thinking_tokens"]))
    dropped = len(all_items) - len(shared)
    return {
        "run_id": run_id,
        "points": points,
        "paired": bool(shared),
        "n_shared_items": len(shared),
        "n_dropped_items": dropped,
        "note": (
            f"Scored on the {len(shared)} items every setting completed. "
            f"{dropped} more were attempted by some settings but excluded "
            f"(usually the machine swapping during long reasoning), and "
            f"including them would compare different question sets."
        )
        if dropped
        else None,
    }


class JudgeRequest(BaseModel):
    limit: int = Field(0, ge=0, le=2000)


@router.post("/runs/{run_id}/judge")
async def start_judging(run_id: str, body: JudgeRequest) -> dict:
    """Score a completed run's answers with the local judge.

    Detached and streamed rather than synchronous: a 9B judge takes a couple of
    seconds per item, so a few hundred items is minutes, and holding an HTTP
    request open for that would make a closed tab look like a failure.
    """
    st = get_state()
    if not query(st.conn, "SELECT id FROM runs WHERE id = ?", (run_id,)):
        raise HTTPException(404, f"no such run {run_id}")

    # Cheapest question first, and the only one that can end the request
    # happily. Demanding a pulled judge model before checking whether there is
    # anything to judge turns "already done" into an error about the wrong
    # thing — re-judging is how the pass resumes, so arriving here with nothing
    # pending is a normal outcome.
    pending = len(pending_rows(st.conn, run_id, body.limit))
    if not pending:
        return {
            "run_id": run_id,
            "pending": 0,
            "note": "Every item has already been judged.",
        }

    try:
        judge = build_judge(MeasuredSession(st.runtime, st.conn, st.sampler), st.cfg)
    except NotImplementedError as exc:
        raise HTTPException(
            501, {"error": "backend_unavailable", "message": str(exc)}
        ) from exc

    installed = {m.ref.name for m in await st.runtime.list_models()}
    if judge.name not in installed:
        raise HTTPException(
            404,
            {
                "error": "judge_missing",
                "model": judge.name,
                "message": (
                    f"The judge model {judge.name} is not installed. Pull it, or "
                    f"point judge.local_model at one you have."
                ),
            },
        )

    # Same refusal as a benchmark: a judge measured while the machine thrashes
    # produces verdicts, but its own latency and token counts are then fiction,
    # and those land in `requests` alongside everything else.
    sample = st.sampler.latest
    if sample and sample.pressure == "critical":
        raise HTTPException(
            409,
            {
                "error": "memory_pressure",
                "free_bytes": sample.free_bytes,
                "message": (
                    f"The machine is under critical memory pressure and the judge "
                    f"needs to load {judge.name}. Unload a model first."
                ),
            },
        )

    try:
        await st.lease.acquire_exclusive("judge", run_id, timeout_s=5)
    except LeaseBusy as busy:
        raise HTTPException(
            409, {"error": "engine_busy", "holder": busy.holder, "run_id": busy.run_id}
        ) from busy

    queue: asyncio.Queue = asyncio.Queue()
    _JUDGING[run_id] = {"queue": queue, "finished": False}

    def on_verdict(row: dict, verdict) -> None:
        queue.put_nowait(
            {
                "kind": "verdict",
                "data": {
                    "item_id": row["item_id"],
                    "model": row["model"],
                    "task": row["task"],
                    "parse_ok": verdict.parse_ok,
                    "rating": verdict.rating,
                    "agreed": (
                        None
                        if verdict.passed is None
                        else bool(verdict.passed) == bool(row["passed"])
                    ),
                },
            }
        )

    async def execute() -> None:
        try:
            outcome = await judge_run(
                st.conn, judge, run_id, limit=body.limit, on_verdict=on_verdict
            )
            queue.put_nowait({"kind": "done", "data": outcome})
        except Exception as exc:  # surfaced on the stream, never swallowed
            queue.put_nowait({"kind": "failed", "data": {"reason": str(exc)}})
        finally:
            _JUDGING[run_id]["finished"] = True
            queue.put_nowait(None)
            await st.lease.release_exclusive()

    # Kept rather than fire-and-forget: asyncio holds only a weak reference, and
    # a collected task would stop judging silently half way through.
    _JUDGING[run_id]["task"] = asyncio.create_task(execute())
    return {"run_id": run_id, "pending": pending, "judge_model": judge.name}


@router.get("/runs/{run_id}/judge/stream")
async def stream_judging(run_id: str) -> object:
    active = _JUDGING.get(run_id)
    if not active:
        raise HTTPException(404, "no judging in progress for this run")

    async def gen():
        queue: asyncio.Queue = active["queue"]
        while True:
            item = await queue.get()
            if item is None:
                break
            yield event(item["kind"], item["data"])

    return sse(gen())


@router.get("/runs/{run_id}/judge-verdicts")
async def run_judge_verdicts(
    run_id: str,
    only: str = Query("all", pattern="^(all|disagreements|parse_failures)$"),
    limit: int = Query(60, ge=1, le=500),
) -> dict:
    """Individual judgements with the evidence behind them.

    The agreement report says how often the judge was right; this says what it
    was doing. A judge that reasons and one that pattern-matches score the same
    until someone reads the verdicts, which is what makes this the actual
    evaluation of the judge.
    """
    return verdicts(get_state().conn, run_id, only=only, limit=limit)


@router.get("/runs/{run_id}/judge-agreement")
async def run_judge_agreement(run_id: str) -> dict:
    """How far the judge can be trusted on this run.

    Points the judge's verdicts at the items a deterministic grader already
    scored, so there is a number for "should I believe this" before any
    open-ended score built on the same judge gets quoted. Read `kappa`, not
    `agreement_rate` — see `bench/agreement.py`.
    """
    return agreement(get_state().conn, run_id)


@router.get("/runs/{run_id}/determinism")
async def run_determinism(run_id: str) -> dict:
    """How stable each cell's output was across repeated samples.

    Needs repeats > 1 to say anything. Grouped per cell so a temperature axis
    reads directly: the same items at t=0 and t=0.7 become two rows whose
    duplicate rates can be compared.
    """
    st = get_state()
    rows = query(
        st.conn,
        """SELECT res.cell_key, res.item_id, res.model, res.score,
                  req.output_text, req.temperature, req.think
           FROM results res
           JOIN requests req ON req.id = res.request_id
           WHERE res.run_id = ? AND req.excluded = 0
           ORDER BY res.cell_key, res.item_id""",
        (run_id,),
    )

    per_cell: dict[str, dict[str, Any]] = {}
    for row in rows:
        cell = per_cell.setdefault(
            row["cell_key"],
            {
                "cell_key": row["cell_key"],
                "model": row["model"],
                "temperature": row["temperature"],
                "think": row["think"],
                "items": {},
            },
        )
        cell["items"].setdefault(row["item_id"], {"texts": [], "scores": []})
        cell["items"][row["item_id"]]["texts"].append(row["output_text"] or "")
        cell["items"][row["item_id"]]["scores"].append(row["score"])

    out = []
    for cell in per_cell.values():
        per_item = {
            item_id: stability(v["texts"], v["scores"]).as_dict()
            for item_id, v in cell["items"].items()
            if len(v["texts"]) > 1
        }
        if not per_item:
            continue
        rates = [m["exact_duplicate_rate"] for m in per_item.values()]
        out.append(
            {
                "cell_key": cell["cell_key"],
                "model": cell["model"],
                "temperature": cell["temperature"],
                "think": cell["think"],
                "n_items": len(per_item),
                "samples_per_item": max(len(v["texts"]) for v in cell["items"].values()),
                "mean_duplicate_rate": sum(rates) / len(rates),
                "fully_deterministic_items": sum(
                    1 for m in per_item.values() if m["fully_deterministic"]
                ),
                "items": per_item,
            }
        )

    return {
        "run_id": run_id,
        "cells": sorted(out, key=lambda c: -c["mean_duplicate_rate"]),
        # Says plainly why the answer is empty, rather than rendering a blank panel.
        "note": (
            None
            if out
            else "Determinism needs repeats > 1 — this run sampled each item once."
        ),
    }


@router.get("/runs/{run_id}/results")
async def run_results(run_id: str, task: str = "", failed_only: bool = False) -> list[dict]:
    st = get_state()
    sql = "SELECT * FROM results WHERE run_id = ?"
    args: list[Any] = [run_id]
    if task:
        sql += " AND task = ?"
        args.append(task)
    if failed_only:
        sql += " AND passed = 0"
    sql += " ORDER BY task, item_id LIMIT 500"
    return [dict(r) for r in query(st.conn, sql, args)]


@router.get("/runs/{run_id}/compare")
async def compare_cells(run_id: str, a: str, b: str, task: str = "") -> dict:
    """Paired comparison of two cells on the items they both attempted.

    Paired because the same items were run through both, which removes the
    variance from some questions simply being harder and roughly halves the
    interval width at these sample sizes.
    """
    st = get_state()
    sql = "SELECT cell_key, item_id, score FROM results WHERE run_id = ? AND cell_key IN (?, ?)"
    args: list[Any] = [run_id, a, b]
    if task:
        sql += " AND task = ?"
        args.append(task)
    rows = query(st.conn, sql, args)
    a_scores = {r["item_id"]: r["score"] for r in rows if r["cell_key"] == a}
    b_scores = {r["item_id"]: r["score"] for r in rows if r["cell_key"] == b}
    if not a_scores or not b_scores:
        raise HTTPException(404, "no overlapping results for those cells")
    return paired_bootstrap(a_scores, b_scores, a, b).as_dict()
