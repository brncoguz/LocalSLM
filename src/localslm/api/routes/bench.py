"""Benchmark suites and runs.

A run holds the inference engine exclusively for its whole duration, so chat is
refused with a 409 while one is active. Progress streams over SSE; the run
itself executes in a background task so a dropped connection doesn't kill it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from localslm.api.deps import get_state
from localslm.api.sse import event, sse
from localslm.bench.runner import BenchRunner
from localslm.bench.spec import BenchSpec
from localslm.bench.stats import Interval, paired_bootstrap, rank_with_ties
from localslm.bench.suites import available, load
from localslm.db import new_id, query
from localslm.db import write as db_write
from localslm.inference.lock import LeaseBusy

router = APIRouter(tags=["bench"])

# Live runs, so the SSE endpoint can attach after the run has started.
_ACTIVE: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    models: list[str]
    suites: list[str]
    repeats: int = Field(default=1, ge=1, le=10)
    think_levels: list[str] = Field(default_factory=lambda: ["false"])
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
        repeats=body.repeats,
        warmups=1,
        thinking_capable=capable,
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
