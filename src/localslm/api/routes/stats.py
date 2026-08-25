"""Aggregations for the charts.

One rule runs through this module: **`ok = 1 AND excluded = 0` lives in the SQL,
not in the component.** Warmups, swapped windows and pressure-poisoned rows are
kept in the table so they stay visible in the trace inspector, and a chart that
had to remember to filter them would eventually forget.

Percentiles are computed in Python rather than SQL because SQLite has no
percentile function, and at this scale (thousands of rows) the difference is
not worth a window-function query nobody can read.
"""

from __future__ import annotations

import statistics
from sqlite3 import Connection, Row
from typing import Any

from fastapi import APIRouter, Query

from localslm.api.deps import get_state
from localslm.db import query

router = APIRouter(prefix="/stats", tags=["stats"])

# Every chart reads through this. Changing it changes every chart at once,
# which is the point.
INCLUDED = "ok = 1 AND excluded = 0"

POINT_COLUMNS = """
    id, ts, model, kind, think, cold,
    decode_tps, prefill_tps, ttft_ms, ttfat_ms, total_ms,
    prompt_tokens, output_tokens, thinking_tokens, answer_tokens,
    peak_rss_bytes, srv_load_ms, itl_stddev_ms
"""


def _spread(values: list[float]) -> dict[str, float] | None:
    """mean/p50/p95/min/max, or None when there is nothing to describe."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    ordered = sorted(clean)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def summarise(rows: list[Row]) -> dict[str, Any]:
    """Per-model rollup. `n` travels with every figure so the UI can refuse to
    draw a trend through two points."""
    cold = [r for r in rows if r["cold"]]
    warm = [r for r in rows if not r["cold"]]
    thinking = sum(r["thinking_tokens"] or 0 for r in rows)
    answering = sum(r["answer_tokens"] or 0 for r in rows)
    generated = thinking + answering
    return {
        "n": len(rows),
        "decode_tps": _spread([r["decode_tps"] for r in rows]),
        "prefill_tps": _spread([r["prefill_tps"] for r in rows]),
        "ttfat_ms": _spread([r["ttfat_ms"] for r in rows]),
        "ttft_ms": _spread([r["ttft_ms"] for r in rows]),
        "total_ms": _spread([r["total_ms"] for r in rows]),
        "peak_rss_bytes": _spread([r["peak_rss_bytes"] for r in rows]),
        "cold": {
            "n": len(cold),
            "load_ms": _spread([r["srv_load_ms"] for r in cold]),
        },
        "warm": {
            "n": len(warm),
            "load_ms": _spread([r["srv_load_ms"] for r in warm]),
        },
        "thinking": {
            "thinking_tokens": thinking,
            "answer_tokens": answering,
            "ratio": thinking / generated if generated else None,
        },
        "first_ts": min((r["ts"] for r in rows), default=None),
        "last_ts": max((r["ts"] for r in rows), default=None),
    }


def model_rows(conn: Connection, model: str, limit: int = 500) -> list[Row]:
    return query(
        conn,
        f"SELECT {POINT_COLUMNS} FROM requests "
        f"WHERE model = ? AND {INCLUDED} ORDER BY ts DESC LIMIT ?",
        (model, limit),
    )


@router.get("/timeline")
async def timeline(limit: int = Query(1000, le=5000)) -> dict:
    """Every included measurement, newest first — the multi-model chart feed."""
    st = get_state()
    rows = query(
        st.conn,
        f"SELECT {POINT_COLUMNS} FROM requests WHERE {INCLUDED} ORDER BY ts DESC LIMIT ?",
        (limit,),
    )
    points = [dict(r) for r in reversed(rows)]
    return {
        "points": points,
        "models": sorted({p["model"] for p in points}),
    }


@router.get("/disagreements")
async def disagreements(run_id: str = "", limit: int = 40) -> dict:
    """Items that discriminate between models, and items nobody got.

    Two different signals, both more useful than an average:

    * **Disagreements** are where the models actually differ. On a suite where
      everything scores 0.9, these few items carry all the information, and
      they are the ones worth hand-verifying first.
    * **Universal failures** usually mean the item is broken rather than the
      models are — a wrong gold label, or an ambiguous prompt. Surfacing them
      turns the benchmark into an aid for authoring the benchmark.
    """
    st = get_state()
    where = "WHERE run_id = ?" if run_id else ""
    args = (run_id,) if run_id else ()
    rows = query(
        st.conn,
        f"""SELECT item_id, task, model, score, passed, pred_label, gold_label,
                   detail_json, request_id, run_id
            FROM results {where}""",
        args,
    )

    by_item: dict[str, list[Any]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)

    split, unanimous_fail = [], []
    for item_id, entries in by_item.items():
        # One model's opinion is not a disagreement.
        if len(entries) < 2:
            continue
        scores = [e["score"] or 0.0 for e in entries]
        spread = max(scores) - min(scores)
        payload = {
            "item_id": item_id,
            "task": entries[0]["task"],
            "run_id": entries[0]["run_id"],
            "spread": spread,
            "models": [
                {
                    "model": e["model"],
                    "score": e["score"],
                    "passed": bool(e["passed"]),
                    "pred": e["pred_label"],
                    "gold": e["gold_label"],
                    "request_id": e["request_id"],
                }
                for e in sorted(entries, key=lambda x: -(x["score"] or 0))
            ],
        }
        if spread > 0:
            split.append(payload)
        elif max(scores) == 0.0:
            unanimous_fail.append(payload)

    split.sort(key=lambda d: -d["spread"])
    return {
        "disagreements": split[:limit],
        "all_failed": unanimous_fail[:limit],
        "n_items": len(by_item),
        "n_disagreements": len(split),
        "n_all_failed": len(unanimous_fail),
    }


@router.get("/summary")
async def summary() -> dict:
    """Per-model rollups plus the exclusion ledger.

    The ledger is surfaced rather than hidden: "8 of 31 measurements excluded"
    is information about the machine, and burying it would be exactly the kind
    of quiet dishonesty this project exists to avoid.
    """
    st = get_state()
    included = query(
        st.conn, f"SELECT {POINT_COLUMNS} FROM requests WHERE {INCLUDED} ORDER BY ts"
    )

    by_model: dict[str, list[Row]] = {}
    for row in included:
        by_model.setdefault(row["model"], []).append(row)

    excluded = {
        r["exclude_reason"] or "unknown": r["n"]
        for r in query(
            st.conn,
            "SELECT exclude_reason, COUNT(*) n FROM requests "
            "WHERE excluded = 1 GROUP BY exclude_reason",
        )
    }
    failed = query(st.conn, "SELECT COUNT(*) n FROM requests WHERE ok = 0")[0]["n"]
    total = query(st.conn, "SELECT COUNT(*) n FROM requests")[0]["n"]

    return {
        "models": [
            {"model": name, **summarise(rows)} for name, rows in sorted(by_model.items())
        ],
        "excluded": excluded,
        "totals": {"requests": total, "included": len(included), "failed": failed},
    }
