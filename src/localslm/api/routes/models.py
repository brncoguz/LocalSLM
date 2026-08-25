"""Model management.

The catalog is opinionated on purpose. The three models installed here are all
Qwen-architecture Q4 thinking models, which makes a "model comparison" between
them mostly a size comparison — so each catalog entry states what contrast it
buys, not just how big it is.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from localslm.api.deps import get_state
from localslm.api.routes.stats import model_rows, summarise
from localslm.api.sse import event, sse
from localslm.db import query
from localslm.host import disk_free_bytes

router = APIRouter(prefix="/models", tags=["models"])

# Never let a pull take the machine below this. macOS gets unhappy long
# before zero, and a benchmark that swaps produces numbers worth nothing.
DISK_GUARD_BYTES = 10_000_000_000

# Sizes are shown from the registry at pull time; these are for planning only.
CATALOG: list[dict[str, Any]] = [
    {
        "name": "llama3.2:3b",
        "size_gb": 2.0,
        "family": "llama",
        "why": "Non-thinking, different family and tokenizer. The cheapest high-contrast baseline.",
    },
    {
        "name": "phi4-mini",
        "size_gb": 2.5,
        "family": "phi",
        "why": "Trained hard on instruction following and structured output — the natural challenger on JSON tasks.",
    },
    {
        "name": "gemma3:4b",
        "size_gb": 3.3,
        "family": "gemma",
        "why": "Third architecture, strong multilingual. Makes the Turkish subset a real comparison.",
    },
    {
        "name": "qwen3.5:4b-q8_0",
        "size_gb": 5.3,
        "family": "qwen35",
        "why": "Quantization ladder rung 2 — identical weights, more precision.",
    },
    {
        "name": "qwen3.5:4b-bf16",
        "size_gb": 9.3,
        "family": "qwen35",
        "why": "Unquantized reference. The anchor that attributes quality loss to quantization rather than model choice.",
    },
    {
        "name": "qwen3.5:4b-mlx",
        "size_gb": 4.0,
        "family": "qwen35",
        "why": "MLX-format weights — same model, Apple-native runtime. Free runtime comparison.",
    },
]


@router.get("")
async def list_models() -> dict:
    st = get_state()
    installed = await st.runtime.list_models()
    loaded = {m.name: m for m in await st.runtime.loaded()}

    # Headline numbers per model, so a card is a scorecard rather than a label.
    stats = {
        r["model"]: dict(r)
        for r in query(
            st.conn,
            """
            SELECT model,
                   COUNT(*)            AS n,
                   AVG(decode_tps)     AS decode_tps,
                   AVG(ttfat_ms)       AS ttfat_ms,
                   MAX(peak_rss_bytes) AS peak_rss_bytes,
                   MAX(model_vram_bytes) AS model_vram_bytes,
                   MIN(on_ac)          AS on_ac_min,
                   MAX(on_ac)          AS on_ac_max
            FROM requests
            WHERE ok = 1 AND excluded = 0 AND decode_tps IS NOT NULL
            GROUP BY model
            """,
        )
    }

    cards = []
    for info in installed:
        live = loaded.get(info.ref.name)
        # `/api/tags` does not carry capabilities, so this needs a `show` per
        # model; the runtime caches them. Worth the calls: without it the bench
        # form cannot tell that a non-thinking model collapses four think
        # levels into one cell, and quotes an estimate four times too large.
        caps = await st.runtime.capabilities(info.ref.name)
        cards.append(
            {
                "name": info.ref.name,
                "digest": info.ref.digest,
                "family": info.family,
                "quant": info.ref.quant,
                "params_b": info.ref.params_b,
                "size_bytes": info.size_bytes,
                "state": "warm" if live else "cold",
                "thinking": caps.thinking,
                "vram_bytes": live.size_vram_bytes if live else None,
                "expires_at": live.expires_at if live else None,
                "headline": stats.get(info.ref.name),
            }
        )
    cards.sort(key=lambda c: c["size_bytes"])

    free = disk_free_bytes()
    installed_names = {c["name"] for c in cards}
    return {
        "installed": cards,
        "disk": {"free_bytes": free},
        "catalog": [
            c
            | {"installed": c["name"] in installed_names, "fits": c["size_gb"] * 1e9 < free}
            for c in CATALOG
        ],
    }


@router.get("/{name:path}/detail")
async def model_detail(name: str) -> dict:
    st = get_state()
    try:
        info = await st.runtime.show(name)
    except Exception as exc:
        raise HTTPException(404, f"model {name!r} not found: {exc}") from exc
    recent = query(
        st.conn,
        "SELECT id, ts, kind, think, ttft_ms, ttfat_ms, decode_tps, output_tokens "
        "FROM requests WHERE model = ? ORDER BY ts DESC LIMIT 20",
        (name,),
    )
    return {
        "name": name,
        "arch": info.arch,
        "family": info.family,
        "quant": info.ref.quant,
        "params_b": info.ref.params_b,
        "ctx_max": info.ref.ctx_max,
        "capabilities": info.capabilities.__dict__,
        "details": info.details,
        "recent": [dict(r) for r in recent],
    }


class PullRequest(BaseModel):
    name: str


def aggregate_layers(layers: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Sum per-layer byte counts into overall (completed, total).

    Ollama reports progress one layer at a time. Passing that through directly
    makes the bar restart on every layer, so we keep the whole map and stream
    the sum — which is the number a human is actually asking for.
    """
    return (
        sum(c for c, _ in layers.values()),
        sum(t for _, t in layers.values()),
    )


def humanise_pull_error(message: str) -> str:
    """Ollama's registry errors are written for its own CLI."""
    low = message.lower()
    if "file does not exist" in low or "manifest" in low:
        return "No such model on the Ollama registry — check the name and tag."
    if "connection" in low or "dial tcp" in low or "no such host" in low:
        return "Couldn't reach the Ollama registry. Check your network."
    if "no space" in low:
        return "Ran out of disk while writing the model."
    return message


@router.get("/{name:path}/history")
async def model_history(name: str, limit: int = 100) -> dict:
    """This model's measured requests over time, plus its rollup."""
    st = get_state()
    rows = model_rows(st.conn, name, limit)
    return {
        "model": name,
        "points": [dict(r) for r in reversed(rows)],
        "summary": summarise(rows),
    }


@router.post("/pull")
async def pull(body: PullRequest) -> object:
    """Pull a model, streaming aggregate progress.

    Ollama reports progress per layer, so a naive passthrough makes the bar
    jump backwards each time a new layer starts. We keep a digest -> bytes map
    and stream the sum, which is the number a human actually wants.
    """
    st = get_state()
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "model name is required")

    # A pull saturates disk and CPU. During a benchmark that would quietly
    # inflate every latency in the run, so it waits.
    lease = st.lease.status()
    if lease.busy and lease.exclusive:
        raise HTTPException(409, {"error": "benchmark_running", "run_id": lease.run_id})

    entry = next((c for c in CATALOG if c["name"] == name), None)
    free = disk_free_bytes()
    # Known size lets us refuse before starting; unknown models are guarded
    # mid-flight instead (below), since Ollama only reveals the size as it goes.
    if entry:
        needed = int(entry["size_gb"] * 1e9)
        if free - needed < DISK_GUARD_BYTES:
            raise HTTPException(
                409,
                {
                    "error": "disk_guard",
                    "free_bytes": free,
                    "needed_bytes": needed,
                    "guard_bytes": DISK_GUARD_BYTES,
                },
            )

    async def gen():
        layers: dict[str, tuple[int, int]] = {}
        started = time.monotonic()
        last_emit = 0.0
        last_bytes = 0
        last_time = started
        speed = 0.0
        try:
            async for progress in st.runtime.pull(name):
                if err := progress.get("error"):
                    yield event("error", {"message": humanise_pull_error(str(err))})
                    return

                digest = progress.get("digest")
                if digest and progress.get("total"):
                    layers[digest] = (
                        progress.get("completed") or 0,
                        progress["total"],
                    )
                done_bytes, total_bytes = aggregate_layers(layers)

                # Refuse to fill the disk. Checked as we go because an
                # uncatalogued model's size isn't knowable up front.
                if disk_free_bytes() < DISK_GUARD_BYTES:
                    yield event(
                        "error",
                        {
                            "message": "Stopped: less than "
                            f"{DISK_GUARD_BYTES / 1e9:.0f} GB of disk would remain.",
                            "code": "disk_guard",
                        },
                    )
                    return

                now = time.monotonic()
                if now - last_time >= 1.0:
                    speed = (done_bytes - last_bytes) / (now - last_time)
                    last_bytes, last_time = done_bytes, now

                # Throttle: layer events arrive far faster than anyone can read.
                if now - last_emit < 0.2 and progress.get("status") != "success":
                    continue
                last_emit = now

                remaining = total_bytes - done_bytes
                yield event(
                    "progress",
                    {
                        "status": progress.get("status"),
                        "completed": done_bytes,
                        "total": total_bytes,
                        "pct": round(done_bytes / total_bytes * 100, 1)
                        if total_bytes
                        else None,
                        "bytes_per_s": round(speed) if speed > 0 else None,
                        "eta_s": round(remaining / speed)
                        if speed > 0 and remaining > 0
                        else None,
                    },
                )
            yield event(
                "done", {"name": name, "elapsed_s": round(time.monotonic() - started, 1)}
            )
        except Exception as exc:
            yield event("error", {"message": humanise_pull_error(str(exc))})

    return sse(gen())


@router.delete("/{name:path}")
async def delete(name: str) -> dict:
    st = get_state()
    before = disk_free_bytes()
    try:
        await st.runtime.delete(name)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": name, "freed_bytes": max(0, disk_free_bytes() - before)}


@router.post("/{name:path}/warm")
async def warm(name: str, keep_alive: str = "10m") -> dict:
    st = get_state()
    return {"load_ms": await st.runtime.warm(name, keep_alive)}


@router.post("/{name:path}/unload")
async def unload(name: str) -> dict:
    st = get_state()
    await st.runtime.unload(name)
    return {"ok": True}
