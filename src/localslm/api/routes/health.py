"""Health and machine context.

The UI renders this in a persistent strip. It is not decoration: memory
headroom and power source change what the numbers below them mean, so they
belong on screen at all times.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from localslm.api.deps import get_state
from localslm.db import query
from localslm.host import disk_free_bytes, power_state

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    st = get_state()
    runtime = await st.runtime.health()
    loaded = await st.runtime.loaded() if runtime["ok"] else []
    sample = st.sampler.latest
    power = power_state()
    counts = query(
        st.conn,
        "SELECT (SELECT COUNT(*) FROM requests) AS requests, (SELECT COUNT(*) FROM runs) AS runs",
    )[0]
    lease = st.lease.status()
    return {
        "ok": runtime["ok"],
        "runtime": runtime,
        "host": asdict(st.host) | {"ram_gb": st.host.ram_gb, "env_hash": st.env_hash},
        "config_hash": st.cfg.fingerprint(),
        "loaded": [
            {
                "name": m.name,
                "size_bytes": m.size_bytes,
                "size_vram_bytes": m.size_vram_bytes,
                "expires_at": m.expires_at,
            }
            for m in loaded
        ],
        "memory": {
            "free_bytes": sample.free_bytes if sample else None,
            "runner_rss_bytes": sample.ollama_rss_bytes if sample else None,
            "swap_used_bytes": sample.swap_used_bytes if sample else None,
            "pressure": sample.pressure if sample else "unknown",
        },
        "disk": {"free_bytes": disk_free_bytes()},
        "power": {"on_ac": power.on_ac, "battery_pct": power.battery_pct},
        "db": {"requests": counts["requests"], "runs": counts["runs"]},
        "lease": {
            "busy": lease.busy,
            "holder": lease.holder,
            "exclusive": lease.exclusive,
            "run_id": lease.run_id,
        },
        "presets": st.cfg.presets,
    }
