"""Shared application state.

Built once in the lifespan and handed to every route. The sampler in
particular must be process-wide: per-request samplers would each see a
different machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from sqlite3 import Connection

from localslm.config import Config, load_config
from localslm.db import connect, record_env, write
from localslm.host import HostInfo, host_info
from localslm.inference.lock import InferenceLease
from localslm.runtimes.ollama import OllamaRuntime
from localslm.telemetry.sampler import SystemSampler


@dataclass
class AppState:
    cfg: Config
    conn: Connection
    runtime: OllamaRuntime
    sampler: SystemSampler
    lease: InferenceLease
    host: HostInfo
    env_hash: str


_state: AppState | None = None


def init_state() -> AppState:
    global _state
    cfg = load_config()
    conn = connect()
    host = host_info()
    sampler = SystemSampler(interval_s=cfg.get("telemetry.sample_interval_s", 0.25))
    sampler.start()
    state = AppState(
        cfg=cfg,
        conn=conn,
        runtime=OllamaRuntime(
            cfg["ollama.host"], timeout_s=cfg.get("ollama.request_timeout_s", 600)
        ),
        sampler=sampler,
        lease=InferenceLease(),
        host=host,
        env_hash=host.hash(),
    )
    record_env(conn, host.__dict__ | {"ollama_env": host.ollama_env}, state.env_hash)
    reconcile_interrupted_runs(conn)
    _state = state
    return state


def reconcile_interrupted_runs(conn: Connection) -> int:
    """Close out runs left marked `running` by a process that died.

    Nothing can still be running: this process has just started and holds no
    lease. Leaving the rows as `running` would show a permanently in-flight
    benchmark in the UI and hide the fact that its results are partial.
    """
    with write(conn):
        cur = conn.execute(
            "UPDATE runs SET status = 'aborted', abort_reason = 'interrupted', "
            "finished_ts = ? WHERE status = 'running'",
            (time.time(),),
        )
    return cur.rowcount


def get_state() -> AppState:
    if _state is None:  # pragma: no cover - lifespan always runs first
        raise RuntimeError("app state not initialised")
    return _state


async def shutdown_state() -> None:
    if _state is None:
        return
    _state.sampler.stop()
    await _state.runtime.aclose()
    _state.conn.close()
