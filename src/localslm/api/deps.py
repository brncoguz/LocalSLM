"""Shared application state.

Built once in the lifespan and handed to every route. The sampler in
particular must be process-wide: per-request samplers would each see a
different machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection

from localslm.config import Config, load_config
from localslm.db import connect, record_env
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
    _state = state
    return state


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
