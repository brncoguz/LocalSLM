"""Offline fixtures.

The whole unit suite runs with no Ollama and no network: a fake runtime
replays a scripted chunk sequence with controlled timestamps, which is the
only way to assert on timing maths deterministically.
"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from localslm.db import connect
from localslm.runtimes.base import Chunk, LoadedModel, ModelRef


class FakeRuntime:
    """Replays chunks at controlled offsets from the moment streaming starts.

    Offsets are anchored to the real `perf_counter` at first read rather than
    to an absolute constant, because the session under test computes its own
    `t_start` from the same clock — that shared origin is exactly what the
    timing assertions are checking.

    `script` is a list of (offset_seconds, thinking, text); a terminal
    done-chunk carrying server counters is appended automatically.
    """

    name = "fake"

    def __init__(self, script, server: dict | None = None, loaded=None, installed=()):
        self.script = script
        self.server = server or {}
        self.installed = list(installed)
        # None means "this runtime cannot answer", which the session must keep
        # distinct from "nothing is loaded" — the first falls back to the
        # load-duration rule, the second is a definitive cold.
        self._loaded = loaded

    async def loaded(self):
        if self._loaded is None:
            raise RuntimeError("runtime cannot say what is loaded")
        return self._loaded

    async def list_models(self):
        return self.installed

    async def stream(self, req) -> AsyncIterator[Chunk]:
        base = time.perf_counter()
        for offset, thinking, text in self.script:
            yield Chunk(t_mono=base + offset, thinking=thinking, text=text)
        yield Chunk(t_mono=base + 99, done=True, raw={"done": True, **self.server})


class FakeSampler:
    """Sampler stub returning a fixed window."""

    def __init__(self, window):
        self._window = window

    def window(self, t0, t1):
        return self._window


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def model_ref():
    return ModelRef(runtime="fake", name="test:4b", quant="Q4_K_M", params_b=4.0)


@pytest.fixture
def loaded_entry():
    """What `/api/ps` says about the model under test while it is resident."""
    return LoadedModel(
        name="test:4b", size_bytes=3_400_000_000, size_vram_bytes=3_400_000_000
    )


@pytest.fixture
def client(conn, monkeypatch):
    """The real FastAPI app, wired to a temporary database.

    Deliberately constructed without `with TestClient(app)`: entering the
    context manager runs the lifespan, which calls `init_state()` and would
    open the developer's actual `data/localslm.db` and start a live sampler
    thread. Routes reach for the module-global `_state` directly rather than
    through `Depends`, so it is replaced here instead of overridden.
    """
    from fastapi.testclient import TestClient

    import localslm.api.deps as deps
    from localslm.api.deps import AppState
    from localslm.api.main import app
    from localslm.config import load_config
    from localslm.host import HostInfo
    from localslm.inference.lock import InferenceLease

    host = HostInfo(
        chip="test",
        cores=1,
        ram_bytes=16 * 1024**3,
        macos="test",
        python="3.13",
        ollama_version=None,
        localslm_version="test",
    )
    state = AppState(
        cfg=load_config(),
        conn=conn,
        runtime=FakeRuntime([], loaded=[]),
        sampler=FakeSampler(None),
        lease=InferenceLease(),
        host=host,
        env_hash="test",
    )
    monkeypatch.setattr(deps, "_state", state, raising=False)
    return TestClient(app)
