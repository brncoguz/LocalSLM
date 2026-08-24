"""Exclusive access to the inference engine.

A benchmark holds this exclusively for its whole run. Chat holds it shared.
The reason is measurement validity: a chat request arriving mid-benchmark
contends for the same GPU and the same unified memory, and would silently
inflate every latency number in the run without leaving any trace in the data.

The honest interface says so out loud — a chat POST during a run gets a 409
with the run id, not a quietly corrupted measurement.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from localslm.config import path as repo_path

# Cross-process guard. The in-process lock alone is not enough: `make bench` in
# a terminal and the API server are separate processes, and each would happily
# believe it had exclusive access while both drove the same GPU.
LOCK_FILE = repo_path("data", "engine.lock")


class LeaseBusy(RuntimeError):
    def __init__(self, holder: str, run_id: str | None, eta_s: float | None) -> None:
        super().__init__(f"inference engine held by {holder}")
        self.holder = holder
        self.run_id = run_id
        self.eta_s = eta_s


@dataclass
class LeaseStatus:
    holder: str | None
    exclusive: bool
    run_id: str | None
    since: float | None
    shared_count: int

    @property
    def busy(self) -> bool:
        return self.holder is not None


class InferenceLease:
    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._exclusive_holder: str | None = None
        self._run_id: str | None = None
        self._since: float | None = None
        self._shared = 0
        self._eta_s: float | None = None
        self._fd: int | None = None

    # -- cross-process ----------------------------------------------------

    def _take_file_lock(self, holder: str, run_id: str | None) -> bool:
        """Try to claim the on-disk lock. False if another process holds it."""
        path = Path(LOCK_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(
            fd,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "holder": holder,
                    "run_id": run_id,
                    "since": time.time(),
                }
            ).encode(),
        )
        os.fsync(fd)
        self._fd = fd
        return True

    def _release_file_lock(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    @staticmethod
    def foreign_holder() -> dict | None:
        """Who holds the lock in another process, if anyone.

        The lock is advisory and released automatically when the holding
        process exits, so a crashed benchmark cannot wedge the engine.
        """
        path = Path(LOCK_FILE)
        if not path.exists():
            return None
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                return json.loads(path.read_text() or "{}") or None
            except (OSError, ValueError):
                return {"holder": "unknown"}
            finally:
                os.close(fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None

    def status(self) -> LeaseStatus:
        return LeaseStatus(
            holder=self._exclusive_holder or ("chat" if self._shared else None),
            exclusive=self._exclusive_holder is not None,
            run_id=self._run_id,
            since=self._since,
            shared_count=self._shared,
        )

    def set_eta(self, eta_s: float | None) -> None:
        self._eta_s = eta_s

    async def acquire_shared(self) -> None:
        """Fail fast rather than queue: a user typing into chat wants to know
        the engine is busy, not to have their message silently stall."""
        async with self._cond:
            if self._exclusive_holder is not None:
                raise LeaseBusy(self._exclusive_holder, self._run_id, self._eta_s)
            if foreign := self.foreign_holder():
                raise LeaseBusy(
                    foreign.get("holder", "another process"),
                    foreign.get("run_id"),
                    self._eta_s,
                )
            self._shared += 1

    async def release_shared(self) -> None:
        async with self._cond:
            self._shared = max(0, self._shared - 1)
            self._cond.notify_all()

    async def acquire_exclusive(
        self, holder: str, run_id: str | None = None, timeout_s: float = 120.0
    ) -> None:
        deadline = time.monotonic() + timeout_s
        async with self._cond:
            while self._exclusive_holder is not None or self._shared:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LeaseBusy(
                        self._exclusive_holder or "chat", self._run_id, self._eta_s
                    )
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    raise LeaseBusy(
                        self._exclusive_holder or "chat", self._run_id, self._eta_s
                    ) from None
            if not self._take_file_lock(holder, run_id):
                foreign = self.foreign_holder() or {}
                raise LeaseBusy(
                    foreign.get("holder", "another process"),
                    foreign.get("run_id"),
                    self._eta_s,
                )
            self._exclusive_holder = holder
            self._run_id = run_id
            self._since = time.time()

    async def release_exclusive(self) -> None:
        async with self._cond:
            self._release_file_lock()
            self._exclusive_holder = None
            self._run_id = None
            self._since = None
            self._eta_s = None
            self._cond.notify_all()
