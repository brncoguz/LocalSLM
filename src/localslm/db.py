"""The SQLite substrate.

One file, raw sqlite3, a write lock, no ORM and no migrations. The reason is
not minimalism: it is that the chat UI, the benchmark runner and the exported
report must read *the same rows*. If the benchmark wrote its own JSON and the
UI aggregated something else, you would eventually publish a number you could
not reproduce in the inspector.

Two structural decisions worth defending:

* `requests` holds every generation ever made, chat and benchmark alike, tagged
  by `kind`. Chatting is therefore free data collection, and the tok/s shown
  under a chat bubble is literally the value the report averages.
* `results` is separate from `requests` because one generation can be graded
  several ways (a JSON extraction is both a schema-validity result and a
  field-accuracy result), and because re-grading after a grader bugfix must not
  require re-running inference.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from localslm.config import path as repo_path

_WRITE_LOCK = threading.Lock()

SCHEMA = """
-- Every generation, chat or bench. The substrate everything else reads.
CREATE TABLE IF NOT EXISTS requests (
    id              TEXT PRIMARY KEY,
    ts              REAL NOT NULL,
    kind            TEXT NOT NULL,          -- chat|compare|bench|arena|judge|calibration
    run_id          TEXT, session_id TEXT, unit_id TEXT,
    parent_id       TEXT,                   -- retry chain / arena sibling / rerun source
    pass_idx        INTEGER NOT NULL DEFAULT 0,   -- which pass over the model list (drift covariate)

    runtime         TEXT NOT NULL, model TEXT NOT NULL,
    model_digest    TEXT, quant TEXT, params_b REAL,

    prompt_ref      TEXT, prompt_hash TEXT,
    task            TEXT, item_id TEXT, cell_key TEXT, rep INTEGER NOT NULL DEFAULT 0,

    -- Warmups, aborted and pressure-poisoned units stay visible in the trace
    -- inspector but are invisible to every aggregate.
    excluded        INTEGER NOT NULL DEFAULT 0,
    exclude_reason  TEXT,

    -- Independent variables, denormalised on purpose: these ARE the experiment.
    temperature REAL, top_p REAL, top_k INTEGER, seed INTEGER,
    num_ctx INTEGER, num_predict INTEGER,
    think           TEXT,                   -- off|low|medium|high|true|unsupported
    constraint_mode TEXT,                   -- none|prompt|fewshot|schema|tools|unsupported
    concurrency     INTEGER NOT NULL DEFAULT 1,
    keep_alive      TEXT,

    -- Client-measured latency.
    queue_ms REAL, ttft_ms REAL, ttfat_ms REAL, total_ms REAL,
    itl_p50_ms REAL, itl_p95_ms REAL, itl_stddev_ms REAL,

    -- Server-reported telemetry. Kept separate from the client numbers and
    -- never merged: a persistent gap between them is a real finding about the
    -- streaming path, and averaging them away would hide it.
    srv_total_ms REAL, srv_load_ms REAL,
    prompt_tokens INTEGER, srv_prompt_ms REAL,
    output_tokens INTEGER, srv_eval_ms REAL,
    prefill_tps REAL, decode_tps REAL,
    thinking_tokens INTEGER, answer_tokens INTEGER,

    -- Systems.
    cold INTEGER NOT NULL DEFAULT 0,
    peak_rss_bytes INTEGER, model_vram_bytes INTEGER, swap_delta_bytes INTEGER,
    mem_pressure TEXT, mean_watts REAL, energy_j REAL,
    on_ac INTEGER, thermal TEXT,

    ok INTEGER NOT NULL DEFAULT 1, error TEXT, finish_reason TEXT,

    prompt_json TEXT, output_text TEXT, thinking_text TEXT,
    tools_json TEXT, options_json TEXT, timings_json TEXT, raw_json TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts    ON requests(ts DESC);
CREATE INDEX IF NOT EXISTS requests_run   ON requests(run_id, cell_key, excluded);
CREATE INDEX IF NOT EXISTS requests_model ON requests(model, kind, ts DESC);
CREATE INDEX IF NOT EXISTS requests_sess  ON requests(session_id, ts);
CREATE INDEX IF NOT EXISTS requests_item  ON requests(run_id, item_id);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, ts REAL NOT NULL, finished_ts REAL,
    suite TEXT NOT NULL, suite_hash TEXT, suite_json TEXT,
    preset TEXT NOT NULL DEFAULT 'default', label TEXT, notes TEXT,
    status TEXT NOT NULL,                   -- planned|running|done|aborted|failed
    abort_reason TEXT, run_seed INTEGER,
    planned_units INTEGER, done_units INTEGER, failed_units INTEGER, excluded_units INTEGER,
    models_json TEXT, env_hash TEXT, config_hash TEXT, config_json TEXT,
    prompts_json TEXT, dataset_hash TEXT,
    judge_model TEXT, judge_cost_usd REAL DEFAULT 0,
    metrics_json TEXT, elapsed_s REAL
);
CREATE INDEX IF NOT EXISTS runs_ts    ON runs(ts DESC);
CREATE INDEX IF NOT EXISTS runs_suite ON runs(suite, ts DESC);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL, request_id TEXT NOT NULL,
    cell_key TEXT, model TEXT NOT NULL,
    item_id TEXT NOT NULL, task TEXT NOT NULL,
    category TEXT, lang TEXT, difficulty TEXT,
    rep INTEGER, prompt_variant INTEGER DEFAULT 0,
    excluded INTEGER NOT NULL DEFAULT 0,
    grader TEXT NOT NULL, judged INTEGER NOT NULL DEFAULT 0,
    score REAL, passed INTEGER,
    pred_label TEXT, gold_label TEXT,
    metrics_json TEXT, detail_json TEXT
);
CREATE INDEX IF NOT EXISTS results_run  ON results(run_id, task, excluded);
CREATE INDEX IF NOT EXISTS results_cell ON results(run_id, cell_key);
CREATE INDEX IF NOT EXISTS results_item ON results(run_id, item_id);

-- Per-cell aggregates, frozen at run completion so the leaderboard and the
-- report quote the same numbers rather than each re-averaging live.
CREATE TABLE IF NOT EXISTS cells (
    run_id TEXT NOT NULL, cell_key TEXT NOT NULL,
    model TEXT NOT NULL, axes_json TEXT,
    n INTEGER, n_excluded INTEGER, n_failed INTEGER,
    ttft_p50 REAL, ttft_p95 REAL, ttfat_p50 REAL, ttfat_p95 REAL,
    total_ms_p50 REAL, total_ms_p95 REAL,
    prefill_tps_mean REAL, decode_tps_mean REAL,
    decode_tps_ci_lo REAL, decode_tps_ci_hi REAL, itl_stddev_mean REAL,
    peak_rss_bytes INTEGER, model_vram_bytes INTEGER, swap_delta_bytes INTEGER,
    energy_per_1k_j REAL,
    score_mean REAL, score_ci_lo REAL, score_ci_hi REAL, pass_rate REAL,
    thinking_tokens_mean REAL, answer_tokens_mean REAL,
    think_ratio_mean REAL, overthink_rate REAL,
    drift_slope REAL,                        -- decode_tps regressed on pass_idx
    metrics_json TEXT,
    PRIMARY KEY (run_id, cell_key)
);
CREATE INDEX IF NOT EXISTS cells_model ON cells(model);

CREATE TABLE IF NOT EXISTS arena_pairings (
    id TEXT PRIMARY KEY, ts REAL NOT NULL, run_id TEXT,
    prompt TEXT NOT NULL, prompt_hash TEXT, item_id TEXT, category TEXT,
    model_a TEXT NOT NULL, model_b TEXT NOT NULL,
    request_a TEXT, request_b TEXT,
    left_is TEXT NOT NULL,                   -- what the voter physically saw on the left
    mirror_of TEXT                           -- order-swapped twin; both cancel position bias
);
CREATE TABLE IF NOT EXISTS arena_votes (
    id TEXT PRIMARY KEY, ts REAL NOT NULL, pairing_id TEXT NOT NULL,
    winner TEXT NOT NULL,                    -- a|b|tie|both_bad
    voter TEXT NOT NULL,                     -- human|judge:<model>
    confidence REAL, rationale TEXT, decision_ms REAL
);
CREATE INDEX IF NOT EXISTS arena_pair_models ON arena_pairings(model_a, model_b);

CREATE TABLE IF NOT EXISTS model_snapshots (
    id TEXT PRIMARY KEY, first_seen REAL, last_seen REAL,
    runtime TEXT NOT NULL, name TEXT NOT NULL, digest TEXT,
    family TEXT, arch TEXT, params_b REAL, quant TEXT, ctx_max INTEGER,
    size_bytes INTEGER, capabilities_json TEXT, details_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS model_snap_key
    ON model_snapshots(runtime, name, IFNULL(digest, ''));

-- Deduped machine provenance; runs.env_hash points here. Kept separate so a
-- deleted model or a reinstalled OS never orphans an old run's context.
CREATE TABLE IF NOT EXISTS env_snapshots (
    hash TEXT PRIMARY KEY, first_seen REAL, info_json TEXT NOT NULL
);

-- 1 Hz telemetry per run; drives the thermal-drift chart. Per-request windows
-- are derived in memory from the sampler ring buffer rather than stored, so
-- the two can never disagree.
CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT NOT NULL, t_rel REAL NOT NULL,
    rss_bytes INTEGER, wired_bytes INTEGER, swap_bytes INTEGER,
    cpu_pct REAL, watts REAL, pressure TEXT, on_ac INTEGER,
    tps_window REAL,
    PRIMARY KEY (run_id, t_rel)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY, ts REAL, updated_ts REAL,
    title TEXT, model TEXT, options_json TEXT
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, ts REAL NOT NULL,
    idx INTEGER NOT NULL, role TEXT NOT NULL,
    content TEXT, thinking TEXT, request_id TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS chat_msgs ON chat_messages(session_id, idx);
"""

DEFAULT_DB = repo_path("data", "localslm.db")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    target = Path(db_path or DEFAULT_DB)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL matters here: the UI polls while a long run writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def write(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Serialise writers. One process, one lock, no surprises."""
    with _WRITE_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with write(conn):
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))


def query(
    conn: sqlite3.Connection, sql: str, args: Sequence[Any] = ()
) -> list[sqlite3.Row]:
    return conn.execute(sql, tuple(args)).fetchall()


def record_env(conn: sqlite3.Connection, info: dict[str, Any], env_hash: str) -> str:
    with write(conn):
        conn.execute(
            "INSERT OR IGNORE INTO env_snapshots (hash, first_seen, info_json) VALUES (?,?,?)",
            (env_hash, time.time(), json.dumps(info, sort_keys=True)),
        )
    return env_hash
