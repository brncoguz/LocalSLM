"""End-to-end measurement, with no Ollama involved.

These assert the two behaviours that decide whether any published number is
trustworthy: that thinking tokens don't get counted as answers, and that a
measurement taken while the machine was swapping is excluded rather than
quietly averaged in as merely slow.
"""

from __future__ import annotations

import pytest
from conftest import FakeRuntime, FakeSampler

from localslm.db import query
from localslm.inference.session import MeasuredSession, Provenance
from localslm.runtimes.base import DecodeOptions, GenRequest, Message
from localslm.telemetry.sampler import SampleWindow

SERVER = {
    "prompt_eval_count": 20,
    "prompt_eval_duration": 200_000_000,
    "eval_count": 6,
    "eval_duration": 300_000_000,
    "load_duration": 0,
}


def clean_window(**kw):
    base = {
        "n": 10,
        "duration_s": 1.0,
        "peak_rss_bytes": 3_000_000_000,
        "mean_rss_bytes": 3e9,
        "swap_delta_bytes": 0,
        "pressure_max": "normal",
    }
    return SampleWindow(**{**base, **kw})


def make(script, window=None, server=None):
    return MeasuredSession(
        FakeRuntime(script, server or SERVER), None, FakeSampler(window or clean_window())
    )


def request(model_ref):
    return GenRequest(
        model=model_ref,
        messages=(Message("user", "hi"),),
        options=DecodeOptions(temperature=0.0),
    )


@pytest.mark.asyncio
async def test_ttfat_excludes_thinking_time(conn, model_ref):
    """A thinking model streams reasoning first. TTFT sees that token at 0.5 s;
    TTFAT must not, because the user sees nothing readable until 3.0 s."""
    script = [
        (0.5, "let me think", ""),
        (1.0, " more", ""),
        (3.0, "", "42"),
        (3.1, "", "!"),
    ]
    session = make(script)
    session.conn = conn
    result = await session.generate(request(model_ref), Provenance(kind="bench"))

    assert result.ttft_ms == pytest.approx(500, abs=1)
    assert result.ttfat_ms == pytest.approx(3000, abs=1)
    assert result.text == "42!"
    assert result.thinking == "let me think more"
    assert result.thinking_chunks == 2
    assert result.answer_chunks == 2


@pytest.mark.asyncio
async def test_non_thinking_model_has_ttft_equal_to_ttfat(conn, model_ref):
    session = make([(0.4, "", "hello")])
    session.conn = conn
    result = await session.generate(request(model_ref), Provenance())
    assert result.ttft_ms == result.ttfat_ms


@pytest.mark.asyncio
async def test_swapped_window_is_excluded_not_merely_slow(conn, model_ref):
    session = make([(0.2, "", "x")], window=clean_window(swap_delta_bytes=50_000_000))
    session.conn = conn
    await session.generate(request(model_ref), Provenance(kind="bench"))

    row = query(conn, "SELECT excluded, exclude_reason FROM requests")[0]
    assert row["excluded"] == 1
    assert row["exclude_reason"] == "swapped"


@pytest.mark.asyncio
async def test_critical_memory_pressure_is_excluded(conn, model_ref):
    session = make([(0.2, "", "x")], window=clean_window(pressure_max="critical"))
    session.conn = conn
    await session.generate(request(model_ref), Provenance(kind="bench"))
    row = query(conn, "SELECT excluded, exclude_reason FROM requests")[0]
    assert row["excluded"] == 1
    assert row["exclude_reason"] == "memory_pressure"


@pytest.mark.asyncio
async def test_explicit_warmup_exclusion_is_preserved(conn, model_ref):
    """A warmup runs for real and stays visible in the trace, but must never
    reach an aggregate."""
    session = make([(0.2, "", "x")])
    session.conn = conn
    await session.generate(
        request(model_ref), Provenance(kind="bench", excluded=True, exclude_reason="warmup")
    )
    row = query(conn, "SELECT excluded, exclude_reason FROM requests")[0]
    assert (row["excluded"], row["exclude_reason"]) == (1, "warmup")


@pytest.mark.asyncio
async def test_row_records_experiment_variables_and_server_counters(conn, model_ref):
    session = make([(0.3, "", "ok")])
    session.conn = conn
    req = GenRequest(
        model=model_ref,
        messages=(Message("user", "hi"),),
        think="high",
        options=DecodeOptions(temperature=0.7, seed=42),
    )
    await session.generate(req, Provenance(kind="bench", run_id="r1", pass_idx=2, rep=1))

    row = query(conn, "SELECT * FROM requests")[0]
    assert row["think"] == "high"
    assert row["temperature"] == 0.7
    assert row["seed"] == 42
    assert row["run_id"] == "r1"
    assert row["pass_idx"] == 2
    assert row["rep"] == 1
    assert row["quant"] == "Q4_K_M"
    assert row["prompt_tokens"] == 20
    assert row["output_tokens"] == 6
    assert row["decode_tps"] == pytest.approx(20.0)  # 6 tokens / 0.3 s
    assert row["prompt_hash"]
    assert row["excluded"] == 0


@pytest.mark.asyncio
async def test_streaming_errors_land_on_the_row_rather_than_raising(conn, model_ref):
    class Boom:
        name = "boom"

        async def stream(self, req):
            yield  # pragma: no cover - never reached
            raise RuntimeError

    class Exploding(Boom):
        async def stream(self, req):
            raise RuntimeError("engine died")
            yield

    session = MeasuredSession(Exploding(), conn, FakeSampler(clean_window()))
    result = await session.generate(request(model_ref), Provenance())
    assert "engine died" in (result.error or "")
    row = query(conn, "SELECT ok, error, finish_reason FROM requests")[0]
    assert row["ok"] == 0
    assert row["finish_reason"] == "error"
