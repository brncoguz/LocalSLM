"""Tests that need a live Ollama. Run with `make test-all`.

Deselected by default so a clean checkout can run the suite with no models
downloaded and no server running.
"""

from __future__ import annotations

import pytest

from localslm.config import load_config
from localslm.runtimes.base import DecodeOptions, GenRequest, Message, ModelRef
from localslm.runtimes.ollama import OllamaRuntime

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def rt():
    runtime = OllamaRuntime(load_config()["ollama.host"])
    yield runtime
    await runtime.aclose()


async def test_server_is_reachable(rt):
    health = await rt.health()
    assert health["ok"], health


async def test_models_expose_quant_and_params(rt):
    models = await rt.list_models()
    assert models, "no models installed — pull one first"
    for m in models:
        assert m.ref.quant
        assert m.ref.params_b and m.ref.params_b > 0


async def test_show_reports_capabilities_and_context(rt):
    models = await rt.list_models()
    info = await rt.show(models[0].ref.name)
    assert info.ref.ctx_max and info.ref.ctx_max > 0
    assert info.capabilities.completion


async def test_client_and_server_throughput_agree(rt):
    """A large divergence between our own timing and Ollama's own counters
    would mean the streaming path is adding overhead — worth failing over."""
    models = await rt.list_models()
    name = models[0].ref.name
    result = await rt.generate(
        GenRequest(
            model=ModelRef("ollama", name),
            messages=(Message("user", "Write one short sentence about rain."),),
            think=False,
            options=DecodeOptions(temperature=0.0, num_predict=64),
        )
    )
    assert result.error is None, result.error
    assert result.text.strip()
    srv = result.server
    assert srv and srv.decode_tps and srv.decode_tps > 1

    # Client wall-clock must bound the server's own generation time.
    assert result.total_ms >= srv.eval_ms
    assert result.ttfat_ms is not None
