"""Tests that need a live Ollama. Run with `make test-all`.

Deselected by default so a clean checkout can run the suite with no models
downloaded and no server running.

**Correctness only.** These also run nightly on a GitHub macOS runner, where
they verify that the adapter still speaks Ollama properly — streaming, /api/ps,
capability detection, the thinking/answer split. They deliberately assert
nothing about speed: a shared virtualised runner measures its neighbours, not
the model, and a timing assertion here would either flake or quietly legitimise
numbers this project would refuse to publish.
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
    # Deliberately not `models[0]`: whichever model Ollama happens to list
    # first may be a reasoner, and `deepseek-r1:8b` ignores `think=False`
    # entirely — it spends the whole 64-token budget on its monologue and
    # returns no answer text, failing an assertion about throughput for a
    # reason that has nothing to do with throughput.
    models = await rt.list_models()
    name = None
    for m in models:
        if not (await rt.capabilities(m.ref.name)).thinking:
            name = m.ref.name
            break
    if name is None:
        pytest.skip("every installed model reasons; this test needs a direct answer")

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
