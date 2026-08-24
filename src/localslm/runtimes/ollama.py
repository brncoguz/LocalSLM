"""Ollama adapter.

The only real runtime for now. Everything Ollama-specific lives behind the
`Runtime` protocol so the bench layer stays portable: `qwen3.5:4b-mlx` and
`qwen3.5:4b` are both just model names here, which is what makes the
MLX-vs-GGUF comparison free rather than a second backend.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from localslm.runtimes.base import (
    Capabilities,
    Chunk,
    GenRequest,
    GenResult,
    LoadedModel,
    ModelInfo,
    ModelRef,
    ServerTiming,
)

# `ollama show` reports parameter counts as "9.7B" / "4.7B" / "494.03M".
_PARAM_RE = re.compile(r"^([\d.]+)\s*([BM])$", re.IGNORECASE)


def _parse_params(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value) / 1e9 if value > 1e6 else float(value)
    m = _PARAM_RE.match(str(value).strip())
    if not m:
        return None
    n = float(m.group(1))
    return n if m.group(2).upper() == "B" else n / 1000


class OllamaRuntime:
    name = "ollama"

    def __init__(self, host: str, timeout_s: float = 600.0) -> None:
        self.host = host.rstrip("/")
        self._timeout = timeout_s
        self._client = httpx.AsyncClient(base_url=self.host, timeout=timeout_s)
        self._caps: dict[str, Capabilities] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- introspection ----------------------------------------------------

    async def health(self) -> dict[str, Any]:
        try:
            r = await self._client.get("/api/version", timeout=5.0)
            r.raise_for_status()
            return {"ok": True, "version": r.json().get("version"), "endpoint": self.host}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc), "endpoint": self.host}

    async def list_models(self) -> list[ModelInfo]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        out = []
        for m in r.json().get("models", []):
            details = m.get("details") or {}
            out.append(
                ModelInfo(
                    ref=ModelRef(
                        runtime=self.name,
                        name=m["name"],
                        digest=(m.get("digest") or "")[:12] or None,
                        quant=details.get("quantization_level"),
                        params_b=_parse_params(details.get("parameter_size")),
                    ),
                    family=details.get("family"),
                    size_bytes=m.get("size", 0),
                    details=details,
                )
            )
        return out

    async def show(self, name: str) -> ModelInfo:
        r = await self._client.post("/api/show", json={"model": name})
        r.raise_for_status()
        data = r.json()
        details = data.get("details") or {}
        info = data.get("model_info") or {}
        caps = set(data.get("capabilities") or [])

        arch = info.get("general.architecture") or details.get("family")
        ctx = next(
            (v for k, v in info.items() if k.endswith(".context_length")),
            None,
        )
        capabilities = Capabilities(
            completion="completion" in caps,
            thinking="thinking" in caps,
            tools="tools" in caps,
            vision="vision" in caps,
            # Ollama constrains any model via `format`; thinking levels are only
            # meaningful where the model advertises thinking at all.
            schema=True,
            think_levels=("low", "medium", "high") if "thinking" in caps else (),
        )
        self._caps[name] = capabilities
        return ModelInfo(
            ref=ModelRef(
                runtime=self.name,
                name=name,
                quant=details.get("quantization_level"),
                params_b=_parse_params(details.get("parameter_size")),
                ctx_max=int(ctx) if ctx else None,
            ),
            family=details.get("family"),
            arch=arch,
            capabilities=capabilities,
            details={"parameters": data.get("parameters"), **details},
        )

    async def capabilities(self, name: str) -> Capabilities:
        if name not in self._caps:
            await self.show(name)
        return self._caps.get(name, Capabilities())

    async def loaded(self) -> list[LoadedModel]:
        r = await self._client.get("/api/ps")
        r.raise_for_status()
        return [
            LoadedModel(
                name=m["name"],
                size_bytes=m.get("size", 0),
                size_vram_bytes=m.get("size_vram", 0),
                expires_at=m.get("expires_at"),
            )
            for m in r.json().get("models", [])
        ]

    # -- lifecycle --------------------------------------------------------

    async def warm(self, name: str, keep_alive: str = "10m") -> float:
        """Load weights without generating. Returns wall-clock load time in ms,
        which is the cold-start penalty a user pays on first message."""
        t0 = time.perf_counter()
        r = await self._client.post(
            "/api/generate", json={"model": name, "keep_alive": keep_alive}
        )
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000

    async def unload(self, name: str) -> None:
        await self._client.post("/api/generate", json={"model": name, "keep_alive": 0})

    async def delete(self, name: str) -> None:
        r = await self._client.request("DELETE", "/api/delete", json={"model": name})
        r.raise_for_status()

    async def pull(self, name: str) -> AsyncIterator[dict[str, Any]]:
        async with self._client.stream(
            "POST", "/api/pull", json={"model": name}, timeout=None
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    # -- generation -------------------------------------------------------

    def _payload(self, req: GenRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model.name,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": True,
            "keep_alive": req.keep_alive,
            "options": req.options.as_ollama(),
        }
        if req.schema is not None:
            body["format"] = req.schema
        if req.tools:
            body["tools"] = list(req.tools)
        if req.think is not None:
            body["think"] = req.think
        return body

    async def stream(self, req: GenRequest) -> AsyncIterator[Chunk]:
        async with self._client.stream(
            "POST", "/api/chat", json=self._payload(req), timeout=req.timeout_s
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                # Stamp arrival before parsing so the timestamp measures the
                # runtime, not our JSON decoding.
                t = time.perf_counter()
                if not line.strip():
                    continue
                payload = json.loads(line)
                msg = payload.get("message") or {}
                yield Chunk(
                    t_mono=t,
                    text=msg.get("content") or "",
                    thinking=msg.get("thinking") or "",
                    done=bool(payload.get("done")),
                    raw=payload if payload.get("done") else None,
                )

    async def generate(self, req: GenRequest) -> GenResult:
        result = GenResult(model=req.model)
        text: list[str] = []
        thinking: list[str] = []
        t_start = time.perf_counter()
        t_prev: float | None = None

        try:
            async for chunk in self.stream(req):
                if chunk.done:
                    result.raw_done = chunk.raw
                    if chunk.raw:
                        result.server = ServerTiming.from_done(chunk.raw)
                        result.finish_reason = chunk.raw.get("done_reason") or "stop"
                    continue

                kind = chunk.kind
                if kind == "empty":
                    continue

                if result.ttft_ms is None:
                    result.ttft_ms = (chunk.t_mono - t_start) * 1000

                if kind == "thinking":
                    thinking.append(chunk.thinking)
                    result.thinking_chunks += 1
                else:
                    if result.ttfat_ms is None:
                        result.ttfat_ms = (chunk.t_mono - t_start) * 1000
                    text.append(chunk.text)
                    result.answer_chunks += 1

                if t_prev is not None:
                    result.inter_token_ms.append((chunk.t_mono - t_prev) * 1000)
                t_prev = chunk.t_mono
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.finish_reason = "error"

        result.total_ms = (time.perf_counter() - t_start) * 1000
        result.text = "".join(text)
        result.thinking = "".join(thinking)
        # Ollama reports load_duration only when it actually loaded weights.
        result.cold = bool(result.server and result.server.load_ms > 100)
        return result
