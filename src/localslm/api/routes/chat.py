"""Chat, with the measurement always on.

Every message is a benchmark datapoint. That is the point of routing chat
through `MeasuredSession`: the tok/s under a chat bubble is the same value the
report averages, so the thing you debug and the thing you publish cannot drift.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from localslm.api.deps import get_state
from localslm.api.sse import event, sse
from localslm.db import insert, new_id, query
from localslm.db import write as db_write
from localslm.inference.lock import LeaseBusy
from localslm.inference.session import MeasuredSession, Provenance
from localslm.runtimes.base import DecodeOptions, GenRequest, Message, ModelRef

router = APIRouter(tags=["chat"])

# How often the UI gets a rolling stats frame while tokens stream.
STAT_INTERVAL_S = 0.4


class ChatRequest(BaseModel):
    model: str
    message: str
    session_id: str | None = None
    system: str | None = None
    think: bool | str | None = None
    temperature: float = 0.7
    top_p: float | None = None
    seed: int | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for constrained decoding"
    )


@router.post("/chat")
async def chat(body: ChatRequest) -> object:
    st = get_state()
    try:
        await st.lease.acquire_shared()
    except LeaseBusy as busy:
        # Refuse rather than pollute a running benchmark's measurements.
        raise HTTPException(
            409,
            {
                "error": "benchmark_running",
                "holder": busy.holder,
                "run_id": busy.run_id,
                "eta_s": busy.eta_s,
            },
        ) from busy

    session_id = body.session_id or new_id()
    now = time.time()
    if not body.session_id:
        insert(
            st.conn,
            "chat_sessions",
            {
                "id": session_id,
                "ts": now,
                "updated_ts": now,
                "title": body.message[:60],
                "model": body.model,
                "options_json": body.model_dump_json(),
            },
        )

    history = query(
        st.conn,
        "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY idx",
        (session_id,),
    )
    idx = len(history)
    messages: list[Message] = []
    if body.system:
        messages.append(Message("system", body.system))
    messages.extend(Message(r["role"], r["content"]) for r in history)
    messages.append(Message("user", body.message))

    insert(
        st.conn,
        "chat_messages",
        {
            "id": new_id(),
            "session_id": session_id,
            "ts": now,
            "idx": idx,
            "role": "user",
            "content": body.message,
        },
    )

    # The model can vanish between the UI listing it and the message being
    # sent — deleting weights is one click away, and past measurements keep
    # the name visible in Stats long after. Fail clearly rather than 500.
    try:
        info = await st.runtime.show(body.model)
    except Exception as exc:
        await st.lease.release_shared()
        raise HTTPException(
            404,
            {
                "error": "model_missing",
                "model": body.model,
                "message": f"{body.model} isn't installed any more. Pull it again from Models.",
            },
        ) from exc
    req = GenRequest(
        model=ModelRef(
            runtime="ollama",
            name=body.model,
            quant=info.ref.quant,
            params_b=info.ref.params_b,
        ),
        messages=tuple(messages),
        think=body.think,
        schema=body.output_schema,
        options=DecodeOptions(
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            num_ctx=body.num_ctx,
            num_predict=body.num_predict,
        ),
        keep_alive=st.cfg.get("ollama.keep_alive", "5m"),
    )

    async def gen():
        session = MeasuredSession(st.runtime, st.conn, st.sampler)
        t0 = time.perf_counter()
        last_stat = 0.0
        answer_chunks = 0
        yield event("meta", {"session_id": session_id, "model": body.model})
        try:
            async for chunk, done in session.stream(
                req, Provenance(kind="chat", session_id=session_id)
            ):
                if chunk is not None:
                    if chunk.thinking:
                        yield event("thinking", {"delta": chunk.thinking})
                    elif chunk.text:
                        answer_chunks += 1
                        yield event("token", {"delta": chunk.text})
                    now_s = time.perf_counter()
                    if now_s - last_stat >= STAT_INTERVAL_S:
                        last_stat = now_s
                        elapsed = now_s - t0
                        yield event(
                            "stat",
                            {
                                "elapsed_ms": round(elapsed * 1000),
                                "chunks": answer_chunks,
                                "tps": round(answer_chunks / elapsed, 1)
                                if elapsed
                                else None,
                            },
                        )
                elif done is not None:
                    request_id = session.last_request_id
                    srv = done.server
                    insert(
                        st.conn,
                        "chat_messages",
                        {
                            "id": new_id(),
                            "session_id": session_id,
                            "ts": time.time(),
                            "idx": idx + 1,
                            "role": "assistant",
                            "content": done.text,
                            "thinking": done.thinking,
                            "request_id": request_id,
                            "error": done.error,
                        },
                    )
                    yield event(
                        "done",
                        {
                            "request_id": request_id,
                            "text": done.text,
                            "thinking": done.thinking,
                            "error": done.error,
                            "stats": {
                                "ttft_ms": done.ttft_ms,
                                "ttfat_ms": done.ttfat_ms,
                                "total_ms": done.total_ms,
                                "decode_tps": srv.decode_tps if srv else None,
                                "prefill_tps": srv.prefill_tps if srv else None,
                                "prompt_tokens": srv.prompt_tokens if srv else None,
                                "output_tokens": srv.output_tokens if srv else None,
                                "thinking_chunks": done.thinking_chunks,
                                "answer_chunks": done.answer_chunks,
                                "think_ratio": done.think_ratio,
                                "itl_stddev_ms": done.itl_stddev_ms,
                                "cold": done.cold,
                                "load_ms": srv.load_ms if srv else None,
                            },
                        },
                    )
        finally:
            await st.lease.release_shared()

    return sse(gen())


@router.get("/sessions")
async def sessions() -> list[dict]:
    st = get_state()
    rows = query(
        st.conn,
        "SELECT s.id, s.ts, s.title, s.model, COUNT(m.id) AS messages "
        "FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id = s.id "
        "GROUP BY s.id ORDER BY s.ts DESC LIMIT 50",
    )
    return [dict(r) for r in rows]


@router.get("/sessions/{session_id}")
async def session_detail(session_id: str) -> dict:
    st = get_state()
    msgs = query(
        st.conn,
        # Every column the StatStrip renders, so a restored conversation shows
        # the numbers it was measured with rather than a degraded summary.
        "SELECT m.*, r.ttft_ms, r.ttfat_ms, r.total_ms, r.decode_tps, r.prefill_tps, "
        "r.prompt_tokens, r.output_tokens, r.thinking_tokens, r.answer_tokens, "
        "r.itl_stddev_ms, r.srv_load_ms, r.cold, r.model, r.think "
        "FROM chat_messages m LEFT JOIN requests r ON r.id = m.request_id "
        "WHERE m.session_id = ? ORDER BY m.idx",
        (session_id,),
    )
    return {"id": session_id, "messages": [dict(r) for r in msgs]}


class SessionPatch(BaseModel):
    title: str


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: SessionPatch) -> dict:
    st = get_state()
    title = body.title.strip()[:120]
    if not title:
        raise HTTPException(422, "title cannot be empty")
    with db_write(st.conn):
        cur = st.conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_ts = ? WHERE id = ?",
            (title, time.time(), session_id),
        )
    if not cur.rowcount:
        raise HTTPException(404, "no such session")
    return {"id": session_id, "title": title}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a conversation, keeping its measurements.

    `chat_messages` is scrollback and goes. The `requests` rows are the
    benchmark record — they outlive the conversation that produced them, so
    deleting a chat never silently shrinks the dataset the charts read.
    """
    st = get_state()
    rows = query(
        st.conn, "SELECT COUNT(*) n FROM requests WHERE session_id = ?", (session_id,)
    )
    kept = rows[0]["n"]
    with db_write(st.conn):
        st.conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cur = st.conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    if not cur.rowcount:
        raise HTTPException(404, "no such session")
    return {"deleted": session_id, "measurements_kept": kept}


@router.get("/requests/{request_id}")
async def request_detail(request_id: str) -> dict:
    """The trace drawer. Every number in the UI is two clicks from this."""
    st = get_state()
    rows = query(st.conn, "SELECT * FROM requests WHERE id = ?", (request_id,))
    if not rows:
        raise HTTPException(404, "no such request")
    return dict(rows[0])
