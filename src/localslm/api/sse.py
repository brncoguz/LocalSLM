"""Server-sent events.

Hand-rolled rather than pulling a dependency: SSE is four lines of framing, and
the one thing we genuinely need — flushing each token the instant it arrives —
is a property of the generator, not the library.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, a proxy will happily buffer the whole stream and every
    # latency number the UI shows becomes a lie.
    "X-Accel-Buffering": "no",
}


def event(kind: str, data: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)
