"""FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from localslm import __version__
from localslm.api.deps import init_state, shutdown_state
from localslm.api.routes import chat, health, models, stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state()
    yield
    await shutdown_state()


app = FastAPI(title="LocalSLM", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5273", "http://127.0.0.1:5273"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (health, models, chat, stats):
    app.include_router(module.router, prefix="/api")
