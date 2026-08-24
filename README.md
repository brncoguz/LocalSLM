# LocalSLM

A lab bench for small language models running entirely on your own machine.
Chat with them, and measure them properly while you do.

## Why

There are real situations where a hosted API isn't an option: privacy rules that
forbid the data leaving, latency budgets that can't absorb a network round trip,
cost at scale, edge deployments with no connectivity.

What's scarce isn't running a small model — `ollama pull` does that. It's knowing
**what it actually costs you** in speed, memory and quality on hardware you own.

## Features

**Chat** with any installed model, streaming, with live measurement. Thinking
collapses into a pill; a footer under each reply shows time-to-answer, tok/s,
token counts and cold-start, expandable to full timings.

**History** — conversations persist in a sidebar, grouped by day, renameable.
Reopening one restores its messages *with the measurements they were recorded
with*. Deleting a conversation keeps its measurements.

**Models** — pull from a curated catalog (each entry says what contrast it buys)
or anything on the Ollama registry, with live progress, throughput and ETA.
Load, unload and delete in place. Each card doubles as a scorecard showing that
model's real numbers.

**Stats** — throughput over time, speed vs memory, time-to-first-answer spread,
cold-start penalty, thinking-vs-answering split. One colour per model across
every chart.

**CLI** — `doctor` (machine state), `models`, `repair`.

## The measurement ethic

An instrument that lies is worse than no instrument. Three rules follow:

**Chat and benchmarks write the same row.** Every generation lands in one
`requests` table, so the tok/s under a chat bubble is the number the report
averages. There's no second code path that could compute it differently.

**Measurements taken on a busy machine are thrown away, not reported as slow.**
If a run swapped or hit critical memory pressure, the row is marked `excluded`
with a reason — visible in the trace, invisible to every aggregate. The filter
lives in SQL, not in components, so no chart can forget it. The exclusion ledger
is shown, not hidden: currently 10 of 43, 3 of them for swapping.

**Time-to-first-token is the wrong headline for a reasoning model.** These models
stream *thinking* tokens first, so TTFT can be under a second while you wait half
a minute for a readable word. **TTFAT** — time to first *answer* token — is
measured separately and leads.

## What it found

Measured on this machine. Small `n`; the suite runner that produces confidence
intervals is phase 2b.

| model | tok/s | to answer | peak RAM | cold start |
|---|---:|---:|---:|---:|
| llama3.2:1b | 198.2 | 0.20 s | 5.2 GB | 78× |
| llama3.2:3b | 91.4 | 0.35 s | 2.7 GB | 8.0× |
| gemma3:4b | 60.6 | 0.48 s | 6.1 GB | 10.9× |
| qwen3.5:9b | 41.7 | 0.53 s | 6.5 GB | — |
| qwen3.5:4b | 38.7 | 0.54 s | 6.2 GB | 14.0× |
| deepseek-r1:8b | 23.0 | 72.9 s | 6.1 GB | — |

- **Reasoning cost 27× latency for an identical answer.** `qwen3.5:4b` solves the
  pens question in 1.4 s with thinking off, 39 s with it on. Both say `$14`.
- **`deepseek-r1:8b` ignores `think=False`** — same ~1,672 tokens either way. "Turn
  reasoning off to go faster" is not portable advice.
- **A 2 GB model can need 17.8 GB.** `llama3.2:3b` at its advertised 131k context
  goes from 2.31 GB resident to 17.77 GB — more than this machine has. Model size
  on disk tells you almost nothing about whether you can actually run it.
- **Cold-start detection was wrong for every row.** Ollama reports 137–305 ms of
  `load_duration` even when weights are resident; real cold loads run 1.5–12 s.
  Residency from `/api/ps` is now authoritative, and `localslm repair` re-scores
  older rows.

## Quick start

Needs [Ollama](https://ollama.com) running and Python 3.13.

```bash
make install     # venv + npm deps
make doctor      # ollama, disk, memory, power source
make serve       # API on :8420, UI on :5273
```

Ports 8420/5273 rather than 8000/5173, which other local projects tend to occupy.

## Architecture

```
  ui/ (React + Tailwind + recharts)
   │  SSE, same-origin via the vite proxy
   ▼
  api/routes/{health,models,chat,stats}          FastAPI
   │
   ▼
  inference/session.py ──── the ONE place a generation is timed and persisted
   │         │        │
   │         │        └──► telemetry/sampler.py   one thread, ring buffer;
   │         │                                    requests take slices of it
   │         └──────────► runtimes/base.py        the Runtime protocol
   │                        └─ runtimes/ollama.py    (the only adapter so far)
   ▼
  db.py ── one SQLite file ── read by the UI, the eval harness and the report
```

**One SQLite file, raw `sqlite3`, no ORM, no migrations.** Not minimalism: the UI,
the benchmark runner and the report must read the *same rows*. If the benchmark
wrote its own JSON and the UI aggregated something else, you'd eventually publish
a number you couldn't reproduce in the inspector.

**The runtime lives behind a protocol.** Ollama serves MLX-format weights
alongside GGUF, so "which runtime is fastest on Apple Silicon" is answerable by
changing a model name. A native llama.cpp backend later means writing one file.

**Chat holds a shared lease; benchmarks an exclusive one.** A chat request during
a benchmark contends for the same GPU and memory and would inflate every latency
without leaving a trace. Chat gets a `409` instead — you cannot both chat and
measure.

## Status

- [x] **Phase 1** — measurement spine, chat, model management
- [x] **Phase 2a** — conversation history, per-model history, charts
- [ ] **Phase 2b** — suites, graders, matrix runner, bootstrap confidence intervals
- [ ] **Phase 3** — context scaling, memory envelope, thermal drift, energy, needle
- [ ] **Phase 4** — reasoning economics, quantization ladder
- [ ] **Phase 5** — judging, arena, generated technical report

## Tests

```bash
make test       # offline: no Ollama, no network, no models needed
make test-all   # adds tests needing a live server
```

The offline suite fakes a runtime replaying scripted chunks at controlled
timestamps — the only way to assert on timing maths deterministically, including
that thinking tokens never count as answer tokens and that a swapped window is
excluded rather than averaged in.
