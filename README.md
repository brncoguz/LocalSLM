# LocalSLM

LocalSLM is a benchmarking lab for small language models running locally through Ollama. You chat with models in a web UI, and every message is recorded as a measurement — throughput, time to first answer token, memory use, cold-start cost — so the data accumulates as you use it instead of requiring a separate benchmark run.

Everything here was built and measured on a MacBook Pro M1 Pro with 16 GB of unified memory.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![React](https://img.shields.io/badge/React-19-61dafb) ![Ollama](https://img.shields.io/badge/Ollama-0.32-white)

---

## Overview

There are plenty of situations where a hosted API isn't an option: data that can't leave the building for privacy or compliance reasons, latency budgets that don't allow a network round trip, cost at scale, or edge deployments without reliable connectivity. Choosing a model for those cases means understanding trade-offs that public leaderboards don't cover, because they weren't measured on your laptop.

## Features

- **Chat:** Stream responses from any installed model. Reasoning traces collapse into an expandable summary, and a stats footer under each reply shows time to answer, throughput, token counts and whether the model had to load from disk.
- **Conversation history:** Sessions persist in a sidebar grouped by day, and can be renamed or deleted. Reopening a conversation restores its messages along with the measurements recorded at the time. Deleting a conversation removes the transcript but keeps the measurements.
- **Model management:** Pull models from a curated catalogue or any tag on the Ollama registry, with live download progress, throughput and ETA. Load, unload and delete without leaving the app. Each model card shows that model's measured performance.
- **Stats dashboard:** Charts for throughput over time, speed against memory use, time-to-first-answer distribution, cold-start penalty, and the split between reasoning and answer tokens. Each model keeps a consistent colour across every chart.
- **CLI:** `doctor` reports machine state (Ollama version, disk, memory pressure, power source), `models` lists what's installed with measured numbers, and `repair` re-scores historical rows when a metric definition changes.

## How the measurements work

Three decisions shape how the numbers are collected.

Chat and benchmarks write to the same table. Every generation, whether it came from the chat UI or a benchmark run, is stored as a row in `requests`. The throughput shown under a chat message is the same value that feeds the reports, so the UI and the analysis can't drift apart.

Requests measured on a loaded machine are excluded rather than reported as slow. If the sampler detects swapping or critical memory pressure while a request is running, the row is flagged with an `excluded` reason. It stays visible in the trace view for debugging but never reaches an average, and the filter lives in SQL so no chart can forget to apply it. On a 16 GB machine this happens regularly; 10 of the 43 requests recorded so far are excluded, 3 of them for swapping.

Time to first answer token is tracked separately from time to first token. Reasoning models stream their internal monologue before the reply, so TTFT can come in under a second while you wait half a minute to read anything useful. TTFAT measures the delay that actually matters, and it's the figure the UI leads with.

## Results so far

| Model | tok/s | Time to answer | Peak RAM | Cold start |
|---|---:|---:|---:|---:|
| llama3.2:1b | 198.2 | 0.20 s | 5.2 GB | 78× |
| llama3.2:3b | 91.4 | 0.35 s | 2.7 GB | 8.0× |
| gemma3:4b | 60.6 | 0.48 s | 6.1 GB | 10.9× |
| qwen3.5:9b | 41.7 | 0.53 s | 6.5 GB | — |
| qwen3.5:4b | 38.7 | 0.54 s | 6.2 GB | 14.0× |
| deepseek-r1:8b | 23.0 | 72.9 s | 6.1 GB | — |

Sample sizes are small and each row comes from a handful of prompts, so treat these as early signal. The suite runner that produces proper confidence intervals is the next phase. A few results stood out:

- **Reasoning cost 27× the latency for an identical answer.** `qwen3.5:4b` solved an arithmetic question in 1.4 seconds with thinking disabled and 39 seconds with it enabled, returning `$14` both times.
- **`deepseek-r1:8b` ignores `think=False`.** It generates roughly 1,672 tokens either way, so "disable reasoning to go faster" doesn't transfer between models.
- **A 2 GB model needed 17.8 GB at full context.** `llama3.2:3b` grows from 2.31 GB resident at 2k context to 17.77 GB at its advertised 131k limit, which is more than this machine has. File size on disk says very little about whether a model will actually run.
- **Cold-start detection was wrong at first.** Ollama reports 137–305 ms of `load_duration` even when the weights are already resident, so an initial threshold of 100 ms flagged every request as cold. Residency is now read from `/api/ps` before the request starts, and `localslm repair` re-scores the older rows.

Some models in the table have since been deleted locally. Their measurements survive because the `requests` rows outlive the weights, which is deliberate.

## Getting started

Requires [Ollama](https://ollama.com) running locally and Python 3.13.

```bash
make install     # virtualenv, Python deps, npm deps
make doctor      # check Ollama, disk, memory and power source
make serve       # API on :8420, UI on :5273
```

The ports are non-standard because 8000 and 5173 tend to already be taken by other local projects.

## Architecture

```
  ui/ (React + Tailwind + recharts)
   │  SSE, same-origin via the Vite proxy
   ▼
  api/routes/{health,models,chat,stats}          FastAPI
   │
   ▼
  inference/session.py ──── where a generation is timed and persisted
   │         │        │
   │         │        └──► telemetry/sampler.py   one thread, ring buffer;
   │         │                                    requests take slices of it
   │         └──────────► runtimes/base.py        the Runtime protocol
   │                        └─ runtimes/ollama.py    (the only adapter so far)
   ▼
  db.py ── one SQLite file ── read by the UI, the eval harness and the report
```

The database is a single SQLite file accessed through raw `sqlite3`, with no ORM and no migrations. The UI, the benchmark runner and the report generator all read the same rows, which avoids the situation where a benchmark writes its own output format and the UI aggregates something slightly different, eventually producing a published number that can't be reproduced in the inspector.

Runtimes sit behind a protocol so the benchmarking layer never talks to Ollama directly. Ollama now serves MLX-format weights alongside GGUF, which makes comparing runtimes on Apple Silicon a matter of changing a model name. Adding llama.cpp as a native backend later means implementing one file.

Chat requests take a shared lease on the inference engine while benchmarks take an exclusive one. A chat message arriving mid-benchmark would compete for the same GPU and unified memory and inflate every latency in the run without leaving any trace in the data, so chat returns a 409 while a benchmark is active.

## Project structure

```
config/app.yaml          behaviour and benchmark presets
src/localslm/
  config.py              dotted-path config with preset overlays
  host.py                machine provenance and power state
  db.py                  schema; one file, one write lock
  runtimes/              Runtime protocol and the Ollama adapter
  telemetry/sampler.py   background sampler and ring buffer
  inference/             measured session and the engine lease
  api/                   FastAPI routes
  cli.py                 doctor · models · repair
ui/src/                  React app: Chat, Models, Stats
tests/                   offline by default; -m integration needs Ollama
```

## Roadmap

- [x] **Phase 1** — measurement spine, chat, model management
- [x] **Phase 2a** — conversation history, per-model history, charts
- [ ] **Phase 2b** — test suites, graders, matrix runner, bootstrap confidence intervals
- [ ] **Phase 3** — context scaling, memory envelope, thermal drift, energy, needle-in-a-haystack
- [ ] **Phase 4** — reasoning economics, quantisation ladder
- [ ] **Phase 5** — LLM-as-judge, blind arena, generated technical report

## Tests

```bash
make test       # offline: no Ollama, no network, no models required
make test-all   # adds tests that need a live server
```

The offline suite replaces the runtime with a fake that replays recorded chunks at controlled timestamps, which makes the timing assertions deterministic. It verifies that reasoning tokens are never counted as answer tokens, and that a request measured during swapping is excluded rather than averaged in.
