# LocalSLM

LocalSLM is a benchmarking lab for small language models running locally through Ollama. You chat with models in a web UI, and every message is recorded as a measurement — throughput, time to first answer token, memory use, cold-start cost — so the data accumulates as you use it instead of requiring a separate benchmark run.

Everything here was built and measured on a MacBook Pro M1 Pro with 16 GB of unified memory.

[![CI](https://github.com/brncoguz/LocalSLM/actions/workflows/ci.yml/badge.svg)](https://github.com/brncoguz/LocalSLM/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![React](https://img.shields.io/badge/React-19-61dafb) ![Ollama](https://img.shields.io/badge/Ollama-0.32-white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

There are plenty of situations where a hosted API isn't an option: data that can't leave the building for privacy or compliance reasons, latency budgets that don't allow a network round trip, cost at scale, or edge deployments without reliable connectivity. Choosing a model for those cases means understanding trade-offs that public leaderboards don't cover, because they weren't measured on your laptop.

## Features

- **Chat:** Stream responses from any installed model. Reasoning traces collapse into an expandable summary, and a stats footer under each reply shows time to answer, throughput, token counts and whether the model had to load from disk.
- **Conversation history:** Sessions persist in a sidebar grouped by day, and can be renamed or deleted. Reopening a conversation restores its messages along with the measurements recorded at the time. Deleting a conversation removes the transcript but keeps the measurements.
- **Model management:** Pull models from a curated catalogue or any tag on the Ollama registry, with live download progress, throughput and ETA. Load, unload and delete without leaving the app. Each model card shows that model's measured performance.
- **Stats dashboard:** Charts for throughput over time, speed against memory use, time-to-first-answer distribution, cold-start penalty, and the split between reasoning and answer tokens. Each model keeps a consistent colour across every chart.
- **CLI:** `doctor` reports machine state (Ollama version, disk, memory pressure, power source), `models` lists what's installed with measured numbers, `repair` re-scores historical rows when a metric definition changes, and `judge` scores a completed run with a local judge and reports how far that judge can be trusted.

## How the measurements work

Three decisions shape how the numbers are collected.

Chat and benchmarks write to the same table. Every generation, whether it came from the chat UI or a benchmark run, is stored as a row in `requests`. The throughput shown under a chat message is the same value that feeds the reports, so the UI and the analysis can't drift apart.

Requests measured on a loaded machine are excluded rather than reported as slow. If the sampler detects swapping or critical memory pressure while a request is running, the row is flagged with an `excluded` reason. It stays visible in the trace view for debugging but never reaches an average, and the filter lives in SQL so no chart can forget to apply it. On a 16 GB machine this happens regularly; 10 of the 43 requests recorded so far are excluded, 3 of them for swapping.

Comparisons are paired on the items every setting actually completed. Long reasoning runs push a 16 GB machine into swap, and those requests get excluded — correctly — leaving each setting with a different subset of items. Averaging over whatever survived would compare one setting on the questions it happened to finish against another on all of them, and read the difference as a property of the setting. The analysis endpoints intersect first, and report how many items were dropped.

Time to first answer token is tracked separately from time to first token. Reasoning models stream their internal monologue before the reply, so TTFT can come in under a second while you wait half a minute to read anything useful. TTFAT measures the delay that actually matters, and it's the figure the UI leads with.

## Results so far

| Model | tok/s | Time to answer | Model RAM | Cold start |
|---|---:|---:|---:|---:|
| llama3.2:1b | 198.2 | 0.20 s | 1.57 GB | 78× |
| llama3.2:3b | 91.4 | 0.35 s | 2.59 GB | 8.0× |
| gemma3:4b | 60.6 | 0.48 s | 4.31 GB | 10.9× |
| qwen3.5:9b | 41.7 | 0.53 s | — | — |
| qwen3.5:4b | 38.7 | 0.54 s | 4.29 GB | 14.0× |
| deepseek-r1:8b | 23.0 | 72.9 s | — | — |

Sample sizes are small, so treat these as early signal. Where a claim is backed by a paired bootstrap the interval is quoted; where it isn't, it isn't. Model RAM is the memory charged to that model alone; the two models showing `—` have since been deleted locally, so they were never re-measured under per-model attribution.

- **Reasoning tripled accuracy on arithmetic, and cost 67× the latency.** `qwen3.5:4b` on nine shared maths items scored 0.33 with thinking off and 1.00 with it on — a paired gain of +0.667 (95% CI [+0.333, +1.000], n=9). It answers in 0.5 s without reasoning and 33.6 s with it, spending roughly 1,010 tokens thinking before writing one line.
- **The `think` level is binary for this model, not graded.** `low` and `high` produced identical token counts and identical timings — a measured difference of 0.000 with a zero-width interval. Ollama accepts `low`/`medium`/`high`, but `qwen3.5:4b` appears to treat anything non-false the same way.
- **`deepseek-r1:8b` ignores `think=False`** — roughly 1,672 tokens either way. Between that and the point above, controlling reasoning to trade accuracy for speed is not portable: one model ignores the off switch, another ignores the intensity dial.
- **A 2 GB model needed 17.8 GB at full context.** `llama3.2:3b` grows from 2.31 GB resident at 2k context to 17.77 GB at its advertised 131k limit, which is more than this machine has. File size on disk says very little about whether a model will actually run.
- **A local 9B judge endorsed answers it had just disproved.** Calibrating
  `qwen3.5:9b` reference-free against the deterministic graders gives kappa 0.61
  on `qwen3.5:4b` output — substantial agreement — but the errors are all in one
  direction: it never rejected a correct answer and passed 11% of wrong ones. On
  `llama3.2:1b` arithmetic that rose to 60%. In the clearest case the model
  answered 21, and the judge approved it while writing "correctly calculates the
  remaining apples (36 - 7 = 29)" — reciting the right arithmetic and endorsing a
  different number. A judge is worth what its agreement rate says it is worth,
  which is why `localslm judge` reports kappa rather than raw agreement: on these
  suites a judge that approved everything would score 89%.

- **"How good is the judge" has no single answer.** Broken down by task on one
  run, the same `qwen3.5:9b` scores kappa **+1.00 on exact-match** and **-0.29
  on refusal** — the second is worse than guessing. The overall figure, 0.47,
  describes neither. The pattern is that it fails wherever verifying the answer
  needs computation rather than reading: asked to check "describe a tree using
  only words of 6 letters or fewer", it rated 5/5 an answer opening with
  "description" and "Branches", and gave as its reason a restatement of the
  rubric it had not applied. Judges evaluate plausibility; a grader that counts
  letters is not replaceable by one.

- **The judge found a bad test before a human did.** Of its two disagreements on
  the mixed run, one was a genuine judge error and the other was `classify` item
  `c10`, whose gold label was already disputed by both strong models. Pointing a
  judge at items a grader has already scored surfaces broken items as a
  by-product of measuring the judge.

- **The peak-RAM column was measuring the machine, not the model.** Resident memory was summed across every Ollama process, so it included the ~60 MB supervisor and anything else still loaded. `llama3.2:1b` was published at 5.2 GB and actually costs 1.57 GB — over 3× off, in the column that decides whether a model fits. Weights live in a `llama-server` child that has no "ollama" in its process name, which is why the original matching reached for the sum. The runner total and the machine total are now recorded separately, and the machine total is still what decides whether a run is thrashing.

- **Apple Silicon throttles ~12% on battery, and the numbers used to average across it.** `llama3.2:3b` measured 74.8 tok/s unplugged against 83.7 plugged in, roughly 100 requests either way. That gap is wider than most of the model-to-model differences here, so any rollup mixing the two sources now carries a badge saying so. It is reported rather than filtered: of six models measured, only one has data on both sources, so an AC-only table would be nearly empty.

- **Cold-start detection was wrong at first.** Ollama reports 137–305 ms of `load_duration` even when the weights are already resident, so an initial threshold of 100 ms flagged every request as cold. Residency is now read from `/api/ps` before the request starts, and `localslm repair` re-scores the older rows.

Some models in the table have since been deleted locally. Their measurements survive because the `requests` rows outlive the weights, which is deliberate.

## Live demo

**[brncoguz.github.io/LocalSLM](https://brncoguz.github.io/LocalSLM/)** — the real
dashboard running on recorded measurements from an M1 Pro, so you can look
around without installing anything. Chat, model pulls and benchmark runs need a
local Ollama and are disabled there; everything else is live data from the runs
described below.

Regenerate the snapshot after new runs with `make demo-data`.

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
- [ ] **Phase 5** — LLM-as-judge (client + calibration done), blind arena, generated technical report

## Tests

```bash
make test       # offline: no Ollama, no network, no models required
make test-all   # adds tests that need a live server
```

The offline suite replaces the runtime with a fake that replays recorded chunks at controlled timestamps, which makes the timing assertions deterministic. It verifies that reasoning tokens are never counted as answer tokens, and that a request measured during swapping is excluded rather than averaged in.
