# LocalSLM

A lab bench for small language models running entirely on your own machine.
Chat with them, and measure them properly while you do.

```
  Apple M1 Pro · 16 GB    1.8 GB free · runner 3.6 GB · warn    181 GB free    AC    ollama 0.32.9

  qwen3.5:4b  Q4_K_M · 3.4 GB · warm

  > A shop sells pens at 3 for $2. How much do 21 pens cost?

  $14

  1.43 s to answer · 46.7 tok/s · 26 in · 4 out
```

The same prompt with reasoning enabled takes **39 seconds** and burns **1,376 tokens**
to reach the same `$14`.

That gap is the reason this exists.

---

## Why bother running models locally

There are real situations where a hosted API is not an option: privacy rules that
forbid the data leaving, latency budgets that can't absorb a network round trip,
cost at scale, and edge deployments with no guaranteed connectivity.

What's scarce isn't the ability to run a small model — `ollama pull` does that.
It's knowing **what it actually costs you** in speed, memory and quality on
hardware you own. This project measures that, on one specific laptop, and shows
its working.

## The measurement ethic

An instrument that lies is worse than no instrument. Three rules follow:

**Chat and benchmarks write the same row.** Every generation — a chat message or
a benchmark unit — lands in one `requests` table. The tok/s under a chat bubble
is the number the report averages. There is no second code path that could
compute it differently.

**A measurement taken on a busy machine is thrown away, not reported as slow.**
If the run swapped or hit critical memory pressure, the row is marked
`excluded` with a reason. It stays visible in the trace inspector and invisible
to every aggregate. On a 16 GB machine this fires often, and hiding it would
quietly corrupt every comparison.

**Time-to-first-token is the wrong headline for a reasoning model.** All three
models installed here stream *thinking* tokens first. TTFT can be under a second
while you wait half a minute to see a readable word. So the app measures
**TTFAT — time to first _answer_ token** — separately, and leads with it.

## Preliminary numbers

Measured on this machine, `n = 2` after discarding a warmup, single prompt.
**These are illustrative, not a study** — too few samples and one task to claim
anything with confidence. The suite runner that produces real confidence
intervals is phase 2.

*"A shop sells pens at 3 for $2. How much do 21 pens cost?"* — correct answer `$14`.

| model | think | TTFAT | output tokens | tok/s | peak RSS | answer |
|---|---|---:|---:|---:|---:|:--|
| llama3.2:3b | n/a | 0.20 s | 3 | 79.3 | — | `$15` ✗ |
| qwen3.5:4b | off | 1.43 s | 4 | 46.7 | 4.4 GB | `$14` ✓ |
| qwen3.5:4b | high | 39.25 s | 1,376 | 35.3 | 4.4 GB | `$14` ✓ |
| qwen3.5:9b | off | 2.62 s | 4 | 27.5 | 6.4 GB | `$14` ✓ |
| deepseek-r1:8b | off | 73.82 s | 1,672 | 23.0 | 6.1 GB | `$14` ✓ |
| deepseek-r1:8b | high | 72.78 s | 1,672 | 23.0 | 5.8 GB | `$14` ✓ |

Three things already stand out, all worth confirming with a real suite:

**The fastest model is the wrong one.** `llama3.2:3b` answers in 200 ms at 79 tok/s
— roughly 200× faster than `deepseek-r1:8b` — and gets it wrong. Any speed
comparison that doesn't carry a correctness column beside it is worse than
useless, which is why the leaderboard is a Pareto frontier and not a ranking.

**Reasoning cost 27× the latency for an identical answer.** `qwen3.5:4b` answers
the arithmetic question in 1.4 s with thinking off and 39 s with it on. Both say
`$14`. Whether that trade ever pays off is exactly what the reasoning-economics
suite is designed to find out — the expectation is that it does on hard multi-step
problems and doesn't on extraction or classification.

**`deepseek-r1:8b` appears to ignore `think=False`.** It generates the same 1,672
tokens and takes the same ~73 s either way. It's a distilled reasoning model, so
this is plausible rather than surprising — but it means "turn reasoning off to go
faster" is not portable advice, and a benchmark that assumed the flag worked would
silently mis-attribute the cost.

## Status

Phase 1 is done: the measurement spine, chat, and model management.

- [x] **Phase 1** — Ollama adapter, telemetry sampler, SQLite substrate, streaming chat with per-message stats, model management (pull / load / unload / delete from the app)
- [x] **Phase 2a** — conversation history, per-model history, and charts
- [ ] **Phase 2** — suites, graders, the matrix runner, bootstrap confidence intervals, report export
- [ ] **Phase 3** — systems probes: context scaling, cold/warm, sustained thermal drift, memory envelope
- [ ] **Phase 4** — reasoning economics and the quantization ladder
- [ ] **Phase 5** — judging, arena, and the generated technical report

## Quick start

Needs [Ollama](https://ollama.com) running and Python 3.13.

```bash
make install     # venv + npm deps, copies .env.example
make doctor      # check ollama, disk, memory, power source
make serve       # API on :8420, UI on :5273
```

`make doctor` prints the machine state that every subsequent number is
conditional on:

```
  machine       Apple M1 Pro · 10 cores · 16 GB
  ollama        ✓ 0.32.9 @ http://localhost:11434
  disk free     ✓ 181 GB
  memory        ✗ 1.8 GB free · swap 8.1 GB · pressure warn
  power         ✓ AC · 52%
  env hash      3226520eea439833
```

## History and charts

Conversations persist. The sidebar lists them grouped by day; reopening one
restores its messages **with the measurements they were recorded with**, because
`chat_messages` joins back to the `requests` row that produced it. Deleting a
conversation removes the scrollback and **keeps the measurements** — they are
the benchmark record, not chat history.

The **Stats** tab charts everything measured so far: throughput over time,
speed vs memory, time-to-first-answer spread, cold-start penalty, and the
thinking/answering split. Every model keeps one colour across every chart.

Two rules the charts follow. `ok = 1 AND excluded = 0` lives in the SQL rather
than in any component, so a chart cannot forget to drop a swapped run. And the
exclusion ledger is shown, not hidden — "10 of 43 measurements excluded, 3
because the machine was swapping" is a fact about the machine, and burying it
would be the exact dishonesty this project exists to avoid.

### A bug this found

The cold-start flag was wrong for every row. Ollama reports a `load_duration`
of 137–305 ms **even when the weights are already resident** — that is handle
setup, not loading — while genuine cold loads here run 1.5–12 s. The original
`load_ms > 100` test therefore marked all 23 early requests cold, which would
have quietly voided the entire cold-start analysis.

Residency from `/api/ps`, checked before the clock starts, is now authoritative;
the load-duration test survives only as a fallback with a threshold sitting in
the wide empty gap between the two clusters. `localslm repair` re-scores older
rows by the same fallback rule and shows you the diff before writing.

| model | cold start | vs warm |
|---|---:|---:|
| llama3.2:3b | 1.28 s | 8.0× |
| gemma3:4b | 3.72 s | 10.9× |
| qwen3.5:4b | 3.60 s | 14.0× |

## Managing models

Pull, load, unload and delete without leaving the app. The **Models** screen has a
curated catalog — each entry says what contrast it buys, not just how big it is —
plus a free-text box for anything else on the Ollama registry.

Pulls stream aggregate progress with live throughput and ETA. Ollama reports
progress one layer at a time, so a naive passthrough makes the bar restart on
every layer; the API sums across layers and streams the total instead.

Two guards sit on the endpoint rather than the UI, because the UI is not the only
caller:

- **Disk guard** — refuses a pull that would leave under 10 GB free. Catalogued
  models are checked before starting; anything else is checked as it downloads,
  since Ollama only reveals the size along the way.
- **Benchmark guard** — refuses while a run holds the exclusive lease. A pull
  saturates disk and CPU, and would inflate every latency in the run without
  leaving a trace in the data.

## Architecture

```
  ui/  (React + Tailwind + recharts)
   │  SSE, same-origin via the vite proxy
   ▼
  api/routes/{health,models,chat}          FastAPI
   │
   ▼
  inference/session.py ──── the ONE place a generation is timed and persisted
   │         │        │
   │         │        └──► telemetry/sampler.py   one thread, ring buffer,
   │         │                                    requests take slices of it
   │         └──────────► runtimes/base.py        the Runtime protocol
   │                        └─ runtimes/ollama.py    (the only adapter so far)
   ▼
  db.py ── one SQLite file ── read by the UI, the eval harness and the report
```

**Why one SQLite file, raw `sqlite3`, no ORM and no migrations.** Not minimalism:
the UI, the benchmark runner and the exported report must read *the same rows*.
If the benchmark wrote its own JSON and the UI aggregated something else, you
would eventually publish a number you couldn't reproduce in the inspector.

**Why `results` is a separate table from `requests`.** One generation can be
graded several ways — a JSON extraction is simultaneously a schema-validity
result and a field-accuracy result. Keeping them apart also means re-grading
after a grader bugfix never requires re-running inference, which you will want
the first time a grader is wrong.

**Why the runtime lives behind a protocol.** Ollama now serves MLX-format
weights alongside GGUF, so "which runtime is fastest on Apple Silicon" is
answerable by pulling `qwen3.5:4b-mlx` and changing a model name. A native
llama.cpp or MLX-server backend later means writing one file, not touching the
bench layer.

**Why chat holds a shared lease and benchmarks an exclusive one.** A chat request
arriving mid-benchmark contends for the same GPU and the same unified memory, and
would inflate every latency in the run without leaving a trace in the data. Chat
gets a `409` during a run instead. You cannot both chat and measure, and the
honest interface says so.

## Layout

```
config/app.yaml          behaviour + presets as dotted-path overrides
src/localslm/
  config.py              dotted-path config, presets, content fingerprint
  host.py                machine provenance + power state
  db.py                  the schema; one file, one lock
  runtimes/              Runtime protocol + Ollama adapter
  telemetry/sampler.py   background sampler, ring buffer, window slices
  inference/             the measured session and the engine lease
  api/                   FastAPI routes
  cli.py                 doctor · models · bench
ui/src/                  React app: Chat, Models
tests/                   offline by default; `-m integration` needs Ollama
```

## Ports

LocalSLM uses **8420** (API) and **5273** (UI) rather than the usual 8000/5173,
which are commonly already taken by other local projects.

## Tests

```bash
make test       # offline: no Ollama, no network, no models needed
make test-all   # adds tests that need a live server
```

The offline suite fakes a runtime that replays scripted chunks at controlled
timestamps, which is the only way to assert on timing maths deterministically —
including that thinking tokens never get counted as answer tokens, and that a
swapped window is excluded rather than averaged in.
