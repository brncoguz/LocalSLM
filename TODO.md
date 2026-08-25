# TODO

Working list. Ordered so each item ships something checkable on its own.

## Phase 2b — close the gaps

- [x] **2b.1** `POST /api/suites/from-session/{id}` — turn a chat session into a suite JSONL
- [x] **2b.2** Chat UI: "Save as suite" action on a conversation
- [x] **2b.3** `GET /api/stats/disagreements` — items where models disagree, or all fail
- [x] **2b.4** UI: failure gallery reading that endpoint
- [x] **2b.5** Expose the temperature axis in the bench API and UI
- [x] **2b.6** Determinism metrics: exact-duplicate rate, output-length variance
- [x] **2b.7** UI: determinism panel (t=0 vs t=0.7)

## Phase 4 — reasoning economics + quantisation

Ahead of phase 3: the `think` axis is already wired end to end, so this is the
cheapest path to the project's strongest finding.

- [x] **4.1** `GET /api/runs/{id}/thinking` — accuracy vs thinking tokens per think level
- [x] **4.2** `ThinkingCurve` chart
- [x] **4.3** Overthink rate metric (thinking tokens > 3x answer tokens)
- [x] **4.4** TTFT vs TTFAT paired bars — the visual that sells the thesis
- [ ] **4.5** Pull `-q8_0` and `-bf16` for the quantisation ladder (`qwen3.5:4b` re-pulled)
- [ ] **4.6** Run the ladder and add a quality-vs-speed-vs-precision view

## Phase 3 — systems probes

- [x] **3.1** Fix per-runner memory attribution — was summing every Ollama
      process, which is why `llama3.2:1b` read 5.2 GB. Runner and machine
      totals are now separate; `llama3.2:1b` measures 1.57 GB.
- [ ] **3.2** `probes/context_sweep.py` — TTFT and prefill rate vs prompt length
- [ ] **3.3** `probes/memory_envelope.py` — resident RAM vs `num_ctx`, "what fits"
- [ ] **3.4** `probes/cold_warm.py` — load penalty, properly repeated
- [ ] **3.5** `probes/sustained.py` — 10-minute thermal drift
- [ ] **3.6** `probes/multiturn.py` — prompt-cache speedup, turn 5 vs turn 1
- [ ] **3.7** `needle` suite + grader — depth x length haystack
- [x] **3.8** Battery-vs-AC axis, plus a guard against averaging the two together
- [ ] **3.9** Charts: context curve, memory envelope, thermal drift, needle heatmap

## Phase 5 — judging, arena, report

- [x] **5.1** Judge client — local `qwen3.5:9b`, measured as `kind='judge'` with
      `parent_id` pointing at the request it graded. Claude backend refuses
      loudly rather than falling back; no SDK or key here to verify it against.
- [x] **5.1b** `localslm judge <run_id>` + `GET /api/runs/{id}/judge-agreement`
      — reference-free calibration against the deterministic graders, reporting
      Cohen's kappa rather than raw agreement
- [x] **5.1c** `JudgePanel` on the run detail — kappa against the Landis-Koch
      bands, a 2x2 confusion grid colouring the two error directions
      differently, and the disagreements with the judge's own reasoning
- [x] **5.1e** Verdict browser — every judgement with the answer it graded, both
      verdicts, the raw reply and the judge's own prompt, plus a rating-spread
      chart. Evaluating the judge needs its reasoning, not just its score.
- [x] **5.1d** Trigger judging from the UI — `POST /runs/{id}/judge` plus an
      SSE progress stream, mirroring how benchmarks start. Batches of 40, since
      a 9B judge is ~2.5 s per item and a ten-minute button feels broken
- [ ] **5.2** Rubric grader for open-ended answers — grader written, registered
      and tested, but not yet usable in a run: there is no open-ended suite, and
      nothing calls the judge during grading. Wire it as a **post-run pass**,
      not inline — swapping a 9B judge in and out per item would thrash a 16 GB
      machine, for the same reason `order_units` blocks by model.
- [ ] **5.3** Pairwise comparison with order swapping + Bradley-Terry Elo
- [ ] **5.4** Arena UI: blind side-by-side with voting
- [ ] **5.5** `prompt_injection` suite + grader
- [ ] **5.6** `probes/long_convo.py` — recall vs conversation depth
- [ ] **5.7** Report generator — markdown + self-contained HTML, dependency-free SVG

## Bench UI

- [x] The item browser works on finished runs too — `GET /runs/{id}/units`
      replays a run in the same shape the live stream sends, so one component
      serves both and they cannot disagree about what happened.
- [x] Suite and label tags on a run card filter the run list. Required
      un-nesting the header: tags inside the expander button were buttons
      inside a button, which is invalid and routes clicks unreliably.
- [x] Clickable heartbeat — every square opens that item in the detail card,
      with prev/next stepping and a "next fail" jump. Pinning is keyed on the
      unit's index rather than its position in the strip, which only keeps a
      recent window.
- [x] Live run view — the unit event now carries the question, the answer, the
      expected value and whether the measurement survived, so a run is watchable
      rather than a counter. Progress with ETA, a per-item heartbeat strip,
      a running per-cell scoreboard, and the current question with its answer.
- [x] Request estimate now mirrors `BenchSpec.cells()` — it multiplied by think
      levels regardless of capability, so selecting four levels for a
      non-thinking model promised 84 requests and ran 20. The form also says
      up front which models will have their think levels dropped.
- [x] Rerun button on a run card — repeats the stored plan exactly. Required
      adding `temperatures` to `summarise_plan`, which the plan had been
      omitting, so a repeat would silently have dropped that axis.

## GitHub automation

- [x] CI — ruff, offline pytest, UI typecheck/build on push and PR
- [x] MIT license, README badges, 9 repo topics, homepage URL
- [x] Dependabot (pip, npm, actions) + CodeQL (python, js)
- [x] Static demo on GitHub Pages, fixture captured via `make demo-data`
- [x] Nightly macOS integration tests against a real Ollama — correctness only
- [ ] Publish the phase-5 technical report to Pages alongside the demo
- [ ] Refresh `ui/src/fixtures/demo.json` whenever notable new runs land

## Judge findings

- [x] ~~One kappa for the judge~~ — agreement is now reported per task. The
      same judge scores +1.00 on `exact` and -0.29 on `refusal` in one run, so
      a single figure was averaging a usable grader with a misleading one.
- [ ] **Never judge a task a grader can verify.** The failures cluster where
      checking needs computation — arithmetic, letter counting — because a
      judge rates plausibility. Consider refusing to judge `numeric`/`ifeval`
      outright, or badging those verdicts as advisory.
- [ ] **The local judge is systematically lenient.** Across 23 reference-free
      verdicts it never rejected a correct answer, but endorsed wrong ones at
      11% on `qwen3.5:4b` and 60% on `llama3.2:1b`. Any score built on it is
      biased upward, and by an amount that depends on the model being graded —
      which is the worst shape for a bias to have.
- [ ] Re-check kappa on a larger sample; 18 items is thin for a rate.
- [ ] Try a reference-based variant as a second axis: the gap between
      reference-free and reference-given is a measurement of how much of the
      judge's verdict is verification rather than plausibility.

## Known issues

- [ ] **Long reasoning runs push this machine into swap**, so units get excluded
      and each think level ends up with a different subset of items. The
      thinking endpoint now pairs on the shared subset and says how many were
      dropped, but the underlying constraint is 16 GB with ~2 GB free.

- [x] ~~Thinking runs were capped at 2048 tokens, truncating `think=high`
      mid-thought so it scored zero for running out of room~~ — now 8192 for
      thinking cells, and truncation is flagged as its own metric

- [ ] `classify` item `c10` has a contested gold label — both strong models say
      neutral, the suite says positive. The judge independently sided with the
      models on 2026-08-25, which is now three votes against the suite.
- [ ] `grounded_qa` may be too easy: hallucination rate is 0.00 for every model,
      and the prompt explicitly offers an "out"
