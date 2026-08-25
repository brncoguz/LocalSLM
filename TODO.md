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

- [ ] **3.1** Fix per-runner memory attribution — currently sums every Ollama
      process, which is why `llama3.2:1b` reads 5.2 GB. Blocks everything below.
- [ ] **3.2** `probes/context_sweep.py` — TTFT and prefill rate vs prompt length
- [ ] **3.3** `probes/memory_envelope.py` — resident RAM vs `num_ctx`, "what fits"
- [ ] **3.4** `probes/cold_warm.py` — load penalty, properly repeated
- [ ] **3.5** `probes/sustained.py` — 10-minute thermal drift
- [ ] **3.6** `probes/multiturn.py` — prompt-cache speedup, turn 5 vs turn 1
- [ ] **3.7** `needle` suite + grader — depth x length haystack
- [ ] **3.8** Battery-vs-AC axis, plus a guard against averaging the two together
- [ ] **3.9** Charts: context curve, memory envelope, thermal drift, needle heatmap

## Phase 5 — judging, arena, report

- [ ] **5.1** Judge client — local `qwen3.5:9b` default, Claude behind a flag
- [ ] **5.2** Rubric grader for open-ended answers
- [ ] **5.3** Pairwise comparison with order swapping + Bradley-Terry Elo
- [ ] **5.4** Arena UI: blind side-by-side with voting
- [ ] **5.5** `prompt_injection` suite + grader
- [ ] **5.6** `probes/long_convo.py` — recall vs conversation depth
- [ ] **5.7** Report generator — markdown + self-contained HTML, dependency-free SVG

## Known issues

- [ ] **Long reasoning runs push this machine into swap**, so units get excluded
      and each think level ends up with a different subset of items. The
      thinking endpoint now pairs on the shared subset and says how many were
      dropped, but the underlying constraint is 16 GB with ~2 GB free.

- [x] ~~Thinking runs were capped at 2048 tokens, truncating `think=high`
      mid-thought so it scored zero for running out of room~~ — now 8192 for
      thinking cells, and truncation is flagged as its own metric

- [ ] Peak-RAM figures sum all Ollama processes (see 3.1)
- [ ] `classify` item `c10` has a contested gold label — both strong models say
      neutral, the suite says positive
- [ ] `grounded_qa` may be too easy: hallucination rate is 0.00 for every model,
      and the prompt explicitly offers an "out"
