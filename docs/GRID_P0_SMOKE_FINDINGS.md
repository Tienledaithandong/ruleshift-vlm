# Grid P0 smoke findings (2026-11-24)

This note records what the first real frozen-VLM smoke runs actually showed. It
is a debugging log, not a scientific result.

## Run 1: `grid_v3` (smoke-9, source commit `99b285e`)

- 2 episodes, 120 decisions, 120 parser fallbacks.
- The report said `fallback rate = 0.0000` because the runner never copied
  `last_fallback` / `last_parse_errors` into the public row. The policy row had
  the fields; the episode row did not. That accounting bug is now fixed in
  `grid_runner.py` and pinned by a runner test.
- The raw Qwen responses were complete six-id JSON objects, but their
  probabilities summed to 1.05 or 1.15, so strict validation correctly rejected
  them. Every rejected decision fell back to `UP`, which produced the reported
  constant-action result.

## Run 2: `grid_v4` (smoke-9 version 2, source commit `970d51f`)

- The arithmetic instruction worked: 68 model responses all summed to 1.00 and
  all 69 decisions were full-menu JSON-valid with zero fallbacks.
- The behavioral gate still failed: every raw action and every chosen action was
  `LEFT`. The prompt had supplied an exact uniform vector and a concrete example
  action, and the 3B model copied them instead of reading the frame.
- The public history also said `LEFT -> move` for every blocked step. The
  environment only wrote `move` after a movement action; it never distinguished
  a wall/closed-door no-op. A policy could not learn from `blocked` because the
  word never appeared.

## Corrections carried by `grid_v5` (source commit `d98cf4c`)

- `grid_env.py` now records `blocked` when a movement action leaves the agent in
  the same cell, while preserving `goal` / `hazard` / interaction outcomes.
- `grid_v5.txt` makes the last history outcome operational: after a `blocked`
  entry, the same action is forbidden for the next decision; after `move`, the
  model should continue only when the direction still reduces distance to the
  target. It also states that the step-0 uniform vector is obsolete once any
  history exists.
- The staging helper writes each chunk map index into the kernel script because
  Kaggle metadata cannot set `MAP_INDEX`.

## Run 3: `grid_v5` (smoke-12, source commit `893b36f`)

- 2 episodes, 69 public rows, 68 model calls, one cache hit.
- All 69 rows were full-menu JSON-valid; zero parser fallbacks; mean output
  tokens 100.
- Chosen actions were `DOWN` 55 and `LEFT` 14; 69 moving steps. The first
  episode still ran to the 60-step cap, but the policy no longer collapsed to a
  single action after a wall hit. The behavioural gate passed.
- Report: `kaggle_outputs/smoke12/reports/grid_real_pilot.smoke.md`. This is a
  scaffold run, not a scientific result.

## Chunk 0

After the technical gate passed, the 55-episode dry chunk for map 0 was launched as
`tiennamlee/ruleshift-grid-chunk-m000` (the title-derived slug, not the
`...-000` id written by the first helper revision). It completed successfully:

- 55 episodes, 55 unique scenarios, 3,086 public decision rows.
- Every row full-menu JSON-valid; zero parser fallbacks; mean output tokens 100.
- 2,143 model calls and 943 cache hits (30.6%).
- Actions: `DOWN` 2,586, `LEFT` 486, `UP` 14; 372 actual moving steps.
- 6/55 episodes succeeded; mean total return -5.93. The low success rate is not
  evidence at this scale, but it does show the policy is still weak.
- Technical gate: **PASS** (JSON/fallback/action diversity). Semantic analysis:
  **FAIL** (see `docs/GRID_CHUNK0_ANALYSIS.md`). Wall time 22,996 s
  (6.39 GPU-hours).
- Report: `reports/grid_chunks/grid_real_pilot.m000.md`. This remains a dry
  chunk and cannot be promoted to a result.

## Run 4: `grid_v6` (smoke-13, source commit `6a8826c`)

- 2 episodes, 120 public rows. The technical gate failed: 13/120 JSON-valid
  (10.8%), 107 parser fallbacks, 0/1 base-rule success.
- The failures were strict-schema omissions, not malformed JSON. Of 33 unique
  responses, 23 omitted only `base` and 7 omitted `base`, `hazard_goal_swap`
  and `sign_composition`; every omitted key is a required candidate id.
  The model is still dropping low-probability candidate keys even though the
  prompt says all six are required.
- The semantic gate also failed: only 1 unique belief vector, 31/64 blocked
  transitions repeated, 0 tool calls, 0/1 base success. The 23 guard
  activations broke some immediate wall loops, but cannot repair an invalid
  belief menu.
- Report: `kaggle_outputs/smoke13/reports/grid_real_pilot.smoke.md`. This is a
  scaffold run, not a scientific result.

## Run 5: `grid_v7` (smoke-14, source commit `6dc75fe`)

`grid_v7` adds an explicit candidate-id line and states that every id,
including `base` and zero-valued ids, must be emitted. The smoke-14 run is
the next 2-episode check. No result is recorded here until it completes.

## Measured cost

The 119 cached responses from run 1 averaged 8.85 s per decision on a T4
(median 8.89 s). The completed 55-episode chunk took 6.39 GPU-hours, so the
earlier 8.1 h upper bound was conservative but the order of magnitude held.
A 5,500-episode grid is therefore roughly 640 GPU-hours before overhead, about
24 weeks at 27 h/week. Even one ten-chunk weekly slice is ~64 GPU-hours, more
than two weeks of quota. This is a feasibility blocker independent of the
behavioural gate.

## Status

Gate A (two-episode smoke) and Gate B (55-episode dry chunk) pass their current
**technical** checks. Gate B is not a behavioural pass: the follow-up analysis
found a constant uniform belief vector, a fixed `LEFT`/`DOWN` action prefix,
97.4% immediate repetition of blocked actions, zero use of `RIGHT` or tools,
and 0/10 success on base-rule episodes. See `docs/GRID_CHUNK0_ANALYSIS.md`.

This chunk must not feed a larger run and is not evidence. The full grid is
still not authorized: `preregistered_test.yaml` remains `DRAFT_NOT_FROZEN`, and
the measured 640 GPU-hour cost exceeds the remaining quota. The next step is to
repair the observation/action contract and add semantic gates, then re-run only
a 2–3 episode smoke. Design resizing comes after that, not before.