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

## Measured cost

The 119 cached responses from run 1 averaged 8.85 s per decision on a T4
(median 8.89 s). At up to 60 decisions per episode, a 55-episode chunk is an
upper bound near 8.1 GPU-hours, not the 2.2 h stated in the earlier runbook.
A 5,500-episode grid is therefore roughly 800 GPU-hours before overhead. This
is a feasibility blocker for the full run, independent of the smoke gate.

## Status

The `grid_v5` smoke must pass the behavioral gate before any 55-episode chunk.
As of this note, `preregistered_test.yaml` remains `DRAFT_NOT_FROZEN`; the full
grid is not authorized.