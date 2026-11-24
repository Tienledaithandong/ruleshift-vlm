# RuleShift-Grid multi-week preregistration (working copy)

**Status: DRAFT_NOT_FROZEN. This document does not authorize the 5,500-episode run.**

The human owner must review and change the status to READY in a separate,
explicit commit before any locked grid run. Until then every grid artifact is
instrumentation, not a confirmatory result.

## 1. Frozen objects

- Model: Qwen/Qwen2.5-VL-3B-Instruct, revision
  66285546d2b821cf421d4f5eb2576359d3770cd3, bfloat16, image + text,
  max_new_tokens=2048.
- Prompt: ruleshift/prompts/grid_v5.txt, version grid_v5.
- Candidate menu: base, remap_swap, remap_rotate, slippery,
  hazard_goal_swap, sign_composition.
- Grid: 100 maps x 5 run seeds x 11 scenario types = 5,500 episodes.
- scenario_id = m{map_seed}-s{run_seed}-{scenario_type}-{initial_rule}-{shift_rule}.

Any change to the model revision, prompt file, candidate menu, grid config, or
scenario construction requires a new freeze manifest and invalidates earlier
chunks.

## 2. Chunking and coverage

- A Kaggle chunk is one map index x five seeds x eleven scenarios = 55 episodes,
  identified m000 through m099.
- A weekly slice is ten consecutive chunks (ten maps, 550 episodes). The slice
  is split across five 55-episode Kaggle sessions because one session cannot
  hold 550 episodes at the measured rate.
- Measured feasibility (2026-11-24, real T4 smoke): mean 8.85 s per model
  decision over 119 calls. A worst-case 55-episode chunk (60 decisions each) is
  ~8.1 GPU-hours; ten chunks per weekly slice is ~81 GPU-hours, and the full
  grid is ~810 GPU-hours before overhead. That is far beyond the ~27 h/week
  quota, so the current 100x5 design cannot complete as specified. Do not mark
  this READY or start the full grid until throughput improves materially or the
  design is resized and re-reviewed.
- Chunks are run in map order. No optional stopping: every one of the 100
  chunks must be attempted and either merged or explicitly declared failed.
- A failed chunk may be rerun under the same mNNN id only after preserving the
  failed artifacts outside the merge directory. Never silently replace them.

## 3. Measurement contract

Every public decision must contain all six candidate rule IDs exactly once and
probabilities summing to 1.0 within 0.02. Missing IDs, unknown IDs, all-zero
vectors, non-finite values, out-of-range values, and sums outside tolerance are
schema failures.

A policy fallback is not a valid decision and is recorded separately. The merge
rejects any chunk containing a non-full-menu row or a fallback row. It also
rejects duplicate scenarios, missing evaluator rows, an advertised scenario that
is absent, and baselines that do not cover the same scenario set.

## 4. Outcomes and analysis

- Primary outcome: total_cost (environment steps + advisor calls x 2).
- Secondary: success, return, environment steps, tool units, adaptation steps,
  false-alarm rate, detection rate, and calibration (Brier/ECE) where defined.
- Comparisons: grid_vlm versus each public CPU baseline
  (no_adaptation, public_advisor, random_probe, info_gain).
- Paired intervals resample scenarios, never seeds or frames. Seeds are averaged
  within a scenario before the paired difference is formed.
- The interval is a percentile bootstrap over scenario-level mean differences;
  report the point estimate, 95% interval, scenario count, and episode count.
- No result is promoted from a smoke or dry chunk. Only the complete 5,500-row
  merge can answer the preregistered question.

## 5. Freeze manifest

The canonical manifest has exactly these keys:

commit, model_revision, config_sha256, scenario_count, scenario_list_sha256.

Generate it with python scripts/grid_freeze_manifest.py. It contains no
timestamp, so regenerating it from the same commit and config must be
byte-identical. Attach the manifest beside the merged summary.

## 6. Stopping and deviations

- Stop and repair the pipeline if the two-episode smoke fails the behavioral
  gate (fewer than two distinct actions, no movement, invalid JSON, or any
  fallback).
- Stop and repair if a 55-episode dry chunk is incomplete or contains a merge
  rejection.
- Any deviation from this document must be recorded in the run log before the
  affected chunks are merged. Post-hoc changes to the primary outcome, candidate
  menu, or resampling unit are prohibited.
