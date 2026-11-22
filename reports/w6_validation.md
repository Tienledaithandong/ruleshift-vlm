# Week 6 — Validation plan and status (no validation data)

**Status: PLANNED, NOT RUN.** This environment has no CUDA device and no cached
frozen-VLM checkpoint, so no method was compared on validation with real model
output. No validation returns, confidence intervals or effect sizes are
reported here. Placeholders are marked `PENDING`; nothing is a result.

## What is implemented and ready

- `research/ablation_matrix.csv` — B0–B6, M1 and five M1 ablations, with
  privilege and call-cap columns. B3 is explicitly labelled an **adapted**
  DIVA-style baseline, not an exact reproduction.
- `agents/diva_style.py` — decision-informed check valuation with physical
  probe cost, using only public costs and VLM predictions.
- `agents/self_consistency.py` — N samples, each sample counted toward the
  shared call cap; deterministic first-occurrence tie-break.
- `agents/value_selector.py` — `ablation_choose` for the five M1 ablations.
- `evaluator/metrics_ci.py` — paired bootstrap CI with the **scenario** as the
  resampling unit, never the frame or decision.
- `scripts/aggregate_results.py` — aggregator from per-scenario CSV to a paired
  summary JSON.

All of the above are exercised by unit tests with hand-specified predictions
or deterministic stubs. Those tests prove arithmetic and contracts, not model
behaviour.

## Planned validation comparison (pending real VLM)

| Comparison | Method | Baseline | Endpoint |
|---|---|---|---|
| H2 primary | M1 | B0 direct VLM | paired mean episodic return |
| H2 secondary | M1 | B3 DIVA-style adapted | paired mean episodic return |
| H2 ablation | M1 | M1_minus_immediate_only | offline decision regret |
| H1 diagnostic | all methods | exact Q* on microtasks | probe error vs Delta_probe bins |

Every method receives the same public action menu, environment step budget and
per-episode VLM-call cap (6). B4 self-consistency spends N calls per decision;
the remaining methods spend one. Privileged rows (B5, B6c–e) are segmented and
never enter a primary comparison.

Statistics: for paired scenario differences `d_i = return_method - return_base`,
report `mean(d)` and a 95% percentile bootstrap CI resampling scenarios. Do not
treat a positive difference in roughly ten episodes as evidence.

## Freeze status

`configs/preregistered_test.yaml` is `DRAFT_NOT_FROZEN`. It records the intended
schema and commitments but has no `frozen_commit`. Freezing requires a real
validation run on a real model; doing it from stub output would be invalid.

| Field | Value |
|---|---|
| test scenarios | PENDING |
| prompt versions | PENDING |
| method code hash | PENDING |
| frozen commit | null |

## Gate W6

**Not met.** The gate requires promising validation return or reduced decision
regret for M1 over both direct VLM and the DIVA-style adapted baseline. With no
real VLM run, the only defensible outcome is the pivot clause: continue as a
rigorous benchmark/diagnostic study and make no method claim. If a checkpoint
becomes available, run the validation comparison, fill the placeholders, then
freeze the test configuration and re-evaluate the gate.
