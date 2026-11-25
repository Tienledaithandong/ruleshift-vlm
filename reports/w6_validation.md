# Week 6 — Validation plan and status (no validation data)

**Status: validation pending.** The archived Week 4 Kaggle P1 pilot exercised
offline inference. A paired real-model validation comparison of M1, direct
and adapted DIVA has still to be run. See the [research checkpoint](../docs/RESEARCH_STATUS.md).

## What is implemented and ready

- `research/ablation_matrix.csv` — B0–B6, M1 and five M1 ablations, with
  privilege and call-cap columns. B3 is explicitly labelled an **adapted**
  DIVA-style baseline, not an exact reproduction.
- `agents/diva_style.py` — decision-informed check valuation with physical
  probe cost, using only public costs and VLM predictions.
- `agents/self_consistency.py` samples N structured responses with a
  deterministic first-occurrence tie-break. Its closed-loop integration and
  per-episode accounting are still pending.
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
| H2 ablation | M1 | Immediate-only selector (`ImmediateOnlySelector`) | offline decision regret |
| H1 diagnostic | all methods | exact Q* on microtasks | probe error vs Delta_probe bins |

The validation design gives each model method the same public action menu,
primitive-step budget and per-episode call cap (6). B4 must spend N calls for
N samples. Direct, M1 and adapted DIVA spend one per model decision. Symbolic
methods use zero model calls. Report privileged methods separately.

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

**Pending evidence.** Evaluate M1 against both direct VLM and adapted DIVA on
validation. Apply the preregistered pivot rule if that comparison fails. The
earlier offline pilot did not measure the validation endpoint.
