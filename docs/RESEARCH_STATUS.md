# Research checkpoint · 2026-11-24

The next experimental milestone in the [12-week plan](../RuleShift_VLM_Research_12_Week_Master_Plan.md)
is a clean Week 4 offline diagnostic on distinct development decisions. Week
numbers here denote acceptance criteria. Later-week modules already exist.

| Milestone | Evidence in the repository | Remaining work |
|---|---|---|
| W1 specification and literature | Public/evaluator contracts, protocol and literature matrix exist. | Reading notes cover metadata, abstracts and READMEs. Full-text comparison and the diagnosticity definition remain open. |
| W2–3 environment and reference | CPU Gymnasium tests, deterministic event streams, cost tests, finite-horizon Family A microtasks and a dev phase diagram. | Exact labels apply only within their declared microtask assumptions. The full shifting environment uses separate references. |
| W4 offline pilot | Archived Qwen2.5-VL-3B P1 inference on Kaggle. v12 parsed all decisions and always chose the same macro. Both revised designs run on CPU. | Assess choice, fallback, label coverage, costs and errors on a fresh authorized pilot. |
| W5–6 method and validation | Public selectors, adapted DIVA, ablation components, matched-budget rollout and paired-CI code. RGB request/cache plumbing, a pixel-dependence test, a validation grid and a real-backend closed-loop CLI now exist. | P0 RGB inference and real paired validation against direct and DIVA are pending. The test configuration is still a draft. |
| W7–11 experiments | Family A rollouts, Family B simulator and analytical reference, synthetic held-out fixtures, component diagnostics and stress tools exist. | Real closed-loop effects, cross-family transfer, held-out estimates and causal attribution remain unmeasured. |
| W12 paper | The master plan defines a write-up and pivot rule. | Evaluate the empirical gates before making a method or novelty claim. |

## What the archived pilot establishes

The following counts were recomputed from `decisions.jsonl` and
`response_cache.jsonl` in the local archive. Both sets identify
`TRANSFORMERS_FROZEN` and the pinned Qwen anchor. The input was text-only.

| Archived output under `kaggle_outputs/` | Decision rows | Actual calls | Cache hits | JSON/schema accepted by that runner | Executed action |
|---|---:|---:|---:|---:|---|
| `previous_attempt/results/real/` | 81 | 27 | 54 | 0/81 | 81 × fallback `WAIT` |
| `attempt_v12_artifacts/results/real/` | 81 | 27 | 54 | 81/81 | 81 × `PROBE_RED` |

The v12 cache has eight byte-distinct response texts in 27 generations.
Each request fingerprint was reused across three noise settings with
different exact probe values. The generator used a fixed 50:50 red/blue
reference belief while public histories started with a red environment and
only used waits. The reported 0.630 over-probe rate and 6.722 mean regret
therefore describe that old scoring procedure. They cannot establish H1, H2,
visual ability or adaptation to a shift. Original files and reports remain
available unchanged in the local archive. Fresh clones omit these raw files.

## Revised CPU references

| Scope | Public inputs or dev cells | Prefer probing | Prefer committing | Tie |
|---|---:|---:|---:|---:|
| W3 Family A cost/noise/lockout grid at the declared fixed belief and horizon | 27 | 3 | 20 | 4 |
| W4 controlled P1, fair card draw and reliable station | 81 | 12 | 51 | 18 |
| W4 history P1, actual Gate/Lock prefixes and noisy sensing under a possible shift | 81 | 13 | 59 | 9 |
| H1 81-cell cost / disclosed-noise / disclosed-prior design | 81 | 27 | 33 | 21 |

The W4 counts compare the best probe and best commitment at a tolerance of
1e-9. Each design has 81 distinct public requests. The history design scores a
6–8 primitive-step continuation from the reached position and remaining
lockout. Sensor reliability, shift probabilities, the posterior and action
values stay with the evaluator in this design. The table measures coverage
of decision conditions before model inference.

H1 is a fractional 81-cell design over detour cost, disclosed sensor noise and
disclosed prior, with lockout length, horizon and rule pair rotated across cells.
It is selected as 27 negative, 27 marginal and 27 positive exact-delta decisions;
the realized exact-sign counts are 27 probe, 33 commit and 21 tie because six
small negative deltas fall in the marginal bin. The prior, sensor reliability,
detour, lockout and horizon are disclosed; the rule and any shift stay hidden and
the candidate menu contains exactly the two rules in the lottery. M1 uses the
structured candidate probabilities plus public costs. Its selector learning rate
(0.5) is a free, unvalidated parameter, and M1 still ranks only the actions the
model predicted rather than a separately constrained menu.

B0 currently has an action-only prompt. M1, the immediate selector and
adapted DIVA share the outcome-prediction path. B1/B2 in the ablation matrix
are planned comparisons. Self-consistency and the five M1 ablations have
standalone implementations. Their matched-budget closed-loop comparisons
remain pending. Family B currently has an environment and an analytical
reference with a perfectly revealing probe in its single-decision abstraction.

Direct CLI execution previously picked up another editable installation on the
review machine. Scripts now run their sibling source tree. The provenance
manifest hashes canonical sources and excludes older Kaggle snapshots.

## Order of work

1. Exercise the corrected Family A references, P1 public contexts, call budgets and result accounting on CPU. Keep calibration and oracle values evaluator-side.
2. At the next explicitly approved model run, use a fresh output directory and record the prompt, input mode, checkpoint revision, source hashes and generation settings. Inspect the distribution of actions and the distinct inputs behind it. A constant policy can be a genuine result once the task and labels are sound.
3. Run a real public-RGB path for P0. Compare M1, direct and adapted DIVA on validation scenarios under the same input, call cap and primitive horizon. Select methods and prompts on validation; the plumbing and validation grid are in place, but no pixel-dependent VLM result exists yet.
4. Apply the W6 decision rule, record a frozen protocol and scenario list if warranted, and only then open the locked evaluation. Keep Family B, CUSUM and the Gemini prototype outside the current W4 claim.
