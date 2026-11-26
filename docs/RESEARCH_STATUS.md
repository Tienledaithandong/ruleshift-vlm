# Research checkpoint · 2026-11-25

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

## 2026-11-25 · P0 freeze readiness (static identification)

The active identification claim now has a separate evaluation contract. The
[Evaluation protocol v1](EVALUATION_PROTOCOL_V1.md) distinguishes task success,
independent terminal MAP, explicit settled declarations, base preservation,
tool spend and zero-tool hazard/sign evidence. Unresolved episodes stay in the
unconditional declaration denominator, and terminal MAP is not substituted for
a declaration. Shift and no-shift endpoints are deferred.

The final candidate is repaired v0.4 at policy commit `e65e191`. The only
post-baseline change is declaration instrumentation at the existing safe-commit
point. The pre-instrumentation fixture covers 300 exposed static scenarios per
method; all 600 action traces, outcomes, estimates, steps and tool units match.
The [declaration instrumentation report](DECLARATION_INSTRUMENTATION.md) records
the contract and equivalence evidence.

The [legacy metric audit](METRIC_AUDIT.md) recomputed existing development rows
without rerunning a policy. On the 300 static maps 0-9 episodes, v3 has 287
task successes and 297 independent MAP matches; v4 has 298 and 297 respectively.
The all-550-row counts (v3 534/544, v4 546/543) mix in shift rows and are not
identification rates. The historical base endpoint remains 49/50 for v3 and
50/50 for v4 on that exposed bank.

The frozen candidate manifest covers maps 210-239 and seeds 5-9: 900 static
entries, 30 unique public layouts, zero collisions with maps 0-199. Its
authoritative content SHA-256 is
`95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`.
No candidate-map episode has been run. The [freeze record](FREEZE_MANIFEST_IDENTIFICATION_V1.md)
and [data exposure audit](DATA_EXPOSURE_AUDIT.md) document validation and the
post-freeze `results/` scan.

The final exposed development run gives the candidate 298/300 task successes,
297/300 terminal MAP, but only 263/300 correct explicit declarations = 87.67%,
with 4 wrong and 33 unresolved. Its sign-composition declaration rate is 44/50
= 88%. These exposed numbers are below the proposed 95% overall and 90% sign
gates, so the candidate may fail its primary identification gate on the frozen
bank. That risk is disclosed, not tuned away. Base task success is 50/50 on the
development base subset; the confirmatory base endpoint is 150 episodes.

The [preregistration](PREREGISTRATION_IDENTIFICATION_V1.md) fixes the 2,700-
episode static design, H1/H2 (v4 and v3 vs no_adaptation, paired terminal MAP,
map-level Holm family, 5-point minimum effect), the absolute G1-G6 gates, the
fail-closed zero-tool rule and the descriptive-only v4−v3 contrast. The
[P0 decision packet](P0_DECISION_PACKET.md) records all nine human decisions,
the candidate and manifest hashes, the development gate risk and the exact GO
condition. Raw development artifacts are archived with checksums under the
ignored `results/archive/p0_2026-11-25/`.

Status: **freeze-ready for a human GO/NO-GO; default NO-GO.** No confirmatory
holdout, candidate tuning, VLM/GPU inference, merge or new draft PR was
performed in this step.
