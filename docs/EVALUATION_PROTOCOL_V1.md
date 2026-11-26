# Evaluation protocol v1: static Grid identification

Status: measurement protocol frozen with the static-only preregistration
(`docs/PREREGISTRATION_IDENTIFICATION_V1.md`). This document defines fields and
estimands; the preregistration defines the experimental design and decision
rules. Neither is a result.

Scope: active rule identification on the six static initial rules. Shift and
post-shift adaptation fields are retained for compatibility but are not part of
the frozen design.

## 1. Unit and denominators

The static unit is one episode identified by `(map_id, run_seed, initial_rule)`.
The design has 30 maps × 5 seeds × 6 rules = 900 static episodes per method. The
map is the independent cluster for paired uncertainty.

No episode is dropped for being unresolved, wrong, unsuccessful, truncated or
API-unavailable. Every rate states its denominator. The primary rates use all 900
static episodes. The base endpoint uses the 150 base episodes. The hazard and sign
endpoints each use 150 episodes of that rule.

## 2. Task success

`task_success` is the environment's terminal task outcome, independent of the
rule estimate. A task success with a wrong rule is a successful misdiagnosis, not
an identification success. A task failure with a correct MAP is a task failure,
not a declaration failure by itself.

## 3. Terminal MAP

At termination, score the terminal posterior's highest-probability rule against
the terminal true rule. Static episodes have the same true rule throughout, so
the initial rule is the terminal true rule. The terminal MAP is a posterior
snapshot. It is **not** an explicit declaration and must not be described as
confirmed active identification.

For paired comparisons, terminal MAP is the endpoint because no_adaptation has no
declaration API. The paired result answers whether terminal behaviour differs, not
whether either policy made an explicit identification.

Ties are unresolved and are never scored as correct. Posterior validity requires
the full six-rule menu, finite probabilities, nonnegative values and a total
within tolerance of 1.0. Invalid posteriors are reported, not repaired.

Brier is the conventional multiclass sum of squared class errors against the
terminal true rule. NLL is the negative log probability assigned to that rule.
Both require a valid posterior; otherwise they are unavailable.

## 4. Explicit declarations

A declaration event has `step`, `declared_rule`, `confidence`,
`posterior_margin` and a source label. It is valid only when:

- the step is within the episode horizon and steps are nondecreasing;
- the rule is one of the six known rules;
- confidence and margin are finite and in [0, 1];
- the margin is not negative.

A declaration is **settled** when confidence ≥ 0.90 and margin ≥ 0.10, with ties
at the top treated as unresolved. The final settled declaration is scored. An
unsettled final declaration is `unresolved`; a settled wrong rule is
`wrong_commit`; a settled correct rule is `correct_commit`; no API is
`unavailable`.

Statuses are mutually exclusive per episode. Unconditional declaration accuracy
is:

    correct_commit / total static episodes

Unresolved, unavailable, wrong and earlier-wrong cases remain in that denominator.
Event coverage is the fraction of episodes with at least one valid declaration
event. Settled coverage is the fraction with a settled declaration. API
availability is the fraction whose policy exposes the API. These are three
different quantities and are reported separately.

Earlier wrong declarations are counted even when a later declaration is correct.
A final correct declaration does not retroactively erase the error.

No terminal MAP, fallback plan, action histogram or post-hoc rule preference may
be converted into a declaration.

## 5. Base, hazard and sign endpoints

The base task gate is the 150 base episodes per method: observed task successes
must be 150/150 for the candidate. It is an engineering readiness gate, not a
population reliability claim. Report the one-sided 95% Clopper–Pearson upper bound
with zero failures and the observed count. Do not relax it after seeing candidate
maps.

Hazard (`hazard_goal_swap`) and sign (`sign_composition`) identification gates use
the 150 episodes of that rule and the unconditional explicit-declaration accuracy
defined above. The candidate requirement is at least 90.0% for each. The
development 50-episode subsets are not the gate.

The zero-tool hazard/sign gate is fail-closed: any raw candidate task success on
a hazard or sign episode with zero tool units blocks the gate. An evidence-coding
exception is not approved in v1. If one were ever approved, it would require a
written coding rule and independent verification, and both raw and
evidence-adjusted counts would still be reported. The adjusted count cannot
rescue a raw failure.

## 6. Tool accounting

Tool units are the sum of READ_SIGN and INSPECT costs using the configured costs.
The cap is 3 per episode. Budget violations are counted; a violation is not
silently clipped. Report total, mean, maximum, zero-tool episodes and zero-tool
successes separately for all static episodes and for hazard/sign. Zero-tool
successes are task successes, not declarations.

## 7. Static and shift separation

Static identification, shift detection and no-shift behaviour are different
endpoints. The frozen experiment uses static identification only. The runner and
record schema retain shift fields for historical compatibility, but no shift or
no-shift episode is part of the 900-per-method design.

`actual_shift_time`, `belief_shift_event_time`, latency, censoring and false-alarm
fields are not used to adjust static accuracy. A shift-detection false alarm on
a static episode, if observed in a deferred design, is reported separately and
never counted as a static declaration error. Conversely, a static success is not
evidence of shift adaptation.

## 8. Aggregation

Records are aggregated by method and endpoint without exclusions:

- `static_identification`: all 900 static episodes;
- `base_identification`: 150 base episodes;
- `by_initial_rule`: 150 episodes per rule;
- success, MAP, joint task+MAP, successful misdiagnosis, task-failure-with-correct-
  MAP, declaration statuses, earlier-wrong counts, Brier, NLL and tool fields;
- the all-row and no-shift blocks remain available for historical comparison but
  are not confirmatory endpoints.

The paired analysis requires the exact same scenario-id set for both methods.
Missing, duplicate or extra ids abort the analysis; there is no intersection or
implicit fill.

## 9. Map-level paired analysis

For each map, average the binary outcome over its 30 static episodes. The paired
effect is the equal-map mean of the 30 within-map differences. Report:

- effect and its 95% percentile bootstrap interval over maps (20,000 resamples);
- one-sided sign-flip randomization p-value over maps (20,000 permutations);
- exact sign-test sensitivity counts;
- maps positive, zero and negative;
- raw and Holm-adjusted p-values for H1/H2 only.

The percentile bootstrap and randomization test are implemented in
`src/ruleshift/evaluator/grid_map_analysis.py`. The default seed is
`20261125`; the exact seed and sample counts are recorded in the export.

Episode-level exact McNemar tests are supplementary. They treat episodes as
independent and can be anti-conservative under shared maps; they never replace the
map-cluster interval and are not confirmatory.

## 10. Multiplicity and gates

Only H1 (v4 > no_adaptation terminal MAP) and H2 (v3 > no_adaptation terminal MAP)
are in the Holm family. The candidate-vs-v3 contrast is descriptive. Gates G1–G6
are deterministic engineering criteria and are not p-valued; they are not added
to Holm. Brier, NLL, calibration, tool counts, base MAP and declaration
subcomponents are descriptive unless a gate explicitly names them.

## 11. Manifest and immutability

The confirmatory manifest is explicit, content-hashed and geometry-hashed. The
frozen manifest is:

- path: `artifacts/manifests/identification_v1_static.json`;
- content SHA-256: `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`;
- entries: 900 static episodes, maps 210–239, seeds 5–9, six rules;
- 30 unique layouts, zero collisions with maps 0–199.

Every episode record carries the manifest hash, config hash and code commit.
Duplicate scenario ids, stale hashes, layout mismatches, unknown methods and
missing paired scenarios fail closed. The working-tree file hash is not the
scientific identity because line endings vary; the content hash is.

## 12. No retrospective scoring

A result is not relabelled after seeing it. Old outputs retain their original
fields and thresholds. A metric correction must be documented and applied before
the frozen candidate run; it cannot be introduced after seeing candidate-map
outcomes. If a bug is found after execution, the confirmatory run is invalid and
must be re-frozen, not patched.

## 13. P0 completion

P0 is complete for review when metric definitions, manifest validation, declaration
instrumentation, leakage checks, behavioural equivalence, archive checksums and
regression tests pass. P0 completion does not authorize candidate-map execution.
That requires the separate human GO in the preregistration.
