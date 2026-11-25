# Preregistration draft: identification v1

Status: DRAFT FOR HUMAN REVIEW. This is not a preregistration, not approval to
run a holdout, and not authorization for model calls, GPU use, remote inference,
candidate tuning or merging.

Evaluation version: grid-identification-v1. Protocol:
docs/EVALUATION_PROTOCOL_V1.md. Metric audit: docs/METRIC_AUDIT.md. Exposure
audit: docs/DATA_EXPOSURE_AUDIT.md.

## 1. Research question

On a frozen, previously unexposed map bank, does a declaration-capable
diagnostic policy identify the active rule more accurately than no_adaptation,
while preserving the base task endpoint and staying inside the tool budget?

Active rule identification is the primary claim. General post-shift adaptation
is out of scope.

## 2. Candidate and baselines

Candidate C is not yet frozen on this branch. To enter this design it must:

- expose an explicit declaration API with step, rule, confidence and posterior
  margin;
- emit only the six-rule public menu, with probabilities summing to one within
  the declared tolerance;
- pass the P0 contract and leakage tests;
- be committed before the manifest is frozen;
- have no map-specific or seed-specific branch, and no hidden-rule access.

Expression of interest does not freeze a candidate. The P0 branch intentionally
contains no candidate tuning.

Comparators:

- no_adaptation: the required paired negative control;
- diagnostic_belief_v4: secondary legacy baseline, reported but not allowed to
  substitute for the candidate.

A VLM arm is not part of this draft. Adding one would require a separate
protocol, model freeze and approval.

## 3. Design

Primary static cohort: six initial rules on every frozen map-seed pair.

Proposed sample for approval:

| Factor | Proposed |
|---|---|
| Maps | 210-239 (30 maps) |
| Run seeds | 5, 6, 7, 8, 9 (5 seeds) |
| Static scenarios per map-seed | 6 (one per initial rule) |
| Static episodes per method | 900 |
| Base endpoint per method | 150 |
| Hazard_goal_swap episodes per method | 150 |
| Sign_composition episodes per method | 150 |
| Full 11-scenario manifest per method | 1,650 |
| Primary methods | Candidate C and no_adaptation |
| Secondary method | diagnostic_belief_v4 |
| Full episodes, three methods | 4,950 |

The 900-static figure is the primary denominator. Shift and no-shift scenarios
are exported for separate descriptive analysis and never enter the static
endpoint.

Sample-size rationale, not a performance assumption: with 900 static episodes,
a 95% Wilson interval near 95% has a half-width of roughly 1.4 percentage points.
The hazard and sign strata have 150 episodes each; a 90% point estimate has a
Wilson half-width of roughly 4.8 percentage points. The base endpoint is 150
episodes, so the 100% gate permits zero base failures. If a smaller budget is
required, these precision claims no longer hold and the sample must be
renegotiated before execution.

## 4. Primary and secondary endpoints

Primary: correct explicit settled declarations divided by all 900 static
episodes, per method. Unresolved, unavailable and wrong are failures in this
denominator. No episode is excluded.

Secondary, always reported separately:

- task success;
- independent terminal MAP accuracy;
- task success with correct identification;
- successful misdiagnosis;
- map-correct but task-failed;
- wrong commitments and unresolved declarations;
- earlier wrong commitments;
- coverage and selective error;
- Brier score and NLL where a valid posterior exists;
- tool spend and zero-tool successes;
- base task success and terminal-base estimate;
- shift timing and censoring, outside the static endpoint.

A policy without a declaration API is commitment-unavailable and cannot pass the
primary gate by substituting MAP.

## 5. Hypotheses and gates

H1 (primary): Candidate C's settled identification accuracy is at least 95% on
the frozen 900-episode static cohort.

H2: Hazard_goal_swap identification is at least 90% (150 episodes).

H3: Sign_composition identification is at least 90% (150 episodes).

H4: Base task preservation is 100% on the 150-episode static base endpoint.
There is no 0.99 tolerance and no substitution of terminal base estimate.

H5: Candidate C has a positive paired terminal-MAP gain over no_adaptation on
the same 900 ids, with a two-sided exact McNemar p < 0.05 after Holm. This is a
secondary paired check, not the primary commitment gate: no_adaptation exposes
no declaration API, so a paired commitment gain cannot be computed without
inventing declarations.

H6: No accidental zero-tool hazard or sign success. The default is fail-closed:
any raw zero-tool success on hazard_goal_swap or sign_composition blocks the
gate. An exception requires a separately approved, public-observation-only
evidence-coding rule; no such rule is part of this draft.

H7: Tool spend is at most 3 units on every episode.

A gate is failed, not waived, if its required field is unavailable. The
historical base result remains 49/50.

## 6. Statistical analysis

- Pairing uses exact scenario-id equality. Missing, extra and duplicate ids
  abort the analysis; intersection is forbidden.
- H1-H3 and the base rate use point estimates with Wilson 95% intervals.
- H5 uses an exact two-sided McNemar test on paired binary identification,
  reporting both discordant counts and the exact p value.
- A cluster bootstrap over maps may be reported as a sensitivity interval. It
  resamples maps, never seeds or frames.
- Multiplicity: Holm correction across the tested identification and paired
  hypotheses. The base gate is a deterministic 150/150 criterion, not a
  p-value.
- Unresolved episodes remain in every denominator.
- No optional stopping, no sequential peeking, no post-hoc rule or map removal.

## 7. Execution and stopping rules

Before execution, freeze and record:

1. candidate commit and code hash;
2. config hash;
3. manifest content hash and every layout hash;
4. model/runtime versions, if any (none in this draft);
5. command line and output paths.

Abort before scoring if the manifest fails validation, a layout hash differs, a
candidate map appears in development output, a method is unknown, or any hidden
truth field reaches the policy input. Abort during execution on duplicate
scenario ids, a tool-budget violation, a parser fallback, or an environment
error. Do not retry individual failed episodes selectively.

The run is fixed at the manifest size. No early success claim is allowed.

## 8. Failure classification

Every non-success is classified as one or more of:

- task failure with correct identification;
- task success with wrong identification (successful misdiagnosis);
- unresolved identification;
- wrong settled commitment;
- earlier wrong commitment followed by correction;
- tool-budget failure;
- shift censoring or no post-shift evidence, for shift scenarios only;
- environment or contract failure.

The classification is descriptive and does not change the primary denominator.

## 9. Assumptions and decisions requiring human approval

1. Candidate identity and declaration API. The draft cannot name or freeze one.
2. Exact map list 210-239 and seeds 5-9. The exposure audit found no
   scenario-id or episode references and no geometry collision on 2026-11-25;
   this prose is only a name reference. They are not yet selected.
3. The 30 x 5 x 6 sample and the 900-episode denominator.
4. Holm correction and the exact McNemar formulation for H5, including whether
   terminal MAP is the accepted paired metric given the baseline's missing
   declaration API.
5. Base endpoint: task success is primary, terminal-base estimate secondary.
6. Whether diagnostic_belief_v4 is a required secondary baseline.
7. Whether any VLM arm is in scope. This draft excludes it.
8. Whether a 100% base gate on 150 episodes is the intended acceptance rule.
9. Whether the fail-closed raw-zero-tool gate is too strict, and if not, the
   exact public-observation-only evidence-coding rule that would permit an
   exception.

Until those decisions are recorded and the candidate and manifest are frozen,
no confirmatory run is authorized. This draft must not be represented as an
approved preregistration.
