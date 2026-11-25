# Preregistration: v0.5 active rule identification holdout

Status: **DRAFT FOR HUMAN APPROVAL. NOT EXECUTED.**

This preregistration is written before the confirmatory holdout is run. It
freezes the candidate, maps, seeds, metrics, gates, and analysis plan. No
confirmatory episode has been executed under this document. The development
runs on maps 0-199 are not part of this confirmation.

The primary claim is active rule identification in the six-rule
RuleShift-Grid. General post-shift adaptation is outside the current scope.
Change-detection and no-shift episodes are secondary descriptives.

## 1. Frozen artifacts

| Artifact | Value |
| --- | --- |
| Candidate code commit | 62eca55b548d9a2c9dc1f80ca8ff4a080d47b6da |
| Candidate policy | ruleshift.agents.grid_baselines.DiagnosticBeliefPolicyV5 |
| Candidate source SHA256 | 4806D83B0B2658CE47F69A1DBDD75A935BDAC27EBFA2E5450255027BF2FE7848 |
| Config | configs/grid_dev.yaml |
| Config SHA256 | F4FAF788641A400043CD7EAA9DA8D1EF204A65985F671ABFA7A2C3CFF2ED3AA3 |
| Baseline code | commit f4b43a4 (v0.3 and v0.4 classes unchanged) |
| Holdout manifest | maps 200-209, seeds 5-9, 550 scenarios |
| Manifest SHA256 | 8dedfe1be2417532180e73b9e1e4635bdf962c0ecfdbbd5b5d2eb73e07284310 |

The manifest hash is SHA256 over the sorted JSON array of selected scenario
records, with keys scenario_id, map_seed, run_seed, scenario_type,
initial_rule, shift_rule, and shift_time, serialized with sorted keys and
compact separators.

The candidate code commit must be the code actually executed. Any source
change after approval requires a new preregistration and a new unseen map set.
The preregistration document itself is frozen at the commit that records the
human approval; it is not edited after the holdout starts.

## 2. Research question and hypotheses

Question: does deterministic cell-role feedback improve active rule
identification without harming base preservation or increasing tool spend?

The candidate's only change from v0.4 is the direct cell-role channel:
physically entering a marker and observing move or hazard updates the belief
about that marker's role. This is public evidence, not a hidden-state read.

Directional hypotheses, fixed before execution:

- H1 (primary): v0.5 identification accuracy is greater than no_adaptation on
  the 300 identification episodes.
- H2: v0.5 identification accuracy is at least 95%.
- H3: v0.5 hazard_goal_swap identification accuracy is at least 90%.
- H4: v0.5 sign_composition identification accuracy is at least 90%.
- H5: v0.5 base preservation is 100% on the 50 true-base identification
  episodes.
- H6: there are no accidental zero-tool hazard/sign successes.
- H7: no primary episode exceeds the 3-unit tool budget.

H2-H7 are conjunctive acceptance gates. H1 is the paired scientific
comparison and is also a required gate (G7); it is not a substitute for the
absolute gates.

The candidate is not expected to improve general post-shift adaptation, and
the change-detection cohort is not used to support that claim.

## 3. Design, maps, and sample size

New holdout design:

- maps: 200, 201, 202, 203, 204, 205, 206, 207, 208, 209;
- seeds: 5, 6, 7, 8, 9;
- 11 scenario types per map and seed: six identification rules, four
  change-detection shifts from base, and one no-shift control;
- 550 episodes per method;
- primary identification cohort: 300 episodes (six rules x 10 maps x 5 seeds);
- base endpoint: 50 true-base identification episodes;
- hazard endpoint: 50 hazard_goal_swap identification episodes;
- sign endpoint: 50 sign_composition identification episodes;
- secondary change-detection cohort: 200 episodes;
- secondary no-shift cohort: 50 episodes.

The maps are proposed as genuinely unseen because no prior result file in the
repository references m200-m209. A scan of the 67 files under results/ found no
such scenario prefix. The map generator also passes the repository's
all-hypothesis solvability gate and readable-side-0 gate for every proposed
map. The audit found ten unique maps, one or two side-0 face cells per map,
and a reachable side-0 face on all ten.

| Map | Generator attempts | Side-0 cells | Side-0 reachable |
| ---: | ---: | ---: | --- |
| 200 | 2 | 1 | yes |
| 201 | 1 | 2 | yes |
| 202 | 3 | 2 | yes |
| 203 | 4 | 2 | yes |
| 204 | 1 | 2 | yes |
| 205 | 2 | 1 | yes |
| 206 | 2 | 1 | yes |
| 207 | 4 | 2 | yes |
| 208 | 2 | 1 | yes |
| 209 | 3 | 2 | yes |

Seeds 5-9 are used instead of the development seeds 0-4 so that the exact
map/seed combinations are new. The maps, not merely the seeds, are the
unseen unit.

Sample size is fixed at 550 episodes per method and 300 primary identification
episodes. There is no power-based optional stopping and no adaptive extension.
The primary accuracy gate of 95% corresponds to at least 285/300 correct; the
hazard and sign gates correspond to at least 45/50 each. These are thresholds
on the observed endpoint, with uncertainty reported separately.

## 4. Baselines

All baselines use the same 550 scenarios, same config, same environment, and
same evaluation code:

- no_adaptation: no belief-directed probing or adaptation; the paired
  reference for H1;
- v0.3: DiagnosticBeliefPolicy, the original official baseline;
- v0.4: DiagnosticBeliefPolicyV4, the repaired probe-allocation baseline;
- v0.5: DiagnosticBeliefPolicyV5, the candidate.

v0.3 and v0.4 are preserved from commit f4b43a4 and are not edited in this
branch. The candidate is additive at 62eca55.

The development no_adaptation run on the exposed 300-episode identification
cohort produced 50/300 correct terminal estimates and 128/300 task successes.
This is context for the paired comparison, not a confirmatory estimate and not
a substitute for running the baseline on the new maps.

## 5. Endpoints and scoring

The full definitions are in docs/EVALUATION_PROTOCOL_V1.md. The confirmatory
analysis uses them unchanged.

Primary identification cohort, N = 300:

- task success rate = successes / 300;
- identification accuracy = resolved terminal estimate equal to the true
  initial rule, divided by 300;
- identification coverage = resolved terminal beliefs / 300;
- incorrect commitment rate = resolved wrong terminal estimates / 300;
- unresolved identification rate = unresolved / 300;
- evidence coverage = episodes with at least one tool unit or at least one
  direct cell-role observation, divided by 300.

Hazard and sign identification:

- correct resolved terminal estimate on the 50 hazard_goal_swap episodes;
- correct resolved terminal estimate on the 50 sign_composition episodes.

Base endpoint, B = 50 true-base identification episodes:

- base preservation = task successes / 50 (primary, original gate);
- base-identification accuracy = resolved terminal estimate base / 50
  (secondary).

Resolution rule:

- terminal posterior saved immediately after the final update;
- resolved only when the leader is unique by more than 1e-6 and has
  probability at least 0.5;
- ties, near-ties, and no-majority beliefs are unresolved;
- unresolved episodes remain in every accuracy denominator and are never
  counted as correct;
- the most_likely_rule() tie-break is not used for scoring resolution.

A successful episode with a wrong terminal estimate is a successful
misdiagnosis, not an identification success. A task failure with a correct
terminal estimate is a navigation or budget failure, not an identification
error. The two axes are reported separately.

Raw zero-tool successes are reported. An accidental zero-tool hazard/sign
success is one with zero tool units and zero direct cell-role observations. An
evidence-backed zero-tool success has at least one direct cell-role observation
and a correct terminal estimate. The gate counts accidental cases only, but the
raw count is always shown.

## 6. Acceptance gates

The candidate passes only if all seven conditions hold on the new holdout:

| Gate | Required |
| --- | --- |
| G1 identification accuracy | at least 285/300 (95%) |
| G2 hazard identification | at least 45/50 (90%) |
| G3 sign identification | at least 45/50 (90%) |
| G4 base preservation | 50/50 task successes on the exact 50-episode endpoint |
| G5 accidental zero-tool hazard/sign successes | 0 |
| G6 tool spend | no episode above 3 tool units |
| G7 paired identification gain | v0.5 > no_adaptation, positive point difference, one-sided exact McNemar p < 0.05 |

G4 is an empirical gate on the defined endpoint, not proof of universal base
preservation. Its 95% interval is reported; with 50/50 the Wilson lower bound
is below 100%, and the report must say so.

The endpoint ambiguity for G4 is acknowledged. The primary definition is task
success on the 50 true-base identification episodes. Base-identification
accuracy is reported separately. If the human reviewer rejects this endpoint,
the holdout must not be run under this preregistration; a new protocol version
is required.

Gates are conjunctive. No multiplicity adjustment is applied because the gates
are fixed, separate requirements, not alternative tests of one hypothesis. All
gate results and all intervals are reported whether the candidate passes or
fails.

## 7. Statistical analysis

- Proportions: Wilson 95% confidence intervals.
- Map-level uncertainty: percentile bootstrap over the 10 maps, resampling
  maps with replacement and retaining all five seeds within each sampled map,
  10,000 resamples. This is a map-cluster bootstrap, not an episode bootstrap.
- Paired method comparison: exact McNemar test on the 300 identification
  episodes, pairing by scenario. H1 uses the one-sided alternative
  v0.5 > no_adaptation. Two-sided p values are also reported.
- Paired difference: map-cluster bootstrap of the paired accuracy difference;
  report the point estimate and 95% percentile interval.
- Base and per-rule percentages: Wilson intervals with their stated
  denominators.
- No imputation, no complete-case deletion, no exclusion of unresolved
  episodes, no post-hoc subgroup as a primary result.
- Raw counts and percentages are both reported. A p value does not override a
  failed absolute gate.

## 8. Stopping and deviation rules

- The sample is fixed. There is no interim analysis, no peeking, no optional
  stopping, and no sample-size re-estimation.
- Run the frozen candidate once on the fixed manifest. Do not rerun to obtain
  a more favorable stochastic outcome.
- A technical abort before any episode result is available may be rerun with
  the same frozen code and manifest. The abort and rerun must be documented.
- An abort after any confirmatory episode result is available does not permit
  reuse of the same maps for a new candidate. The partial result is preserved
  and reported; a new unseen map set and a new preregistration are required.
- If the code hash, config hash, or manifest hash differs from Section 1,
  abort; do not run.
- If the map audit fails, abort; do not substitute maps.
- If a gate fails, stop and report the failure. Do not change the policy,
  endpoint, denominator, map set, or gate after seeing the result.
- Any deviation is recorded in a deviations section with time, reason, and
  whether it occurred before or after outcome inspection.

## 9. Failure classification

Every failed or anomalous episode receives a primary label before aggregation:

- task_failure;
- incorrect_commitment;
- unresolved_identification;
- successful_misdiagnosis;
- evidence_conflict;
- inspection_noise;
- stale_evidence;
- plan_loop;
- post_shift_not_rediagnosed;
- accidental_short_success;
- budget_exhausted;
- environment_or_evaluator_error.

The classification is descriptive, not a way to remove an episode. An episode
with a task failure and a wrong estimate is both a task failure and an
incorrect identification. An unresolved episode remains in the denominator.

Expected residual risks from development, to be checked rather than assumed:
noisy INSPECT dominance, early accidental goal arrival before evidence, and a
correct task outcome with a wrong terminal estimate. The development run
contains one accidental short slippery success with a wrong estimate; the
confirmatory analysis must count any analogous episode as an identification
error, not as a success for the primary claim.

## 10. Execution procedure

1. Human approval of this document and of the endpoint in Section 6.
2. Verify the branch worktree is clean and HEAD contains the candidate code
   commit and this preregistration; record the HEAD hash.
3. Verify source, config, and manifest hashes.
4. Build the 550 scenarios from maps 200-209 and seeds 5-9; verify the
   manifest hash.
5. Run each method once on the same scenario objects:
   no_adaptation, v0.3, v0.4, v0.5.
6. Save one JSONL row per episode with method, scenario, success, env_steps,
   tool_units, final estimate, terminal posterior, direct role observation
   count, scenario type, initial rule, shift rule, and shift time. Save the
   failure traces separately.
7. Compute the protocol metrics and gates from the saved rows, without
   editing the raw data.
8. Report all gate outcomes, both pass and fail, with intervals and paired
   tests. Preserve raw outputs under the git-ignored results/ tree.
9. Do not merge the research branch until the protocol and candidate have been
   reviewed and the CI is green.

No GPU, VLM, or external model call is part of this holdout. It is a CPU
baseline experiment.

## 11. Interpretation limits

A pass supports only the claim that v0.5 met the fixed gates on this 10-map,
5-seed holdout. It does not establish universal base preservation, general
post-shift adaptation, or robustness to arbitrary map distributions. A fail is
a fail; it is not repaired by switching to task success, to a different
denominator, or to a post-hoc subgroup.

The development result (v0.5 550/550 task successes and 299/300 correct
terminal labels on maps 0-9) is reported only as development context. It is
not part of the confirmatory evidence and must not be cited as a gate result.

## 12. Approval checklist

- [ ] Primary claim limited to active rule identification.
- [ ] Base-preservation endpoint accepted as task success on the 50 true-base
      identification episodes, with base-identification accuracy separate.
- [ ] Evidence-backed direct cell-role observations accepted as non-tool
      evidence, with raw zero-tool counts reported.
- [ ] Candidate code commit frozen at 62eca55b548d9a2c9dc1f80ca8ff4a080d47b6da.
- [ ] Maps 200-209 and seeds 5-9 accepted as genuinely unseen.
- [ ] Seven gates accepted as written.
- [ ] No execution before approval.

Approval date and approver: ______________________________

No confirmatory holdout has been run under this preregistration.
