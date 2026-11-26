# Evaluation Protocol v1.0

Status: **proposed; not executed; pending human approval**.

Scope: active rule identification in RuleShift-Grid. This protocol does not
establish general post-shift adaptation. Change-detection and no-shift episodes
are reported as secondary cohorts but are not part of the primary claim.

This document exists to prevent three failure modes in the earlier analysis:
changing the endpoint after seeing results, silently dropping unresolved
episodes, and treating task success as if it were rule identification. It is
written before the confirmatory holdout is run. The historical development
results in the accompanying documents are not evidence for the confirmatory
claim.

## 1. Cohorts and unit of analysis

The unit is one deterministic scenario episode. A scenario is identified by
map seed, run seed, scenario type, initial rule, and shift rule. All episodes
that are started are retained in the denominator. No episode is removed for
timeout, unresolved belief, tool failure, navigation loop, or a result that
disagrees with the hypothesis.

The planned primary cohort is the identification cohort:

- 10 genuinely unseen maps;
- 5 seeds per map;
- 6 public rules (base, remap_swap, remap_rotate, slippery,
  hazard_goal_swap, sign_composition);
- 300 identification episodes.

The base-preservation endpoint is the 50 true-base identification episodes in
that cohort: scenario type identification, initial rule base, and no shift.

Secondary cohorts, reported but not used for the primary claim:

- 200 change-detection episodes (base to each of the four shift rules);
- 50 no-shift control episodes;
- 550 episodes in total.

The primary analyses use the 300 identification episodes. Change-detection
results must not be described as evidence that the policy adapts in general.

## 2. Exact base-preservation endpoint and its ambiguity

The original 100% base gate is preserved. Its exact endpoint is:

- the 10-map development suite (maps 0-9), seeds 0-4;
- scenario type identification;
- initial rule base;
- shift rule None;
- 50 episodes;
- endpoint value is the environment's task-success flag, success == true, for
  every episode.

In the official v0.3 run this endpoint is **49/50**. The failing episode is
m2-s2-identification-base-None. The literal 100% gate therefore fails. A 100%
observed result is an empirical statement about this endpoint, not proof of
universal base preservation.

There is an ambiguity in the historical prose. "Base preservation" could mean
task success on those 50 episodes, or it could mean that the terminal rule
estimate is base. The original gate is defined by task success in the code and
in the failure-analysis tables. The alternative estimate endpoint also happens
to be 49/50 in the official v0.3 run, because the same m2-s2 episode has a
terminal estimate of slippery. The two endpoints are not interchangeable, even
though they coincide here.

Rules for reporting:

1. The primary base-preservation gate uses task success, as above.
2. Base-identification accuracy uses terminal estimate == base on the same 50
   episodes and is reported separately.
3. The gate is never moved to a different cohort, such as all base-initial
   change-detection and no-shift episodes.
4. A future result is reported as 50/50 on the defined endpoint, not as proof
   that base is always preserved.

The development v0.5 candidate reached 50/50 task success on this endpoint.
That is a development observation on maps already exposed to the authors. It
is not a confirmatory gate result.

## 3. Primary and secondary metrics

For an identification cohort of size N_id = 300, let:

- S = number of episodes with success == true;
- C = number of episodes whose resolved terminal estimate equals the true
  initial rule;
- R = number of episodes with a resolved terminal belief;
- U = N_id - R = unresolved identification episodes;
- B = 50 true-base identification episodes;
- C_base = number of those episodes whose resolved estimate is base;
- S_base = number of those episodes with success == true.

The metrics are:

| Metric | Definition | Denominator |
| --- | --- | --- |
| Task success | S / N_id | all identification episodes |
| Identification accuracy | C / N_id | all identification episodes; unresolved counts as not correct |
| Misdiagnosis rate | resolved base episodes with estimate != base, divided by B | all 50 base episodes |
| Identification coverage | R / N_id | all identification episodes |
| Unresolved identification rate | U / N_id | all identification episodes |
| Base preservation | S_base / B | all 50 base episodes |
| Base-identification accuracy | C_base / B | all 50 base episodes |
| Incorrect commitment rate | resolved episodes with estimate != truth, divided by N_id | all identification episodes |
| Evidence coverage | episodes with at least one tool unit or at least one direct cell-role observation, divided by N_id | all identification episodes |

Identification accuracy and task success are never substituted for one
another. A successful episode with a wrong terminal estimate is a successful
misdiagnosis, not an identification success. An episode that ends without a
resolved estimate is unresolved even if its task return is high.

"Identification coverage" means that the policy produced a resolved terminal
belief. It does not mean that the belief is correct. Accuracy is conditional on
the full denominator, not conditional on coverage.

The hazard and sign identification gates use the 50 hazard_goal_swap and 50
sign_composition identification episodes. They require a correct resolved
terminal estimate, not merely task success.

## 4. Terminal belief, ties, and unsettled beliefs

The terminal belief is the normalized posterior immediately after the final
environment transition and the final policy update, before any evaluation
cleanup. It must be saved with the episode. A policy's displayed
most_likely_rule() label is not sufficient to determine resolution, because
that method may break ties by rule order.

For a terminal posterior p:

- the leader is the rule with the largest probability;
- the runner-up is the rule with the second-largest probability;
- the leader is unique only when p(leader) - p(runner-up) > 1e-6;
- the belief is resolved only when the leader is unique and
  p(leader) >= 0.5;
- otherwise the belief is unresolved (a tie, a near-tie, or no majority).

Exact ties (including 0.5/0.5) are unresolved. A tie is never resolved by rule
name, menu order, map identity, or the most_likely_rule() tie-break. A resolved
wrong estimate is an incorrect identification; it is not an unresolved
episode. An unresolved episode remains in the accuracy denominator and counts
as not correct.

If a historical artifact did not save the terminal posterior, coverage and
resolution cannot be reconstructed from the label alone. Such artifacts may be
used for task-success and final-label comparisons, but not for a new coverage
claim. New confirmatory runs must save the posterior.

## 5. Incorrect commitments

An identification commitment is a resolved terminal estimate. It is incorrect
when the resolved estimate is not the true rule. This is an identification
error whether or not the episode also reached the goal.

A separate diagnostic, not used to change the endpoint, is an action-level
incorrect commitment: a non-diagnostic action chosen while the current belief
is resolved and wrong. Action-level commitments are recorded in traces and
classified in failure analysis. They can explain a task failure, but they do
not turn a successful episode into an identification success or vice versa.

Thus a policy can have:

- correct identification and task success;
- correct identification and task failure (navigation or budget failure);
- incorrect identification and task success (successful misdiagnosis);
- incorrect identification and task failure (incorrect commitment, possibly
  with navigation failure);
- unresolved identification and either task outcome.

These categories are reported separately.

## 6. Early termination and zero-tool successes

For identification episodes, early termination is not an error by itself. A
short successful episode is valid task success. Its terminal belief is still
scored normally, so a short accidental success with a wrong estimate is still
an identification error.

For change-detection episodes, an episode that terminates before its scheduled
shift time is flagged pre-shift termination. Its task success is retained as
task success on the initial rule, but it is not evidence of post-shift
adaptation and is not used in adaptation-step summaries. Truncation at the
decision limit is reported separately and is never converted into a success.

A raw zero-tool success is a hazard_goal_swap or sign_composition
identification episode with tool_units == 0 and success == true. It is
classified as:

- evidence-backed when the episode records at least one direct cell-role
  observation and the terminal estimate is correct; the public movement
  outcome, not a tool, supplied the evidence;
- accidental when no tool unit and no direct cell-role observation were
  recorded. A successful misdiagnosis is always treated as accidental for this
  purpose.

The acceptance gate is zero accidental zero-tool hazard/sign successes. Raw
zero-tool successes are also reported without classification. This distinction
is not a way to turn a raw zero-tool count into zero: both numbers are shown,
and the reason for each evidence-backed case is auditable in its trace.

## 7. Uncertainty and paired comparisons

- Proportions: Wilson 95% intervals.
- Map-level uncertainty: percentile bootstrap over the 10 maps, resampling
  maps with replacement and retaining all seeds within a sampled map, 10,000
  resamples. This respects the map cluster and does not resample individual
  seeds as if they were independent maps.
- Paired method comparisons: exact McNemar test on the 300 identification
  episodes. The primary directional test is one-sided for
  candidate > baseline; the two-sided p value is also reported.
- Paired difference intervals: map-cluster bootstrap of the paired accuracy
  difference.
- No interim look, optional stopping, or sample-size re-estimation. The
  planned sample is fixed before execution.
- Unresolved episodes remain in every denominator. No complete-case analysis
  is allowed for the primary endpoint.

## 8. Gates for the v0.5 confirmatory candidate

All gates apply to the new, unseen 300-episode identification cohort. The base
gate applies to its predefined 50 true-base subset. A gate is not a claim about
other maps, other rules, or general adaptation.

The candidate passes only if all of the following hold:

1. Identification accuracy is at least 95% (at least 285/300 correct), with
   unresolved episodes in the denominator.
2. Hazard identification accuracy is at least 90% (at least 45/50 correct).
3. Sign identification accuracy is at least 90% (at least 45/50 correct).
4. Base preservation is 100% (50/50 task successes) on the exact endpoint in
   Section 2. This is an empirical gate; its uncertainty is reported and it is
   not presented as universal proof.
5. There are zero accidental zero-tool hazard/sign successes under Section 6.
   Raw zero-tool counts are also reported.
6. No episode uses more than 3 tool units. Tool-spend mean and median are also
   reported.
7. The paired identification accuracy of v0.5 is greater than no_adaptation,
   with a positive point difference and a one-sided exact McNemar p < 0.05.
   The two-sided p value and the map-cluster interval are reported.

If any gate fails, the candidate fails the confirmatory test. A failure is
reported as a failure; it is not rescued by switching to task success, to a
different denominator, or to a post-hoc subset. If the candidate passes, the
claim is limited to the observed endpoint and cohort.

## 9. Failure classification

Every failed or anomalous episode receives one primary classification before
any aggregate is interpreted:

- task_failure: success == false without a resolvable identification issue;
- incorrect_commitment: resolved wrong terminal estimate, or a documented
  action-level commitment to a wrong rule;
- unresolved_identification: no resolved terminal estimate;
- successful_misdiagnosis: task success with a resolved wrong estimate;
- evidence_conflict: a tool or movement outcome contradicts the belief update;
- inspection_noise: a noisy tool observation dominates otherwise valid
  evidence;
- plan_loop: repeated actions fail to make progress after the belief has
  settled;
- post_shift_not_rediagnosed: a change-detection episode retains a pre-shift
  hypothesis without obtaining post-shift evidence;
- budget_exhausted: the tool budget is spent without resolving the relevant
  ambiguity;
- environment_or_evaluator_error: infrastructure failure, invalid action, or
  schema failure.

Secondary labels may be attached, but the primary label is fixed before
aggregation. A failure may have more than one contributing cause; the trace
must show the decisive one.

## 10. Historical comparisons

The following historical numbers remain as previously reported and are not
recomputed to make the new candidate look better:

- full 550-episode development grid: v0.3 534/550, v0.4 546/550;
- official v0.3 base endpoint: 49/50;
- repaired v0.4 holdout (maps 100-199, base episodes): 0/500 task failures and
  7/500 misdiagnoses; this was a post-hoc analysis and is not confirmatory.

These runs used previously exposed maps. They may motivate the candidate and
the failure analysis, but they are not confirmatory evidence. Historical
artifacts are preserved with their original commit hashes, configurations, and
result files.

The development v0.5 run is reported separately as 550/550 task successes,
299/300 correct terminal estimates, and 50/50 base task successes on maps 0-9.
Its terminal posterior was not saved for every development episode, so no
independent resolved-coverage claim is made for that run. It also contains
three raw zero-tool hazard/sign successes, all evidence-backed, and one
accidental short slippery success with a wrong estimate. These are development
observations, not gate results.

## 11. Reproducibility requirements

A confirmatory report must include:

- the frozen candidate commit and a clean worktree;
- the exact configuration file and its hash;
- the map range, seeds, and scenario manifest hash;
- the policy method name and class;
- the terminal posterior for every identification episode;
- the per-episode success, tool units, env steps, final estimate, and direct
  cell-role observation count;
- the code and command used for the paired and cluster analyses;
- a statement that the 300 episodes and 50 base episodes were fixed before
  execution.

Any change to the policy, scoring code, map set, or gate after the holdout is
started invalidates the confirmatory claim and requires a new protocol version
and a new unseen map set.

## 12. Open decisions for human approval

The following must be approved before the holdout is run:

1. Whether the base-preservation gate is accepted as task success (primary)
   with base-identification accuracy reported separately, given the ambiguity
   in Section 2.
2. Whether evidence-backed direct cell-role observations count as legitimate
   non-tool evidence for the zero-tool gate, with raw counts also reported.
3. Whether the 10-map x 5-seed identification cohort is the correct primary
   sample, with change detection secondary only.
4. Whether the candidate commit is frozen at the commit containing this
   protocol plus the v0.5 policy and tests, before any holdout execution.

No confirmatory holdout has been run under this protocol.
