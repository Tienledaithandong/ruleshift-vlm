# Evaluation Protocol v1: RuleShift-Grid identification

Status: P0 measurement-repair draft. This document does not authorize a
confirmatory holdout, model calls, GPU use or a merged policy change.

Scope: active rule identification is the primary research claim. General
post-shift adaptation is outside this protocol. Task success, identification,
commitment, tool spend and shift timing are reported separately.

Evaluation version: grid-identification-v1. Manifest version: grid-manifest-v1.

## 1. Cohorts and denominators

An episode is never removed from a cohort after its outcome is known.

Static identification cohort: scenario_type is identification. It contains six
initial rules (base, remap_swap, remap_rotate, slippery, hazard_goal_swap,
sign_composition). Every static episode is in the denominator.

Base endpoint: the static identification cohort with initial_rule base, using
the frozen map and seed list. The historical endpoint is maps 0-9, seeds 0-4
(50 episodes). A future endpoint must be specified in the manifest before
execution.

Shift cohort: scenarios with a non-null shift_rule. Shift metrics never enter
the static primary endpoint. The no-shift control is reported separately.

## 2. Independent outcome fields

Task success is the environment's success flag. It is not identification.

Terminal MAP is the highest-probability rule at the end of the episode, with a
documented deterministic tie break. It is compared with the true rule at
termination (the terminal true rule), not with the pre-shift rule on a shifted
episode. Terminal MAP accuracy is secondary and is computed independently of
task success.

Misdiagnosis means task success with an incorrect terminal MAP. It is not an
identification success.

A joint field records task success and correct identification together. A
policy may succeed by accident, and must not receive identification credit for
doing so.

## 3. Explicit commitments

The primary identification score is:

    correct explicit settled declarations / all static episodes

A declaration is an event produced by the policy's declaration API. It contains
the decision step, declared rule, confidence and top-second posterior margin. A
declaration is settled only when confidence >= 0.90 and margin >= 0.10. Both
conditions are required. The final declaration before termination is scored.
Commitments are scored only on the static identification cohort; a shifted
episode contributes to the separate shift summary, not to the primary
commitment numerator or denominator.

Commitment status is exactly one of:

- correct_commit: settled declaration equals the true initial rule;
- wrong_commit: settled declaration names another rule;
- unresolved: no declaration, an unsettled declaration, a missing posterior, or
  any other failure to make a scorable declaration;
- unavailable: the policy exposes no declaration API.

Unresolved and unavailable are never counted as correct. Ties are unresolved.
A terminal MAP is not a declaration and may not be substituted for one.

Earlier settled wrong declarations are counted separately, even if a later
declaration is correct. Reporting only the final status would hide a wrong
commitment that was later repaired.

For policies without a declaration API, commitment accuracy is reported as
unavailable. Their terminal MAP and posterior may be reported descriptively.
They cannot pass the primary commitment gate by substituting MAP for
commitment.

A successful goal before any declaration is task success plus unresolved
identification.

## 4. Coverage, selective error and calibration

Coverage is the fraction of static episodes with a valid full-menu posterior or
a declaration, as separately named. Selective error is:

    wrong commitments / (correct commitments + wrong commitments)

When the denominator is zero it is unavailable, not zero.

Brier score and negative log-likelihood are reported only for valid full-menu
posteriors whose probabilities sum to one within tolerance. Brier is the
conventional multiclass sum of squared class errors; NLL is the negative log
probability assigned to the terminal true rule. Neither is imputed for policies
without a posterior. Raw and normalized calibration may be reported separately;
neither replaces identification accuracy.

## 5. Base preservation

Base preservation is measured on the static base endpoint, never on shift
episodes. The original gate is 50/50 task successes, with no 0.99 tolerance and
no substitution of a terminal base estimate for task success. The terminal-base
estimate is reported as a secondary field.

A 100% observed result is an empirical gate result for the observed endpoint,
not a universal proof about all maps or all possible base transitions.

## 6. Shift timing

The shift export distinguishes:

- scheduled shift time;
- actual shift time, detected from the evaluator-only true rule change;
- censoring, when the episode ended before the scheduled shift occurred;
- policy shift-probability event and its step;
- false alarm before the actual shift;
- shift away and back to base, when the true terminal rule is base;
- no post-shift evidence.

The policy's shift-probability threshold crossing is a policy proxy, not an
explicit detection declaration. The current baselines do not expose a separate
detection declaration. Shift fields are never used to score the static primary
endpoint.

## 7. Tools and accidental successes

Tool spend is the configured tool-unit spend, with the episode cap of 3. Tool
events are exported separately. Zero-tool successes are reported as raw counts.

A zero-tool success on hazard_goal_swap or sign_composition is not automatically
called evidence-backed or accidental. An accidental-success classification
requires a separate evidence audit, such as direct public cell-role feedback.
Without that evidence the report says zero-tool success and nothing stronger.

## 8. Pairing and statistics

Paired comparisons require exactly the same unique scenario ids. Missing ids,
extra ids and duplicates are errors; implementation silently intersecting the
two sets is forbidden.

The primary paired statistic is an exact McNemar test on the paired binary
outcome. Report both discordant cells and the exact two-sided p value. Do not
use a normal approximation for small discordant counts.

Rates use Wilson intervals. Confirmatory uncertainty may additionally use a
cluster bootstrap over maps, never over seeds or frames. Multiplicity control
must be specified before the run.

## 9. Gates retained from the program

- Identification accuracy: at least 95% on the frozen static endpoint, using
  explicit settled commitments.
- Hazard identification: at least 90%.
- Sign identification: at least 90%.
- Base preservation: 100% task success on the static base endpoint.
- No accidental zero-tool hazard or sign success. Until a separate evidence
  audit is approved, the gate is fail-closed: any raw zero-tool success on
  hazard_goal_swap or sign_composition blocks it. An evidence-backed exception
  requires a pre-registered coding rule that uses only public observations.
- Tool spend at most 3 units per episode.
- Positive paired terminal-MAP gain over no_adaptation. A paired commitment
  gain is required only when both methods expose a declaration API; a
  commitment-unavailable baseline cannot supply it.

A gate whose required field is unavailable is not passed by a fallback metric.
The historical 49/50 base result remains labelled 49/50. It is not rewritten as
100%.

## 10. Reproducibility and manifests

Every run uses an explicit scenario manifest containing scenario ids, map ids,
run seeds, scenario types, rules, shift times and public-layout SHA-256 hashes.
The manifest content hash is recorded in every episode record. Layout hashes
cover geometry, not only integer map ids.

Duplicate manifests, stale hashes, unknown methods, layout mismatches and
missing paired scenarios fail closed. Development mode refuses maps above 199.
Confirmatory execution is not available in the P1 scaffold.

## 11. No retrospective scoring

A result is not relabelled after seeing it. Old outputs keep their original
labels and thresholds. New reports use the grid-identification-v1 names and
state which fields were unavailable in the old rows.

## 12. Decision rule for this P0 stage

P0 is complete only when the metric definitions, manifest validation, shift
export, leakage checks and regression tests pass. P0 completion does not
authorize candidate tuning or a holdout. Those require a separately approved
preregistration and a frozen candidate commit.
