# v0.5 separated epistemic/control design

Status: development candidate, CPU-only, frozen before the official 300-episode
development run. Protocol: docs/EVALUATION_PROTOCOL_V2.md at commit
ae1c112.

## 1. Scope and hypothesis

The only authorized candidate is v0.5-separated-epistemic. It is not a
confirmatory candidate and this work does not authorize reserved-map,
GPU, VLM, or holdout evaluation.

Hypothesis: separating explicit epistemic decisions (declare, abstain,
request evidence) from control decisions (commit, probe sign, probe blue)
improves explicit rule identification without sacrificing task performance
under the existing information budget.

The previous candidate, repaired v0.4, remains closed as a confirmatory
candidate at e65e1915503246f0f07ed15c4884ac6952925ee8. Its historical
results and failure modes are not rewritten by this work. Protocol v1 remains
frozen for those results.

## 2. Candidate identity

- Candidate class: DiagnosticBeliefPolicyV5
  (ruleshift.agents.grid_v05).
- Control implementation: inherited from DiagnosticBeliefPolicyV4 without
  changing navigation, safe-commit logic, sign/blue probe order, or budget
  reservation.
- Epistemic implementation: _SeparatedV2Policy, evaluated before each v4
  control decision and once after the environment closes.
- Method name in records: v5.

The explicit event API is read-only to the evaluator:

- epistemic_events();
- declaration_events();
- abstention_events();
- request_events();
- evidence_ledger().

commitment_events() deliberately returns an empty tuple so the frozen v1
evaluator cannot mix v1 safe-commit declarations with v2 events.

## 3. Decision timing

Each pre-action round performs, in order:

1. observe the new public posterior and maintain the evidence ledger;
2. make one epistemic decision (DECLARE_RULE, ABSTAIN, or
   REQUEST_EVIDENCE);
3. make one v4 control decision (COMMIT, PROBE_SIGN, PROBE_BLUE).

The epistemic decision never directly changes the environment. A request is
not an action and gives no observation or tool unit. The control layer is the
only component that emits actions.

After success, truncation, or the decision bound, the runner calls
finalize(observation, info) exactly once. This is a readout only: no
environment action, no tool call, and no new observation. It uses the terminal
public posterior and the existing evidence ledger. The runner then captures
the terminal posterior for evaluation. A final declaration therefore cannot be
a retrospective conversion of terminal MAP.

## 4. Epistemic interface

### 4.1 Declaration

A declaration is emitted only when all of the following hold at that moment:

- full normalized six-rule public posterior;
- top probability >= 0.90;
- top-minus-second margin >= 0.10;
- at least one compatible public evidence reference.

The event records rule_id, confidence, posterior_margin, step, decision_index,
source (pre_action or final_observation), terminal, and evidence_refs.
Duplicate declarations for the current leader are suppressed; a different
leader is a revision. At most six declaration events are retained. The final
declaration is the last retained declaration event.

An abstention never retracts a declaration. If the episode closes with no
declaration, a terminal abstention is emitted with a closed reason. Consecutive
identical abstentions are de-duplicated.

### 4.2 Evidence requests

REQUEST_EVIDENCE(channel, target, purpose) is enabled only in the candidate
(C). Channels are SIGN and BLUE; BLUE is the public request to inspect the
blue marker (the control action is the normal INSPECT).

The request:

- consumes no budget and creates no observation;
- is satisfied only when the control layer executes the corresponding legal
  action and pays its normal cost;
- is marked executed, unmet, or expired;
- is limited to one pending request.

The candidate's request order mirrors v4's probe priority. At exactly the
inspection cost it first reserves the budget for a blue face that separates
the two leading rules; otherwise it requests an informative unread sign face,
then an eligible blue inspection. The request can only prefer an action the
control layer can already take legally; if it cannot, the request is unmet
and v4's normal probe logic applies.

### 4.3 Evidence eligibility

The ledger contains only public observations: movement outcomes, sign clauses,
inspect roles, and interaction outcomes. Each entry has a public id, step,
decision index, observed value, and public likelihoods over the six rules. A
declaration's refs must be a non-empty subset of compatible entries. No entry
contains a true rule, hidden role, evaluator field, or simulator-only
transition. The 5% inspect error rate is part of the public likelihood model
and is not changed after observing outcomes.

## 5. Controls and ablation

- A (historical): v3 and repaired v4 under their original v1 protocols.
  Compared descriptively only; it is not a controlled v2 arm.
- B (v4_readout_v2): repaired v4 control plus the v2 declaration and
  final-observation readout, with active requests disabled. This isolates the
  added declaration opportunity from the control policy.
- C (v5): candidate; same v4 control, separated declaration readout, and
  active REQUEST_EVIDENCE.
- D (v5_no_request): candidate class with active requests disabled. By
  construction B and D share the same control and request-disabled behavior;
  D is therefore expected to duplicate B. Any request effect is estimated by
  C versus B, not by D versus B. If C's requests do not alter a control action,
  the report says so rather than claiming a causal request benefit.

Declarations are scored independently of task success and terminal MAP. Task
success with a wrong or absent declaration is a successful misdiagnosis, not
an identification success.

## 6. Frozen configuration and data

GridConfig() defaults are frozen for this run:

- width 7, height 7;
- max_env_steps=60, max_decisions=200;
- tool_budget=3;
- inspect_cost=2, read_sign_cost=1;
- inspect_range=2, inspect_error=0.05;
- shift_probability=0.35, shift_window=(10, 40);
- wall_density=0.18, step_cost=-0.05, goal_reward=10.0;
- the remaining defaults are recorded by config_sha256.

Data are the 300 static identification scenarios on exposed maps 0-9 and
seeds 0-4, built from the frozen development manifest
cfc6504ec4ad4a27494947e0200581ec86d4e781cb278719da149dc6e5b38b00.
Reserved maps 210-239 are refused. No confirmatory or holdout run is
performed.

Run command:

    PYTHONPATH=src python scripts/run_grid_identification_v2.py --manifest results/p0_dev_static_manifest.json --mode development --out results/grid_v05_official_final --code-commit 751a3414ad8815087f64d85ad5a1523af11077f4

## 7. Readiness criteria (recorded before the official run)

1. Overall unconditional declaration accuracy >= 95% (285/300).
2. Hazard final declarations >= 90% (45/50).
3. Sign-composition final declarations >= 90% (45/50).
4. Base task success 100% (50/50 observed).
5. Zero raw zero-tool hazard/sign task successes.
6. Tool spend <= 3 units per episode.

These are development targets, not confirmatory gates. Coverage gains are also
reported against new final wrong declarations, any-wrong events, revisions,
strict clean accuracy, and task regressions.

## 8. Freeze rule

This is the final authorized symbolic candidate-development cycle. After the
official run, the candidate identity, thresholds, evidence model, probe order,
and data slice are frozen. No v0.6, threshold relaxation, policy patching,
holdout access, GPU/VLM run, or PR #22 modification is permitted. If the
candidate fails a target, the result is reported as negative and the decision
packet proposes closing or redesigning the benchmark rather than tuning again.
