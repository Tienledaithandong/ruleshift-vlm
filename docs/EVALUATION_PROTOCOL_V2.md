# Evaluation Protocol v2 — separated epistemic/control decisions

**Status:** proposed protocol for the authorized `v0.5-separated-epistemic`
development cycle. It is not the frozen v1 protocol and it does not
retroactively change any v0.3/v0.4 record, gate, manifest, or preregistration.

**Authorization boundary:** CPU-only development on previously exposed maps
0-9 and seeds 0-4. No confirmatory holdout, no maps 210-239, no VLM/GPU
inference, no policy tuning after the official development run, and no merge.

## 1. Purpose and scope

v1 coupled identification to control: a rule was recorded only as a side
effect of the safe control-commit branch. Protocol v2 makes the epistemic
assertion an explicit, time-stamped decision that is independent of the
control action. The purpose is to test whether that separation improves
explicit identification without sacrificing task performance under the
existing information budget.

Protocol v2 defines:

- two decision channels, epistemic and control;
- an explicit declaration event, abstention event, and evidence request;
- one optional final-observation declaration opportunity;
- scoring that never converts a terminal posterior into a declaration;
- matched controls and one no-request ablation;
- development criteria recorded before the candidate is evaluated.

The existing v1 implementation and all historical results remain valid
records of the v1 protocol. They are not rescored under v2.

## 2. Decision timing

An episode is a sequence of **rounds**. Round `t` begins with the current
public observation `o_t` and the posterior produced by observations through
the previous round. The policy makes:

1. **one epistemic decision** (optional event): `DECLARE_RULE`, `ABSTAIN`, or
   `REQUEST_EVIDENCE`; and then
2. **one control decision**: `COMMIT`, `PROBE_SIGN`, or `PROBE_BLUE`.

The control decision is translated into exactly one environment action. An
epistemic event never directly changes the environment. The environment
returns a public observation and outcome; the belief is updated at the next
`record`/round boundary.

The initial observation is round 0. It has no preceding public outcome, so it
cannot support a declaration unless the posterior already meets the
requirements and at least one evidence reference exists (it normally cannot).
The number of pre-action rounds is bounded by `max_decisions` (200 in the
development config). There is no separate unbounded epistemic loop.

After the environment terminates, truncates, or the decision bound is reached,
there is exactly one **final-observation opportunity** before scoring. It
uses the terminal public observation and the posterior after the last
environment action. It permits one epistemic decision only. It permits no
environment action, no tool call, and no additional observation.

## 3. Epistemic interface

### 3.1 `DECLARE_RULE(rule_id)`

A time-stamped public assertion that `rule_id` is the active rule. A
declaration event carries at least:

- `rule_id`;
- `step` (the current `env_steps`);
- `decision_index` (the pre-action round index, or the terminal round index
  for a final-observation declaration);
- `confidence` = the top posterior probability;
- `margin` = top minus second;
- `evidence_refs`: identifiers of prior public observations used as support;
- `source`: `pre_action` or `final_observation`.

A declaration is valid only when all of the following hold at the moment of
the event:

1. the public posterior is a full, normalized menu over the six rules;
2. the top probability is at least **0.90**;
3. the top-minus-second margin is at least **0.10**;
4. at least one referenced public observation is compatible with the declared
   rule under the public likelihood model; and
5. the declaration does not duplicate the currently active declaration for
   the same rule.

The thresholds are unchanged from v1. Protocol v2 adds no confidence
relaxation. The two-consecutive-leader control-stability condition is **not**
a declaration requirement; it remains a property of the control layer where
the control layer uses it.

An `INSPECT` is error-prone (public 5% error rate). A declaration may cite an
inspect as evidence; the protocol does not claim that one inspect is
corroborated. Noise-related errors are reported, not hidden by changing the
likelihood model after the fact.

### 3.2 `ABSTAIN(reason)`

An explicit, time-stamped decision not to assert a rule at this decision.
Reasons are from a closed public vocabulary: `insufficient_evidence`,
`ambiguous`, `noisy_evidence_unconfirmed`, `budget_exhausted`, `terminal_only`,
or `evidence_requested`.

An abstention is **not** a declaration, not a terminal MAP conversion, and not
a correct identification. If no later declaration exists, the episode is
scored `unresolved`. An abstention does not retract an earlier declaration.
Consecutive identical abstentions may be de-duplicated; non-consecutive
changes of reason are retained. A final abstention is required when the
episode closes without a declaration, so unresolved status has an explicit
reason.

### 3.3 `REQUEST_EVIDENCE(channel, target, purpose)`

A public request for a future observation. `channel` is `SIGN` or `BLUE`.
`target` is a public map location or `next_informative_face`; `purpose` is a
short reason string. A request:

- provides **no observation and no free information**;
- consumes no tool budget by itself;
- is not an environment action;
- is satisfied only if the control layer later executes the corresponding
  legal `READ_SIGN` or `INSPECT` action and pays its normal cost;
- may remain pending while the control layer navigates;
- is marked `executed`, `unmet`, or `expired` when it is resolved or the
  episode ends.

At most one request is pending at a time. A new request supersedes a pending
request only if the old one is explicitly marked `superseded`. Request
targets must be derived from the public observation; evaluator state, hidden
roles, and true-rule access are prohibited. A request cannot create an action
mask entry or budget that the environment did not publish.

## 4. Control interface

### 4.1 `COMMIT`

Navigate toward the current public MAP rule's goal/subgoal. `COMMIT` is a
navigation decision; it does not declare a rule and does not itself end the
episode. The environment still decides goal/hazard/termination from its public
transition. A declaration and a `COMMIT` may occur in the same round; the
declaration is not a side effect of the commit and the commit is not evidence
for the declaration.

### 4.2 `PROBE_SIGN`

Move to, or execute, a public `READ_SIGN` action. A read is legal only when
the agent is orthogonally adjacent to the signpost, the `READ_SIGN` action is
in the public mask, and the remaining budget is at least the public
`read_sign_cost` (1). If the agent is not adjacent, the control layer may
navigate toward an informative face; that navigation is still one control
decision and one physical step/action.

### 4.3 `PROBE_BLUE`

Move to, or execute, a public `INSPECT` of the blue marker. An inspect is legal
only when the target is public, in range, inspectable, and the remaining
budget is at least the public `inspect_cost` (2). Navigation toward the target
has the same status as in `PROBE_SIGN`.

### 4.4 Interaction with epistemic decisions

Epistemic and control decisions are logically independent:

- A declaration does not force, replace, or terminate a control action.
- A control action does not create or suppress a declaration.
- A `REQUEST_EVIDENCE` may cause the control layer to prefer the requested
  probe; this is an explicit, logged influence, not an implicit coupling.
- If the request cannot be executed legally, the control layer falls back to
  its normal public policy; the request is marked `unmet`.
- `ABSTAIN` never blocks a control action.

## 5. Evidence eligibility and public isolation

The policy maintains an evidence ledger containing only observations returned
by the environment: movement outcomes, sign clauses, inspect roles, and
interaction outcomes. Each entry has a monotonically increasing public
reference id, the decision/step at which it was observed, and the public
outcome. No entry contains a true rule, hidden role, evaluator field, or
simulator-only transition.

A declaration's `evidence_refs` must be a non-empty subset of that ledger.
An evidence entry is **compatible** with a rule when the public likelihood of
the observed outcome under that rule is positive. Compatibility is necessary,
not sufficient, for truth; the posterior confidence and margin are the
quantitative requirement. Evidence cannot be fabricated, copied from the
terminal posterior, or inferred from an evaluator label.

The final-observation declaration uses the same ledger and the same posterior.
The terminal transition itself may add a movement evidence entry; if it does,
it is a legitimate final observation. The final opportunity never reads the
true rule or calls the environment again.

## 6. Revision semantics

- The first valid `DECLARE_RULE` is the initial declaration.
- A later valid declaration with a different rule is a **revision**. The
  revision is retained in the event ledger and counts in the revision total.
- A later declaration of the same rule is suppressed; it is not a new event.
- The **final declaration** is the last retained declaration event, not the
  terminal MAP, not the last abstention, and not the first declaration.
- An abstention does not erase a prior declaration. If a policy wants to
  withdraw an assertion, v2 has no retraction event; it must make a new
  declaration or remain with the old one. This is deliberate: silent
  retraction would make time-stamped accounting ambiguous.
- At most six declaration events are retained per episode; further attempted
  declarations are suppressed and logged as `suppressed_declaration_cap`.

Final-observation revisions are allowed under the same rules. A final
declaration is an actual assertion made before scoring, using only public
evidence; it is not a retrospective conversion of the terminal MAP.

## 7. Costs and termination

- Physical actions advance `env_steps` by one and are bounded by
  `max_env_steps` (60).
- `READ_SIGN` costs 1 tool unit; `INSPECT` costs 2; total tool budget is 3.
- A tool action is one high-level decision and does not advance `env_steps`.
- Epistemic declarations and abstentions consume no tool units and no
  environment step; they are not free observations.
- The high-level decision count is the number of pre-action rounds; the
  final-observation opportunity is reported separately and is not a control
  decision.
- Success terminates the episode. Truncation (`max_env_steps`) and the
  decision bound also close the episode.
- No control action or tool action is permitted after termination. The one
  final observation is the already-returned terminal observation.

## 8. Scoring

The development set is the 300 static identification episodes on maps 0-9,
seeds 0-4. The denominator for every headline declaration rate is all 300
episodes; unresolved and unavailable count as failures.

Primary outcomes:

- **Task success:** goal reached / 300.
- **Terminal MAP accuracy:** final posterior MAP equals true terminal rule /
  300. Reported separately; never counted as a declaration.
- **Unconditional correct declaration rate:** episodes whose final retained
  declaration is correct / 300.
- **Declaration coverage:** episodes with any retained declaration / 300.
- **Incorrect declarations:** final wrong declarations and episodes with any
  wrong declaration event, reported separately.
- **Unresolved:** no retained declaration / 300.
- **Strict clean declaration accuracy:** final declaration correct **and** no
  wrong declaration event anywhere in the episode / 300.
- **Any-wrong-event rate:** episodes with at least one wrong declaration event /
  300.
- **Revisions:** number of revisions; episodes with a revision; final
  correctness conditional on revision.
- **Time to first correct declaration:** `env_steps` of the first correct
  declaration; right-censored if none.
- **Tool units / physical steps / decisions / return / cost.**
- **Per-rule and per-map outcomes**, with denominators stated.
- **Paired transitions** between methods on identical scenario ids.

`COMMIT`, probes, and declarations are separate events. A correct declaration
does not imply a correct terminal MAP and vice versa. A task success with a
wrong or absent declaration is a successful-misdiagnosis case, not an
identification success.

## 9. Development readiness criteria (recorded before outcomes)

These are **development targets**, not confirmatory gates. They are applied to
one frozen candidate run after the protocol and candidate are committed.

1. Overall unconditional declaration accuracy: at least **95%** (285/300).
2. Hazard final declarations: at least **90%** (45/50).
3. Sign-composition final declarations: at least **90%** (45/50).
4. Base task success: **100%** observed (50/50).
5. Zero raw zero-tool hazard/sign-composition task successes.
6. Tool spend at most **3 units** in every episode.

Also report, without treating them as pass/fail substitutions:

- whether the declaration-coverage change increases final wrong declarations
  or any-wrong events;
- task-success regressions versus the matched no-request control;
- strict clean accuracy;
- revisions and time to first correct declaration;
- per-map and per-rule variation.

Passing these targets does not establish confirmatory validity, does not
authorize the 2,700-episode holdout, and does not relax any frozen v1 gate.
Failure is reported as a negative development result; thresholds are not
moved and seeds are not selectively rerun.

## 10. Matched controls

The official development comparison contains:

- **A. Historical v0.3 and repaired v0.4** under their original protocols.
  These are descriptive cross-protocol baselines, not controlled causal
  estimates.
- **B. Repaired v0.4 control + v2 matched declaration readout.** Same v4
  navigation and tool policy, but declarations/abstentions/final readout
  follow v2. This isolates the added declaration opportunity from the
  separated request channel.
- **C. v0.5 separated epistemic/control policy.** v4 control behavior where
  possible, plus explicit evidence requests that may influence control.
- **D. v0.5 no-request ablation.** The same separated interface and readout,
  with active requests disabled; control falls back to the v4-compatible
  path.

Posterior updates, confidence thresholds, navigation, tool costs, evidence
ledger, observation stream, flags, and budget are kept as matched as the
implementation permits. Differences are reported as descriptive because B, C,
and D share the same scenario set but may take different trajectories.

## 11. Determinism and protection

- No new RNG source; all decisions use public observations and the existing
  deterministic environment draws.
- The candidate and controls reject maps outside the exposed development
  range, reject maps 210-239 explicitly, and validate the static manifest
  before executing episodes.
- Event ledgers are finite: one pre-action event per round plus at most one
  final event; declaration events capped at six; at most one pending request.
- No evaluator import in agent code; tests check that public decisions do not
  contain true-rule or hidden-role values.
- The official run records a candidate commit, configuration hash, manifest
  hash, and scenario ids; the run is not repeated selectively.

## 12. Preservation

This protocol does not modify the frozen v1 candidate, manifest,
preregistration, archive, PR #22, or any historical record. It creates a new
protocol version and a new candidate identity. The historical v0.4 result
remains a v1 result even if v2 produces different numbers on the same exposed
episodes.
