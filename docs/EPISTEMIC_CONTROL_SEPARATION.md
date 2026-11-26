# Epistemic-control separation

**Status:** design analysis. No protocol or candidate is changed by this
document. The frozen v0.4 protocol remains the only evaluated protocol.

## 1. The current coupling

The frozen policy has one decision channel. At each `choose()` call it selects
a grid action (move, `READ_SIGN`, `INSPECT`, or a plan step). A declaration is
not an action; it is a side effect of entering the safe-commit branch:

```
choose()
  -> update posterior
  -> if _commit_is_safe():
         _record_commitment()     # declaration event
         return _commit()         # control plan
  -> else choose a probe / move
```

The existing labels therefore bundle four different questions:

| Question | Current representation | Consequence |
|---|---|---|
| Do I know the rule well enough to assert it? | Only if I also choose to commit control | A high-confidence posterior without a control commit is invisible |
| Do I choose to act on a rule now? | `_commit()` plan | Commitment can occur only with a declaration event |
| Do I need more evidence? | An ordinary probe action | No explicit request or reason |
| Do I decline to assert? | No event | `unresolved` conflates abstention, no opportunity, and insufficient evidence |

This is why v4 can improve task success while losing declarations: the control
repair changes when the episode ends, and the declaration is attached to the
control event rather than to the epistemic state.

## 2. Evidence that the coupling matters

The 37 target episodes provide three distinct coupling failures.

### 2.1 A qualified belief with no commit opportunity

In 14 episodes, the posterior met the frozen confidence rule (top >= .90,
margin >= .10) at a decision, but v4's two-consecutive-leader condition made
the safe-commit branch false. The next action reached the goal, so no second
decision existed. All 14 were correctly declared by v3. Examples:
`m1-s0-remap_swap` (top .947, run 1), `m0-s3-slippery` (top ~1.0, run 1),
and `m5-s0-sign_composition` (top .974 after the full 3-unit tool budget,
run 1). The belief was qualified; the control branch was not eligible.

### 2.2 A qualified terminal belief with no timed assertion

In 11 episodes, no decision met the rule, but the terminal posterior did. The
episode ended by success. Five were v3 correct because v3's path had one more
decision; six were unresolved in both. The terminal estimate is a post-episode
summary, not a time-stamped declaration. Allowing a "terminal declaration"
would be retroactive and would make the metric incomparable with v3.

### 2.3 A wrong declaration forced by a control commitment

In two episodes, a single noisy `INSPECT` moved the posterior to .974; the
policy then committed and declared the wrong rule. The inspect error rate is
public (5%) and is included in the likelihood, so the posterior is rational.
The failure is not a missing update; it is a commitment after one noisy
observation with no remaining budget for corroboration. In two other episodes,
ambiguous movement/sign evidence crossed the threshold and the policy committed
and declared too early. The two-leader gate did not block either premature
case, because both had run >= 2.

These are different mechanisms, but they share one structural cause: the
protocol has no way to say **"I have a qualified belief"**, **"I do not have
enough evidence"**, or **"I need this specific evidence"** independently of
**"I will now execute a control plan."**

## 3. Proposed separation

This is a protocol design proposal, not an implementation.

### 3.1 Decision channels

At each decision, expose two channels:

**Epistemic output** (one of):

- `DECLARE_RULE(rule, confidence, margin, evidence_refs)`
- `ABSTAIN(reason)`, where reason is one of
  `insufficient_evidence`, `ambiguous_tie`, `terminal_only`,
  `no_decision_remaining`, `noisy_evidence_unconfirmed`;
- `REQUEST_EVIDENCE(channel, target, purpose)`, where channel is
  `SIGN` or `INSPECT` and target is the public target cell.

**Control output** (one of):

- `COMMIT(plan)` — execute a plan; may be emitted with or without a
  declaration;
- `PROBE_SIGN(target)` / `PROBE_BLUE(target)` — execute an evidence tool;
  this is the control realization of a request;
- `NAVIGATE(action)` — ordinary movement or a plan prefix;
- `STOP` — end only when the environment's terminal condition is met.

The two channels are recorded separately. A declaration is **non-terminal** and
does not by itself change the environment action. A commitment is **not** a
declaration. A probe is **not** a declaration and does not imply a commitment.

### 3.2 Declaration rule

For a first, minimal candidate:

- Keep the frozen quantitative thresholds: top >= 0.90 and margin >= 0.10.
- Remove the **two-consecutive-decision** condition. The diagnosis shows it is
  a control-stability filter, not an evidence-quality filter: it cost 14 clean
  declarations, all v3 correct, while the four final wrong declarations all
  satisfied the leader-run condition.
- Require at least one public evidence event supporting the declared rule, and
  record the evidence references in the event.
- If the only decisive support is a single `INSPECT` and a corroborating
  observation is available within budget, either request corroboration or
  abstain with `noisy_evidence_unconfirmed`. Do not silently treat a 5%-error
  tool as certain after one sample.
- Do not declare from a terminal posterior. If the last decision has no
  qualified belief, emit `ABSTAIN(terminal_only)` or `ABSTAIN(insufficient)`.
  A post-episode `terminal_report` is a separate field, never a declaration.

The exact evidence-sufficiency rule is deliberately left to the new protocol
preregistration; this document does not tune it on the 300 development
episodes.

### 3.3 Control rule

Keep the v4 control repairs for a fair comparison:

- safe-plan fallback for slippery/base loops;
- the existing probe/expected-information selection;
- the existing budget and navigation rules.

Control may commit without a declaration when the plan is safe but the rule is
not epistemically qualified. Conversely, the agent may declare and continue
probing if the declaration is non-terminal and the protocol allows it. The
control objective (reach the goal) and the identification objective (declare
the rule) are scored separately.

### 3.4 Early success

Separation does not make terminal evidence time-stamped. To reduce the 9
early-success losses, the new policy needs an explicit choice **before** the
final move:

- if the belief is qualified, declare at that decision even if the next action
  ends the episode;
- if it is near threshold and the remaining budget allows, request the
  highest-value evidence before the final move;
- otherwise abstain explicitly and do not pretend the terminal MAP was a
  declaration.

This can trade task steps for evidence. That trade-off must be measured, not
assumed. The current diagnosis does not authorize that experiment.

## 4. Evaluation semantics for the separated protocol

A separated protocol needs metrics that cannot be gamed by changing the
termination point or by re-labeling terminal estimates.

| Metric | Definition | Why |
|---|---|---|
| Unconditional declaration coverage | correct declarations / all episodes (300) | Preserves the current identification denominator; abstention is still a failure |
| Opportunity-conditional coverage | correct declarations / episodes with >= 1 qualified decision | Separates evidence failure from no-opportunity failure |
| Time-stamped precision | correct declarations / all declarations, at event time | Does not allow revision to hide earlier errors |
| First-event accuracy | correctness of the first declaration | Measures the cost of premature assertion |
| Last-event accuracy | correctness of the final declaration | Comparable to the frozen record, but reported alongside first-event |
| Revision count | number of rule changes after a declaration | Measures instability, not just final correctness |
| Wrong-event rate | episodes with any wrong declaration | Penalizes trial-and-error declarations |
| Abstention calibration | abstain rate by reason vs actual error rate | Tests whether abstention is used honestly |
| Task success | goal reached / episodes | Separate control objective |
| Terminal report accuracy | terminal MAP correctness | Reported separately; never counted as declaration |

The existing 300-episode denominator should remain for the headline. A
conditional metric is diagnostic, not a substitute: it must always be reported
with its denominator.

## 5. Why this is a new protocol, not a retrospective edit

The frozen v0.4 records have no `DECLARE_RULE`, `ABSTAIN`, or
`REQUEST_EVIDENCE` field. Reinterpreting their terminal MAPs, probe actions, or
control commitments as those objects would be post hoc. It would also change
the measured 263/300:

- adding terminal MAP as a declaration would retroactively count 34 v4
  episodes and produce 297/300, without any time-stamped assertion;
- treating every no-commit decision as an abstention would change the
  denominator and make v3/v4 incomparable;
- treating every probe as a request would not distinguish navigation from
  evidence gathering.

The correct path is a **new protocol version** (working name `v0.5`) with new
events, new tests, and a new preregistration. The frozen candidate, manifest,
preregistration, archive, and PR #22 stay unchanged as historical records.

## 6. Risks and safeguards

- **Earlier declarations increase wrong-event exposure.** The 14 stability
  cases were all correct in v3, but a different sample can contain a noisy
  single observation. Require evidence references and corroboration when the
  decisive evidence is a single error-prone tool.
- **Noisy inspect is not solved by separation.** Two of the four final wrong
  declarations were caused by a 5% inspect error. The new protocol must model
  redundancy or explicitly abstain; it must not simply lower the threshold.
- **Early evidence gathering can reduce task success.** Requesting a probe
  before the final move costs steps. Measure both objectives and report the
  trade-off.
- **Abstention can hide failures.** Keep the unconditional 300-episode
  denominator; an abstention is not a correct declaration.
- **Revision can look like accuracy.** Report first-event accuracy and
  revision count; do not count only the last event.
- **No free architecture change.** The separated protocol is a new candidate
  with its own code, tests, and freeze; it cannot be merged into the v0.4
  evaluation.

## 7. Conclusion

The current protocol's declaration is a control-commit side effect. That
coupling explains the observed trade-off better than any single threshold:
v4's control repair raises task success to 298/300, while the same trajectory
changes and the added stability filter reduce final declarations to 263/300.
The four final errors are evidence/commitment errors, not hook misses.

A separated epistemic/control interface is the only intervention supported at
the protocol level. It should be designed and tested as a new version, with
explicit abstention and evidence requests, no retroactive terminal
declarations, and separate reporting of task success and declaration coverage.
