# Next candidate proposal (design only)

**Decision:** close repaired v0.4 as a confirmatory candidate. Propose **one**
new development candidate, `v0.5-separated-epistemic`, and do not implement or
tune it without separate human approval.

This is a proposal, not an authorized experiment. No candidate-map episode,
holdout episode, GPU/VLM call, policy edit, protocol edit, or merge was made.

## 1. Why v0.4 stops here

Under the current protocol, v0.4 has:

- task success 298/300 (v3: 287/300);
- final declaration 263/300 (v3: 284/300);
- final incorrect declaration 4/300 (v3: 4/300);
- terminal MAP 297/300 (v3: 297/300).

The task gain is real and comes from control repair. The declaration loss is
not a hook miss: all 33 unresolved episodes had zero safe-commit-eligible
decisions. It is a coverage failure caused by the declaration being attached to
control commitment:

- 14 losses: the confidence rule was met, but the two-consecutive-leader gate
  blocked the commit and the episode succeeded next step;
- 9 additional losses: v4 reached the goal earlier than v3, so v3's later
  declaration opportunity disappeared (5 terminal-only, 4 insufficient);
- 10 remaining unresolved episodes were unresolved in v3 too;
- 2 gains were different-path successes.

The four final errors are unchanged from v3: two rational declarations after a
5% noisy `INSPECT`, and two premature commitments on ambiguous movement/sign
evidence. The two-leader gate did not prevent any of them.

A same-protocol patch (for example, simply deleting the two-leader gate, or
counting terminal MAP) is not justified: it would not address the 9
early-success losses, could increase wrong events, and terminal MAP conversion
would be retroactive. The diagnosis supports a **protocol-level** intervention,
not threshold tuning.

## 2. The one proposed candidate

**Working name:** `v0.5-separated-epistemic`.

**Core hypothesis:** if declaration is a non-terminal epistemic action with
explicit abstention and evidence requests, the policy can recover qualified
declarations currently blocked by control commitment and early success, without
increasing first-event or final wrong declarations beyond the v3 baseline.

**Control policy:** the v4 control repairs (safe-plan fallback for
slippery/base loops, expected-information probes, existing budget and
navigation). The candidate tests the epistemic interface, not a new planner.

**Epistemic policy:** explicit `DECLARE_RULE`, `ABSTAIN`, and
`REQUEST_EVIDENCE` channels, as specified in
`docs/EPISTEMIC_CONTROL_SEPARATION.md`.

### 2.1 Declaration behavior

At each decision, after the public posterior update:

1. If top >= 0.90 and margin >= 0.10 and at least one public evidence event
   supports the leader, emit `DECLARE_RULE(rule, confidence, margin,
   evidence_refs)`.
2. Declarations are **non-terminal**. They do not force a control commit and
   do not end the episode.
3. Do not require two consecutive decisions. The diagnosis shows this filter
   is a control-stability condition, not an evidence condition.
4. If a declaration is qualified only because of a single `INSPECT`, and the
   remaining budget allows a corroborating observation, prefer
   `REQUEST_EVIDENCE` or `ABSTAIN(noisy_evidence_unconfirmed)`; do not
   assert certainty from one error-prone sample.
5. If the episode ends before a qualified decision, emit
   `ABSTAIN(terminal_only)` or `ABSTAIN(insufficient_evidence)` at the last
   decision. A terminal report is recorded separately and is never counted as
   a declaration.
6. If two rules tie or the margin is below the threshold, emit
   `ABSTAIN(ambiguous_tie)`; do not commit a declaration on a coin flip.

### 2.2 Evidence behavior

- `REQUEST_EVIDENCE(channel, target, purpose)` selects the public sign or
  blue-cell inspection with the highest expected information gain.
- The control layer executes it as `PROBE_SIGN` or `PROBE_BLUE`; the
  epistemic request is recorded separately.
- Requests are budget-aware. If no remaining budget can change the decision,
  emit an abstention rather than spending a tool without a purpose.
- A request is not a declaration and does not imply that the next action is a
  commit.

### 2.3 Control behavior

- `COMMIT(plan)` may be emitted with or without a declaration.
- The planner keeps the v4 fallback for slippery/base loops.
- Task success and declaration coverage are scored separately; a control
  success does not repair a wrong declaration, and a wrong declaration does not
  retroactively invalidate a task success.
- No new thresholds, rule-specific bonuses, or map-specific logic.

## 3. What the candidate is expected to recover, and what it cannot

Mechanism-based expectations, not measured results:

| Failure class | Count | Expected effect of separation |
|---|---:|---|
| Stability gate | 14 | Directly recoverable: declare at the qualified decision instead of waiting for a second one |
| Terminal-only | 11 | Only recoverable if the policy declares before the final move or requests evidence earlier; no retroactive terminal declaration |
| Insufficient evidence | 8 | Honest `ABSTAIN`; some may become declarations if the policy chooses evidence over an early goal, but this trades task steps |
| Noisy inspect | 2 | Requires corroboration or explicit uncertainty; separation alone does not remove the 5% error |
| Premature ambiguous | 2 | Requires evidence sufficiency, not a lower threshold; may remain wrong |

The candidate must not be credited with the 11 terminal-only/insufficient
recoveries unless the trajectory actually contains a time-stamped declaration.

## 4. Evaluation plan and guardrails

Use only the same development data: maps 0-9, seeds 0-4, 300 static episodes.
CPU only. Maps 210-239 remain reserved and must not be touched.

### 4.1 Primary endpoints

1. **Unconditional declaration coverage:** correct declarations / 300. This is
   the headline and preserves the current denominator.
2. **First-event accuracy:** correctness of the first declaration; this is the
   cost of premature assertion.
3. **Any-wrong-event rate:** episodes with at least one wrong declaration;
   revisions do not hide it.
4. **Final-event accuracy:** comparable to the frozen record, reported
   alongside first-event.
5. **Task success:** separate control endpoint.
6. **Terminal report accuracy:** reported separately, never summed into
   declaration coverage.
7. **Abstention calibration:** abstain reason vs actual error; no hidden
   conversion of `ABSTAIN` into success.

### 4.2 Proposed go/no-go guardrails for the development candidate

These are proposed for human approval before implementation; they are not
current gates and no gate has been relaxed.

- **Task:** no worse than v4's 298/300; if separation costs task success, report
  the trade-off and do not claim a control improvement.
- **Coverage:** correct final declarations >= 284/300 (the v3 level); a
  protocol that merely relabels abstentions does not pass.
- **Wrong assertions:** any-wrong-event episodes <= 10/300 (the v3 event-level
  baseline); final wrong <= 4/300.
- **No retroactive credit:** terminal report and post-episode MAP are excluded
  from all declaration endpoints.
- **Reproducibility:** deterministic replay, paired scenario-level tables,
  trace schema tests, and a new preregistration.

If the candidate cannot meet these guardrails, stop and close it. Do not tune
on the 300 episodes and do not open the reserved maps.

### 4.3 Required implementation artifacts (after approval only)

- new protocol version and action/event schema;
- pure trace exporter with time-stamped declaration, abstention, request, and
  control fields;
- paired scorer for unconditional and conditional coverage, first/last event,
  revisions, and task;
- tests for event ordering, no terminal conversion, non-terminal declaration,
  explicit abstention, and evidence-request/control separation;
- a new freeze manifest and preregistration before any candidate-map use.

## 5. Why only one candidate

The diagnosis identifies one generalizable mechanism (declaration-control
coupling) and one control repair already present in v4. Proposing a second
candidate in parallel would confound the mechanism and invite threshold
tuning. If this candidate fails, the correct conclusion is that the current
static-identification formulation is not feasible at the required coverage,
not that another threshold should be tried.

## 6. Approval boundary

This proposal does **not** authorize:

- implementation or tuning;
- candidate-map rollouts;
- the 2,700-episode confirmatory experiment;
- GPU/VLM runs;
- changes to the frozen candidate, manifest, preregistration, archive, or
  PR #22;
- merging the diagnosis branch.

The next action is a human decision on whether to authorize a separate v0.5
protocol/candidate development cycle. Until then, the diagnosis is the
deliverable.
