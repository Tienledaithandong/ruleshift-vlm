# Declaration semantics audit

**Scope:** audit of the frozen v3/v4 declaration object and its use in the
development records. This document does not alter the frozen protocol, the
candidate, the evaluator, or PR #22. All counts are from the 300 static
development episodes on maps 0-9, seeds 0-4.

## 1. The object that is currently recorded

A declaration is an entry appended by
`DiagnosticBeliefPolicy._record_commitment()`
(`src/ruleshift/agents/grid_baselines.py:673`). It is called only inside the
existing safe-commit branch of `choose()`:

```
if self._commit_is_safe():
    self._record_commitment(observation, info)
    return self._commit(observation)
```

The entry contains `step`, `declared_rule`, `confidence`,
`posterior_margin`, and `source="policy.safe_commit"`. The hook is
instrumentation around an existing control decision; it does not add an
epistemic action and does not change the returned action.

The record builder (`src/ruleshift/evaluator/grid_identification.py:331-370`)
then:

1. validates all events against the episode length;
2. sorts them by step;
3. treats every event except the last as an "earlier wrong" event when its
   rule differs from the true rule and its confidence/margin meet the
   thresholds;
4. reports the **last** event as `declared_rule`, `declared_step`,
   `declared_confidence`, and `declared_margin`;
5. assigns `commitment_status` from that last event: correct, wrong, or
   unresolved.

This means the current object is a **revocable last declaration**, not an
immutable first assertion. The distinction matters for the v3/v4 comparison.

## 2. Five distinct things that are often called "declaration"

| Concept | In the current code | Timed? | Counted as a declaration? | Audit finding |
|---|---|---|---|---|
| Terminal MAP estimate | `terminal_map`, derived after the episode from `terminal_posterior` | No | No | Must not be converted into a declaration. 34 v4 episodes have a correct terminal MAP but no correct final declaration. |
| Confidence-qualified rule declaration | An event at the safe-commit branch with top >= .90 and margin >= .10 | Yes | Yes, as the last event | The event is real and public-only, but it is a byproduct of a control commit. |
| Control commitment | `_commit(observation)` selected at the same branch | Yes, as an action | Not separately | The declaration and the commitment are the same branch; there is no declaration without a control commit. |
| Early termination without declaration | Episode ends by success or decision limit; no event | No | No | In v4 all 33 unresolved targets ended by success; no event exists, and the terminal posterior is not a declaration. |
| Incorrect explicit declaration | A final event whose rule differs from the true rule | Yes | Yes, `wrong_commit` | Four final cases each in v3 and v4; additionally, v3 has six earlier wrong events later revised. |

The five categories are not interchangeable. In particular, treating the
terminal MAP as a declaration would erase the timing and evidence conditions
that the protocol is supposed to test.

## 3. What the records say, at two levels

### 3.1 Final-event accounting (the frozen `commitment_status`)

| Method | Final correct | Final wrong | Unresolved | Declared episodes | Final precision |
|---|---:|---:|---:|---:|---:|
| v3 | 284 | 4 | 12 | 288 | 284/288 = 98.6% |
| v4 | 263 | 4 | 33 | 267 | 263/267 = 98.5% |

This is the accounting that produces the headline 284 vs 263. The four final
wrong episodes are the same four in both methods. The difference is almost
entirely unresolved coverage, not a change in final precision.

### 3.2 Event-level accounting (including revisions)

| Method | Any episode with a wrong explicit event | Final correct with an earlier wrong event | Final correct and clean (no earlier wrong) | Terminal MAP correct | Terminal MAP correct but not correctly declared |
|---|---:|---:|---:|---:|---:|
| v3 | 10 | 6 | 278 | 297 | 13 |
| v4 | 5 | 1 | 262 | 297 | 34 |

The v3 earlier-wrong cases are:

`m1-s2-remap_rotate`, `m2-s1-remap_rotate`, `m3-s0-slippery`,
`m3-s4-slippery`, `m5-s1-remap_rotate`, `m5-s2-remap_rotate`.

The v4 earlier-wrong case is:

`m2-s3-hazard_goal_swap`.

Consequences:

- **The 21-point final coverage gap is not a 21-point clean-accuracy gap.**
  Strict clean final correct is 278 vs 262, a 16-point gap, because v4 has five
  fewer revise-after-wrong episodes.
- At the event level, v4 makes **fewer** wrong explicit assertions than v3
  (5 vs 10 episodes), even though its final coverage is lower. The stability
  gate appears to suppress some early wrong assertions, but it also suppresses
  14 correct declarations and does not prevent the four final errors.
- A metric that reports only the last event rewards revision and hides the
  earlier wrong assertion. A metric that reports only the first event would
  make different episodes wrong. Both should be reported separately. Neither
  should be replaced by terminal MAP.

### 3.3 The four final incorrect declarations

| Episode | True | Declared | Confidence | Mechanism | Terminal MAP |
|---|---|---|---:|---|---|
| `m0-s3-sign_composition` | sign_composition | hazard_goal_swap | .9744 | noisy `INSPECT` contradicted true role | wrong (hazard) |
| `m5-s1-hazard_goal_swap` | hazard_goal_swap | sign_composition | .9744 | noisy `INSPECT` contradicted true role | wrong (sign) |
| `m1-s1-base` | base | slippery | .9143 | ambiguous movement/sign evidence; no contradictory tool | correct (base) |
| `m2-s2-base` | base | slippery | .9343 | movement-only ambiguous evidence; no contradictory tool | correct (base) |

These are explicit, time-stamped, confidence-qualified assertions, not
terminal estimates. They remain failures even when the terminal MAP later
recovers. In `m2-s2`, v4's control fallback makes the task succeed despite the
wrong declaration; task success does not repair the declaration.

## 4. Instrumentation coverage

- **Safe-branch coverage:** all 33 v4 unresolved episodes had zero decisions on
  which `_commit_is_safe()` returned true. There is no case of an eligible
  safe branch without an event. The hook is complete for the branch it wraps.
- **Event coverage:** every wrong final declaration has an event; no wrong
  status is inferred from a posterior without a time-stamped assertion.
- **De-duplication:** repeated same-rule events are suppressed. A rule change
  can append a new event; the final event wins in the record. This is why the
  event-level and final-event tables differ.
- **No separate abstention:** an episode with no event is `unresolved`; the
  record cannot distinguish "I deliberately abstain" from "the episode ended
  before a decision" from "the posterior never qualified." The traces can
  distinguish them (terminal-only vs insufficient vs stability), but the record
  cannot.
- **No separate evidence request:** `READ_SIGN` and `INSPECT` are ordinary
  control actions. The record does not say whether a request was an epistemic
  probe, a navigation detour, or a fallback.
- **No declaration without control commitment:** the hook is on the commit
  branch, so a high-confidence posterior at a non-commit decision cannot become
  an event. The 14 stability cases are exactly this situation.

Thus the hook has **no observed missed-call bug** and a **real semantic
coupling limitation**.

## 5. What would count as a correct declaration?

For the current protocol, the only defensible operational definition is:

> A time-stamped event emitted at a decision, before the episode ends, with a
> rule, confidence, and margin meeting the frozen thresholds, and with the
> event recorded at the safe-commit branch.

Under that definition:

- 263 v4 episodes have a correct final event;
- 33 have no event, and are unresolved regardless of terminal MAP;
- 4 have a wrong final event;
- 6 v3 and 1 v4 episodes have a correct final event preceded by a wrong event,
  which final-event accounting hides.

The diagnosis uses this definition. It does **not** count a terminal MAP, a
plan, a task success, or a post-hoc reinterpretation as a declaration.

## 6. Recommended reporting changes (not applied to the frozen protocol)

For any future protocol version, report at least:

1. **Unconditional coverage:** correct declarations / all episodes (the
   current G1 denominator), so abstention cannot be hidden.
2. **Opportunity-conditional coverage:** correct declarations / episodes with
   at least one decision meeting the confidence rule; report the denominator.
3. **Event-level accuracy:** first event, last event, any wrong event, and
   revision count, all separately.
4. **Abstention reasons:** explicit `insufficient_evidence`,
   `terminal_only`, `ambiguous_tie`, `no_decision_remaining`.
5. **Terminal report as a separate field:** never folded into declaration
   coverage.
6. **Time-stamped calibration:** confidence at declaration, not terminal
   confidence.

These changes require a new protocol version and human approval. They must not
be applied retroactively to the frozen v0.4 records.

## 7. Audit conclusion

The current declaration instrumentation is faithful to the **control-commit
branch** and is not missing eligible calls. It is not a faithful representation
of **epistemic declaration** as a separate act. The 33 unresolved episodes
contain at least 23 episodes that v3 declared correctly; 14 of those are
blocked only by the two-consecutive-leader condition, and 9 by v4's earlier
success. The four final wrong declarations are unchanged and are evidence/
commitment failures, not instrumentation failures.

Any change to this semantics must be a new protocol version. The proposal is in
`docs/EPISTEMIC_CONTROL_SEPARATION.md` and `docs/NEXT_CANDIDATE_PROPOSAL.md`;
no frozen object is modified by this audit.
