# Declaration diagnosis decision packet

**Decision:** NO-GO for the confirmatory holdout. Close repaired v0.4 as a
development-stage feasibility failure under the current protocol. Preserve the
frozen candidate, manifest, preregistration, archive, and PR #22 as historical
records.

## What was established

- All 33 v4 unresolved static episodes were replayed with public per-decision
  traces, plus matched v3 runs (74 episodes total).
- **No instrumentation miss:** 0/33 had a safe-commit-eligible decision
  without an event. All 4 incorrect declarations had an event.
- The 33 unresolved are: **14 stability gate**, **11 terminal-only evidence**,
  **8 insufficient evidence**. None ended at the decision limit.
- All 14 stability cases had a qualified posterior at a decision but only one
  leader observation; all 14 were correctly declared by v3.
- 9 more new losses are early-success cases: 5 terminal-only and 4
  insufficient were v3 correct. The other 10 unresolved were unresolved in v3.
- The same four episodes are the final incorrect declarations in v3 and v4,
  with the same declared rules: 2 rational updates after a 5% noisy `INSPECT`,
  2 premature commitments on ambiguous evidence. The stability gate prevented
  none of them.

## Trade-off in one table

| Metric | v3 | v4 | Change |
|---|---:|---:|---:|
| Task success | 287/300 | **298/300** | +11, no regressions |
| Final correct declaration | **284/300** | 263/300 | -21 |
| Final incorrect declaration | 4/300 | 4/300 | 0 |
| Unresolved | 12/300 | 33/300 | +21 |
| Terminal MAP correct | 297/300 | 297/300 | 0 |
| Any wrong explicit event | 10 episodes | 5 episodes | -5 |
| Strict clean final correct | 278/300 | 262/300 | -16 |

The task gain is control repair (slippery/base plan fallback), not improved
identification. Final declaration precision is nearly unchanged (98.6% vs
98.5%); the failure is coverage. Terminal MAP must not be converted into a
declaration; doing so would retroactively produce 297/300 and erase the
timing/evidence requirement.

## Recommendation

Propose **one** design-only protocol candidate,
`v0.5-separated-epistemic`: explicit, non-terminal `DECLARE_RULE`,
`ABSTAIN`, and `REQUEST_EVIDENCE` channels; control `COMMIT`,
`PROBE_SIGN`, `PROBE_BLUE` separated from declaration; drop the
two-consecutive-leader gate; retain the confidence rule and v4 control repairs;
require evidence references and corroboration for noisy inspect; never count a
terminal report as a declaration.

This is not an authorization to implement or tune. It requires a new protocol
version, new tests, a new preregistration, and separate human approval. No
candidate-map or holdout episode may be run. If the candidate cannot meet the
proposed guardrails, close it and do not try another threshold patch.

## Deliverables

- `docs/DECLARATION_FAILURE_DIAGNOSIS.md`
- `docs/DECLARATION_SEMANTICS_AUDIT.md`
- `docs/EPISTEMIC_CONTROL_SEPARATION.md`
- `docs/NEXT_CANDIDATE_PROPOSAL.md`
- `docs/DECLARATION_FAILURE_EPISODES.md`
- `docs/declaration_failure_traces_v1.jsonl`
- `scripts/diagnose_declaration_failures.py`
- `src/ruleshift/evaluator/declaration_diagnosis.py`
- `tests/test_declaration_diagnosis.py`

## Next action

Human decision: authorize a separately versioned v0.5 protocol/candidate
development cycle, or close the static-identification candidate line. In
either case, do not promote v0.4 to confirmation and do not touch PR #22.
