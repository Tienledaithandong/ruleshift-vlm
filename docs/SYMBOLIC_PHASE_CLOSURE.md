# Symbolic Phase Closure

**Status:** closed for confirmatory purposes.
**Date:** 2026-11-26.
**Scope:** RuleShift symbolic identification development, v0.3 through v0.5.
**Decision:** retain v0.5 as a development reference, do not promote it as
confirmatory evidence, and defer the 900-episode reserved-map confirmation.

This document closes the symbolic candidate-development cycle. It does not
rewrite any frozen protocol, result, manifest, test, or historical decision.
Where an earlier document contains an error, the error is recorded here and the
original artifact is left intact.

## 1. What is retained

The following are retained without modification:

- PR #22 and PR #23, including their draft state and CI history.
- Frozen commits, manifests, results, tests, and historical decision records.
- The archived v0.5 official result folder, including records.jsonl and
  summary.json.
- Maps 210-239 as reserved, unauthorized, and unused.

The following are **not** retained as evidence:

- v0.5 is not a confirmatory result.
- No v0.6 is authorized.
- No reserved-map rollout is authorized.
- No confirmatory evaluation, model inference, or GPU run is authorized by this
  closure.

## 2. Historical development record

The earlier symbolic phase established the following development picture. These
numbers are historical and descriptive; they are not confirmatory claims.

| Stage | Recorded result | Disposition |
| --- | --- | --- |
| v0.3 repaired phase 0 | 534/550 (97.1%) on the 10-map CPU replication | development reference only; failed the base gate in its original protocol |
| v0.4 repaired holdout | 546/550 overall; task 298/300, MAP 297/300, joint 298/300 in the recorded table | development reference only; not promoted |
| v0.5 final development | see the v0.5 table below | development reference only; no confirmation |

The v0.3/v0.4 details remain in `docs/GRID_PHASE0_10MAP.md` and
`docs/GRID_V04_HOLDOUT.md`. Those files are historical records, not inputs
to a new promotion decision.

## 3. v0.5 official development summary

Official file: `results/grid_v05_official_final/`.

The archived 900-episode file has three method labels: v4_readout_v2, v5, and
v5_no_request. The B/C/D comparison in the closure decision refers to v5,
v5_no_request, and a replay of v5 (D duplicates B). In the archived file, v5 and
v5_no_request are identical on every reported aggregate and every episode-level
comparison used below. The embedded v4_readout_v2 arm also reports the same
final aggregate counts in this file; this is a property of the archived readout
and does not erase the separate repaired-v4 replay described in section 5.

| Quantity | v5 (B) | v5_no_request (C) | v4_readout_v2 (archived) |
| --- | ---: | ---: | ---: |
| Episodes | 300 | 300 | 300 |
| Task success | 298/300 | 298/300 | 298/300 |
| Terminal MAP correct | 297/300 | 297/300 | 297/300 |
| Joint task and MAP correct | 297/300 | 297/300 | 297/300 |
| Final declaration correct | 289/300 | 289/300 | 289/300 |
| Final declaration wrong | 3/300 | 3/300 | 3/300 |
| Final unresolved | 8/300 | 8/300 | 8/300 |
| Declaration coverage | 292/300 | 292/300 | 292/300 |
| Unconditional declaration accuracy | 96.33% | 96.33% | 96.33% |
| Strict clean | 282/300 (94.0%) | 282/300 (94.0%) | 282/300 (94.0%) |
| Episodes with any wrong declaration | 10 | 10 | 10 |
| Tool units total / mean | 383 / 1.2767 | 383 / 1.2767 | 383 / 1.2767 |
| Mean environment steps | 8.21 | 8.21 | 8.21 |
| Mean decisions / maximum | 9.1467 / 62 | 9.1467 / 62 | 9.1467 / 62 |

The six registered point-estimate readiness criteria are recorded as nominally
passed. This is a **nominal point-estimate pass**, not a confidence-interval
gate and not a confirmatory result. The corresponding Wilson lower bounds
remain below the registered targets. That fact is recorded as a limitation; it
is not used to replace the point criteria after the run, and it does not turn
the nominal pass into a post-hoc failure.

The 900-episode reserved-map confirmation is deferred. No reserved map was used
to produce the table above.

## 4. Active REQUEST_EVIDENCE: no observed behavioral effect

The active request arm (`v5_no_request` compared with `v5`) produced no
observed behavioral effect on the evaluated development scenarios.

- C recorded 445 request events: 281 executed, 84 expired, 80 unmet.
- B, C, and D have identical action traces, tool traces, outcomes, and final
  declarations on the evaluated development scenarios.
- D duplicates B.
- The request machinery therefore did not change the decisions observed in this
  development set. The count of request events is not evidence that the policy
  used requests to improve control.

This is a null development finding. It is retained as such; it is not repaired
by retroactively adding a request-sensitive policy or a new v0.6.

## 5. Repaired-v4 replay and the paired comparison

A separate repaired-v4 replay, scored with its own v4 readout, recorded:

| Quantity | Repaired v4 |
| --- | ---: |
| Task success | 298/300 |
| Terminal MAP correct | 297/300 |
| Final declaration correct | 263/300 |
| Final unresolved | 33/300 |
| Final declaration wrong | 4/300 |
| Strict clean | 262/300 |
| Episodes with any wrong declaration | 5 |

The recorded paired comparison v4 -> B is: +26 correct final declarations,
-25 unresolved, -1 wrong; task success and terminal MAP are invariant.

The comparison is a development comparison. It is not a confirmatory estimate,
and it does not establish that the change would generalize beyond these
development scenarios. The archived v4_readout_v2 arm and the separate repaired
v4 replay are different records; both are preserved. No attempt is made here to
reconcile their provenance after the fact.

## 6. The m3-s2 correction

Scenario m3-s2 is a true slippery episode. Under v5 the recorded trajectory is
`DOWN`, `DOWN`: 2 environment steps, 0 tool units, task success. The terminal
MAP is remap_rotate, and the final declaration is remap_rotate. Relative to the
repaired-v4 record, this is one new final wrong declaration: v5 committed to a
wrong rule, whereas v4 left the same MAP unresolved.

The official terminal posterior for m3-s2 under the archived results is:

| Rule | Terminal posterior |
| --- | ---: |
| base | 0.0029069767441860465 |
| hazard_goal_swap | 0.0029069767441860465 |
| remap_rotate | 0.9418604651162791 |
| remap_swap | 0.0029069767441860465 |
| sign_composition | 0.0029069767441860465 |
| slippery | 0.046511627906976744 |

The values sum to 1.0. Earlier secondary documentation for PR #23 reports
slippery = 0.1538461538 for this episode. That value is the first-step
posterior (0.2 / 1.3 = 0.1538461538), not the terminal posterior. It is not a
typo in the terminal record and it is not silently rewritten here. PR #23 is
preserved as published; this closure document supplies the correction and the
provenance explanation.

The episode is a concrete failure mode for the closed symbolic phase: with no
tool use and a two-step DOWN trajectory, the terminal MAP was confidently wrong
on a true slippery map. This is a reason to study decision quality under
constrained information, not a reason to develop v0.6.

## 7. Why the symbolic cycle is closed

The v0.5 development record is strong enough to serve as a reference and weak
enough that promotion would overstate it:

- The point estimates are high, but the lower confidence bounds do not meet the
  registered targets.
- The active-request manipulation produced no observed behavioral difference.
- The remaining errors include confident wrong declarations and unresolved
  terminal MAPs.
- The archived records are internally consistent only when the readout
  provenance is kept explicit; a post-hoc relabeling would damage the record.
- The reserved-map confirmation remains the correct place for a stronger claim,
  and it is explicitly deferred.

The correct closure is therefore: retain, document, do not promote, do not
rerun with a new candidate, and do not spend reserved maps on a development
question.

## 8. Preservation ledger

| Artifact | Identifier |
| --- | --- |
| v0.5 candidate commit | 751a3414ad8815087f64d85ad5a1523af11077f4 |
| v0.5 documentation commit | bc79ce1d7a54aea38fa7502452d646450121c09a |
| PR #23 | draft; recorded CI passed; preserved |
| PR #22 | draft/open; head a3e614b093dfd8237898cb19ac83cab92b8bad50; preserved |
| PR #21 | superseded; preserved |
| records.jsonl SHA-256 | 00B6F22A5C39AE2BE14C1C422ADC57C992C77F73B72761A6532BBEE471879EBD |
| summary.json SHA-256 | B2D8D18611EFA9AC15FE7DBEF369760986DC81A15666389FE64254A62805C922 |

No merge, rebase, force-push, or history rewrite is part of this closure.

## 9. Authorization boundary

- Maps 210-239 remain reserved and unauthorized.
- No reserved-map rollout, confirmatory experiment, GPU run, or VLM inference is
  authorized by this document.
- No v0.6 is authorized.
- No frozen protocol is modified.
- No automatic merge is authorized.

The next research direction is the VLM probe-versus-commit decision
microbenchmark. Its CPU infrastructure is documented separately and remains
development-only until a separate human GO.
