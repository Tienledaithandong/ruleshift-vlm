# v0.4 base-policy candidate: protocol

**Status: exploratory candidate. Not a new confirmatory benchmark, not a gate
change, and not a Phase 1 authorization.**

## Motivation

The v0.3 base-failure trace identified three policy mechanisms:

1. when the MAP rule's plan is empty, the policy emits the first legal
   physical action; on `m2-s2` and `m57-s4` this produced 60-step loops;
2. `READ_SIGN` always tries side 0 first and stores only one bit, so side 1
   is never read after side 0; on `m78-s1` this left `base` versus
   `remap_rotate` unresolved;
3. the commit test can act on one-step posterior spikes, so a wrong MAP rule
   can be committed repeatedly.

The candidate fixes (1) and (2), and adds a conservative stability condition
to (3). It does not change the environment, the map generator, the 100% base
gate, or the v0.3 policy.

## Candidate changes

- **Plan fallback.** If the MAP rule's plan is empty, plan under `base`; only
  if that is also empty, use the old first-legal-action fallback. The fallback
  does not bypass the MAP belief permanently; the normal belief update still
  runs after each move.
- **Per-side sign memory.** Track the set of sign faces already read, not just
  one boolean. `READ_SIGN` chooses an unread face by expected information
  gain over the current posterior, so side 1 is available when it separates
  `base` from `remap_rotate` or `remap_swap`.
- **Stable commit.** Require the same MAP rule to lead on two consecutive
  decisions before committing, in addition to the existing `top >= 0.90` and
  `top - second >= 0.10` thresholds. This is intended to stop one-step
  oscillation; it is not a change to the probability thresholds.

All three changes live in a new `DiagnosticBeliefPolicyV4`;
`DiagnosticBeliefPolicy` remains byte-for-byte the v0.3 policy.

## Holdout

- Maps: generator map indices **100-199**, which are not part of the v0.3
  10-map replication, the 100-map base-stress study, or the 100-map audit.
- Seeds: 0-4, the same five seeds as the earlier CPU studies.
- Episodes: 100 maps x 5 seeds x 1 true-base identification scenario = 500.
- Policies: v0.3 diagnostic, v0.4 candidate, and `no_adaptation`.
- Primary endpoint: task failure on the true-base episodes.
- Secondary endpoints: base misdiagnosis, tool units, steps, and paired
  differences versus v0.3 and `no_adaptation`.

The holdout is fixed before the v0.4 code is run. If the candidate fails, do
not edit it and rerun these maps; a later candidate must use a new map offset.

## Decision rules

- The literal 100% base gate still applies to v0.3 as the benchmark policy.
  A v0.4 result cannot retroactively pass it.
- If v0.4 has **any** task failure on 500 holdout episodes, the literal 100%
  gate still fails for v0.4 and Phase 1 stays blocked.
- If v0.4 has 0/500 task failures, report the Wilson and map-cluster upper
  bounds. This is evidence, not proof of a 100% population rate, and does not
  freeze or promote the candidate.
- Misdiagnosis is reported separately. An early success can still end with a
  wrong or tied final belief; ties are called out rather than silently
  credited as correct.
- No prompt, benchmark, environment, or gate tuning after seeing holdout
  outcomes.

## Reproduce

```bash
python scripts/run_grid_v04_holdout.py
```

Outputs are ignored under `results/grid_phase0/`; only code and docs are
committed.