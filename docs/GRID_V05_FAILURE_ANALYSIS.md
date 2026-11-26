# v0.5 failure analysis: the four shared v0.3/v0.4 failures

Status: development analysis; not confirmatory evidence.

This document analyses the four episodes that fail under both v0.3
(DiagnosticBeliefPolicy) and v0.4 (DiagnosticBeliefPolicyV4) on the ten-map
development grid (maps 0-9, seeds 0-4). The machine-readable trace was produced
by:

    python scripts/trace_grid_failures.py --out results/grid_v05_dev/traces.json

The script records privileged true-rule and belief fields for analysis only; it
is not used by any policy. The trace artifact is under the git-ignored results/
tree. All four action/outcome sequences are identical for v0.3 and v0.4; the
tables below are the v0.3 traces and therefore document both methods.

## 1. Summary

The four failures are two identification errors caused by a single noisy tool
observation, and two post-shift failures caused by a stale pre-shift sign
observation. They are not map-generation or evaluator defects. In all four
cases the policy had enough public evidence available to recover, but did not
use it: movement outcomes (safe, hazard, goal) were not treated as rule
evidence, and a noisy INSPECT was allowed to dominate an unresolved sign
ambiguity.

The v0.5 candidate adds deterministic cell-role feedback from movement
outcomes. It recovers all four development episodes without adding tools or
changing the benchmark. The v0.5 result is a development result on exposed
maps; it is not a gate result.

| Scenario | True rule | v0.3/v0.4 end state | v0.5 end state | v0.3/v0.4 tools | v0.5 tools |
| --- | --- | --- | --- | ---: | ---: |
| m0-s3-identification-sign_composition-None | sign_composition | failure, hazard_goal_swap, 60 steps | success, sign_composition, 24 steps | 3 | 3 |
| m5-s1-identification-hazard_goal_swap-None | hazard_goal_swap | failure, sign_composition, 60 steps | success, hazard_goal_swap, 6 steps | 3 | 3 |
| m7-s2-change_detection-base-sign_composition | sign_composition after shift at step 10 | failure, base, 60 steps | success, sign_composition, 29 steps | 1 | 3 |
| m9-s1-change_detection-base-sign_composition | sign_composition after shift at step 12 | failure, base, 60 steps | success, sign_composition, 20 steps | 1 | 3 |

The full development-grid result is v0.3 534/550, v0.4 546/550, and the v0.5
prototype 550/550. These are task-success counts on previously exposed maps.
They are not the primary identification endpoint and are not confirmatory.

## 2. Public rule roles used in the analysis

The public menu has three coloured markers. Their roles in the six hypotheses
are:

| Rule | Goal | Hazard | Safe |
| --- | --- | --- | --- |
| base | red | blue | green |
| remap_swap | red | blue | green |
| remap_rotate | red | blue | green |
| slippery | red | blue | green |
| hazard_goal_swap | blue | red | green |
| sign_composition | green | red | blue |

The sign has three faces. Side 0 separates base/remap/slippery from
hazard_goal_swap/sign_composition. On the maps in these failures the side-0
text is either "The red tile is not the goal." or "The red tile is not the
hazard." The first leaves hazard_goal_swap and sign_composition tied at 0.5;
the second leaves base and slippery as the leading alternatives. Side 1
separates the remaps; side 2 is shared.

## 3. Failure 1: m0-s3-identification-sign_composition-None

### Ground truth and map

- True rule: sign_composition (goal green, hazard red, safe blue).
- Map 0, run seed 3; start (1,5); markers blue (5,5), green (2,1), red (5,3);
  signpost (4,2).
- Scenario type: identification; no shift.
- Correct terminal estimate: sign_composition.

### v0.3/v0.4 trace

Initial posterior is uniform. The policy spends decisions 0-6 moving toward
the sign. At decision 7 it reads side 0 and receives "The red tile is not the
goal." That rules out base/remap/slippery but leaves hazard_goal_swap and
sign_composition at 0.500 each. The policy then moves to (4,4).

At decision 9 it spends the second tool on INSPECT at blue (5,5). The tool
returns "goal" from the 5% noise channel. The posterior becomes
hazard_goal_swap 0.974, sign_composition 0.026. The policy commits to
hazard_goal_swap and plans toward blue, which is actually the safe cell under
sign_composition. It reaches (5,5) without a hazard, but the belief is already
fixed and the planner keeps targeting a non-goal cell. It circles around
(5,4)/(5,5) and never reaches green (2,1). With no tool budget left, there is
no way to repeat the inspection. The episode ends at the 60-step limit with
final estimate hazard_goal_swap.

### Belief evolution

- Initial: uniform over six rules.
- After side-0 sign at decision 7: hazard_goal_swap 0.500,
  sign_composition 0.500.
- After noisy INSPECT blue at decision 9: hazard_goal_swap 0.974,
  sign_composition 0.026.
- Final: hazard_goal_swap, no recovery.

### Root cause

A single 5%-noise INSPECT was treated as near-certain evidence. The posterior
jumped from 0.500/0.500 to 0.974/0.026, and the existing commit threshold
accepted it. The policy did not check the tool result against the public
consequence of the planned action: walking onto blue would have produced a
safe movement, which would have contradicted hazard_goal_swap and supported
sign_composition. It also had no replication, no consistency check, and no
fallback after the tool budget reached zero. The failure is primarily a policy
error: the likelihood model is doing what it was written to do, but the
decision rule does not require independent confirmation before committing.

### Proposed fix and v0.5 behaviour

Direct cell-role feedback. If the agent physically enters a marker and the
public outcome is "move", that marker is safe; if the outcome is "hazard", it
is a hazard. In v0.5, the agent eventually enters red (5,3), receives
"hazard", and the posterior moves to sign_composition. It then plans green
(2,1) and succeeds in 24 steps. The fix does not use the true rule, does not
change the 5% noise, and does not special-case map 0.

### Origin

- Primary: policy (single-sample overconfidence and no movement-outcome
  evidence).
- Contributing: environment (stochastic 5% inspection noise).
- Evaluator: none.
- Benchmark: not defective; the side-0 ambiguity is the intended active
  identification problem.

### Regression test

tests/test_grid_v05_policy.py covers this exact scenario and asserts success,
correct terminal estimate, and tool spend at most 3. The same test also
asserts that v0.3 and v0.4 still fail it, preserving the historical result.

### Full decision trace (v0.3; v0.4 identical)


| # | action | position before -> after | env steps | tools left | outcome / value | top two (posterior) |
|---:|---|---|---:|---:|---|---|
| 0 | UP | (1, 5) -> (1, 4) | 1 | 3 | move | base 0.305 / hazard_goal_swap 0.305 |
| 1 | UP | (1, 4) -> (1, 3) | 2 | 3 | move | base 0.328 / hazard_goal_swap 0.328 |
| 2 | RIGHT | (1, 3) -> (2, 3) | 3 | 3 | move | base 0.332 / hazard_goal_swap 0.332 |
| 3 | RIGHT | (2, 3) -> (2, 2) | 4 | 3 | move | base 0.329 / hazard_goal_swap 0.329 |
| 4 | RIGHT | (2, 2) -> (2, 3) | 5 | 3 | move | base 0.315 / hazard_goal_swap 0.315 |
| 5 | RIGHT | (2, 3) -> (3, 3) | 6 | 3 | move | base 0.329 / hazard_goal_swap 0.329 |
| 6 | RIGHT | (3, 3) -> (4, 3) | 7 | 3 | move | base 0.332 / hazard_goal_swap 0.332 |
| 7 | READ_SIGN | (4, 3) -> (4, 3) | 7 | 2 | The red tile is not the goal. / The red tile is not the goal. | hazard_goal_swap 0.500 / sign_composition 0.500 |
| 8 | DOWN | (4, 3) -> (4, 4) | 8 | 2 | move | hazard_goal_swap 0.500 / sign_composition 0.500 |
| 9 | INSPECT (5, 5) | (4, 4) -> (4, 4) | 8 | 0 | goal / goal | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 10 | DOWN | (4, 4) -> (3, 4) | 9 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 11 | DOWN | (3, 4) -> (3, 5) | 10 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 12 | RIGHT | (3, 5) -> (3, 5) | 11 | 0 | blocked | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 13 | RIGHT | (3, 5) -> (4, 5) | 12 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 14 | RIGHT | (4, 5) -> (5, 5) | 13 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 15 | UP | (5, 5) -> (5, 4) | 14 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 16 | DOWN | (5, 4) -> (5, 5) | 15 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 17 | UP | (5, 5) -> (4, 5) | 16 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 18 | RIGHT | (4, 5) -> (5, 5) | 17 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 19 | UP | (5, 5) -> (5, 4) | 18 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 20 | DOWN | (5, 4) -> (5, 5) | 19 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 21 | UP | (5, 5) -> (5, 4) | 20 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 22 | DOWN | (5, 4) -> (5, 5) | 21 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 23 | UP | (5, 5) -> (5, 4) | 22 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 24 | DOWN | (5, 4) -> (5, 5) | 23 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 25 | UP | (5, 5) -> (5, 4) | 24 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 26 | DOWN | (5, 4) -> (5, 5) | 25 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 27 | UP | (5, 5) -> (5, 4) | 26 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 28 | DOWN | (5, 4) -> (5, 5) | 27 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 29 | UP | (5, 5) -> (5, 4) | 28 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 30 | DOWN | (5, 4) -> (5, 5) | 29 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 31 | UP | (5, 5) -> (5, 4) | 30 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 32 | DOWN | (5, 4) -> (5, 5) | 31 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 33 | UP | (5, 5) -> (5, 4) | 32 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 34 | DOWN | (5, 4) -> (5, 5) | 33 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 35 | UP | (5, 5) -> (5, 4) | 34 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 36 | DOWN | (5, 4) -> (5, 5) | 35 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 37 | UP | (5, 5) -> (5, 4) | 36 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 38 | DOWN | (5, 4) -> (5, 5) | 37 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 39 | UP | (5, 5) -> (5, 4) | 38 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 40 | DOWN | (5, 4) -> (5, 5) | 39 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 41 | UP | (5, 5) -> (5, 4) | 40 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 42 | DOWN | (5, 4) -> (5, 5) | 41 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 43 | UP | (5, 5) -> (5, 4) | 42 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 44 | DOWN | (5, 4) -> (5, 5) | 43 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 45 | UP | (5, 5) -> (5, 4) | 44 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 46 | DOWN | (5, 4) -> (5, 5) | 45 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 47 | UP | (5, 5) -> (5, 5) | 46 | 0 | blocked | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 48 | UP | (5, 5) -> (5, 4) | 47 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 49 | DOWN | (5, 4) -> (5, 5) | 48 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 50 | UP | (5, 5) -> (5, 4) | 49 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 51 | DOWN | (5, 4) -> (5, 5) | 50 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 52 | UP | (5, 5) -> (5, 5) | 51 | 0 | blocked | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 53 | UP | (5, 5) -> (5, 4) | 52 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 54 | DOWN | (5, 4) -> (5, 5) | 53 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 55 | UP | (5, 5) -> (5, 4) | 54 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 56 | DOWN | (5, 4) -> (5, 5) | 55 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 57 | UP | (5, 5) -> (5, 4) | 56 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 58 | DOWN | (5, 4) -> (5, 4) | 57 | 0 | blocked | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 59 | DOWN | (5, 4) -> (5, 4) | 58 | 0 | blocked | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 60 | DOWN | (5, 4) -> (5, 5) | 59 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |
| 61 | UP | (5, 5) -> (5, 4) | 60 | 0 | move | hazard_goal_swap 0.974 / sign_composition 0.026 |

## 4. Failure 2: m5-s1-identification-hazard_goal_swap-None

### Ground truth and map

- True rule: hazard_goal_swap (goal blue, hazard red, safe green).
- Map 5, run seed 1; start (3,3); markers blue (1,1), green (1,4), red (5,2);
  signpost (2,2).
- Scenario type: identification; no shift.
- Correct terminal estimate: hazard_goal_swap.

### v0.3/v0.4 trace

The policy moves LEFT to (2,3), reads side 0 at decision 1, and receives "The
red tile is not the goal." The posterior is hazard_goal_swap 0.500 and
sign_composition 0.500. It moves to (1,3) and spends the second tool on
INSPECT at blue (1,1), which is the true goal.

The tool returns "safe" from the 5% noise channel. The posterior becomes
sign_composition 0.974, hazard_goal_swap 0.026. The policy commits to
sign_composition, whose goal is green (1,4). It moves DOWN into green safely,
then cycles between (1,3) and (1,4) because green is not a goal under the true
rule. It never walks to blue (1,1), even though blue is adjacent. It ends at
the 60-step limit with final estimate sign_composition.

### Belief evolution

- Initial: uniform.
- After side-0 sign: hazard_goal_swap 0.500, sign_composition 0.500.
- After noisy INSPECT blue at decision 3: sign_composition 0.974,
  hazard_goal_swap 0.026.
- Final: sign_composition, no recovery.

### Root cause

This is the mirror image of Failure 1. The single noisy INSPECT is treated as
near-certain, and the policy plans toward green without ever testing the
adjacent blue cell. The true goal is the cell that was inspected, but the
policy's action does not exploit the difference between "inspected as safe"
and "physically entered": under hazard_goal_swap, entering blue would end the
episode successfully. A deterministic movement outcome would have resolved
the ambiguity at no additional tool cost. The primary error is again the
policy's single-sample commitment, not the map or the evaluator.

### Proposed fix and v0.5 behaviour

At decision 4, v0.5 enters green (1,4) and observes "move". Green is safe under
hazard_goal_swap but is the goal under sign_composition. The direct cell-role
update therefore rules out sign_composition and moves the posterior to
hazard_goal_swap 1.000. The policy turns around and enters blue (1,1) at
decision 7, succeeding in 6 total steps. It still paid for the noisy
inspection; it recovered because it used the physical outcome.

### Origin

- Primary: policy.
- Contributing: environment (5% tool noise).
- Evaluator: none.
- Benchmark: not defective; the side-0 ambiguity is intentional.

### Regression test

The same exact-scenario test covers m5, including terminal estimate and tool
budget, and checks that v0.3/v0.4 still fail it.

### Full decision trace (v0.3; v0.4 identical)


| # | action | position before -> after | env steps | tools left | outcome / value | top two (posterior) |
|---:|---|---|---:|---:|---|---|
| 0 | LEFT | (3, 3) -> (2, 3) | 1 | 3 | move | base 0.305 / hazard_goal_swap 0.305 |
| 1 | READ_SIGN | (2, 3) -> (2, 3) | 1 | 2 | The red tile is not the goal. / The red tile is not the goal. | hazard_goal_swap 0.500 / sign_composition 0.500 |
| 2 | LEFT | (2, 3) -> (1, 3) | 2 | 2 | move | hazard_goal_swap 0.500 / sign_composition 0.500 |
| 3 | INSPECT (1, 1) | (1, 3) -> (1, 3) | 2 | 0 | safe / safe | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 4 | DOWN | (1, 3) -> (1, 4) | 3 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 5 | UP | (1, 4) -> (1, 3) | 4 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 6 | DOWN | (1, 3) -> (1, 4) | 5 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 7 | UP | (1, 4) -> (1, 3) | 6 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 8 | DOWN | (1, 3) -> (1, 4) | 7 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 9 | UP | (1, 4) -> (1, 3) | 8 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 10 | DOWN | (1, 3) -> (1, 4) | 9 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 11 | UP | (1, 4) -> (1, 3) | 10 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 12 | DOWN | (1, 3) -> (1, 4) | 11 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 13 | UP | (1, 4) -> (1, 3) | 12 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 14 | DOWN | (1, 3) -> (2, 3) | 13 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 15 | LEFT | (2, 3) -> (1, 3) | 14 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 16 | DOWN | (1, 3) -> (1, 4) | 15 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 17 | UP | (1, 4) -> (1, 3) | 16 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 18 | DOWN | (1, 3) -> (1, 4) | 17 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 19 | UP | (1, 4) -> (1, 3) | 18 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 20 | DOWN | (1, 3) -> (1, 4) | 19 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 21 | UP | (1, 4) -> (1, 3) | 20 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 22 | DOWN | (1, 3) -> (1, 4) | 21 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 23 | UP | (1, 4) -> (1, 3) | 22 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 24 | DOWN | (1, 3) -> (1, 4) | 23 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 25 | UP | (1, 4) -> (1, 3) | 24 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 26 | DOWN | (1, 3) -> (1, 4) | 25 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 27 | UP | (1, 4) -> (1, 3) | 26 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 28 | DOWN | (1, 3) -> (2, 3) | 27 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 29 | LEFT | (2, 3) -> (1, 3) | 28 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 30 | DOWN | (1, 3) -> (1, 4) | 29 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 31 | UP | (1, 4) -> (1, 3) | 30 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 32 | DOWN | (1, 3) -> (1, 4) | 31 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 33 | UP | (1, 4) -> (1, 3) | 32 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 34 | DOWN | (1, 3) -> (1, 4) | 33 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 35 | UP | (1, 4) -> (1, 3) | 34 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 36 | DOWN | (1, 3) -> (1, 4) | 35 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 37 | UP | (1, 4) -> (1, 3) | 36 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 38 | DOWN | (1, 3) -> (1, 4) | 37 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 39 | UP | (1, 4) -> (1, 4) | 38 | 0 | blocked | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 40 | UP | (1, 4) -> (1, 4) | 39 | 0 | blocked | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 41 | UP | (1, 4) -> (1, 3) | 40 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 42 | DOWN | (1, 3) -> (1, 4) | 41 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 43 | UP | (1, 4) -> (1, 3) | 42 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 44 | DOWN | (1, 3) -> (1, 4) | 43 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 45 | UP | (1, 4) -> (1, 3) | 44 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 46 | DOWN | (1, 3) -> (1, 4) | 45 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 47 | UP | (1, 4) -> (1, 3) | 46 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 48 | DOWN | (1, 3) -> (1, 4) | 47 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 49 | UP | (1, 4) -> (1, 3) | 48 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 50 | DOWN | (1, 3) -> (1, 4) | 49 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 51 | UP | (1, 4) -> (1, 3) | 50 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 52 | DOWN | (1, 3) -> (1, 4) | 51 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 53 | UP | (1, 4) -> (1, 3) | 52 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 54 | DOWN | (1, 3) -> (1, 4) | 53 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 55 | UP | (1, 4) -> (1, 3) | 54 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 56 | DOWN | (1, 3) -> (1, 4) | 55 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 57 | UP | (1, 4) -> (1, 3) | 56 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 58 | DOWN | (1, 3) -> (1, 4) | 57 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 59 | UP | (1, 4) -> (1, 3) | 58 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 60 | DOWN | (1, 3) -> (1, 4) | 59 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |
| 61 | UP | (1, 4) -> (1, 3) | 60 | 0 | move | sign_composition 0.974 / hazard_goal_swap 0.026 |

## 5. Failure 3: m7-s2-change_detection-base-sign_composition

### Ground truth and map

- Initial rule: base (goal red, hazard blue, safe green).
- Shift rule: sign_composition (goal green, hazard red, safe blue).
- Shift time: step 10.
- Map 7, run seed 2; start (4,4); markers blue (5,5), green (1,5), red (5,1);
  signpost (2,4).
- Scenario type: change_detection.
- Correct post-shift terminal estimate: sign_composition.

### v0.3/v0.4 trace

Before the shift, the policy moves to (2,5) and reads side 0 at decision 4.
The text is "The red tile is not the hazard." That supports base and slippery;
the posterior becomes base 0.884, slippery 0.116. Further movement raises base
to 1.000. The policy plans toward red (5,1), which is the base goal.

The rule shifts to sign_composition at step 10. The public map is unchanged, so
the policy receives no explicit shift signal. At decision 13 it enters red
(5,1) and receives "hazard". Under base, red is supposed to be the goal and
blue the hazard, but the outcome is a hazard on red. The policy does not treat
this as a contradiction requiring re-diagnosis: its posterior remains base
0.999. It repeatedly attempts to move UP into (5,1), receives "hazard" or
"recovering", and loops until the 60-step limit. It has two tool units left
but never performs a post-shift sign read. Final estimate: base.

### Belief evolution

- Before shift: base 1.000 after the pre-shift sign and movement.
- After shift (step 10): unchanged; no shift observation.
- At first post-shift red hazard (decision 13): base 0.999, slippery 0.001;
  the hazard contradicts the base model's red-goal role, but it is not
  treated as new rule evidence.
- Final: base 1.000.

### Root cause

The sign was read once before the shift and cached. The policy never re-read
the sign after the shift, and it did not maintain an uncertainty state for
"the rule may have changed". The observed hazard at red directly
contradicts base, whose red cell is the goal and would terminate the episode.
The policy nevertheless leaves the posterior at base 0.999 because the
movement outcome is used for recovery and navigation, not for a rule-belief
update. More importantly, the policy had no post-shift probe policy: the two
remaining tools were never used to re-test side 0 or blue.
This is a policy failure to gather post-shift evidence, not an environment
bug: the shift is intended and the public sign remains readable.

### Proposed fix and v0.5 behaviour

v0.5 treats the first post-shift movement outcome as cell-role evidence. On
entering red (5,1) and observing "hazard", the posterior becomes
base/hazard_goal_swap/sign_composition at 0.333 each: all three assign red the
hazard role, while the remap and slippery hypotheses are eliminated. The
posterior is now ambiguous, so the policy uses its remaining budget to inspect
blue (5,5). Blue is safe only under sign_composition, so the posterior becomes
sign_composition 0.950. It plans green (1,5) and succeeds in 29 steps with 3
tools. The recovery is not map-specific and does not read the true rule.

This is a combination of two justified mechanisms: direct cell-role feedback
creates the ambiguity, and the existing v0.4 probe allocation spends the
remaining tool on the informative blue cell. Neither mechanism alone is
sufficient here.

### Origin

- Primary: policy (stale pre-shift observation and no post-shift re-diagnosis).
- Contributing: environment (the intended rule shift).
- Evaluator: none.
- Benchmark: not defective; the scenario is a valid change-detection test.

### Regression test

The exact-scenario test covers m7, its post-shift success, and the 3-tool cap.
The failure is also a regression test for the stale-sign mechanism because the
same ID is asserted to fail under v0.3/v0.4.

### Full decision trace (v0.3; v0.4 identical)


| # | action | position before -> after | env steps | tools left | outcome / value | top two (posterior) |
|---:|---|---|---:|---:|---|---|
| 0 | DOWN | (4, 4) -> (4, 5) | 1 | 3 | move | base 0.269 / hazard_goal_swap 0.269 |
| 1 | LEFT | (4, 5) -> (3, 5) | 2 | 3 | move | base 0.317 / hazard_goal_swap 0.317 |
| 2 | LEFT | (3, 5) -> (3, 5) | 3 | 3 | blocked | base 0.278 / hazard_goal_swap 0.278 |
| 3 | LEFT | (3, 5) -> (2, 5) | 4 | 3 | move | base 0.319 / hazard_goal_swap 0.319 |
| 4 | READ_SIGN | (2, 5) -> (2, 5) | 4 | 2 | The red tile is not the hazard. / The red tile is not the hazard. | base 0.884 / slippery 0.116 |
| 5 | RIGHT | (2, 5) -> (3, 5) | 5 | 2 | move | base 0.972 / slippery 0.028 |
| 6 | UP | (3, 5) -> (3, 4) | 6 | 2 | move | base 0.994 / slippery 0.006 |
| 7 | UP | (3, 4) -> (3, 3) | 7 | 2 | move | base 0.999 / slippery 0.001 |
| 8 | UP | (3, 3) -> (3, 2) | 8 | 2 | move | base 1.000 / slippery 0.000 |
| 9 | UP | (3, 2) -> (3, 1) | 9 | 2 | move | base 1.000 / slippery 0.000 |
| 10 | RIGHT | (3, 1) -> (4, 1) | 10 | 2 | move | base 1.000 / slippery 0.000 |
| 11 | RIGHT | (4, 1) -> (4, 2) | 11 | 2 | move | base 1.000 / slippery 0.000 |
| 12 | UP | (4, 2) -> (5, 2) | 12 | 2 | move | base 0.999 / slippery 0.001 |
| 13 | UP | (5, 2) -> (5, 1) | 13 | 2 | hazard | base 0.999 / slippery 0.001 |
| 14 | UP | (5, 1) -> (5, 1) | 14 | 2 | recovering | base 1.000 / slippery 0.000 |
| 15 | UP | (5, 1) -> (5, 1) | 15 | 2 | hazard | base 1.000 / slippery 0.000 |
| 16 | UP | (5, 1) -> (5, 1) | 16 | 2 | recovering | base 1.000 / slippery 0.000 |
| 17 | UP | (5, 1) -> (5, 1) | 17 | 2 | hazard | base 1.000 / slippery 0.000 |
| 18 | UP | (5, 1) -> (5, 1) | 18 | 2 | recovering | base 1.000 / slippery 0.000 |
| 19 | UP | (5, 1) -> (5, 1) | 19 | 2 | hazard | base 1.000 / slippery 0.000 |
| 20 | UP | (5, 1) -> (5, 1) | 20 | 2 | recovering | base 1.000 / slippery 0.000 |
| 21 | UP | (5, 1) -> (5, 1) | 21 | 2 | hazard | base 1.000 / slippery 0.000 |
| 22 | UP | (5, 1) -> (5, 1) | 22 | 2 | recovering | base 1.000 / slippery 0.000 |
| 23 | UP | (5, 1) -> (5, 1) | 23 | 2 | hazard | base 1.000 / slippery 0.000 |
| 24 | UP | (5, 1) -> (5, 1) | 24 | 2 | recovering | base 1.000 / slippery 0.000 |
| 25 | UP | (5, 1) -> (5, 1) | 25 | 2 | hazard | base 1.000 / slippery 0.000 |
| 26 | UP | (5, 1) -> (5, 1) | 26 | 2 | recovering | base 1.000 / slippery 0.000 |
| 27 | UP | (5, 1) -> (5, 1) | 27 | 2 | hazard | base 1.000 / slippery 0.000 |
| 28 | UP | (5, 1) -> (5, 1) | 28 | 2 | recovering | base 1.000 / slippery 0.000 |
| 29 | UP | (5, 1) -> (5, 1) | 29 | 2 | hazard | base 1.000 / slippery 0.000 |
| 30 | UP | (5, 1) -> (5, 1) | 30 | 2 | recovering | base 1.000 / slippery 0.000 |
| 31 | UP | (5, 1) -> (5, 1) | 31 | 2 | hazard | base 1.000 / slippery 0.000 |
| 32 | UP | (5, 1) -> (5, 1) | 32 | 2 | recovering | base 1.000 / slippery 0.000 |
| 33 | UP | (5, 1) -> (5, 1) | 33 | 2 | hazard | base 1.000 / slippery 0.000 |
| 34 | UP | (5, 1) -> (5, 1) | 34 | 2 | recovering | base 1.000 / slippery 0.000 |
| 35 | UP | (5, 1) -> (5, 1) | 35 | 2 | hazard | base 1.000 / slippery 0.000 |
| 36 | UP | (5, 1) -> (5, 1) | 36 | 2 | recovering | base 1.000 / slippery 0.000 |
| 37 | UP | (5, 1) -> (5, 1) | 37 | 2 | hazard | base 1.000 / slippery 0.000 |
| 38 | UP | (5, 1) -> (5, 1) | 38 | 2 | recovering | base 1.000 / slippery 0.000 |
| 39 | UP | (5, 1) -> (5, 1) | 39 | 2 | hazard | base 1.000 / slippery 0.000 |
| 40 | UP | (5, 1) -> (5, 1) | 40 | 2 | recovering | base 1.000 / slippery 0.000 |
| 41 | UP | (5, 1) -> (5, 1) | 41 | 2 | hazard | base 1.000 / slippery 0.000 |
| 42 | UP | (5, 1) -> (5, 1) | 42 | 2 | recovering | base 1.000 / slippery 0.000 |
| 43 | UP | (5, 1) -> (5, 1) | 43 | 2 | hazard | base 1.000 / slippery 0.000 |
| 44 | UP | (5, 1) -> (5, 1) | 44 | 2 | recovering | base 1.000 / slippery 0.000 |
| 45 | UP | (5, 1) -> (5, 1) | 45 | 2 | hazard | base 1.000 / slippery 0.000 |
| 46 | UP | (5, 1) -> (5, 1) | 46 | 2 | recovering | base 1.000 / slippery 0.000 |
| 47 | UP | (5, 1) -> (5, 1) | 47 | 2 | hazard | base 1.000 / slippery 0.000 |
| 48 | UP | (5, 1) -> (5, 1) | 48 | 2 | recovering | base 1.000 / slippery 0.000 |
| 49 | UP | (5, 1) -> (5, 1) | 49 | 2 | hazard | base 1.000 / slippery 0.000 |
| 50 | UP | (5, 1) -> (5, 1) | 50 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 51 | UP | (5, 1) -> (5, 1) | 51 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 52 | UP | (5, 1) -> (5, 1) | 52 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 53 | UP | (5, 1) -> (5, 1) | 53 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 54 | UP | (5, 1) -> (5, 1) | 54 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 55 | UP | (5, 1) -> (5, 1) | 55 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 56 | UP | (5, 1) -> (5, 1) | 56 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 57 | UP | (5, 1) -> (5, 1) | 57 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 58 | UP | (5, 1) -> (5, 1) | 58 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 59 | UP | (5, 1) -> (5, 1) | 59 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 60 | UP | (5, 1) -> (5, 1) | 60 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |

## 6. Failure 4: m9-s1-change_detection-base-sign_composition

### Ground truth and map

- Initial rule: base (goal red, hazard blue, safe green).
- Shift rule: sign_composition (goal green, hazard red, safe blue).
- Shift time: step 12.
- Map 9, run seed 1; start (5,1); markers blue (1,3), green (3,4), red (2,1);
  signpost (2,5).
- Scenario type: change_detection.
- Correct post-shift terminal estimate: sign_composition.

### v0.3/v0.4 trace

The policy reads side 0 at decision 8, before the shift, and receives "The red
tile is not the hazard." The posterior is base 1.000. It plans toward red
(2,1), the base goal.

The shift occurs at step 12. At decision 13 the policy enters red (2,1) and
receives "hazard". It keeps the base hypothesis at 1.000, attempts to move
UP into the same cell, receives repeated hazard/recovering outcomes, and loops
until the 60-step limit. It has two tools left and never re-reads the sign or
inspects blue. Final estimate: base.

### Belief evolution

- Before shift: base 1.000.
- After shift: unchanged.
- First post-shift red hazard: base 1.000; no shift update.
- Final: base 1.000.

### Root cause

The mechanism is the same as Failure 3: a cached pre-shift sign read, no
post-shift re-diagnosis, and no use of the unexpected movement outcome to
create a new ambiguity. The different map, seed, and shift time make it a
second independent instance rather than a duplicate: it checks that the fix is
not tied to the timing or geometry of m7.

### Proposed fix and v0.5 behaviour

At decision 13, v0.5's direct cell-role feedback turns the red hazard into
base/hazard_goal_swap/sign_composition at 0.333 each. It then inspects blue
(1,3); blue is safe only under sign_composition, and the posterior becomes
0.950. It plans green (3,4) and succeeds in 20 steps with 3 tools. The fix is
the same general evidence channel and the same existing probe allocation; it
is not keyed to map 9 or seed 1.

### Origin

- Primary: policy.
- Contributing: environment (intended shift).
- Evaluator: none.
- Benchmark: not defective.

### Regression test

The exact-scenario test covers m9 independently of m7, plus the historical
v0.3/v0.4 failure assertion. This is important because a map-specific patch
could fix one and not the other.

### Full decision trace (v0.3; v0.4 identical)


| # | action | position before -> after | env steps | tools left | outcome / value | top two (posterior) |
|---:|---|---|---:|---:|---|---|
| 0 | DOWN | (5, 1) -> (5, 2) | 1 | 3 | move | base 0.269 / hazard_goal_swap 0.269 |
| 1 | LEFT | (5, 2) -> (4, 2) | 2 | 3 | move | base 0.317 / hazard_goal_swap 0.317 |
| 2 | LEFT | (4, 2) -> (3, 2) | 3 | 3 | move | base 0.330 / hazard_goal_swap 0.330 |
| 3 | DOWN | (3, 2) -> (3, 3) | 4 | 3 | move | base 0.333 / hazard_goal_swap 0.333 |
| 4 | LEFT | (3, 3) -> (2, 3) | 5 | 3 | move | base 0.333 / hazard_goal_swap 0.333 |
| 5 | DOWN | (2, 3) -> (2, 4) | 6 | 3 | move | base 0.333 / hazard_goal_swap 0.333 |
| 6 | LEFT | (2, 4) -> (1, 4) | 7 | 3 | move | base 0.333 / hazard_goal_swap 0.333 |
| 7 | DOWN | (1, 4) -> (1, 5) | 8 | 3 | move | base 0.333 / hazard_goal_swap 0.333 |
| 8 | READ_SIGN | (1, 5) -> (1, 5) | 8 | 2 | The red tile is not the hazard. / The red tile is not the hazard. | base 1.000 / slippery 0.000 |
| 9 | UP | (1, 5) -> (1, 4) | 9 | 2 | move | base 1.000 / slippery 0.000 |
| 10 | UP | (1, 4) -> (2, 4) | 10 | 2 | move | base 1.000 / slippery 0.000 |
| 11 | UP | (2, 4) -> (2, 3) | 11 | 2 | move | base 1.000 / slippery 0.000 |
| 12 | UP | (2, 3) -> (2, 2) | 12 | 2 | move | base 1.000 / slippery 0.000 |
| 13 | UP | (2, 2) -> (2, 1) | 13 | 2 | hazard | base 1.000 / slippery 0.000 |
| 14 | UP | (2, 1) -> (2, 1) | 14 | 2 | recovering | base 1.000 / slippery 0.000 |
| 15 | UP | (2, 1) -> (2, 1) | 15 | 2 | hazard | base 1.000 / slippery 0.000 |
| 16 | UP | (2, 1) -> (2, 1) | 16 | 2 | recovering | base 1.000 / slippery 0.000 |
| 17 | UP | (2, 1) -> (2, 1) | 17 | 2 | hazard | base 1.000 / slippery 0.000 |
| 18 | UP | (2, 1) -> (2, 1) | 18 | 2 | recovering | base 1.000 / slippery 0.000 |
| 19 | UP | (2, 1) -> (2, 1) | 19 | 2 | hazard | base 1.000 / slippery 0.000 |
| 20 | UP | (2, 1) -> (2, 1) | 20 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 21 | UP | (2, 1) -> (2, 1) | 21 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 22 | UP | (2, 1) -> (2, 1) | 22 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 23 | UP | (2, 1) -> (2, 1) | 23 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 24 | UP | (2, 1) -> (2, 1) | 24 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 25 | UP | (2, 1) -> (2, 1) | 25 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 26 | UP | (2, 1) -> (2, 1) | 26 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 27 | UP | (2, 1) -> (2, 1) | 27 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 28 | UP | (2, 1) -> (2, 1) | 28 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 29 | UP | (2, 1) -> (2, 1) | 29 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 30 | UP | (2, 1) -> (2, 1) | 30 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 31 | UP | (2, 1) -> (2, 1) | 31 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 32 | UP | (2, 1) -> (2, 1) | 32 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 33 | UP | (2, 1) -> (2, 1) | 33 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 34 | UP | (2, 1) -> (2, 1) | 34 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 35 | UP | (2, 1) -> (2, 1) | 35 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 36 | UP | (2, 1) -> (2, 1) | 36 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 37 | UP | (2, 1) -> (2, 1) | 37 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 38 | UP | (2, 1) -> (2, 1) | 38 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 39 | UP | (2, 1) -> (2, 1) | 39 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 40 | UP | (2, 1) -> (2, 1) | 40 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 41 | UP | (2, 1) -> (1, 1) | 41 | 2 | move | base 1.000 / hazard_goal_swap 0.000 |
| 42 | RIGHT | (1, 1) -> (2, 1) | 42 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 43 | UP | (2, 1) -> (2, 1) | 43 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 44 | UP | (2, 1) -> (2, 1) | 44 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 45 | UP | (2, 1) -> (2, 1) | 45 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 46 | UP | (2, 1) -> (2, 1) | 46 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 47 | UP | (2, 1) -> (2, 1) | 47 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 48 | UP | (2, 1) -> (2, 1) | 48 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 49 | UP | (2, 1) -> (2, 1) | 49 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 50 | UP | (2, 1) -> (2, 1) | 50 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 51 | UP | (2, 1) -> (2, 1) | 51 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 52 | UP | (2, 1) -> (2, 1) | 52 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 53 | UP | (2, 1) -> (2, 1) | 53 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 54 | UP | (2, 1) -> (2, 1) | 54 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 55 | UP | (2, 1) -> (2, 1) | 55 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 56 | UP | (2, 1) -> (2, 1) | 56 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 57 | UP | (2, 1) -> (2, 1) | 57 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 58 | UP | (2, 1) -> (2, 1) | 58 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |
| 59 | UP | (2, 1) -> (2, 1) | 59 | 2 | recovering | base 1.000 / hazard_goal_swap 0.000 |
| 60 | UP | (2, 1) -> (2, 1) | 60 | 2 | hazard | base 1.000 / hazard_goal_swap 0.000 |

## 7. Cross-cutting findings

1. **A single noisy INSPECT can erase a correct sign inference.** In m0 and
   m5 the side-0 sign left a 0.500/0.500 ambiguity. A 5%-noise tool result
   moved the posterior to 0.974/0.026. The commit threshold accepted it. The
   policy did not repeat the tool, cross-check it against movement, or treat
   the remaining budget as a resource for resolving the exact ambiguity.

2. **Movement outcomes were not part of the belief update.** Entering a marker
   and observing move, hazard, or goal is deterministic public evidence. The
   v0.3/v0.4 policies used the outcome for reward and navigation but not for
   rule identification. This is the common missing mechanism behind all four
   failures.

3. **Stale sign observations survived a rule shift.** m7 and m9 read side 0
   before the shift and never re-read it. The post-shift red hazard is
   compatible with all three hypotheses that assign red the hazard role, so the
   correct response is to become ambiguous and probe the distinguishing blue
   cell, not to retain the pre-shift base belief.

4. **The tool budget was not allocated to the unresolved distinction.** In m0
   and m5 the policy spent the last tool on a noisy blue inspection even though
   the sign ambiguity was the unresolved question. In m7 and m9 it left two
   tools unused. The v0.4 probe-allocation repair fixes some allocation cases,
   but these four episodes still share the underlying evidence-use failure.

5. **The evaluator and map generator are not the cause.** The scenarios are
   valid, solvable, and have a readable side-0 face. The true rule is available
   only to the evaluator. No failure required a ground-truth leak, a map
   change, or a scoring change.

## 8. Candidate change and regression evidence

v0.5 is a separate class, DiagnosticBeliefPolicyV5, subclassing v0.4. It adds
one public observation channel:

    on a MOVE that ends on a marker cell:
        outcome "move"    -> marker is safe
        outcome "hazard"  -> marker is hazard
        outcome "goal"    -> episode terminates; no role update needed

The implementation calls the existing deterministic GridBelief.update_role.
It never reads the true rule, the scenario, the shift time, or the evaluator
state. It does not add tools, change the tool noise, change the map set, or
special-case a scenario ID.

The four exact scenarios are covered by tests/test_grid_v05_policy.py. The
same test asserts that v0.3 and v0.4 still fail the four scenarios, so the
historical result is preserved rather than edited.

Development counts:

| Method | Full grid | Identification task success | Identification correct | Base task success |
| --- | ---: | ---: | ---: | ---: |
| v0.3 | 534/550 | 287/300 | 297/300 | 49/50 |
| v0.4 | 546/550 | 298/300 | 297/300 | 50/50 |
| v0.5 prototype | 550/550 | 300/300 | 299/300 | 50/50 |

Paired development comparisons:

- v0.5 vs v0.3: 16 discordant successes, all in v0.5; exact two-sided McNemar
  p = 3.0517578125e-05.
- v0.5 vs v0.4: 4 discordant successes, all in v0.5; exact two-sided McNemar
  p = 0.125.

These are development results, not gate results. The v0.5 vs v0.4 paired
difference on the full grid is not statistically strong, and the four fixes
are a small subset of 550 episodes. The confirmatory claim needs the new
unseen holdout in the preregistration.

## 9. Residual anomalies in the v0.5 development run

The v0.5 candidate is not perfect, and the anomalies are reported rather than
tuned away.

### 9.1 Accidental short success with a wrong estimate

Scenario: m3-s2-identification-slippery-None.

- True rule: slippery (goal red, hazard blue, safe green).
- Start (4,2); red goal (2,2).
- v0.5 moves DOWN twice, enters red at step 2, and succeeds before using a
  tool or observing a cell role. The terminal posterior is remap_rotate 0.942;
  the terminal estimate is wrong.
- This is a successful misdiagnosis: task success, wrong identification. It is
  an accidental zero-tool success (zero tools, zero direct cell-role
  observations) and must be counted as an identification error.

This anomaly was not one of the four shared failures, and it was not used to
change the gate or the policy after the fact. It shows that a stochastic
slippery episode can terminate before the policy has evidence. The protocol
retains it in the denominator and reports it as an identification error; it is
not converted into a correct identification because the task succeeded.

### 9.2 Evidence-backed zero-tool successes

Three v0.5 hazard/sign identification episodes succeed with zero tools:

- m1-s2-identification-hazard_goal_swap-None: enters red, observes hazard,
  identifies hazard_goal_swap, then reaches blue; 10 steps.
- m2-s3-identification-hazard_goal_swap-None: enters red, observes hazard,
  identifies hazard_goal_swap, then reaches blue; 17 steps.
- m6-s4-identification-sign_composition-None: enters red (hazard) and then
  blue (safe), identifies sign_composition, then reaches green; 7 steps.

These are raw zero-tool successes, but they are not accidental under the
protocol definition: each has direct cell-role evidence and a correct
terminal estimate. The raw counts and the evidence-backed classification are
both reported. If a reviewer intends the zero-tool gate to be literal rather
than accident-based, the candidate fails that literal gate; the protocol
states this explicitly.

## 10. What was deliberately not changed

- No map was regenerated or patched.
- No seed was selected because it produced a success.
- No true rule, shift time, or evaluator field is read by the policy.
- No tool-noise parameter, success threshold, or scoring rule was changed to
  make a failure pass.
- No historical v0.3/v0.4 result, config, or commit hash was rewritten.
- The v0.5 class is additive; existing policy classes are unchanged.

## 11. Reproduction and artifact

- Trace script: scripts/trace_grid_failures.py.
- Trace artifact: results/grid_v05_dev/traces.json (git-ignored).
- Full development rows: results/grid_v05_dev/fullgrid_10x5.jsonl (git-ignored).
- Full development summary: results/grid_v05_dev/summary.json (git-ignored).
- Policy: src/ruleshift/agents/grid_baselines.py,
  DiagnosticBeliefPolicyV5.
- Tests: tests/test_grid_v05_policy.py.

Command:

    python scripts/trace_grid_failures.py --out results/grid_v05_dev/traces.json
    python -m pytest tests/test_grid_v05_policy.py -q

The four full traces follow. They are long by design: every decision is
retained, including repeated loops, so the failure is auditable rather than
summarised away.
