# Phase 0 10-map CPU replication: gate failure and validity audit

**Status: development result, not preregistered, gates failed, Phase 1 not started.**

This report records the 10-map replication requested after the map-0 and five-map
checks. It is deliberately a negative-result report. The symbolic
`DiagnosticBeliefPolicy` is much better than `no_adaptation` at active rule
identification, but the preregistered acceptance gates do not pass. Per the
stop rule, no Phase 1 VLM smoke, no 55-episode evaluation, and no GPU grid were
run. No prompt was tuned to force a pass.

## 1. Decision

- Overall success is **518/550 (94.2%)** for `diagnostic_belief` versus
  **382/550 (69.5%)** for `no_adaptation`.
- Identification is **276/300 (92.0%)**, below the 95% gate.
- `hazard_goal_swap` is **39/50 (78.0%)**, below 90%; `sign_composition` is
  **38/50 (76.0%)**, below 90%; `base` is **49/50 (98.0%)**, below 100%.
- There are **20 zero-tool hazard/sign identification episodes**; 19 fail and
  one succeeds accidentally. This violates the no-zero-tool gate.
- The tool cap passes: maximum is 3 units, and the paired gain is positive
  (147 diagnostic-only versus 11 baseline-only, exact McNemar
  `p = 1.59e-31`).
- Two benchmark-validity defects explain much of the shortfall: duplicate map
  geometry and signposts with no reachable informative face. A separate policy
  limitation remains on map 5. The eligible-map sensitivity check still fails
  the 100% base-preservation gate, so the benchmark defect is not the whole
  story.

**Decision:** stop the Phase 0 branch here. Do not implement Phase 1, do not run
the two-episode smoke, and do not run 55 episodes on the present benchmark.
Repairing the benchmark is a separate, preregistered step.

## 2. What was run

- Branch: `research/belief-state`, policy commit `ca249e5`.
- Scenarios: 10 maps x 5 run seeds x 11 scenario types = **550 episodes**.
- Methods: `diagnostic_belief` and `no_adaptation`; **1,100 method rows**.
- Hardware: CPU only. The frozen VLM was not loaded or called.
- `DiagnosticBeliefPolicy` is a deterministic symbolic reference, not a VLM.
  It uses the public map, Bayesian `GridBelief`, a fixed sign/blue diagnostic
  tree, A* navigation, and the 3-unit tool budget.

Reproduction:

```powershell
$env:PYTHONPATH = 'src'
python scripts/export_grid_baselines.py `
  --methods diagnostic_belief,no_adaptation `
  --maps 10 --seeds 5 `
  --out results/grid_phase0/maps10.jsonl
python scripts/analyze_grid_phase0.py `
  --input results/grid_phase0/maps10.jsonl `
  --out-dir results/grid_phase0/maps10_analysis `
  --maps 10 --seeds 5
```

Raw and analyzed artifacts are generated and ignored by Git:

- `results/grid_phase0/maps10.jsonl` (SHA-256
  `E04955C7193BA21C4C811C900707BAE5C2D546358B94B80EB1055290E9C2CB2B`)
- `results/grid_phase0/maps10_analysis/aggregate.json`
- `results/grid_phase0/maps10_analysis/per_map.json`
- `results/grid_phase0/maps10_analysis/paired.json`
- `results/grid_phase0/maps10_analysis/failures.json`
- `results/grid_phase0/maps10_analysis/sign_affordance.json`
- `results/grid_phase0/maps10_analysis/map_identity.json`
- `results/grid_phase0/maps10_analysis/eligibility.json`
- `results/grid_phase0/maps10_analysis/zero_tool_hazard_or_sign.json`
- `results/grid_phase0/map_suite_audit_100.json` (100-map independence audit)

## 3. Main results

### 3.1 Overall

| Method | Success | Rate | Env steps | Tool units |
| --- | ---: | ---: | ---: | ---: |
| `diagnostic_belief` | 518/550 | 0.942 | 5,683 | 688 |
| `no_adaptation` | 382/550 | 0.695 | 12,218 | 0 |

Paired counts: 371 both, **147 only diagnostic**, **11 only baseline**, 21
neither. Exact McNemar `p = 1.5885257017954777e-31`.

### 3.2 By scenario type

| Type | Diagnostic | Baseline |
| --- | ---: | ---: |
| identification | 276/300 (0.920) | 132/300 (0.440) |
| change_detection | 192/200 (0.960) | 200/200 (1.000) |
| no_shift | 50/50 (1.000) | 50/50 (1.000) |

The overall gain is entirely from identification. On change detection the
diagnostic policy loses 8 episodes relative to `no_adaptation` (paired exact
`p = 0.0078125`), consistent with spending the 3-unit budget before the shift
and not re-probing.

### 3.3 Identification by initial rule

| Initial rule | Diagnostic | Baseline |
| --- | ---: | ---: |
| `base` | 49/50 | 50/50 |
| `hazard_goal_swap` | 39/50 | 10/50 |
| `sign_composition` | 38/50 | 10/50 |
| `remap_swap` | 50/50 | 0/50 |
| `remap_rotate` | 50/50 | 12/50 |
| `slippery` | 50/50 | 50/50 |

The movement-identifiable rules are perfect. The two sign-dependent rules are
the weak point, and the zero-tool benchmark failures are concentrated in them.

### 3.4 Per map

| Map | Diagnostic | Baseline | Env steps | Tool units |
| --- | ---: | ---: | ---: | ---: |
| m0 | 54/55 | 38/55 | 473 | 60 |
| m1 | 55/55 | 35/55 | 347 | 68 |
| m2 | 55/55 | 35/55 | 326 | 69 |
| m3 | 53/55 | 47/55 | 448 | 72 |
| m4 | 51/55 | 45/55 | 1,225 | 120 |
| m5 | 53/55 | 37/55 | 475 | 120 |
| m6 | 55/55 | 35/55 | 258 | 63 |
| m7 | 45/55 | 35/55 | 753 | 0 |
| m8 | 46/55 | 36/55 | 723 | 0 |
| m9 | 51/55 | 39/55 | 655 | 116 |

Maps 7 and 8 are the unreadable-signpost maps described in section 5. Their
zero tool spend is **not** evidence of a no-tool diagnostic success.

## 4. Acceptance gates

| Gate | Target | Observed | Result |
| --- | ---: | ---: | --- |
| identification | >= 95% | 92.0% | FAIL |
| hazard | >= 90% | 78.0% | FAIL |
| sign | >= 90% | 76.0% | FAIL |
| base preserved | 100% | 98.0% | FAIL |
| no zero-tool hazard/sign | 0 | 20 | FAIL |
| maximum tool units | <= 3 | 3 | PASS |
| paired gain positive | diagnostic > baseline | 147 > 11 | PASS |

The zero-tool count is 20 raw rows. One of the 20 succeeds:
`m8-s3-identification-sign_composition-None` reaches the goal in 14 steps with
zero tools and `base` as the final estimate. That is an accidental success, not
diagnostic evidence; section 5.2 explains why the map cannot support the
hazard/sign diagnostic.

## 5. Why the gates fail

### 5.1 Duplicate map geometry

The 10 requested maps contain only **8 unique public geometries**:
`m1 = m2` and `m7 = m8`. The 100-map audit is worse: 100 requested maps contain
only **62 unique public geometries**, with 25 duplicate groups and 38 redundant
map indices. Examples include `m21..m25`, `m44..m47`, `m48..m51`, and
`m88..m91`.

Cause: `generate_solvable_map` retries with `random.Random(seed + attempt)`.
For example, `seed=1, attempt=1` has the same RNG stream as
`seed=2, attempt=0`. Scenario ids remain unique, but their geometry does not.

Consequences:

1. The nominal 550-episode replication is not 550 independent maps; it is 8
   geometries with five run seeds each.
2. The nominal 5,500-episode grid is not 5,500 unique scenarios at the map
   level; the audit predicts 62 unique geometries. Paired intervals that
   resample scenarios would treat duplicate geometry as independent evidence.
3. Any claim of "5,500 unique scenarios" is false until the retry RNG is fixed
   and the benchmark version is changed.

This is a benchmark-construction bug, not a policy bug. It must not be fixed by
silently regenerating maps after seeing results; that would require a new
preregistered benchmark version.

### 5.2 Signposts without a reachable informative face

Map generation checks that the goal is reachable, but not that the signpost has
a free, reachable cell on an informative side. On the 10-map set, only **8/10
maps** have a reachable informative face. Maps 7 and 8 share the same geometry:
a 7x7 map with signpost `(4,5)`. Side 0 candidates are out of bounds or walled
off; side 1 is walled off; side 2 is shared/uninformative. There is no legal
`READ_SIGN` that can separate `hazard_goal_swap` from `sign_composition`.

This produces exactly the 20 zero-tool episodes: 2 maps x 5 seeds x 2
hazard/sign rules. Nineteen fail at the 60-step limit. The twentieth
(`m8-s3-sign_composition`) happens to reach the goal with the initial `base`
estimate, so it is recorded as a success with zero tools. It is not a diagnosis
and must not be counted as one.

On the 100-map audit, **6 of 62 unique geometries** (11 of 100 map indices)
have no informative face. The affected index groups are `m7/m8`, `m12/m13`,
`m60`, `m64/m65`, `m92/m93/m94`, and `m95`.

This is also a benchmark-construction bug. A valid benchmark must gate on both
goal reachability and a reachable informative sign face (or declare the map
sign-free and exclude sign-dependent scenarios).

### 5.3 Base failure on map 5: face selection, not prompt tuning

The single base identification failure is
`m5-s4-identification-base-None`. Map 5 has signpost `(1,4)` with informative
cells on both sides: side 0 at `(1,5)` and side 1 at `(2,4)`. Both are
reachable. The policy chooses the nearer side 1. Side 1 rules out the remaps,
but it does not separate `base` from `hazard_goal_swap`/`sign_composition`; only
side 0 does that. The policy then spends 2 units inspecting blue. Under `base`
the true blue is a hazard, but the 5% inspect error returns `goal`, so the
filter shifts to `hazard_goal_swap` and the episode times out. The run uses 3
tools and is classified `inspect_noise`.

This is a genuine policy limitation: the current face choice prefers proximity
over expected discrimination. It is not repaired here, because changing it
after seeing the gate would be post-hoc tuning. A future policy could rank
faces by expected information gain, not distance.

### 5.4 Inspect noise

Five identification failures are classified `inspect_noise`:
`m0-s3-sign`, `m3-s0-hazard`, `m3-s1-sign`, `m5-s4-base`, `m9-s0-sign`. The
public `INSPECT` error is 0.05. With one inspect and a 3-unit budget there is
no affordable retry. The analysis reproduces the failed episode and checks the
tool outcome before assigning this class; it is not a catch-all.

### 5.5 Post-shift failures are out of scope, but disclosed

Eight change-detection failures are `post_shift_no_recovery` or
`post_shift_late`. The initial diagnosis spends the budget; after the shift the
policy has no probe left. This is the scoped limitation identified earlier.
Active rule identification is the primary claim; general post-shift adaptation
remains a future extension. It should not be bundled into the identification
success rate.

## 6. Failure taxonomy

| Classification | Count |
| --- | ---: |
| `zero_tool_unidentified` | 19 |
| `post_shift_no_recovery` | 7 |
| `post_shift_late` | 1 |
| `inspect_noise` | 5 |
| **Total failures** | **32** |

Failure rows sum to 32, plus the one accidental zero-tool success. The
`zero_tool_unidentified` count is 19 failures; the raw zero-tool count is 20.

## 7. Eligible-map sensitivity (post hoc, not acceptance)

Excluding maps 7 and 8 gives 8 map indices, but only 7 unique geometries because
`m1 = m2`. This subset removes the unreadable-signpost maps, not the duplicate
geometry problem:

| Initial rule | Diagnostic | Baseline |
| --- | ---: | ---: |
| `base` | 39/40 | 40/40 |
| `hazard_goal_swap` | 39/40 | 10/40 |
| `sign_composition` | 37/40 | 9/40 |
| `remap_swap` | 40/40 | 0/40 |
| `remap_rotate` | 40/40 | 12/40 |
| `slippery` | 40/40 | 40/40 |

Eligible overall: **427/440 (97.0%)** versus **311/440 (70.7%)**. Eligible
identification: **235/240 (97.9%)**. Eligible hazard: 39/40 (97.5%). Eligible
sign: 37/40 (92.5%). Eligible paired: 300 both, 127 diagnostic-only, 11
baseline-only, 2 neither, `p = 3.61e-26`.

Six of the seven gates pass on this post-hoc subset; `base_preserved_100pct`
still fails at 39/40. The subset is useful for diagnosis, not as a substitute
acceptance result: it is selected after inspection and still contains duplicate
geometry.

## 8. Interpretation

1. The symbolic diagnostic protocol works where the benchmark is well formed.
   `remap_swap`, `remap_rotate`, and `slippery` are perfect; hazard is 78%
   overall but 97.5% on eligible maps; sign is 76% overall but 92.5% on eligible
   maps. This supports the mechanism, not the current benchmark-level claim.
2. The reported 94.2% is conditional on generated maps that include impossible
   episodes. It is not a clean estimate of identification ability. The
   accidental zero-tool success and the 19 impossible failures are benchmark
   artifacts.
3. The 10-map replication does **not** establish the 95% active-identification
   claim. The pre-shift identification gate fails even before considering VLM
   behaviour.
4. The 3-unit budget makes initial identification and post-shift
   re-identification mutually exclusive. On change detection the policy is
   worse than no adaptation (192/200 versus 200/200). This is a design
   limitation to address in a future post-shift study, not a result to hide in
   the identification aggregate.
5. No frozen-VLM conclusion follows from this run. The VLM Phase 1 experiment
   was intentionally not started because its prerequisite gate failed.

## 9. What would make the next replication valid

These are recommendations, not changes made here:

1. Make map retries collision-free, e.g. use a per-seed stream or
   `attempt * 1009 + seed`; re-audit uniqueness for 100 maps.
2. Add a generation-time affordance gate: every sign-dependent map must have a
   free, reachable cell on an informative side. Otherwise regenerate or mark
   the map sign-free and exclude sign-dependent scenarios.
3. Preserve the diagnostic protocol and the 3-unit cap; do not tune prompts or
   face selection to rescue this run.
4. Freeze the repaired benchmark as a new version, rerun the 10-map CPU
   replication, and only then decide whether Phase 1 is eligible.
5. Report both raw and opportunity-conditioned metrics, but label the latter
   as benchmark repair, not as the preregistered result.

## 10. Artifacts and checks

- Analysis script: `scripts/analyze_grid_phase0.py`.
- Full suite on the research branch: `python -m pytest -q` -> 418 passed.
- Analysis script lint: `python -m ruff check scripts/analyze_grid_phase0.py`
  -> clean. The repository-wide ruff backlog is pre-existing and not claimed
  clean here.
- Raw rows are ignored because `results/` is generated; the analysis JSON and
  this report are the durable record.

## 11. Limitations

- Symbolic policy only; no VLM calls, no visual encoder, no external validity.
- 10 map indices, 8 unique geometries, 7 unique in the eligible subset.
- One deterministic run seed per method per scenario; no repeated stochastic
  samples.
- Inspect noise is irreducible at the current budget; five failures are
  classified accordingly.
- The 100-map audit is a generation audit, not a run of the grid.
- The preregistration remains `DRAFT_NOT_FROZEN`; this report cannot promote it.

## 12. Bottom line

Active rule identification remains the primary research claim, and the
symbolic diagnostic is promising on well-formed maps. The present benchmark is
not yet valid enough to test that claim at 95%: duplicate geometry and
unreadable signposts create impossible episodes, and a remaining face-selection
limitation keeps base below 100% even on the eligible subset. The correct next
action is to repair and re-freeze the benchmark, not to implement Phase 1 or
tune the policy against the failed gates.
