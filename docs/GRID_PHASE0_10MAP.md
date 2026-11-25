# Phase 0 10-map CPU replication: repaired benchmark, remaining base-gate failure

**Status: development result, not preregistered, acceptance gates not passed, Phase 1 not started.**

This report has two parts. Sections 3-4 describe the original v0.1 run, which
failed five of seven gates and exposed two benchmark-construction defects, and
the repairs made in response. Sections 5-12 report the v0.3 replication. The
repair was performed after seeing v0.1 and is therefore **post-hoc benchmark
repair**, not a preregistered confirmatory result. v0.3 passes six of seven
gates; the base-preservation gate still fails at 49/50. Under the stop rule,
no Phase 1 VLM smoke, no 55-episode evaluation, and no GPU grid were run. No
prompt or face-selection rule was tuned against the failed gate.

## 1. Decision

- The original v0.1 run scored **518/550 (94.2%)** for `diagnostic_belief`
  versus **382/550 (69.5%)** for `no_adaptation`, but identification was only
  **276/300 (92.0%)** and 20 hazard/sign episodes used zero tools.
- The repaired v0.3 run scores **534/550 (97.1%)** versus **378/550 (68.7%)**.
  Identification is **287/300 (95.7%)**, hazard **49/50 (98.0%)**, sign
  **49/50 (98.0%)**, and base **49/50 (98.0%)**. There are **zero** zero-tool
  hazard/sign episodes; the maximum is 3 tool units.
- Six gates pass, including the 95% identification gate. The hard
  **base-preserved 100%** gate fails on one stochastic navigation episode
  (`m2-s2-identification-base-None`).
- The overall paired gain is positive: 364 both, **170 diagnostic-only**, **14
  baseline-only**, 2 neither; exact McNemar `p = 3.127131016108e-35`.
- The benchmark is now unique and sign-readable: 10/10 v0.3 maps are distinct
  and have a reachable side-0 face; the 100-map audit is 100/100 unique with
  no missing side-0 face.
- **Decision:** stop again. The v0.3 result does not establish the 100% base
  gate, so Phase 1 remains blocked. Do not relax the gate, change the
  stochastic environment, or tune the policy after seeing this result.

## 2. What was run

- Branch: `research/belief-state`.
- v0.1: 10 maps x 5 run seeds x 11 scenario types = 550 episodes; methods
  `diagnostic_belief` and `no_adaptation`; 1,100 rows.
- v0.3: same scenario grid on the repaired generator and side-0-first policy;
  550 episodes and 1,100 rows.
- Hardware: CPU only. The frozen VLM was not loaded or called.
- `DiagnosticBeliefPolicy` is a deterministic symbolic reference, not a VLM.
  It uses the public map, Bayesian `GridBelief`, a fixed sign/blue diagnostic
  tree, A* navigation, and the 3-unit tool budget.

Reproduction of v0.3:

```powershell
$env:PYTHONPATH = 'src'
python scripts/export_grid_baselines.py `
  --methods diagnostic_belief,no_adaptation `
  --maps 10 --seeds 5 `
  --out results/grid_phase0/maps10_v3.jsonl
python scripts/analyze_grid_phase0.py `
  --input results/grid_phase0/maps10_v3.jsonl `
  --out-dir results/grid_phase0/maps10_v3_analysis `
  --maps 10 --seeds 5
```

Raw and analyzed artifacts are generated and ignored by Git:

- `results/grid_phase0/maps10_v3.jsonl` (SHA-256
  `A6DA6D7087D0B70BD25928EE5A463BBC2725FE29D77C336A3ED8D25E16BCC61B`)
- `results/grid_phase0/maps10_v3_analysis/` (`aggregate.json`, `per_map.json`,
  `paired.json`, `failures.json`, `sign_affordance.json`,
  `map_identity.json`, `eligibility.json`,
  `zero_tool_hazard_or_sign.json`, `summary.md`)
- `results/grid_phase0/map_suite_audit_100_v3.json`

## 3. Original v0.1 result (historical)

| Method | Success | Rate | Env steps | Tool units |
| --- | ---: | ---: | ---: | ---: |
| `diagnostic_belief` | 518/550 | 0.942 | 5,683 | 688 |
| `no_adaptation` | 382/550 | 0.695 | 12,218 | 0 |

Identification was 276/300; hazard 39/50; sign 38/50; base 49/50; there were
20 raw zero-tool hazard/sign rows (19 failures and one accidental success).
Paired: 371 both, 147 diagnostic-only, 11 baseline-only, 21 neither;
`p = 1.5885257017954777e-31`. The failure taxonomy was 19
`zero_tool_unidentified`, 7 `post_shift_no_recovery`, 1 `post_shift_late`, and
5 `inspect_noise`.

### 3.1 Duplicate map geometry

The 10 requested v0.1 maps contained only **8 unique geometries**: `m1 = m2`
and `m7 = m8`. The 100-map audit contained only **62 unique geometries**, with
25 duplicate groups and 38 redundant indices. The retry RNG used
`random.Random(seed + attempt)`, so `seed=1,attempt=1` and `seed=2,attempt=0`
were the same stream. Scenario ids stayed unique while their geometry did not.

### 3.2 Unreadable signposts

v0.1 maps 7 and 8 had no legal informative sign face: side 0 was out of bounds
or walled, side 1 was walled, and side 2 was shared. That produced the 20
zero-tool hazard/sign rows. On the 100-map v0.1 audit, 11/100 indices had no
informative face. A valid benchmark must gate on sign readability, not only
goal reachability.

### 3.3 Proximity-based face selection

On map 5 the policy chose the nearer side 1 instead of side 0. Side 1 rules
out the remap hypotheses but does not separate `base` from
`hazard_goal_swap`/`sign_composition`; only side 0 does. This was a real
policy limitation, not a prompt problem. It is repaired in v0.3 by implementing
the already-documented side-0 priority.

## 4. v0.3 benchmark repairs

1. **Collision-free retry streams.** Map generation now uses
   `random.Random(seed * 1_000_003 + attempt)`; adjacent seeds cannot alias.
   The 100-map audit is **100 unique geometries**, with maximum 11 attempts and
   no duplicate groups.
2. **Side-0 generation gate.** A candidate is rejected unless side 0 has a free
   cell reachable from the start without crossing a marker, the signpost, or
   the closed door. Side 0 is the only face that separates `base` from the
   hazard/sign hypotheses; side 1 is not sufficient for the base gate. The
   v0.3 100-map audit has **no map without a side-0 face**; the 10-map suite
   is 10/10.
3. **Side-0-first policy.** `_go_read_sign` now tries side 0 first and leaves
   side 1 even when side 1 is nearer. Side 1 remains a fallback only when side
   0 has no legal route. This implements the stated protocol; it was not
   selected by searching face policies against the metric.
4. **Audit correction.** `scripts/analyze_grid_phase0.py` now blocks the closed
   door in the affordance audit, reports side-0 reachability separately, and
   no longer states the stale `seed + attempt` cause.

## 5. v0.3 main results

### 5.1 Overall

| Method | Success | Rate | Env steps | Tool units |
| --- | ---: | ---: | ---: | ---: |
| `diagnostic_belief` | **534/550** | 0.971 | 4,953 | 612 |
| `no_adaptation` | 378/550 | 0.687 | 12,782 | 0 |

Paired counts: 364 both, **170 only diagnostic**, **14 only baseline**, 2
neither. Exact McNemar `p = 3.127131016108e-35`.

### 5.2 By scenario type

| Type | Diagnostic | Baseline |
| --- | ---: | ---: |
| identification | 287/300 (0.957) | 128/300 (0.427) |
| change_detection | 197/200 (0.985) | 200/200 (1.000) |
| no_shift | 50/50 (1.000) | 50/50 (1.000) |

The overall gain is still concentrated in identification. On change detection
the diagnostic loses 3 episodes; paired exact `p = 0.25`, so this run does not
show the v0.1 post-shift deficit at conventional significance, but the
structural budget limitation remains.

### 5.3 Identification by initial rule

| Initial rule | Diagnostic | Baseline |
| --- | ---: | ---: |
| `base` | 49/50 | 50/50 |
| `hazard_goal_swap` | 49/50 | 3/50 |
| `sign_composition` | 49/50 | 7/50 |
| `remap_swap` | 50/50 | 0/50 |
| `remap_rotate` | 50/50 | 18/50 |
| `slippery` | 40/50 | 50/50 |

The two sign-dependent rules improved from 39/50 and 38/50 to 49/50 each,
and zero-tool episodes disappeared. `slippery` remains the largest
identification deficit: 40/50 versus 50/50 for the baseline, with 10
navigation timeouts. The diagnostic identifies the rule but does not finish
within 60 steps on maps 2 and 4.

### 5.4 Per map

| Map | Diagnostic | Baseline | Tool units |
| --- | ---: | ---: | ---: |
| m0 | 54/55 | 38/55 | 60 |
| m1 | 55/55 | 35/55 | 65 |
| m2 | 49/55 | 38/55 | 59 |
| m3 | 55/55 | 40/55 | 61 |
| m4 | 50/55 | 35/55 | 60 |
| m5 | 54/55 | 40/55 | 62 |
| m6 | 55/55 | 40/55 | 63 |
| m7 | 54/55 | 35/55 | 60 |
| m8 | 55/55 | 41/55 | 60 |
| m9 | 53/55 | 36/55 | 62 |

All 10 geometries are distinct. All 10 have a reachable side-0 face.

## 6. Acceptance gates

| Gate | Target | Observed | Result |
| --- | ---: | ---: | --- |
| identification | >= 95% | 95.7% | PASS |
| hazard | >= 90% | 98.0% | PASS |
| sign | >= 90% | 98.0% | PASS |
| base preserved | 100% | 98.0% | FAIL |
| no zero-tool hazard/sign | 0 | 0 | PASS |
| maximum tool units | <= 3 | 3 | PASS |
| paired gain positive | diagnostic > baseline | 170 > 14 | PASS |

Six of seven gates pass. The single base failure is enough to fail the run
under the stated rule.

## 7. Failure taxonomy and root causes

| Classification | Count |
| --- | ---: |
| `navigation_timeout` | 11 |
| `inspect_noise` | 2 |
| `post_shift_no_recovery` | 3 |
| **Total failures** | **16** |

### 7.1 The base failure is stochastic navigation, not a benchmark defect

`m2-s2-identification-base-None` uses zero tools and reaches the 60-step cap
with final estimate `slippery`. The true rule is `base`; the base policy has a
0.1 slip probability. The observed slips move the filter toward `slippery`, and
the policy commits before reading the sign. Both rules have the red goal, but
the planner follows the estimated rule's movement model; committing to
`slippery` changes the plan and expects two-cell slides. The baseline, which
always plans for `base`, wins this episode.

Map 2 has a reachable side-0 face, so this is not the old unreadable-signpost
defect. It is a posterior/commit decision under stochastic movement. Changing
the commit threshold, forcing a base plan, or adding a retry after seeing this
one failure would be post-hoc tuning; none was done.

### 7.2 Slippery navigation

Ten `slippery` identification episodes time out: five on map 2 and five on map
4. The diagnostic posterior can identify the rule, but the 60-step cap leaves
too little margin under the 0.8 slip / two-cell-slide dynamics; in most of these
episodes the tool budget is not the binding constraint. The baseline goes
directly to red and wins 50/50. This is a real cost of the diagnostic protocol,
not an unreadable-signpost artifact.

### 7.3 Inspect noise

Two failures are `inspect_noise`: `m0-s3-identification-sign_composition-None`
and `m5-s1-identification-hazard_goal_swap-None`. The public `INSPECT` error is
0.05; with one inspect and a 3-unit budget there is no affordable retry. The
analysis reproduces the episode and checks the tool outcome before assigning
this class.

### 7.4 Post-shift failures

Three change-detection episodes are `post_shift_no_recovery`: `m7-s2` to
`sign_composition`, `m9-s1` to `remap_swap`, and `m9-s1` to
`sign_composition`. The initial diagnosis spends the budget; after the shift
the policy has no probe left. Active rule identification remains the primary
claim; general post-shift adaptation is a future extension, not part of this
gate.

## 8. Eligible-map sensitivity

All 10 v0.3 maps have a reachable side-0 face, so the eligible subset is the
full run: no map is excluded and the sensitivity table is identical to section
5.3. This is not a substitute for the failed base gate; it confirms that the
remaining failure is not caused by unreadable signposts.

## 9. Artifacts and checks

- Analysis script: `scripts/analyze_grid_phase0.py`.
- Full suite on the research branch: `python -m pytest -q` -> **422 passed**.
- Changed-file lint: `python -m ruff check` on the modified source, script, and
  map-gen tests -> clean. The repository-wide ruff backlog is pre-existing and
  not claimed clean here.
- Raw rows are ignored because `results/` is generated; the analysis JSON and
  this report are the durable record.

## 10. Limitations

- Symbolic policy only; no VLM calls, no visual encoder, no external validity.
- The v0.3 repair was chosen after v0.1 failed. It is a development benchmark,
  not a preregistered confirmatory test.
- 10 map indices, 10 unique geometries, one deterministic run seed per method
  per scenario; no repeated stochastic samples.
- Inspect noise is irreducible at the current budget; two failures are
  classified accordingly.
- The 100-map audit is a generation audit, not a run of the grid.
- The preregistration remains `DRAFT_NOT_FROZEN`; this report cannot promote it.

## 11. What would make the next study valid

These are decisions for a new preregistration, not changes made here:

1. Decide in advance whether a 100% base gate is compatible with a stochastic
   base rule at 60 steps. If not, preregister an appropriate repeated-seed or
   tolerance rule before looking at more results. Do not relax it post hoc.
2. Keep the collision-free generator and side-0 affordance gate.
3. Pre-register the side-0-first protocol and the 3-unit budget.
4. Run a fresh 10-map CPU replication as a new benchmark version, then decide
   whether Phase 1 is eligible.
5. Report raw and opportunity-conditioned metrics, but label post-hoc
   subsets as such.

## 12. Bottom line

The benchmark repair worked: v0.3 has unique maps, readable side-0 faces, zero
zero-tool episodes, and a 95.7% identification rate. Six gates pass. The
remaining base failure is one stochastic navigation episode, not the old
benchmark defect and not something to tune away after inspection. The 100% base
gate therefore still blocks Phase 1. The correct next action is a new,
preregistered decision about stochastic base evaluation, not a VLM run or a
prompt change.