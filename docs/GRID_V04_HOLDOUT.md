# v0.4 base-policy candidate: exploratory holdout result

**Status: exploratory candidate evaluation. Not a gate result, not a frozen
benchmark, and not a Phase 1 authorization.**

Protocol: [`GRID_V04_CANDIDATE.md`](GRID_V04_CANDIDATE.md), committed before
the candidate was run.

## Candidate

The preregistered candidate `DiagnosticBeliefPolicyV4` changes three things
relative to v0.3:

1. an empty MAP plan falls back to a `base` plan before the first-legal-action
   fallback;
2. sign reads are stored per face, and the face with the highest expected
   information gain is chosen; side 1 is therefore available when it separates
   `base` from the remap rules;
3. a rule must lead on two consecutive decisions before the existing commit
   thresholds can fire.

v0.3 remains `DiagnosticBeliefPolicy` and is unchanged.

## Preregistered fresh-map holdout (before repair)

- Maps: generator indices **100-199**, unused in the v0.3 10-map study, the
  100-map base-stress study, and the 100-map audit.
- Seeds: 0-4; 500 true-base identification episodes per policy.
- Raw: `results/grid_phase0/v04_holdout_100x5.jsonl`; SHA-256
  `02315d1474bfdfb535cfe6406e748bd039a7878a9e0cbd7d92647a00b5315779`.

| Policy | Task failures | Misdiagnoses |
| --- | ---: | ---: |
| v0.3 diagnostic | 0/500 (0.0%) | 5/500 (1.0%) |
| v0.4 candidate | 0/500 (0.0%) | 4/500 (0.8%) |
| `no_adaptation` | 0/500 (0.0%) | 0/500 (0.0%) |

Task-failure Wilson 95% for both diagnostic policies: `[0.0%, 0.76%]`. The
map-cluster interval is degenerate at `[0.0%, 0.0%]` because every map has
the same zero event rate; that is not evidence of certainty.

v0.4 misdiagnosis: 0.8%, Wilson 95% `[0.31%, 2.04%]`, map-cluster 95%
`[0.0%, 1.8%]`. v0.3 misdiagnosis: 1.0%, Wilson 95% `[0.43%, 2.32%]`,
map-cluster `[0.2%, 2.0%]`.

### Paired comparison

- Task success: **500 both, 0 only v0.4, 0 only v0.3, 0 neither**; exact
  McNemar `p = 1`. The holdout has a task-success ceiling: v0.3 also has no
  failures, so it cannot show that v0.4 fixes task failures.
- Misdiagnosis: 4 both, **1 only v0.3** (`m166-s0`), 0 only v0.4; exact
  McNemar `p = 1`. The difference is one episode, not a reliable improvement.

All four v0.4 misdiagnoses still succeeded, in 4-9 steps, with a final
`slippery` belief of 0.542, 0.894, 0.849, and 0.714. They are early
terminations, not task failures. The `base` versus `slippery` sign ambiguity
identified in the v0.3 diagnosis remains.

### Mechanism use (before repair)

On the 500 holdout episodes the candidate used the base-plan fallback 4 times
in 2 episodes, and read side 1 in 13 episodes. The per-side set was `{0}` in
479 episodes, `{0, 1}` in 13, and empty in 8. The fallback and side-1 path
are therefore exercised, but the holdout did not contain a v0.3 task failure
for them to repair.

## Full-grid regression (before repair)

After the base holdout, v0.4 was run on the original 10-map all-scenario CPU
benchmark (550 episodes, all six rules and shift types). This is a post-hoc
safety check, not a gate.

| Policy | Overall | Base initial | Slippery identification | Sign-composition identification | Hazard identification |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0.3 | 534/550 | 296/300 | 40/50 | 49/50 | 49/50 |
| v0.4 | 538/550 | 297/300 | 50/50 | **41/50** | 50/50 |

v0.4 gained ten slippery episodes and one hazard episode, but lost eight
sign-composition episodes (`m5-s1` through `m5-s4` and `m6-s0` through
`m6-s3`); those eight ended after 60 steps with a wrong `hazard_goal_swap`
commit. The rule-level regression made the candidate unsafe to promote.

The traced cause was a tool-allocation error. After the first sign face,
`hazard_goal_swap` and `sign_composition` remain at 0.5/0.5. A negligible
remap tail created a second-face expected gain around `3e-9`, above the
candidate's `1e-9` floor. v0.4 spent its remaining tool unit on that face
instead of inspecting blue, so it could not separate the two rules. Raising
the gain floor was explored and rejected: it transferred the `m5-s1` hazard
case to a failure and did not robustly remove all six remaining failures.

## Post-hoc repair (not preregistered)

The candidate was changed **after** these outcomes. This is not a valid rerun
under the v0.4 protocol: maps 100-199 are now development data for the
repaired policy, and the repaired results below are descriptive only. They
cannot be used as confirmatory holdout evidence.

The repair is confined to `DiagnosticBeliefPolicyV4` through two feature flags
that remain false in v0.3. It adds three rules:

1. **Leader-separating sign face.** Skip a sign face whose clause is identical
   for the two leading rules; reading it cannot change which rule leads.
2. **Blue budget reservation.** When exactly the two-unit inspection budget
   remains, inspect blue before another sign read if blue gives the two
   leading rules different roles. This prevents a sign read from stranding the
   budget needed to separate `hazard_goal_swap` from `sign_composition`.
3. **Slippery commit fallback.** When the MAP rule is `slippery`, commit the
   `base` plan. Slippery and base share movement support, and the slippery MAP
   plan can cycle under its own high slip rate.

v0.3 has both flags false and keeps its previous behavior; the full
550-episode v0.3 count below is unchanged at 534/550.

### Post-hoc full-grid check (same 10 maps)

| Policy | Overall | Base initial | Slippery identification | Sign-composition identification | Hazard identification |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0.3 | 534/550 | 296/300 | 40/50 | 49/50 | 49/50 |
| v0.4 repaired | **546/550** | 298/300 | 50/50 | 49/50 | 49/50 |

By scenario type, the repaired v0.4 has 198/200 change-detection, 298/300
identification, and 50/50 no-shift successes (v0.3: 197/200, 287/300, 50/50).
All eight sign-composition identification failures are recovered. The four
remaining v0.4 failures are a strict subset of the v0.3 failures:

- `m0-s3-identification-sign_composition-None`;
- `m5-s1-identification-hazard_goal_swap-None`;
- `m7-s2-change_detection-base-sign_composition`;
- `m9-s1-change_detection-base-sign_composition`.

Paired task success against v0.3: **534 both, 0 only v0.3, 12 only v0.4, 4
neither**; exact McNemar `p = 0.00048828125`. This is a post-hoc same-map
safety check, not a new benchmark result.

Raw: `results/grid_phase0/v04_fixed_fullgrid_10x5.jsonl`; SHA-256
`f5236a5ae4cc15fc11ad9fd08d6094c00a95f2a0af5b924750903820ab57d259`.
Summary: `results/grid_phase0/v04_fixed_fullgrid_analysis/summary.json`.

### Post-hoc base holdout rerun (same maps; descriptive only)

| Policy | Task failures | Misdiagnoses |
| --- | ---: | ---: |
| v0.3 diagnostic | 0/500 (0.0%) | 5/500 (1.0%) |
| v0.4 repaired | 0/500 (0.0%) | 7/500 (1.4%) |
| `no_adaptation` | 0/500 (0.0%) | 0/500 (0.0%) |

Raw: `results/grid_phase0/v04_fixed_holdout_100x5.jsonl`; SHA-256
`98c4d2acc80b1bcd3571b1c70950f0021c570d5a32b114ec0b7f727e5225edf6`.

Task-failure Wilson 95% is still `[0.0%, 0.76%]`; the map-cluster interval is
still degenerate at `[0.0%, 0.0%]`. Repaired v4 misdiagnosis is 1.4%, Wilson
95% `[0.68%, 2.86%]`, map-cluster `[0.4%, 2.6%]`; v0.3 is 1.0%, `[0.43%,
2.32%]`, map-cluster `[0.2%, 2.0%]`. Paired task success is 500 both, 0
discordant, `p = 1`. Paired misdiagnosis: 4 both, **1 only v0.3**, **3 only
v0.4**, 492 neither; exact McNemar `p = 0.625`. The three v0.4-only
misdiagnoses are early successful `slippery` terminations (`m143-s1`,
`m155-s3`, `m179-s0`, 4-6 steps); they are not task failures.

On this true-base holdout the repaired policy did not read side 1, inspect
blue, or use the empty-plan fallback. The sign-side set was `{0}` in 485
episodes and empty in 15. The repaired probe rules are exercised by the
full-grid sign and hazard cases, not by this holdout. That is a limitation of
this descriptive rerun, not evidence that the probe repair is generally
unnecessary.

## Interpretation

- The v0.3 base gate remains failed: **49/50** on the official 10-map run.
  Neither the holdout nor any v0.4 result can pass or replace it.
- The preregistered v0.4 holdout does not regress on the fresh task endpoint,
  but it cannot demonstrate a task-failure improvement because v0.3 already
  has zero failures there.
- The original full-grid regression is real and was traced to tool allocation.
  The post-hoc repair removes it and produces a strict full-grid improvement
  over v0.3 on the same maps. It also leaves four failures, all of which v0.3
  also has.
- The repaired holdout has more misdiagnoses than the original candidate (7
  versus 4) and more than v0.3 (5). The paired difference is not significant
  (`p = 0.625`), and all remain successful early episodes; the terminal belief
  metric still counts unsettled or tied beliefs as wrong. This does not
  justify another tuning pass on these maps.
- The repair was chosen after seeing the eight sign failures. It is therefore
  not a preregistered candidate, and its full-grid and holdout numbers cannot
  be reported as confirmation. A future candidate must use a new map offset
  and a fresh preregistration.
- The literal 100% gate is still the binding blocker. v0.4 is not frozen, not
  promoted, and not used by the benchmark.

## Decision

Do not start Phase 1, the GPU grid, or VLM work. Do not promote the repaired
v0.4 as the benchmark policy. The next step remains a human decision about the
base endpoint and a fully preregistered candidate evaluation on unused maps.

## Reproduce

```bash
# Preregistered holdout (original candidate; result above).
python scripts/run_grid_v04_holdout.py

# Post-hoc repaired holdout (descriptive only; same maps).
python scripts/run_grid_v04_holdout.py \
  --out results/grid_phase0/v04_fixed_holdout_100x5.jsonl \
  --analysis-dir results/grid_phase0/v04_fixed_holdout_analysis
```

Generated artifacts stay under `results/` and are not committed.
