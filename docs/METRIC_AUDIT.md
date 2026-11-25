# P0 metric audit: legacy Grid rows

Date: 2026-11-25. Status: development-only audit, no reruns.

Inputs were existing JSONL rows under results/. The audit script reads the rows,
parses scenario ids, and compares the legacy fields with independent task and
terminal-MAP fields. It does not instantiate an environment or a policy.

Raw audit outputs are generated, git-ignored artifacts:

- reports/p0_legacy_maps0_9.json
- reports/p0_legacy_v4_maps0_9.json
- reports/p0_legacy_holdout_100_199.json

## 1. The old identification gate

The legacy analyzer used task success_rate for its identification row. It also
used final_rule_matches_initial, but that field required success. The old field
is therefore a joint success-and-estimate match, not independent identification
accuracy.

The legacy exporter also wrote shift_detected as:

    episode.final_rule_estimate != "base"

That is a terminal-label proxy, not an actual shift, a detection event, or a
detection time. It is retained only as a historical label.

## 2. Maps 0-9, seeds 0-4

These are the exposed development maps used by v0.3 and v0.4. Each method has
550 episodes: 300 static identification episodes, 200 change-detection episodes
and 50 no-shift episodes. The v3 and v4 rows are in
results/grid_phase0/v04_fixed_fullgrid_10x5.jsonl. The v3 rows are also present
in maps10_v3_postmerge.jsonl with the same counts.

The primary identification view is the 300 static episodes:

| Method | Static episodes | Task success | Independent terminal MAP | Joint task+MAP | Successful misdiagnosis | MAP correct, task failure | Base task | Base MAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_belief_v3 | 300 | 287 (95.67%) | 297 (99.00%) | 287 | 0 | 10 | 49/50 | 49/50 |
| diagnostic_belief_v4 | 300 | 298 (99.33%) | 297 (99.00%) | 297 | 1 | 0 | 50/50 | 50/50 |
| no_adaptation | 300 | 128 (42.67%) | 50 (16.67%) | 50 | 78 | 0 | 50/50 | 50/50 |

For all 550 rows, the legacy context is:

| Method | Task success | Initial-rule agreement | Legacy joint field |
|---|---:|---:|---:|
| diagnostic_belief_v3 | 534 (97.09%) | 544 (98.91%) | 532 |
| diagnostic_belief_v4 | 546 (99.27%) | 543 (98.73%) | 541 |
| no_adaptation | 378 (68.73%) | 300 (54.55%) | 300 |

The 550-row initial-rule agreement is not a static identification rate. On
change-detection rows it compares the final estimate with the initial rule, not
with the shifted rule, and it rewards the no-adaptation baseline for assuming
base everywhere.

Reading the static table:

- v3 identified 297 static rules but succeeded on only 287 tasks: 10 episodes
  identified the rule yet failed the task. The legacy joint field omitted those
  10.
- v4 succeeded on 298 static tasks but matched only 297 rules: 1 episode
  succeeded with the wrong terminal rule. Task success cannot stand in for
  identification.
- The v3 base endpoint is exactly the known 49/50. The failed episode is
  m2-s2-identification-base-None. This audit neither repairs nor relabels it.
- no_adaptation reached only 50/300 static terminal MAP and 16.67% accurate
  identification. Its 78 successful misdiagnoses are task successes with the
  wrong rule. Its 300/550 initial-rule agreement on the full file is not an
  identification result.

### 2.1 Explicit-declaration audit of the final development run

The legacy rows above have no declaration API, so they cannot supply an explicit
identification metric. The final exposed development run at candidate commit
`e65e191` added declaration observations without changing actions. Its 300
static episodes per method give:

| Method | Declaration events | Episodes with an event | Correct | Wrong | Unresolved | Unconditional accuracy | Event coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_belief_v3 | 294 | 288 | 284 | 4 | 12 | 94.67% | 96.0% |
| diagnostic_belief_v4 | 268 | 267 | 263 | 4 | 33 | 87.67% | 89.0% |
| no_adaptation | 0 | 0 | 0 | 0 | 300 unavailable | unavailable | 0.0% |

The v4 mean Brier score is 0.025507 and NLL is 0.056736; v3 is 0.024447 and
0.054596. v3 has 6 episodes with an earlier wrong declaration; v4 has 1. A later
correct declaration does not erase the earlier error.

Unconditional accuracy keeps all 300 episodes in the denominator. The earlier
draft reported API availability as `declaration_coverage` and therefore showed
1.0 for v4 despite only 267/300 episodes with an event. The corrected fields are
`declaration_event_episodes`, `settled_declaration_episodes`,
`declaration_coverage`, `settled_declaration_coverage` and
`declaration_available_coverage`. The correction is evaluator-only.

The v4 base subset has 40/50 explicit correct declarations (8 unresolved), even
though base task success is 50/50. Base task success and base declaration
accuracy are different endpoints and must not be substituted for each other.

These are development counts, not the 900-episode confirmatory endpoint and not
a gate adjudication. They show the candidate's unconditional declaration
accuracy below the proposed 95% gate and its sign subset below 90%; that risk is
recorded in the decision packet before freeze.

## 3. Tool and zero-tool observations

| Method | 0 tools | 1 tool | 2 tools | 3 tools | Zero-tool hazard/sign task successes |
|---|---:|---:|---:|---:|---:|
| v3 | 140 | 309 | 0 | 101 | 0 |
| v4 | 126 | 316 | 0 | 108 | 0 |
| no_adaptation | 550 | 0 | 0 | 0 | 10 |

The last column is a raw count. Legacy rows do not expose direct cell-role
evidence, so this audit does not classify those ten as evidence-backed or
accidental. That classification requires a separate evidence audit.

## 4. Shift-proxy observations

In maps10_v3_postmerge.jsonl, shift_detected agrees with the terminal-label
proxy on all 550 rows. This is agreement between two proxies, not validation of
detection. The v04_fixed_fullgrid rows predate the field and omit it entirely.
The new record schema exports scheduled time, actual time, censoring, the policy
shift-probability event and pre-shift false alarms separately.

## 5. Legacy 100-199 base-only export

results/grid_phase0/v04_fixed_holdout_100x5.jsonl contains 500 base static rows
per method, not the full six-rule cohort. Within that limited base-only slice:

| Method | Base task | Base MAP |
|---|---:|---:|
| v3 | 500/500 | 495/500 |
| v4 | 500/500 | 493/500 |
| no_adaptation | 500/500 | 500/500 |

This is not a full holdout audit and is not used to claim a full-cohort
identification rate. It confirms the old exporter's terminal-MAP behavior on a
different exposed bank.

## 6. Reproduction

From the repository root, with the package importable from src:

    python scripts/audit_grid_existing_rows.py results/grid_phase0/v04_fixed_fullgrid_10x5.jsonl --map-max 9 --out reports/p0_legacy_v4_maps0_9.json
    python scripts/audit_grid_existing_rows.py results/grid_phase0/maps10_v3_postmerge.jsonl --map-max 9 --out reports/p0_legacy_maps0_9.json
    python scripts/audit_grid_existing_rows.py results/grid_phase0/v04_fixed_holdout_100x5.jsonl --map-min 100 --map-max 199 --out reports/p0_legacy_holdout_100_199.json

The exposure audit is separate:

    python scripts/audit_grid_map_exposure.py --out reports/p0_map_exposure_audit.json --git-refs

On Windows PowerShell, prefix these commands with $env:PYTHONPATH='src';.
The generated reports/ files are advisory outputs, not tracked sources.

## 7. Consequences

1. New reports use independent fields: task_success, map_correct,
   task_and_map_correct, successful_misdiagnosis, commitment status, Brier/NLL
   where available, tools and shift timing.
2. A policy without a declaration API is commitment-unavailable. Its MAP and
   posterior are descriptive; it cannot pass the primary commitment gate by
   substitution.
3. No old row is re-scored into the new primary endpoint. The absence of
   posterior and declaration data in the old rows is recorded as unavailable.
4. The 49/50 base result remains a literal historical gate result.
