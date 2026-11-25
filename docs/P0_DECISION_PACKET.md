# P0 decision packet

Date: 2026-11-25. Status: **measurement repair complete for review; no confirmatory
claim and no merge authorization.**

This packet records what was repaired, what the old rows actually say, and what a
human must decide before any candidate tuning or unseen-map run. The new plan is
treated as a sequencing correction, not as approval for a holdout.

## 1. Scope completed

- Independent task, terminal-MAP, commitment, base, tool and shift fields are
  implemented in `grid-identification-v1`.
- Explicit manifest construction and validation are implemented; manifests carry
  public-layout hashes and the full 11-scenario map-seed contract.
- The development runner accepts only an explicit manifest, verifies layouts,
  refuses maps above 199, and never invokes a model, GPU or remote service.
- The no-rerun audits recompute legacy rows, detect exposure and compare the
  proposed candidate layouts.
- No policy behavior was changed. v0.3 and repaired v0.4 are unchanged.
- No candidate was selected or tuned. No holdout, VLM smoke, GPU run or merge was
  performed.

## 2. Exact validation

- `python -m pytest -q` -> **486 passed**, exit 0.
- Focused P0 tests (`test_grid_identification.py`,
  `test_grid_identification_runner.py`, `test_grid_audits.py`) -> **24 passed**,
  exit 0.
- `python scripts/reproduce_all.py` -> exit 0;
  `determinism=PASS`; no tracked file changed.
- New CLI entrypoints all parse `--help`.

The full suite is byte-level green; it does not authorize a scientific claim.

## 3. Legacy audit correction

The old gate used `success_rate` as the identification gate and
`final_rule_matches_initial` as a joint task-plus-estimate field. On maps 0-9,
seeds 0-4, the **300 static identification episodes per method** are:

| Method | Task | Terminal MAP | Joint | Successful misdiagnosis | MAP correct, task failed | Base task |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 287/300 (95.67%) | 297/300 (99.00%) | 287 | 0 | 10 | 49/50 |
| v4 | 298/300 (99.33%) | 297/300 (99.00%) | 297 | 1 | 0 | 50/50 |
| no_adaptation | 128/300 (42.67%) | 50/300 (16.67%) | 50 | 78 | 0 | 50/50 |

The all-550-row legacy counts (v3 534 task, 544 initial-rule agreement; v4 546
and 543; no_adaptation 378 and 300) mix in change-detection and no-shift rows.
They are not static identification rates. On change rows, initial-rule
agreement rewards a stale pre-shift answer and must not be substituted for
terminal-rule identification.

Base remains literally 49/50 for v3 and 50/50 for v4 on the exposed bank. The
failed v3 episode is `m2-s2-identification-base-None`.

The historical 100-199 export is base-only (500 rows per method), not a
six-rule holdout: v3 500/500 task and 495/500 base MAP; v4 500/500 and 493/500;
no_adaptation 500/500 and 500/500.

## 4. Exposure and candidate bank

The worktree audit found 56 text/result files, 14,938 scenario-id references, all
map ids 0-199 and none above 199. A committed-ref scan found no candidate
scenario ids. It did find six prose lines on the prior
`research/v05-protocol-and-candidate` branch proposing maps 200-209; no episode
used them.

To avoid reusing even those named ids, the draft proposes **210-239**, 30 maps.
All 30 public layouts are unique, none collide with any 0-199 layout, and there
are no scenario-id references or episodes. The candidate bank is **not selected
or authorized**.

## 5. Decision required before any next run

1. Name and freeze the declaration-capable candidate commit.
2. Approve or replace maps 210-239 and seeds 5-9, and freeze the manifest hash.
3. Approve or change the 30 x 5 x 6 = 900 static episodes per method (1,650 full
   episodes; 4,950 across three methods).
4. Approve Holm correction, exact McNemar and the H5 terminal-MAP paired metric.
5. Approve the 150-episode 100% base gate.
6. Decide whether v4 is a required secondary baseline.
7. Decide whether a VLM arm is in scope (this draft excludes it).
8. Approve or replace the fail-closed raw zero-tool gate and its evidence-coding
   exception rule.

## 6. Artifacts and availability

- Protocol: `docs/EVALUATION_PROTOCOL_V1.md`
- Metric audit: `docs/METRIC_AUDIT.md`
- Exposure audit: `docs/DATA_EXPOSURE_AUDIT.md`
- Draft preregistration: `docs/PREREGISTRATION_IDENTIFICATION_V1.md`
- Manifest builder: `scripts/build_grid_manifest.py`
- Development runner: `scripts/run_grid_identification.py`
- Legacy audit: `scripts/audit_grid_existing_rows.py`
- Exposure audit: `scripts/audit_grid_map_exposure.py`
- Regression tests: `tests/test_grid_identification.py`,
  `tests/test_grid_identification_runner.py`, `tests/test_grid_audits.py`

Generated audit outputs are `reports/p0_legacy_maps0_9.json`,
`reports/p0_legacy_v4_maps0_9.json`,
`reports/p0_legacy_holdout_100_199.json` and
`reports/p0_map_exposure_audit.json`. They are advisory, untracked and must not
be committed as results.

## 7. Stop condition

This branch is a draft P0 review artifact. **Do not merge, do not tune a
candidate, do not build or run a new holdout, and do not start a VLM or GPU
smoke** until a human records the decisions in section 5 and freezes the
candidate plus manifest.
