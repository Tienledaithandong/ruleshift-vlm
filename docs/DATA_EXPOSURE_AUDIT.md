# P0 data-exposure audit

Date: 2026-11-25. Status: development-only governance audit, no policy calls.

The audit scans existing text and result artifacts for scenario ids of the form
m<map>-s<seed>-... and hashes public map layouts. It does not instantiate a
policy, run an episode, call a model or use a GPU. Raw output is the git-ignored
reports/p0_map_exposure_audit.json.

## 1. Existing exposure

| Field | Observed |
|---|---:|
| Text/result files containing scenario ids | 56 |
| Total scenario-id references | 14,938 |
| Distinct map ids referenced | 200 |
| Minimum map id | 0 |
| Maximum map id | 199 |
| References to maps 200 or above | 0 |

All map ids 0-199 have appeared in at least one existing artifact. Maps 0-9 are
the heavily reused development bank; maps 100-199 were used by the previous
holdout-style development work. This inventory does not claim every one of
those references is an evaluation episode; it establishes that the ids and
layouts are exposed.

The result files also include duplicate scenario ids across methods. That is
expected when a combined file stores v3, v4 and no_adaptation rows together. The
new paired code rejects duplicates within a method instead of silently
intersecting them.

## 2. Frozen candidate namespace

The candidate bank is **210-239**, frozen in
`artifacts/manifests/identification_v1_static.json`:

| Check | Result |
|---|---:|
| Candidate map ids | 30 |
| Unique public-layout hashes | 30 |
| Duplicate candidate layouts | 0 |
| Candidate layouts colliding with any 0-199 layout | 0 |
| Manifest entries | 900 |
| Candidate scenario-id references in `results/` | 0 |
| Candidate scenario-id references outside the manifest | 0 |
| Candidate episode rows | 0 |

The manifest content SHA-256 is
`95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`
and its scope is `static_identification`. It contains 900 public scenario
entries: 30 maps × 5 seeds × 6 static initial rules, exactly six per map-seed.

The worktree scan before the manifest was written found 56 files and 14,938
scenario references, all maps 0-199. A read-only scan of committed refs found no
candidate scenario ids. Six prose lines on
`research/v05-protocol-and-candidate` and its remote copy propose maps 200-209
in `docs/PREREGISTRATION_V05.md`; those lines are a written proposal, not an
episode, and the raw result inventory has no m200+ row. The frozen bank starts
at 210 so it does not reuse even that prose namespace.

The post-freeze scan necessarily finds 900 candidate references **inside the
manifest itself**, because a scenario manifest lists scenario ids. That is not
exposure and no environment was reset. The decisive check is that `results/`
contains zero candidate scenario ids and that no candidate row or trace exists.
The integer ids are not the scientific unit; the 30 layout hashes are. The
manifest is frozen but not authorized: any execution still needs the separate
human GO.

## 3. Boundaries

- No episode was run on maps 200-239.
- Maps 200-209 were named only in a prior-branch prose proposal; they were not
  run and are not used by the frozen design.
- No candidate layout was rendered into a trajectory or inspected by a policy.
- No unseen-map evaluation, VLM smoke, GPU inference or remote inference was
  performed.
- Maps 0-199 remain development-only. Their results are not confirmatory.
- The 100-199 bank is already exposed and cannot be reused as an unseen holdout.
- The frozen manifest is public metadata; its entries are not confirmatory
  outcomes.

## 4. Required re-audit

The pre-execution re-audit is complete: manifest hash, exact id set, layout
recomputation, 0-199 collision check and `results/` scan all pass. Rerun the
audit after any confirmatory execution and before any later analysis. Abort if a
candidate layout hash appears in development output, if candidate layouts
collide with 0-199, if the manifest cannot be reproduced from its recorded
config, or if a candidate episode id appears outside an authorized run record.
