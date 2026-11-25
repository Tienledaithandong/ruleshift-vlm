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

## 2. Candidate namespace

A proposed 30-map namespace, 210-239, was hashed only for feasibility:

| Check | Result |
|---|---:|
| Candidate map ids | 30 |
| Unique public-layout hashes | 30 |
| Duplicate candidate layouts | 0 |
| Candidate layouts colliding with any 0-199 layout | 0 |
| Candidate scenario-id references in the worktree | 0 |
| Candidate scenario-id references in committed refs | 0 |

The worktree scan found no scenario-id references above 199. A separate
read-only scan of 25 committed refs found no candidate scenario ids either. It
did find six prose lines on refs/heads/research/v05-protocol-and-candidate and
its remote copy proposing maps 200-209 in docs/PREREGISTRATION_V05.md. Those
lines are a written proposal, not an executed episode: the raw result inventory
contains no m200+ rows.

The newly proposed bank starts at 210 rather than 200 so it does not reuse even
the map ids named by that prior proposal. The candidate namespace is unexposed
as of this audit. It is not selected, frozen or authorized. The integer ids are
not the scientific unit: the layout hashes are. If any candidate layout later
appears in a development run, the whole proposed manifest must be re-audited
before approval.

This document itself will create name references when committed. That is why
the manifest, not this prose, must carry the authoritative hash list, and why
the exposure audit must be rerun after the manifest is frozen.

## 3. Boundaries

- No episode was run on maps 200-239.
- Maps 200-209 were named only in a prior-branch prose proposal; they were not
  run and are not used by the current draft.
- No candidate layout was rendered into a trajectory or inspected by a policy.
- No unseen-map evaluation, VLM smoke, GPU inference or remote inference was
  performed.
- Maps 0-199 remain development-only. Their results are not confirmatory.
- The 100-199 bank is already exposed and cannot be reused as an unseen holdout.

## 4. Required re-audit

Before any later holdout request, rerun the audit against the frozen manifest
and all artifacts created after this date. Abort if a candidate layout hash
appears in development output, if candidate layouts collide with 0-199, or if
the manifest cannot be reproduced from its recorded config.
