# Freeze record: identification v1 static manifest

Date: 2026-11-25. Status: **frozen proposal; not executed.**

This record identifies the exact candidate, manifest and analysis code proposed
for the 2,700-episode static CPU experiment. It is a freeze-readiness artifact,
not a confirmatory result.

## 1. Frozen identities

| Object | Identity |
|---|---|
| Candidate policy commit | `e65e1915503246f0f07ed15c4884ac6952925ee8` |
| Behavioural baseline | `f4b43a4ede7f76ab9d5b1f45178fe648ae419042` |
| Coverage/reporting correction | `a46ed21` (evaluator only) |
| Manifest commit | `21db1e2` |
| Manifest path | `artifacts/manifests/identification_v1_static.json` |
| Manifest content SHA-256 | `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80` |
| Config SHA-256 | `67f7be9716947a9b30f6d30fdcdd72f0abb95277ea7f2f5e8eb09cde03cb8421` |
| Scope | `static_identification` |
| Entries | 900 |
| Maps | 210–239 |
| Seeds | 5–9 |
| Initial rules | 6 |
| Unique public layouts | 30 |
| Collisions with maps 0–199 | 0 |

The content hash is the authoritative identity. It is computed over canonical
manifest entries, independent of checkout line endings. A local Windows
working-file hash of `646f89632319a443943fc0e1add5cffb9579764e18cef07cdc7b325620d9fef5`
is recorded only for this working copy; a Git checkout can normalize line endings,
so the file-byte hash is not portable and must not be used as the scientific
identity. The manifest scope is separately validated against its exact scenario
membership.

## 2. Construction and validation

Construction used the public map generator only:

    python scripts/build_grid_manifest.py \
      --maps 210-239 --seeds 5-9 \
      --scope static_identification \
      --out artifacts/manifests/identification_v1_static.json

No environment reset, policy instantiation or episode occurred. Validation,
without rollouts, established:

- 30 maps × 5 seeds × 6 rules = 900 entries;
- exactly six scenarios per map-seed, one per initial rule;
- 900 unique scenario ids and 30 unique public layout hashes;
- no duplicate layout within the candidate bank;
- no layout collision with any map 0–199;
- manifest content hash, scope, exact id set and layout hashes consistent;
- missing, extra, duplicate or relabelled scenarios fail closed.

The validation code is `scripts/run_grid_identification.py::validate_manifest`
and `src/ruleshift/evaluator/grid_manifest.py`. The manifest builder has no
policy import and cannot run an episode.

## 3. Exposure check

The pre-freeze worktree audit found:

- 56 text/result files with scenario references;
- 14,938 scenario-id references;
- map ids 0–199 only;
- zero references to maps 210–239;
- 30 candidate layouts, all unique, zero collisions with 0–199.

A committed-ref scan found no candidate scenario ids. Six prose lines on
`research/v05-protocol-and-candidate` mentioned maps 200–209; no episode used
them. The candidate bank is 210–239, outside that prose namespace.

After the manifest was written, a scan of `results/` found **zero** candidate
scenario ids. The only `m21x-s…` strings in the working tree are the 900 entries
inside the frozen manifest itself. The manifest is therefore a public scenario
listing, not evidence of a rollout.

## 4. Development evidence and gate risk

The final exposed development run used maps 0–9, seeds 0–4 (900 episodes total;
300 static per method) and candidate commit `e65e191`. It is recorded in the
ignored archive and is not confirmatory.

| Endpoint (development) | Result |
|---|---|
| v4 static task | 298/300 |
| v4 terminal MAP | 297/300 |
| v4 explicit declarations correct | 263/300 (87.67%) |
| v4 declared wrong | 4 |
| v4 unresolved | 33 |
| v4 event coverage | 267/300 (89.0%) |
| v4 base task | 50/50 |
| v4 hazard declarations | 47/50 (94%) |
| v4 sign declarations | 44/50 (88%) |
| v4 zero-tool hazard/sign successes | 0 |
| v3 static task / MAP | 287/300 / 297/300 |
| v3 explicit declarations correct | 284/300 (94.67%) |
| no_adaptation task / MAP | 128/300 / 50/300 |

The candidate is at material risk of failing G1 (95% unconditional declaration
accuracy) and G4 (90% sign declarations), and its sign development subset is
already below the gate. This is disclosed before GO; no tuning was performed and
no gate was moved. If the reviewer requires a candidate that is already above the
gates on development data, this freeze should be rejected and a new candidate
pre-registered, not adjusted after seeing candidate maps.

## 5. Artifact archive

Raw development and audit artifacts are preserved outside Git at:

    results/archive/p0_2026-11-25/

It contains 85 files, 8,135,664 bytes, including 69 historical `grid_phase0`
files (18 top-level plus 51 nested analysis files), the prior `grid_v05_dev`
rows, the final 900-episode exposed development run, its 300-entry manifest and
four audit JSON files. The archive README states
that it is not confirmatory. `SHA256SUMS.txt` lists 84 files; all 84 were verified
with zero mismatches. The checksum manifest's SHA-256 is:

    4735b8fc393359f3b23d9ce4c01d6fc049f0fab8f1ae035740ae5b762516df59

The archive is ignored by Git. No raw archive file is committed as a result.

## 6. Test and reproducibility status

Verified before the freeze commit:

- full CPU suite: **496 passed**, exit 0;
- focused declaration equivalence: **3 passed**, 600/600 exposed scenarios
  behaviourally identical;
- focused declaration/manifest/map-analysis tests: **30 passed**;
- `scripts/reproduce_all.py`: exit 0, all synthetic artifacts regenerated,
  `determinism=PASS`;
- CLI `--help`: exit 0 for the manifest builder, identification runner,
  reproducibility driver and exposure audit;
- working tree contains only the freeze documents and generated artifacts at
  commit time; raw evidence is in the ignored archive.

## 7. Exact GO condition

The confirmatory run is not armed. The development runner refuses maps above 199.
After a separate explicit GO, a reviewed authorization commit may add a
manifest-hash-pinned confirmatory entry point; it may not change candidate policy
or analysis code.

GO requires acceptance of:

1. candidate `e65e191` and behavioural equivalence;
2. manifest content hash `95c4e1…be80`, maps 210–239, seeds 5–9;
3. 900 static episodes per method, 2,700 total CPU episodes;
4. H1/H2 only with Holm, 5-point minimum effect and map-cluster intervals;
5. G1–G6, including the disclosed G1/G4 risk;
6. v4−v3 descriptive only; no baseline declaration API;
7. no VLM/GPU, no shift, no tuning, no post-holdout amendment.

Default: **NO-GO**. Stop after this freeze-readiness packet and draft PR.
