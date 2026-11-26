# P0 raw-evidence archive inventory

Date: 2026-11-25. Status: preserved development evidence, **not confirmatory**.

## Location

    results/archive/p0_2026-11-25/

The directory is under `results/`, which is git-ignored. It is intentionally not
committed as a result set. The tracked record of its existence and checksum is
this inventory plus the freeze record.

## Contents

| Group | Files | Description |
|---|---:|---|
| `grid_phase0/` | 69 | historical exposed-map rows, base stress, baselines, audits and tables, including their nested analysis directories |
| `grid_v05_dev/` | 7 | prior candidate development rows, traces and summary on exposed maps |
| `p0_dev_static_e65e191/` | 2 | final 900-episode exposed development records and summary |
| `reports/` | 4 | legacy audit and pre-freeze exposure JSON |
| root | 3 | `p0_dev_static_manifest.json`, `ARCHIVE_README.md` and `SHA256SUMS.txt` |

Totals: **85 files, 8,135,664 bytes**. The checksum manifest lists 84 files
(everything except itself).

## Integrity

- checksum-manifest SHA-256:
  `4735b8fc393359f3b23d9ce4c01d6fc049f0fab8f1ae035740ae5b762516df59`
- verification result at archive creation: **84 checked, 0 mismatches**
- format: lowercase SHA-256, two spaces, relative path with forward separators.

Verification from the repository root:

    $dest = Join-Path (Get-Location) "results/archive/p0_2026-11-25"
    foreach ($line in Get-Content (Join-Path $dest "SHA256SUMS.txt")) {
      $hash, $relative = $line -split "  ", 2
      $actual = (Get-FileHash -LiteralPath (Join-Path $dest $relative) -Algorithm SHA256).Hash.ToLower()
      if ($actual -ne $hash) { Write-Error "MISMATCH $relative" }
    }

## Scope caveats

- The archive contains **no candidate-map episode** (maps 210–239). It contains
  only public layout hashes in the frozen manifest and exposed development data.
- The historical rows include legacy labels that are not identification rates;
  they are preserved for audit, not pooled with the new endpoints.
- The archive is not a substitute for the frozen 2,700-episode experiment and
  must not be cited as a holdout result.
- The final frozen manifest is tracked at
  `artifacts/manifests/identification_v1_static.json`; its authoritative content
  SHA-256 is `95c4e13e533d1161af9cf6971df905c3222d94fbf091d3bf71e5ff3b37cabe80`.
