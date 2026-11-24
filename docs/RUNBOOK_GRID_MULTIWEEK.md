# Runbook — RuleShift-Grid multi-week real-VLM grid

This runbook is for any agent or human joining the 100 maps × 5 seeds real
frozen-VLM grid. It is self-contained: read this before running or merging a
chunk. The implementation plan is
`docs/superpowers/plans/2026-11-24-ruleshift-grid-multiweek-run.md`.

## Status and scope

- This is a **real-VLM** benchmark run, not the synthetic CPU smoke.
- The full grid is 100 maps × 5 seeds × 11 scenarios = **5,500 episodes**.
- It is split into 10 weekly chunks of 10 maps × 5 seeds = **550 episodes**.
- One chunk costs about **22 GPU-hours** at the measured ~2.6 s/decision; Kaggle
  has ~27 GPU-hours/week. Run at most one GPU chunk per week.
- **No result may be claimed until all ten chunks are merged** and paired CIs
  are computed at scenario level.
- `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN` until a human
  explicitly approves the freeze before week 1.

## Roles

| Role | Responsibility |
| --- | --- |
| Tooling agent | Implements and tests Tasks 1–5 of the plan; owns `scripts/grid_freeze_manifest.py`, pilot slicing, chunk artifacts, baseline exporter, merge script. |
| Run agent | Executes one Kaggle chunk per week; never changes code or config; records checksums. |
| Merge agent | After chunk `c09`, downloads all chunk artifacts, verifies coverage, runs the merge and CI script. |
| Review agent | Checks freeze manifest, artifact completeness, privilege separation, and CI resampling unit before any claim. |

Agents may work in separate worktrees. Every agent runs `codegraph init --yes`
in its own worktree and uses `codegraph explore` for symbol questions.

## One-time setup (before week 1)

1. Agree the frozen commit and record it. From the integration worktree:

   ```bash
   git rev-parse HEAD
   python scripts/grid_freeze_manifest.py --out grid_manifest.json
   ```

   The manifest must contain the same commit sha, model revision
   `66285546d2b821cf421d4f5eb2576359d3770cd3`, `scenario_count = 5500`, and a
   stable `scenario_list_sha256`.

2. Human approves the freeze and updates the preregistration document at
   `docs/superpowers/specs/2026-11-24-ruleshift-grid-multiweek-preregistration.md`.
   Do not flip `configs/preregistered_test.yaml` automatically.

3. Stage the source dataset from the frozen commit; its `SOURCE_COMMIT` marker
   must match the manifest commit.

   ```bash
   python scripts/stage_kaggle_dataset.py
   python -m kaggle datasets version -p kaggle_stage\ruleshift-source -r zip -m "grid multiweek freeze <sha>"
   python -m kaggle datasets status tiennamlee/ruleshift-source
   ```

## Weekly loop (chunks `c00` … `c09`)

Chunk `cXX` uses `map_offset = 10 * XX`, `map_count = 10`, `seeds = 5`.

1. Copy `kaggle/kernel/run_grid_chunk.py` and a `kernel-metadata.json` into a
   temporary kernel directory. Set `id` to
   `tiennamlee/ruleshift-grid-chunk-cXX` and confirm `enable_gpu = true`.

2. Set kernel environment (in Kaggle notebook settings or the kernel script):

   ```text
   GRID_CHUNK_ID=cXX
   GRID_MAP_OFFSET=<10*XX>
   GRID_MAP_COUNT=10
   RULESHIFT_EXPECTED_COMMIT=<manifest commit sha>
   ```

3. Push and wait:

   ```bash
   python -m kaggle kernels push -p <kernel-dir>
   python -m kaggle kernels status tiennamlee/ruleshift-grid-chunk-cXX
   ```

   Accept only `KernelWorkerStatus.COMPLETE`. Inspect `HF_TOKEN present` in the
   log; the token value must never be printed.

4. Download artifacts into a shared, git-ignored store:

   ```bash
   python -m kaggle kernels output tiennamlee/ruleshift-grid-chunk-cXX -p results/grid_chunks/cXX
   ```

   Expected files:

   - `reports/grid_real_pilot.cXX.md`
   - `results/grid_real/grid_real_decisions.cXX.jsonl`
   - `results/grid_real/grid_real_evaluator_only.cXX.jsonl`
   - `results/grid_real/response_cache.cXX.jsonl`

   Record file sizes and SHA-256 checksums in `results/grid_chunks/cXX/SHA256SUMS`.

5. Upload the chunk directory to a versioned Kaggle dataset or shared storage so
   the Merge agent can retrieve it. Do not commit `results/`.

## Merge (after `c09`)

1. Export CPU baselines once for the full grid:

   ```bash
   python scripts/export_grid_baselines.py --maps 100 --seeds 5 --out results/grid_baselines_100x5.jsonl
   ```

2. Merge and compute paired CIs:

   ```bash
   python scripts/merge_grid_chunks.py --chunks-dir results/grid_chunks \
     --baselines results/grid_baselines_100x5.jsonl --out reports/grid_multiweek
   ```

3. The merge must report exactly 5,500 unique `scenario_id`s and complete
   coverage of `build_grid_scenarios(100, 5)`. Paired CIs use scenario-level
   bootstrap (`n_boot=10000`), never seed- or frame-level resampling.

## Gates and recovery

- A chunk with `json_valid_rate < 1.0` is rejected at merge; re-run that chunk
  with the same manifest sha.
- Never mix chunks produced under different manifest shas.
- Never report a partial merge as a benchmark result.
- No local GPU job. The only CUDA execution happens inside the Kaggle kernel.
- Synthetic CPU smoke output is plumbing evidence only and must never appear in
  a findings table.
