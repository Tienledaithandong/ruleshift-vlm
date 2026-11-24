# Runbook: RuleShift-Grid multi-week run

This runbook is for the 100-map x 5-seed grid (5,500 episodes). It is not a
license to start that run. The preregistration must first be marked READY by a
human, and the gates below must pass in order.

## 0. Preconditions

- Clean tree at the frozen commit; staged SOURCE_COMMIT matches it.
- Model revision pinned in configs/models.yaml.
- preregistered_test.yaml status is DRAFT_NOT_FROZEN until the human freeze
  commit; do not edit that field from an automated agent.
- Kaggle GPU quota checked before a weekly slice. The real smoke measured
  ~8.85 s per model decision on a T4; 55 episodes x up to 60 decisions is an
  upper bound near 8.1 GPU-hours per chunk, not 2.2. The full 5,500-episode grid
  is therefore ~800 GPU-hours (about 17 weeks at 27 h/week) before overhead.
  Treat throughput as a blocking feasibility issue; do not schedule the full
  grid until inference is materially faster or the design is resized.
- Never commit results/, reports/, outputs/, or the Kaggle token.

## 1. Freeze the provenance

The human READY commit must land first. The manifest hashes configs/preregistered_test.yaml, so generating it while that file is still DRAFT produces a manifest that cannot describe the locked run. After the READY commit, generate the manifest twice and compare byte-for-byte:

    python scripts/grid_freeze_manifest.py --out reports/grid_freeze_manifest.json
    python scripts/grid_freeze_manifest.py --out reports/grid_freeze_manifest.recheck.json
    fc reports/grid_freeze_manifest.json reports/grid_freeze_manifest.recheck.json

The manifest must contain exactly commit, model_revision, config_sha256,
scenario_count=5500, and scenario_list_sha256. Attach it to the run log.

## 2. Gate A: two-episode smoke

Build and push a source dataset only from a clean commit:

    git status --short
    python scripts/stage_kaggle_dataset.py
    cd kaggle_stage/ruleshift-source
    python -m kaggle datasets version -p . -r zip -m "grid smoke <commit>"

Push kaggle/kernel/run_grid.py and confirm its printed `source commit` equals the
frozen commit. Kaggle kernel metadata has no environment field, so
RULESHIFT_EXPECTED_COMMIT is normally unset; the printed SOURCE_COMMIT marker is
the traceability check. Download the smoke artifacts and check the behavioral gate:

- full-menu JSON valid rate = 1.0
- parser fallbacks = 0
- at least two distinct actions
- at least one moving step
- model_calls and cache_hits are lifetime totals, not one episode's counters

If the gate fails, stop. Do not reinterpret fallback actions as valid and do not
scale to a larger run.

## 3. Gate B: one 55-episode dry chunk

Stage a literal map index (Kaggle cannot set MAP_INDEX in kernel metadata):

    python scripts/stage_grid_chunk.py --map-index 0
    cd kaggle_stage/grid_chunks/m000
    python -m kaggle kernels push -p . --accelerator NvidiaTeslaT4

Push exactly one version and leave it alone. A repush while the first version is
queued creates a second batch session; two queued versions fill the account
limit and produce "Maximum batch GPU session count of 2 reached" or an
apparently endless QUEUED state. If that happens, delete the kernel, wait for
the slot to clear, and push one fresh version. Do not repush to "fix" a queue.

The kernel asserts exactly 55 evaluator rows and 55 unique scenarios. After download:

    python -c "import json,pathlib; rows=[json.loads(x) for x in pathlib.Path('results/grid_chunks/grid_real_decisions.m000.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; print(len(rows), sum(r.get('json_valid') is True for r in rows), sum(r.get('fallback') is True for r in rows))"

Expected: nonzero rows, every row json_valid, zero fallbacks. Inspect action
diversity and movement; a constant-action chunk is a failed gate.

## 4. Weekly slices

One weekly slice is ten maps = ten chunks. Stage each one (the helper writes the
index into the kernel, so every chunk is traceable even though Kaggle has no env
field):

    python scripts/stage_grid_chunk.py --map-index 0
    python scripts/stage_grid_chunk.py --map-index 1
    ...
    python scripts/stage_grid_chunk.py --map-index 9

Use a fresh Kaggle session per chunk, with the same frozen dataset. Download each
chunk into the same results/grid_chunks directory. Never overwrite an existing
mNNN file; move failed artifacts aside.

Suggested weekly order: run all ten chunks first, verify each has 55 scenarios,
then merge that slice for a sanity check. Do not inspect outcomes to decide
whether to continue; coverage is fixed in advance.

## 5. Baselines and merge

Export the public CPU baselines once at the frozen commit:

    python scripts/export_grid_baselines.py --config configs/grid_dev.yaml --out results/grid_baselines/baselines.jsonl

After all 100 chunks are present:

    python scripts/merge_grid_chunks.py --chunks results/grid_chunks --baselines results/grid_baselines/baselines.jsonl --out results/grid_merged/summary.json

The merge refuses to write a summary unless:

- every scenario appears exactly once;
- there are 5,500 scenarios;
- every public decision is full-menu JSON-valid;
- there are zero parser fallbacks;
- every baseline covers the same scenario set.

The paired interval resamples scenarios, never seeds or frames.

## 6. Failure handling

- Incomplete chunk (<55 scenarios): keep the artifacts, rerun mNNN, and record
  the retry.
- Invalid JSON or fallback in any row: stop the slice. Repair the parser or
  prompt, bump the prompt version, regenerate the freeze manifest, and treat all
  earlier chunks as a different experiment.
- Missing evaluator row: do not infer it from public rows; rerun the chunk.
- Baseline mismatch: regenerate the baseline export from the same manifest; do
  not drop scenarios to make coverage pass.
- Quota exhaustion: stop at a chunk boundary. Do not shrink max_new_tokens and
  do not run a partial merge as a final result.

## 7. What not to do

- Do not run the 5,500-episode grid before Gates A and B pass.
- Do not use a 550-episode session if the measured rate puts it near the Kaggle
  limit; use the 55-episode chunk kernel.
- Do not call the smoke or a dry chunk a benchmark result.
- Do not tune the prompt on the same scenarios used for the final merge.
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
