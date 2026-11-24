# RuleShift-Grid Multi-Week Real-VLM Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it possible to execute the 100 maps × 5 seeds real frozen-VLM grid in deterministic, weekly Kaggle chunks, then merge chunks into one report with scenario-level paired confidence intervals.

**Architecture:** The existing evaluator already generates a deterministic scenario grid and computes metrics/CIs. We add scenario slicing to the Kaggle pilot, chunk-stamped raw artifacts, a CPU baseline exporter, and a merge/CI script. Each Kaggle run remains an ordinary kernel over a versioned source dataset; nothing is persisted only in chat or in a notebook session.

**Tech Stack:** Python >=3.11,<3.14, pytest, PyYAML, Kaggle CLI (`python -m kaggle`), Kaggle private dataset `tiennamlee/ruleshift-source`, CodeGraph v1.6.4 (`codegraph init`, `codegraph explore`).

**Spec:** `docs/superpowers/specs/2026-11-23-ruleshift-grid-design.md` and the Decision Record below.

---

## Decision Record (why this plan exists)

- Full real-VLM grid size: 100 maps × 5 seeds × 11 scenarios = **5,500 episodes**.
- Measured Kaggle T4 cost: ~2.6 s/decision × ~55 decisions/episode ≈ **143 s/episode**, so the full grid is ~**218 GPU-hours**.
- Kaggle quota at decision time: **27.22 GPU-hours remaining** of 30 h/week, refreshing weekly.
- Laptop GPU: RTX 4060 Laptop, 8 GB VRAM. The pinned `bfloat16` 3B model (~6 GB weights plus image tower/KV cache) is too large for reliable local execution; no local GPU job is permitted for this plan.
- Chosen chunk: **10 maps × 5 seeds = 550 episodes ≈ 22 GPU-hours**, ten chunks over ten weeks. Use a shared source dataset snapshot and per-chunk result datasets.

This is a **real-VLM** plan. The CPU `run_grid_smoke.py` grid is synthetic plumbing validation and must never be reported as a finding.

## Global Constraints

- Additive to RuleShift-Grid; do not change Family A/B, `MACRO_ACTIONS`, or existing validators beyond the documented VLM-output tolerance already committed.
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, `dtype: bfloat16`, image + text in one message, `max_new_tokens: 2048`.
- Config: `configs/grid_dev.yaml`; `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN` until a human explicitly approves the freeze in Task 1.
- No local GPU job. Kaggle only for CUDA. `python -m pytest -q` must stay green after every task.
- Scenario identity is `scenario_id = f"m{map_seed}-s{run_seed}-{scenario_type}-{initial_rule}-{shift_rule}"`. It is the unit of pairing and bootstrap resampling; never resample seeds or frames.
- Raw artifacts are privileged/public separated: `*_decisions.jsonl` (public), `*_evaluator_only.jsonl` (privileged), separate files, never joined into model input.
- `reports/`, `results/`, `data/`, `kaggle_stage/`, `kaggle.json`, `.codegraph/` are generated/ignored; never commit them.
- CodeGraph v1.6.4 is installed; every agent runs `codegraph init --yes` in its own worktree and uses `codegraph explore` before grep/read for symbol questions.

## Review Focus

- **Scenario overlap or omission:** a merged report must contain each of the 5,500 `scenario_id`s exactly once; Task 5 tests this.
- **Silent fallback actions:** a chunk report with JSON-valid < 1.0 must fail the merge gate; Task 5 tests this.
- **Wrong model/config revision:** freeze manifest must record commit sha, model revision, config hashes and scenario-list hash; Task 1 tests determinism.
- **Privileged leakage into public artifacts:** chunk artifacts must keep evaluator rows separate; Task 3 tests this.
- **Bootstrap resampling the wrong unit:** paired CIs must resample scenarios, never seeds/frames; Task 5 tests this.

---

### Task 1: Freeze manifest and preregistration note

**Files:**
- Create: `docs/superpowers/specs/2026-11-24-ruleshift-grid-multiweek-preregistration.md`
- Create: `scripts/grid_freeze_manifest.py`
- Test: `tests/test_grid_freeze_manifest.py`

**Interfaces:**
- Consumes: `configs/grid_dev.yaml`, `configs/models.yaml`, `build_grid_scenarios(100, 5, config)`.
- Produces: `grid_freeze_manifest(paths: Sequence[str]) -> dict[str, Any]`; CLI writes JSON to stdout or `--out PATH`. Keys: `commit`, `model_revision`, `config_sha256`, `scenario_count`, `scenario_list_sha256`.

- [ ] **Step 1: Write failing test for manifest determinism and exact keys**

```python
def test_freeze_manifest_is_deterministic():
    first = grid_freeze_manifest(("configs/grid_dev.yaml", "configs/models.yaml"))
    second = grid_freeze_manifest(("configs/grid_dev.yaml", "configs/models.yaml"))
    assert first == second
    assert set(first) == {"commit", "model_revision", "config_sha256", "scenario_count", "scenario_list_sha256"}
    assert first["scenario_count"] == 5500
```

- [ ] **Step 2: Run the test and watch it fail** (`pytest tests/test_grid_freeze_manifest.py -q`, expect import error).

- [ ] **Step 3: Implement `grid_freeze_manifest`**

Use `subprocess.check_output(["git", "rev-parse", "HEAD"])`, PyYAML for the two config files, and `hashlib.sha256` over canonical JSON. The scenario list hash uses `[s.scenario_id for s in build_grid_scenarios(100, 5, config)]` sorted, joined with newlines.

- [ ] **Step 4: Run the test until it passes** (`pytest tests/test_grid_freeze_manifest.py -q`).

- [ ] **Step 5: Write the preregistration document**

Record: hypothesis, primary metrics (success rate, adaptation steps, total cost, false-alarm rate), secondary metrics, scenario-level paired CI method, chunk scheme 10 maps × 5 seeds, ten weekly runs, exclusion rules, and the explicit statement that no result is claimed before all ten chunks are merged. State that flipping `configs/preregistered_test.yaml` to `READY` requires a human decision and is **not** done by this task.

- [ ] **Step 6: Commit**

```bash
git add scripts/grid_freeze_manifest.py tests/test_grid_freeze_manifest.py docs/superpowers/specs/2026-11-24-ruleshift-grid-multiweek-preregistration.md
git commit -m "feat: add grid multiweek freeze manifest and preregistration"
```

---

### Task 2: Slice the real-VLM pilot by map range

**Files:**
- Modify: `scripts/run_grid_real_pilot.py:40-53, 213-215`
- Test: `tests/test_grid_scripts.py`

**Interfaces:**
- Consumes: `build_grid_scenarios(n_maps, seeds, config) -> tuple[GridScenario, ...]`; `GridScenario.map_seed: int`.
- Produces: `parse_args(argv)` accepts `--map-offset` (int, default 0), `--map-count` (int, default 0 = all maps), and `--chunk-id` (str, default `"default"`). `main` selects scenarios whose `map_seed` is in `[map_offset, map_offset + map_count)` before applying `max_episodes`.

- [ ] **Step 1: Write failing tests**

```python
def test_parse_args_accepts_multiweek_slicing():
    args = parse_args(["--maps", "100", "--seeds", "5", "--map-offset", "20", "--map-count", "10", "--chunk-id", "c02"])
    assert (args.map_offset, args.map_count, args.chunk_id) == (20, 10, "c02")

def test_selected_scenarios_are_the_requested_map_range():
    config = load_grid_config("configs/grid_dev.yaml")
    all_scenarios = build_grid_scenarios(30, 5, config)
    selected = select_chunk_scenarios(all_scenarios, 20, 10)
    assert {s.map_seed for s in selected} == set(range(20, 30))
    assert len(selected) == 550
```

- [ ] **Step 2: Run the tests and watch them fail** (`pytest tests/test_grid_scripts.py -q`).

- [ ] **Step 3: Implement**

Define `select_chunk_scenarios(scenarios, map_offset, map_count) -> tuple[GridScenario, ...]` in `scripts/run_grid_real_pilot.py`. In `main`, call it before the existing `[: args.max_episodes]` slice. Keep `--max-episodes` as an optional cap; default it to `None` (no cap) so a chunk runs all 550 episodes unless explicitly capped.

- [ ] **Step 4: Run the full suite** (`python -m pytest -q`, expect 0 failures).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_grid_real_pilot.py tests/test_grid_scripts.py
git commit -m "feat: slice grid real pilot by map range"
```

---

### Task 3: Chunk-stamped artifacts and Kaggle chunk kernel

**Files:**
- Modify: `scripts/run_grid_real_pilot.py`
- Create: `kaggle/kernel/run_grid_chunk.py`
- Test: `tests/test_grid_scripts.py`

**Interfaces:**
- Consumes: `args.chunk_id`, `args.out`, `args.report` from Task 2.
- Produces: filenames `grid_real_decisions.{chunk_id}.jsonl`, `grid_real_evaluator_only.{chunk_id}.jsonl`, `response_cache.{chunk_id}.jsonl`, `grid_real_pilot.{chunk_id}.md`, all under `args.out`/`args.report` directories. The report includes `Chunk: {chunk_id}` and `Map range: [{offset}, {offset+count})`.

- [ ] **Step 1: Write failing test for artifact naming**

```python
def test_chunk_artifact_names_include_chunk_id(tmp_path):
    paths = chunk_artifact_paths(tmp_path, "c02")
    assert paths["decisions"].name == "grid_real_decisions.c02.jsonl"
    assert paths["evaluator"].name == "grid_real_evaluator_only.c02.jsonl"
    assert paths["cache"].name == "response_cache.c02.jsonl"
```

- [ ] **Step 2: Run the test and watch it fail.**

- [ ] **Step 3: Implement `chunk_artifact_paths(out_dir: Path, chunk_id: str) -> dict[str, Path]`**

Use it in `main` instead of hardcoded names. Keep privilege separation: evaluator rows only in the evaluator file.

- [ ] **Step 4: Create `kaggle/kernel/run_grid_chunk.py`**

It reads env vars `GRID_CHUNK_ID`, `GRID_MAP_OFFSET`, `GRID_MAP_COUNT`, `RULESHIFT_EXPECTED_COMMIT`; copies `/kaggle/input/.../ruleshift-source` to `/kaggle/working/ruleshift-vlm`; installs `.[dev]`; invokes the pilot with `--maps 100 --seeds 5 --map-offset ... --map-count ... --chunk-id ...`. It prints `HF_TOKEN present: <bool>` only, never the token.

- [ ] **Step 5: Run the full suite** (`python -m pytest -q`).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_grid_real_pilot.py kaggle/kernel/run_grid_chunk.py tests/test_grid_scripts.py
git commit -m "feat: stamp RuleShift-Grid chunk artifacts"
```

---

### Task 4: Export CPU baselines for the full grid

**Files:**
- Create: `scripts/export_grid_baselines.py`
- Test: `tests/test_grid_scripts.py`

**Interfaces:**
- Consumes: `METHODS` from `scripts/run_grid_smoke.py`, `build_grid_scenarios(100, 5, config)`, `run_grid_benchmark`.
- Produces: `export_grid_baselines(config, methods, out_path) -> None`, writing one JSON object per episode with keys `scenario_id`, `method`, `success`, `env_steps`, `total_cost`, `adaptation_steps`, `false_alarm`, `scenario_type`, `oracle_return`.

- [ ] **Step 1: Write failing test for exported schema**

```python
def test_exported_baseline_rows_have_required_keys(tmp_path):
    out = tmp_path / "baselines.jsonl"
    export_grid_baselines(load_grid_config("configs/grid_dev.yaml"), ("NoAdaptationPolicy",), out)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert {"scenario_id", "method", "success", "total_cost", "scenario_type"} <= set(row)
```

- [ ] **Step 2: Run the test and watch it fail.**

- [ ] **Step 3: Implement the exporter**

Reuse the policy factories from `scripts/run_grid_smoke.py`; never modify their behaviour. Run on CPU only.

- [ ] **Step 4: Run the full suite and commit.**

```bash
git add scripts/export_grid_baselines.py tests/test_grid_scripts.py
git commit -m "feat: export CPU baseline episodes for the full grid"
```

---

### Task 5: Merge chunks and compute paired CIs

**Files:**
- Create: `scripts/merge_grid_chunks.py`
- Test: `tests/test_grid_scripts.py`

**Interfaces:**
- Consumes: chunk decision files from Task 3, baseline file from Task 4, `summarize_grid`, `paired_grid_ci`.
- Produces: `merge_grid_chunks(chunk_dir: Path, baselines: Path, out_dir: Path) -> dict[str, Any]`; writes `grid_multiweek_results.md` and `grid_multiweek_results.json`.

- [ ] **Step 1: Write failing tests**

```python
def test_merge_rejects_duplicate_scenario_ids(tmp_path):
    write_two_chunks_with_overlap(tmp_path)
    try:
        merge_grid_chunks(tmp_path, baselines_path, tmp_path / "out")
    except ValueError as exc:
        assert "duplicate scenario_id" in str(exc)
    else:
        raise AssertionError("overlap must be rejected")

def test_merge_reports_scenario_level_paired_ci(tmp_path):
    result = merge_grid_chunks(tmp_path, baselines_path, tmp_path / "out")
    assert result["n_scenarios"] == 5500
    assert "paired_95_ci" in result
```

- [ ] **Step 2: Run the tests and watch them fail.**

- [ ] **Step 3: Implement the merge**

Collect all decision rows, assert exactly 5,500 unique `scenario_id`s and complete coverage of `build_grid_scenarios(100, 5)`; reject any chunk whose `json_valid_rate` < 1.0; reconstruct `GridEpisode` rows; call `summarize_grid` for primary/secondary metrics and `paired_grid_ci(method_rows, baseline_rows, n_boot=10000)` for scenario-level CIs.

- [ ] **Step 4: Run the full suite and commit.**

```bash
git add scripts/merge_grid_chunks.py tests/test_grid_scripts.py
git commit -m "feat: merge grid chunks and compute paired CIs"
```

---

### Task 6: Weekly runbook and agent handoff

**Files:**
- Create: `docs/RUNBOOK_GRID_MULTIWEEK.md`
- Modify: `AGENTS.md` (add pointer outside the `omh:agent-instructions` marker)

**Interfaces:**
- Consumes: all of the above.
- Produces: a self-contained runbook with weekly commands, artifact naming, agent roles, and gates.

- [ ] **Step 1: Write the runbook**

Include:
  1. One-time setup: human approves freeze, run `grid_freeze_manifest.py`, record sha, `codegraph init --yes` in every agent worktree.
  2. Weekly loop for chunk `c00…c09`: stage source dataset, verify `SOURCE_COMMIT`, set `GRID_CHUNK_ID`, `GRID_MAP_OFFSET`, `GRID_MAP_COUNT`, push `kaggle/kernel/run_grid_chunk.py`, wait for `COMPLETE`, download chunk artifacts to a shared dataset/local directory, record checksums.
  3. Agent roles: **Tooling agent** (Tasks 1–5), **Run agent** (weekly Kaggle chunks), **Merge agent** (after c09), **Review agent** (checks freeze/coverage/CI before any claim).
  4. Gates: no result claimed before all ten chunks; no local GPU; no synthetic smoke in findings; preregistration remains DRAFT until human freeze.
  5. Recovery: re-run a chunk only with same manifest sha; never mix manifests in one merge.

- [ ] **Step 2: Add AGENTS.md pointer**

Add a short section: "Multi-week real-VLM grid: see `docs/RUNBOOK_GRID_MULTIWEEK.md` and `docs/superpowers/plans/2026-11-24-ruleshift-grid-multiweek-run.md`."

- [ ] **Step 3: Commit**

```bash
git add docs/RUNBOOK_GRID_MULTIWEEK.md AGENTS.md
git commit -m "docs: add multiweek grid runbook and agent handoff"
```

---

## Final Verification

- [ ] `python -m pytest -q` exits 0 after each task.
- [ ] `python scripts/grid_freeze_manifest.py` produces identical JSON on two consecutive runs before week 1.
- [ ] A dry chunk of 1 map × 5 seeds produces exactly 55 scenarios and chunk-stamped files.
- [ ] `merge_grid_chunks` rejects duplicate `scenario_id` and incomplete coverage.
- [ ] After c09, merged report shows 5,500 scenarios and paired CIs resampled by scenario.
- [ ] No local GPU job was run at any point.
