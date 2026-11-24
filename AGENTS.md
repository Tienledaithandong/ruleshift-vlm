<!-- omh:agent-instructions:begin -->

# Agent instructions — project_0 (RuleShift / Agentic VLM harness)

Repo: `C:\LENAMTIEN\research\Agentic_VLM\project_0`. Package: `ruleshift-vlm` (src layout), Python >=3.11,<3.14. CPU-first harness; no GPU needed for the core suite.

## 1. Build and test

Set up the dev environment (CPU, no GPU needed):

```bash
python -m pip install -e ".[dev]"        # or: uv run --with ".[dev]"
```

Commands and their verified status (observed 2026-11-22):

| Command | Status |
| --- | --- |
| `python -m pytest -q` | verified (exit 0) |
| `python scripts/preflight.py` | verified (exit 0; prints NO-GO — see Pitfalls) |
| `python scripts/preflight.py --strict` | verified (exit 2 on NO-GO) |
| `python scripts/check_reproducibility.py` | unverified as written (exit 1); see Pitfalls — prepend `PYTHONPATH=src` to run |
| `python scripts/phase_diagram.py --config configs/dev.yaml --out reports` | verified (exit 0) |

Run from the repo root with the package importable from `src` (pytest sets `pythonpath=["src"]` via `pyproject.toml`). `python -m pytest` is the regression gate; never edit `tests/` and leave it green.

## 2. Generated files

Never hand-edit generated files — regenerate from source:

| File | Source | Regenerate command | Gate |
| --- | --- | --- | --- |
| `reports/experiment_manifest.json`, `reports/w11_reproducibility.md` | `scripts/check_reproducibility.py` | `python scripts/check_reproducibility.py` (from repo root) | `determinism_check` returns PASS; file is rewritten, never hand-edited |
| `reports/phase_diagram.csv`, `reports/phase_diagram.md`, `reports/phase_diagram*.png` | `scripts/phase_diagram.py` | `python scripts/phase_diagram.py --config configs/dev.yaml --out reports` | files exist and are non-empty after a clean run |
| `reports/real_pilot.md`, `reports/real_smoke*.md` | `scripts/run_real_pilot.py`, `scripts/run_diagnostics.py` (real backend only) | see script `if __name__ == "__main__"` blocks | reviewed before committing (privileged labels are separate files) |

## 3. Gates

- **Byte-exact regression gate:** `python -m pytest -q` exits 0. Any source change must keep this green.
- **Reproducibility gate:** `check_reproducibility.py` `determinism_check` = PASS (paired intervals resample scenarios, never seeds or frames — see RUNBOOK §0).
- **Freeze gate:** `preflight.py --strict` exits 2 only when `NO-GO` (model revision not pinned, or `configs/preregistered_test.yaml` not `READY`).

## 4. Pitfalls

- **Stale ablation count.** `tests/test_ablations.py::test_all_named_ablations_return_valid_macros` previously asserted a fixed count that drifted from `ABLECTION_NAMES`: the committed menu had 5 entries and the test asserted 6 (stale). The working tree added two ablations (`M1_multi_step`, `M1_minus_selector`) and the working-tree test was corrected to assert `len(ABLECTION_NAMES)` — keep the test's count exactly equal to `len(ABLECTION_NAMES)` instead of hard-coding it. (Cost: two red CI runs before alignment.)
- **`check_reproducibility.py` needs `PYTHONPATH=src`.** The script imports `ruleshift` from the editable install's `site-packages`, where the `prompts/*.txt` data files are absent, so it fails with `FileNotFoundError` on `ruleshift/prompts/outcomes_v1.txt`. Prepend `PYTHONPATH=src` so it imports from `src` instead.
- **`torch`/`transformers` are not in the `dev` extra.** They live in the separate `vlm` extra. Preflight reports `NO-GO` until a frozen checkpoint + `configs/preregistered_test.yaml` `READY` status exists.

## 5. Conventions

- Python >=3.11,<3.14; core is standard-library only; `gymnasium`, `numpy`, `Pillow`, `PyYAML`, `matplotlib` are the `dev` extras.
- src layout; import as `from src.ruleshift import ...`; tests set `pythonpath=["src"]` and run `python -m pytest` from the repo root.
- Ablation names (the `ABLECTION_NAMES` tuple) are the public-only selector variants; every `ablation_choose(name, …)` call keeps all inputs public — no simulator, oracle, or hidden transitions inside the selector.
- Real-VLM pipelines use the `vlm` extra, a local frozen checkpoint (`configs/models.yaml`: `revision`, `local_files_only`, `device`), and the `READY` test config. Held-out data is never used to tune prompts.
- `data/`, `outputs/`, `results/`, `reports/` (except `experiment_manifest.json`), and `kaggle.json` are generated or secrets — never commit them (see `.gitignore`).

## 6. Grid multi-week runs

- The 100-map x 5-seed grid (5,500 episodes) is governed by `docs/superpowers/specs/2026-11-24-ruleshift-grid-multiweek-preregistration.md` and `docs/RUNBOOK_GRID_MULTIWEEK.md`. It is not authorized until that preregistration is marked `READY` by a human.
- A full-menu decision is required: all six candidate ids, probabilities sum to 1.0 within 0.02, and no parser fallback. A partial menu is a schema failure, not a valid belief.
- Never promote a smoke or a 55-episode dry chunk to a result. The merge only accepts complete 55-episode chunks, zero fallbacks, and 5,500 unique scenarios.
- Use `scripts/grid_freeze_manifest.py` before staging, `scripts/export_grid_baselines.py` for public CPU baselines, and `scripts/merge_grid_chunks.py` for scenario-level paired intervals.

<!-- omh:agent-instructions:end -->
