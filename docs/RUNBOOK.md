# Runbook — setup now, experiment later

This repository is a **prepared research harness**. The setup work is done:
contracts, environment, evaluator references, baselines, ablations, closed-loop
runner, held-out evaluation, diagnostics, stress tests and CI all exist and are
covered by tests. **No training, experiment or evaluation with a real model has
been run**, and generated result files are intentionally absent so that
stub-produced numbers cannot be mistaken for measurements.

## 1. What is ready

- CPU environment: Family A `RuleShiftEnv` and Family B `BridgeHazardEnv`
  (Gymnasium API, semantic RNG, public/privileged split, shared macro contract).
- Evaluator side: shift-aware Bayes filter, exact finite-horizon microtask
  solver, labelled truncated-rollout reference, paired bootstrap CIs, surgical
  diagnostics and statistical stress sweeps.
- Agent side: public-only prompts, response cache keyed by public history,
  structured JSON parsing, value-aware selector with one-step continuation,
  DIVA-style adapted baseline, self-consistency and all five M1 ablations.
- Orchestration: offline pilot, closed-loop, held-out and diagnostic runners.
- All of the above run with a deterministic stub in tests. A real backend can
  now be injected into the closed-loop and held-out runners.

## 2. Verify the setup

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/preflight.py
python scripts/check_reproducibility.py
```

`pytest` must be green before any experiment. `check_reproducibility.py`
writes `reports/experiment_manifest.json` with pinned versions, seeds, source
hashes and a determinism check.

## 3. Preparing the real model (later)

1. Download a frozen VLM checkpoint yourself; nothing in this repo downloads
   weights implicitly. Candidates are listed in `configs/models.yaml`.
2. Edit `configs/models.yaml`: set `revision` to a pinned commit hash, choose
   `device` (`cuda` only if a GPU actually exists), and keep
   `local_files_only: true` unless you explicitly want a download.
3. Run `python scripts/preflight.py --strict`. It reports NO-GO until a local
   checkpoint exists, the revision is pinned, and
   `configs/preregistered_test.yaml` is set to `READY` with a frozen commit.

## 4. Real offline pilot (later)

```bash
python scripts/run_real_pilot.py --model-key qwen2.5-vl-3b \
    --out results/real --report reports/real_pilot.md
```

This uses `TransformersVLMBackend` and **never falls back to the stub**. It
writes public decision rows and privileged labels to separate files. Review
the failure taxonomy before changing any prompt.

## 5. Validation and freeze (later)

1. Tune prompts only on `validation`; never on held-out data.
2. Run closed-loop comparisons with the real backend (see snippet below) and
   compute paired 95% CIs resampled at the scenario level.
3. Only if the preregistered gate passes, set `configs/preregistered_test.yaml`
   to `READY`, list the frozen scenarios, and record the Git commit hash.
4. Do not freeze a test set before this step. The current `DRAFT_NOT_FROZEN`
   status is deliberate.

## 6. Closed loop and held-out with a real backend (later)

The runners accept an optional backend and shared cache; method labels drop
the `_stub` suffix automatically when a real backend is supplied:

```python
from pathlib import Path
from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import TransformersVLMBackend
from ruleshift.evaluator.closed_loop import build_family_a_scenarios, run_closed_loop
from ruleshift.evaluator.heldout import build_heldout_family_a, run_heldout_evaluation

backend = TransformersVLMBackend("Qwen/Qwen2.5-VL-3B-Instruct", device="cuda")
cache = ResponseCache(Path("results/real/cache.jsonl"))

dev = run_closed_loop(
    build_family_a_scenarios(),
    ("direct", "diva", "m1", "never_probe"),
    seeds=(1, 2),
    comparisons=(("m1", "direct"), ("m1", "diva")),
    backend=backend,
    cache=cache,
)
held = run_heldout_evaluation(
    build_heldout_family_a(),
    ("direct", "diva", "m1", "immediate"),
    seeds=(101, 102),
    backend=backend,
    cache=cache,
)
```

Method ids are `direct`, `diva`, `m1`, `immediate` with a real backend; with
the default stub they are `direct_stub`, `diva_stub`, etc.

## 7. Diagnostics, stress and write-up (later)

```bash
python scripts/run_diagnostics.py
python scripts/run_stress.py
python scripts/reproduce_all.py
```

`reproduce_all.py` regenerates the stub-labelled artifacts and reruns the
suite. For a real write-up, replace those with real-backend runs, fill the
`PENDING` fields in `reports/w6_validation.md`, and only then draft a paper.

## 8. Guardrails

- No agent module imports `envs` or `evaluator`; CI checks this statically.
- No simulator likelihood, oracle belief, Q* or true rule enters prompts,
  caches or public logs.
- Probes cost primitive steps in the same reward stream; all methods share the
  same menu and step budget.
- Never report stub-generated numbers as empirical results, and never claim
  novelty or publishability before a real held-out result survives matched
  budgets and paired intervals.
