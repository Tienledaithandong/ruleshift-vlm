# RuleShift runbook

[Research progress and archived evidence](RESEARCH_STATUS.md) records the
historical runs. The current task is to establish interpretable Week 4
public-text decisions before a visual or closed-loop claim.

## CPU verification

From the repository root, install the CPU extras and run:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/phase_diagram.py
python scripts/run_offline_audit.py
python scripts/run_offline_audit.py --design history --out results/history --report results/history
python scripts/check_reproducibility.py
```

`python scripts/reproduce_all.py` rebuilds the CPU pipeline and runs the
tests. Its Week 7/9/10/11 reports exercise synthetic fixtures. The exact
Family A microtask and Family B single-decision calculations have their own
scope. To reproduce the analytical Family B table separately, run
`python scripts/family_b_transfer.py --out results/w8_transfer.md`.

## Offline pilot readiness

`configs/models.yaml` pins the Qwen anchor and defaults to local-only loading.
Preflight reads configuration and checkpoint inventory without loading
weights. Use the device intended for the run. For a CPU-only inspection:

```sh
python scripts/preflight.py --stage pilot --device cpu
python scripts/preflight.py --stage locked --device cpu
```

`--strict` returns exit code 2 on NO-GO. Pilot readiness requires the pinned
model, required packages, requested device and a usable local snapshot or an
explicit download choice. A development pilot can run while the test
configuration is a draft. The locked stage additionally checks the READY
status and frozen source revision. Model preflight alone cannot validate
the scientific freeze.

## Later real P1 pilot

On an approved inference runtime, install `.[dev,vlm]`, check readiness with
`--stage pilot --strict`, and run the following using a fresh run directory.
Both commands accept `--allow-download` for an explicitly chosen fetch.
The default model requests CUDA. Select `--device cpu` for CPU inference.

```sh
python scripts/run_real_pilot.py --model-key qwen2.5-vl-3b --contexts 81 --out results/real/pilot-next --report results/real/pilot-next/report.md
```

The default `--design controlled` measures first decisions. Use `--design history`
with its own output directory for decisions after noisy feedback and possible
rule shifts. Both designs use the action-only B0 prompt.

Inspect the real manifest, public decisions, private labels and response cache
separately. Check distinct public requests, output/action diversity, invalid
outputs, fallback regret, exact-label coverage, tokens and actual generation
latency. Compare results only when the input protocol, model, prompt and
reference versions match. Historical v12 used a different protocol.

## Validation and locked evaluation

Use `ruleshift.evaluator.closed_loop.run_closed_loop` with explicit validation
scenarios, paired seeds, a pinned real backend and a shared response cache.
Use the same public observation mode, physical step limit, per-episode call
cap and decoding settings for M1, direct and adapted DIVA. Cached responses
still count against the logical budget. Publish real call and token costs.

Average paired seeds within each scenario and resample scenarios for the 95%
interval. Record a practically meaningful effect and sample size before
looking at held-out data. Complete `reports/w6_validation.md` before freezing
the test configuration. The `build_heldout_family_a()` grid is a synthetic
fixture. Locked inference requires the actual frozen scenarios, seeds, methods
and budgets. Record `implementation_sha256` under `locked_run` using
`ruleshift.evaluator.provenance.implementation_digest()` on the reviewed
implementation. The evaluator verifies that digest before issuing a model
call. P0 and empirical transfer each require their own verification.

For hosted runtime preparation, see [Colab and Kaggle](COLAB_KAGGLE_SETUP.md).
