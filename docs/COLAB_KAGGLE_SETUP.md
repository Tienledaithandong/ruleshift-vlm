# Remote GPU setup — Google Colab and Kaggle (setup only)

**Nothing in this setup has been executed remotely.** No model was downloaded,
no kernel was started, and no experiment was run. This document prepares the
repository so that a later run can happen on Colab or Kaggle instead of your
local machine.

## 0. What I need from you (placeholders)

Edit `configs/runtime.yaml`:

- `repo_url`: already set to `https://github.com/Tienledaithandong/ruleshift-vlm.git`.
- `checkpoint.source`: `hf` or `kaggle_dataset`.
- `checkpoint.dataset_slug`: required only if using a Kaggle dataset.

`configs/models.yaml` is now pinned for the pilot:
`Qwen/Qwen2.5-VL-3B-Instruct` at revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`, with `device: cuda`.

## 1. GitHub

The project folder was not a Git repository. I initialized one locally with no
remote and no commit, so you control the remote, credentials and first commit.

```bash
git remote add origin <YOUR_GITHUB_HTTPS_URL>
git add .
git commit -m "RuleShift prepared research harness"
git push -u origin main
```

Never commit `kaggle.json`, API tokens, credentials, or generated result files.
They are git-ignored.

## 2. Google Colab — offline pilot

Open `notebooks/colab_pilot.ipynb` for the ready-to-run pilot. It clones this
repo, installs the package, checks the GPU and runs
`scripts/run_real_pilot.py --model-key qwen2.5-vl-3b --contexts 81`.
The pilot is **text-input only** (P1 diagnostic mode): the backend sends the
public prompt but no RGB tensor yet. No training, closed loop or evaluation
happens in this step.

The general setup notebook (no run) is `notebooks/colab_setup.ipynb`, described
below.

1. Open `notebooks/colab_setup.ipynb` in Colab.
2. Runtime → Change runtime type → GPU (T4 is enough for a 3B model at 4-bit;
   for BF16 prefer L4/A100 if available).
3. Set `RULESHIFT_REPO_URL` in Colab secrets or a setup cell.
4. Run all cells. They clone the repo, install `.[dev]`, check `nvidia-smi`,
   run `pytest -q`, and run `scripts/preflight.py`.
5. `preflight.py` will still say NO-GO until a checkpoint exists and the test
   config is frozen. That is intentional.

## 3. Kaggle

1. Create an API token at `https://www.kaggle.com/account` → Create New API Token.
2. Place `kaggle.json` at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and
   `KAGGLE_KEY`). The file is git-ignored; `kaggle/kaggle.json.example` is a
   placeholder only.
3. Open `notebooks/kaggle_gpu.ipynb` as a new Kaggle notebook, enable
   Settings → Accelerator → GPU T4 x2 or P100, and set `RULESHIFT_REPO_URL`.
4. Run all cells. They clone, install, check GPU, run tests and call
   `scripts/kaggle_preflight.py --strict` (local checks only — no Kaggle API
   calls are made by that script).
5. To use a model hosted as a Kaggle dataset, set its slug in
   `configs/runtime.yaml` and download it yourself in the notebook when ready.

## 4. What the preflight scripts check

- `scripts/preflight.py` — model revision pinned, local checkpoint found,
  CUDA availability, and freeze state of `configs/preregistered_test.yaml`.
- `scripts/kaggle_preflight.py` — `kaggle` package installed, credentials
  present as a file or env vars, GPU visible, repo URL not a placeholder, and
  `execution.allow_remote_run` enabled. It makes **no network calls**.

Both print GO/NO-GO. `--strict` exits non-zero on NO-GO so it can gate a run.

## 5. Guardrails

- Local GPU is not used; no local training/evaluation happens in this setup.
- The stub remains the only backend that has actually run. `run_real_pilot.py`
  never falls back to the stub.
- `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN`.
- No result files are committed; `reports/`, `results/`, `data/` are empty.
