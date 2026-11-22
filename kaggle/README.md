# Kaggle credentials and GPU — setup only

Nothing here has been executed. No Kaggle API call, dataset download or kernel
run was made during setup.

## What I set up

- `kaggle/kaggle.json.example` — placeholder only. **Never commit a real key.**
- `scripts/kaggle_preflight.py` — local-only checks: package presence,
  credential file/env presence, CUDA availability, and whether the GitHub repo
  URL is still a placeholder. It makes **no network calls**.
- `notebooks/kaggle_gpu.ipynb` — notebook skeleton for a Kaggle GPU kernel.

## What you must do yourself

1. Create a Kaggle API token at `https://www.kaggle.com/account` → "Create New
   API Token". This downloads `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` (Windows:
   `C:\Users\<you>\.kaggle\kaggle.json`) and restrict permissions to your user.
   On Linux: `chmod 600 ~/.kaggle/kaggle.json`.
   Alternatively set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
3. Fill `configs/runtime.yaml`:
   - `repo_url`: your real GitHub HTTPS URL,
   - `checkpoint.source`: `hf` or `kaggle_dataset`,
   - if using a Kaggle dataset, set the real `dataset_slug`.
4. Run `python scripts/kaggle_preflight.py`. It must print GO before any run.
5. In the Kaggle notebook, enable **GPU T4 x2 or P100** in Settings → Accelerator.

## Guardrails

- Real `kaggle.json` is git-ignored.
- `scripts/run_real_pilot.py` still refuses to run until a checkpoint exists;
  the remote setup does not bypass that gate.
- `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN`.
