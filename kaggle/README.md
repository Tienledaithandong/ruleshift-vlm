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
2. Authenticate the CLI. Any one of these forms works; `scripts/kaggle_preflight.py`
   recognises all four:

   | Form | Where |
   | --- | --- |
   | Username + key file | `~/.kaggle/kaggle.json` = `{"username":"…","key":"…"}` |
   | Username + key env | `KAGGLE_USERNAME` and `KAGGLE_KEY` |
   | API token env | `KAGGLE_API_TOKEN` (token from **Settings → API → Create New Token**) |
   | API token file | `~/.kaggle/access_token`, raw token as the only content |

   On Windows the directory is `C:\Users\<you>\.kaggle\`. Keep credential files
   private: `chmod 700 ~/.kaggle && chmod 600 ~/.kaggle/*`. Kaggle CLI 2.2.5+
   accepts `KAGGLE_API_TOKEN` and `~/.kaggle/access_token` directly, so a
   username/key pair is no longer required.
3. Fill `configs/runtime.yaml`:
   - `repo_url`: your real GitHub HTTPS URL,
   - `checkpoint.source`: `hf` or `kaggle_dataset`,
   - if using a Kaggle dataset, set the real `dataset_slug`.
4. Run `python scripts/kaggle_preflight.py`. It must print GO before any run.
5. In the Kaggle notebook, enable **GPU T4 x2 or P100** in Settings → Accelerator.

## Snapshot provenance

`scripts/stage_kaggle_dataset.py` refuses to stage a dirty working tree. It
stamps the commit it copied in two places:

- `dataset-metadata.json` → `description`, visible on the dataset page;
- `SOURCE_COMMIT` at the dataset root → `{commit, short_commit, subject, repo_url}`.

`kaggle/kernel/run.py` reads `SOURCE_COMMIT` right after copying the snapshot,
prints the commit and subject, and exits unless the marker holds a full 40-hex
commit id. It also fails on a mismatch when `RULESHIFT_EXPECTED_COMMIT` is set,
so a run can be pinned to one revision:

```bash
RULESHIFT_EXPECTED_COMMIT=$(git rev-parse HEAD) python scripts/stage_kaggle_dataset.py
```

Because Kaggle mounts the dataset's latest version, re-run the staging and
`datasets version` steps after every commit you want the kernel to see.

## Guardrails

- Real `kaggle.json` is git-ignored.
- `scripts/run_real_pilot.py` still refuses to run until a checkpoint exists;
  the remote setup does not bypass that gate.
- `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN`.
