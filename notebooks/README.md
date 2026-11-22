# notebooks/

Two **setup-only** notebooks:

- `colab_setup.ipynb` — clone the GitHub repo in Google Colab, install the
  package, check GPU, run tests and preflight.
- `kaggle_gpu.ipynb` — the same for a Kaggle GPU kernel.

Neither notebook downloads a model, runs an experiment, or evaluates anything.
Both require you to set `RULESHIFT_REPO_URL` (your real GitHub HTTPS URL) first.
See `docs/COLAB_KAGGLE_SETUP.md` for the full instructions and guardrails.
