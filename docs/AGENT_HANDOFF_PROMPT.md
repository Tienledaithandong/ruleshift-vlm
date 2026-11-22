# Handoff prompt for the next agent

Copy the prompt below verbatim into a fresh agent session. It is self-contained.

---

> You are taking over a research-engineering repo. Read this whole prompt before changing anything. Do not run training, do not fine-tune, and do not freeze the test set.
>
> ## 1. Where things are
>
> - Private GitHub repo: `https://github.com/Tienledaithandong/ruleshift-vlm` (branch `main`, HEAD `35c123e` at handoff). Clone/read it first.
> - Key docs: `README.md`, `docs/RUNBOOK.md`, `docs/problem_spec.md`, `docs/benchmark_protocol.md`, `docs/COLAB_KAGGLE_SETUP.md`, `docs/KAGGLE_HANDOFF_PROMPT.md`.
> - Research question: do frozen VLMs choose probes whose control value justifies their physical cost under hidden, unannounced dynamics shifts, and can counterfactual outcome reasoning reduce probe-decision regret?
> - Architecture: `src/ruleshift/public_api/` = agent-visible contracts; `src/ruleshift/envs/` = simulator (agents must never import); `src/ruleshift/evaluator/` = privileged labels/references (agents must never import); `src/ruleshift/agents/` = public-only policies, VLM interface, cache, selector.
>
> ## 2. State of the work
>
> - Setup is complete. Week 1-11 code exists: two Gymnasium environments (Family A Gate/Lock with regime-dependent lockout; Family B Bridge/Hazard with regime-dependent slip probabilities), semantic RNG, macro executor, exact Bayes microtask solver, labelled approximate reference, baselines/ablations, closed-loop runner, held-out evaluator, diagnostics, stress tests, paired bootstrap CIs, CI workflow.
> - Local verification: `python -m pip install -e '.[dev]'` then `python -m pytest -q` → **193 passed**. All scripts pass `--help`.
> - Local machine: Windows, Python 3.13.15, no local GPU used for the final run (an RTX 4060 laptop GPU exists and CUDA torch `2.11.0+cu128` + torchvision `0.26.0+cu128` are installed). Qwen2.5-VL-3B weights are cached in `~/.cache/huggingface/hub`.
> - Real Kaggle GPU run **completed successfully**:
>   - Kernel: `https://www.kaggle.com/code/tiennamlee/ruleshift-offline-pilot`
>   - Private source dataset: `https://www.kaggle.com/datasets/tiennamlee/ruleshift-source`
>   - Model `Qwen/Qwen2.5-VL-3B-Instruct`, pinned revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, device `cuda`, GPU T4.
>   - Artifacts downloaded locally to `kaggle_outputs/` (git-ignored): `reports/real_pilot.md`, `results/real/{decisions,evaluator_only,response_cache}.jsonl`.
>   - Measurements: 81 contexts, 27 model calls, JSON valid 1.000, valid actions 1.000, over-probe 0.630, under-probe 0.000, mean decision regret 6.722, mean output tokens 539, cache hit 0.667, ~10.7 s/call, GPU quota used 0.16 h.
>   - `backend_id` is `TRANSFORMERS_FROZEN`, confirming a real model ran, not the stub.
>
> ## 3. The critical unresolved problem
>
> The model copied the prompt's JSON example **verbatim** for every distinct input (`0.34/0.33/0.33`, `0.5/0.5`, `PROBE_RED`, `COMMIT_RED`). Output is valid JSON but not genuine reasoning. Therefore the 0.630 over-probe rate is an artifact of an over-prescriptive example, NOT an empirical finding. Do not report it as evidence for H1/H2.
>
> The parser already strips ```json fences (`src/ruleshift/agents/vlm_backend.py:parse_structured_output`), and the backend uses the chat template, so those earlier failures are fixed. The remaining problem is prompt design.
>
> ## 4. Your next tasks, in order
>
> 1. **Fix the prompt example.** Edit `src/ruleshift/prompts/direct_v1.txt` and `outcomes_v1.txt`. Describe the required JSON shape without supplying concrete, copyable values: use neutral placeholders or require the model to derive values from the public text. Add an explicit instruction that repeated identical-looking examples are unacceptable and that probabilities must reflect the described uncertainty. Add a unit test asserting the templates do not contain the literal strings `0.34`, `0.33`, `PROBE_RED`, `COMMIT_RED`. Run `pytest -q`.
> 2. **Rebuild the Kaggle source dataset.** Repackage the repo with the `kaggle/kaggle_dataset_build` helper or the same staging approach (`kaggle/kernel/run.py` copies from `/kaggle/input/datasets/tiennamlee/ruleshift-source`), create a new dataset version with `python -m kaggle datasets version -p <stage> -r zip -m "fix prompt example"`. Confirm `python -m kaggle datasets status tiennamlee/ruleshift-source` says `ready`.
> 3. **Re-run the pilot on Kaggle.** `python -m kaggle kernels push -p kaggle/kernel`, wait for `COMPLETE` via `python -m kaggle kernels status tiennamlee/ruleshift-offline-pilot`, then `python -m kaggle kernels output ... -p kaggle_outputs/`. Verify: (a) outputs differ across public inputs, (b) no response equals the template example, (c) action distribution is not a single constant, (d) JSON validity remains high.
> 4. **Report honestly.** Record JSON validity, valid-action rate, over/under-probe, regret, tokens, latency, and any remaining pathologies. If outputs are still degenerate, diagnose and iterate, reporting every attempt. Never substitute the stub backend.
> 5. **Only after a clean pilot:** add an RGB image path (`ModelRequest.image` -> backend `processor(images=...)`) so the primary P0 mode can run instead of text-only P1. Then consider validation/closed-loop with paired scenario-level CIs.
>
> ## 5. Guardrails (non-negotiable)
>
> - No agent module imports `ruleshift.envs` or `ruleshift.evaluator`; CI checks this statically.
> - No simulator likelihood, oracle belief, Q*, true rule, shift time, or hidden RNG in prompts, caches, or public logs.
> - Every primitive step cost is charged once in the same reward stream; all methods share the same macro menu and step budget.
> - `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN` until a real validation run passes the preregistered gate.
> - No fabricated results, no novelty or publishability claims, no VLM training.
>
> ## 6. Useful commands
>
> ```
> python -m pip install -e '.[dev]'
> python -m pytest -q
> python -m kaggle datasets status tiennamlee/ruleshift-source
> python -m kaggle kernels push -p kaggle/kernel
> python -m kaggle kernels status tiennamlee/ruleshift-offline-pilot
> python -m kaggle kernels logs tiennamlee/ruleshift-offline-pilot
> python -m kaggle kernels output tiennamlee/ruleshift-offline-pilot -p kaggle_outputs
> python -m kaggle quota
> ```
>
> Commit and push your changes to `main`. Keep generated outputs out of Git.
>

---

## Notes for the human owner

- This handoff supersedes `docs/KAGGLE_HANDOFF_PROMPT.md`, which was written before the Kaggle run succeeded.
- The earlier failed Kaggle kernel versions (1-6) are not a concern; version 7 completed. Failures were: wrong dataset path, missing tests in dataset, wrong config file patched, and download disabled.
- The local RTX 4060 can run the model too, but the Kaggle T4 path is already working and preserves your local machine.
