# Handoff prompt for another agent with Kaggle/GPU access

> **Archived handoff.** These instructions describe an earlier attempt. The current task order and pilot evidence are in [research progress](RESEARCH_STATUS.md) and the [runbook](RUNBOOK.md). Remote execution is disabled in the current configuration.

Give this prompt to the other agent. It is self-contained.

---

> We have a prepared research repo at `https://github.com/Tienledaithandong/ruleshift-vlm` (**private**). The setup is complete; no experiment has been run yet. Your job is to run the real offline VLM pilot on a Kaggle GPU and return the artifacts. Do not train anything, do not fine-tune, do not run the full closed-loop evaluation, and do not freeze the test set.
>
> 1. Read `docs/RUNBOOK.md`, `docs/COLAB_KAGGLE_SETUP.md` and `configs/models.yaml` in the repo.
> 2. The model is `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, `max_new_tokens=256`.
> 3. Because the repo is private, either (a) add a Kaggle Secret `RULESHIFT_GITHUB_TOKEN` containing a read-only GitHub token, or (b) ask me to make the repo public first. Never print the token.
> 4. Use `kaggle/kernel/kernel-metadata.json` and `kaggle/kernel/run.py` as the kernel, but replace `<KAGGLE_USERNAME>` with the actual Kaggle username. Set GPU accelerator (T4 x2 or P100) and internet enabled.
> 5. Push and run the kernel with the Kaggle CLI from a directory containing those two files:
>    `kaggle kernels push -p <dir-with-kernel-metadata-and-run.py>`
>    Then monitor with `kaggle kernels status <username>/ruleshift-pilot`.
> 6. When it finishes, download outputs:
>    `kaggle kernels output <username>/ruleshift-pilot -p <local-output-dir>`
> 7. Verify the run was real: `reports/real_pilot.md` must show `Device: cuda`, model calls > 0, and the response cache must contain genuine model text (not the stub). Report the exact JSON-valid rate, valid-action rate, over/under-probe rates, mean decision regret, tokens and latency.
> 8. If the model output is not parseable JSON, fix the prompt/parser in the repo and re-run, but report every attempt. Do not substitute the stub backend.
> 9. Return: (a) the report text, (b) the raw decisions/manifest paths, (c) any failures and fixes, (d) the Kaggle kernel URL. Do not claim any hypothesis is supported; validation and test freezing come later.
>
> Guardrails: no agent module may import `envs` or `evaluator`; no simulator likelihood, oracle belief or Q-value may enter prompts/caches; costs must be charged once. `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN`.

---

## Notes on why I am handing this off

- The browse extension is offline, so I cannot drive a Kaggle notebook through the browser.
- The repository is private, so a Kaggle kernel cannot clone it without a token that I do not have.
- Pushing a kernel consumes Kaggle GPU quota and creates a remote artifact; that is a real-world side effect I will not trigger blindly.

Everything else is prepared locally: pinned model revision, real VLM backend, parser, pilot script and cache writer.
