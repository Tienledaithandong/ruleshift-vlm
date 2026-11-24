# RuleShift / Agentic VLM — prepared research harness

**Status: CPU harness complete, including the additive RuleShift-Grid benchmark
(Tier A implemented; Tier B is a text-only VLM scaffold). No training,
experiment, or evaluation with a real model has been run. Generated result
files are intentionally absent so that stub-produced numbers cannot be
mistaken for measurements.**

CPU-first Python 3.11 research repository for value-aware probing under hidden
dynamics shifts. This is a 12-week research prototype, not a claim of a paper
or of a novel method. See `docs/RUNBOOK.md` for the later experiment sequence.

## Research question (frozen)

> When hidden environment dynamics change without warning, do frozen
> vision-language agents choose probes whose control value justifies their
> cost, and can structured counterfactual outcome reasoning—without access to
> the environment's generative model—reduce probe-decision regret and improve
> closed-loop return?

The agent sees public RGB, its own action/outcome history, a public macro-action
menu and textual descriptions of candidate mechanics. The evaluator alone holds
the hidden mechanic, shift time, transition kernel, belief and exact action
values. Probes cost real primitive steps in the same reward stream.

## What is implemented

| Area | Files | State |
|---|---|---|
| Contracts | `docs/problem_spec.md`, `docs/agent_evaluator_permissions.md`, `docs/benchmark_protocol.md` | specified |
| Static interfaces | `src/ruleshift/public_api/{schemas,interface}.py`, `src/ruleshift/permissions.py` | implemented and tested |
| Family A env | `src/ruleshift/envs/gate_lock.py` | implemented, Gymnasium checked |
| Family B env | `src/ruleshift/envs/bridge_hazard.py` | implemented; regime-dependent slip transitions |
| Evaluator | `src/ruleshift/evaluator/{bayes_filter,exact_solver,approx_reference,metrics_ci}.py` | filter, exact microtask solver, labelled approximate reference, paired CIs |
| Agent pipeline | `src/ruleshift/agents/{vlm_backend,cache,structured_vlm,consistency,value_selector,diva_style,self_consistency,baselines}.py` | implemented; public-only, no simulator imports |
| Baselines and ablations | `research/ablation_matrix.csv`, `src/ruleshift/agents/baselines.py`, `src/ruleshift/evaluator/baselines.py` | implemented; privileged rows labelled |
| Orchestration | `src/ruleshift/evaluator/{closed_loop,heldout,diagnostics,stress,family_b_reference}.py`, `scripts/` | implemented; backend/cache injectable for real runs |
| RuleShift-Grid | `src/ruleshift/{public_api/grid_*,envs/grid/,agents/grid_*,evaluator/grid_*}`, `scripts/run_grid_*.py`, `configs/grid_dev.yaml` | additive Tier A benchmark and text-only Tier B scaffold; implemented, CPU-only, no scientific run |
| Freeze scaffolding | `configs/preregistered_test.yaml` | `DRAFT_NOT_FROZEN` — intentionally unfrozen |
| Literature | `research/literature_matrix.csv`, `research/reading_notes.md` | canonical URLs verified; abstracts/READMEs only, full texts not read |
| Reproducibility | `requirements.lock`, `scripts/preflight.py`, `scripts/check_reproducibility.py`, `scripts/reproduce_all.py`, `docs/RUNBOOK.md` | implemented |

## Repository tree (clean setup)

```
project_0/
├── README.md                      docs/RUNBOOK.md
├── pyproject.toml                 requirements.lock
├── AGENT_TASK_WEEK1_RuleShift.md
├── RuleShift_VLM_Research_12_Week_Master_Plan.md
├── .github/workflows/tests.yml
├── configs/{dev.yaml,preregistered_test.yaml,models.yaml,grid_dev.yaml}
├── docs/{problem_spec,agent_evaluator_permissions,benchmark_protocol,week2_issues,RUNBOOK,REPO_REVIEW_AND_IMPROVEMENT_PLAN}.md
├── research/{literature_matrix.csv,reading_notes.md,ablation_matrix.csv}
├── scripts/*.py                   # setup, verification, grid audit/smoke/pilot, later-run entrypoints
├── reports/ results/ data/         # clean setup; generated files are git-ignored
├── src/ruleshift/
│   ├── public_api/                # agent-visible schemas, dataclasses and grid contracts/planner
│   ├── envs/                      # simulator; envs/grid/ is the additive RuleShift-Grid env
│   ├── evaluator/                 # privileged labels and references, including grid_* evaluators
│   ├── agents/                    # public-only policies, grid_* baselines/selector/VLM scaffold
│   └── prompts/                   # direct_v1.txt, outcomes_v1.txt, grid_v1.txt
└── tests/                         # 377 tests, all passing
```

## Verify the setup

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/preflight.py
python scripts/check_reproducibility.py
python scripts/run_grid_audit.py --maps 10 --out reports
python scripts/run_grid_smoke.py --quick --maps 5 --seeds 2 --out reports
```

The grid audit and smoke outputs are synthetic/pipeline checks; the smoke report
starts with `SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING`. Neither is a
measurement. Tier B's real pilot is remote-only:
`RULESHIFT_KAGGLE=1 python scripts/run_grid_real_pilot.py --device cuda --maps 3 --seeds 1 --out ...`
requires the Kaggle environment and is never run in local CI.

The full stub-labelled artifact set can be regenerated on demand with
`python scripts/reproduce_all.py`; those artifacts are git-ignored and are not
stored in the clean setup. CI regenerates them and then asserts the working
tree is still clean, so the ignore list cannot drift.

The only model-backend dependencies are in the optional extra:
`python -m pip install -e '.[vlm]'` installs `torch`/`transformers` for
`scripts/run_real_pilot.py`. The CPU harness and its tests never need them.

## Remote GPU setup (Colab / Kaggle)

Setup scaffolding exists but has **not been executed remotely**:

- `notebooks/colab_pilot.ipynb` — ready-to-run **real** offline pilot on Colab (pinned Qwen2.5-VL-3B, text-input only).
- `notebooks/colab_setup.ipynb`, `notebooks/kaggle_gpu.ipynb` — clone/install/check notebooks.
- `configs/runtime.yaml` — GitHub URL and checkpoint source (currently placeholders).
- `scripts/preflight.py`, `scripts/kaggle_preflight.py` — local-only GO/NO-GO gates.
- `scripts/run_grid_real_pilot.py`, `kaggle/kernel/run_grid.py` — remote-only grid pilot scaffold; the local guard refuses `cuda` unless `RULESHIFT_KAGGLE=1`.
- `kaggle/kaggle.json.example` — placeholder; real credentials are git-ignored.

See `docs/COLAB_KAGGLE_SETUP.md`. You must supply the GitHub URL, model revision
and credentials. No local GPU is used and nothing has been downloaded or run.

## Later: real experiment and evaluation

1. Obtain a frozen VLM checkpoint and pin its revision in `configs/models.yaml`.
2. `python scripts/preflight.py --strict` must print GO.
3. `python scripts/run_real_pilot.py` runs the offline pilot with the real
   backend and never falls back to the stub.
4. Use the injected-backend example in `docs/RUNBOOK.md` for closed-loop and
   held-out runs, compute paired scenario-level CIs, then freeze the test
   configuration with a commit hash.
5. Only then write up results. No empirical result exists yet.

## Non-goals

- No VLM fine-tuning, RL training, multi-agent debate or open-set rule discovery.
- No privileged information in agent inputs, prompts, caches or logs.
- No "Bayes-optimal" wording where a reference is approximate.
- No fabricated results, plots or method/novelty claims.
- No test-set freeze before a real validation run.
