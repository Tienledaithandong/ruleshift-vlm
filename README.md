# RuleShift / Agentic VLM

**Research checkpoint (2026-11-24) — Week 4, revised pilot verified on CPU.**
The frozen Qwen anchor has run on Kaggle in public-text mode (P1). The v12
archive contains 81 decisions from 27 distinct requests. All 81 chose
`PROBE_RED`. Several noise conditions received the same input despite
different exact labels. The pilot demonstrates inference and parsing.
A clean Week 4 diagnostic and real Week 6 validation are still pending.
**Status: CPU harness complete, including the additive RuleShift-Grid benchmark
(Tier A implemented; Tier B is an RGB+text VLM scaffold). The frozen 3B grid
is closed after diagnostic smokes and one map-0 dry chunk.** See the
[grid feasibility note](docs/GRID_NEGATIVE_FEASIBILITY_NOTE.md).

The revised P1 harness covers controlled first decisions and decisions
following noisy public histories under a possible shift. Each default design
contains 81 distinct inputs with exact microtask labels.

[Progress and evidence](docs/RESEARCH_STATUS.md) · [CPU and experiment runbook](docs/RUNBOOK.md)

## P0 measurement repair (2026-11-25)

The active identification claim now has an independent metric protocol, explicit
manifests, shift-timing exports and a corrected pairing path. The legacy audit
found that the old gate used task success and a terminal-label shift proxy; it
is preserved under its original labels in
[Metric audit](docs/METRIC_AUDIT.md). Map exposure and candidate-layout checks
are in [Data exposure audit](docs/DATA_EXPOSURE_AUDIT.md). The new
[Evaluation protocol](docs/EVALUATION_PROTOCOL_V1.md) separates task success,
terminal MAP, explicit settled commitment, base preservation, shift timing and
tool spend. A [draft preregistration](docs/PREREGISTRATION_IDENTIFICATION_V1.md)
proposes a sample and gates but is not approved. The
[P0 decision packet](docs/P0_DECISION_PACKET.md) records the tests, corrected
legacy numbers and decisions still owed. No holdout, model call or GPU run was
performed, and no candidate was tuned.

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

## Implementation map

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
| RuleShift-Grid | `src/ruleshift/{public_api/grid_*,envs/grid/,agents/grid_*,evaluator/grid_*}`, `scripts/run_grid_*.py`, `configs/grid_dev.yaml` | additive Tier A benchmark and Tier B RGB+text scaffold (prompt `grid_v2`, 2048-token budget); CPU-only, no scientific run |
| Freeze scaffolding | `configs/preregistered_test.yaml` | `DRAFT_NOT_FROZEN` — intentionally unfrozen |
| Literature | `research/literature_matrix.csv`, `research/reading_notes.md` | canonical URLs verified; abstracts/READMEs only, full texts not read |
| Reproducibility | `requirements.lock`, `scripts/preflight.py`, `scripts/check_reproducibility.py`, `scripts/reproduce_all.py`, `docs/RUNBOOK.md` | implemented |
| Optional shift detection | `src/ruleshift/agents/change_detection.py`, `docs/world_model_trigger.md` | CPU-only CUSUM/residual utility; not yet wired to a learned model or closed-loop runner |
| Gemini prototype | `prototypes/gemini_ruleshift_grid/` | Original four-script prototype, kept separate from the main implementation |


## Verify on CPU
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
└── tests/                         # 486 tests, all passing
```

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_reproducibility.py
python scripts/reproduce_all.py --skip-tests
```

The reproduction command exercises the synthetic reports using deterministic
stubs and the CPU references. Archived Kaggle runs live separately under the
git-ignored `kaggle_outputs/` directory in the original workspace.
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

Setup scaffolding for remote diagnostics:

- `notebooks/colab_pilot.ipynb` — ready-to-run **real** offline pilot on Colab (pinned Qwen2.5-VL-3B, text-input only).
- `notebooks/colab_setup.ipynb`, `notebooks/kaggle_gpu.ipynb` — clone/install/check notebooks.
- `configs/runtime.yaml` — GitHub URL and checkpoint source (currently placeholders).
- `scripts/preflight.py`, `scripts/kaggle_preflight.py` — local-only GO/NO-GO gates.
- `scripts/run_grid_real_pilot.py`, `kaggle/kernel/run_grid.py` — remote-only grid pilot scaffold; the local guard refuses `cuda` unless `RULESHIFT_KAGGLE=1`.
- `kaggle/kaggle.json.example` — placeholder; real credentials are git-ignored.

See `docs/COLAB_KAGGLE_SETUP.md` for runtime configuration and credentials.

For the multi-week real-VLM grid (100 maps x 5 seeds, split into weekly Kaggle
chunks), see `docs/RUNBOOK_GRID_MULTIWEEK.md` and the plan at
`docs/superpowers/plans/2026-11-24-ruleshift-grid-multiweek-run.md`. That grid
is **closed for the frozen 3B model**; see `docs/GRID_FEASIBILITY_DECISION.md`
before scheduling anything.

## Later: real experiment and evaluation

## Current scope

The default model path still consumes public text. The P0 plumbing now attaches
the rendered public RGB frame to the observation and forwards it to the
processor when `TransformersVLMBackend(input_mode="rgb")` is selected; a
pixel-dependence test covers request identity and forwarding. A real
pixel-dependent VLM run has not been executed, so P0 remains unvalidated. The
primary endpoint is paired closed-loop return after real validation and a test
freeze. Prompt changes, the P0 path and agent comparisons belong on development
or validation inputs. The locked configuration remains `DRAFT_NOT_FROZEN`.

The optional [CUSUM detector](docs/world_model_trigger.md) and the
[Gemini grid prototype](prototypes/gemini_ruleshift_grid/README.md) remain
separate experiments. The canonical benchmark lives in `src/ruleshift/`.
