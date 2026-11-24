# RuleShift / Agentic VLM

**Research checkpoint (2026-11-24) — Week 4, revised pilot verified on CPU.**
The frozen Qwen anchor has run on Kaggle in public-text mode (P1). The v12
archive contains 81 decisions from 27 distinct requests. All 81 chose
`PROBE_RED`. Several noise conditions received the same input despite
different exact labels. The pilot demonstrates inference and parsing.
A clean Week 4 diagnostic and real Week 6 validation are still pending.

The revised P1 harness covers controlled first decisions and decisions
following noisy public histories under a possible shift. Each default design
contains 81 distinct inputs with exact microtask labels.

[Progress and evidence](docs/RESEARCH_STATUS.md) · [CPU and experiment runbook](docs/RUNBOOK.md)

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
| Freeze scaffolding | `configs/preregistered_test.yaml` | `DRAFT_NOT_FROZEN` — intentionally unfrozen |
| Literature | `research/literature_matrix.csv`, `research/reading_notes.md` | canonical URLs verified; abstracts/READMEs only, full texts not read |
| Reproducibility | `requirements.lock`, `scripts/preflight.py`, `scripts/check_reproducibility.py`, `scripts/reproduce_all.py`, `docs/RUNBOOK.md` | implemented |
| Optional shift detection | `src/ruleshift/agents/change_detection.py`, `docs/world_model_trigger.md` | CPU-only CUSUM/residual utility; not yet wired to a learned model or closed-loop runner |
| Gemini prototype | `prototypes/gemini_ruleshift_grid/` | Original four-script prototype, kept separate from the main implementation |


## Verify on CPU

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_reproducibility.py
python scripts/reproduce_all.py --skip-tests
```

The reproduction command exercises the synthetic reports using deterministic
stubs and the CPU references. Archived Kaggle runs live separately under the
git-ignored `kaggle_outputs/` directory in the original workspace.

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
