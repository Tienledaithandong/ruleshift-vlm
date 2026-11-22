# TASK BRIEF FOR CODING AGENT — WEEK 1

You are an experienced research engineer. Your goal is to set up the **RuleShift / Agentic VLM** research repository and create a falsifiable, unambiguous problem specification for value-aware probing under hidden dynamics shifts. **Do not implement a neural model, VLM runner, Bayes oracle or full environment yet.** Focus on source-grounded literature, contracts, static interfaces and test plans. See `RuleShift_VLM_Research_12_Week_Master_Plan.md` for the full project.

## Core RQ

> When hidden environment dynamics change without warning, do frozen vision-language agents choose probes whose control value justifies their cost, and can structured counterfactual outcome reasoning—without access to the environment's generative model—reduce probe-decision regret and improve closed-loop return?

## Non-negotiable permissions boundary

- Agent can use only public RGB observation, its own previous action/outcome history, valid macro-action menu and textual descriptions of candidate mechanics.
- Evaluator alone may read hidden active mechanic, shift schedule, full stochastic transition kernel, oracle belief and exact action values.
- No code in `src/ruleshift/agents/` may import `src/ruleshift/envs/` or `src/ruleshift/evaluator/` or run `env.copy().step` to score counterfactual actions.
- Separate evaluator traces from public agent input. Test this boundary via static import checks and schema checks.
- The common macro executor may use only public static geometry, and all macro travel/probe/lockout costs must count as primitive environment steps and in the same reward stream.

## Required outputs

1. `README.md` (research goal, non-goals, reproducibility plan; no invented results).
2. `pyproject.toml` (CPU-first Python 3.11, minimal dependencies, pytest).
3. `docs/problem_spec.md` — a full technical contract, including family A Gate/Lock, eventual family B Hazard/Bridge, state and observation definitions, legal actions, `reset/step` semantics, reward and cost once-only, single unannounced shift and no-shift episodes, feedback noise, random seeds and macro-duration semantics.
4. `docs/agent_evaluator_permissions.md` — explicit public/privileged field inventory, examples of forbidden leakage, test plan.
5. `docs/benchmark_protocol.md` — hypotheses H1/H2/H3; primary metric total undiscounted episodic return; microtask exact decision regret, paired CI and cost/call accounting; development/validation/locked-test split by scenario; sample-plan and preregistration requirements.
6. `research/literature_matrix.csv` — at least DIVA 2026, ZendoWorld 2026, EVOLvE ICML 2025, Piriyakulkij NeurIPS 2024 and One Life to Learn ICLR 2026, with URLs and contrast columns; clearly flag any paper that was not fully read.
7. `configs/dev.yaml` — development-only probe-cost / noise / diagnosticity grid with deterministic seeds. Do NOT freeze test scenarios yet.
8. `.github/workflows/tests.yml`, `tests/test_schemas.py`, `tests/test_agent_permissions.py` with tests feasible before environment implementation.
9. `docs/week2_issues.md` — ordered tasks for Gymnasium env, semantic RNG, public view, macro executor, oracle isolation, env-checker and smoke tests.

## Operational constraints

- Ground the literature matrix in actual canonical papers/official repository README, not hallucinated titles, years or numbers.
- Make exact Bayes reference only for enumerated microtasks, approximate elsewhere.
- Do not hard-code a narrow shift time window or expose shift schedule through an episode-progress feature.
- Include at least one development setting where direct task action is preferable and one where probing is preferable; those are hypotheses to validate when the environment is implemented, not results.
- The main planned VLM method cannot use exact environment likelihoods or Bayes Q*; it will depend only on VLM-estimated beliefs/outcomes and public cost descriptions. Its continuation-value estimator must account for future decisions after a probe, not only immediate reward.
- Be conservative about GPU requirements and free-tier availability. The week-1 repo must run fully on CPU.
- Use typing, docstrings, simple tests and deterministically seeded placeholders; no fabricated research plots.

## Review and acceptance

Do not declare completion until `pytest -q` passes, README and design docs exist, literature matrix references verified sources and the permissions contract is explicit. Print a file tree and summarize exactly what is implemented vs merely specified. Include unresolved scientific questions as a short issue list rather than filling gaps with assumptions. Do not claim that a paper is publishable or that a novel method has been demonstrated.

## Canonical references

- DIVA: https://arxiv.org/abs/2610.32181 ; https://github.com/seunghan96/diva
- ZendoWorld: https://arxiv.org/abs/2607.08233 ; https://github.com/ml-research/ZendoWorld
- EVOLvE: https://proceedings.mlr.press/v267/nie25b.html
- Doing Experiments and Revising Rules: https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html
- One Life to Learn: https://proceedings.iclr.cc/paper_files/paper/2026/hash/a24d08f2eecc6c96f4a0ad37a74ddeca-Abstract-Conference.html
- Gymnasium custom environment guide: https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/
