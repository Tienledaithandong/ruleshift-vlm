# Agent / Evaluator Permissions Contract (v0.1)

**Status:** specified and statically enforced in CI. The runtime environment is
Week 2 work; the inventory and the static checks exist now.

---

## 1. Principle

The agent sees only what a real deployed vision-language agent could observe:
public RGB, its own action/outcome history, a public macro-action menu, and
natural-language descriptions of candidate mechanics and their public costs.
The evaluator alone holds the simulator and the labels used to score decisions.
These are **two separate data paths**, not one `info` dictionary with a
voluntary "don't look" convention.

```
env (hidden) ──public_view──> public_api ──> agent  ──macro id──> executor
     │
     └──evaluator_state──> evaluation-only adapter ──> evaluator_only.jsonl
                                      (NEVER into prompts/cache/decisions.jsonl)
```

---

## 2. Privileged field inventory (evaluator only)

Every name in this table is forbidden in any agent-visible payload, prompt,
prompt template, cache key/payload, or `decisions.jsonl` row. The set is
maintained in code as `ruleshift.permissions.PRIVILEGED_FIELDS` and checked
against this document by `tests/test_agent_permissions.py`.

| Field | Meaning |
|---|---|
| `mechanic_id` | id of the active hidden mechanic |
| `true_rule` | the actual active rule label |
| `true_rule_before_after` | rule before and after a shift |
| `active_mechanic` | alias for the currently active rule |
| `shift_time` | primitive step at which the shift occurred |
| `shift_schedule` | preselected shift time sequence |
| `shift_hazard` | hidden per-step shift hazard value |
| `latent_hazard` | hidden hazard state (Family B) |
| `hidden_rng` | RNG state/stream of the simulator |
| `hidden_state` | full latent state `x_t` |
| `transition_kernel` | `P(x' | x, a)` |
| `transition_probabilities` | numeric transition table |
| `oracle_belief` | evaluator exact posterior over hidden state |
| `bayes_belief` | evaluator filtered belief |
| `q_star` | exact optimal action values |
| `exact_q_values` | evaluator action-value labels |
| `delta_probe` | evaluator probe-vs-task value difference |
| `actual_noise_outcome` | realised noisy feedback draw |
| `privileged_oracle_return` | return of a true-rule controller |
| `scenario_generating_seed` | seed that generated the scenario |

This inventory is deliberately over-inclusive. Rejecting a harmless name is
preferable to leaking a privileged one.

---

## 3. Public field allowlist

Exactly these top-level observation keys are allowed
(`ruleshift.permissions.PUBLIC_OBSERVATION_FIELDS`):

```
observation_id, public_obs_hash, public_history_hash, public_history, public_text,
available_actions, candidate_rules, public_costs
```

`public_obs_hash` and `public_history_hash` are digests of immutable public
snapshots; they let a cache key include history without embedding hidden state.
Cache keys must also include model id/revision, decoding config/seed, prompt
template version, task family, image digests, public history, candidate-rule
descriptions and available actions. **Never cache by current frame alone** —
identical images with different histories require different actions.

---

## 4. Forbidden leakage examples

All of the following are violations even if they occur only in a debug branch:

- `from ruleshift.evaluator import bayes_filter` inside `src/ruleshift/agents/`.
- `from ..envs.gate_lock import RuleShiftEnv` inside an agent module.
- `score = env.copy().step(a)` to score a counterfactual probe outcome.
- `info["delta_probe"]` copied into a prompt string or a cache key.
- A text description such as "the active rule is blue" in `public_text`.
- A feature `phase = step / MAX_ENV_STEPS` combined with the hidden shift
  schedule, or any prescheduled narrow shift window.
- A `true_rule` field in structured VLM output — the output schema rejects it
  and the parser must log and sanitise it deterministically.
- Joining `evaluator_only.jsonl` columns back into model-facing datasets.
- An observation-named cache key that omits history, causing reuse across
  different posterior states.

---

## 5. Static and schema test plan (implemented in Week 1)

`tests/test_agent_permissions.py`:

1. AST scan of every `*.py` under `src/ruleshift/agents/`; forbid imports whose
   path segments include `envs` or `evaluator` (including relative imports).
2. AST scan forbidding `env.copy().step(...)` and direct `<env-like>.step(...)`
   counterfactual/direct simulator calls.
3. Tests of the scanner itself using synthetic bad and good source strings, so
   the checker cannot silently pass by scanning nothing.
4. Recursive payload scan of dict keys, nested structures and string values for
   every name in `PRIVILEGED_FIELDS`.
5. Allowlist/disjointness assertions between public and privileged field sets.
6. A check that this document names every privileged field, so documentation
   cannot drift away from code.

`tests/test_schemas.py`:

- Valid/invalid structured VLM outputs and public observations.
- Probability bounds and sums, action-id validity, feedback vocabulary.
- Explicit injection of `true_rule`, `shift_time` and `oracle_belief` must fail.
- Validators never mutate their inputs.

`CI` (` .github/workflows/tests.yml`) runs `pytest -q` on Python 3.11 on every
push/PR. Week 2 adds `gymnasium.utils.env_checker.check_env` and runtime
assertions that `public_view()` and `evaluator_state()` share no privileged
keys.

---

## 6. Caveats

Static import checks catch direct imports and obvious counterfactual calls;
they cannot prove the absence of all information flow (e.g. a hand-copied
numeric table inside an agent file). Runtime logging, code review and the
`evaluator_only.jsonl` separation remain necessary. Oracle-belief and
perfect-outcome input modes (P2/P3) are diagnostic upper bounds only, must be
segmented in every table, and are never reported as agent-valid methods.

Week 1 verifies that this contract is **expressible and testable**; it does not
verify a running environment, because none exists yet.
