# RuleShift-Grid: additive benchmark design

**Status:** draft for human approval
**Date:** 2026-11-23
**Scope:** add a new grid environment beside the existing Family A/B
environments. Do not replace, rename, or weaken Family A/B.
**Primary track:** value-aware probing under hidden dynamics shifts with a
frozen VLM.
**Implementation horizon:** Tier A in Weeks 1-2 on CPU; Tier B behind an
explicit interface, using the frozen VLM only when Tier A is validated.

---

## 1. Why this environment exists

Family A (Gate/Lock) and Family B (Bridge/Hazard) test macro-level probe
decisions, but their public action menus hide the low-level control problem.
RuleShift-Grid adds a small, fully inspectable gridworld in which the agent
must:

1. act physically (move, interact) under stochastic dynamics;
2. decide when a transition discrepancy is evidence of a rule shift rather
   than ordinary aleatoric noise;
3. spend a separate epistemic budget on `INSPECT` and `READ_SIGN`;
4. plan with a reference planner (A*) in Tier A and with a pluggable learned
   world model plus MPC/beam search in Tier B.

The environment is deliberately small enough to enumerate beliefs exactly in
Tier A, so every claim about probing can be checked against an oracle. It is
also rich enough that a signpost cannot be read once at `t=0` to solve the
episode.

### Research questions

- **RQ1 (detection):** under controlled aleatoric noise, can an agent separate
  a hidden rule shift from ordinary stochasticity and keep false alarms low?
- **RQ2 (probing):** does information-gain probing beat random probing and
  advisor-only advice under the same total budget?
- **RQ3 (tools):** does partial, costly visual/signpost evidence plus a frozen
  VLM improve adaptation over physical probing alone, without becoming a
  one-shot signpost shortcut?
- **RQ4 (planner):** how far is a learned world model plus MPC/beam search from
  the A* reference when the transition model is no longer hand-coded?

### Non-goals

- No VLM training or fine-tuning.
- No replacement of Family A/B, their contracts, or their evaluator.
- No frozen test configuration until a real validation run passes the
  preregistered gate.
- No claim of novelty, SOTA, or publishability from a smoke run.
- No hidden-state leakage into prompts, caches, public logs, or renders.

### Anti-goals

These are failure conditions even if the code runs:

- a deterministic residual that makes `residual > 0` equivalent to "shift";
- a signpost or HUD that reveals the complete active rule at no cost;
- irreversible death as the normal Tier A probing cost;
- A* hard-coded as the only possible planner;
- metrics that mix env steps with tool calls without a fixed exchange rate;
- any agent import of `ruleshift.envs` or `ruleshift.evaluator`.

---

## 2. Additive integration with the existing repository

The existing package boundaries stay intact:

- `src/ruleshift/public_api/` remains the agent-visible contract.
- `src/ruleshift/envs/` remains simulator-only.
- `src/ruleshift/evaluator/` remains privileged.
- `src/ruleshift/agents/` remains public-only.

### 2.1 New files

```
src/ruleshift/envs/grid/
  __init__.py
  config.py          # GridConfig, defaults, YAML loading
  mapgen.py          # controlled random maps + solvability check
  entities.py        # CellType, Entity, GridState
  rules.py           # rule hypotheses and shift families
  noise.py           # aleatoric observation/transition models
  grid_env.py        # Gymnasium Env: reset/step/render
  observations.py    # symbolic and visual public views
  render.py          # 224x224 RGB, Set-of-Mark, HUD
  tools.py           # INSPECT / READ_SIGN semantics and costs
src/ruleshift/evaluator/
  grid_trace.py      # privileged GridEvaluatorTrace
  grid_reference.py  # exact belief, Q*, oracle return for Tier A
  grid_metrics.py    # detection/probing/tool metrics
scripts/
  run_grid_smoke.py
  run_grid_audit.py
configs/
  grid_dev.yaml
tests/
  test_grid_mapgen.py
  test_grid_env.py
  test_grid_noise.py
  test_grid_belief.py
  test_grid_tools.py
  test_grid_metrics.py
  test_grid_permissions.py
```

### 2.2 Contract extensions

Add, do not mutate:

- `GRID_PHYSICAL_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT")`
- `GRID_TOOL_ACTIONS = ("INSPECT", "READ_SIGN")`
- `GRID_ACTIONS = GRID_PHYSICAL_ACTIONS + GRID_TOOL_ACTIONS`
- `GRID_RULE_IDS` for the new rule hypotheses.
- `GRID_PUBLIC_OBSERVATION_FIELDS` for the optional image payload.

`MACRO_ACTIONS` becomes the union of the existing macro ids and
`GRID_ACTIONS`. Existing environments continue to advertise only their own
subset through `PublicObservation.available_actions`, so their tests and
behaviour are unchanged.

Do not add an image field to `PublicObservation`. Add a separate
`GridPublicObservation` that carries the existing public fields plus:

- `image`: optional `numpy.ndarray` (H x W x 3, uint8) or `None` in symbolic
  mode;
- `action_mask`: tuple of legal grid actions at this step;
- `tool_budget_remaining`: public integer.

The grid decision log uses a new `GridDecisionRecord` that extends the existing
public record fields with `tool_calls_used`, `tool_cost`, and
`information_gain_bits`. The evaluator-only trace is a separate
`GridEvaluatorTrace`; it never enters a prompt or cache.

### 2.3 Compatibility rule

Every new schema validator takes an explicit `allowed_rules` and
`allowed_actions` argument, defaulting to the existing Family A/B sets. The
existing `validate_public_observation` and `validate_structured_vlm_output`
keep their current signatures and behaviour. New grid validators are separate
functions, not a breaking rewrite.

---

## 3. Environment specification

### 3.1 Map

- Default size: `7 x 7`. Stress size: `9 x 9`.
- Generated from a seed with a fixed, documented distribution:
  wall density, start/goal placement, key/door placement, hazard count,
  signpost placement.
- Every generated map must pass a solvability check before it is used:
  BFS from start to goal under the base rule, with the key and door reachable
  and at least one safe route that does not require crossing a lethal cell.
- Public geometry is fixed for an episode and shared by every compared method.
- The hidden rule, shift time, noise draws, and true cell properties are not
  part of the public map.

### 3.2 Entities

- `Agent`: position, health, inventory.
- `Wall`: blocks movement.
- `Goal`: episode success when reached (or when the door is opened and the
  goal is reached, depending on the map).
- `Hazard`: soft damage in Tier A; lethal only in Tier B and only when
  `INSPECT` is available.
- `Key`: opens the matching door.
- `Door`: locked until the matching key is used.
- `Signpost`: a public text/icon surface for partial, possibly ambiguous
  evidence. It is not an oracle.
- `HUD`: public status line (health, inventory, steps, tool budget); it never
  shows the hidden rule.

### 3.3 Actions and budget

Physical actions cost one environment step:

`UP`, `DOWN`, `LEFT`, `RIGHT`, `INTERACT`

Epistemic tools cost one tool-budget unit, not an environment step:

- `INSPECT(x, y)`: reveal one hidden cell property (hazard/key/door) with a
  documented sensor error rate; out-of-range or already-inspected cells are
  rejected or charged according to the public rule.
- `READ_SIGN()`: return the signpost/HUD text visible at the current position,
  or a partial clause if the sign is multi-part. The result may be ambiguous.

Total budget for every method:

```
TotalCost = N_env_steps + lambda * (N_tool_calls + N_advisor_calls)
```

`lambda` is fixed before the first comparison run and is part of the public
cost description. Tool calls are logged separately from environment steps.

### 3.4 Gymnasium API

`GridEnv(gym.Env)` implements:

- `reset(*, seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render() -> np.ndarray | None`
- `action_space`: `spaces.Discrete(len(GRID_ACTIONS))` with an action mask in
  the observation/info.
- `observation_space`: a `spaces.Dict` for symbolic mode, or `spaces.Box` for
  visual mode.

`info` contains only public quantities: step count, tool budget, last public
outcome, action mask, and public cost fields. Hidden state is available only
through `evaluator_state()`, which is called by the evaluator package, never by
an agent.

### 3.5 Observation modes

Both modes come from the same engine and the same public state.

**Symbolic (Tier A).** A small tensor/ASCII grid, agent position, health,
inventory, public history, candidate rule descriptions, public costs, and the
action mask. Runs thousands of episodes per second on CPU.

**Visual-semantic (Tier B).** A `224 x 224` RGB frame with:

- distinct icons for agent, wall, goal, hazard, key, door, signpost;
- a Set-of-Mark coordinate grid so the VLM can name `(x, y)` for `INSPECT`;
- a HUD with health, inventory, steps, and tool budget;
- no pixel that encodes the hidden rule except the public signpost/HUD text.

The renderer is deterministic for a given public state and seed.

---

## 4. Rule shifts

Each episode samples one hidden rule `r*` from a finite hypothesis set. A
single shift may occur at a hidden time `t_shift`; no-shift control episodes
use the same maps and noise. Every rule is stochastic; no rule is detectable
by a single deterministic residual.

| Family | Rule | Mechanism | What the agent can observe | Intended probe |
|---|---|---|---|---|
| A1 | Control Remap | Movement keys are remapped (swap or rotate). | Intended vs actual displacement, with slip noise. | Try one safe move and compare likelihoods of the mappings. |
| A2 | Slippery / Inertia | Baseline slip probability rises; movement may slide two cells or until a wall. | Displacement distribution, not a guaranteed offset. | Probe toward open space, never toward a suspected hazard. |
| A3 | Hazard/Goal Swap | Hazard and goal semantics swap; key/door colours may swap. | Reward/damage and interaction outcome. | Soft-hazard probe in Tier A; `INSPECT` in Tier B. |
| B4 | Signpost Composition | The rule is a composition of clauses hinted by the signpost/HUD (e.g. key colour + floor condition). | Partial, ambiguous text/icon; transition evidence. | `READ_SIGN` plus `INSPECT` plus physical evidence; no single call solves it. |

### 4.1 Stochasticity parameters (initial defaults)

These are builder calls, not frozen values:

- baseline slip: `0.10` (one cell);
- slippery shift: `0.80` (one or two cells, or slide until wall);
- `INSPECT` sensor error: `0.05` false positive / `0.05` false negative;
- `READ_SIGN` ambiguity: one clause is hidden or one clause has two plausible
  readings;
- shift hazard: one shift per episode with probability `p_shift`; no-shift
  controls use `p_shift = 0`;
- soft hazard damage: `-2` reward and one-step recovery, not death, in Tier A;
- lethal trap: Tier B only, and only on maps where `INSPECT` is available.

### 4.2 Hypothesis space

Tier A uses a finite, enumerable hypothesis set:

- base rule;
- each A1/A2/A3 variant;
- a small cross-product for B4 clauses when the signpost is present.

The exact size is fixed in `configs/grid_dev.yaml` and logged with every run.
Tier B may propose new compositional hypotheses, but the VLM output is parsed
into the same structured hypothesis format; the selector never receives a
free-form rule.

### 4.3 Detection is probabilistic

The agent maintains `b_t(h)`, a belief over hypotheses. A transition residual is
not a shift flag. The update uses the public observation likelihood:

```
b_{t+1}(h) proportional to P(o_t | s_t, a_t, h) * b_t(h)
```

with a small prior probability of a shift at each step. The trigger is a
calibrated statistic (posterior entropy, likelihood ratio, or surprise), not
`actual_pos != predicted_pos`. False alarms are measured on no-shift episodes.

### 4.4 Active probing

When the trigger fires, the agent chooses a probe that maximizes expected
information gain per total cost:

```
a_probe = argmax_a  I(H; O | s, a, b) / (env_cost(a) + lambda * tool_cost(a))
```

subject to a public safety constraint: the estimated probability of taking
lethal damage is below a threshold. In Tier A the expectation is enumerated
exactly over the finite hypothesis set. In Tier B the VLM proposes candidate
tools/actions and the selector scores them with the same public objective.

### 4.5 Belief to control

- **Tier A reference planner:** A* on the highest-probability hypothesis, with
  the public map and the public action costs. A* is a reference, not the only
  planner.
- **Planner interface:** a `DynamicsModel` protocol with
  `predict_proba(state, action)` and `sample(state, action, rng)`. A* uses a
  hand-coded implementation of that protocol; a learned MLP/CNN world model can
  implement the same protocol later.
- **Tier B planner:** learned world model plus MPC/beam search over the same
  action space. The planner receives only public state and the model's
  predictions; it never receives the true rule.
- **Oracle:** evaluator-only A* with the true rule, used as an upper bound.

---

## 5. Tier B: frozen VLM and epistemic tools

### 5.1 Inputs

The VLM receives only:

- the current RGB frame (or a symbolic view in a text-only ablation);
- public history of actions, outcomes, and tool results;
- candidate rule descriptions;
- public costs and remaining budgets;
- the action mask.

It never receives the true rule, shift time, noise draw, transition kernel,
Bayes belief, Q*, or oracle return.

### 5.2 Outputs

A structured JSON object with:

- `hypothesis_probabilities`: one entry per candidate hypothesis;
- `tool_call`: `INSPECT(x, y)`, `READ_SIGN`, or `NONE`;
- `action`: one legal physical action;
- `predicted_outcomes`: optional per-hypothesis outcome probabilities;
- `confidence`: optional scalar, used only for calibration diagnostics.

The parser rejects unknown fields and privileged names. The cache key includes
the image hash, public history hash, action mask, prompt version, and model
revision.

### 5.3 Tool semantics

- `INSPECT(x, y)` reveals one hidden property with sensor noise; it does not
  reveal the rule.
- `READ_SIGN()` returns one clause or an ambiguous reading; it does not reveal
  the complete rule.
- The signpost is placed so that reaching it costs physical steps and the
  information is partial. A scripted "read sign at t=0, then walk to goal"
  policy must fail or underperform on the held-out maps; this is an explicit
  acceptance test.
- Tool calls are charged once, in the same total-cost stream as environment
  steps.

### 5.4 Ablations

- VLM without tools (physical probing only).
- Scripted tool use without VLM (information-gain selector).
- VLM plus tools.
- A* plus exact belief (Tier A reference).
- Learned world model plus MPC (Tier B planner ablation).

---

## 6. Experimental protocol

### 6.1 Scenarios

- At least 100 random maps for development and 100 held-out maps for
  evaluation; 5 seeds per map.
- Four shift families plus no-shift controls.
- Shift times sampled inside a fixed window; the window is public, the exact
  time is hidden.
- Held-out maps use different seeds, sizes, and wall densities from
  development.

### 6.2 Baselines (all with the same total budget)

1. **No-Adaptation:** A* with the base rule for the whole episode.
2. **Uncertainty-Gated Action Advice:** on a detected discrepancy, ask an
   advisor for one action; no belief update.
3. **Random Probing:** on a detected discrepancy, take one or two random
   probes, then update.
4. **Info-Gain Probing:** the proposed method; choose the probe with maximum
   expected information gain per cost, subject to safety.
5. **Oracle:** knows the shifted rule at `t_shift`; upper bound, evaluator-only.
6. **Bayes-optimal Tier A reference:** exact belief plus A*; reference for
   detection and planning.
7. **VLM-only and VLM+tools:** Tier B ablations.

### 6.3 Metrics

Log from day one:

- success rate before the step limit;
- adaptation steps from `t_shift` to the first correct-rule decision;
- total cost (`env_steps + lambda * tool_calls`);
- false alarm rate on no-shift episodes;
- detection delay and detection precision/recall;
- calibration: Brier score and expected calibration error for shift
  probability;
- information gain per tool call and per total cost;
- decision regret against the oracle;
- hazard damage and deaths (Tier B);
- tool precision/recall for `INSPECT`;
- VLM JSON validity, valid-action rate, output tokens, latency;
- cache hit rate.

### 6.4 Statistics

- Paired scenario-level bootstrap confidence intervals; resample scenarios,
  never individual steps or frames.
- Report all seeds, not only the best.
- Pre-register the primary metric and the shift/no-shift split before the
  held-out run.
- Keep `configs/preregistered_test.yaml` at `DRAFT_NOT_FROZEN` until a real
  validation run passes the gate.

### 6.5 Ablations

- Noise level: baseline slip `0.05 / 0.10 / 0.20`.
- Signpost informativeness: full clause / partial clause / ambiguous clause.
- Tool cost: `lambda = 1 / 2 / 5`.
- Shift family: A1, A2, A3, B4, no-shift.
- Trigger: entropy / likelihood ratio / fixed threshold.
- Planner: A* / learned world model + MPC.
- Budget: tight / medium / loose.

---

## 7. Implementation phases

### Phase 1 — Tier A symbolic (Weeks 1-2)

- `GridConfig`, map generator, solvability check.
- Entities, actions, stochastic transitions, soft hazards.
- Rules A1-A3 and the finite hypothesis set.
- Bayesian belief update, entropy/likelihood trigger.
- A* reference planner and `DynamicsModel` protocol.
- Baselines 1-4 and oracle.
- Metrics, paired CIs, smoke script, tests.

### Phase 2 — Visual and tools (Weeks 2-3)

- 224x224 renderer with Set-of-Mark and HUD.
- `INSPECT` / `READ_SIGN` semantics and costs.
- `GridPublicObservation` with image.
- Frozen-VLM prompt, parser, cache, permission checks.
- Signpost-shortcut acceptance test.
- Tier B smoke on a small map subset.

### Phase 3 — Learned planner (after Tier A is validated)

- Learned world model behind `DynamicsModel`.
- MPC/beam search.
- Compare against A* reference.
- Do not start this phase until Phase 1 metrics are stable.

---

## 8. Acceptance criteria

The addition is acceptable only when:

1. The existing 217 tests still pass unchanged.
2. New tests cover map solvability, stochastic transitions, soft hazards,
   partial signposts, tool costs, budget accounting, belief updates, action
   masks, renderer determinism, and the agent/evaluator permission boundary.
3. A no-shift control produces a measurable false-alarm rate, and a
   deterministic-residual detector is explicitly shown to be inadequate.
4. A scripted signpost-only policy fails or underperforms on held-out maps.
5. `INSPECT` improves outcomes only when its cost is included in the total.
6. Tier A smoke runs at least 100 maps x 5 seeds for every baseline.
7. Tier B smoke produces valid JSON with a real frozen VLM and no privileged
   leakage.
8. No generated result is presented as a scientific finding before
   preregistration and a real validation run.

---

## 9. Builder calls (defaults, not user decisions)

These are sensible defaults; the human can override them before
implementation:

- default grid `7 x 7`, stress `9 x 9`;
- baseline slip `0.10`, slippery slip `0.80`;
- `INSPECT` error `0.05`, `READ_SIGN` cost `2` tool units;
- `lambda = 2` for the first comparison;
- soft hazard damage `-2` and one-step recovery in Tier A;
- one shift per episode, sampled uniformly in a public window;
- A1/A2/A3 use the existing `red`/`blue`/`green` colour vocabulary where
  possible; B4 adds compositional clause ids.

## 10. Open questions for the human

Only these change what gets built:

1. Is the first deliverable Tier A only, or do you want the Tier B VLM path
   scaffolded in the same change?
2. Should the default map be `7 x 7` or `9 x 9` for the first benchmark table?
3. Is `lambda = 2` the intended exchange rate between one environment step and
   one tool/advisor call, or should the first run use a different value?

## 11. Risks and mitigations

- **Too many moving parts.** Phase the work; do not implement Tier B before
  Tier A is stable.
- **Signpost shortcut.** Explicit scripted-policy test and partial clauses.
- **Threshold tuning on test.** Calibrate only on validation; keep the test
  configuration unfrozen.
- **VLM prompt sensitivity.** Freeze prompt version, cache by hash, report
  invalid JSON separately.
- **Compute.** Tier A is CPU-only; Tier B uses the existing Kaggle path and
  the frozen model revision.
- **Metric ambiguity.** Fix the total-cost formula and log env steps and tool
  calls separately.
