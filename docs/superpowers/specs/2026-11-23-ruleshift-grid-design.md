# RuleShift-Grid: additive benchmark design

**Status:** approved for implementation planning (2026-11-23)
**Date:** 2026-11-23
**Approved scope:** Tier A CPU environment plus Tier B renderer/tool/VLM
scaffold; default map `7 x 7`; advisor cost `lambda = 2`. The real frozen-VLM
run remains a remote Kaggle gate.
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
- metrics that mix env steps, tool units, and advisor calls without a stated
  accounting rule;
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
src/ruleshift/public_api/
  grid_schemas.py    # grid actions, observation, decision record, validators
  grid_rules.py      # public hypothesis set and transition semantics
  grid_planner.py    # DynamicsModel protocol, hypothesis model, A* reference
src/ruleshift/envs/grid/
  __init__.py
  config.py          # GridConfig, defaults, YAML loading
  mapgen.py          # controlled random maps + all-hypothesis solvability
  entities.py        # CellType, Entity, GridState, GridMap
  rules.py           # shift schedule and active-rule application
  noise.py           # aleatoric transition and sensor models
  grid_env.py        # Gymnasium Env: reset/step/render
  observations.py    # symbolic and visual public views
  render.py          # 224x224 RGB, Set-of-Mark, HUD
  tools.py           # INSPECT / READ_SIGN semantics and costs
src/ruleshift/agents/
  grid_belief.py     # change-point belief filter
  grid_selector.py   # info-gain and public advisor policies
  grid_baselines.py  # no-adaptation and random-probing baselines
  grid_vlm.py        # grid prompt, parser, cache, policy adapter
src/ruleshift/evaluator/
  grid_trace.py      # privileged GridEvaluatorTrace
  grid_reference.py  # exact belief, Q* microtasks, oracle return
  grid_metrics.py    # detection/probing/tool metrics
  grid_runner.py     # closed-loop grid runner and oracle policy
src/ruleshift/prompts/
  grid_v1.txt        # public-only grid prompt
scripts/
  run_grid_smoke.py
  run_grid_audit.py
  run_grid_real_pilot.py
configs/
  grid_dev.yaml
tests/
  test_grid_schemas.py
  test_grid_rules.py
  test_grid_mapgen.py
  test_grid_env.py
  test_grid_noise.py
  test_grid_belief.py
  test_grid_tools.py
  test_grid_planner.py
  test_grid_selector.py
  test_grid_runner.py
  test_grid_metrics.py
  test_grid_observations.py
  test_grid_render.py
  test_grid_vlm.py
  test_grid_permissions.py
  test_grid_signpost_shortcut.py
```

### 2.2 Contract extensions

Add, do not mutate, in `src/ruleshift/public_api/grid_schemas.py`:

- `GRID_PHYSICAL_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT")`
- `GRID_TOOL_ACTIONS = ("INSPECT", "READ_SIGN")`
- `GRID_ACTIONS = GRID_PHYSICAL_ACTIONS + GRID_TOOL_ACTIONS`
- `GRID_ACTIONS_FROZEN = frozenset(GRID_ACTIONS)`
- `GRID_RULE_IDS` for the new rule hypotheses.
- `GRID_PUBLIC_OBSERVATION_FIELDS` for the optional image payload.
- `ALL_ACTIONS = MACRO_ACTIONS + GRID_ACTIONS` only for the new grid prompt
  and the new grid validator.

`MACRO_ACTIONS` stays exactly as it is. Existing validators, including
`validate_structured_vlm_output`, keep accepting only the Family A/B macro set.
The grid has its own `validate_grid_vlm_output` and its own accepted action
set. This is deliberate: widening the global macro set would let a Family A
response legally emit `"UP"`, which the closed-loop runner would then clamp and
count as invalid. Do not do that.

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
- Every generated map must pass a solvability check before it is used: for
  every hypothesis in the finite Tier A set, BFS from start to goal with the
  key and door reachable and at least one safe route that does not require
  crossing a lethal cell.
- Public geometry is fixed for an episode and shared by every compared method.
- **Public:** wall positions, tile appearance (colour/shape), start/goal
  locations, key/door/signpost locations, the agent, the HUD, public costs, and
  the public history.
- **Hidden:** which appearance class is currently safe/goal/hazard, which key
  opens which door, the shift time, noise draws, and true cell properties.
- A rule shift never repaints the map. A3 swaps the *semantics* of the same
  visible tiles; it does not swap the icons. The agent must infer the new
  meaning from interaction outcomes, tool evidence, and the signpost.
- Solvability is checked under every hypothesis in the finite Tier A set, not
  only under the base rule. The generator rejects maps that are unwinnable
  under any hypothesis and logs the rejection rate. This keeps the oracle upper
  bound and success-rate comparisons meaningful.

### 3.2 Entities

- `Agent`: position, health, inventory.
- `Wall`: blocks movement.
- `Appearance class`: a visible colour/shape tile. The renderer shows the
  appearance, never the semantic label. For example, a red tile and a blue tile
  are visible; which one is the goal is rule-dependent.
- `Goal tile`: the appearance class that currently completes the episode. The
  agent learns this from reward or interaction, not from a "goal" icon.
- `Hazard tile`: soft damage in Tier A; lethal only in Tier B and only when
  `INSPECT` is available.
- `Key`: a visible colour. Which door it opens is hidden and rule-dependent.
- `Door`: a visible colour. The required key is hidden and rule-dependent.
- `Signpost`: a visible icon whose text is illegible in the 224x224 frame.
  `READ_SIGN` is the only way to read a clause. It is not an oracle.
- `HUD`: public status line (health, inventory, steps, tool budget); it never
  shows the hidden rule.

### 3.3 Actions and budget

**One decision is one action.** A decision is either a physical action or a
tool call, never both at once. A physical action advances the environment step
counter by one. A tool call does not advance the environment step counter; it
returns a new observation, and the agent decides again. This keeps the MDP and
the budget accounting unambiguous.

Physical actions cost one environment step:

`UP`, `DOWN`, `LEFT`, `RIGHT`, `INTERACT`

Epistemic tools cost tool-budget units, not environment steps:

- `INSPECT(x, y)`: reveal one hidden semantic property of a cell (current
  hazard/goal role, or a door's required key) with a documented sensor error
  rate. The target must be within public Manhattan range 2 of the agent. Cost:
  2 tool units. The same cell may be inspected again only by paying again.
- `READ_SIGN()`: the agent must be adjacent to the signpost. It returns one
  clause from one side of the sign, not the whole rule. Cost: 1 tool unit. The
  sign has two or three sides; reading another side requires moving around the
  sign and paying again. The result may be ambiguous.
- The public tool budget is 3 units per episode. `INSPECT` therefore cannot be
  used to scan the whole map, which preserves the Tier B risk story.

Total budget for every method:

```
TotalCost = N_env_steps + lambda * N_advisor_calls
```

Tool calls are priced by the separate tool budget, not by `lambda`: the
budget is 3 units, `INSPECT` costs 2, and `READ_SIGN` costs 1. Tool units and
advisor calls are logged separately from environment steps. `lambda` is fixed
before the first comparison run and is part of the public cost description.

### 3.4 Gymnasium API

`GridEnv(gym.Env)` implements:

- `reset(*, seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render() -> np.ndarray | None`
- `action_space`: `spaces.Dict({"kind": spaces.Discrete(len(GRID_ACTIONS)),
  "target": spaces.MultiDiscrete([width, height])})`. `target` is ignored for
  every action except `INSPECT`. The observation/info carries an action mask.
- `step()` accepts a `GridAction(kind, target=None)` or the equivalent dict.
  A physical action increments `env_steps`; a tool action does not.
  `max_env_steps` counts physical steps only.
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

- distinct appearance classes for agent, wall, goal tile, hazard tile, key,
  door, and signpost. The render shows colour/shape, not a semantic label;
- a Set-of-Mark coordinate grid so the VLM can name `(x, y)` for `INSPECT`;
- a HUD with health, inventory, steps, and tool budget;
- a signpost icon whose text is blurred/illegible at this resolution;
  `READ_SIGN` is the only way to read a clause;
- no pixel that encodes the hidden rule. Tile appearance is stable across
  shifts; only the hidden semantics change.

The renderer is deterministic for a given public state and seed.

---

## 4. Rule shifts

Two scenario types share the same maps, noise, and agent-facing prompt:

- **Identification:** `r*` is sampled at `t=0` from the finite hypothesis set;
  no shift occurs. This measures initial rule identification and calibration.
- **Change detection:** the episode starts under the base rule `r0`, then one
  shift to `r*` occurs at a hidden `t_shift`. **No-shift controls** are the
  same generative process with no shift sampled. The agent is told the shift
  window and the hazard rate, but not whether this episode contains a shift.
  This is what makes false alarms measurable.

Every rule is stochastic; no rule is detectable by a single deterministic
residual.

| Family | Rule | Mechanism | What the agent can observe | Intended probe |
|---|---|---|---|---|
| A1 | Control Remap | Movement keys are remapped (swap or rotate). | Intended vs actual displacement, with slip noise. | Try one safe move and compare likelihoods of the mappings. |
| A2 | Slippery / Inertia | Baseline slip probability rises; movement may slide two cells or until a wall. | Displacement distribution, not a guaranteed offset. | Probe toward open space, never toward a suspected hazard. |
| A3 | Hazard/Goal Swap | The semantics of the same visible tiles swap; key/door colours may swap meaning. Appearance does not change. | Reward/damage and interaction outcome. | Soft-hazard probe in Tier A; `INSPECT` in Tier B. |
| B4 | Signpost Composition | The rule is a composition of clauses; the sign gives one clause per side, and at least one clause must be inferred from transition evidence. | Partial, ambiguous text/icon; transition evidence. | `READ_SIGN` plus `INSPECT` plus physical evidence; no single call solves it. |

### 4.1 Stochasticity parameters (initial defaults)

These are builder calls, not frozen values:

- baseline slip: `0.10` (one cell);
- slippery shift: `0.80` (one or two cells, or slide until wall);
- `INSPECT` sensor error: `0.05` false positive / `0.05` false negative;
- `READ_SIGN` ambiguity: one clause is hidden or one clause has two plausible
  readings;
- shift hazard: one shift per episode with probability `p_shift`; the
  agent-facing prior is the same for shift and no-shift episodes, so the agent
  never receives a "this is a control" label. No-shift controls are a dataset
  slice, not a different prompt;
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

The agent maintains a belief over `(h, tau)`, where `h` is a hypothesis and
`tau` is the last shift time (`None` before any shift). The shift window and
hazard rate are public; the exact time is hidden. For change-detection
episodes the update is a hazard-rate change-point filter:

```
b_{t+1}(h, tau) proportional to
  P(o_t | s_t, a_t, h) *
  [ (1 - p_shift) * b_t(h, tau)
    + p_shift * prior(h) * 1(tau = t) ]
```

For identification episodes the filter is static over `h` only. The exact
state space is `|H| x (T + 1)`, which is small enough to enumerate in Tier A.
A transition residual is not a shift flag. The trigger is a calibrated
statistic (posterior entropy, likelihood ratio, or surprise), not
`actual_pos != predicted_pos`. False alarms are measured on no-shift episodes
that share the same agent-facing prior.

### 4.4 Active probing

When the trigger fires, the agent chooses a probe that maximizes expected
information gain subject to the public tool budget and the safety constraint:

```
a_probe = argmax_a  I(H; O | s, a, b)
```

subject to:

- `tool_cost(a) <= tool_budget_remaining`;
- the estimated probability of taking lethal damage is below the public
  threshold;
- ties are broken by lower `env_cost(a)`, then lower `tool_cost(a)`.

In Tier A the expectation is enumerated exactly over the finite hypothesis
set. In Tier B the VLM proposes candidate tools/actions and the selector
scores them with the same public objective.

### 4.5 Belief to control

- **Tier A reference planner:** exact change-point belief plus A* on the
  highest-probability hypothesis, with the public map and the public action
  costs. This is a point-estimate reference, **not** a full Bayes-optimal POMDP
  policy. A* is a reference, not the only planner.
- **Exact Q* scope:** `grid_reference.py` computes Q* only for small microtasks
  (for example 5 x 5, two hypotheses, short horizon) used in unit tests. It is
  not the benchmark-scale solver. Do not call the point-estimate baseline
  "Bayes-optimal".
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
- `action`: one legal grid action, which may be a physical action or a tool
  call (`INSPECT` or `READ_SIGN`);
- `tool_target`: optional `(x, y)`, required only for `INSPECT`;
- `predicted_outcomes`: optional per-hypothesis outcome probabilities;
- `confidence`: optional scalar, used only for calibration diagnostics.

One output is one decision. A tool result returns a new observation before the
next decision; the VLM does not choose a tool and a physical action
simultaneously.

The parser rejects unknown fields and privileged names. The cache key includes
the image hash, public history hash, action mask, prompt version, and model
revision.

### 5.3 Tool semantics

- `INSPECT(x, y)` reveals one hidden semantic property with sensor noise; it
  does not reveal the rule. It is range-limited and budget-limited, so it
  cannot scan the whole map.
- `READ_SIGN()` requires adjacency and returns one clause from one side of the
  sign. The text is not legible in the frame; `READ_SIGN` is the only way to
  read it. The sign has two or three sides, and reading another side requires
  moving around the sign and paying again.
- The full rule is a composition. At least one clause must be inferred from
  transition evidence; the sign never reveals the complete rule, even if every
  side is read.
- The signpost is placed so that reaching it costs physical steps. A scripted
  "read one side at t=0, then walk to goal" policy must fail or underperform on
  the held-out maps; this is an explicit acceptance test.
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
- Two scenario types: identification (rule sampled at `t=0`, no shift) and
  change detection (base rule then one shift). No-shift controls are
  change-detection episodes with no shift sampled; the agent-facing prompt is
  identical.
- Four shift families plus no-shift controls.
- Shift times sampled inside a fixed window; the window and hazard rate are
  public, the exact time is hidden.
- Held-out maps use different seeds, sizes, and wall densities from
  development. Every map passes solvability under every Tier A hypothesis.

### 6.2 Baselines (same physical-step cap and tool budget)

1. **No-Adaptation:** A* with the base rule for the whole episode.
2. **Uncertainty-Gated Action Advice:** on a detected discrepancy, ask a
   **public** advisor for one action. The advisor sees only public state, does
   not reveal the rule, and does not update the agent's belief. Each call costs
   `lambda` in the total-cost metric. A privileged oracle advisor is reported
   separately as an upper bound, never as a baseline.
3. **Random Probing:** on a detected discrepancy, take one or two random
   probes, then update.
4. **Info-Gain Probing:** the proposed method; choose the probe with maximum
   expected information gain per cost, subject to safety.
5. **Oracle:** knows the shifted rule at `t_shift`; upper bound, evaluator-only.
6. **Exact-belief + A* reference:** exact change-point belief plus A*
   point-estimate planning; reference for detection and planning. This is not
   a full POMDP policy; the exact Q* microtask reference is separate.
7. **VLM-only and VLM+tools:** Tier B ablations.

### 6.3 Reward, budget, and metrics

The reward is the same currency for every method: goal `+10`, physical step
`-0.05`, soft hazard `-2` with one-step recovery, lethal trap `-10` and
termination in Tier B. `max_env_steps` is `60` for `7 x 7` and `90` for
`9 x 9`. All methods share the same physical-step cap and the same tool
budget. `TotalCost` is a reported efficiency metric, **not** an episode
termination rule. This avoids giving No-Adaptation extra physical steps and
confounding the probing comparison.

Log from day one:

- success rate before `max_env_steps`;
- undiscounted return;
- adaptation steps from `t_shift` to the first correct-rule decision;
- total cost (`env_steps + lambda * advisor_calls`);
- tool units used and tool calls by type, separate from total cost;
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
- Calibrate the shift trigger only on validation; report false alarms on the
  no-shift controls and detection delay on the shifted episodes separately.
- Pre-register the primary metric and the shift/no-shift split before the
  held-out run.
- Keep `configs/preregistered_test.yaml` at `DRAFT_NOT_FROZEN` until a real
  validation run passes the gate.

### 6.5 Ablations

- Noise level: baseline slip `0.05 / 0.10 / 0.20`.
- Signpost informativeness: full clause / partial clause / ambiguous clause.
- Advisor cost: `lambda = 1 / 2 / 5`.
- Tool budget: `1 / 3 / 5` units.
- Shift family: A1, A2, A3, B4, no-shift.
- Trigger: entropy / likelihood ratio / fixed threshold.
- Planner: A* / learned world model + MPC.
- Budget: tight / medium / loose.

---

## 7. Implementation phases

### Phase 1 — Tier A symbolic (Weeks 1-2)

- `GridConfig`, map generator, all-hypothesis solvability check.
- Entities, appearance classes, actions, stochastic transitions, soft hazards.
- Rules A1-A3 and the finite hypothesis set.
- Change-point belief update, entropy/likelihood trigger.
- Exact-belief + A* reference, `DynamicsModel` protocol, Q* only for microtasks.
- Public advisor, baselines 1-4, oracle.
- Matched physical-step and tool budgets; metrics, paired CIs, smoke script,
  tests.

### Phase 2 — Visual and tools (Weeks 2-3)

- 224x224 renderer with Set-of-Mark and HUD; illegible signpost icon.
- `INSPECT` / `READ_SIGN` semantics, range, adjacency, and costs.
- `GridPublicObservation` with image and action mask.
- Frozen-VLM prompt with one action per decision; parser, cache, permission
  checks.
- Signpost-shortcut acceptance test.
- Tier B smoke on a small map subset, run on the existing Kaggle path. Local
  CI uses the stub backend and parser/permission tests only; a real-VLM run is
  a remote gate, not a local requirement.

### Phase 3 — Learned planner (after Tier A is validated)

- Learned world model behind `DynamicsModel`.
- MPC/beam search.
- Compare against A* reference.
- Do not start this phase until Phase 1 metrics are stable.

---

## 8. Acceptance criteria

The addition is acceptable only when:

1. The existing 217 tests still pass unchanged.
2. New tests cover all-hypothesis map solvability, stochastic transitions,
   soft hazards, partial signposts, tool range/adjacency/costs, budget
   accounting, one-action-per-decision semantics, change-point belief updates,
   action masks, renderer determinism, and the agent/evaluator permission
   boundary.
3. A no-shift control produces a measurable false-alarm rate, and a
   deterministic-residual detector is explicitly shown to be inadequate.
4. A scripted signpost-only policy fails or underperforms on held-out maps,
   even if it reads every side of the sign.
5. `INSPECT` improves outcomes under the matched 3-unit tool budget; an
   unlimited-tool ablation shows that the budget, not free information,
   creates the trade-off.
6. Tier A smoke runs at least 100 maps x 5 seeds for every baseline with
   matched physical-step and tool budgets.
7. Tier B smoke produces valid JSON with a real frozen VLM and no privileged
   leakage on the Kaggle path; local CI only needs the stub, parser, and
   permission tests.
8. No generated result is presented as a scientific finding before
   preregistration and a real validation run.

---

## 9. Builder calls (defaults, not user decisions)

These are sensible defaults; the human can override them before
implementation:

- default grid `7 x 7`, stress `9 x 9`;
- public appearance, hidden semantics; a shift never repaints the map;
- one action per decision; tools return a new observation without advancing
  the environment step counter;
- change-point belief over `(hypothesis, shift time)`; public window and hazard
  rate, hidden exact time;
- exact-belief + A* reference; Q* only for microtasks; no full POMDP solver;
- all-hypothesis solvability for Tier A maps;
- public advisor baseline; privileged oracle advisor only as an upper bound;
- baseline slip `0.10`, slippery slip `0.80`;
- `INSPECT` error `0.05`, range 2, cost 2 tool units; `READ_SIGN` cost 1 tool
  unit, adjacency required, one clause per call;
- tool budget 3 units per episode;
- `lambda = 2` per advisor (VLM) call; tool units are priced by the tool
  budget, not by `lambda`;
- soft hazard damage `-2` and one-step recovery in Tier A;
- one shift per episode, sampled uniformly in a public window;
- A1/A2/A3 use the existing `red`/`blue`/`green` colour vocabulary where
  possible; B4 adds compositional clause ids;
- separate `GRID_ACTIONS` and grid validators; do not widen `MACRO_ACTIONS`;
- Tier B real-VLM smoke is a remote Kaggle gate, not local CI.

## 10. Approved decisions

The three scope questions were answered on 2026-11-23:

1. **Tier B scope:** Tier A CPU environment plus Tier B renderer, tool,
   prompt/parser, and stub tests. The real frozen-VLM run is a remote Kaggle
   gate, not part of the local build.
2. **Default map:** `7 x 7`; `9 x 9` remains a stress condition.
3. **Advisor cost:** `lambda = 2` per advisor (VLM) call. Tool units are priced
   separately by the 3-unit budget.

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

---

## 12. Review resolutions

The independent builder review raised eleven ambiguities. They are resolved
here as builder calls; the three scope questions in section 10 were answered
by the human on 2026-11-23.

| Review question | Resolution |
|---|---|
| What is public vs. hidden, and what changes visually on a shift? | Geometry and tile appearance are public; semantic roles, key-door mapping, shift time, and noise are hidden. A shift never repaints the map. `INSPECT` reveals hidden semantics. |
| What is one decision? | One action per decision, either physical or tool, never both. A tool returns a new observation without advancing the env step counter. `INSPECT` has a target, range 2, cost 2; `READ_SIGN` requires adjacency, cost 1. |
| Identification or base-then-shift? | Both. Identification samples `r*` at `t=0`; change detection starts at `r0` and shifts. No-shift controls are change-detection episodes with no shift sampled and an identical agent-facing prompt. |
| Static or change-point belief? | Change-point hazard filter over `(h, tau)` for change detection; static filter for identification. Window and hazard rate are public; exact time is hidden. |
| Objective and budget? | Goal `+10`, step `-0.05`, soft hazard `-2`, lethal trap `-10`; max 60/90 physical steps. All methods share the physical-step cap and the 3-unit tool budget. `TotalCost = env_steps + lambda * advisor_calls`; tools are priced by the separate tool budget. `TotalCost` is reported, not used to truncate. |
| Which Tier A reference? | Exact belief + A* point-estimate reference; Q* only for microtasks. No full POMDP solver. |
| Solvability under which rules? | Every hypothesis in the finite Tier A set; reject and log unwinnable maps. |
| What is the advisor? | A public advisor that returns one action and no rule; privileged oracle advisor is a separate upper bound. |
| Does `READ_SIGN` add information? | The sign text is illegible in the frame; `READ_SIGN` is the only way to read it. One clause per side, adjacency required, repeated reads cost; at least one clause must come from transition evidence. |
| Union `MACRO_ACTIONS`? | No. Keep `MACRO_ACTIONS` unchanged; add `GRID_ACTIONS` and separate grid validators. |
| Where does the real-VLM smoke run? | Kaggle remote gate. Local CI uses the stub, parser, and permission tests. |
