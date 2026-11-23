# RuleShift-Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RuleShift-Grid as an additive grid benchmark: a CPU Tier A environment with stochastic dynamics, partial epistemic tools, a change-point belief, and matched-budget baselines, plus a Tier B 224x224 renderer and frozen-VLM scaffold.

**Architecture:** Public contracts, rule hypotheses, and the planner live in `public_api/grid_*` so agents never import `envs` or `evaluator`. The simulator lives in `envs/grid/`, the privileged runner and metrics in `evaluator/grid_*`, and agent-side belief/selector/baselines in `agents/grid_*`. One engine produces both the symbolic Tier A view and the 224x224 Tier B view.

**Tech Stack:** Python >=3.11,<3.14; `gymnasium`, `numpy`, `Pillow`, `PyYAML`, `pytest` from the `dev` extra. No `torch` in the dev environment; the real frozen VLM runs only on Kaggle.

**Spec:** `docs/superpowers/specs/2026-11-23-ruleshift-grid-design.md`

## Global Constraints

- Do not replace, rename, or weaken Family A/B. Their tests and public contracts stay unchanged.
- `MACRO_ACTIONS` and `MACRO_ACTIONS_FROZEN` stay exactly as they are. Grid actions live in a separate `GRID_ACTIONS` set.
- `configs/preregistered_test.yaml` stays `DRAFT_NOT_FROZEN`; no held-out result is reported before a real validation run.
- `python -m pytest -q` must stay green after every task.
- CPU only. Never run a local GPU job. Real VLM work is a remote Kaggle gate.
- No hidden state in public observations, prompts, caches, or renders. The permissions scan must pass.
- One decision is one action: physical or tool, never both. Tool actions do not advance `env_steps`.
- Tool budget is 3 units per episode: `INSPECT` costs 2, `READ_SIGN` costs 1.
- `TotalCost = env_steps + 2 * advisor_calls`. Tool units are reported separately; they are priced by the tool budget.
- Default map is `7 x 7`; `9 x 9` is a stress condition. Reward: goal `+10`, step `-0.05`, soft hazard `-2`, lethal trap `-10` in Tier B.
- Soft hazard in Tier A never terminates the episode. Lethal traps exist only in Tier B.
- Every Tier A map is solvable under every hypothesis in the finite set.
- Every sign side is partial: no single side, and no full set of sides, identifies the active rule without transition evidence.
- Do not hand-edit generated reports. Regenerate them from scripts.

## Review Focus

These are the failure modes most likely to bite a user; each has a test in the owning task.

1. **A shift leaks through the render or HUD.** A user expects the map's appearance to stay fixed and only its meaning to change. Test: `test_renderer_is_rule_independent` (Task 6) and `test_public_view_has_no_hidden_fields` (Task 4).
2. **A tool call advances the environment or bypasses the budget.** A user expects `INSPECT`/`READ_SIGN` to cost tool units only. Test: `test_inspect_does_not_advance_steps` and `test_tool_budget_never_negative` (Task 5).
3. **Aleatoric noise is mistaken for a shift.** A user expects a no-shift control to produce a measurable but low false-alarm rate. Test: `test_no_shift_false_alarm_rate_is_bounded` (Task 7).
4. **One signpost side solves the episode.** A user expects partial information, not an oracle. Test: `test_every_sign_side_leaves_at_least_two_hypotheses` and `test_full_sign_set_still_needs_transition_evidence` (Tasks 2 and 12).
5. **A map is solvable at base but unwinnable after a shift.** A user expects every Tier A map to be solvable under every hypothesis. Test: `test_generated_map_is_solvable_under_every_hypothesis` (Task 3).

## File Map

| Path | Responsibility |
|---|---|
| `src/ruleshift/public_api/grid_schemas.py` | Grid actions, observation, decision record, validators |
| `src/ruleshift/public_api/grid_rules.py` | Public hypothesis set, public map, transition semantics |
| `src/ruleshift/public_api/grid_planner.py` | `DynamicsModel` protocol, hypothesis model, A* |
| `src/ruleshift/envs/grid/config.py` | `GridConfig`, YAML loading |
| `src/ruleshift/envs/grid/entities.py` | `CellType`, `Entity`, `GridMap`, `GridState`, `GridScenario` |
| `src/ruleshift/envs/grid/mapgen.py` | Seeded map generation and all-hypothesis solvability |
| `src/ruleshift/envs/grid/rules.py` | Shift schedule and active-rule application |
| `src/ruleshift/envs/grid/noise.py` | Seeded slip, sensor, and sign draws |
| `src/ruleshift/envs/grid/grid_env.py` | `GridEnv(gym.Env)` |
| `src/ruleshift/envs/grid/observations.py` | Public symbolic/visual observation assembly |
| `src/ruleshift/envs/grid/render.py` | 224x224 RGB, Set-of-Mark, HUD |
| `src/ruleshift/envs/grid/tools.py` | `INSPECT` / `READ_SIGN` semantics |
| `src/ruleshift/agents/grid_belief.py` | Change-point belief filter |
| `src/ruleshift/agents/grid_selector.py` | Info-gain selector and public advisor |
| `src/ruleshift/agents/grid_baselines.py` | No-adaptation and random-probing policies |
| `src/ruleshift/agents/grid_vlm.py` | Grid prompt, parser, cache, policy |
| `src/ruleshift/evaluator/grid_trace.py` | Privileged `GridEvaluatorTrace` |
| `src/ruleshift/evaluator/grid_reference.py` | Exact belief and Q* microtasks |
| `src/ruleshift/evaluator/grid_metrics.py` | Metrics and paired CIs |
| `src/ruleshift/evaluator/grid_runner.py` | Closed-loop runner and oracle policy |
| `src/ruleshift/prompts/grid_v1.txt` | Public-only grid prompt |
| `configs/grid_dev.yaml` | Default grid configuration |
| `scripts/run_grid_smoke.py` | Tier A smoke run |
| `scripts/run_grid_audit.py` | Leakage, solvability, and shortcut audit |
| `scripts/run_grid_real_pilot.py` | Remote Kaggle frozen-VLM pilot |
| `kaggle/kernel/run_grid.py` | Remote grid-pilot kernel |
| `tests/test_grid_scripts.py` | Smoke, audit, and pilot-CLI guard tests |
| `tests/test_grid_permissions.py` | Agent/evaluator boundary tests |
| `tests/test_grid_signpost_shortcut.py` | Signpost-only shortcut acceptance test |

Before Task 1, create an isolated branch or worktree. Use `superpowers:using-git-worktrees` if available; otherwise run `git switch -c feat/ruleshift-grid` from a clean `main`.

---

### Task 1: Grid Public Contracts

**Files:**
- Create: `src/ruleshift/public_api/grid_schemas.py`
- Create: `tests/test_grid_schemas.py`
- Modify: `src/ruleshift/public_api/__init__.py`
- Do not modify: `src/ruleshift/public_api/schemas.py`

**Interfaces:**
- Consumes: `CandidateRule` from `ruleshift.public_api.interface`; `find_forbidden_fields` from `ruleshift.permissions`.
- Produces:
  - `GRID_PHYSICAL_ACTIONS`, `GRID_TOOL_ACTIONS`, `GRID_ACTIONS`, `GRID_ACTIONS_FROZEN`, `GRID_RULE_IDS`, `GRID_PUBLIC_OBSERVATION_FIELDS`.
  - `GridAction(kind: str, target: tuple[int, int] | None = None)`.
  - `GridPublicObservation` with `to_payload() -> dict`.
  - `GridDecisionRecord` with `to_row() -> dict`.
  - `validate_grid_action(action, action_mask, grid_size) -> list[str]`.
  - `validate_grid_vlm_output(payload) -> list[str]`.
  - `validate_grid_public_observation(payload) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_schemas.py
from ruleshift.public_api.grid_schemas import (
    GRID_ACTIONS, GRID_ACTIONS_FROZEN, GRID_RULE_IDS,
    GridAction, validate_grid_action, validate_grid_vlm_output,
)
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN


def test_grid_actions_are_disjoint_from_macro_actions():
    assert not (set(GRID_ACTIONS) & MACRO_ACTIONS_FROZEN)


def test_inspect_requires_a_target():
    errors = validate_grid_action(GridAction("INSPECT"), ("INSPECT",), (7, 7))
    assert errors


def test_inspect_accepts_an_in_range_target():
    errors = validate_grid_action(GridAction("INSPECT", (3, 4)), ("INSPECT",), (7, 7))
    assert errors == []


def test_valid_grid_vlm_output():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "tool_target": None,
        "confidence": 0.5,
    }
    assert validate_grid_vlm_output(payload) == []


def test_grid_vlm_output_rejects_privileged_field():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "shift_time": 3,
    }
    errors = validate_grid_vlm_output(payload)
    assert any("privileged" in error for error in errors)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.public_api.grid_schemas`.

- [ ] **Step 3: Implement `grid_schemas.py`**

Use the same style as `public_api/schemas.py`: frozen dataclasses, validators returning a list of error strings, no exceptions for bad payloads.

Define the constants exactly as in the spec. `GRID_RULE_IDS` is the tuple of ids from Task 2; use the literal tuple here and add a cross-check test in Task 2.

`GridPublicObservation` fields:

```python
observation_id: str
public_obs_hash: str
public_history_hash: str
public_history: str
public_text: str
available_actions: tuple[str, ...]
candidate_rules: tuple[CandidateRule, ...]
public_costs: Mapping[str, float]
public_map: Any          # PublicGridMap from grid_rules; avoid a circular import
agent_position: tuple[int, int]
health: int
inventory: tuple[str, ...]
door_open: bool
steps: int
action_mask: tuple[str, ...]
tool_budget_remaining: int
image: Any | None = None
image_hash: str = ""
```

`to_payload()` returns exactly `GRID_PUBLIC_OBSERVATION_FIELDS`; serialize `public_map` with `.to_payload()` and use `image_hash`, never the raw image.

`GridDecisionRecord` reuses the existing `DecisionRecord` field names and adds `tool_calls_used`, `tool_cost`, `advisor_calls_used`, `information_gain_bits`, `env_steps_elapsed`, `success`.

`validate_grid_vlm_output` rules:
- Top-level keys must be a subset of `{"hypothesis_probabilities", "action", "tool_target", "predicted_outcomes", "confidence"}`.
- `hypothesis_probabilities` keys must equal `GRID_RULE_IDS` and sum to 1 within `1e-6`.
- `action` must be in `GRID_ACTIONS_FROZEN`.
- `tool_target` is required only for `INSPECT`; it must be a two-integer pair if present.
- `confidence`, if present, must be a finite number in `[0, 1]`.
- `find_forbidden_fields(payload)` must be empty.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all existing tests plus the new ones pass.

```bash
git add src/ruleshift/public_api/grid_schemas.py src/ruleshift/public_api/__init__.py tests/test_grid_schemas.py
git commit -m "feat: add grid public contracts and validators"
```

---

### Task 2: Public Rule Hypotheses and Public Map

**Files:**
- Create: `src/ruleshift/public_api/grid_rules.py`
- Create: `tests/test_grid_rules.py`
- Modify: `src/ruleshift/public_api/__init__.py`

**Interfaces:**
- Consumes: nothing from the simulator. This module must be importable without `numpy`, `gymnasium`, or the environment.
- Produces:
  - `Coord = tuple[int, int]`.
  - `PublicGridMap(width, height, walls, start, markers, key, door, signpost)` with `to_payload() -> dict`.
  - `GridRule(rule_id, description, move_map, slip_prob, slide_two_prob, goal_color, hazard_color, safe_color, key_color, door_color, sign_clauses)`.
  - `BASE_RULE`, `GRID_HYPOTHESES`, `RULE_BY_ID`.
  - `movement_outcomes(rule, position, action, public_map) -> dict[Coord, float]`.
  - `cell_role(rule, color) -> str` (`"goal"`, `"hazard"`, or `"safe"`).
  - `key_opens(rule, key_color, door_color) -> bool`.
  - `sign_clause(rule, side) -> str`.
  - `rules_consistent_with_clause(side, clause) -> tuple[str, ...]`.
  - `rules_consistent_with_all_clauses(clauses) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_rules.py
from dataclasses import replace
from ruleshift.public_api.grid_rules import (
    BASE_RULE, GRID_HYPOTHESES, RULE_BY_ID, PublicGridMap,
    cell_role, key_opens, movement_outcomes,
    rules_consistent_with_all_clauses, rules_consistent_with_clause, sign_clause,
)
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS

MAP = PublicGridMap(
    width=5, height=5, walls=frozenset(),
    start=(1, 1),
    markers={"red": (3, 1), "blue": (3, 3), "green": (1, 3)},
    key=(0, 4), door=(4, 4), signpost=(2, 2),
)


def test_hypotheses_cover_every_declared_rule_id():
    assert tuple(rule.rule_id for rule in GRID_HYPOTHESES) == GRID_RULE_IDS


def test_base_rule_identity_movement():
    rule = replace(BASE_RULE, slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", MAP) == {(1, 0): 1.0}


def test_remap_swap_reverses_vertical():
    rule = replace(RULE_BY_ID["remap_swap"], slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", MAP) == {(1, 2): 1.0}


def test_slippery_distribution_sums_to_one():
    outcomes = movement_outcomes(RULE_BY_ID["slippery"], (2, 2), "RIGHT", MAP)
    assert abs(sum(outcomes.values()) - 1.0) < 1e-9
    assert len(outcomes) >= 2


def test_blocked_move_collapses_to_current_cell():
    blocked = replace(MAP, walls=frozenset({(1, 0)}))
    rule = replace(BASE_RULE, slip_prob=0.0)
    assert movement_outcomes(rule, (1, 1), "UP", blocked) == {(1, 1): 1.0}


def test_every_sign_side_leaves_at_least_two_hypotheses():
    for side in range(3):
        for rule in GRID_HYPOTHESES:
            clause = sign_clause(rule, side)
            assert len(rules_consistent_with_clause(side, clause)) >= 2


def test_full_sign_set_still_needs_transition_evidence():
    for rule in GRID_HYPOTHESES:
        clauses = tuple(sign_clause(rule, side) for side in range(3))
        assert len(rules_consistent_with_all_clauses(clauses)) >= 2


def test_key_opens_green_door():
    assert key_opens(BASE_RULE, "green", "green")
    assert not key_opens(BASE_RULE, "blue", "green")


def test_cell_roles_are_rule_dependent():
    assert cell_role(BASE_RULE, "red") == "goal"
    assert cell_role(RULE_BY_ID["hazard_goal_swap"], "red") == "hazard"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.public_api.grid_rules`.

- [ ] **Step 3: Implement `grid_rules.py`**

Use these six hypotheses and clauses exactly; the partiality tests depend on the sharing pattern.

```python
BASE_RULE = GridRule(
    rule_id="base",
    description="Movement follows the chosen direction; the red tile is the goal.",
    move_map={"UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"},
    slip_prob=0.10, slide_two_prob=0.0,
    goal_color="red", hazard_color="blue", safe_color="green",
    key_color="green", door_color="green",
    sign_clauses=(
        "The red tile is not the hazard.",
        "Movement may slip.",
        "The green key opens the green door.",
    ),
)
```

Add `remap_swap` (UP<->DOWN, LEFT<->RIGHT), `remap_rotate` (UP->RIGHT, RIGHT->DOWN, DOWN->LEFT, LEFT->UP), `slippery` (identity, `slip_prob=0.80`, `slide_two_prob=0.5`), `hazard_goal_swap` (identity, goal `"blue"`, hazard `"red"`), and `sign_composition` (identity, goal `"green"`, hazard `"red"`, safe `"blue"`). Use the clause sharing below so no side and no full set identifies a rule:

| Rule | Side 0 | Side 1 |
|---|---|---|
| base, slippery | The red tile is not the hazard. | Movement may slip. |
| remap_swap, remap_rotate | The red tile is not the hazard. | Movement may be remapped. |
| hazard_goal_swap, sign_composition | The red tile is not the goal. | Movement may slip. |

All rules share side 2: `"The green key opens the green door."`

`movement_outcomes` algorithm:
1. Map the intended action through `rule.move_map` to the actual direction.
2. Build the outcome distribution: intended with `1 - slip_prob`; two-cell slide with `slip_prob * slide_two_prob`; each perpendicular with `slip_prob * (1 - slide_two_prob) / 2`.
3. Resolve each move against `public_map.walls` and the grid bounds. A blocked one-cell move stays put. A blocked two-cell slide stops on the first cell; if the first cell is blocked, it stays put.
4. Aggregate equal positions and return the distribution.

`PublicGridMap.to_payload()` returns a JSON-serializable dict with sorted wall coordinates and list coordinates for start/markers/key/door/signpost.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_rules.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/public_api/grid_rules.py src/ruleshift/public_api/__init__.py tests/test_grid_rules.py
git commit -m "feat: add public grid rule hypotheses and map geometry"
```

---

### Task 3: Grid Config and Map Generator

**Files:**
- Create: `src/ruleshift/envs/grid/__init__.py`
- Create: `src/ruleshift/envs/grid/config.py`
- Create: `src/ruleshift/envs/grid/entities.py`
- Create: `src/ruleshift/envs/grid/mapgen.py`
- Create: `configs/grid_dev.yaml`
- Create: `tests/test_grid_mapgen.py`

**Interfaces:**
- Consumes: `PublicGridMap`, `GridRule`, `GRID_HYPOTHESES` from `public_api.grid_rules`.
- Produces:
  - `GridConfig` dataclass and `load_grid_config(path: str | Path) -> GridConfig`.
  - `GridMap(public: PublicGridMap, map_seed: int, door_required: bool = False)`.
  - `MapGenResult(map: GridMap, attempts: int)`.
  - `generate_solvable_map(seed: int, config: GridConfig, rules=GRID_HYPOTHESES, max_attempts=200) -> MapGenResult`.
  - `is_solvable(public_map: PublicGridMap, rule: GridRule, door_required: bool = False) -> bool`.
  - `all_hypotheses_solvable(public_map, rules, door_required=False) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_mapgen.py
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.mapgen import generate_solvable_map, is_solvable
from ruleshift.public_api.grid_rules import GRID_HYPOTHESES, PublicGridMap


def test_generated_map_has_expected_dimensions():
    result = generate_solvable_map(0, GridConfig(), GRID_HYPOTHESES)
    public = result.map.public
    assert (public.width, public.height) == (7, 7)
    assert public.start not in public.walls
    assert set(public.markers) == {"red", "blue", "green"}
    assert result.attempts >= 1


def test_generated_map_is_solvable_under_every_hypothesis():
    for seed in range(20):
        result = generate_solvable_map(seed, GridConfig(), GRID_HYPOTHESES)
        for rule in GRID_HYPOTHESES:
            assert is_solvable(result.map.public, rule)


def test_unsolvable_map_is_rejected():
    public = PublicGridMap(
        width=5, height=5,
        walls=frozenset({(3, 0), (2, 1), (4, 1), (3, 2)}),
        start=(0, 0),
        markers={"red": (3, 1), "blue": (1, 3), "green": (0, 4)},
        key=None, door=None, signpost=None,
    )
    rule = GRID_HYPOTHESES[0]
    assert not is_solvable(public, rule)


def test_generator_is_deterministic():
    first = generate_solvable_map(7, GridConfig(), GRID_HYPOTHESES)
    second = generate_solvable_map(7, GridConfig(), GRID_HYPOTHESES)
    assert first.map.public.to_payload() == second.map.public.to_payload()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_mapgen.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.envs.grid`.

- [ ] **Step 3: Implement config, entities, and mapgen**

`GridConfig` defaults: `width=7`, `height=7`, `wall_density=0.18`, `max_env_steps=60`, `tool_budget=3`, `inspect_cost=2`, `read_sign_cost=1`, `inspect_range=2`, `inspect_error=0.05`, `baseline_slip=0.10`, `slippery_slip=0.80`, `shift_probability=0.35`, `shift_window=(10, 40)`, `soft_hazard_damage=-2.0`, `soft_hazard_recovery_steps=1`, `goal_reward=10.0`, `step_cost=-0.05`, `max_health=3`, `lethal_trap=False`, `door_required=False`, `advisor_cost=2`, `max_decisions=200`. There is no `max_damage` field: health is an integer that floors at zero in Tier A. `baseline_slip` and `slippery_slip` are the simulator's source of truth; `GridEnv` applies them to the public rule objects at reset (see Task 4).

`load_grid_config` reads YAML with `yaml.safe_load`, keeps only keys in `GridConfig.__dataclass_fields__`, and converts `shift_window` to a tuple. Unknown keys such as `render_size` and `observation_mode` are informational in v1 and ignored.

`generate_solvable_map`:
1. Draw a candidate map from `random.Random(seed + attempt)`.
2. Put walls on the border and sample interior wall cells at `wall_density`.
3. Choose start, the three marker cells, key, door, and signpost from open cells with a minimum Manhattan separation.
4. Keep the candidate only if `all_hypotheses_solvable` is true.
5. Return `MapGenResult(map=GridMap(public, seed, config.door_required), attempts=attempt + 1)`.
6. Raise `RuntimeError` after `max_attempts`.

`is_solvable`:
- BFS from `start` to `public.markers[rule.goal_color]`.
- Treat walls and the rule's hazard marker as blocked.
- If `door_required`, require the key to be reachable and the door to be openable; v1 keeps the key/door mapping constant, so a path through the door is allowed after the key.
- The other markers are walkable but may be hazards under another rule; the per-rule check handles that.

Write `configs/grid_dev.yaml` with the defaults above plus `render_size: 224` and `observation_mode: symbolic`.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_mapgen.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/envs/grid/ configs/grid_dev.yaml tests/test_grid_mapgen.py
git commit -m "feat: add grid config and all-hypothesis map generator"
```

---

### Task 4: Grid Environment Physical Dynamics and Shifts

**Files:**
- Create: `src/ruleshift/envs/grid/noise.py`
- Create: `src/ruleshift/envs/grid/rules.py`
- Create: `src/ruleshift/envs/grid/grid_env.py`
- Create: `tests/test_grid_env.py`
- Create: `tests/test_grid_noise.py`

**Interfaces:**
- Consumes: `GridConfig`, `GridMap`, `PublicGridMap`, `GridRule`, `movement_outcomes`, `cell_role`, `SemanticRNG`.
- Produces:
  - `GridScenario(scenario_id, scenario_type, initial_rule, shift_rule, shift_time, map_seed, run_seed=0)`. `run_seed` drives noise draws; `map_seed` drives the map. Keeping them separate prevents the map from changing when only the seed changes.
  - `sample_scenario(seed: int, config: GridConfig, scenario_type: str | None = None) -> GridScenario`.
  - `GridState(position, health, inventory, steps, tool_budget, recovery, door_open, terminated, truncated, success, last_outcome)`.
  - `GridEnv(config=None, render_mode="rgb_array", observation_mode="symbolic")` with `reset`, `step`, `render`, `public_view`, `evaluator_state`.
  - `GridEnv.reset` accepts test/evaluator options `map: PublicGridMap`, `scenario: GridScenario`, and individual scenario overrides.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_env.py
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction


def make_env(public_map, *, scenario=None, **config_overrides):
    # Tests assert exact positions, so slip is off unless a test opts in.
    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0, **config_overrides)
    env = GridEnv(config)
    scenario = scenario or GridScenario(
        scenario_id="test", scenario_type="identification",
        initial_rule="base", shift_rule=None, shift_time=None, map_seed=0,
    )
    env.reset(seed=0, options={"map": public_map, "scenario": scenario})
    return env


def test_reset_returns_public_observation():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    view = env.public_view()
    assert view.public_map.start == (1, 1)
    assert find_forbidden_fields(view.to_payload()) == []


def test_physical_step_increments_env_steps():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, _, _, _, info = env.step(GridAction("DOWN"))
    assert info["env_steps"] == 1


def test_wall_blocks_movement():
    public = PublicGridMap(5, 5, frozenset({(1, 0)}), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    env.step(GridAction("UP"))
    assert env.evaluator_state()["position"] == (1, 1)


def test_soft_hazard_damages_and_recovers_without_termination():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (3, 3), "blue": (1, 2), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, reward, terminated, _, _ = env.step(GridAction("DOWN"))
    assert reward <= -2.0
    assert env.evaluator_state()["health"] == 1   # max_health=3, damage=2
    assert terminated is False
    env.step(GridAction("DOWN"))
    assert env.evaluator_state()["position"] == (1, 2)
    assert env.evaluator_state()["recovery"] == 0


def test_goal_terminates_with_reward():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 2), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    env = make_env(public)
    _, reward, terminated, _, info = env.step(GridAction("DOWN"))
    assert terminated is True
    assert info["success"] is True
    assert reward >= 9.9


def test_key_and_door():
    public = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (1, 4), "blue": (3, 3), "green": (3, 1)},
        key=(1, 2), door=(1, 3), signpost=None,
    )
    env = make_env(public)
    env.step(GridAction("DOWN"))       # onto the key
    env.step(GridAction("INTERACT"))   # pick it up
    env.step(GridAction("INTERACT"))   # open the adjacent door
    assert env.evaluator_state()["door_open"] is True
    env.step(GridAction("DOWN"))       # onto the door cell
    _, _, terminated, _, _ = env.step(GridAction("DOWN"))
    assert terminated is True


def test_shift_changes_rule_at_shift_time():
    public = PublicGridMap(7, 7, frozenset(), (3, 3),
                           {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                           None, None, None)
    scenario = GridScenario(
        scenario_id="shift", scenario_type="change_detection",
        initial_rule="base", shift_rule="remap_swap", shift_time=2, map_seed=0,
    )
    env = make_env(public, scenario=scenario)
    assert env.evaluator_state()["true_rule"] == "base"
    env.step(GridAction("UP"))
    env.step(GridAction("DOWN"))
    assert env.evaluator_state()["true_rule"] == "remap_swap"


def test_truncates_at_max_env_steps():
    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (3, 3), "blue": (3, 1), "green": (1, 3)},
                           None, None, None)
    env = make_env(public, max_env_steps=3)
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(GridAction("RIGHT"))
    assert truncated is True
    assert terminated is False


def test_slip_is_deterministic_for_seed():
    public = PublicGridMap(7, 7, frozenset(), (3, 3),
                           {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                           None, None, None)
    scenario = GridScenario("slip", "identification", "slippery", None, None, 0)
    first = make_env(public, scenario=scenario, slippery_slip=0.8)
    second = make_env(public, scenario=scenario, slippery_slip=0.8)
    for _ in range(10):
        first.step(GridAction("RIGHT"))
        second.step(GridAction("RIGHT"))
    assert first.evaluator_state()["position"] == second.evaluator_state()["position"]
```

```python
# tests/test_grid_noise.py
from ruleshift.envs.events import SemanticRNG


def test_semantic_rng_is_deterministic():
    a = SemanticRNG(11, "scenario")
    b = SemanticRNG(11, "scenario")
    assert [a.draw("move", step) for step in range(5)] == [
        b.draw("move", step) for step in range(5)
    ]


def test_different_seeds_differ():
    assert SemanticRNG(11, "scenario").draw("move", 0) != SemanticRNG(
        12, "scenario"
    ).draw("move", 0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_env.py tests/test_grid_noise.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.envs.grid.grid_env`.

- [ ] **Step 3: Implement `noise.py`, `rules.py`, and `grid_env.py`**

`noise.py` is a thin wrapper around `SemanticRNG`:
- `draw_move(rng, step, attempt=0) -> float` with event `"move"`.
- `draw_shift(rng, step) -> float` with event `"shift"`.
- `draw_inspect(rng, step, attempt=0) -> float` with event `"inspect"`.
- `draw_sign(rng, step, attempt=0) -> float` with event `"sign"`.

`rules.py`:
- `sample_scenario(seed, config, scenario_type=None)`: use `random.Random(seed)` to choose the rule(s) and shift time. `identification` samples `initial_rule` uniformly and no shift. `change_detection` starts at `"base"`, samples a non-base `shift_rule`, and samples `shift_time` in `config.shift_window` clamped to `[1, max_env_steps - 1]`. `no_shift` is change detection with `shift_rule=None`, `shift_time=None`. The default type is `change_detection` with probability `config.shift_probability`, otherwise `no_shift`. Set `run_seed=seed` on the returned scenario.
- `RULE_OBJECTS = RULE_BY_ID`.

**Config-driven slip:** the public `GridRule` objects carry default probabilities
for the agent's planner, but the simulator must use `GridConfig`. Define
`_rule_object(rule_id)` in `GridEnv` as
`replace(RULE_BY_ID[rule_id], slip_prob=config.slippery_slip if rule_id == "slippery" else config.baseline_slip)`.
Use it for movement, cell roles, and key/door checks. This is why
`GridConfig(baseline_slip=0.0)` makes the environment deterministic while the
agent's public planner still reasons about the documented 0.10 baseline.

`grid_env.py`:
- `GridState` as a mutable dataclass.
- `GridEnv.__init__` sets `action_space = spaces.Dict({"kind": spaces.Discrete(len(GRID_ACTIONS)), "target": spaces.MultiDiscrete([width, height])})`. Set `observation_space` to a `spaces.Dict` with `public_text` (Text), `action_mask` (MultiBinary), `tool_budget_remaining` (Discrete), and, in visual mode, `image` (Box).
- `reset`:
  - `super().reset(seed=seed)`.
  - If `options["map"]` is a `PublicGridMap`, wrap it in `GridMap(public=..., map_seed=scenario.map_seed)`; otherwise call `generate_solvable_map(scenario.map_seed, config)` so the same map seed always yields the same map.
  - If `options["scenario"]` is a `GridScenario`, use it; otherwise `sample_scenario(seed or 0, config, options.get("scenario_type"))`.
  - Apply individual overrides for `initial_rule`, `shift_rule`, `shift_time`, `scenario_type`.
  - Initialize `GridState(position=public.start, health=config.max_health, inventory=(), steps=0, tool_budget=config.tool_budget, recovery=0, door_open=False, terminated=False, truncated=False, success=False, last_outcome="none")`.
  - Store `_history=[]`, `_shifted=False`, `_active_rule=scenario.initial_rule`.
  - Return `_obs()`, `_info()`.
- `step`:
  - Normalize dict/`GridAction`; reject invalid actions without advancing.
  - If terminated/truncated, raise `RuntimeError`.
  - If `action.kind in GRID_TOOL_ACTIONS`: call `_apply_tool` (Task 5); in this task return a rejection `last_outcome="tool_not_available"`.
  - Otherwise `_apply_physical`.
- `_apply_physical`:
  - `reward = config.step_cost`.
  - If `recovery > 0`: decrement recovery, set `last_outcome="recovering"`, increment steps, maybe shift, check truncation, return.
  - If `INTERACT`: pick up a key on the current cell, or open an adjacent closed door when the inventory contains the key color. Set `last_outcome` accordingly.
  - Else: build `blocked = public.walls | ({door} if not door_open else set()) | ({signpost} if signpost else set())`; call `movement_outcomes(rule, position, action.kind, replace(public, walls=blocked))`; sample one outcome using `draw_move(rng, step)`.
  - Apply the cell effect: goal terminates with `goal_reward` and `success=True`; hazard adds `soft_hazard_damage`, decrements health to a floor of 0, sets `recovery`; safe does nothing.
  - Increment steps, call `_maybe_shift()`, `_check_truncation()`.
- `_maybe_shift`: if `scenario.shift_time is not None` and `state.steps >= scenario.shift_time` and not `_shifted`, set `_active_rule=scenario.shift_rule`, `_shifted=True`.
- `_action_mask`: physical actions + `INTERACT`; add `INSPECT` when `tool_budget >= config.inspect_cost` and a target is within range; add `READ_SIGN` when `tool_budget >= config.read_sign_cost` and the agent is adjacent to the signpost. In this task, tools are not yet enabled, so the mask contains physical actions only; Task 5 adds them.
- `public_view`: build a `GridPublicObservation` using `public_map`, state, history, candidate rules from `GRID_RULE_IDS`, public costs (`step_cost`, `goal_reward`, `inspect_cost`, `read_sign_cost`, `advisor_cost`, `tool_budget`), action mask, and `tool_budget_remaining`. Do not include `_active_rule`.
- `evaluator_state`: return `scenario_id`, `true_rule`, `initial_rule`, `shift_time`, `scenario_type`, `map_seed`, `position`, `health`, `inventory`, `steps`, `tool_budget`, `recovery`, `door_open`, `success`, `terminated`, `truncated`.
- `render`: return `None` until Task 6.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_env.py tests/test_grid_noise.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/envs/grid/noise.py src/ruleshift/envs/grid/rules.py src/ruleshift/envs/grid/grid_env.py tests/test_grid_env.py tests/test_grid_noise.py
git commit -m "feat: add grid environment physical dynamics and rule shifts"
```

---

### Task 5: Epistemic Tools and Tool Budget

**Files:**
- Create: `src/ruleshift/envs/grid/tools.py`
- Modify: `src/ruleshift/envs/grid/grid_env.py`
- Create: `tests/test_grid_tools.py`

**Interfaces:**
- Consumes: `GridConfig`, `PublicGridMap`, `GridRule`, `cell_role`, `sign_clause`.
- Produces:
  - `ToolResult(kind, target, value, cost, rejected, reason)` with `to_payload()`.
  - `tool_cost(kind: str, config: GridConfig) -> int`.
  - `inspect_cell(public_map, rule, target, draw: float, wrong_index: int, config) -> ToolResult`.
  - `read_sign(rule, side: int, config) -> ToolResult`.
  - `sign_side(position: Coord, signpost: Coord) -> int`.
  - `validate_tool_action(kind, target, position, public_map, budget, config) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_tools.py
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.tools import inspect_cell, read_sign, sign_side
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction


def make_env():
    public = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (3, 3), "blue": (1, 3), "green": (3, 1)},
        key=None, door=None, signpost=(2, 1),
    )
    env = GridEnv(GridConfig())
    env.reset(seed=0, options={"map": public})
    return env, public


def test_inspect_costs_two_units_and_does_not_advance_steps():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("INSPECT", (1, 3)))
    assert info["tool_budget_remaining"] == 1
    assert info["env_steps"] == 0


def test_inspect_rejects_out_of_range_target():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("INSPECT", (3, 3)))
    assert info["tool_budget_remaining"] == 3
    assert info["tool_rejected"] == "out_of_range"


def test_inspect_rejects_when_budget_insufficient():
    env, _ = make_env()
    env.step(GridAction("INSPECT", (1, 3)))
    _, _, _, _, info = env.step(GridAction("INSPECT", (2, 2)))
    assert info["tool_budget_remaining"] == 1
    assert info["tool_rejected"] == "insufficient_budget"


def test_inspect_is_noisy_but_truthful_without_error():
    _, public = make_env()
    truthful = inspect_cell(public, BASE_RULE, (1, 3), draw=0.99, wrong_index=0,
                            config=GridConfig())
    assert truthful.value == "hazard"
    noisy = inspect_cell(public, BASE_RULE, (1, 3), draw=0.0, wrong_index=0,
                         config=GridConfig())
    assert noisy.value != "hazard"


def test_read_sign_requires_adjacency():
    env, _ = make_env()
    env.step(GridAction("LEFT"))   # move away from the signpost at (2, 1)
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 3
    assert info["tool_rejected"] == "not_adjacent"


def test_read_sign_costs_one_unit_and_returns_one_clause():
    env, _ = make_env()
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 2
    assert info["env_steps"] == 0
    assert "last_tool_value" in info


def test_repeated_reads_cost_again():
    env, _ = make_env()
    env.step(GridAction("READ_SIGN"))
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 1


def test_tool_budget_never_negative():
    env, _ = make_env()
    env.step(GridAction("INSPECT", (1, 3)))
    env.step(GridAction("INSPECT", (2, 2)))
    env.step(GridAction("READ_SIGN"))
    _, _, _, _, info = env.step(GridAction("READ_SIGN"))
    assert info["tool_budget_remaining"] == 0
    assert info["tool_rejected"] == "insufficient_budget"


def test_sign_side_is_deterministic():
    assert sign_side((2, 1), (2, 2)) == 2   # north of the sign
    assert sign_side((3, 2), (2, 2)) == 1   # east of the sign
    assert sign_side((2, 3), (2, 2)) == 0   # south of the sign
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.envs.grid.tools`.

- [ ] **Step 3: Implement `tools.py` and wire it into `GridEnv`**

`inspect_cell`:
- Valid targets are marker cells, the key cell, and the door cell. Walls and floor are rejected.
- The true value is `cell_role(rule, marker_color)` for a marker, `"key"` for the key, and `"required_key:<color>"` for the door. In v1 every rule uses the same key/door mapping, so inspecting the door is truthful but uninformative for rule identification.
- With probability `config.inspect_error`, return a wrong role chosen from the other roles by `wrong_index`; otherwise return the true role.
- Cost is `config.inspect_cost`; `rejected=False`.

`read_sign`:
- `side` in `0, 1, 2`; returns `sign_clause(rule, side)`.
- Cost is `config.read_sign_cost`; `rejected=False`.
- The clause is partial by construction; no extra noise in v1.

`sign_side(position, signpost)`:
- `dx, dy = position[0] - signpost[0], position[1] - signpost[1]`.
- `(0, -1) -> 2` (north), `(1, 0) -> 1` (east), `(0, 1) -> 0` (south), `(-1, 0) -> 0` (west). Adjacent only; raise `ValueError` otherwise.

`validate_tool_action`:
- `INSPECT`: target is a two-int tuple, in bounds, Manhattan distance `<= config.inspect_range`, target is a marker/key/door cell, budget `>= inspect_cost`. Return reasons `"missing_target"`, `"out_of_range"`, `"invalid_target"`, `"insufficient_budget"`.
- `READ_SIGN`: signpost exists, Manhattan distance `== 1`, budget `>= read_sign_cost`. Return reasons `"no_signpost"`, `"not_adjacent"`, `"insufficient_budget"`.

`GridEnv._apply_tool`:
- Call `validate_tool_action`; if errors, set `last_outcome="rejected:<reason>"`, `info["tool_rejected"]=reason`, do not deduct or advance.
- On success, deduct the cost, set `last_outcome="inspect"` or `"read_sign"`, store `last_tool_value` in the env's info, append to history.
- Do not increment `state.steps`; do not call `_maybe_shift`.
- Update `_action_mask` to add `INSPECT` when a valid target exists and `READ_SIGN` when adjacent and affordable.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_tools.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/envs/grid/tools.py src/ruleshift/envs/grid/grid_env.py tests/test_grid_tools.py
git commit -m "feat: add INSPECT and READ_SIGN with a matched tool budget"
```

---

### Task 6: Public Observations and 224x224 Renderer

**Files:**
- Create: `src/ruleshift/envs/grid/observations.py`
- Create: `src/ruleshift/envs/grid/render.py`
- Modify: `src/ruleshift/envs/grid/grid_env.py`
- Create: `tests/test_grid_observations.py`
- Create: `tests/test_grid_render.py`

**Interfaces:**
- Consumes: `PublicGridMap`, `GridRule`, `GridState`, `GridPublicObservation`, `CandidateRule`, `digest` from `envs/observations.py`.
- Produces:
  - `build_grid_observation(public_map, state, history, last_outcome, candidate_rules, public_costs, action_mask, tool_budget_remaining, image=None) -> GridPublicObservation`.
  - `render_grid(public_map, state, size=224) -> np.ndarray`.
  - `image_digest(image: np.ndarray) -> str`.
  - `GRID_LINE_COLOR`, `HUD_COLOR` constants for tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_observations.py
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_PUBLIC_OBSERVATION_FIELDS
from ruleshift.permissions import find_forbidden_fields


def make_env(mode="symbolic"):
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode=mode)
    env.reset(seed=0, options={"map": public})
    return env


def test_symbolic_observation_contains_only_public_fields():
    view = make_env().public_view()
    payload = view.to_payload()
    assert set(payload) == set(GRID_PUBLIC_OBSERVATION_FIELDS)
    assert find_forbidden_fields(payload) == []


def test_visual_observation_has_224x224_uint8():
    image = make_env("visual").render()
    assert image.shape == (224, 224, 3)
    assert image.dtype.name == "uint8"


def test_signpost_clause_is_not_in_public_text():
    view = make_env().public_view()
    assert all(clause not in view.public_text for clause in BASE_RULE.sign_clauses)
```

```python
# tests/test_grid_render.py
import numpy as np
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.envs.grid.render import GRID_LINE_COLOR, render_grid
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GridAction


def make_env(rule="base"):
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode="visual")
    env.reset(seed=0, options={"map": public, "initial_rule": rule})
    return env


def test_renderer_is_deterministic():
    env = make_env()
    assert np.array_equal(env.render(), env.render())


def test_renderer_is_rule_independent():
    base = make_env("base").render()
    swapped = make_env("hazard_goal_swap").render()
    assert np.array_equal(base, swapped)


def test_set_of_mark_grid_is_drawn():
    image = make_env().render()
    assert np.any(np.all(image == GRID_LINE_COLOR, axis=2))


def test_hud_changes_with_public_state():
    env = make_env()
    first = env.render()
    env.step(GridAction("RIGHT"))
    second = env.render()
    assert not np.array_equal(first, second)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_observations.py tests/test_grid_render.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.envs.grid.render`.

- [ ] **Step 3: Implement `observations.py` and `render.py`**

`observations.py`:
- `build_grid_observation` assembles the `GridPublicObservation`.
- `public_text` includes `step`, `pos`, `health`, `tool`, `door`, `last`, marker coordinates, key/door/signpost coordinates. It never includes `true_rule`, `shift_time`, or the active rule.
- `public_obs_hash` uses `digest` from `envs/observations.py` over the public state and `public_map.to_payload()`.
- `public_history_hash` uses `digest(history)`.
- `candidate_rules` are `CandidateRule(rule_id, description)` for every `GRID_HYPOTHESES` entry.
- `public_costs` uses `step_cost`, `goal_reward`, `inspect_cost`, `read_sign_cost`, `advisor_cost`, `tool_budget`.
- If `image` is not None, set `image_hash=image_digest(image)`.

`render.py`:
- Use Pillow `Image`, `ImageDraw`, `ImageFont.load_default()`.
- Canvas `size x size x 3`, white background.
- `margin = 8`, `label_margin = 14`, `hud_height = 28`, `grid_px = size - 2 * margin - label_margin - hud_height`, `cell = grid_px // max(width, height)`; the grid is drawn at `(margin + label_margin, margin + label_margin)` and the HUD at `y = size - hud_height - margin`.
- Draw floor cells, walls, marker tiles, key icon, door icon, signpost icon (gray post plus white board with two short gray lines, no readable text), agent circle.
- Draw thin grid lines in `GRID_LINE_COLOR = (210, 210, 210)`.
- Draw coordinate labels `0..width-1` along the top margin and `0..height-1` along the left margin in dark gray.
- Draw the HUD at the bottom in `HUD_COLOR = (30, 30, 30)` with white text `HP={health} STEPS={steps} TOOL={tool_budget}`.
- The function takes only `public_map` and `state`; it cannot see the active rule. This is the structural guarantee against render leakage.
- `image_digest` returns `"sha256:" + hashlib.sha256(image.tobytes()).hexdigest()[:16]`.

Modify `GridEnv`:
- `public_view()` calls `build_grid_observation`, passing `render_grid(...)` when `observation_mode == "visual"`, else `None`.
- `render()` returns `self.public_view().image` when visual, else `None`.
- `_obs()` returns `public_view().to_payload()` and, in visual mode, adds `"image": view.image`.
- `observation_space` for visual mode is `spaces.Dict({"image": spaces.Box(0, 255, (224, 224, 3), np.uint8), ...})`; for symbolic mode, a `spaces.Dict` without `image`.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_observations.py tests/test_grid_render.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/envs/grid/observations.py src/ruleshift/envs/grid/render.py src/ruleshift/envs/grid/grid_env.py tests/test_grid_observations.py tests/test_grid_render.py
git commit -m "feat: add grid public observations and 224x224 renderer"
```

---

### Task 7: Public Planner and DynamicsModel Protocol

**Files:**
- Create: `src/ruleshift/public_api/grid_planner.py`
- Create: `tests/test_grid_planner.py`
- Modify: `src/ruleshift/public_api/__init__.py`

**Interfaces:**
- Consumes: `GridAction`, `PublicGridMap`, `GridRule`, `movement_outcomes`.
- Produces:
  - `GridPlannerState(position: Coord, has_key: bool, door_open: bool)`.
  - `DynamicsModel` Protocol with `predict_proba(state, action) -> dict[GridPlannerState, float]` and `sample(state, action, rng) -> GridPlannerState`.
  - `HypothesisDynamics(rule: GridRule, public_map: PublicGridMap)`.
  - `astar(start: GridPlannerState, goal: Coord, model: DynamicsModel, max_expansions=10000) -> list[GridAction]`.
  - `plan_to_goal(rule, public_map, start: Coord, has_key=False, door_open=False) -> list[GridAction]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_planner.py
from pathlib import Path
from ruleshift.public_api.grid_planner import (
    GridPlannerState, HypothesisDynamics, astar, plan_to_goal,
)
from ruleshift.public_api.grid_rules import BASE_RULE, RULE_BY_ID, PublicGridMap


def test_astar_finds_shortest_path_on_open_grid():
    public = PublicGridMap(5, 5, frozenset(), (0, 0),
                           {"red": (4, 4), "blue": (4, 0), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    plan = astar(GridPlannerState((0, 0), False, False), (4, 4), model)
    assert len(plan) == 8
    assert plan[0].kind in {"UP", "DOWN", "LEFT", "RIGHT"}


def test_astar_avoids_walls():
    # A wall column with a single gap at the bottom forces a detour; the
    # Manhattan path from (0,0) to (4,0) would be 4 steps.
    walls = frozenset({(2, y) for y in range(4)})
    public = PublicGridMap(5, 5, walls, (0, 0),
                           {"red": (4, 0), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    plan = astar(GridPlannerState((0, 0), False, False), (4, 0), model)
    assert plan
    assert len(plan) > 4


def test_astar_uses_remapped_movement():
    public = PublicGridMap(5, 5, frozenset(), (2, 2),
                           {"red": (2, 0), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    rule = RULE_BY_ID["remap_swap"]
    model = HypothesisDynamics(rule, public)
    plan = astar(GridPlannerState((2, 2), False, False), (2, 0), model)
    state = GridPlannerState((2, 2), False, False)
    for action in plan:
        state = max(model.predict_proba(state, action).items(), key=lambda kv: kv[1])[0]
    assert state.position == (2, 0)


def test_astar_collects_key_and_opens_door():
    public = PublicGridMap(
        5, 5, frozenset({(1, 0), (3, 0), (1, 2), (3, 2)}), (2, 2),
        {"red": (2, 0), "blue": (4, 4), "green": (0, 4)},
        key=(2, 3), door=(2, 1), signpost=None,
    )
    model = HypothesisDynamics(BASE_RULE, public)
    plan = plan_to_goal(BASE_RULE, public, (2, 2))
    state = GridPlannerState((2, 2), False, False)
    for action in plan:
        state = max(model.predict_proba(state, action).items(), key=lambda kv: kv[1])[0]
    assert state.position == (2, 0)


def test_astar_returns_empty_when_unreachable():
    public = PublicGridMap(5, 5, frozenset({(3, 0), (2, 1), (4, 1), (3, 2)}),
                           (0, 0), {"red": (3, 1), "blue": (4, 4), "green": (0, 4)},
                           None, None, None)
    model = HypothesisDynamics(BASE_RULE, public)
    assert astar(GridPlannerState((0, 0), False, False), (3, 1), model) == []


def test_planner_is_public_only():
    source = Path("src/ruleshift/public_api/grid_planner.py").read_text(encoding="utf-8")
    assert "ruleshift.envs" not in source
    assert "ruleshift.evaluator" not in source
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.public_api.grid_planner`.

- [ ] **Step 3: Implement `grid_planner.py`**

`HypothesisDynamics.predict_proba`:
- Build `blocked = public_map.walls | ({door} if not state.door_open else set()) | ({signpost} if signpost else set())`.
- For a move, call `movement_outcomes(rule, state.position, action.kind, replace(public_map, walls=blocked))` and map each resulting position to a `GridPlannerState` preserving `has_key` and `door_open`.
- For `INTERACT`:
  - On the key and no key: one transition to `has_key=True`.
  - Adjacent to the door, has key, door closed: one transition to `door_open=True`.
  - Otherwise: one self-loop.
- `sample` uses `random.Random` and the distribution.

`astar`:
- `heapq` priority queue of `(f, g, counter, state)`.
- `came_from` maps `(state, action)`.
- For each action, `predict_proba`; use the most likely outcome; skip self-loops; cost 1.
- Heuristic: Manhattan distance to `goal`.
- Goal test: `state.position == goal`.
- `max_expansions` guard; return `[]` if exhausted.
- Reconstruct the action list.

`plan_to_goal`: goal is `public_map.markers[rule.goal_color]`; start state from the arguments.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_planner.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/public_api/grid_planner.py src/ruleshift/public_api/__init__.py tests/test_grid_planner.py
git commit -m "feat: add public grid DynamicsModel and A* reference planner"
```

---

### Task 8: Change-Point Belief and Info-Gain Selector

**Files:**
- Create: `src/ruleshift/agents/grid_belief.py`
- Create: `src/ruleshift/agents/grid_selector.py`
- Create: `tests/test_grid_belief.py`
- Create: `tests/test_grid_selector.py`

**Interfaces:**
- Consumes: `GRID_RULE_IDS`, `GRID_HYPOTHESES`, `RULE_BY_ID`, `movement_outcomes`, `cell_role`, `plan_to_goal`, `GridAction`.
- Produces:
  - `GridBelief(hypotheses=GRID_RULE_IDS, p_shift=0.02, shift_window=(0, 60), prior=None)`.
  - `update_move(previous_position, action, observed_position, public_map, step)`.
  - `update_role(color, observed_role, step)`.
  - `update_inspect(color, observed_role, step)`.
  - `update_sign(side, clause, step)`.
  - `update_interact(outcome, step)`.
  - `copy()`, `marginal()`, `most_likely()`, `entropy()`, `shift_probability()`.
  - `InfoGainSelector(config)` with `choose(belief, public_map, *, position, inventory=(), door_open=False, tool_budget, action_mask, step, safety_threshold=None) -> GridAction`.
  - `PublicAdvisor(config)` with `advise(belief, public_map, *, position, inventory=(), door_open=False) -> GridAction`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_belief.py
import random
from ruleshift.agents.grid_belief import GridBelief
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap, movement_outcomes

MAP = PublicGridMap(7, 7, frozenset(), (3, 3),
                    {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                    None, None, None)


def test_belief_normalizes():
    belief = GridBelief()
    assert abs(sum(belief.marginal().values()) - 1.0) < 1e-9


def test_belief_updates_on_move():
    belief = GridBelief()
    belief.update_move((3, 3), "UP", (3, 2), MAP, step=0)
    assert belief.marginal()["base"] > belief.marginal()["remap_swap"]


def test_belief_updates_on_role():
    belief = GridBelief()
    belief.update_role("red", "goal", step=0)
    assert belief.marginal()["base"] > belief.marginal()["hazard_goal_swap"]


def test_belief_updates_on_inspect():
    belief = GridBelief()
    belief.update_inspect("red", "goal", step=0)
    assert belief.marginal()["base"] > belief.marginal()["hazard_goal_swap"]


def test_belief_updates_on_sign():
    belief = GridBelief()
    belief.update_sign(0, "The red tile is not the goal.", step=0)
    assert belief.marginal()["hazard_goal_swap"] > belief.marginal()["base"]


def test_no_shift_false_alarm_rate_is_bounded():
    belief = GridBelief(p_shift=0.02, shift_window=(0, 60))
    rng = random.Random(0)
    position = (3, 3)
    for step in range(30):
        outcomes = movement_outcomes(BASE_RULE, position, "RIGHT", MAP)
        draw = rng.random()
        cumulative = 0.0
        next_position = position
        for candidate, probability in sorted(outcomes.items()):
            cumulative += probability
            if draw <= cumulative:
                next_position = candidate
                break
        belief.update_move(position, "RIGHT", next_position, MAP, step)
        position = next_position
    assert belief.shift_probability() < 0.5


def test_deterministic_residual_is_not_a_shift_detector():
    belief = GridBelief(p_shift=0.02, shift_window=(0, 60))
    rng = random.Random(0)
    position = (3, 3)
    residual_fired = False
    for step in range(30):
        outcomes = movement_outcomes(BASE_RULE, position, "RIGHT", MAP)
        draw = rng.random()
        cumulative = 0.0
        next_position = position
        for candidate, probability in sorted(outcomes.items()):
            cumulative += probability
            if draw <= cumulative:
                next_position = candidate
                break
        expected = (position[0] + 1, position[1])
        residual_fired = residual_fired or next_position != expected
        belief.update_move(position, "RIGHT", next_position, MAP, step)
        position = next_position
    assert residual_fired is True
    assert belief.shift_probability() < 0.5


def test_change_point_belief_detects_shift():
    belief = GridBelief(p_shift=0.3, shift_window=(0, 60))
    for step in range(3):
        belief.update_move((3, 3), "UP", (3, 4), MAP, step=step)
    assert belief.most_likely() == "remap_swap"
    assert belief.shift_probability() > 0.5
```

```python
# tests/test_grid_selector.py
from ruleshift.agents.grid_belief import GridBelief
from ruleshift.agents.grid_selector import InfoGainSelector, PublicAdvisor
from ruleshift.envs.grid.config import GridConfig
from ruleshift.public_api.grid_rules import PublicGridMap

# No signpost: the test isolates INSPECT from the cheaper READ_SIGN probe.
MAP = PublicGridMap(5, 5, frozenset(), (1, 1),
                    {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                    None, None, None)
MASK = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN")


def test_info_gain_prefers_an_informative_tool():
    belief = GridBelief(hypotheses=("base", "hazard_goal_swap"))
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=3, action_mask=MASK, step=0,
    )
    assert action.kind == "INSPECT"
    assert action.target is not None


def test_info_gain_respects_tool_budget():
    belief = GridBelief()
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=0,
        action_mask=("UP", "DOWN", "LEFT", "RIGHT", "INTERACT"), step=0,
    )
    assert action.kind not in {"INSPECT", "READ_SIGN"}


def test_info_gain_returns_a_legal_action():
    belief = GridBelief()
    selector = InfoGainSelector(GridConfig())
    action = selector.choose(
        belief, MAP, position=(1, 1), tool_budget=3, action_mask=MASK, step=0,
    )
    assert action.kind in MASK


def test_public_advisor_returns_a_legal_action():
    belief = GridBelief()
    advisor = PublicAdvisor(GridConfig())
    action = advisor.advise(belief, MAP, position=(1, 1))
    assert action.kind in {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT"}
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_belief.py tests/test_grid_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.agents.grid_belief`.

- [ ] **Step 3: Implement the belief and selector**

`GridBelief` stores a dict `(rule_id, tau) -> probability`, where `tau` is `None` or the step at which the last shift occurred. `predict(step)` applies the hazard-rate transition when `shift_window[0] <= step <= shift_window[1]`; otherwise `p_shift=0`. The prior is uniform over the hypotheses. The initial state is `(rule, None)` for each rule.

Update methods call `predict(step)` first, then multiply by the likelihood and normalize:
- `update_move`: likelihood is `movement_outcomes(rule, previous_position, action, public_map).get(observed_position, 1e-9)`.
- `update_role`: likelihood is `1.0` if `cell_role(rule, color) == observed_role`, else `1e-6`.
- `update_inspect`: likelihood is `1 - inspect_error` on a match, `inspect_error / 2` otherwise.
- `update_sign`: likelihood is `1.0` if `rule.sign_clauses[side] == clause`, else `1e-6`.
- `update_interact`: `"key_picked"` is consistent with every rule; `"door_opened"` is consistent when `key_opens(rule, key_color, door_color)`; `"nothing"` is uninformative. Use a small epsilon for inconsistent rules.

`shift_probability` is the sum of probabilities with `tau is not None`. `marginal` sums over `tau`. `entropy` is the Shannon entropy of the marginal in bits. `copy` returns a deep copy of the dict.

`InfoGainSelector`:
- Candidate list from `action_mask`. For `INSPECT`, targets are marker cells, the key, and the door within `config.inspect_range`, excluding walls. For `READ_SIGN`, only if adjacent.
- `_expected_ig` uses `belief.copy()`, enumerates outcomes with their probabilities, updates the copy, and computes `H_before - mean(H_after)`.
- For a move, outcomes are the next positions under each rule; for a tool, outcomes are roles/clauses; for `INTERACT`, outcomes are the interaction results.
- Safety: skip a move if the belief-weighted probability that the destination is a hazard exceeds `safety_threshold` (default `1.0` in Tier A; the runner passes `0.1` in Tier B).
- Choose the highest expected information gain **per unit cost** (a physical move costs 1 environment step; `INSPECT` costs 2 tool units and `READ_SIGN` costs 1). Tie-break by lower raw cost, then by the order of the action mask. If no candidate has positive IG, return the first action of `plan_to_goal(belief.most_likely(), public_map, position)`; if that is empty, return the first legal move.

`PublicAdvisor`:
- Plan to `public_map.markers[belief.most_likely() goal color]` with `plan_to_goal`.
- Return the first action, or `INTERACT` when the plan requires the key/door.
- It never updates the belief and never reads the true rule.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_belief.py tests/test_grid_selector.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/agents/grid_belief.py src/ruleshift/agents/grid_selector.py tests/test_grid_belief.py tests/test_grid_selector.py
git commit -m "feat: add change-point belief and information-gain selector"
```

---

### Task 9: Baselines, Evaluator Runner, and Privileged Trace

**Files:**
- Create: `src/ruleshift/agents/grid_baselines.py`
- Create: `src/ruleshift/evaluator/grid_trace.py`
- Create: `src/ruleshift/evaluator/grid_runner.py`
- Create: `tests/test_grid_runner.py`

**Interfaces:**
- Consumes: `GridEnv`, `GridScenario`, `GridBelief`, `InfoGainSelector`, `PublicAdvisor`, `plan_to_goal`, `BASE_RULE`, `RULE_BY_ID`, `GRID_RULE_IDS`.
- Produces:
  - `GridPolicy` Protocol with `reset(observation, info)`, `choose(observation, info) -> GridAction`, `record(action, observation, reward, info)`, `most_likely_rule() -> str`, `shift_detected() -> bool`.
  - `NoAdaptationPolicy(config)`, `PublicAdvisorPolicy(config)`, `RandomProbePolicy(config)`, `InfoGainPolicy(config)` in `agents/grid_baselines.py`.
  - `OraclePolicy(env, config)` in `evaluator/grid_runner.py`.
  - `GridEpisode` and `GridEvaluatorTrace` dataclasses.
  - `run_grid_episode(env, policy, config, scenario, observation, info) -> GridEpisode`.
  - `build_grid_scenarios(n_maps: int, seeds: int, config: GridConfig) -> tuple[GridScenario, ...]`.
  - `run_grid_benchmark(methods, scenarios, config) -> list[GridEpisode]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_runner.py
from ruleshift.agents.grid_baselines import InfoGainPolicy, NoAdaptationPolicy
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import OraclePolicy, run_grid_episode
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS


def scenario():
    return GridScenario("runner-test", "change_detection", "base",
                        "remap_swap", 3, 0)


def env():
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(4, 3),
    )
    env = GridEnv(GridConfig(baseline_slip=0.0, slippery_slip=0.0))
    observation, info = env.reset(
        seed=0, options={"map": public, "scenario": scenario()}
    )
    return env, observation, info


def run(policy):
    env, observation, info = env()
    return run_grid_episode(env, policy, GridConfig(), scenario(), observation, info)


def test_no_adaptation_keeps_the_base_rule():
    policy = NoAdaptationPolicy(GridConfig())
    episode = run(policy)
    assert policy.most_likely_rule() == "base"


def test_info_gain_uses_a_tool_when_informative():
    episode = run(InfoGainPolicy(GridConfig()))
    assert episode.tool_units > 0


def test_total_cost_formula():
    config = GridConfig()
    episode = run(InfoGainPolicy(config))
    assert episode.total_cost == episode.env_steps + config.advisor_cost * episode.advisor_calls


def test_budgets_are_matched():
    config = GridConfig()
    episode = run(InfoGainPolicy(config))
    assert episode.env_steps <= config.max_env_steps
    assert episode.tool_units <= config.tool_budget


def test_public_rows_have_no_privileged_fields():
    episode = run(InfoGainPolicy(GridConfig()))
    for row in episode.public_rows:
        assert find_forbidden_fields(row) == []


def test_evaluator_trace_has_the_true_rule():
    episode = run(InfoGainPolicy(GridConfig()))
    assert episode.evaluator_rows[0]["true_rule"] in GRID_RULE_IDS


def test_oracle_is_an_upper_bound_on_a_deterministic_map():
    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0)
    no_adapt = run(NoAdaptationPolicy(config))
    oracle_env, observation, info = env()
    oracle = run_grid_episode(
        oracle_env, OraclePolicy(oracle_env, config), config,
        scenario(), observation, info,
    )
    assert oracle.total_return >= no_adapt.total_return
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.agents.grid_baselines`.

- [ ] **Step 3: Implement the policies and runner**

`agents/grid_baselines.py`:
- `_BasePolicy` stores `previous_observation`, `previous_action`, and a `GridBelief` when the policy uses one. `reset` stores the first observation and clears counters.
- `_update_belief(action, observation, info)` runs before the previous observation is overwritten:
  - `INSPECT`: if `info.get("last_tool_value")` is present, find the target's color. A marker target uses its marker color; the key/door target uses the key/door color. Call `update_inspect(color, value, step)`.
  - `READ_SIGN`: `side = sign_side(previous_observation.agent_position, signpost)`; call `update_sign(side, value, step)`.
  - A physical move where `info["env_steps"] > previous_info["env_steps"]`: call `update_move(previous_position, action.kind, observation.agent_position, public_map, step)`.
  - `INTERACT`: call `update_interact(info["last_outcome"], step)`.
  - `step` is `info["env_steps"]`; tool calls at the same physical step are allowed and do not advance it.
- `NoAdaptationPolicy`: `most_likely_rule` returns `"base"`; `shift_detected` returns `False`; `choose` replans with `plan_to_goal(BASE_RULE, ...)` every decision and returns the first action, or the first legal physical action if the plan is empty.
- `PublicAdvisorPolicy`: keeps a public `GridBelief` only for discrepancy detection. On `shift_probability() > 0.5`, it calls `PublicAdvisor.advise`, increments `advisor_calls`, and returns the advised action. Otherwise it follows the base plan. It never lets the advisor change its belief. `most_likely_rule` returns `"base"`; `shift_detected` is `advisor_calls > 0`.
- `RandomProbePolicy`: on `shift_probability() > 0.5`, chooses a random legal tool from the action mask if one is affordable; otherwise follows the base plan. Use a seeded `random.Random` so runs are reproducible.
- `InfoGainPolicy`: chooses with `InfoGainSelector`, updates the belief in `record`, and delegates `most_likely_rule`/`shift_detected` to the belief.

`evaluator/grid_trace.py`:
- `GridEvaluatorTrace` with `scenario_id`, `run_seed`, `method`, `true_rule`, `initial_rule`, `shift_time`, `scenario_type`, `map_seed`, `success`, `total_return`, `env_steps`, `tool_units`, `advisor_calls`, `adaptation_steps`, `false_alarm`, `final_rule_estimate`, and `to_row()`.
- It is imported only by `evaluator/` and tests; never by `agents/`.

`evaluator/grid_runner.py`:
- `GridEpisode` fields: `scenario_id`, `run_seed`, `method`, `success`, `total_return`, `env_steps`, `tool_units`, `advisor_calls`, `total_cost`, `adaptation_steps`, `false_alarm`, `final_rule_estimate`, `public_rows`, `evaluator_rows`, plus defaults `scenario_type="change_detection"`, `shift_detected=False`, `oracle_return=None`, `json_valid=True`, `valid_action=True`, `output_tokens=0`, `cache_hit=False`, `tool_calls=()`, `information_gain_bits=0.0`.
- `OraclePolicy(env, config)` reads `env.evaluator_state()["true_rule"]` and plans with `RULE_BY_ID[true_rule]`. It is evaluator-only and is never passed to an agent.
- `run_grid_episode(env, policy, config, scenario, observation, info)`:
  - Call `policy.reset(observation, info)`.
  - Loop up to `config.max_decisions`:
    - `action = policy.choose(observation, info)`.
    - `next_observation, reward, terminated, truncated, next_info = env.step(action)`.
    - `policy.record(action, next_observation, reward, next_info)`.
    - Append a public row: the observation payload plus `chosen_action`, `tool_target`, `actual_reward`, `env_steps`, `tool_budget_remaining`, `advisor_calls`, `json_valid`, `valid_action`, `output_tokens`, `cache_hit`. It never contains `true_rule`, `shift_time`, or a belief.
    - Track the first step at which `policy.most_likely_rule() == scenario.shift_rule` and `next_info["env_steps"] >= scenario.shift_time`.
    - Set `observation, info = next_observation, next_info`; break on `terminated or truncated`.
  - `success = info.get("success", False)`; `total_return = sum(rewards)`.
  - `tool_units = config.tool_budget - info["tool_budget_remaining"]`.
  - `advisor_calls = getattr(policy, "advisor_calls", 0)`.
  - `adaptation_steps = max(0, first_correct_step - scenario.shift_time)` when a shift exists and detection happened; otherwise `None`.
  - `false_alarm = scenario.scenario_type == "no_shift" and policy.shift_detected()`.
  - `evaluator_rows = (GridEvaluatorTrace(...).to_row(),)`.
  - `total_cost = env_steps + config.advisor_cost * advisor_calls`.
- `build_grid_scenarios(n_maps, seeds, config)`:
  - Map seeds `0..n_maps-1`, run seeds `0..seeds-1`.
  - For each pair, add an identification scenario cycling through `GRID_RULE_IDS`, a change-detection scenario for each of `("remap_swap", "slippery", "hazard_goal_swap", "sign_composition")`, and a no-shift control with `initial_rule="base"`.
  - Shift times are sampled deterministically from `config.shift_window` with `random.Random(map_seed * 1000 + run_seed)`.
  - `scenario_id` is `m{map_seed}-s{run_seed}-{scenario_type}-{initial_rule}-{shift_rule}`; `run_seed` is stored on the scenario.
- `run_grid_benchmark(methods, scenarios, config)`:
  - `methods` is a mapping `name -> factory(config, env) -> GridPolicy`.
  - For each method and scenario: create `GridEnv(config)`, reset with `seed=scenario.run_seed` and `options={"scenario": scenario}`, build the policy with the factory, run the episode, collect.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/agents/grid_baselines.py src/ruleshift/evaluator/grid_trace.py src/ruleshift/evaluator/grid_runner.py tests/test_grid_runner.py
git commit -m "feat: add grid baselines, privileged trace, and closed-loop runner"
```

---

### Task 10: Metrics, Paired CIs, Smoke and Audit Scripts

**Files:**
- Create: `src/ruleshift/evaluator/grid_reference.py`
- Create: `src/ruleshift/evaluator/grid_metrics.py`
- Create: `scripts/run_grid_smoke.py`
- Create: `scripts/run_grid_audit.py`
- Create: `tests/test_grid_metrics.py`
- Create: `tests/test_grid_scripts.py`

**Interfaces:**
- Consumes: `GridEpisode`, `PairedSummary`, `summarize_paired`, `GRID_HYPOTHESES`, `rules_consistent_with_all_clauses`, `scan_package`, `HypothesisDynamics`.
- Produces:
  - `q_star_microtask(belief, public_map, position, horizon=2) -> float` — evaluator-only exhaustive depth-limited reference for calibration diagnostics, never a policy.
  - `oracle_return(episode: GridEpisode) -> float | None`.
  - `summarize_grid(episodes: Sequence[GridEpisode]) -> dict[str, float]`.
  - `paired_grid_ci(method_episodes, baseline_episodes, n_boot=2000) -> PairedSummary`.
  - `audit_grid(config: GridConfig, n_maps: int = 20) -> dict[str, Any]`.
  - `scripts/run_grid_smoke.py` with `--config`, `--maps`, `--seeds`, `--methods`, `--out`, `--quick`.
  - `scripts/run_grid_audit.py` with `--config`, `--maps`, `--out`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_metrics.py
from ruleshift.evaluator.grid_metrics import paired_grid_ci, summarize_grid
from ruleshift.evaluator.grid_runner import GridEpisode


def episode(scenario_id, seed, success, cost, false_alarm=False, adaptation=None,
            scenario_type="change_detection"):
    return GridEpisode(
        scenario_id=scenario_id, run_seed=seed, method="m", success=success,
        total_return=10.0 if success else 0.0, env_steps=cost,
        tool_units=0, advisor_calls=0, total_cost=float(cost),
        adaptation_steps=adaptation, false_alarm=false_alarm,
        scenario_type=scenario_type,
        final_rule_estimate="base", public_rows=(), evaluator_rows=(),
    )


def test_summary_counts_success_and_false_alarms():
    episodes = [
        episode("a", 0, True, 10),
        episode("a", 1, False, 20, false_alarm=True, scenario_type="no_shift"),
        episode("b", 0, True, 15, scenario_type="no_shift"),
    ]
    summary = summarize_grid(episodes)
    assert summary["n_episodes"] == 3
    assert abs(summary["success_rate"] - 2 / 3) < 1e-9
    assert abs(summary["false_alarm_rate"] - 0.5) < 1e-9


def test_paired_ci_resamples_scenarios_not_seeds():
    method = [
        episode("a", 0, True, 10), episode("a", 1, True, 12),
        episode("b", 0, True, 20), episode("b", 1, True, 22),
    ]
    baseline = [
        episode("a", 0, False, 30), episode("a", 1, False, 32),
        episode("b", 0, False, 40), episode("b", 1, False, 42),
    ]
    summary = paired_grid_ci(method, baseline, n_boot=200)
    assert summary.n_scenarios == 2
    assert summary.mean_difference < 0


def test_q_star_microtask_is_a_depth_limited_upper_bound():
    from ruleshift.agents.grid_belief import GridBelief
    from ruleshift.evaluator.grid_reference import q_star_microtask
    from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap

    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    belief = GridBelief(hypotheses=("base",))
    value = q_star_microtask(belief, public, (1, 1), horizon=2)
    assert 0.0 < value <= 11.0   # goal reward 10 minus at most two step costs
```

```python
# tests/test_grid_scripts.py
import json
from pathlib import Path
import subprocess
import sys


def test_smoke_script_writes_outputs(tmp_path):
    out = tmp_path / "grid"
    subprocess.run(
        [sys.executable, "scripts/run_grid_smoke.py", "--quick",
         "--maps", "2", "--seeds", "1", "--out", str(out)],
        check=True,
    )
    assert (out / "grid_smoke.json").exists()
    assert (out / "grid_smoke.md").exists()
    payload = json.loads((out / "grid_smoke.json").read_text(encoding="utf-8"))
    assert "summary" in payload


def test_audit_script_passes_on_small_config(tmp_path):
    out = tmp_path / "audit"
    subprocess.run(
        [sys.executable, "scripts/run_grid_audit.py", "--maps", "3",
         "--out", str(out)],
        check=True,
    )
    assert (out / "grid_audit.md").exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_metrics.py tests/test_grid_scripts.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.evaluator.grid_metrics`.

- [ ] **Step 3: Implement metrics and scripts**

`grid_reference.py`:
- `q_star_microtask` enumerates actions and the most likely outcome of each
  `HypothesisDynamics` model up to `horizon`, accumulating the public reward
  (`goal_reward`, `step_cost`, `soft_hazard_damage`). It is an evaluator-only
  diagnostic: the module is never imported by `agents/` or `public_api/`.
- `oracle_return` returns `episode.oracle_return` when present, else `None`.

`summarize_grid`:
- Return `n_episodes`, `success_rate`, `mean_return`, `mean_env_steps`, `mean_tool_units`, `mean_total_cost`, `mean_adaptation_steps` (only non-`None` values), `false_alarm_rate` (no-shift episodes only), `detection_rate` / `detection_precision` (shifted episodes), `mean_regret` (when `oracle_return` is present), `brier_score` and `expected_calibration_error` for the shift trigger (when calibration labels are present), `information_gain_per_tool_call` and `information_gain_per_total_cost`, `mean_hazard_damage` and `deaths` (Tier B), `tool_precision` / `tool_recall` (when tool accounting is present), `json_valid_rate`, `valid_action_rate`, `mean_output_tokens`, `mean_latency_ms`, `cache_hit_rate`.
- Guard empty groups by omitting the key rather than returning `nan`; a missing key is how a report shows "not measured".

`paired_grid_ci`:
- Group episodes by `scenario_id`; average the metric per scenario over `run_seed`; pair the method and baseline by scenario id; call `summarize_paired` with the scenario-level means. `n_scenarios` must be the number of scenarios, not episodes.
- The test's method/baseline lists have two seeds per scenario; the function averages them to 11 and 21 for scenario `a`, and 21 and 41 for `b`; the mean difference is -15; `n_scenarios == 2`.

`audit_grid`:
- For `n_maps` seeds, generate a map and check all hypotheses solvable.
- Render the same public state under two scenarios with different rules; assert the images are equal.
- Build a public observation and assert `find_forbidden_fields(payload) == []`.
- Check tool accounting: a fresh env with budget 3; inspect (2), inspect again (rejected), read sign (1), read sign again (rejected); budget 0; env steps 0.
- Check sign partiality: for every rule, every side and the full set leave >=2 hypotheses.
- Check permissions: `scan_package(Path("src/ruleshift/agents"))` and `scan_package(Path("src/ruleshift/public_api"))` return no violations.
- Return `{"passed": bool, "checks": {...}, "failures": [...]}`.

`run_grid_smoke.py`:
- Load config.
- Build scenarios with `build_grid_scenarios`.
- Build methods: `no_adaptation`, `public_advisor`, `random_probe`, `info_gain`, `oracle`.
- Run `run_grid_benchmark`.
- Summarize per method; write JSON and Markdown.
- The Markdown header must say `SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING`.
- `--quick` uses the provided `--maps`/`--seeds` (default 5/2); the full run uses 100/5.
- The script must not touch `configs/preregistered_test.yaml`.

`run_grid_audit.py`:
- Call `audit_grid`; write `grid_audit.md`; exit 1 if `passed` is false.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_metrics.py tests/test_grid_scripts.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/evaluator/grid_reference.py src/ruleshift/evaluator/grid_metrics.py scripts/run_grid_smoke.py scripts/run_grid_audit.py tests/test_grid_metrics.py tests/test_grid_scripts.py
git commit -m "feat: add grid reference, metrics, paired CIs, smoke and audit scripts"
```

---

### Task 11: Tier B VLM Prompt, Parser, Cache, and Policy

**Files:**
- Create: `src/ruleshift/prompts/grid_v1.txt`
- Create: `src/ruleshift/agents/grid_vlm.py`
- Create: `tests/test_grid_vlm.py`

**Interfaces:**
- Consumes: `GridPublicObservation`, `GridAction`, `GRID_ACTIONS`, `GRID_PHYSICAL_ACTIONS`, `GRID_RULE_IDS`, `GRID_PUBLIC_OBSERVATION_FIELDS`, `validate_grid_vlm_output`, `ModelRequest`, `ResponseCache`, `find_forbidden_fields`.
- Produces:
  - `GRID_PROMPT_PATH`, `GRID_PROMPT_VERSION = "grid_v1"`.
  - `render_grid_prompt(observation: GridPublicObservation) -> str`.
  - `parse_grid_output(text, *, action_mask, grid_size, position=None, inspect_range=None) -> tuple[dict | None, list[str]]`.
  - `GridVLMConfig(model_id, revision, max_new_tokens=1024, decoding_seed=0)`.
  - `GridStubBackend` — deterministic local stand-in; emits the grid schema, never Family A/B.
  - `GridVLMPolicy(backend, config, cache=None)` implementing the `GridPolicy` protocol.
  - `grid_candidate_hash(observation) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_vlm.py
from ruleshift.agents.grid_vlm import (
    GRID_PROMPT_VERSION, GridStubBackend, GridVLMConfig, GridVLMPolicy,
    parse_grid_output, render_grid_prompt,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GridAction


def observation():
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode="visual")
    obs, _ = env.reset(seed=0, options={"map": public})
    return obs


def test_prompt_is_public_only():
    text = render_grid_prompt(observation())
    assert find_forbidden_fields({"prompt": text}) == []
    assert "true_rule" not in text
    assert "shift_time" not in text


def test_prompt_lists_the_grid_actions():
    text = render_grid_prompt(observation())
    for action in ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN"):
        assert action in text


def test_parser_accepts_a_valid_grid_output():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "tool_target": None,
        "confidence": 0.5,
    }
    import json
    parsed, errors = parse_grid_output(
        json.dumps(payload),
        action_mask=("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN"),
        grid_size=(7, 7),
    )
    assert errors == []
    assert parsed["action"] == "UP"


def test_parser_rejects_privileged_field():
    import json
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "shift_time": 3,
    }
    parsed, errors = parse_grid_output(json.dumps(payload), action_mask=("UP",), grid_size=(7, 7))
    assert parsed is None
    assert any("privileged" in error for error in errors)


def test_parser_rejects_out_of_range_tool_target():
    import json
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "INSPECT",
        "tool_target": [99, 99],
    }
    parsed, errors = parse_grid_output(
        json.dumps(payload), action_mask=("INSPECT",), grid_size=(7, 7)
    )
    assert parsed is None
    assert errors


def test_policy_uses_the_cache_for_identical_observations():
    backend = GridStubBackend()
    policy = GridVLMPolicy(backend, config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    first = policy.choose(obs, {})
    policy.record(first, obs, 0.0, {"env_steps": 0})
    policy.choose(obs, {})
    assert backend.calls == 1


def test_policy_returns_a_legal_action():
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    assert isinstance(action, GridAction)
    assert action.kind in obs.action_mask


def test_policy_falls_back_on_malformed_output():
    class BrokenBackend:
        calls = 0

        def generate(self, request):
            from ruleshift.agents.vlm_backend import ModelResponse
            self.calls += 1
            return ModelResponse("{ broken", 0, 3, 0.0, "broken", "GRID_BROKEN")

    policy = GridVLMPolicy(BrokenBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    assert action.kind in {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT"}
    assert policy.last_json_valid is False


def test_policy_rows_are_public_only():
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    policy.record(action, obs, 0.0, {"env_steps": 0, "tool_budget_remaining": 3})
    row = policy.last_public_row()
    assert find_forbidden_fields(row) == []


def test_prompt_version_is_pinned():
    assert GRID_PROMPT_VERSION == "grid_v1"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_vlm.py -v`
Expected: FAIL with `ModuleNotFoundError: ruleshift.agents.grid_vlm`.

- [ ] **Step 3: Write the public-only prompt**

`src/ruleshift/prompts/grid_v1.txt` follows `direct_v1.txt` in tone but never mentions hidden state. Placeholders: `{public_history}`, `{public_text}`, `{candidate_rules}`, `{available_actions}`, `{public_costs}`, `{action_mask}`, `{tool_budget}`, `{position}`, `{grid_size}`. State explicitly:
- the sign text is illegible in the frame and `READ_SIGN` returns one clause from one adjacent side;
- `INSPECT(x, y)` is range-limited to Manhattan distance 2 and costs 2 tool units; `READ_SIGN` costs 1; the total budget is 3;
- one JSON object per decision, one action only; a tool call never also moves the agent;
- output keys are exactly `hypothesis_probabilities`, `action`, `tool_target`, `predicted_outcomes`, `confidence`;
- probabilities have one entry per candidate rule and sum to 1;
- `tool_target` is required only for `INSPECT`;
- keep the object compact on one line and return no prose or code fence.

- [ ] **Step 4: Implement `grid_vlm.py`**

`render_grid_prompt` fills the template from `observation.to_payload()` fields only. Never pass `image` bytes into the prompt; pass `image_hash`.

`parse_grid_output`:
- Strip a leading ``` fence and a trailing ``` like `parse_structured_output` does.
- `json.loads`; on failure return `(None, ["invalid JSON: ..."])`.
- Call `validate_grid_vlm_output(payload)`; also require `payload["action"] in action_mask`.
- If the action is `INSPECT`, require `tool_target` to be a two-integer pair, inside `grid_size`, and within `inspect_range` of `position` when both are supplied.
- Never accept extra keys. Return errors, never raise.

`GridStubBackend.generate(request)`:
- Seed `random.Random` from `request.fingerprint()` exactly like `StubVLMBackend`.
- Emit a compact JSON object with uniform `hypothesis_probabilities` over `GRID_RULE_IDS`, an `action` chosen from the physical actions in `request.available_actions` (never `INSPECT`, because the stub cannot choose a meaningful target), `tool_target: None`, and `confidence: 0.5`.
- `backend_id="GRID_STUB_SYNTHETIC"` so no report can mistake it for evidence.

`GridVLMPolicy`:
- `reset` clears counters, `last_json_valid=True`, and the previous output.
- `choose(observation, info)` builds a `ModelRequest` with `model_id`, the rendered prompt, `image_hash=observation.image_hash`, `history_hash=observation.public_history_hash`, `available_actions=observation.action_mask`, `candidate_hash=grid_candidate_hash(observation)`, `decoding_seed`, `max_new_tokens`, and `prompt_version=GRID_PROMPT_VERSION`.
- Look up `ResponseCache.get`; on a miss call `backend.generate` and `cache.put`. Set `last_cache_hit`.
- Parse with `parse_grid_output`, passing the observation's `action_mask`, `(public_map.width, public_map.height)`, `agent_position`, and `inspect_range` from the public costs.
- On error, return the first `GRID_PHYSICAL_ACTIONS` member in the mask and set `last_json_valid=False`, `last_valid_action=False`.
- On success, return `GridAction(action, tuple(tool_target))` and set `last_json_valid=True`, `last_valid_action=True`, `last_probabilities`.
- Track `output_tokens`, `input_tokens`, `latency_ms`, `model_calls`, and `cache_hits`. `advisor_calls` stays 0: Tier B is not priced by the Tier A `lambda` comparison.
- `record` stores the last decision row; `most_likely_rule` returns the argmax of the last probabilities or `"base"`; `shift_detected` returns whether that rule is not `"base"`; `last_public_row()` returns the stored row.
- This module imports only `public_api`, `agents`, and `permissions`. It must not import `envs` or `evaluator`. `GridVLMConfig` is a frozen dataclass carrying `model_id`, `revision=None`, `decoding_seed=0`, and `max_new_tokens=1024`. `ModelRequest` already fingerprints `prompt_version`, so pass `f"{GRID_PROMPT_VERSION}|{revision or 'none'}"` as that field to pin the model revision in the cache key without corrupting the backend's `model_id`.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_vlm.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add src/ruleshift/prompts/grid_v1.txt src/ruleshift/agents/grid_vlm.py tests/test_grid_vlm.py
git commit -m "feat: add grid VLM prompt, parser, cache, and stub policy scaffold"
```

---

### Task 12: Permission Boundary and Signpost-Shortcut Acceptance

**Files:**
- Create: `tests/test_grid_permissions.py`
- Create: `tests/test_grid_signpost_shortcut.py`

**Interfaces:**
- Consumes: `scan_package`, `find_forbidden_fields`, `GRID_PUBLIC_OBSERVATION_FIELDS`, `rules_consistent_with_all_clauses`, `sign_clause`, `sign_side`, `GridEnv`, `GridBelief`, `run_grid_episode`, `plan_to_goal`, `RULE_BY_ID`.
- Produces: no new production code. If a test fails because a module leaks privileged names, fix that module in its owning task and re-run this task.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_grid_permissions.py
from pathlib import Path
import yaml
from ruleshift.permissions import find_forbidden_fields, scan_package
from ruleshift.public_api.grid_rules import GRID_HYPOTHESES
from ruleshift.public_api.grid_schemas import GRID_PUBLIC_OBSERVATION_FIELDS
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN


def test_agent_and_public_api_packages_have_no_forbidden_imports():
    for package in ("src/ruleshift/agents", "src/ruleshift/public_api"):
        assert scan_package(Path(package)) == []


def test_grid_rules_are_public_descriptions_only():
    for rule in GRID_HYPOTHESES:
        payload = {"rule_id": rule.rule_id, "description": rule.description}
        assert find_forbidden_fields(payload) == []


def test_grid_observation_fields_cover_the_required_public_fields():
    required = {
        "observation_id", "public_obs_hash", "public_history_hash",
        "public_history", "public_text", "available_actions",
        "candidate_rules", "public_costs", "image_hash",
    }
    assert required <= set(GRID_PUBLIC_OBSERVATION_FIELDS)


def test_grid_actions_do_not_widen_the_macro_set():
    from ruleshift.public_api.grid_schemas import GRID_ACTIONS
    assert not (set(GRID_ACTIONS) & MACRO_ACTIONS_FROZEN)


def test_preregistered_config_is_still_a_draft():
    data = yaml.safe_load(
        Path("configs/preregistered_test.yaml").read_text(encoding="utf-8")
    )
    assert data["status"] == "DRAFT_NOT_FROZEN"
```

```python
# tests/test_grid_signpost_shortcut.py
from ruleshift.agents.grid_baselines import _BasePolicy
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import run_grid_episode
from ruleshift.public_api.grid_rules import (
    GRID_HYPOTHESES, RULE_BY_ID, PublicGridMap,
    rules_consistent_with_all_clauses, sign_clause,
)
from ruleshift.public_api.grid_schemas import GridAction


def test_full_sign_set_is_behaviorally_ambiguous():
    for rule in GRID_HYPOTHESES:
        clauses = tuple(sign_clause(rule, side) for side in range(3))
        consistent = rules_consistent_with_all_clauses(clauses)
        assert len(consistent) >= 2
        signatures = {
            (RULE_BY_ID[rule_id].goal_color,
             tuple(sorted(RULE_BY_ID[rule_id].move_map.items())))
            for rule_id in consistent
        }
        assert len(signatures) >= 2


class SignOnlyPolicy(_BasePolicy):
    """Read all three sides via a fixed walk, then commit to the first rule."""

    ROUTE = ("READ_SIGN", "RIGHT", "DOWN", "READ_SIGN",
             "DOWN", "LEFT", "READ_SIGN")

    def __init__(self, config):
        super().__init__(use_belief=False)
        self._clauses = {}
        self._route_index = 0
        self._committed = None

    def choose(self, observation, info):
        if self._route_index < len(self.ROUTE):
            return GridAction(self.ROUTE[self._route_index])
        if self._committed is None:
            clauses = tuple(self._clauses[side] for side in range(3))
            consistent = rules_consistent_with_all_clauses(clauses)
            self._committed = consistent[0] if consistent else "base"
        plan = self.base_plan(self._committed, observation)
        return plan[0] if plan else GridAction("UP")

    def record(self, action, observation, reward, info):
        if action.kind == "READ_SIGN" and "last_tool_value" in info:
            from ruleshift.public_api.grid_rules import sign_side
            side = sign_side(self.previous_observation.agent_position,
                             observation.public_map.signpost)
            self._clauses[side] = info["last_tool_value"]
        if self._route_index < len(self.ROUTE):
            self._route_index += 1
        super().record(action, observation, reward, info)

    def most_likely_rule(self):
        return self._committed or "base"

    def shift_detected(self):
        return False


def test_scripted_signpost_only_policy_underperforms():
    # start (2,2), signpost (2,3): north side is adjacent at the start.
    public = PublicGridMap(
        7, 7, frozenset(), (2, 2),
        {"red": (1, 1), "blue": (5, 5), "green": (4, 4)},
        key=None, door=None, signpost=(2, 3),
    )
    scenario = GridScenario(
        "sign-shortcut", "identification", "sign_composition",
        None, None, 0,
    )
    config = GridConfig(baseline_slip=0.0, slippery_slip=0.0, max_env_steps=20)
    env = GridEnv(config, observation_mode="symbolic")
    obs, info = env.reset(seed=0, options={"map": public, "scenario": scenario})
    episode = run_grid_episode(env, SignOnlyPolicy(config), config, scenario, obs, info)
    assert episode.tool_units == 3
    assert episode.success is False


def test_unlimited_tool_budget_is_an_ablation_not_a_baseline():
    from ruleshift.agents.grid_baselines import InfoGainPolicy

    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(4, 3),
    )
    scenario = GridScenario("ablation", "identification", "base", None, None, 0)

    def run(budget):
        config = GridConfig(tool_budget=budget, baseline_slip=0.0)
        env = GridEnv(config)
        obs, info = env.reset(seed=0, options={"map": public, "scenario": scenario})
        return run_grid_episode(env, InfoGainPolicy(config), config, scenario, obs, info)

    limited = run(3)
    unlimited = run(99)
    assert unlimited.tool_units > limited.tool_units
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/test_grid_permissions.py tests/test_grid_signpost_shortcut.py -v`
Expected: FAIL (the shortcut policy and `_BasePolicy.base_plan` do not exist yet).

- [ ] **Step 3: Support the test policy and verify the failure is real**

In `agents/grid_baselines.py`, make `_BasePolicy` accept `use_belief: bool = True`, expose `previous_observation`, and add `base_plan(rule_id, observation)` returning `plan_to_goal(RULE_BY_ID[rule_id], public_map, position, has_key, door_open)`. These are public helpers for scripted policies; they read only the observation.

If `test_full_sign_set_is_behaviorally_ambiguous` fails, the clause-sharing table in Task 2 is wrong: fix the clauses so each rule's full sign set still leaves at least two behaviorally distinct rules, then re-run Tasks 2, 8, 10, and 12.

If `test_scripted_signpost_only_policy_underperforms` does not fail, check that:
- the true rule is `sign_composition` (goal green) while the first consistent rule is `hazard_goal_swap` (goal blue);
- the three `READ_SIGN` calls cost exactly 3 units;
- the commit action is computed only from the sign clauses, never from transition evidence.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_permissions.py tests/test_grid_signpost_shortcut.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass.

```bash
git add tests/test_grid_permissions.py tests/test_grid_signpost_shortcut.py src/ruleshift/agents/grid_baselines.py
git commit -m "test: enforce grid permission boundary and signpost shortcut failure"
```

---

### Task 13: Remote Kaggle Grid Pilot Scaffold

**Files:**
- Create: `scripts/run_grid_real_pilot.py`
- Create: `kaggle/kernel/run_grid.py`
- Modify: `tests/test_grid_scripts.py` (one CLI guard test)

**Interfaces:**
- Consumes: `configs/models.yaml` (`qwen2.5-vl-3b`, pinned `revision`, `device`), `GridVLMConfig`, `GridVLMPolicy`, `TransformersVLMBackend`, `ResponseCache`, `build_grid_scenarios`, `run_grid_episode`, `summarize_grid`.
- Produces:
  - `python scripts/run_grid_real_pilot.py --maps N --seeds N --device cuda --out DIR --report FILE`.
  - A remote-only kernel `kaggle/kernel/run_grid.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_grid_scripts.py
def test_grid_real_pilot_refuses_local_cuda_without_kaggle_env(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("RULESHIFT_KAGGLE", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_grid_real_pilot.py", "--device", "cuda",
         "--out", str(tmp_path), "--report", str(tmp_path / "report.md")],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "RULESHIFT_KAGGLE" in result.stdout + result.stderr
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_grid_scripts.py -k real_pilot -v`
Expected: FAIL with a missing script or exit 0.

- [ ] **Step 3: Implement the pilot script**

`scripts/run_grid_real_pilot.py`:
- Parse `--model-key` (default `qwen2.5-vl-3b`), `--device` (default `cuda`), `--maps` (default 3), `--seeds` (default 1), `--out`, `--report`, and `--config` (default `configs/grid_dev.yaml`).
- **Guard before importing `torch` or `transformers`:** if `device` starts with `cuda` and `os.environ.get("RULESHIFT_KAGGLE") != "1"`, print an explanation naming `RULESHIFT_KAGGLE` and exit 2. This is the hard barrier against a local GPU run.
- Load `configs/models.yaml`; assert the chosen model has a 40-hex `revision`, `local_files_only` is true in the committed config, and `device` agrees with the CLI unless the Kaggle kernel patches it.
- Build `TransformersVLMBackend(model_id, revision=..., device=..., dtype=...)` and `GridVLMPolicy` with `max_new_tokens=1024`. Reuse the v10 lesson: never shrink the token budget, because a truncated response is counted as invalid JSON.
- Build scenarios with `build_grid_scenarios(--maps, --seeds, config)`; run only the `identification` and `change_detection` subsets, no-shift controls included, then cap at a small number of episodes via `--max-episodes` (default 20).
- Write `grid_real_decisions.jsonl` (public rows only) and `grid_real_evaluator_only.jsonl` (privileged trace, separate file). Verify with `find_forbidden_fields` on every public row before writing.
- Write the Markdown report with the header `REAL FROZEN VLM — SCAFFOLD RUN, NOT A SCIENTIFIC FINDING`, plus JSON validity, valid-action rate, mean output tokens, cache hit rate, and the paired-CI placeholders. `configs/preregistered_test.yaml` is read but never written.
- Do not report success rate as evidence: the map/seed count is far below the 100x5 gate.

`kaggle/kernel/run_grid.py`:
- Reuse the source-snapshot and `SOURCE_COMMIT` verification logic from `kaggle/kernel/run.py` (copy it; do not import it, because the kernel runs as a script).
- After `pip install -e ".[dev]"`, `pip install -U transformers`, and `nvidia-smi`, set `RULESHIFT_KAGGLE=1` in the environment and run `scripts/run_grid_real_pilot.py` with `--device cuda`.
- Print the report path and the two JSONL paths at the end.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python -m pytest tests/test_grid_scripts.py -v`
Expected: PASS. No real model is loaded or downloaded locally.

- [ ] **Step 5: Run the full suite and commit**

Run: `python -m pytest -q`
Expected: all existing 217 tests plus all new grid tests pass.

```bash
git add scripts/run_grid_real_pilot.py kaggle/kernel/run_grid.py tests/test_grid_scripts.py
git commit -m "feat: add remote-only grid pilot scaffold and Kaggle kernel"
```

---

## Final Verification (run before presenting as complete)

- [ ] `python -m pytest -q` — exit 0; record the test count.
- [ ] `python scripts/run_grid_audit.py --maps 10 --out reports` — exit 0; `reports/grid_audit.md` exists and `passed: true`.
- [ ] `python scripts/run_grid_smoke.py --quick --maps 5 --seeds 2 --out reports` — exit 0; `reports/grid_smoke.md` starts with `SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING`.
- [ ] `git diff --name-only origin/main...HEAD` contains only spec/plan/source/test/config files — no `reports/`, `results/`, `data/`, `outputs/`, `kaggle.json`, or `configs/preregistered_test.yaml` change.
- [ ] `git status --short` is clean after the commits.
- [ ] `python scripts/preflight.py` still prints NO-GO; this is expected and not a failure.
- [ ] Confirm no local GPU job was run: no `nvidia-smi` or `transformers` import in local CI tests.

## Out of Scope (do not implement in this change)

- Learned world model and MPC behind `DynamicsModel` (Phase 3).
- Full POMDP Q* solver; only the exact belief and Q* microtasks are in scope.
- Freezing `configs/preregistered_test.yaml`.
- Running the real frozen VLM locally or reporting synthetic smoke output as a finding.
- Changing `MACRO_ACTIONS`, Family A/B environments, or existing validators.
