"""Closed-loop runner, scenario builder and benchmark for RuleShift-Grid.

The runner owns the loop: the caller resets the environment and passes its
initial ``(observation, info)``; the runner never resets it. Public decision
rows and the privileged trace are produced separately so hidden state can never
be joined back into a model input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ruleshift.agents.grid_baselines import GridPolicy, as_observation
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_trace import GridEvaluatorTrace
from ruleshift.public_api.grid_planner import plan_to_goal
from ruleshift.public_api.grid_rules import RULE_BY_ID
from ruleshift.public_api.grid_schemas import (
    GRID_RULE_IDS,
    GRID_TOOL_ACTIONS,
    GridAction,
)

_CHANGE_RULES = ("remap_swap", "slippery", "hazard_goal_swap", "sign_composition")
_PHYSICAL_KINDS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT")


@dataclass(frozen=True)
class GridEpisode:
    """One episode's public summary plus separately stored traces."""

    scenario_id: str
    run_seed: int
    method: str
    success: bool
    total_return: float
    env_steps: int
    tool_units: int
    advisor_calls: int
    total_cost: float
    adaptation_steps: int | None
    false_alarm: bool
    final_rule_estimate: str
    public_rows: tuple[dict[str, Any], ...]
    evaluator_rows: tuple[dict[str, Any], ...]
    scenario_type: str = "change_detection"
    shift_detected: bool = False
    oracle_return: float | None = None
    json_valid: bool = True
    valid_action: bool = True
    output_tokens: int = 0
    cache_hit: bool = False
    tool_calls: tuple[Any, ...] = ()
    #: ``None`` means the policy never measured information gain; a float is
    #: the episode-wide total it reported (0.0 is a genuine measurement).
    information_gain_bits: float | None = None
    #: Information gain attributable to INSPECT/READ_SIGN decisions only.
    #: This is the numerator for ``information_gain_per_tool_call``.
    tool_information_gain_bits: float | None = None
    #: Information gain attributable to physical moves only.
    move_information_gain_bits: float | None = None
    #: Number of tool actions actually issued, independent of their unit cost.
    tool_calls_made: int = 0


class OraclePolicy:
    """Privileged diagnostic: plans against the evaluator-only true rule.

    This class lives in the evaluator package and is never passed to an agent.
    It re-reads the active rule each decision so it also adapts after a shift.
    """

    def __init__(self, env: GridEnv, config: GridConfig) -> None:
        self.env = env
        self.config = config
        self.previous_observation: Any = None
        self.previous_info: dict[str, Any] = {}
        self.advisor_calls = 0

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        self.previous_observation = as_observation(observation)
        self.previous_info = dict(info or {})

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        observation = as_observation(observation)
        # Evaluator-only ground truth; never exposed in a public row.
        true_rule = str(self.env.evaluator_state()["true_rule"])
        plan = plan_to_goal(
            RULE_BY_ID[true_rule],
            observation.public_map,
            tuple(observation.agent_position),
            has_key=bool(observation.inventory),
            door_open=bool(observation.door_open),
        )
        if plan:
            return plan[0]
        for kind in _PHYSICAL_KINDS:
            if kind in observation.action_mask:
                return GridAction(kind)
        return GridAction("INTERACT")

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        self.previous_observation = as_observation(observation)
        self.previous_info = dict(info or {})

    def most_likely_rule(self) -> str:
        return "base"

    def shift_detected(self) -> bool:
        return False


def _method_name(policy: Any) -> str:
    """Best-effort public name for a policy without a declared method name."""

    for attribute in ("name", "method"):
        value = getattr(policy, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(policy).__name__


def run_grid_episode(
    env: GridEnv,
    policy: GridPolicy,
    config: GridConfig,
    scenario: GridScenario,
    observation: Any,
    info: dict[str, Any],
) -> GridEpisode:
    """Drive one episode from an already-reset environment."""

    observation = as_observation(observation)
    policy.reset(observation, info)

    method = _method_name(policy)
    max_decisions = int(getattr(config, "max_decisions", 200))
    rewards: list[float] = []
    public_rows: list[dict[str, Any]] = []
    first_correct_step: int | None = None
    tool_calls_made = 0

    for decision in range(max_decisions):
        action = policy.choose(observation, info)
        if action.kind in GRID_TOOL_ACTIONS:
            tool_calls_made += 1
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        next_observation = as_observation(next_observation)
        policy.record(action, next_observation, reward, next_info)

        # Public row: the observation payload plus this decision's accounting.
        # It never contains the true rule, the shift time, or a belief.
        row: dict[str, Any] = dict(observation.to_payload())
        row.update(
            {
                "chosen_action": action.kind,
                "tool_target": list(action.target) if action.target is not None else None,
                "actual_reward": float(reward),
                "env_steps": int(next_info.get("env_steps", 0)),
                "tool_budget_remaining": int(
                    next_info.get("tool_budget_remaining", config.tool_budget)
                ),
                "advisor_calls": int(getattr(policy, "advisor_calls", 0)),
                "json_valid": bool(getattr(policy, "last_json_valid", True)),
                "valid_action": bool(getattr(policy, "last_valid_action", True)),
                "fallback": bool(getattr(policy, "last_fallback", False)),
                "guard_applied": bool(
                    getattr(policy, "last_guard_applied", False)
                ),
                "hypothesis_probabilities": dict(
                    getattr(policy, "last_probabilities", {}) or {}
                ),
                "parse_error_count": int(
                    len(getattr(policy, "last_parse_errors", ()) or ())
                ),
                "output_tokens": int(getattr(policy, "last_output_tokens", 0)),
                "cache_hit": bool(getattr(policy, "last_cache_hit", False)),
                # A tool's revealed value must travel with the decision in the
                # public log, not stay hidden in the transient info dict.
                "last_tool_value": next_info.get("last_tool_value"),
            }
        )
        public_rows.append(row)

        env_steps_now = int(next_info.get("env_steps", 0))
        if (
            scenario.shift_rule is not None
            and scenario.shift_time is not None
            and policy.most_likely_rule() == scenario.shift_rule
            and env_steps_now >= int(scenario.shift_time)
            and first_correct_step is None
        ):
            first_correct_step = env_steps_now

        rewards.append(float(reward))
        observation, info = next_observation, next_info
        if terminated or truncated:
            break

    success = bool(info.get("success", False))
    total_return = sum(rewards)
    env_steps = int(info.get("env_steps", 0))
    tool_units = int(config.tool_budget) - int(
        info.get("tool_budget_remaining", config.tool_budget)
    )
    advisor_calls = int(getattr(policy, "advisor_calls", 0))
    if scenario.shift_rule is not None and first_correct_step is not None:
        adaptation_steps: int | None = max(
            0, first_correct_step - int(scenario.shift_time or 0)
        )
    else:
        adaptation_steps = None
    scenario_type = str(getattr(scenario, "scenario_type", "change_detection"))
    false_alarm = scenario_type == "no_shift" and policy.shift_detected()
    final_rule_estimate = policy.most_likely_rule()

    trace = GridEvaluatorTrace(
        scenario_id=scenario.scenario_id,
        run_seed=int(scenario.run_seed),
        method=method,
        true_rule=str(env.evaluator_state()["true_rule"]),
        initial_rule=str(scenario.initial_rule),
        shift_time=scenario.shift_time,
        scenario_type=scenario_type,
        map_seed=int(scenario.map_seed),
        success=success,
        total_return=float(total_return),
        env_steps=env_steps,
        tool_units=tool_units,
        advisor_calls=advisor_calls,
        adaptation_steps=adaptation_steps,
        false_alarm=bool(false_alarm),
        final_rule_estimate=final_rule_estimate,
    ).to_row()

    raw_information_gain = getattr(policy, "information_gain_bits", None)
    information_gain_bits = (
        float(raw_information_gain)
        if isinstance(raw_information_gain, (int, float))
        and not isinstance(raw_information_gain, bool)
        else None
    )
    raw_tool_gain = getattr(policy, "tool_information_gain_bits", None)
    tool_information_gain_bits = (
        float(raw_tool_gain)
        if isinstance(raw_tool_gain, (int, float))
        and not isinstance(raw_tool_gain, bool)
        else None
    )
    raw_move_gain = getattr(policy, "move_information_gain_bits", None)
    move_information_gain_bits = (
        float(raw_move_gain)
        if isinstance(raw_move_gain, (int, float))
        and not isinstance(raw_move_gain, bool)
        else None
    )
    total_cost = env_steps + int(config.advisor_cost) * advisor_calls
    return GridEpisode(
        scenario_id=scenario.scenario_id,
        run_seed=int(scenario.run_seed),
        method=method,
        success=success,
        total_return=float(total_return),
        env_steps=env_steps,
        tool_units=tool_units,
        advisor_calls=advisor_calls,
        total_cost=total_cost,
        adaptation_steps=adaptation_steps,
        false_alarm=bool(false_alarm),
        final_rule_estimate=final_rule_estimate,
        public_rows=tuple(public_rows),
        evaluator_rows=(trace,),
        scenario_type=scenario_type,
        shift_detected=bool(policy.shift_detected()),
        oracle_return=None,
        json_valid=bool(getattr(policy, "last_json_valid", True)),
        valid_action=bool(getattr(policy, "last_valid_action", True)),
        output_tokens=int(getattr(policy, "last_output_tokens", 0)),
        cache_hit=bool(getattr(policy, "last_cache_hit", False)),
        tool_calls=tuple(getattr(policy, "last_tool_calls", ())),
        information_gain_bits=information_gain_bits,
        tool_information_gain_bits=tool_information_gain_bits,
        move_information_gain_bits=move_information_gain_bits,
        tool_calls_made=tool_calls_made,
    )


def build_grid_scenarios(
    n_maps: int,
    seeds: int,
    config: GridConfig,
) -> tuple[GridScenario, ...]:
    """Build the paired map/run scenario grid with deterministic shift times."""

    scenarios: list[GridScenario] = []

    def add(
        scenario_type: str,
        initial_rule: str,
        shift_rule: str | None,
        shift_time: int | None,
    ) -> None:
        scenarios.append(
            GridScenario(
                scenario_id=(
                    f"m{map_seed}-s{run_seed}-{scenario_type}-"
                    f"{initial_rule}-{shift_rule}"
                ),
                scenario_type=scenario_type,
                initial_rule=initial_rule,
                shift_rule=shift_rule,
                shift_time=shift_time,
                map_seed=map_seed,
                run_seed=run_seed,
            )
        )

    for map_seed in range(int(n_maps)):
        for run_seed in range(int(seeds)):
            rng = random.Random(map_seed * 1000 + run_seed)

            for rule_id in GRID_RULE_IDS:
                add("identification", rule_id, None, None)

            start, end = (int(part) for part in config.shift_window)
            for shift_rule in _CHANGE_RULES:
                add("change_detection", "base", shift_rule, rng.randint(start, end))

            add("no_shift", "base", None, None)
    return tuple(scenarios)


def run_grid_benchmark(
    methods: Mapping[str, Callable[[GridConfig, GridEnv], GridPolicy]],
    scenarios: Sequence[GridScenario],
    config: GridConfig,
) -> list[GridEpisode]:
    """Run every method on every paired scenario and collect episodes."""

    episodes: list[GridEpisode] = []
    for name, factory in methods.items():
        for scenario in scenarios:
            env = GridEnv(config)
            observation, info = env.reset(
                seed=scenario.run_seed,
                options={"scenario": scenario},
            )
            policy = factory(config, env)
            try:
                policy.name = name  # type: ignore[attr-defined]
            except AttributeError:  # pragma: no cover - defensive for frozen policies
                pass
            episodes.append(
                run_grid_episode(
                    env, policy, config, scenario, observation, info
                )
            )
    return episodes


__all__ = [
    "GridEpisode",
    "OraclePolicy",
    "build_grid_scenarios",
    "run_grid_benchmark",
    "run_grid_episode",
]
