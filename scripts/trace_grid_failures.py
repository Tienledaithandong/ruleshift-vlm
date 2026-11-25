"""Write development-only traces for named RuleShift-Grid scenarios.

This script is deliberately outside the benchmark runner: it records privileged
true-rule and belief fields for failure analysis. Its output is not a model
input and must stay under the git-ignored results/ tree.

Example:
    python scripts/trace_grid_failures.py --out results/grid_v05_dev/traces.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from ruleshift.agents.grid_baselines import (
    DiagnosticBeliefPolicy,
    DiagnosticBeliefPolicyV4,
    DiagnosticBeliefPolicyV5,
    as_observation,
)
from ruleshift.envs.grid.config import load_grid_config
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import build_grid_scenarios
from ruleshift.public_api.grid_rules import sign_side

DEFAULT_SCENARIOS: tuple[str, ...] = (
    "m0-s3-identification-sign_composition-None",
    "m5-s1-identification-hazard_goal_swap-None",
    "m7-s2-change_detection-base-sign_composition",
    "m9-s1-change_detection-base-sign_composition",
)

METHODS = {
    "v3": DiagnosticBeliefPolicy,
    "v4": DiagnosticBeliefPolicyV4,
    "v5": DiagnosticBeliefPolicyV5,
}


def _belief(policy: Any) -> dict[str, float]:
    belief = getattr(policy, "belief", None)
    if belief is None:
        return {}
    return {rule: float(value) for rule, value in belief.marginal().items()}


def trace_episode(
    config: Any, scenario: Any, policy_class: type, method: str
) -> dict[str, Any]:
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    observation = as_observation(observation)
    policy = policy_class(config)
    policy.reset(as_observation(observation), info)
    steps: list[dict[str, Any]] = []

    for decision in range(int(getattr(config, "max_decisions", 200))):
        before = env.evaluator_state()
        before_belief = _belief(policy)
        action = policy.choose(observation, info)
        side = None
        signpost = observation.public_map.signpost
        if action.kind == "READ_SIGN" and signpost is not None:
            try:
                side = sign_side(tuple(before["position"]), tuple(signpost))
            except ValueError:
                side = None
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        next_observation = as_observation(next_observation)
        policy.record(action, next_observation, reward, next_info)
        after = env.evaluator_state()
        steps.append(
            {
                "decision": decision,
                "action": action.kind,
                "target": (list(action.target) if action.target is not None else None),
                "sign_side": side,
                "position_before": list(before["position"]),
                "position_after": list(after["position"]),
                "env_steps_before": int(before["steps"]),
                "env_steps_after": int(after["steps"]),
                "tool_budget_before": int(before["tool_budget"]),
                "tool_budget_after": int(after["tool_budget"]),
                "true_rule_before": before["true_rule"],
                "true_rule_after": after["true_rule"],
                "last_outcome": next_info.get("last_outcome"),
                "last_tool_value": next_info.get("last_tool_value"),
                "reward": float(reward),
                "success": bool(after["success"]),
                "belief_before": before_belief,
                "belief_after": _belief(policy),
            }
        )
        observation, info = next_observation, next_info
        if terminated or truncated:
            break

    return {
        "scenario_id": scenario.scenario_id,
        "method": method,
        "scenario_type": scenario.scenario_type,
        "map_seed": int(scenario.map_seed),
        "run_seed": int(scenario.run_seed),
        "initial_rule": scenario.initial_rule,
        "shift_rule": scenario.shift_rule,
        "shift_time": scenario.shift_time,
        "success": bool(env.evaluator_state()["success"]),
        "final_rule_estimate": policy.most_likely_rule(),
        "final_probabilities": _belief(policy),
        "env_steps": int(env.evaluator_state()["steps"]),
        "tool_units": int(config.tool_budget)
        - int(env.evaluator_state()["tool_budget"]),
        "direct_role_observations": int(getattr(policy, "direct_role_observations", 0)),
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--maps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario id to trace; repeat for several (defaults to four).",
    )
    parser.add_argument("--out", default="results/grid_v05_dev/traces.json")
    args = parser.parse_args()

    config = load_grid_config(_ROOT / args.config)
    wanted = tuple(args.scenarios or DEFAULT_SCENARIOS)
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(args.maps, args.seeds, config)
        if scenario.scenario_id in wanted
    }
    missing = sorted(set(wanted) - set(scenarios))
    if missing:
        raise SystemExit(f"unknown scenarios: {missing}")

    payload: dict[str, Any] = {
        "config": args.config,
        "scenarios": {},
        "episodes": {},
    }
    for scenario_id in wanted:
        scenario = scenarios[scenario_id]
        payload["scenarios"][scenario_id] = {
            "map_seed": int(scenario.map_seed),
            "run_seed": int(scenario.run_seed),
            "scenario_type": scenario.scenario_type,
            "initial_rule": scenario.initial_rule,
            "shift_rule": scenario.shift_rule,
            "shift_time": scenario.shift_time,
        }
        for method, policy_class in METHODS.items():
            episode = trace_episode(config, scenario, policy_class, method)
            payload["episodes"][f"{method}:{scenario_id}"] = episode
            print(
                method,
                scenario_id,
                "success=",
                episode["success"],
                "final=",
                episode["final_rule_estimate"],
                "env_steps=",
                episode["env_steps"],
                "tools=",
                episode["tool_units"],
            )

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
