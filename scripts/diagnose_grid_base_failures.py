"""Trace selected true-base failures with the diagnostic policy.

This is a post-hoc root-cause diagnostic, not a new benchmark run. It replays
only scenarios named by the base-stress analysis, records the policy posterior
before each decision, and writes public decision traces plus the privileged
true rule at the episode level. No policy or benchmark code is changed.
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

from ruleshift.agents.grid_baselines import DiagnosticBeliefPolicy, as_observation
from ruleshift.envs.grid.config import GridConfig, load_grid_config
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.evaluator.grid_runner import build_grid_scenarios
from ruleshift.public_api.grid_rules import sign_side

ROOT = _ROOT
DEFAULT_ANALYSIS = "results/grid_phase0/base_stress_100x5_analysis/analysis.json"
DEFAULT_OUT = "results/grid_phase0/base_failure_traces.json"


class TracedDiagnosticBeliefPolicy(DiagnosticBeliefPolicy):
    """Record belief/action state without changing the policy decision."""

    def __init__(self, config: GridConfig | None = None) -> None:
        super().__init__(config)
        self.trace: list[dict[str, Any]] = []

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self.trace = []

    def choose(self, observation: Any, info: dict[str, Any]) -> Any:
        observation = as_observation(observation)
        action = super().choose(observation, info)
        belief = self.belief.marginal() if self.belief is not None else {}
        top = max(belief.values(), default=0.0)
        second = sorted(belief.values(), reverse=True)[1] if len(belief) > 1 else 0.0
        map_rule = self.most_likely_rule()
        map_plan = self.base_plan(map_rule, observation) if self.belief is not None else []
        signpost = observation.public_map.signpost
        try:
            current_sign_side = (
                sign_side(tuple(observation.agent_position), tuple(signpost))
                if signpost is not None
                else None
            )
        except ValueError:
            current_sign_side = None
        self.trace.append({
            "map_rule": map_rule,
            "map_plan_length": len(map_plan),
            "signpost": list(signpost) if signpost is not None else None,
            "sign_side": current_sign_side,
            "decision": len(self.trace),
            "env_step": int(info.get("env_steps", 0)),
            "position": list(observation.agent_position),
            "tool_budget_remaining": int(observation.tool_budget_remaining),
            "inventory": list(observation.inventory),
            "door_open": bool(observation.door_open),
            "action": action.kind,
            "target": list(action.target) if action.target is not None else None,
            "belief": belief,
            "top_probability": top,
            "second_probability": second,
            "commit_safe": bool(self._commit_is_safe()),
            "sign_observed": bool(self.sign_observed),
            "blue_observed": bool(self.blue_observed),
            "tool_attempts": int(self.tool_attempts),
        })
        return action

    def record(
        self,
        action: Any,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        super().record(action, observation, reward, info)
        observation = as_observation(observation)
        if self.trace:
            self.trace[-1].update(
                {
                    "reward": float(reward),
                    "next_position": list(observation.agent_position),
                    "next_env_step": int(info.get("env_steps", 0)),
                    "next_tool_budget_remaining": int(
                        info.get("tool_budget_remaining", observation.tool_budget_remaining)
                    ),
                    "last_tool_value": info.get("last_tool_value"),
                    "success_after_step": bool(info.get("success", False)),
                }
            )


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [*payload["diagnostic_failures"], *payload["diagnostic_misdiagnoses"]]
    return sorted({str(record["scenario_id"]) for record in records})


def _run_scenario(
    scenario: Any, config: GridConfig
) -> dict[str, Any]:
    env = GridEnv(config)
    observation, info = env.reset(
        seed=scenario.run_seed, options={"scenario": scenario}
    )
    policy = TracedDiagnosticBeliefPolicy(config)
    policy.reset(observation, info)
    success = False
    max_decisions = int(getattr(config, "max_decisions", 200))
    for _ in range(max_decisions):
        action = policy.choose(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        policy.record(action, observation, reward, info)
        success = bool(info.get("success", False))
        if terminated or truncated:
            break
    final_info = info
    tool_units = int(config.tool_budget) - int(
        final_info.get("tool_budget_remaining", config.tool_budget)
    )
    return {
        "scenario_id": scenario.scenario_id,
        "map_seed": scenario.map_seed,
        "run_seed": scenario.run_seed,
        "true_rule": str(env.evaluator_state()["true_rule"]),
        "success": success,
        "final_rule_estimate": policy.most_likely_rule(),
        "final_belief": policy.belief.marginal() if policy.belief is not None else {},
        "env_steps": int(final_info.get("env_steps", 0)),
        "tool_units": tool_units,
        "trace": policy.trace,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_grid_config(ROOT / args.config)
    scenario_ids = args.scenario_id or _load_ids(ROOT / args.analysis)
    scenarios = {
        scenario.scenario_id: scenario
        for scenario in build_grid_scenarios(100, 5, config)
        if scenario.initial_rule == "base" and scenario.shift_rule is None
    }
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios]
    if missing:
        raise SystemExit(f"unknown base scenario ids: {missing}")
    episodes = [_run_scenario(scenarios[scenario_id], config) for scenario_id in scenario_ids]
    payload = {
        "study": "base_failure_trace",
        "note": (
            "Post-hoc diagnostic trace; the true rule is stored only at episode "
            "level and is never fed back to the policy."
        ),
        "policy": "diagnostic_belief",
        "episodes": episodes,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "episodes": [
            {
                "scenario_id": episode["scenario_id"],
                "success": episode["success"],
                "final_rule_estimate": episode["final_rule_estimate"],
                "env_steps": episode["env_steps"],
                "tool_units": episode["tool_units"],
            }
            for episode in episodes
        ],
    }, indent=2))


if __name__ == "__main__":
    main()