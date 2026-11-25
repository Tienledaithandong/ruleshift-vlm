"""Deterministic CPU smoke test for the Week 2 Gate/Lock environment.

Usage:
    python scripts/smoke_env.py --seed 42 --family gate --decisions 5

Prints one JSON line per macro decision using ONLY public fields, followed by
one clearly labelled evaluator-only line. The two streams must never be joined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv

DEFAULT_ACTIONS = ["PROBE_RED", "WAIT", "COMMIT_RED", "COMMIT_BLUE", "WAIT"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", choices=["gate"], default="gate")
    parser.add_argument("--decisions", type=int, default=5)
    args = parser.parse_args()

    env = RuleShiftEnv(config=EnvConfig(scenario_id=f"smoke-{args.seed}"))
    env.reset(seed=args.seed)

    total = 0.0
    for index, action in enumerate(DEFAULT_ACTIONS[: args.decisions]):
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        public = env.public_view().to_payload()
        print(
            json.dumps(
                {
                    "decision": index,
                    "action": action,
                    "reward": reward,
                    "observation_id": public["observation_id"],
                    "public_obs_hash": info["public_obs_hash"],
                    "public_history_hash": info["public_history_hash"],
                    "public_text": public["public_text"],
                    "available_actions": public["available_actions"],
                    "physical_steps_elapsed": info["physical_steps_elapsed"],
                },
                sort_keys=True,
            )
        )
        if terminated or truncated:
            break

    print("--- evaluator-only trace (never pass to an agent) ---")
    print(json.dumps({"return": total, **env.evaluator_state()}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
