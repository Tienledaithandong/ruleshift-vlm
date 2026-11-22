"""Simulator-side package: hidden state, transitions and privileged labels.

Agent code must never import this package. The public boundary is exposed by
:mod:`ruleshift.public_api`; the evaluator boundary by
:mod:`ruleshift.evaluator`.
"""

import gymnasium as gym

from .bridge_hazard import BridgeConfig, BridgeHazardEnv
from .events import SemanticRNG
from .gate_lock import RuleShiftEnv

if "RuleShift-GateLock-v0" not in gym.registry:
    gym.register(
        id="RuleShift-GateLock-v0",
        entry_point="ruleshift.envs:RuleShiftEnv",
        max_episode_steps=60,
    )
if "RuleShift-BridgeHazard-v0" not in gym.registry:
    gym.register(
        id="RuleShift-BridgeHazard-v0",
        entry_point="ruleshift.envs:BridgeHazardEnv",
        max_episode_steps=60,
    )

__all__ = ["BridgeConfig", "BridgeHazardEnv", "RuleShiftEnv", "SemanticRNG"]
