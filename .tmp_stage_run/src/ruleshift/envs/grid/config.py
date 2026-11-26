"""Grid scenario configuration and YAML loading.

This module is additive: it does not touch the Family A/B ``envs.config``.
``baseline_slip`` and ``slippery_slip`` are the simulator's source of truth;
the environment applies them to the public rule objects at reset. Health is an
integer that floors at zero in Tier A (hence ``max_health``, not ``max_damage``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GridConfig:
    """One fully resolved RuleShift-Grid development scenario."""

    width: int = 7
    height: int = 7
    wall_density: float = 0.18
    max_env_steps: int = 60
    tool_budget: int = 3
    inspect_cost: int = 2
    read_sign_cost: int = 1
    inspect_range: int = 2
    inspect_error: float = 0.05
    baseline_slip: float = 0.10
    slippery_slip: float = 0.80
    shift_probability: float = 0.35
    shift_window: tuple[int, int] = (10, 40)
    soft_hazard_damage: float = -2.0
    soft_hazard_recovery_steps: int = 1
    goal_reward: float = 10.0
    step_cost: float = -0.05
    max_health: int = 3
    lethal_trap: bool = False
    door_required: bool = False
    advisor_cost: int = 2
    max_decisions: int = 200


def load_grid_config(path: str | Path) -> GridConfig:
    """Load a grid config from YAML, keeping only known dataclass fields.

    Informational keys such as ``render_size`` and ``observation_mode`` are
    accepted but ignored in v1.
    """

    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("grid config must be a YAML mapping")

    known = GridConfig.__dataclass_fields__
    overrides = {key: value for key, value in data.items() if key in known}
    if "shift_window" in overrides:
        overrides["shift_window"] = tuple(overrides["shift_window"])
    return GridConfig(**overrides)
