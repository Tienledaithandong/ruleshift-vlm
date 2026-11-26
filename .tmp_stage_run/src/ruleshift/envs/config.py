"""Scenario configuration and development-grid loading."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import yaml

def default_config_path(name: str) -> Path:
    """Find the checkout config or the config installed alongside a wheel."""
    package = Path(__file__).resolve()
    if package.parents[2].name == "src":
        checkout = package.parents[3] / "configs" / name
        if checkout.is_file():
            return checkout
    dist = distribution("ruleshift-vlm")
    shared = Path("share", "ruleshift", "configs", name)
    # pip --target keeps data under the target itself. Standard installations
    # record the installed data path relative to the package directory.
    target = Path(dist.locate_file(shared))
    if target.is_file():
        return target
    for entry in dist.files or ():
        if Path(str(entry)).parts[-4:] == shared.parts:
            installed = Path(dist.locate_file(entry))
            if installed.is_file():
                return installed
    raise FileNotFoundError(f"missing RuleShift config {name}")


DEFAULT_DEV_PATH = default_config_path("dev.yaml")

HAZARD_BY_CONDITION = {
    "no_shift": 0.0,
    "low_hazard": 0.02,
    "high_hazard": 0.10,
}


@dataclass(frozen=True)
class EnvConfig:
    """One fully resolved scenario (dev hypotheses, never a frozen test)."""

    grid_width: int = 5
    grid_height: int = 5
    max_env_steps: int = 60
    seed_rule: str = "red"
    candidate_rules: tuple[str, ...] = ("red", "blue", "green")
    delivery_reward: float = 10.0
    step_cost: float = -0.05
    lockout_steps: int = 3
    probe_detour_steps: int = 2
    feedback_noise: float = 0.10
    shift_hazard: float = 0.03
    disable_shift: bool = False
    initial_mechanic: str = "red"
    shift_targets: tuple[str, ...] = ("blue", "green")
    scenario_id: str = "dev-default"

    def __post_init__(self) -> None:
        if min(self.grid_width, self.grid_height, self.max_env_steps) < 1:
            raise ValueError("grid dimensions and primitive horizon must be positive")
        if not 0 <= self.probe_detour_steps <= self.grid_width + self.grid_height - 2:
            raise ValueError("station detour must fit inside the grid")
        if self.lockout_steps < 0:
            raise ValueError("lockout steps must be non-negative")
        if not 0 <= self.feedback_noise <= 1 or not 0 <= self.shift_hazard <= 1:
            raise ValueError("feedback noise and shift hazard must be probabilities")


def config_with_overrides(config: EnvConfig, overrides: dict[str, Any]) -> EnvConfig:
    """Apply only known config fields from reset options."""

    known = {field.name for field in fields(config)}
    return replace(config, **{key: value for key, value in overrides.items() if key in known})


def load_dev_grid(path: Path | None = None) -> dict[str, Any]:
    """Load the development grid from YAML (without importing agent code)."""

    target = Path(path) if path is not None else DEFAULT_DEV_PATH
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("dev config must be a YAML mapping")
    return data


def config_from_dev_cell(cell_id: str, path: Path | None = None) -> EnvConfig:
    """Resolve one hypothesised dev cell into a concrete EnvConfig."""

    grid = load_dev_grid(path)
    cells = {cell["id"]: cell for cell in grid["hypothesised_cells"]}
    if cell_id not in cells:
        raise KeyError(f"unknown dev cell: {cell_id}")
    cell = cells[cell_id]
    condition = cell["shift_condition"]
    return EnvConfig(
        grid_width=int(grid["environment"]["grid_size"][0]),
        grid_height=int(grid["environment"]["grid_size"][1]),
        max_env_steps=int(grid["environment"]["max_env_steps"]),
        seed_rule=grid["environment"]["seed_rule"],
        candidate_rules=tuple(grid["environment"]["candidate_rules"]),
        delivery_reward=float(grid["environment"]["delivery_reward"]),
        step_cost=float(grid["environment"]["step_cost"]),
        probe_detour_steps=int(cell["probe_detour_steps"]),
        feedback_noise=float(cell["feedback_noise_rate"]),
        lockout_steps=int(cell["lockout_steps"]),
        shift_hazard=HAZARD_BY_CONDITION[condition],
        disable_shift=(condition == "no_shift"),
        initial_mechanic=grid["environment"]["seed_rule"],
        scenario_id=f"dev-{cell_id}",
    )
