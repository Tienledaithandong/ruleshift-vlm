"""Rule/scenario sampling for RuleShift-Grid.

The simulator samples an entire episode specification from one integer seed.
The public rule menu is reused verbatim from :mod:`public_api.grid_rules`; this
module only chooses which rule is active (and when it shifts) -- it never
changes the public descriptions or probabilities.
"""

from __future__ import annotations

import random

from ruleshift.public_api.grid_rules import RULE_BY_ID

from .config import GridConfig
from .entities import GridScenario

#: Public rule objects, keyed by id. Re-exported for simulator callers.
RULE_OBJECTS = RULE_BY_ID

_SCENARIO_TYPES = ("identification", "change_detection", "no_shift")


def sample_scenario(
    seed: int, config: GridConfig, scenario_type: str | None = None
) -> GridScenario:
    """Sample a deterministic scenario from ``seed`` and ``config``.

    ``map_seed`` and ``run_seed`` are both set to ``seed``: the former drives
    map generation, the latter the semantic noise streams. Callers that want a
    different map for the same draws construct :class:`GridScenario` directly.
    """

    rng = random.Random(seed)
    if scenario_type is None:
        scenario_type = (
            "change_detection"
            if rng.random() < config.shift_probability
            else "no_shift"
        )
    if scenario_type not in _SCENARIO_TYPES:
        raise ValueError(f"unknown grid scenario type {scenario_type!r}")

    shift_rule: str | None = None
    shift_time: int | None = None
    if scenario_type == "identification":
        initial_rule = rng.choice(tuple(RULE_OBJECTS))
    else:
        initial_rule = "base"
        if scenario_type == "change_detection":
            non_base = tuple(rule_id for rule_id in RULE_OBJECTS if rule_id != "base")
            shift_rule = rng.choice(non_base)
            latest = max(1, int(config.max_env_steps) - 1)
            low = min(max(1, int(config.shift_window[0])), latest)
            high = min(max(1, int(config.shift_window[1])), latest)
            if high < low:
                high = low
            shift_time = rng.randint(low, high)

    return GridScenario(
        scenario_id=f"grid-{seed}",
        scenario_type=scenario_type,
        initial_rule=initial_rule,
        shift_rule=shift_rule,
        shift_time=shift_time,
        map_seed=seed,
        run_seed=seed,
    )


__all__ = ["RULE_OBJECTS", "sample_scenario"]
