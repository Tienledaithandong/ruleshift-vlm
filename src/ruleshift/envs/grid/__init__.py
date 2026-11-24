"""Simulator-side RuleShift-Grid package.

The public boundary stays in :mod:`ruleshift.public_api`; this package owns
configuration, entities and the deterministic map generator only. Grid
scenarios and the environment itself arrive in a later task.
"""

from .config import GridConfig, load_grid_config
from .entities import GridMap, MapGenResult
from .mapgen import all_hypotheses_solvable, generate_solvable_map, is_solvable

__all__ = [
    "GridConfig",
    "GridMap",
    "MapGenResult",
    "all_hypotheses_solvable",
    "generate_solvable_map",
    "is_solvable",
    "load_grid_config",
]
