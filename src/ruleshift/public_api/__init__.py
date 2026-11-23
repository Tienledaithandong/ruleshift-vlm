"""Agent-visible interface package.

Only observation schemas, action affordances and public cost descriptions live
here. Privileged simulator state and evaluator labels never enter this package.
"""

from . import grid_rules, grid_schemas, interface, schemas

__all__ = ["grid_rules", "grid_schemas", "interface", "schemas"]
