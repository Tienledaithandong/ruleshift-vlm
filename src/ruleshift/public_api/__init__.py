"""Agent-visible interface package.

Only observation schemas, action affordances and public cost descriptions live
here. Privileged simulator state and evaluator labels never enter this package.
"""

from . import interface, schemas

__all__ = ["interface", "schemas"]
