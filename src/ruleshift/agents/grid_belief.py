"""Public change-point belief over RuleShift-Grid rule hypotheses.

This module is purely agent-side and dependency-light: it consumes only the
public rule/menu helpers in :mod:`ruleshift.public_api` and the standard
library. It never imports the simulator or the evaluator and it never reads the
true rule -- every likelihood below is computed from public helper functions.

The state is a distribution over ``(rule_id, tau)`` pairs. ``tau is None`` means
"no change point has been recorded yet"; an integer ``tau`` is the step at which
the most recent change point occurred. A change point is a hazard event: with
probability ``p_shift`` (only inside ``shift_window``) the filter records a new
change point at the current step while retaining the current rule label. The
rule label itself is learned from observations; a change point is not itself a
claim that the rule changed (a shift may leave it unchanged, and that case is
observationally indistinguishable from "no shift").

All update methods share the same shape: apply the hazard, multiply by a public
likelihood, and renormalize. Zero-probability observations are floored with a
small epsilon so the filter is never degenerate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from ruleshift.public_api.grid_rules import (
    Coord,
    PublicGridMap,
    RULE_BY_ID,
    cell_role,
    key_opens,
    movement_outcomes,
)
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GridAction

__all__ = ["GridBelief"]

#: Floor for an impossible movement outcome (a cell with zero public mass).
MOVE_EPS = 1e-9
#: Mismatch floor for a deterministic role/clause observation.
MISMATCH_EPS = 1e-6
#: Mismatch floor for an interaction outcome inconsistent with a rule.
INTERACT_EPS = 1e-6
#: Default tool noise, matching the development ``GridConfig``.
DEFAULT_INSPECT_ERROR = 0.05

_State = tuple[str, int | None]


class GridBelief:
    """A hazard-rate change-point filter over ``(rule_id, tau)`` states.

    Args:
        hypotheses: Public rule ids to consider, in tie-break order.
        p_shift: Per-step change-point hazard, applied only inside the window.
        shift_window: Inclusive ``(start, end)`` step range for the hazard.
        prior: Optional mapping ``rule_id -> mass``; missing ids get zero mass.
            Defaults to a uniform prior over ``hypotheses``.
        inspect_error: Noise of an INSPECT observation. On a match the
            likelihood is ``1 - inspect_error``; on a mismatch it is
            ``inspect_error / 2`` (one of the two other public roles).
    """

    def __init__(
        self,
        hypotheses: tuple[str, ...] = GRID_RULE_IDS,
        p_shift: float = 0.02,
        shift_window: tuple[int, int] = (0, 60),
        prior: Mapping[str, float] | None = None,
        inspect_error: float = DEFAULT_INSPECT_ERROR,
    ) -> None:
        self.hypotheses = tuple(hypotheses)
        if not self.hypotheses:
            raise ValueError("hypotheses must be non-empty")
        unknown = [rule for rule in self.hypotheses if rule not in RULE_BY_ID]
        if unknown:
            raise ValueError(f"unknown rule hypotheses: {unknown}")
        if len(set(self.hypotheses)) != len(self.hypotheses):
            raise ValueError("hypotheses must not contain duplicates")

        self.p_shift = float(p_shift)
        if not 0.0 <= self.p_shift <= 1.0:
            raise ValueError("p_shift must lie in [0, 1]")
        self.shift_window = (int(shift_window[0]), int(shift_window[1]))
        self.inspect_error = float(inspect_error)
        if not 0.0 <= self.inspect_error < 1.0:
            raise ValueError("inspect_error must lie in [0, 1)")

        masses = {
            rule: float(prior.get(rule, 0.0)) if prior is not None else 1.0
            for rule in self.hypotheses
        }
        total = sum(masses.values())
        if total <= 0.0:
            masses = {rule: 1.0 for rule in self.hypotheses}
            total = float(len(self.hypotheses))
        self.states: dict[_State, float] = {
            (rule, None): masses[rule] / total for rule in self.hypotheses
        }
        # The hazard is a per-physical-step rate: repeated predictions at the
        # same step must not compound it. Copies inherit this marker.
        self._last_predicted_step: int | None = None

    # ------------------------------------------------------------------
    # Prediction and conditioning
    # ------------------------------------------------------------------

    def predict(self, step: int) -> None:
        """Apply the change-point hazard for ``step`` in place.

        The hazard is a rate *per physical step*, so this method is idempotent
        within a step: a second call with the same ``step`` is a no-op, which
        allows several observation updates at one step without compounding the
        hazard. A higher step advances the hazard normally. Outside
        ``shift_window`` the hazard is zero and the distribution is unchanged.
        Inside it, each state ``(rule, tau)`` keeps its rule and either retains
        ``tau`` (probability ``1 - p_shift``) or records a change point at
        ``step`` (probability ``p_shift``).
        """
        if step == self._last_predicted_step:
            return
        self._last_predicted_step = step
        start, end = self.shift_window
        hazard = self.p_shift if start <= step <= end else 0.0
        if hazard <= 0.0:
            return

        updated: dict[_State, float] = {}
        for (rule, tau), probability in self.states.items():
            if probability == 0.0:
                continue
            updated[(rule, tau)] = updated.get((rule, tau), 0.0) + probability * (
                1.0 - hazard
            )
            updated[(rule, step)] = updated.get((rule, step), 0.0) + probability * hazard
        self.states = {key: value for key, value in updated.items() if value > 0.0}

    def _condition(
        self, step: int, likelihood: Callable[[str], float]
    ) -> None:
        """Predict, multiply by ``likelihood(rule_id)``, and renormalize."""

        before = dict(self.states)
        before_step = self._last_predicted_step
        self.predict(step)
        weighted: dict[_State, float] = {}
        total = 0.0
        for (rule, tau), probability in self.states.items():
            weight = probability * likelihood(rule)
            weighted[(rule, tau)] = weight
            total += weight
        if total <= 0.0:
            # Never let a degenerate likelihood destroy the belief.
            self.states = before
            self._last_predicted_step = before_step
            return
        self.states = {key: value / total for key, value in weighted.items()}

    # ------------------------------------------------------------------
    # Public observations
    # ------------------------------------------------------------------

    def update_move(
        self,
        previous_position: Coord,
        action: str,
        observed_position: Coord,
        public_map: PublicGridMap,
        step: int,
    ) -> None:
        """Condition on a physical move's observed landing cell."""

        if isinstance(action, GridAction):
            action = action.kind
        cache: dict[str, dict[Coord, float]] = {}

        def likelihood(rule_id: str) -> float:
            if rule_id not in cache:
                cache[rule_id] = movement_outcomes(
                    RULE_BY_ID[rule_id], previous_position, action, public_map
                )
            return cache[rule_id].get(tuple(observed_position), MOVE_EPS)

        self._condition(step, likelihood)

    def update_role(self, color: str, observed_role: str, step: int) -> None:
        """Condition on a directly observed marker role (1e-6 mismatch)."""

        self._condition(
            step,
            lambda rule_id: 1.0
            if cell_role(RULE_BY_ID[rule_id], color) == observed_role
            else MISMATCH_EPS,
        )

    def update_inspect(self, color: str, observed_role: str, step: int) -> None:
        """Condition on a noisy INSPECT role for one marker color."""

        error = self.inspect_error

        def likelihood(rule_id: str) -> float:
            if cell_role(RULE_BY_ID[rule_id], color) == observed_role:
                return 1.0 - error
            return error / 2.0

        self._condition(step, likelihood)

    def update_sign(self, side: int, clause: str, step: int) -> None:
        """Condition on one observed signpost clause (1e-6 mismatch)."""

        def likelihood(rule_id: str) -> float:
            clauses = RULE_BY_ID[rule_id].sign_clauses
            if not isinstance(side, int) or not 0 <= side < len(clauses):
                return MISMATCH_EPS
            return 1.0 if clauses[side] == clause else MISMATCH_EPS

        self._condition(step, likelihood)

    def update_interact(
        self,
        outcome: str,
        step: int,
        key_color: str = "green",
        door_color: str = "green",
    ) -> None:
        """Condition on an INTERACT outcome.

        ``key_picked`` is consistent with every rule; ``door_opened`` is
        consistent only when ``key_opens`` holds under the candidate rule;
        everything else (``nothing``, ``door_locked``, ``interact``) is
        uninformative. In v1 every public rule shares the same key/door colors,
        so the door check is truthful but carries no rule information.
        """

        normalized = str(outcome).lower()

        def likelihood(rule_id: str) -> float:
            rule = RULE_BY_ID[rule_id]
            if normalized in {"key_picked", "key_picked_up", "key"}:
                return 1.0
            if normalized in {"door_opened", "door"}:
                return 1.0 if key_opens(rule, key_color, door_color) else INTERACT_EPS
            return 1.0  # "nothing" and other uninformative outcomes

        self._condition(step, likelihood)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def marginal(self) -> dict[str, float]:
        """Collapse the change-point dimension: ``rule_id -> probability``."""

        result = {rule: 0.0 for rule in self.hypotheses}
        for (rule, _tau), probability in self.states.items():
            result[rule] = result.get(rule, 0.0) + probability
        return result

    def most_likely(self) -> str:
        """Return the highest-mass rule id with hypothesis-order tie-break."""

        marginal = self.marginal()
        return max(self.hypotheses, key=lambda rule: marginal[rule])

    def entropy(self) -> float:
        """Shannon entropy of the rule marginal, in bits."""

        return -sum(
            probability * math.log2(probability)
            for probability in self.marginal().values()
            if probability > 0.0
        )

    def shift_probability(self) -> float:
        """Total mass assigned to states with a recorded change point."""

        return sum(p for (_rule, tau), p in self.states.items() if tau is not None)

    def copy(self) -> GridBelief:
        """Return a deep copy that can be updated without touching this one."""

        return deepcopy(self)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable public snapshot (no hidden fields)."""

        return {
            "marginal": dict(self.marginal()),
            "shift_probability": self.shift_probability(),
            "entropy": self.entropy(),
        }
