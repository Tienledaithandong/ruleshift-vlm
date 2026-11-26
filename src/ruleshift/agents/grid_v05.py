"""Separated epistemic/control policies for evaluation protocol v2.

These policies keep the repaired v0.4 control layer and add an explicit,
time-stamped epistemic channel. Agent-side only: no simulator or evaluator
imports, and no access to hidden state.
"""

from __future__ import annotations

from typing import Any

from ruleshift.agents.grid_baselines import (
    _MOVE_KINDS,
    DiagnosticBeliefPolicyV4,
    as_observation,
)
from ruleshift.public_api.grid_rules import (
    RULE_BY_ID,
    cell_role,
    effective_map,
    movement_outcomes,
    sign_clause,
    sign_side,
)
from ruleshift.public_api.grid_schemas import GridAction

__all__ = [
    "DiagnosticBeliefPolicyV4ReadoutV2",
    "DiagnosticBeliefPolicyV5",
    "DiagnosticBeliefPolicyV5NoRequest",
]

_ROLES = frozenset({"goal", "hazard", "safe"})
_DECLARATION_CAP = 6


def _copy_event(event: dict[str, Any]) -> dict[str, Any]:
    clone = {key: value for key, value in event.items() if key != "likelihoods"}
    if "likelihoods" in event:
        clone["likelihoods"] = dict(event["likelihoods"])
    return clone


class _SeparatedV2Policy(DiagnosticBeliefPolicyV4):
    """v4 control plus explicit v2 epistemic decisions.

    The control layer is intentionally inherited without changing its
    navigation, probe selection, two-leader commit gate, or safety fallbacks.
    The epistemic channel is evaluated before every control decision and once
    after termination. A request may influence probe selection only when the
    subclass enables requests.
    """

    _requests_enabled = False
    name = "separated_v2"

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config)
        self._decision_count = 0
        self._events: list[dict[str, Any]] = []
        self._evidence: list[dict[str, Any]] = []
        self._last_declared_rule: str | None = None
        self._last_abstain_reason: str | None = None
        self._pending_request: dict[str, Any] | None = None
        self._request_serial = 0
        self._evidence_serial = 0
        self._finalized = False

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        super().reset(observation, info)
        self._decision_count = 0
        self._events = []
        self._evidence = []
        self._last_declared_rule = None
        self._last_abstain_reason = None
        self._pending_request = None
        self._request_serial = 0
        self._evidence_serial = 0
        self._finalized = False
        self.last_probabilities = self._marginal()

    # -- public ledger ------------------------------------------------------

    def epistemic_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(_copy_event(event) for event in self._events)

    def declaration_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _copy_event(event)
            for event in self._events
            if event["kind"] == "DECLARE_RULE"
        )

    def abstention_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _copy_event(event)
            for event in self._events
            if event["kind"] == "ABSTAIN"
        )

    def request_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _copy_event(event)
            for event in self._events
            if event["kind"] == "REQUEST_EVIDENCE"
        )

    def evidence_ledger(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **{key: value for key, value in item.items() if key != "likelihoods"},
                "likelihoods": dict(item["likelihoods"]),
            }
            for item in self._evidence
        )

    def commitment_events(self) -> tuple[dict[str, Any], ...]:
        """Keep the v1 safe-commit hook separate from v2 declarations."""

        return ()

    # -- decision timing ----------------------------------------------------

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        self._observe_top_rule()
        observation = as_observation(observation)
        self.last_probabilities = self._marginal()
        self._maybe_epistemic_decision(observation, info)
        # The control layer is v4's, including its two-leader commit gate.
        if self._commit_is_safe():
            return self._commit(observation)
        action = self._diagnostic_action(observation)
        if action is not None:
            return action
        return self._commit(observation)

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        previous = self.previous_observation
        previous_info = dict(self.previous_info)
        pending = self._pending_request
        super().record(action, observation, reward, info)
        self._capture_evidence(action, previous, observation, info, previous_info)
        if pending is not None:
            expected = "READ_SIGN" if pending["channel"] == "SIGN" else "INSPECT"
            if action.kind == expected and info.get("last_tool_value") is not None:
                pending["status"] = "executed"
                self._pending_request = None

    def finalize(self, observation: Any, info: dict[str, Any]) -> None:
        """One final public readout after termination; never an env action."""

        if self._finalized:
            return
        self._finalized = True
        observation = as_observation(observation)
        step = int(info.get("env_steps", getattr(observation, "steps", 0)))
        leader, top, margin, refs = self._declaration_status()
        if top >= 0.90 and margin >= 0.10 and refs:
            if leader != self._last_declared_rule:
                self._emit_declaration(
                    leader,
                    top,
                    margin,
                    refs,
                    step,
                    source="final_observation",
                    terminal=True,
                    decision_index=self._decision_count + 1,
                )
        elif self._last_declared_rule is None:
            reason = self._abstain_reason(top, margin, refs)
            self._emit_abstain(
                reason,
                top,
                margin,
                step,
                terminal=True,
                force=True,
                decision_index=self._decision_count + 1,
            )
        if self._pending_request is not None:
            self._pending_request["status"] = "expired"
            self._pending_request = None

    # -- epistemic decision -------------------------------------------------

    def _maybe_epistemic_decision(
        self, observation: Any, info: dict[str, Any]
    ) -> None:
        self._decision_count += 1
        step = int(info.get("env_steps", getattr(observation, "steps", 0)))
        leader, top, margin, refs = self._declaration_status()

        if top >= 0.90 and margin >= 0.10 and refs:
            if leader != self._last_declared_rule:
                self._emit_declaration(
                    leader,
                    top,
                    margin,
                    refs,
                    step,
                    source="pre_action",
                    terminal=False,
                )
            return

        if self._requests_enabled and self._pending_request is None:
            request = self._request_candidate(observation, top, margin)
            if request is not None:
                self._emit_request(request, top, margin, step)
                return

        reason = self._abstain_reason(top, margin, refs)
        self._emit_abstain(reason, top, margin, step, terminal=False, force=False)

    def _declaration_status(self) -> tuple[str, float, float, tuple[int, ...]]:
        marginal = self._marginal()
        if not marginal:
            return "base", 0.0, 0.0, ()
        ordered = sorted(marginal.items(), key=lambda item: (-item[1], item[0]))
        leader, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        refs = tuple(
            int(item["evidence_id"])
            for item in self._evidence
            if float(item["likelihoods"].get(leader, 0.0)) > 1e-12
        )
        return str(leader), float(top), float(top - second), refs

    @staticmethod
    def _abstain_reason(
        top: float, margin: float, refs: tuple[int, ...]
    ) -> str:
        if top < 0.90:
            return "ambiguous" if margin <= 0.05 else "insufficient_evidence"
        if margin < 0.10:
            return "ambiguous"
        if not refs:
            return "insufficient_evidence"
        return "budget_exhausted"

    def _emit_declaration(
        self,
        rule: str,
        top: float,
        margin: float,
        refs: tuple[int, ...],
        step: int,
        *,
        source: str,
        terminal: bool,
        decision_index: int | None = None,
    ) -> None:
        declarations = sum(
            1 for event in self._events if event["kind"] == "DECLARE_RULE"
        )
        if declarations >= _DECLARATION_CAP:
            return
        event = {
            "kind": "DECLARE_RULE",
            "rule_id": str(rule),
            "declared_rule": str(rule),
            "confidence": float(top),
            "posterior_margin": float(margin),
            "step": int(step),
            "decision_index": int(
                self._decision_count if decision_index is None else decision_index
            ),
            "source": str(source),
            "terminal": bool(terminal),
            "evidence_refs": list(refs[-8:]),
        }
        self._events.append(event)
        self._last_declared_rule = str(rule)
        self._last_abstain_reason = None
        if self._pending_request is not None:
            # A declaration resolves the pending request without executing it;
            # it is not a free observation and it is not a new request.
            self._pending_request["status"] = "expired"
            self._pending_request = None

    def _emit_abstain(
        self,
        reason: str,
        top: float,
        margin: float,
        step: int,
        *,
        terminal: bool,
        force: bool,
        decision_index: int | None = None,
    ) -> None:
        if not force and reason == self._last_abstain_reason:
            return
        self._events.append(
            {
                "kind": "ABSTAIN",
                "reason": str(reason),
                "top": float(top),
                "margin": float(margin),
                "step": int(step),
                "decision_index": int(
                    self._decision_count if decision_index is None else decision_index
                ),
                "terminal": bool(terminal),
            }
        )
        self._last_abstain_reason = str(reason)

    # -- evidence requests --------------------------------------------------

    def _request_candidate(
        self, observation: Any, top: float, margin: float
    ) -> dict[str, Any] | None:
        budget = int(observation.tool_budget_remaining)
        if budget <= 0:
            return None
        sign_cost = int(getattr(self.config, "read_sign_cost", 1))
        inspect_cost = int(getattr(self.config, "inspect_cost", 2))
        public_map = observation.public_map
        leaders = sorted(
            self._marginal().items(), key=lambda item: (-item[1], item[0])
        )[:2]

        # Mirror the v4 probe order so a request cannot strand the two-unit
        # inspection budget. First, at exactly the inspection cost, reserve
        # the budget for a blue face that separates the leading rules.
        if (
            budget == inspect_cost
            and self.sign_observed
            and not self.blue_observed
            and public_map.markers.get("blue") is not None
            and self._blue_separates_leaders()
        ):
            return {
                "channel": "BLUE",
                "target": tuple(public_map.markers["blue"]),
            }

        if budget >= sign_cost and public_map.signpost is not None:
            for side in self._informative_sides():
                if side in self.sign_sides_observed:
                    continue
                if len(leaders) == 2 and sign_clause(
                    RULE_BY_ID[leaders[0][0]], side
                ) == sign_clause(RULE_BY_ID[leaders[1][0]], side):
                    continue
                if self._sign_side_gain(side) <= 1e-9:
                    continue
                if not self._sign_cells(
                    public_map, tuple(public_map.signpost), side
                ):
                    continue
                return {"channel": "SIGN", "target": "next_informative_face"}

        if (
            self.sign_observed
            and not self.blue_observed
            and budget >= inspect_cost
            and public_map.markers.get("blue") is not None
            and (
                self._blue_separates_leaders()
                or sum(
                    self._marginal().get(rule, 0.0)
                    for rule in ("hazard_goal_swap", "sign_composition")
                )
                >= 0.05
            )
        ):
            return {
                "channel": "BLUE",
                "target": tuple(public_map.markers["blue"]),
            }
        return None

    def _emit_request(
        self,
        request: dict[str, Any],
        top: float,
        margin: float,
        step: int,
    ) -> None:
        self._request_serial += 1
        event = {
            "kind": "REQUEST_EVIDENCE",
            "request_id": int(self._request_serial),
            "channel": str(request["channel"]),
            "target": request["target"],
            "purpose": "active_evidence_request",
            "status": "pending",
            "top": float(top),
            "margin": float(margin),
            "step": int(step),
            "decision_index": int(self._decision_count),
        }
        self._events.append(event)
        self._pending_request = event

    def _diagnostic_action(self, observation: Any) -> GridAction | None:
        if self._requests_enabled and self._pending_request is not None:
            request = self._pending_request
            action = (
                self._go_read_sign(observation)
                if request["channel"] == "SIGN"
                else self._go_inspect_blue(observation)
            )
            if action is not None:
                return action
            request["status"] = "unmet"
            self._pending_request = None
        return super()._diagnostic_action(observation)

    # -- evidence ledger ----------------------------------------------------

    def _capture_evidence(
        self,
        action: GridAction,
        previous: Any,
        observation: Any,
        info: dict[str, Any],
        previous_info: dict[str, Any],
    ) -> None:
        if previous is None:
            return
        previous = as_observation(previous)
        observation = as_observation(observation)
        step = int(info.get("env_steps", getattr(observation, "steps", 0)))
        kind = action.kind
        likelihoods: dict[str, float] | None = None
        observed: Any = None
        evidence_kind = ""

        if kind == "READ_SIGN" and info.get("last_tool_value") is not None:
            signpost = previous.public_map.signpost
            if signpost is not None:
                try:
                    side = sign_side(tuple(previous.agent_position), tuple(signpost))
                except ValueError:
                    side = None
                if side is not None:
                    observed = str(info["last_tool_value"])
                    evidence_kind = "sign"
                    likelihoods = {
                        rule: 1.0
                        if sign_clause(RULE_BY_ID[rule], side) == observed
                        else 1e-6
                        for rule in RULE_BY_ID
                    }
        elif kind == "INSPECT" and info.get("last_tool_value") is not None:
            color = self._inspect_color(previous.public_map, action.target)
            observed = str(info["last_tool_value"])
            if color and observed in _ROLES:
                evidence_kind = "inspect"
                error = float(getattr(self.config, "inspect_error", 0.05))
                likelihoods = {
                    rule: (
                        1.0 - error
                        if cell_role(RULE_BY_ID[rule], color) == observed
                        else error / 2.0
                    )
                    for rule in RULE_BY_ID
                }
        elif kind in _MOVE_KINDS:
            previous_step = int(previous_info.get("env_steps", 0))
            if step > previous_step:
                observed = tuple(observation.agent_position)
                evidence_kind = "move"
                public_map = effective_map(
                    previous.public_map,
                    door_open=bool(getattr(previous, "door_open", False)),
                )
                likelihoods = {}
                for rule in RULE_BY_ID:
                    outcomes = movement_outcomes(
                        RULE_BY_ID[rule],
                        tuple(previous.agent_position),
                        kind,
                        public_map,
                    )
                    likelihoods[rule] = float(outcomes.get(tuple(observed), 0.0))

        if likelihoods is None or not evidence_kind:
            return
        self._evidence_serial += 1
        self._evidence.append(
            {
                "evidence_id": int(self._evidence_serial),
                "decision_index": int(self._decision_count),
                "step": int(step),
                "kind": evidence_kind,
                "observed": observed,
                "likelihoods": likelihoods,
            }
        )


class DiagnosticBeliefPolicyV4ReadoutV2(_SeparatedV2Policy):
    """Matched control B: v4 control, v2 readout, no active requests."""

    _requests_enabled = False
    name = "diagnostic_belief_v4_readout_v2"


class DiagnosticBeliefPolicyV5(_SeparatedV2Policy):
    """Candidate C: v4-compatible control with explicit evidence requests."""

    _requests_enabled = True
    name = "diagnostic_belief_v5"


class DiagnosticBeliefPolicyV5NoRequest(_SeparatedV2Policy):
    """Ablation D: separated readout, active requests disabled."""

    _requests_enabled = False
    name = "diagnostic_belief_v5_no_request"
