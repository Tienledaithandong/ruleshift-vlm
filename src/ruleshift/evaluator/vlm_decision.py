"""CPU reference for the VLM probe-versus-commit decision microbenchmark.

This module is evaluator-side. A context contains only public facts: the public
map, the candidate rule menu, the public prior, the observed tool outcomes, the
remaining tool budget, and the feasible high-level actions. The exact reference
values and the latent rule used for declaration scoring are stored separately
in ReferenceLabel and must never be placed in a model prompt.

The decision is intentionally a local macro decision: navigation to a probe or
to the committed goal is delegated to the public planner. The reference values
information and control consequences, not primitive movement optimisation.
Family A's exact solver is not reused unchanged: it has three Gate/Lock rules
and different probe/commit macros, whereas this benchmark has six grid rules
and sign/inspect evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_planner import plan_to_goal
from ruleshift.public_api.grid_rules import (
    Coord,
    PublicGridMap,
    RULE_BY_ID,
    cell_role,
    sign_clause,
    sign_side,
)
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS

__all__ = [
    "ACTIONS",
    "ABSTAIN",
    "DECLARE_RULE",
    "DecisionContext",
    "DecisionLibrary",
    "DecisionPrediction",
    "DecisionScore",
    "DeclarationPrediction",
    "Evidence",
    "ReferenceLabel",
    "REFERENCE_VERSION",
    "ToolCosts",
    "audit_library",
    "baseline_predictions",
    "build_development_contexts",
    "build_development_library",
    "declaration_prediction",
    "information_gain_bits",
    "reference_for",
    "score_declarations",
    "score_decisions",
    "summarize_scores",
]

ACTIONS = ("COMMIT", "PROBE_SIGN", "PROBE_BLUE")
TIE_ORDER = ("COMMIT", "PROBE_SIGN", "PROBE_BLUE")
REFERENCE_VERSION = "vlm-decision-reference-v1"
DECLARE_RULE = "DECLARE_RULE"
ABSTAIN = "ABSTAIN"
DEFAULT_DECLARATION_THRESHOLD = 0.90
DEFAULT_DECLARATION_MARGIN = 0.10
TIE_TOLERANCE = 1e-9
NEAR_TIE_MARGIN = 0.25
DEFAULT_SUCCESS_REWARD = 10.0
DEFAULT_HAZARD_PENALTY = -2.0
DEFAULT_STEP_COST = 0.05
ROLES = ("goal", "hazard", "safe")


@dataclass(frozen=True)
class ToolCosts:
    sign: int = 1
    blue: int = 2

    def __post_init__(self) -> None:
        if self.sign < 0 or self.blue < 0:
            raise ValueError("tool costs must be non-negative")
        if self.sign == 0 and self.blue == 0:
            raise ValueError("at least one probe must have a positive cost")

    def cost(self, action: str) -> int:
        if action == "PROBE_SIGN":
            return self.sign
        if action == "PROBE_BLUE":
            return self.blue
        return 0

    def as_payload(self) -> dict[str, int]:
        return {"sign": self.sign, "blue": self.blue}


@dataclass(frozen=True)
class Evidence:
    kind: str
    value: str
    side: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("sign", "blue"):
            raise ValueError("evidence kind must be sign or blue")
        if self.kind == "sign":
            if self.side not in (0, 1, 2):
                raise ValueError("sign evidence needs side in {0,1,2}")
            if not self.value:
                raise ValueError("sign evidence needs a clause")
        elif self.side is not None:
            raise ValueError("blue evidence has no side")
        if self.kind == "blue" and self.value not in ROLES:
            raise ValueError("blue evidence must be goal/hazard/safe")

    def cost(self, costs: ToolCosts) -> int:
        return costs.sign if self.kind == "sign" else costs.blue

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "value": self.value}
        if self.side is not None:
            payload["side"] = self.side
        return payload


@dataclass(frozen=True)
class DecisionContext:
    context_id: str
    layout_id: str
    public_map: PublicGridMap
    agent_position: Coord
    candidate_rules: tuple[str, ...]
    prior: tuple[float, ...]
    posterior: tuple[float, ...]
    evidence: tuple[Evidence, ...]
    sign_side: int | None
    inspect_error: float
    costs: ToolCosts
    tool_budget: int
    max_probes: int
    commit_steps_by_rule: Mapping[str, int]
    step_cost: float = DEFAULT_STEP_COST
    success_reward: float = DEFAULT_SUCCESS_REWARD
    hazard_penalty: float = DEFAULT_HAZARD_PENALTY
    stratum: str = "unassigned"

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id is required")
        if not self.candidate_rules or len(set(self.candidate_rules)) != len(
            self.candidate_rules
        ):
            raise ValueError("candidate_rules must be non-empty and unique")
        unknown = [rule for rule in self.candidate_rules if rule not in RULE_BY_ID]
        if unknown:
            raise ValueError(f"unknown candidate rules: {unknown}")
        if len(self.prior) != len(self.candidate_rules):
            raise ValueError("prior length does not match candidate_rules")
        if len(self.posterior) != len(self.candidate_rules):
            raise ValueError("posterior length does not match candidate_rules")
        for name, masses in (("prior", self.prior), ("posterior", self.posterior)):
            if any(not math.isfinite(value) or value < 0.0 for value in masses):
                raise ValueError(f"{name} masses must be finite and non-negative")
            if abs(sum(masses) - 1.0) > 1e-8:
                raise ValueError(f"{name} must sum to one")
        if self.tool_budget < 0:
            raise ValueError("tool_budget must be non-negative")
        if self.max_probes < 0:
            raise ValueError("max_probes must be non-negative")
        if not 0.0 <= self.inspect_error < 1.0:
            raise ValueError("inspect_error must lie in [0, 1)")
        if self.step_cost < 0.0:
            raise ValueError("step_cost must be non-negative")
        if self.sign_side is not None and self.sign_side not in (0, 1, 2):
            raise ValueError("sign_side must be 0, 1, 2, or None")
        if self.sign_side is not None and self.public_map.signpost is None:
            raise ValueError("sign_side requires a public signpost")
        if "blue" not in self.public_map.markers:
            raise ValueError("PROBE_BLUE requires a public blue marker")
        for rule in self.candidate_rules:
            if rule not in self.commit_steps_by_rule:
                raise ValueError(f"missing commit steps for {rule}")
            if self.commit_steps_by_rule[rule] < 0:
                raise ValueError("commit steps must be non-negative")

    @property
    def spent_units(self) -> int:
        return sum(item.cost(self.costs) for item in self.evidence)

    @property
    def remaining_budget(self) -> int:
        return self.tool_budget - self.spent_units

    @property
    def used_sign(self) -> bool:
        return any(item.kind == "sign" for item in self.evidence)

    @property
    def used_blue(self) -> bool:
        return any(item.kind == "blue" for item in self.evidence)

    def feasible_actions(self) -> tuple[str, ...]:
        actions = ["COMMIT"]
        if (
            self.sign_side is not None
            and not self.used_sign
            and self.remaining_budget >= self.costs.sign
        ):
            actions.append("PROBE_SIGN")
        if not self.used_blue and self.remaining_budget >= self.costs.blue:
            actions.append("PROBE_BLUE")
        return tuple(actions)

    def candidate_descriptions(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"rule_id": rule, "description": RULE_BY_ID[rule].description}
            for rule in self.candidate_rules
        )

    def to_public_payload(self, *, include_posterior: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "context_id": self.context_id,
            "layout_id": self.layout_id,
            "public_map": self.public_map.to_payload(),
            "agent_position": list(self.agent_position),
            "candidate_rules": list(self.candidate_descriptions()),
            "prior": dict(zip(self.candidate_rules, self.prior)),
            "evidence": [item.as_payload() for item in self.evidence],
            "sign_side": self.sign_side,
            "inspect_error": self.inspect_error,
            "tool_costs": self.costs.as_payload(),
            "tool_budget": self.tool_budget,
            "spent_units": self.spent_units,
            "remaining_budget": self.remaining_budget,
            "max_probes": self.max_probes,
            "horizon": self.max_probes + 1,
            "feasible_actions": list(self.feasible_actions()),
            "commit_steps_by_rule": dict(self.commit_steps_by_rule),
            "step_cost": self.step_cost,
            "success_reward": self.success_reward,
            "hazard_penalty": self.hazard_penalty,
        }
        if include_posterior:
            payload["posterior"] = dict(zip(self.candidate_rules, self.posterior))
        return payload

    def assert_public(self, *, include_posterior: bool = True) -> None:
        payload = self.to_public_payload(include_posterior=include_posterior)
        forbidden = find_forbidden_fields(payload)
        extra = {
            "latent_rule",
            "reference",
            "reference_label",
            "q_commit",
            "q_sign",
            "q_blue",
            "best_action",
            "optimal_actions",
            "declaration_truth",
            "label",
            "stratum",
        }
        leaked = sorted(extra.intersection(payload))
        if forbidden or leaked:
            raise ValueError(
                "public context leaked privileged fields: "
                + repr(forbidden)
                + " "
                + repr(leaked)
            )

    def public_key(self) -> tuple[Any, ...]:
        evidence_key = tuple(
            tuple(sorted(item.as_payload().items())) for item in self.evidence
        )
        return (
            self.layout_id,
            self.agent_position,
            evidence_key,
            self.costs.sign,
            self.costs.blue,
            self.tool_budget,
            self.inspect_error,
            self.prior,
        )


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("cannot normalise zero mass")
    return tuple(float(value) / total for value in weights)


def _entropy(posterior: Sequence[float]) -> float:
    return -sum(
        float(probability) * math.log2(float(probability))
        for probability in posterior
        if probability > 0.0
    )


def _commit_target(context: DecisionContext, posterior: Sequence[float]) -> str:
    index = max(
        range(len(context.candidate_rules)),
        key=lambda i: (posterior[i], -i),
    )
    return context.candidate_rules[index]


def _commit_value(context: DecisionContext, posterior: Sequence[float]) -> float:
    commit_rule = _commit_target(context, posterior)
    goal_color = RULE_BY_ID[commit_rule].goal_color
    steps = context.commit_steps_by_rule[commit_rule]
    expected = 0.0
    for index, rule_id in enumerate(context.candidate_rules):
        role = cell_role(RULE_BY_ID[rule_id], goal_color)
        if role == "goal":
            payoff = context.success_reward
        elif role == "hazard":
            payoff = context.hazard_penalty
        else:
            payoff = 0.0
        expected += posterior[index] * payoff
    return expected - context.step_cost * steps


def _sign_observations(
    context: DecisionContext, posterior: Sequence[float]
) -> list[tuple[str, float]]:
    if context.sign_side is None:
        return []
    masses: dict[str, float] = {}
    for probability, rule_id in zip(posterior, context.candidate_rules):
        if probability <= 0.0:
            continue
        clause = sign_clause(RULE_BY_ID[rule_id], context.sign_side)
        masses[clause] = masses.get(clause, 0.0) + probability
    return list(masses.items())


def _blue_observations(
    context: DecisionContext, posterior: Sequence[float]
) -> list[tuple[str, float]]:
    masses: dict[str, float] = {}
    error = context.inspect_error
    for probability, rule_id in zip(posterior, context.candidate_rules):
        if probability <= 0.0:
            continue
        observed = cell_role(RULE_BY_ID[rule_id], "blue")
        for role in ROLES:
            likelihood = 1.0 - error if role == observed else error / 2.0
            if likelihood <= 0.0:
                continue
            masses[role] = masses.get(role, 0.0) + probability * likelihood
    return list(masses.items())


def _condition(
    posterior: Sequence[float], likelihoods: Sequence[float]
) -> tuple[float, ...]:
    return _normalise(
        [
            max(0.0, float(probability)) * max(0.0, float(likelihood))
            for probability, likelihood in zip(posterior, likelihoods)
        ]
    )


def _sign_posterior(
    context: DecisionContext, posterior: Sequence[float], clause: str
) -> tuple[float, ...]:
    likelihoods = [
        1.0
        if context.sign_side is not None
        and sign_clause(RULE_BY_ID[rule_id], context.sign_side) == clause
        else 0.0
        for rule_id in context.candidate_rules
    ]
    return _condition(posterior, likelihoods)


def _blue_posterior(
    context: DecisionContext, posterior: Sequence[float], observed_role: str
) -> tuple[float, ...]:
    error = context.inspect_error
    likelihoods = []
    for rule_id in context.candidate_rules:
        truth = cell_role(RULE_BY_ID[rule_id], "blue")
        likelihoods.append(1.0 - error if truth == observed_role else error / 2.0)
    return _condition(posterior, likelihoods)


@dataclass(frozen=True)
class ReferenceLabel:
    context_id: str
    reference_version: str
    q_commit: float
    q_sign: float | None
    q_blue: float | None
    best_value: float
    optimal_actions: tuple[str, ...]
    best_action: str
    decision_class: str
    decision_margin: float
    commit_rule: str
    commit_target: Coord
    commit_steps: int
    latent_rule: str
    declaration: str
    declaration_rule: str | None
    declaration_confidence: float
    declaration_margin: float
    declaration_correct: bool | None
    posterior_entropy: float

    def as_payload(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "reference_version": self.reference_version,
            "q_commit": self.q_commit,
            "q_sign": self.q_sign,
            "q_blue": self.q_blue,
            "best_value": self.best_value,
            "optimal_actions": list(self.optimal_actions),
            "best_action": self.best_action,
            "decision_class": self.decision_class,
            "decision_margin": self.decision_margin,
            "commit_rule": self.commit_rule,
            "commit_target": list(self.commit_target),
            "commit_steps": self.commit_steps,
            "latent_rule": self.latent_rule,
            "declaration": self.declaration,
            "declaration_rule": self.declaration_rule,
            "declaration_confidence": self.declaration_confidence,
            "declaration_margin": self.declaration_margin,
            "declaration_correct": self.declaration_correct,
            "posterior_entropy": self.posterior_entropy,
        }


def _latent_rule(context: DecisionContext) -> str:
    seed = int(hashlib.sha256(context.context_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return rng.choices(
        list(context.candidate_rules), weights=list(context.posterior), k=1
    )[0]


def _declaration_label(
    context: DecisionContext,
    *,
    threshold: float = DEFAULT_DECLARATION_THRESHOLD,
    margin_threshold: float = DEFAULT_DECLARATION_MARGIN,
) -> tuple[str, str | None, float, float, bool | None]:
    order = sorted(
        range(len(context.candidate_rules)),
        key=lambda i: (context.posterior[i], -i),
        reverse=True,
    )
    top = order[0]
    runner_up = order[1] if len(order) > 1 else top
    top_prob = float(context.posterior[top])
    margin = top_prob - float(context.posterior[runner_up])
    if top_prob + 1e-12 >= threshold and margin + 1e-12 >= margin_threshold:
        rule = context.candidate_rules[top]
        truth = _latent_rule(context)
        return DECLARE_RULE, rule, top_prob, margin, rule == truth
    return ABSTAIN, None, top_prob, margin, None


def reference_for(context: DecisionContext) -> ReferenceLabel:
    context.assert_public()
    p0 = tuple(float(value) for value in context.posterior)
    sign_cost = context.costs.sign
    blue_cost = context.costs.blue
    budget0 = context.remaining_budget
    sign_used0 = context.used_sign
    blue_used0 = context.used_blue
    max_probes = max(0, int(context.max_probes))

    @lru_cache(maxsize=None)
    def value(
        posterior: tuple[float, ...],
        budget: int,
        sign_used: bool,
        blue_used: bool,
        probes_left: int,
    ) -> float:
        best = _commit_value(context, posterior)
        if probes_left <= 0:
            return best
        if context.sign_side is not None and not sign_used and budget >= sign_cost:
            probe = -float(sign_cost)
            for clause, mass in _sign_observations(context, posterior):
                probe += mass * value(
                    _sign_posterior(context, posterior, clause),
                    budget - sign_cost,
                    True,
                    blue_used,
                    probes_left - 1,
                )
            best = max(best, probe)
        if not blue_used and budget >= blue_cost:
            probe = -float(blue_cost)
            for role, mass in _blue_observations(context, posterior):
                probe += mass * value(
                    _blue_posterior(context, posterior, role),
                    budget - blue_cost,
                    sign_used,
                    True,
                    probes_left - 1,
                )
            best = max(best, probe)
        return best

    def probe_value(action: str) -> float | None:
        if action == "PROBE_SIGN":
            if context.sign_side is None or sign_used0 or budget0 < sign_cost:
                return None
            total = -float(sign_cost)
            for clause, mass in _sign_observations(context, p0):
                total += mass * value(
                    _sign_posterior(context, p0, clause),
                    budget0 - sign_cost,
                    True,
                    blue_used0,
                    max_probes - 1,
                )
            return total
        if action == "PROBE_BLUE":
            if blue_used0 or budget0 < blue_cost:
                return None
            total = -float(blue_cost)
            for role, mass in _blue_observations(context, p0):
                total += mass * value(
                    _blue_posterior(context, p0, role),
                    budget0 - blue_cost,
                    sign_used0,
                    True,
                    max_probes - 1,
                )
            return total
        raise ValueError(f"unknown probe action: {action}")

    q_commit = _commit_value(context, p0)
    q_sign = probe_value("PROBE_SIGN")
    q_blue = probe_value("PROBE_BLUE")
    feasible_values = {"COMMIT": q_commit}
    if q_sign is not None:
        feasible_values["PROBE_SIGN"] = q_sign
    if q_blue is not None:
        feasible_values["PROBE_BLUE"] = q_blue
    best_value = max(feasible_values.values())
    optimal = tuple(
        action
        for action in TIE_ORDER
        if action in feasible_values
        and feasible_values[action] >= best_value - TIE_TOLERANCE
    )
    best_action = optimal[0]
    if len(optimal) > 1:
        decision_class = "tie"
    elif len(feasible_values) == 1:
        decision_class = "commit_preferred"
    else:
        others = [
            value
            for action, value in feasible_values.items()
            if action != best_action
        ]
        second = max(others)
        if best_value - second <= NEAR_TIE_MARGIN:
            decision_class = "near_tie"
        elif best_action == "COMMIT":
            decision_class = "commit_preferred"
        else:
            decision_class = "probe_preferred"
    commit_rule = _commit_target(context, p0)
    goal_color = RULE_BY_ID[commit_rule].goal_color
    declaration, declaration_rule, confidence, margin, declaration_correct = (
        _declaration_label(context)
    )
    return ReferenceLabel(
        context_id=context.context_id,
        reference_version=REFERENCE_VERSION,
        q_commit=q_commit,
        q_sign=q_sign,
        q_blue=q_blue,
        best_value=best_value,
        optimal_actions=optimal,
        best_action=best_action,
        decision_class=decision_class,
        decision_margin=best_value
        - max(
            [
                value
                for action, value in feasible_values.items()
                if action != best_action
            ]
            or [best_value]
        ),
        commit_rule=commit_rule,
        commit_target=context.public_map.markers[goal_color],
        commit_steps=context.commit_steps_by_rule[commit_rule],
        latent_rule=_latent_rule(context),
        declaration=declaration,
        declaration_rule=declaration_rule,
        declaration_confidence=confidence,
        declaration_margin=margin,
        declaration_correct=declaration_correct,
        posterior_entropy=_entropy(p0),
    )


@dataclass(frozen=True)
class _Layout:
    layout_id: str
    public_map: PublicGridMap
    start: Coord
    sign_side: int


def _layouts() -> tuple[_Layout, ...]:
    return (
        _Layout(
            "layout-a",
            PublicGridMap(
                width=7,
                height=7,
                walls=frozenset({(0, 0), (6, 0), (0, 6), (6, 6)}),
                start=(3, 4),
                markers={"red": (3, 2), "blue": (3, 5), "green": (5, 5)},
                key=None,
                door=None,
                signpost=(4, 4),
            ),
            (3, 4),
            0,
        ),
        _Layout(
            "layout-b",
            PublicGridMap(
                width=7,
                height=7,
                walls=frozenset({(1, 1), (5, 3), (3, 5), (5, 5)}),
                start=(2, 2),
                markers={"red": (2, 4), "blue": (3, 2), "green": (1, 2)},
                key=None,
                door=None,
                signpost=(2, 1),
            ),
            (2, 2),
            0,
        ),
        _Layout(
            "layout-c",
            PublicGridMap(
                width=6,
                height=6,
                walls=frozenset({(0, 0), (5, 5), (2, 2)}),
                start=(4, 2),
                markers={"red": (1, 4), "blue": (2, 1), "green": (4, 4)},
                key=None,
                door=None,
                signpost=(5, 2),
            ),
            (4, 2),
            0,
        ),
    )


def _commit_steps(layout: _Layout) -> dict[str, int]:
    steps: dict[str, int] = {}
    for rule_id in GRID_RULE_IDS:
        rule = RULE_BY_ID[rule_id]
        plan = plan_to_goal(rule, layout.public_map, layout.start)
        if not plan and layout.public_map.markers[rule.goal_color] != layout.start:
            raise ValueError(f"no plan for {rule_id} in {layout.layout_id}")
        steps[rule_id] = len(plan)
    return steps


@dataclass(frozen=True)
class _PriorTemplate:
    name: str
    weights: Mapping[str, float]


def _priors() -> tuple[_PriorTemplate, ...]:
    def prior(name: str, **weights: float) -> _PriorTemplate:
        return _PriorTemplate(name, weights)

    return (
        prior("base-hazard-50", base=0.5, hazard_goal_swap=0.5),
        prior("base-remap-50", base=0.5, remap_swap=0.5),
        prior("hazard-sign-50", hazard_goal_swap=0.5, sign_composition=0.5),
        prior(
            "base-hazard-sign-uniform",
            base=1 / 3,
            hazard_goal_swap=1 / 3,
            sign_composition=1 / 3,
        ),
        prior("all-uniform", **{rule: 1 / 6 for rule in GRID_RULE_IDS}),
        prior(
            "base-confident",
            base=0.8,
            hazard_goal_swap=0.1,
            sign_composition=0.1,
        ),
        prior(
            "hazard-confident",
            hazard_goal_swap=0.8,
            base=0.1,
            sign_composition=0.1,
        ),
        prior(
            "sign-confident",
            sign_composition=0.8,
            base=0.1,
            hazard_goal_swap=0.1,
        ),
        prior("base-rotate-50", base=0.5, remap_rotate=0.5),
        prior("swap-rotate-50", remap_swap=0.5, remap_rotate=0.5),
        prior("base-hazard-90", base=0.9, hazard_goal_swap=0.1),
    )


@dataclass(frozen=True)
class _CostRegime:
    name: str
    costs: ToolCosts
    budget: int


def _cost_regimes() -> tuple[_CostRegime, ...]:
    return (
        _CostRegime("default", ToolCosts(sign=1, blue=2), 3),
        _CostRegime("cheap", ToolCosts(sign=1, blue=1), 2),
        _CostRegime("expensive", ToolCosts(sign=2, blue=3), 4),
        _CostRegime("exhausted", ToolCosts(sign=1, blue=2), 0),
    )


def _evidence_options() -> tuple[tuple[Evidence, ...], ...]:
    clauses = sorted({sign_clause(RULE_BY_ID[rule], 0) for rule in GRID_RULE_IDS})
    roles = sorted({cell_role(RULE_BY_ID[rule], "blue") for rule in GRID_RULE_IDS})
    options: list[tuple[Evidence, ...]] = [()]
    options.extend((Evidence("sign", clause, side=0),) for clause in clauses)
    options.extend((Evidence("blue", role),) for role in roles)
    options.extend(
        (Evidence("sign", clause, side=0), Evidence("blue", role))
        for clause in clauses
        for role in roles
    )
    return tuple(options)


def _posterior_for(
    prior: Sequence[float],
    evidence: Sequence[Evidence],
    inspect_error: float,
) -> tuple[float, ...] | None:
    masses = [float(value) for value in prior]
    for item in evidence:
        if item.kind == "sign":
            assert item.side is not None
            likelihoods = [
                1.0
                if sign_clause(RULE_BY_ID[rule_id], item.side) == item.value
                else 0.0
                for rule_id in GRID_RULE_IDS
            ]
        else:
            likelihoods = [
                1.0 - inspect_error
                if cell_role(RULE_BY_ID[rule_id], "blue") == item.value
                else inspect_error / 2.0
                for rule_id in GRID_RULE_IDS
            ]
        masses = [
            mass * likelihood
            for mass, likelihood in zip(masses, likelihoods)
        ]
        total = sum(masses)
        if total <= 0.0:
            return None
        masses = [mass / total for mass in masses]
    return tuple(masses)


def _prior_vector(template: _PriorTemplate) -> tuple[float, ...]:
    return _normalise(
        [float(template.weights.get(rule, 0.0)) for rule in GRID_RULE_IDS]
    )


def _valid_layout(layout: _Layout) -> None:
    if layout.public_map.start != layout.start:
        raise ValueError(f"layout {layout.layout_id} start mismatch")
    if sign_side(layout.start, layout.public_map.signpost) != layout.sign_side:
        raise ValueError(f"layout {layout.layout_id} sign side mismatch")


def build_development_contexts() -> tuple[DecisionContext, ...]:
    contexts: list[DecisionContext] = []
    seen: set[tuple[Any, ...]] = set()
    for layout in _layouts():
        _valid_layout(layout)
        commit_steps = _commit_steps(layout)
        for template in _priors():
            prior = _prior_vector(template)
            for evidence in _evidence_options():
                for regime in _cost_regimes():
                    spent = sum(item.cost(regime.costs) for item in evidence)
                    if spent > regime.budget:
                        continue
                    for noise in (0.0, 0.05, 0.15, 0.30):
                        posterior = _posterior_for(prior, evidence, noise)
                        if posterior is None:
                            continue
                        provisional_id = (
                            f"vlm-dev-{layout.layout_id}-{template.name}-"
                            f"{regime.name}-{noise:.2f}-"
                            + "-".join(f"{item.kind}:{item.value}" for item in evidence)
                        )
                        context = DecisionContext(
                            context_id=provisional_id,
                            layout_id=layout.layout_id,
                            public_map=layout.public_map,
                            agent_position=layout.start,
                            candidate_rules=GRID_RULE_IDS,
                            prior=prior,
                            posterior=posterior,
                            evidence=evidence,
                            sign_side=layout.sign_side,
                            inspect_error=noise,
                            costs=regime.costs,
                            tool_budget=regime.budget,
                            max_probes=2,
                            commit_steps_by_rule=commit_steps,
                            stratum="unassigned",
                        )
                        key = context.public_key()
                        if key in seen:
                            continue
                        seen.add(key)
                        context.assert_public()
                        contexts.append(context)
    contexts.sort(key=lambda item: item.context_id)
    return tuple(contexts)


@dataclass(frozen=True)
class DecisionLibrary:
    contexts: tuple[DecisionContext, ...]
    labels: tuple[ReferenceLabel, ...]
    manifest: Mapping[str, Any]

    def label_map(self) -> dict[str, ReferenceLabel]:
        return {label.context_id: label for label in self.labels}

    def context_map(self) -> dict[str, DecisionContext]:
        return {context.context_id: context for context in self.contexts}

    def public_payloads(self) -> tuple[dict[str, Any], ...]:
        return tuple(context.to_public_payload() for context in self.contexts)


def _selection_key(context: DecisionContext) -> str:
    return hashlib.sha256(context.context_id.encode("utf-8")).hexdigest()


def _select_contexts(
    solved: Sequence[tuple[DecisionContext, ReferenceLabel]],
) -> tuple[DecisionContext, ...]:
    selected: list[DecisionContext] = []
    selected_ids: set[str] = set()

    def pick(predicate: Any, count: int, stratum: str) -> None:
        pool = [
            pair
            for pair in solved
            if pair[0].context_id not in selected_ids and predicate(pair)
        ]
        pool.sort(key=lambda pair: _selection_key(pair[0]))
        for context, _label in pool[:count]:
            selected.append(replace(context, stratum=stratum))
            selected_ids.add(context.context_id)

    def same_goal_ambiguous(pair: tuple[DecisionContext, ReferenceLabel]) -> bool:
        context = pair[0]
        goals = {
            RULE_BY_ID[rule].goal_color
            for rule, probability in zip(context.candidate_rules, context.posterior)
            if probability > 0.0
        }
        return len(goals) == 1 and _entropy(context.posterior) >= 0.5

    pick(lambda pair: pair[0].remaining_budget == 0, 8, "budget_exhausted")
    pick(same_goal_ambiguous, 8, "ambiguous_task_solvable")
    pick(
        lambda pair: pair[1].decision_class == "commit_preferred",
        36,
        "commit_preferred",
    )
    pick(
        lambda pair: pair[1].best_action == "PROBE_SIGN",
        14,
        "probe_sign_preferred",
    )
    pick(
        lambda pair: pair[1].best_action == "PROBE_BLUE",
        14,
        "probe_blue_preferred",
    )
    pick(
        lambda pair: pair[1].decision_class in {"near_tie", "tie"},
        14,
        "near_tie",
    )
    pick(lambda pair: pair[0].inspect_error >= 0.15, 8, "high_noise")
    pick(lambda pair: pair[0].inspect_error == 0.0, 8, "zero_noise")
    pick(
        lambda pair: pair[1].decision_class == "commit_preferred",
        8,
        "commit_preferred",
    )
    selected.sort(key=lambda context: context.context_id)
    return tuple(selected)


def _manifest(
    contexts: Sequence[DecisionContext], labels: Sequence[ReferenceLabel]
) -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_stratum: dict[str, int] = {}
    by_cost: dict[str, int] = {}
    by_noise: dict[str, int] = {}
    by_layout: dict[str, int] = {}
    for context, label in zip(contexts, labels):
        by_class[label.decision_class] = by_class.get(label.decision_class, 0) + 1
        by_stratum[context.stratum] = by_stratum.get(context.stratum, 0) + 1
        cost_key = f"{context.costs.sign}/{context.costs.blue}/{context.tool_budget}"
        by_cost[cost_key] = by_cost.get(cost_key, 0) + 1
        by_noise[str(context.inspect_error)] = (
            by_noise.get(str(context.inspect_error), 0) + 1
        )
        by_layout[context.layout_id] = by_layout.get(context.layout_id, 0) + 1
    return {
        "reference_version": REFERENCE_VERSION,
        "n_contexts": len(contexts),
        "decision_classes": dict(sorted(by_class.items())),
        "strata": dict(sorted(by_stratum.items())),
        "tool_cost_budget": dict(sorted(by_cost.items())),
        "inspect_error": dict(sorted(by_noise.items())),
        "layouts": dict(sorted(by_layout.items())),
        "split": "development",
        "reserved_maps_used": False,
        "vlm_inference": False,
        "actions": list(ACTIONS),
        "declaration_actions": [DECLARE_RULE, ABSTAIN],
        "candidate_rules": list(GRID_RULE_IDS),
    }


@lru_cache(maxsize=1)
def build_development_library() -> DecisionLibrary:
    all_contexts = build_development_contexts()
    solved = [(context, reference_for(context)) for context in all_contexts]
    selected = _select_contexts(solved)
    labels = tuple(reference_for(context) for context in selected)
    return DecisionLibrary(selected, labels, _manifest(selected, labels))


@dataclass(frozen=True)
class DecisionPrediction:
    context_id: str
    action: str
    fallback: bool = False


@dataclass(frozen=True)
class DeclarationPrediction:
    context_id: str
    action: str
    rule: str | None = None
    fallback: bool = False


@dataclass(frozen=True)
class DecisionScore:
    context_id: str
    predicted_action: str
    optimal: bool
    regret: float
    over_probe: bool
    under_probe: bool
    probe_target_correct: bool | None
    invalid: bool
    fallback: bool


def _action_value(label: ReferenceLabel, action: str) -> float | None:
    if action == "COMMIT":
        return label.q_commit
    if action == "PROBE_SIGN":
        return label.q_sign
    if action == "PROBE_BLUE":
        return label.q_blue
    return None


def score_decisions(
    library: DecisionLibrary,
    predictions: Sequence[DecisionPrediction],
    *,
    allow_missing: bool = False,
) -> tuple[DecisionScore, ...]:
    contexts = library.context_map()
    labels = library.label_map()
    by_id = {prediction.context_id: prediction for prediction in predictions}
    scores: list[DecisionScore] = []
    for context_id, context in contexts.items():
        prediction = by_id.get(context_id)
        if prediction is None and not allow_missing:
            raise ValueError(f"missing prediction for {context_id}")
        label = labels[context_id]
        raw_action = "COMMIT" if prediction is None else prediction.action
        fallback = True if prediction is None else bool(prediction.fallback)
        invalid = raw_action not in ACTIONS or raw_action not in context.feasible_actions()
        action = "COMMIT" if invalid else raw_action
        used_value = _action_value(label, action)
        if used_value is None:
            used_value = label.q_commit
        scores.append(
            DecisionScore(
                context_id=context_id,
                predicted_action=action,
                optimal=action in label.optimal_actions and not invalid,
                regret=max(0.0, label.best_value - used_value),
                over_probe=action.startswith("PROBE") and label.best_action == "COMMIT",
                under_probe=action == "COMMIT" and label.best_action.startswith("PROBE"),
                probe_target_correct=(
                    action in label.optimal_actions if action.startswith("PROBE") else None
                ),
                invalid=invalid,
                fallback=fallback,
            )
        )
    return tuple(scores)


def summarize_scores(scores: Sequence[DecisionScore]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {"n": 0}
    regrets = sorted(score.regret for score in scores)
    probe_scores = [score for score in scores if score.probe_target_correct is not None]
    return {
        "n": total,
        "optimal_action_accuracy": sum(score.optimal for score in scores) / total,
        "mean_regret": sum(regrets) / total,
        "max_regret": max(regrets),
        "median_regret": regrets[total // 2],
        "over_probe_rate": sum(score.over_probe for score in scores) / total,
        "under_probe_rate": sum(score.under_probe for score in scores) / total,
        "tool_selection_accuracy": (
            sum(bool(score.probe_target_correct) for score in probe_scores)
            / len(probe_scores)
            if probe_scores
            else None
        ),
        "invalid_rate": sum(score.invalid for score in scores) / total,
        "fallback_rate": sum(score.fallback for score in scores) / total,
        "action_counts": {
            action: sum(score.predicted_action == action for score in scores)
            for action in ACTIONS
        },
    }


def _declaration_confidence(
    context: DecisionContext, declaration: DeclarationPrediction
) -> float:
    if declaration.rule not in context.candidate_rules:
        return 0.0
    return float(context.posterior[context.candidate_rules.index(declaration.rule)])


def score_declarations(
    library: DecisionLibrary,
    predictions: Sequence[DeclarationPrediction],
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    contexts = library.context_map()
    labels = library.label_map()
    by_id = {prediction.context_id: prediction for prediction in predictions}
    rows: list[dict[str, Any]] = []
    for context_id, context in contexts.items():
        prediction = by_id.get(context_id)
        if prediction is None and not allow_missing:
            raise ValueError(f"missing declaration for {context_id}")
        action = ABSTAIN if prediction is None else prediction.action
        rule = None if prediction is None else prediction.rule
        invalid = action not in (DECLARE_RULE, ABSTAIN)
        if action == DECLARE_RULE:
            if rule not in context.candidate_rules:
                invalid = True
                correct: bool | None = False
            else:
                correct = rule == labels[context_id].latent_rule
        else:
            correct = None
        confidence = (
            _declaration_confidence(context, prediction)
            if prediction is not None
            else 0.0
        )
        rows.append(
            {
                "context_id": context_id,
                "action": action,
                "rule": rule,
                "invalid": invalid,
                "correct": correct,
                "confidence": confidence,
                "declared": action == DECLARE_RULE and not invalid,
            }
        )
    declared = [row for row in rows if row["declared"]]
    correct_rows = [row for row in declared if row["correct"]]
    pairs = [
        (float(row["confidence"]), 1.0 if row["correct"] else 0.0)
        for row in declared
    ]
    ece = 0.0
    if pairs:
        for index in range(10):
            lower, upper = index / 10, (index + 1) / 10
            bucket = [
                (confidence, value)
                for confidence, value in pairs
                if lower <= confidence < upper
                or (index == 9 and confidence >= lower)
            ]
            if bucket:
                mean_confidence = sum(pair[0] for pair in bucket) / len(bucket)
                mean_correct = sum(pair[1] for pair in bucket) / len(bucket)
                ece += len(bucket) / len(pairs) * abs(mean_confidence - mean_correct)
    return {
        "n": len(rows),
        "coverage": len(declared) / len(rows) if rows else 0.0,
        "accuracy_given_declared": (
            len(correct_rows) / len(declared) if declared else None
        ),
        "abstention_rate": (
            sum(row["action"] == ABSTAIN for row in rows) / len(rows) if rows else 0.0
        ),
        "invalid_rate": (
            sum(row["invalid"] for row in rows) / len(rows) if rows else 0.0
        ),
        "ece_10_bin": ece if pairs else None,
        "declared": len(declared),
        "correct": len(correct_rows),
        "rows": rows,
    }


def declaration_prediction(
    library: DecisionLibrary, context_id: str
) -> DeclarationPrediction:
    label = library.label_map()[context_id]
    return DeclarationPrediction(
        context_id=context_id,
        action=label.declaration,
        rule=label.declaration_rule,
    )


def information_gain_bits(context: DecisionContext, action: str) -> float:
    posterior = context.posterior
    before = _entropy(posterior)
    if action == "COMMIT":
        return 0.0
    if action == "PROBE_SIGN":
        if context.sign_side is None or context.used_sign:
            return 0.0
        after = sum(
            mass * _entropy(_sign_posterior(context, posterior, clause))
            for clause, mass in _sign_observations(context, posterior)
        )
        return max(0.0, before - after)
    if action == "PROBE_BLUE":
        if context.used_blue:
            return 0.0
        after = sum(
            mass * _entropy(_blue_posterior(context, posterior, role))
            for role, mass in _blue_observations(context, posterior)
        )
        return max(0.0, before - after)
    raise ValueError(f"unknown action: {action}")


def baseline_predictions(
    library: DecisionLibrary, name: str, *, seed: int = 20261126
) -> tuple[DecisionPrediction, ...]:
    predictions: list[DecisionPrediction] = []
    for context in library.contexts:
        feasible = context.feasible_actions()
        if name == "always_commit":
            action = "COMMIT"
        elif name == "always_probe":
            action = (
                "PROBE_SIGN"
                if "PROBE_SIGN" in feasible
                else "PROBE_BLUE"
                if "PROBE_BLUE" in feasible
                else "COMMIT"
            )
        elif name == "random_valid":
            rng = random.Random(
                seed
                + int(hashlib.sha256(context.context_id.encode()).hexdigest()[:8], 16)
            )
            action = rng.choice(feasible)
        elif name == "information_gain":
            action = "COMMIT"
            best_gain = 0.0
            for candidate in feasible:
                if candidate == "COMMIT":
                    continue
                gain = information_gain_bits(context, candidate) / context.costs.cost(
                    candidate
                )
                if gain > best_gain + TIE_TOLERANCE:
                    best_gain = gain
                    action = candidate
        elif name == "exact":
            action = library.label_map()[context.context_id].best_action
        else:
            raise ValueError(f"unknown baseline: {name}")
        predictions.append(DecisionPrediction(context.context_id, action))
    return tuple(predictions)


def audit_library(library: DecisionLibrary) -> dict[str, Any]:
    baseline_names = (
        "always_commit",
        "always_probe",
        "random_valid",
        "information_gain",
        "exact",
    )
    labels = library.labels
    summary: dict[str, Any] = {
        "manifest": dict(library.manifest),
        "best_action_counts": {
            action: sum(label.best_action == action for label in labels)
            for action in ACTIONS
        },
        "baselines": {},
        "labels": {
            "mean_entropy": sum(label.posterior_entropy for label in labels)
            / len(labels),
            "mean_best_value": sum(label.best_value for label in labels)
            / len(labels),
            "declaration_coverage_opportunity": sum(
                label.declaration == DECLARE_RULE for label in labels
            )
            / len(labels),
        },
    }
    for name in baseline_names:
        summary["baselines"][name] = summarize_scores(
            score_decisions(library, baseline_predictions(library, name))
        )
    oracle_declarations = score_declarations(
        library,
        [
            declaration_prediction(library, context.context_id)
            for context in library.contexts
        ],
    )
    oracle_declarations.pop("rows", None)
    summary["declarations"] = {"reference_gate": oracle_declarations}
    return summary
