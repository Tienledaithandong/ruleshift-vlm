"""Surgical offline diagnostics: one component replaced at a time."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.diagnostics import (
    INTERVENTIONS,
    run_diagnostic,
    write_diagnostic_report,
)
from ruleshift.evaluator.offline_audit import build_offline_contexts
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN

ALL_CONTEXTS = build_offline_contexts()
CONTEXTS = ALL_CONTEXTS[:8]
_ALL_RULES = {
    rule.rule_id: rule
    for context in ALL_CONTEXTS
    for rule in context.public_observation.candidate_rules
}


@pytest.mark.parametrize("intervention", INTERVENTIONS)
def test_each_intervention_runs_and_chooses_valid_actions(intervention: str) -> None:
    result = run_diagnostic(
        CONTEXTS,
        StubVLMBackend(),
        ResponseCache(None),
        intervention,
    )
    assert result.intervention == intervention
    assert result.mean_regret >= 0.0
    assert 0.0 <= result.error_rate <= 1.0
    for decision in result.decisions:
        assert decision["chosen_action"] in MACRO_ACTIONS_FROZEN
        assert decision["decision_regret"] >= 0.0


def test_diagnostic_is_deterministic() -> None:
    first = run_diagnostic(
        CONTEXTS, StubVLMBackend(), ResponseCache(None), "oracle_belief"
    )
    second = run_diagnostic(
        CONTEXTS, StubVLMBackend(), ResponseCache(None), "oracle_belief"
    )
    assert first == second


def test_report_states_bottleneck_without_generality_claims(tmp_path) -> None:
    results = {
        intervention: run_diagnostic(
            CONTEXTS, StubVLMBackend(), ResponseCache(None), intervention
        )
        for intervention in INTERVENTIONS
    }
    path = write_diagnostic_report(results, tmp_path / "w10.md")
    text = path.read_text(encoding="utf-8")
    assert "no vlm" in text.lower()
    assert "bottleneck" in text.lower()
    assert "no generality" in text.lower()
    for intervention in INTERVENTIONS:
        assert intervention in text


# ---------------------------------------------------------------------------
# Intervention wiring: each row must actually change the selector's inputs.
# ---------------------------------------------------------------------------
from ruleshift.agents.vlm_backend import ModelResponse  # noqa: E402


class _FixedBackend:
    """Backend that always returns one hand-built, schema-valid payload."""

    model_id = "fixed-test-backend"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate(self, request) -> ModelResponse:  # noqa: ANN001 - test double
        import json

        text = json.dumps(self.payload, sort_keys=True)
        return ModelResponse(
            text=text,
            input_tokens=1,
            output_tokens=1,
            latency_ms=0.0,
            model_id=self.model_id,
            backend_id="FIXED_TEST",
        )


def _tie_breaking_payload() -> dict:
    """A confident blue model prior switches after the lottery excludes blue."""

    def under_rules(accepted_rule: str) -> list[dict]:
        return [
            {
                "rule_id": rule,
                "possible_outcomes": [
                    {
                        "observation": "accept" if rule == accepted_rule else "reject",
                        "p": 1.0,
                    }
                ],
            }
            for rule in ("red", "blue", "green")
        ]

    return {
        "candidate_probabilities": {"red": 0.0, "blue": 1.0, "green": 0.0},
        "action_predictions": [
            {
                "action_id": "COMMIT_BLUE",
                "under_rules": under_rules("blue"),
                "estimated_future_best_action": "COMMIT_BLUE",
            },
            {
                "action_id": "PROBE_RED",
                "under_rules": under_rules("red"),
                "estimated_future_best_action": "COMMIT_RED",
            },
        ],
        "selected_action": "COMMIT_BLUE",
    }


def test_oracle_belief_intervention_changes_the_decision() -> None:
    # This test isolates the belief intervention: the test-only public menu
    # retains the model's confident blue rule so that a wrong-color commit is
    # available. Production contexts expose only the two rules in the lottery.
    contexts = tuple(
        replace(
            c,
            public_observation=replace(
                c.public_observation,
                candidate_rules=tuple(
                    _ALL_RULES[rule] for rule in ("red", "blue", "green")
                ),
            ),
        )
        for c in ALL_CONTEXTS
        if ("blue", False) not in c.belief
        and c.public_observation.public_costs["probe_detour"] == 0
    )
    assert contexts
    backend = _FixedBackend(_tie_breaking_payload())
    baseline = run_diagnostic(contexts, backend, ResponseCache(None), "baseline")
    oracle = run_diagnostic(contexts, backend, ResponseCache(None), "oracle_belief")
    assert {d["chosen_action"] for d in baseline.decisions} == {"COMMIT_BLUE"}
    assert {d["chosen_action"] for d in oracle.decisions} == {"PROBE_RED"}


def test_oracle_outcomes_intervention_rewrites_predictions() -> None:
    from ruleshift.evaluator.diagnostics import _true_outcomes

    payload = _tie_breaking_payload()
    rewritten = _true_outcomes(payload)
    for prediction in rewritten["action_predictions"]:
        color = prediction["action_id"].split("_", 1)[1].lower()
        for rule_prediction in prediction["under_rules"]:
            assert rule_prediction["possible_outcomes"] == [
                {
                    "observation": (
                        "accept" if rule_prediction["rule_id"] == color else "reject"
                    ),
                    "p": 1.0,
                }
            ]
    # The original payload must not be mutated in place.
    assert payload["action_predictions"][0]["under_rules"][0][
        "possible_outcomes"
    ] == [{"observation": "reject", "p": 1.0}]


def test_diagnostic_regret_uses_covered_labels_only() -> None:
    result = run_diagnostic(CONTEXTS, StubVLMBackend(), ResponseCache(None), "baseline")
    assert result.label_coverage_rate == pytest.approx(1.0)
    assert all(d["label_covered"] for d in result.decisions)
