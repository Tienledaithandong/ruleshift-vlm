"""Surgical offline diagnostics: one component replaced at a time."""

from __future__ import annotations

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

CONTEXTS = build_offline_contexts()[:8]


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
