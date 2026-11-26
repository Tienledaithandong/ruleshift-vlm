"""Deterministic CPU tests for the VLM decision microbenchmark."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from ruleshift.evaluator.vlm_decision import (
    ACTIONS,
    ABSTAIN,
    DECLARE_RULE,
    DecisionPrediction,
    DeclarationPrediction,
    _commit_steps,
    _layouts,
    build_development_contexts,
    build_development_library,
    declaration_prediction,
    reference_for,
    score_declarations,
    score_decisions,
    summarize_scores,
)
from ruleshift.evaluator.vlm_decision_visual import (
    blue_role_perturbation,
    render_decision_card,
    visual_digest,
)
from ruleshift.public_api.grid_planner import plan_to_goal
from ruleshift.public_api.grid_rules import RULE_BY_ID


@pytest.fixture(scope="module")
def contexts():
    return build_development_contexts()


@pytest.fixture(scope="module")
def library():
    return build_development_library()


def _find(contexts, *, prior: str, regime: str, noise: float = 0.05, evidence: str = "none"):
    matches = []
    for context in contexts:
        if prior not in context.context_id or regime not in context.context_id:
            continue
        if context.inspect_error != noise:
            continue
        if evidence == "none" and context.evidence:
            continue
        if evidence != "none" and not any(item.kind == evidence for item in context.evidence):
            continue
        matches.append(context)
    assert matches, (prior, regime, noise, evidence)
    return matches[0]


def test_commit_preferred_when_all_candidates_share_goal(contexts):
    context = _find(contexts, prior="base-remap-50", regime="default")
    label = reference_for(context)
    assert label.best_action == "COMMIT"
    assert label.q_commit > label.q_sign
    assert label.q_commit > label.q_blue
    assert label.optimal_actions == ("COMMIT",)


def test_sign_probe_preferred_when_it_separates_goal_from_hazard(contexts):
    context = _find(contexts, prior="base-hazard-50", regime="default")
    label = reference_for(context)
    assert label.best_action == "PROBE_SIGN"
    assert label.q_sign > label.q_commit
    assert label.q_sign > label.q_blue
    assert label.optimal_actions == ("PROBE_SIGN",)


def test_blue_probe_preferred_when_sign_is_uninformative(contexts):
    context = _find(contexts, prior="hazard-sign-50", regime="default")
    label = reference_for(context)
    assert label.best_action == "PROBE_BLUE"
    assert label.q_blue > label.q_commit
    assert label.q_blue > label.q_sign
    assert label.optimal_actions == ("PROBE_BLUE",)


def test_budget_exhaustion_forces_commit(contexts):
    context = _find(contexts, prior="base-hazard-50", regime="exhausted")
    label = reference_for(context)
    assert context.feasible_actions() == ("COMMIT",)
    assert label.q_sign is None
    assert label.q_blue is None
    assert label.best_action == "COMMIT"
    assert label.decision_class == "commit_preferred"


def test_expensive_cost_can_remove_blue_after_sign(contexts):
    context = _find(contexts, prior="base-hazard-50", regime="expensive", evidence="sign")
    assert "PROBE_BLUE" not in context.feasible_actions()
    label = reference_for(context)
    assert label.q_blue is None
    assert label.q_sign is None


def test_posteriors_are_normalised_and_consistent(contexts):
    for context in contexts:
        assert abs(sum(context.prior) - 1.0) < 1e-9
        assert abs(sum(context.posterior) - 1.0) < 1e-9
        assert all(value >= 0.0 and math.isfinite(value) for value in context.posterior)
        assert context.assert_public() is None


def test_contexts_carry_no_privileged_or_label_fields(contexts):
    forbidden_keys = {
        "latent_rule",
        "reference",
        "q_commit",
        "q_sign",
        "q_blue",
        "best_action",
        "optimal_actions",
        "label",
        "stratum",
    }
    for context in contexts:
        payload = context.to_public_payload()
        assert not forbidden_keys.intersection(payload)
        serialized = json.dumps(payload, sort_keys=True)
        assert "latent_rule" not in serialized
        assert "q_commit" not in serialized
        assert "best_action" not in serialized


def test_public_posterior_can_be_withheld(contexts):
    context = contexts[0]
    without = context.to_public_payload(include_posterior=False)
    assert "posterior" not in without
    assert "prior" in without
    context.assert_public(include_posterior=False)


def test_commit_steps_match_public_planner(contexts):
    layouts = {layout.layout_id: layout for layout in _layouts()}
    for context in contexts:
        layout = layouts[context.layout_id]
        expected = _commit_steps(layout)
        assert dict(context.commit_steps_by_rule) == expected
        for rule_id, steps in expected.items():
            plan = plan_to_goal(RULE_BY_ID[rule_id], layout.public_map, layout.start)
            assert len(plan) == steps


def test_library_coverage_and_determinism(contexts, library):
    assert len(library.contexts) >= 80
    assert library.manifest["n_contexts"] == len(library.contexts)
    assert {context.layout_id for context in library.contexts} == {
        "layout-a",
        "layout-b",
        "layout-c",
    }
    counts = library.manifest["decision_classes"]
    assert counts.get("commit_preferred", 0) > 0
    assert counts.get("probe_preferred", 0) > 0
    assert counts.get("near_tie", 0) + counts.get("tie", 0) > 0
    assert library.manifest["strata"].get("budget_exhausted", 0) == 8
    assert any(
        context.inspect_error >= 0.15 for context in library.contexts
    )
    assert any(context.inspect_error == 0.0 for context in library.contexts)
    assert any(context.costs.sign == 2 for context in library.contexts)
    assert any(context.costs.sign == 1 and context.costs.blue == 1 for context in library.contexts)
    assert any(context.used_sign for context in library.contexts)
    assert any(context.used_blue for context in library.contexts)
    assessed = [reference_for(context).best_action for context in library.contexts]
    assert set(assessed) == set(ACTIONS)
    assert assessed.count("COMMIT") > assessed.count("PROBE_SIGN") + assessed.count("PROBE_BLUE")
    assert [context.context_id for context in build_development_contexts()] == [
        context.context_id for context in contexts
    ]
    assert [
        reference_for(context).as_payload() for context in library.contexts
    ] == [label.as_payload() for label in library.labels]


def test_invalid_action_falls_back_and_is_counted(library):
    context = next(
        item
        for item in library.contexts
        if reference_for(item).best_action == "COMMIT"
    )
    scores = score_decisions(
        library,
        [
            DecisionPrediction(context.context_id, "NOT_AN_ACTION"),
            *[
                DecisionPrediction(item.context_id, "COMMIT")
                for item in library.contexts
                if item.context_id != context.context_id
            ],
        ],
    )
    target = next(score for score in scores if score.context_id == context.context_id)
    assert target.invalid
    assert target.fallback is False
    assert target.predicted_action == "COMMIT"


def test_over_and_under_probe_are_separated(library):
    commit_context = next(
        item
        for item in library.contexts
        if reference_for(item).best_action == "COMMIT"
    )
    probe_context = next(
        item
        for item in library.contexts
        if reference_for(item).best_action.startswith("PROBE")
    )
    predictions = [
        DecisionPrediction(
            item.context_id,
            "PROBE_SIGN"
            if item.context_id == commit_context.context_id
            else "COMMIT",
        )
        for item in library.contexts
    ]
    scores = score_decisions(library, predictions)
    commit_score = next(
        score for score in scores if score.context_id == commit_context.context_id
    )
    probe_score = next(
        score for score in scores if score.context_id == probe_context.context_id
    )
    assert commit_score.over_probe
    assert probe_score.under_probe
    summary = summarize_scores(scores)
    assert 0.0 <= summary["over_probe_rate"] <= 1.0
    assert 0.0 <= summary["under_probe_rate"] <= 1.0
    assert summary["mean_regret"] >= 0.0


def test_declaration_scoring_and_invalid_rule(library):
    predictions = [
        declaration_prediction(library, context.context_id)
        for context in library.contexts
    ]
    oracle = score_declarations(library, predictions)
    assert 0.0 < oracle["coverage"] <= 1.0
    assert 0.0 < oracle["accuracy_given_declared"] <= 1.0
    assert oracle["ece_10_bin"] is not None
    first = library.contexts[0]
    abstain = [
        DeclarationPrediction(context.context_id, ABSTAIN, None)
        for context in library.contexts
    ]
    aborted = score_declarations(library, abstain)
    assert aborted["coverage"] == 0.0
    assert aborted["accuracy_given_declared"] is None
    invalid = [
        DeclarationPrediction(
            context.context_id,
            DECLARE_RULE,
            "not_a_rule",
        )
        for context in library.contexts
    ]
    invalid_result = score_declarations(library, invalid)
    assert invalid_result["invalid_rate"] == 1.0
    assert invalid_result["coverage"] == 0.0


def test_visual_card_nuisance_control_is_semantically_invariant(library):
    context = library.contexts[0]
    base = render_decision_card(context, variant="base")
    nuisance = render_decision_card(context, variant="nuisance")
    assert base.shape == (320, 512, 3)
    assert base.dtype == np.uint8
    assert np.array_equal(base, render_decision_card(context, variant="base"))
    assert visual_digest(base) != visual_digest(nuisance)
    assert not np.array_equal(base, nuisance)
    assert reference_for(context).as_payload() == reference_for(context).as_payload()


def test_visual_diagnostic_is_a_semantic_paired_context(library):
    context = next(
        item
        for item in library.contexts
        if item.used_blue and item.remaining_budget >= item.costs.blue
    )
    diagnostic = None
    for role in ("goal", "hazard", "safe"):
        try:
            diagnostic = blue_role_perturbation(context, role)
            break
        except ValueError:
            continue
    assert diagnostic is not None
    assert diagnostic.context_id != context.context_id
    assert diagnostic.evidence != context.evidence
    assert diagnostic.posterior != context.posterior
    assert visual_digest(render_decision_card(context)) != visual_digest(
        render_decision_card(diagnostic)
    )
    reference_for(diagnostic)


def test_one_probe_exact_values_match_hand_calculation(contexts):
    context = _find(contexts, prior="base-hazard-50", regime="default")
    label = reference_for(context)
    # Sign is side 0 and separates red goal from red hazard. Both branches end
    # with a commit; the best commit is red (base) or blue (hazard).
    assert label.q_sign == pytest.approx(-1.0 + (9.9 + 9.95) / 2.0)
    assert label.q_commit == pytest.approx(4.0 - 0.05 * 2)
    assert label.q_blue < label.q_sign
