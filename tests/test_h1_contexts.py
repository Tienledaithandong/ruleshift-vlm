from collections import Counter

import pytest

from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.structured_vlm import build_prediction_request
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.evaluator.h1_contexts import build_h1_contexts
from ruleshift.evaluator.offline_audit import run_offline_audit


@pytest.fixture(scope="module")
def h1_contexts():
    return build_h1_contexts(81)


def test_h1_design_is_balanced_and_fully_disclosed(h1_contexts) -> None:
    assert len(h1_contexts) == 81
    assert Counter(c.evaluator_meta["delta_bin"] for c in h1_contexts) == {
        "negative": 27,
        "marginal": 27,
        "positive": 27,
    }
    assert sum(c.delta_probe > 1e-9 for c in h1_contexts) == 27
    assert sum(c.delta_probe < -1e-9 for c in h1_contexts) >= 27
    assert sum(abs(c.delta_probe) <= 1e-9 for c in h1_contexts) >= 20
    assert min(c.delta_probe for c in h1_contexts) <= -30
    assert max(c.delta_probe for c in h1_contexts) >= 5
    assert {int(c.public_observation.public_costs["probe_detour"]) for c in h1_contexts} == {
        0,
        2,
        5,
        8,
    }
    assert sorted({int(c.evaluator_meta["lockout"]) for c in h1_contexts}) == [0, 3, 8]
    assert sorted({c.feedback_noise for c in h1_contexts}) == [0.0, 0.1, 0.25]
    assert sorted({c.prior for c in h1_contexts}) == [0.5, 0.8]
    assert sorted({c.horizon for c in h1_contexts}) == [12, 24]
    assert len({c.public_observation.public_text for c in h1_contexts}) == 81
    assert all(len(c.q_values) == 8 for c in h1_contexts)


def test_h1_prompts_disclose_rates_without_hidden_fields(h1_contexts) -> None:
    backend = StubVLMBackend()
    prompts = [
        build_prediction_request(c.public_observation, backend, direct=True).prompt
        for c in h1_contexts
    ]
    assert len(set(prompts)) == 81
    joined = "\n".join(prompts)
    for field in ("true_rule", "initial_rule", "shift_hazard", "feedback_noise"):
        assert field not in joined
    assert "equal probability" in joined
    assert "0.80" in joined
    assert "always reports correctly" in joined
    assert "0.90" in joined
    assert "0.75" in joined
    assert "detour costs 8" in joined
    assert "locks the gate for 8" in joined


def test_h1_candidate_menu_matches_the_two_rule_lottery(h1_contexts) -> None:
    for context in h1_contexts:
        rule_ids = {rule.rule_id for rule in context.public_observation.candidate_rules}
        assert len(rule_ids) == 2
        assert set(context.belief) == {(rule, False) for rule in rule_ids}
        assert all(
            rule.rule_id in rule_ids for rule in context.public_observation.candidate_rules
        )


def test_h1_m1_path_covers_every_decision(tmp_path, h1_contexts) -> None:
    records, metrics = run_offline_audit(
        h1_contexts[:9],
        backend=StubVLMBackend(),
        cache=ResponseCache(tmp_path / "cache.jsonl"),
        method="m1",
    )
    assert len(records) == 9
    assert all(record["method"] == "m1" for record in records)
    assert all(record["chosen_action"] in record["available_actions"] for record in records)
    assert metrics.label_coverage_rate == pytest.approx(1.0)
    assert metrics.fallback_count == 0