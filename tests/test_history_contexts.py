"""Independent posterior and horizon checks for the noisy-history P1 pilot."""

import pytest
import yaml

from ruleshift.agents.structured_vlm import build_prediction_request
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.envs.config import load_dev_grid
from ruleshift.evaluator.history_contexts import build_history_contexts
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS, validate_public_observation


def test_noisy_probe_posterior_matches_hand_calculation() -> None:
    (context,) = build_history_contexts(1)
    assert context.public_observation.public_history == "PROBE_RED outcome=accept reward=-0.05 steps=1"
    # At the test step P(red)=.9, P(blue)=P(green)=.05. Accept has
    # probability .9 under red and .1 under each alternative.
    assert context.belief == pytest.approx({
        ("red", False): 81 / 82, ("blue", True): 1 / 164, ("green", True): 1 / 164,
    })


def test_public_prompts_are_distinct_and_complete_under_one_hidden_process() -> None:
    contexts = build_history_contexts()
    assert len(contexts) == 81
    backend = StubVLMBackend()
    prompts = set()
    for context in contexts:
        obs = context.public_observation
        assert validate_public_observation(obs.to_payload()) == []
        prompt = build_prediction_request(obs, backend, direct=True).prompt
        assert obs.public_history in prompt
        assert all(word not in prompt for word in (
            "shift_time", "shift_hazard", "feedback_noise", "delta_probe", "q_values",
        ))
        prompts.add(prompt)
        assert set(context.q_values) == set(FAMILY_A_ACTIONS)
    assert len(prompts) == 81
    assert {(c.config.shift_hazard, c.config.feedback_noise) for c in contexts} == {(0.1, 0.1)}
    assert any(c.delta_probe > 0 for c in contexts)
    assert any(c.delta_probe < 0 for c in contexts)
    budgets_by_first_and_detour = {}
    for context in contexts:
        obs = context.public_observation
        group = (obs.public_history.split()[0], obs.public_costs["probe_detour"])
        budgets_by_first_and_detour.setdefault(group, set()).add(context.horizon)
    assert len(budgets_by_first_and_detour) == 6 * 3
    assert all(horizons == {6, 7, 8} for horizons in budgets_by_first_and_detour.values())


def test_history_prompt_discloses_rates_and_records_the_public_prefix() -> None:
    contexts = build_history_contexts(9)
    backend = StubVLMBackend()
    for context in contexts:
        prompt = build_prediction_request(
            context.public_observation, backend, direct=True
        ).prompt
        assert "0.90" in prompt  # station accuracy
        assert "0.10" in prompt  # per-step switch hazard
        assert context.prefix
        assert context.evaluator_meta["shift_hazard"] == pytest.approx(0.10)
        assert context.evaluator_meta["feedback_noise"] == pytest.approx(0.10)
        assert context.evaluator_meta["true_rule"] in {"red", "blue", "green"}


def test_history_census_is_pinned() -> None:
    contexts = build_history_contexts(81)
    counts = (
        sum(context.delta_probe < -1e-9 for context in contexts),
        sum(context.delta_probe > 1e-9 for context in contexts),
        sum(abs(context.delta_probe) <= 1e-9 for context in contexts),
    )
    assert counts == (59, 13, 9)


def test_one_remaining_step_after_probe_obeys_public_episode_limit(tmp_path) -> None:
    dev = load_dev_grid()
    dev["environment"]["max_env_steps"] = 2
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(dev), encoding="utf-8")
    (context,) = build_history_contexts(1, path)
    assert context.horizon == 1
    assert all(q == pytest.approx(-0.05) for q in context.q_values.values())
    assert context.delta_probe == 0
