"""Hand-computed checks for the evaluator-only shift-aware Bayes filter.

All expected values below are arithmetic literals derived by hand from the
transition and observation model in docs/problem_spec.md, never from the
filter's own helpers.
"""

from __future__ import annotations

import pytest

from ruleshift.evaluator.bayes_filter import RuleShiftFilter


def make_filter(noise: float = 0.0, hazard: float = 0.1) -> RuleShiftFilter:
    return RuleShiftFilter(
        rules=("red", "blue", "green"),
        initial_rule="red",
        shift_hazard=hazard,
        shift_targets=("blue", "green"),
        feedback_noise=noise,
    )


def test_initial_belief_is_certain_on_old_default() -> None:
    belief = make_filter().initial_belief()
    assert belief[("red", False)] == 1.0
    assert sum(belief.values()) == pytest.approx(1.0)
    assert set(belief) == {
        (rule, shifted)
        for rule in ("red", "blue", "green")
        for shifted in (False, True)
    }


def test_one_primitive_step_transition_matches_hand_calculation() -> None:
    belief = make_filter().advance(make_filter().initial_belief())
    # no shift: 0.9 on red/c=0; shift: 0.1 split evenly over blue and green
    assert belief[("red", False)] == pytest.approx(0.9)
    assert belief[("blue", True)] == pytest.approx(0.05)
    assert belief[("green", True)] == pytest.approx(0.05)
    assert belief[("red", True)] == pytest.approx(0.0)


def test_at_most_one_shift_and_no_shift_mass_decays_geometrically() -> None:
    belief = make_filter().advance(make_filter().initial_belief(), primitive_steps=5)
    unshifted = sum(value for (_, shifted), value in belief.items() if not shifted)
    assert unshifted == pytest.approx(0.9**5)


def test_one_step_posterior_after_clean_probe_accept() -> None:
    bayes = make_filter(noise=0.0, hazard=0.1)
    posterior = bayes.step(bayes.initial_belief(), "PROBE_BLUE", "accept")
    # accept is impossible under red/green when eps=0, certain under blue
    assert posterior[("blue", True)] == pytest.approx(1.0)
    assert posterior[("red", False)] == pytest.approx(0.0)
    assert posterior[("green", True)] == pytest.approx(0.0)


def test_one_step_posterior_after_noisy_probe_accept() -> None:
    bayes = make_filter(noise=0.1, hazard=0.1)
    posterior = bayes.step(bayes.initial_belief(), "PROBE_BLUE", "accept")
    # unnormalised: red .9*.1=.09, blue .05*.9=.045, green .05*.1=.005, Z=.14
    assert posterior[("red", False)] == pytest.approx(9 / 14)
    assert posterior[("blue", True)] == pytest.approx(9 / 28)
    assert posterior[("green", True)] == pytest.approx(1 / 28)
    assert sum(posterior.values()) == pytest.approx(1.0)


def test_commit_reject_is_deterministic_and_renormalises() -> None:
    bayes = make_filter(noise=0.1, hazard=0.1)
    prior = bayes.advance(bayes.initial_belief())
    posterior = bayes.observe(prior, "COMMIT_BLUE", "reject")
    # reject impossible under blue; red .9 and green .05 survive, Z=.95
    assert posterior[("red", False)] == pytest.approx(18 / 19)
    assert posterior[("blue", True)] == pytest.approx(0.0)
    assert posterior[("green", True)] == pytest.approx(1 / 19)


def test_non_informative_observation_leaves_belief_unchanged() -> None:
    bayes = make_filter()
    prior = bayes.advance(bayes.initial_belief())
    assert bayes.observe(prior, "WAIT", "none") == prior
    assert bayes.observe(prior, "GO_TO_STATION", "none") == prior


def test_impossible_observation_is_rejected() -> None:
    bayes = make_filter()
    with pytest.raises(ValueError):
        bayes.observe(bayes.initial_belief(), "COMMIT_RED", "reject")
