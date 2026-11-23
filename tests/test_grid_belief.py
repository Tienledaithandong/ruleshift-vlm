import random

from ruleshift.agents.grid_belief import GridBelief
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap, movement_outcomes

MAP = PublicGridMap(7, 7, frozenset(), (3, 3),
                    {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
                    None, None, None)


def test_belief_normalizes():
    belief = GridBelief()
    assert abs(sum(belief.marginal().values()) - 1.0) < 1e-9


def test_belief_updates_on_move():
    belief = GridBelief()
    belief.update_move((3, 3), "UP", (3, 2), MAP, step=0)
    assert belief.marginal()["base"] > belief.marginal()["remap_swap"]


def test_belief_updates_on_role():
    belief = GridBelief()
    belief.update_role("red", "goal", step=0)
    assert belief.marginal()["base"] > belief.marginal()["hazard_goal_swap"]


def test_belief_updates_on_inspect():
    belief = GridBelief()
    belief.update_inspect("red", "goal", step=0)
    assert belief.marginal()["base"] > belief.marginal()["hazard_goal_swap"]


def test_belief_updates_on_sign():
    belief = GridBelief()
    belief.update_sign(0, "The red tile is not the goal.", step=0)
    assert belief.marginal()["hazard_goal_swap"] > belief.marginal()["base"]


def test_no_shift_false_alarm_rate_is_bounded():
    belief = GridBelief(p_shift=0.02, shift_window=(0, 60))
    rng = random.Random(0)
    position = (3, 3)
    for step in range(30):
        outcomes = movement_outcomes(BASE_RULE, position, "RIGHT", MAP)
        draw = rng.random()
        cumulative = 0.0
        next_position = position
        for candidate, probability in sorted(outcomes.items()):
            cumulative += probability
            if draw <= cumulative:
                next_position = candidate
                break
        belief.update_move(position, "RIGHT", next_position, MAP, step)
        position = next_position
    assert belief.shift_probability() < 0.5


def test_deterministic_residual_is_not_a_shift_detector():
    belief = GridBelief(p_shift=0.02, shift_window=(0, 60))
    rng = random.Random(0)
    position = (3, 3)
    residual_fired = False
    for step in range(30):
        outcomes = movement_outcomes(BASE_RULE, position, "RIGHT", MAP)
        draw = rng.random()
        cumulative = 0.0
        next_position = position
        for candidate, probability in sorted(outcomes.items()):
            cumulative += probability
            if draw <= cumulative:
                next_position = candidate
                break
        expected = (position[0] + 1, position[1])
        residual_fired = residual_fired or next_position != expected
        belief.update_move(position, "RIGHT", next_position, MAP, step)
        position = next_position
    assert residual_fired is True
    assert belief.shift_probability() < 0.5


def test_change_point_belief_detects_shift():
    belief = GridBelief(p_shift=0.3, shift_window=(0, 60))
    for step in range(3):
        belief.update_move((3, 3), "UP", (3, 4), MAP, step=step)
    assert belief.most_likely() == "remap_swap"
    assert belief.shift_probability() > 0.5


def test_predict_is_idempotent_within_a_step():
    belief = GridBelief(p_shift=0.3, shift_window=(0, 60))
    belief.predict(0)
    once = belief.shift_probability()
    assert abs(once - 0.3) < 1e-9

    belief.predict(0)
    assert abs(belief.shift_probability() - once) < 1e-9

    belief.predict(1)
    assert abs(belief.shift_probability() - (1 - 0.7 ** 2)) < 1e-9


def test_multiple_updates_at_one_step_do_not_compound_hazard():
    belief = GridBelief(p_shift=0.3, shift_window=(0, 60))
    for _ in range(4):
        belief.update_interact("nothing", step=0)  # uninformative observation
    assert abs(belief.shift_probability() - 0.3) < 1e-9
