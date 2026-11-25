import random

from ruleshift.agents.grid_belief import MOVE_EPS, GridBelief
from ruleshift.public_api.grid_rules import (
    BASE_RULE,
    RULE_BY_ID,
    PublicGridMap,
    cell_role,
    effective_map,
    movement_outcomes,
)

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


def test_sign_then_blue_inspect_identifies_movement_equivalent_rules():
    hypotheses = ("base", "hazard_goal_swap", "sign_composition")
    for true_rule in hypotheses:
        belief = GridBelief(hypotheses=hypotheses)
        belief.update_sign(0, RULE_BY_ID[true_rule].sign_clauses[0], step=0)
        belief.update_inspect(
            "blue", cell_role(RULE_BY_ID[true_rule], "blue"), step=1
        )
        assert belief.most_likely() == true_rule


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


def test_bump_into_signpost_or_closed_door_favours_the_blocking_rule():
    """A blocked move must be likely under the rule that intended it."""

    for field in ("signpost", "door"):
        public = PublicGridMap(
            5, 5, frozenset(), (1, 1),
            {"red": (4, 4), "blue": (4, 0), "green": (0, 4)},
            None,
            (2, 1) if field == "door" else None,
            (2, 1) if field == "signpost" else None,
        )
        belief = GridBelief(hypotheses=("base", "remap_swap"))
        belief.update_move((1, 1), "RIGHT", (1, 1), public, step=0, door_open=False)
        marginal = belief.marginal()
        assert marginal["base"] > marginal["remap_swap"], field


def test_update_move_honours_door_open_for_blocking_geometry():
    public = PublicGridMap(
        5, 5, frozenset({(0, 1)}), (1, 1),
        {"red": (4, 4), "blue": (4, 0), "green": (0, 4)},
        None, (2, 1), None,
    )
    belief = GridBelief(hypotheses=("base", "remap_swap"))
    # With the door open, RIGHT reaches (2,1) under base but remapped RIGHT
    # bumps into the wall at (0,1), so staying put favours remap_swap.
    belief.update_move((1, 1), "RIGHT", (1, 1), public, step=0, door_open=True)
    assert belief.marginal()["remap_swap"] > belief.marginal()["base"]


def test_bump_likelihood_matches_effective_map_reference():
    public = PublicGridMap(
        5, 5, frozenset(), (1, 1),
        {"red": (4, 4), "blue": (4, 0), "green": (0, 4)},
        None, None, (2, 1),
    )
    movement_map = effective_map(public, door_open=False)
    likelihoods = {
        rule_id: max(
            movement_outcomes(
                RULE_BY_ID[rule_id], (1, 1), "RIGHT", movement_map
            ).get((1, 1), 0.0),
            MOVE_EPS,
        )
        for rule_id in ("base", "remap_swap")
    }
    belief = GridBelief(hypotheses=("base", "remap_swap"))
    belief.update_move((1, 1), "RIGHT", (1, 1), public, step=0)

    base_mass = belief.marginal()["base"]
    expected = likelihoods["base"] / sum(likelihoods.values())
    assert abs(base_mass - expected) < 1e-9
