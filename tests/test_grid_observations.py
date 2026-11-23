from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_PUBLIC_OBSERVATION_FIELDS
from ruleshift.permissions import find_forbidden_fields


def make_env(mode="symbolic"):
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode=mode)
    env.reset(seed=0, options={"map": public})
    return env


def test_symbolic_observation_contains_only_public_fields():
    view = make_env().public_view()
    payload = view.to_payload()
    assert set(payload) == set(GRID_PUBLIC_OBSERVATION_FIELDS)
    assert find_forbidden_fields(payload) == []


def test_visual_observation_has_224x224_uint8():
    image = make_env("visual").render()
    assert image.shape == (224, 224, 3)
    assert image.dtype.name == "uint8"


def test_signpost_clause_is_not_in_public_text():
    view = make_env().public_view()
    assert all(clause not in view.public_text for clause in BASE_RULE.sign_clauses)
