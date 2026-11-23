from ruleshift.agents.grid_baselines import as_observation
from ruleshift.agents.grid_vlm import (
    GRID_PROMPT_VERSION, GridStubBackend, GridVLMConfig, GridVLMPolicy,
    parse_grid_output, render_grid_prompt,
)
from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.grid_env import GridEnv
from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_rules import PublicGridMap
from ruleshift.public_api.grid_schemas import GRID_RULE_IDS, GridAction


def observation():
    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=(5, 5), door=(5, 6), signpost=(4, 3),
    )
    env = GridEnv(GridConfig(), observation_mode="visual")
    obs, _ = env.reset(seed=0, options={"map": public})
    # The runner wraps env payloads in an attribute-accessible view before
    # calling a policy; direct-call tests should use the same shape.
    return as_observation(obs)


def test_prompt_is_public_only():
    text = render_grid_prompt(observation())
    assert find_forbidden_fields({"prompt": text}) == []
    assert "true_rule" not in text
    assert "shift_time" not in text


def test_prompt_lists_the_grid_actions():
    text = render_grid_prompt(observation())
    for action in ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN"):
        assert action in text


def test_parser_accepts_a_valid_grid_output():
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "tool_target": None,
        "confidence": 0.5,
    }
    import json
    parsed, errors = parse_grid_output(
        json.dumps(payload),
        action_mask=("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "INSPECT", "READ_SIGN"),
        grid_size=(7, 7),
    )
    assert errors == []
    assert parsed["action"] == "UP"


def test_parser_rejects_privileged_field():
    import json
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "UP",
        "shift_time": 3,
    }
    parsed, errors = parse_grid_output(json.dumps(payload), action_mask=("UP",), grid_size=(7, 7))
    assert parsed is None
    assert any("privileged" in error for error in errors)


def test_parser_rejects_out_of_range_tool_target():
    import json
    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "INSPECT",
        "tool_target": [99, 99],
    }
    parsed, errors = parse_grid_output(
        json.dumps(payload), action_mask=("INSPECT",), grid_size=(7, 7)
    )
    assert parsed is None
    assert errors


def test_policy_uses_the_cache_for_identical_observations():
    backend = GridStubBackend()
    policy = GridVLMPolicy(backend, config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    first = policy.choose(obs, {})
    policy.record(first, obs, 0.0, {"env_steps": 0})
    policy.choose(obs, {})
    assert backend.calls == 1


def test_policy_returns_a_legal_action():
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    assert isinstance(action, GridAction)
    assert action.kind in obs.action_mask


def test_policy_falls_back_on_malformed_output():
    class BrokenBackend:
        calls = 0

        def generate(self, request):
            from ruleshift.agents.vlm_backend import ModelResponse
            self.calls += 1
            return ModelResponse("{ broken", 0, 3, 0.0, "broken", "GRID_BROKEN")

    policy = GridVLMPolicy(BrokenBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    assert action.kind in {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT"}
    assert policy.last_json_valid is False


def test_policy_rows_are_public_only():
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None))
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    policy.record(action, obs, 0.0, {"env_steps": 0, "tool_budget_remaining": 3})
    row = policy.last_public_row()
    assert find_forbidden_fields(row) == []


def test_policy_tolerates_raw_dict_observations():
    raw = observation().to_payload()
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0"))
    policy.reset(raw, {})
    action = policy.choose(raw, {})
    assert isinstance(action, GridAction)
    assert action.kind in raw["action_mask"]
    policy.record(action, raw, 0.0, {})
    assert find_forbidden_fields(policy.last_public_row()) == []


def test_prompt_version_is_pinned():
    assert GRID_PROMPT_VERSION == "grid_v1"
