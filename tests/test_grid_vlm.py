from ruleshift.agents.grid_baselines import as_observation
from ruleshift.agents.grid_vlm import (
    GRID_PROMPT_VERSION, GridStubBackend, GridVLMConfig, GridVLMPolicy,
    _last_blocked_action, parse_grid_output, render_grid_prompt,
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


def test_prompt_requires_every_candidate_id():
    text = render_grid_prompt(observation())
    for rule in (
        "base", "remap_swap", "remap_rotate", "slippery",
        "hazard_goal_swap", "sign_composition",
    ):
        assert rule in text
    assert "every id is required" in text
    assert "If they total 1.05 or 1.15" in text


def test_merged_prompt_uses_the_public_image():
    text = render_grid_prompt(observation())
    assert "TEXT-ONLY" not in text
    assert "image" in text.lower()


def test_public_rows_never_include_the_raw_image():
    view = observation()
    assert view.image is not None
    assert "image" not in view.to_payload()
    assert view.to_payload()["image_hash"]


def test_policy_sends_public_image_to_backend():
    class CapturingBackend(GridStubBackend):
        def __init__(self):
            super().__init__()
            self.request = None

        def generate(self, request):
            self.request = request
            return super().generate(request)

    backend = CapturingBackend()
    obs = observation()
    policy = GridVLMPolicy(backend, GridVLMConfig(model_id="stub-grid-v0"))
    policy.choose(obs, {})
    assert backend.request is not None
    assert backend.request.image is obs.image
    assert backend.request.image_hash == obs.image_hash


def test_grid_token_budget_is_raised():
    assert GridVLMConfig(model_id="stub-grid-v0").max_new_tokens == 2048


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
    assert policy.last_fallback is True
    assert policy.last_parse_errors


def test_policy_falls_back_on_partial_menu():
    import json

    from ruleshift.agents.vlm_backend import ModelResponse

    class PartialBackend:
        def generate(self, request):
            payload = {
                "hypothesis_probabilities": {
                    "base": 0.6, "remap_swap": 0.3, "remap_rotate": 0.1
                },
                "action": "LEFT",
                "confidence": 0.8,
            }
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return ModelResponse(text, 0, 20, 0.0, "partial", "GRID_PARTIAL")

    policy = GridVLMPolicy(
        PartialBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None)
    )
    obs = observation()
    policy.reset(obs, {})
    action = policy.choose(obs, {})
    assert action.kind in obs.action_mask
    assert policy.last_json_valid is False
    assert policy.last_fallback is True
    assert any("missing ids" in error for error in policy.last_parse_errors)
    policy.record(action, obs, 0.0, {})
    row = policy.last_public_row()
    assert row["fallback"] is True
    assert row["parse_error_count"] >= 1
    assert find_forbidden_fields(row) == []


def test_policy_lifetime_totals_survive_episode_reset():
    policy = GridVLMPolicy(
        GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0", revision=None)
    )
    obs = observation()
    policy.reset(obs, {})
    policy.choose(obs, {})
    assert policy.total_model_calls == 1
    policy.reset(obs, {})
    policy.choose(obs, {})
    assert policy.total_model_calls == 1
    assert policy.total_cache_hits == 1


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


def test_record_works_without_explicit_reset():
    policy = GridVLMPolicy(GridStubBackend(), config=GridVLMConfig(model_id="stub-grid-v0"))
    assert policy.advisor_calls == 0
    policy.record(GridAction("UP"), observation(), 0.0, {})
    assert policy.last_public_row()["advisor_calls"] == 0


def test_prompt_version_is_pinned():
    assert GRID_PROMPT_VERSION == "grid_v9"


def test_guard_uses_only_the_latest_outcome():
    payload = observation().to_payload()
    payload["public_history"] = "Step 1: LEFT -> blocked; Step 2: UP -> move"
    assert _last_blocked_action(payload) is None
    payload["public_history"] = "Step 1: LEFT -> blocked; Step 2: UP -> blocked"
    assert _last_blocked_action(payload) == "UP"

def test_policy_guard_replaces_a_blocked_action():
    class FixedBackend:
        def generate(self, request):
            import json

            from ruleshift.agents.vlm_backend import ModelResponse
            from ruleshift.public_api.grid_schemas import GRID_RULE_IDS

            payload = {
                "hypothesis_probabilities": {
                    rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS
                },
                "action": "LEFT",
                "tool_target": None,
                "confidence": 0.5,
            }
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return ModelResponse(text, 0, 20, 0.0, "fixed", "GRID_FIXED")

    payload = observation().to_payload()
    payload["public_history"] = "Step 1: LEFT -> blocked"
    payload["public_text"] = "last=blocked"
    obs = as_observation(payload)
    policy = GridVLMPolicy(
        FixedBackend(), config=GridVLMConfig(model_id="fixed")
    )
    policy.reset(obs, {})
    action = policy.choose(obs, {})

    assert action.kind in obs.action_mask
    assert action.kind != "LEFT"
    assert policy.last_guard_applied is True

def test_tool_result_reaches_history_text_and_prompt():
    """A READ_SIGN value is a public outcome, not just a transient info key."""

    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (1, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=(3, 4),
    )
    env = GridEnv(GridConfig(baseline_slip=0.0, slippery_slip=0.0))
    obs, _ = env.reset(seed=0, options={"map": public})
    nxt, _, _, _, info = env.step(GridAction("READ_SIGN"))

    clause = info["last_tool_value"]
    assert clause
    assert clause in nxt["public_history"]
    assert clause in nxt["public_text"]
    # The rendered prompt is what the VLM actually sees.
    assert clause in render_grid_prompt(as_observation(nxt))


def test_parser_rejects_target_outside_inspect_range():
    import json

    payload = {
        "hypothesis_probabilities": {rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS},
        "action": "INSPECT",
        "tool_target": [6, 6],
        "confidence": 0.5,
    }
    parsed, errors = parse_grid_output(
        json.dumps(payload),
        action_mask=("INSPECT",),
        grid_size=(7, 7),
        position=(0, 0),
        inspect_range=2,
    )
    assert parsed is None
    assert any("inspect_range" in error for error in errors)


def test_policy_rejects_out_of_range_inspect_target_end_to_end():
    """The env publishes inspect_range in public_costs so the policy can enforce it."""

    import json

    from ruleshift.agents.vlm_backend import ModelResponse
    from ruleshift.public_api.grid_schemas import GRID_PHYSICAL_ACTIONS

    class FixedBackend:
        def generate(self, request):
            payload = {
                "hypothesis_probabilities": {
                    rule: 1 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS
                },
                "action": "INSPECT",
                "tool_target": [6, 6],
                "confidence": 0.5,
            }
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return ModelResponse(text, 0, len(text.split()), 0.0, "fixed", "GRID_FIXED")

    public = PublicGridMap(
        7, 7, frozenset(), (3, 3),
        {"red": (3, 1), "blue": (1, 5), "green": (5, 1)},
        key=None, door=None, signpost=None,
    )
    env = GridEnv(GridConfig())
    obs, _ = env.reset(seed=0, options={"map": public})
    assert obs["public_costs"]["inspect_range"] == 2
    view = as_observation(obs)

    policy = GridVLMPolicy(FixedBackend(), config=GridVLMConfig(model_id="fixed"))
    policy.reset(view, {})
    action = policy.choose(view, {})

    assert policy.last_valid_action is False
    assert action.kind in GRID_PHYSICAL_ACTIONS
