"""Tests for public observation hashing and cache-key separation."""

from __future__ import annotations

from ruleshift.envs.macro import Geometry
from ruleshift.envs.observations import RenderState, build_public_observation


def make_observation(history: list[tuple[str, str | None, str]]):
    return build_public_observation(
        geometry=Geometry(start=(0, 0), gate=(4, 0), station=(0, 0), grid=(5, 5)),
        state=RenderState(position=(0, 0), gate_locked=False, deliveries=0, steps=1),
        history=history,
        last_outcome=history[-1][2],
        candidate_rules=("red", "blue", "green"),
        public_costs={"primitive_step": 0.05},
    )


def test_same_public_state_and_history_gives_same_hashes() -> None:
    first = make_observation([("wait", None, "none")])
    second = make_observation([("wait", None, "none")])
    assert first.public_obs_hash == second.public_obs_hash
    assert first.public_history_hash == second.public_history_hash


def test_same_frame_with_different_history_gives_different_cache_key_material() -> None:
    probed = make_observation([("test", "blue", "accept"), ("commit", "blue", "accept")])
    passive = make_observation([("wait", None, "none"), ("commit", "blue", "accept")])
    assert probed.public_obs_hash == passive.public_obs_hash
    assert probed.public_history_hash != passive.public_history_hash


def test_public_text_does_not_reveal_active_rule() -> None:
    observation = make_observation([("test", "blue", "accept")])
    lowered = observation.public_text.lower()
    assert "active" not in lowered
    assert "true_rule" not in lowered
    assert "shift_time" not in lowered
