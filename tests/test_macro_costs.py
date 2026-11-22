"""Macro duration and hand-computed cost/return tests.

Expected numbers are derived by hand from the contract in docs/problem_spec.md:
every primitive step costs -0.05, an accepted delivery pays +10, a rejected
commit costs 1 attempt step plus k lockout steps, and a probe costs 2d+1 steps.
"""

from __future__ import annotations

import pytest

from ruleshift.envs.config import EnvConfig, config_from_dev_cell
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.envs.macro import Geometry, MacroExecutor


def test_probe_macro_has_round_trip_cost() -> None:
    geometry = Geometry(start=(0, 0), gate=(4, 0), station=(4, 1), grid=(5, 5))
    ops = MacroExecutor(geometry).plan("PROBE_RED", position=(0, 0)).ops
    assert len(ops) == 11
    # Outbound leg is a plain move; the return leg is marked move_back so the
    # environment can keep the test feedback for the decision boundary.
    assert sum(op.kind == "move" for op in ops) == 5
    assert sum(op.kind == "move_back" for op in ops) == 5
    assert sum(op.kind == "test" for op in ops) == 1
    kinds = [op.kind for op in ops]
    assert kinds.index("test") < kinds.index("move_back")


def test_commit_macro_is_approach_plus_one_commit() -> None:
    geometry = Geometry(start=(0, 0), gate=(4, 0), station=(0, 0), grid=(5, 5))
    ops = MacroExecutor(geometry).plan("COMMIT_BLUE", position=(0, 0)).ops
    assert len(ops) == 5
    assert [op.kind for op in ops] == ["move"] * 4 + ["commit"]
    assert ops[-1].color == "blue"


def test_wait_and_goto_have_public_durations() -> None:
    geometry = Geometry(start=(0, 0), gate=(4, 0), station=(2, 0), grid=(5, 5))
    executor = MacroExecutor(geometry)
    assert len(executor.plan("WAIT", (0, 0)).ops) == 1
    assert all(op.kind == "move" for op in executor.plan("GO_TO_STATION", (0, 0)).ops)
    assert len(executor.plan("GO_TO_STATION", (0, 0)).ops) == 2


def run_actions(config: EnvConfig, actions: list[str], seed: int = 5) -> float:
    env = RuleShiftEnv(config=config)
    env.reset(seed=seed)
    total = 0.0
    for action in actions:
        _, reward, _, truncated, _ = env.step(action)
        total += reward
        if truncated:
            break
    return total


def test_probe_disadvantageous_when_lockout_cheap_and_detour_long() -> None:
    config = EnvConfig(
        probe_detour_steps=5,
        lockout_steps=0,
        feedback_noise=0.0,
        shift_hazard=0.0,
        initial_mechanic="red",
    )
    blind = run_actions(config, ["COMMIT_RED"])
    probe = run_actions(config, ["PROBE_RED", "COMMIT_RED"])
    # blind: 10 - 5*0.05 = 9.75 ; probe: 10 - 16*0.05 = 9.20
    assert blind == pytest.approx(9.75)
    assert probe == pytest.approx(9.20)
    assert blind > probe


def test_probe_advantageous_when_lockout_expensive_and_feedback_clean() -> None:
    config = EnvConfig(
        probe_detour_steps=0,
        lockout_steps=8,
        feedback_noise=0.0,
        shift_hazard=0.0,
        initial_mechanic="blue",
    )
    blind = run_actions(config, ["COMMIT_RED"] + ["WAIT"] * 8 + ["COMMIT_BLUE"])
    probe = run_actions(config, ["PROBE_BLUE", "COMMIT_BLUE"])
    # Blind: 5 + 8 + 1 = 14 primitive steps (no teleport after rejection),
    # so 10 - 14*0.05 = 9.30. Probe: 1 + 5 = 6 steps, 10 - 6*0.05 = 9.70.
    assert blind == pytest.approx(9.30)
    assert probe == pytest.approx(9.70)
    assert probe > blind


def test_dev_cells_expose_both_regimes_without_freezing_test() -> None:
    direct = config_from_dev_cell("dev_direct_action_preferable")
    probing = config_from_dev_cell("dev_probe_preferable")
    assert direct.lockout_steps == 0
    assert direct.probe_detour_steps == 5
    assert probing.lockout_steps == 8
    assert probing.probe_detour_steps == 0
    assert probing.feedback_noise == 0.0
