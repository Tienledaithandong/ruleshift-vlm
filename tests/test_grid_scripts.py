import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def _load_pilot_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_grid_real_pilot.py"
    spec = importlib.util.spec_from_file_location("run_grid_real_pilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module



def test_smoke_script_writes_outputs(tmp_path):
    out = tmp_path / "grid"
    subprocess.run(
        [sys.executable, "scripts/run_grid_smoke.py", "--quick",
         "--maps", "2", "--seeds", "1", "--out", str(out)],
        check=True,
    )
    assert (out / "grid_smoke.json").exists()
    assert (out / "grid_smoke.md").exists()
    payload = json.loads((out / "grid_smoke.json").read_text(encoding="utf-8"))
    assert "summary" in payload
    header = (out / "grid_smoke.md").read_text(encoding="utf-8").splitlines()[0]
    assert header == "# SYNTHETIC CPU SMOKE \u2014 NOT A SCIENTIFIC FINDING"


def test_audit_script_passes_on_small_config(tmp_path):
    out = tmp_path / "audit"
    subprocess.run(
        [sys.executable, "scripts/run_grid_audit.py", "--maps", "3",
         "--out", str(out)],
        check=True,
    )
    assert (out / "grid_audit.md").exists()


def test_grid_real_pilot_refuses_local_cuda_without_kaggle_env(tmp_path):
    import os

    env = dict(os.environ)
    env.pop("RULESHIFT_KAGGLE", None)
    result = subprocess.run(
        [sys.executable, "scripts/run_grid_real_pilot.py", "--device", "cuda",
         "--out", str(tmp_path), "--report", str(tmp_path / "report.md")],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "RULESHIFT_KAGGLE" in result.stdout + result.stderr

def test_behavioral_gate_rejects_constant_action():
    pilot = _load_pilot_module()
    rows = [
        {"chosen_action": "LEFT", "agent_position": [1, 1], "env_steps": 0,
         "json_valid": True, "fallback": False},
        {"chosen_action": "LEFT", "agent_position": [1, 2], "env_steps": 1,
         "json_valid": True, "fallback": False},
    ]
    gate = pilot.behavioral_gate(rows, 2)
    assert gate["passed"] is False
    assert gate["distinct_actions"] == ["LEFT"]


def test_behavioral_gate_passes_two_actions_and_movement():
    pilot = _load_pilot_module()
    rows = [
        {"chosen_action": "LEFT", "agent_position": [1, 1], "env_steps": 0,
         "json_valid": True, "fallback": False},
        {"chosen_action": "UP", "agent_position": [1, 2], "env_steps": 1,
         "json_valid": True, "fallback": False},
    ]
    gate = pilot.behavioral_gate(rows, 2)
    assert gate["passed"] is True
    assert gate["moved_steps"] == 1


def test_behavioral_gate_rejects_fallback_and_invalid_json():
    pilot = _load_pilot_module()
    rows = [
        {"chosen_action": "LEFT", "agent_position": [1, 1], "env_steps": 0,
         "json_valid": False, "fallback": True},
        {"chosen_action": "UP", "agent_position": [1, 2], "env_steps": 1,
         "json_valid": True, "fallback": False},
    ]
    assert pilot.behavioral_gate(rows, 2)["passed"] is False


def _semantic_row(scenario, action, position, steps, probs=None, history="no prior steps"):
    return {
        "scenario_id": scenario,
        "chosen_action": action,
        "agent_position": list(position),
        "env_steps": steps,
        "public_history": history,
        "action_mask": ["UP", "DOWN", "LEFT", "RIGHT", "INSPECT"],
        "hypothesis_probabilities": probs or {
            "base": 0.2, "remap_swap": 0.2, "remap_rotate": 0.2,
            "slippery": 0.2, "hazard_goal_swap": 0.1, "sign_composition": 0.1,
        },
    }


def test_semantic_gate_accepts_belief_change_tool_use_and_base_success():
    pilot = _load_pilot_module()
    rows = [
        _semantic_row("a", "RIGHT", (1, 5), 0, {
            "base": 0.5, "remap_swap": 0.1, "remap_rotate": 0.1,
            "slippery": 0.1, "hazard_goal_swap": 0.1, "sign_composition": 0.1,
        }),
        _semantic_row("a", "INSPECT", (2, 5), 1, {
            "base": 0.7, "remap_swap": 0.1, "remap_rotate": 0.05,
            "slippery": 0.05, "hazard_goal_swap": 0.05, "sign_composition": 0.05,
        }),
        _semantic_row("b", "READ_SIGN", (1, 5), 0, {
            "base": 0.2, "remap_swap": 0.5, "remap_rotate": 0.1,
            "slippery": 0.1, "hazard_goal_swap": 0.05, "sign_composition": 0.05,
        }),
    ]
    episodes = [
        SimpleNamespace(scenario_id="a", success=True),
        SimpleNamespace(scenario_id="b", success=False),
    ]
    evaluator = [
        {"scenario_id": "a", "true_rule": "base", "final_rule_estimate": "base"},
        {"scenario_id": "b", "true_rule": "remap_swap", "final_rule_estimate": "remap_swap"},
    ]
    gate = pilot.semantic_gate(rows, episodes, evaluator)
    assert gate["passed"] is True
    assert gate["unique_belief_vectors"] >= 2
    assert gate["tool_calls"] > 0
    assert gate["base_success"] == 1


def test_semantic_gate_rejects_constant_beliefs_and_blocked_repeat():
    pilot = _load_pilot_module()
    constant = {
        "base": 0.17, "remap_swap": 0.17, "remap_rotate": 0.17,
        "slippery": 0.17, "hazard_goal_swap": 0.16, "sign_composition": 0.16,
    }
    rows = [
        _semantic_row("a", "LEFT", (1, 5), 0, constant),
        _semantic_row(
            "a", "LEFT", (1, 5), 1, constant,
            history="Step 1: LEFT -> blocked",
        ),
        _semantic_row("b", "LEFT", (1, 5), 0, constant),
        _semantic_row(
            "b", "LEFT", (1, 5), 1, constant,
            history="Step 1: LEFT -> blocked",
        ),
    ]
    gate = pilot.semantic_gate(rows)
    assert gate["passed"] is False
    assert any("belief" in reason for reason in gate["reasons"])
    assert any("blocked" in reason for reason in gate["reasons"])

def test_semantic_gate_does_not_treat_hazard_as_blocked():
    pilot = _load_pilot_module()
    rows = [
        _semantic_row("a", "LEFT", (1, 5), 0),
        _semantic_row(
            "a", "LEFT", (1, 5), 1,
            history="Step 1: LEFT -> hazard",
        ),
        _semantic_row("b", "RIGHT", (1, 5), 0),
    ]
    gate = pilot.semantic_gate(rows)
    assert gate["blocked_transitions"] == 0
    assert gate["blocked_repeats"] == 0

def test_select_chunk_scenarios_uses_map_range():
    pilot = _load_pilot_module()
    scenarios = tuple(SimpleNamespace(map_seed=map_seed) for map_seed in range(100))
    selected = pilot.select_chunk_scenarios(scenarios, 20, 10)
    assert {scenario.map_seed for scenario in selected} == set(range(20, 30))
    assert len(selected) == 10
    assert len(pilot.select_chunk_scenarios(scenarios, 95, 0)) == 5


def test_one_map_chunk_has_exactly_55_scenarios():
    pilot = _load_pilot_module()
    from ruleshift.envs.grid.config import load_grid_config
    from ruleshift.evaluator.grid_runner import build_grid_scenarios

    config = load_grid_config(Path(__file__).resolve().parents[1] / "configs" / "grid_dev.yaml")
    scenarios = build_grid_scenarios(100, 5, config)
    assert len(scenarios) == 5500
    chunk = pilot.select_chunk_scenarios(scenarios, 0, 1)
    assert len(chunk) == 55
    assert {scenario.map_seed for scenario in chunk} == {0}
    assert {scenario.run_seed for scenario in chunk} == {0, 1, 2, 3, 4}


def test_chunk_artifact_paths_are_stamped():
    pilot = _load_pilot_module()
    paths = pilot.chunk_artifact_paths(Path("out"), "c02")
    assert paths["decisions"].name == "grid_real_decisions.c02.jsonl"
    assert paths["evaluator"].name == "grid_real_evaluator_only.c02.jsonl"
    assert paths["cache"].name == "response_cache.c02.jsonl"
    report = pilot.chunk_report_path(Path("reports/pilot.md"), "c02")
    assert report.name == "pilot.c02.md"
