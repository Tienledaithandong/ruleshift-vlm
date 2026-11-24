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


def test_select_chunk_scenarios_uses_map_range():
    pilot = _load_pilot_module()
    scenarios = tuple(SimpleNamespace(map_seed=map_seed) for map_seed in range(100))
    selected = pilot.select_chunk_scenarios(scenarios, 20, 10)
    assert {scenario.map_seed for scenario in selected} == set(range(20, 30))
    assert len(selected) == 10
    assert len(pilot.select_chunk_scenarios(scenarios, 95, 0)) == 5


def test_chunk_artifact_paths_are_stamped():
    pilot = _load_pilot_module()
    paths = pilot.chunk_artifact_paths(Path("out"), "c02")
    assert paths["decisions"].name == "grid_real_decisions.c02.jsonl"
    assert paths["evaluator"].name == "grid_real_evaluator_only.c02.jsonl"
    assert paths["cache"].name == "response_cache.c02.jsonl"
    report = pilot.chunk_report_path(Path("reports/pilot.md"), "c02")
    assert report.name == "pilot.c02.md"
