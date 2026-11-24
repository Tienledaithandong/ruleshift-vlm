import importlib.util
import json
from pathlib import Path

import pytest


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _scenario_ids(count: int = 5500) -> list[str]:
    return [f"s{index:04d}" for index in range(count)]


def _public_row(scenario_id: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "json_valid": True,
        "fallback": False,
        "valid_action": True,
        "output_tokens": 10,
        "cache_hit": False,
        "chosen_action": "LEFT",
    }


def _evaluator_row(scenario_id: str, index: int) -> dict:
    return {
        "scenario_id": scenario_id,
        "run_seed": 0,
        "success": index % 2 == 0,
        "total_return": float(index % 3),
        "env_steps": 3 + index % 5,
        "tool_units": 0,
        "advisor_calls": 0,
        "total_cost": float(3 + index % 5),
        "adaptation_steps": 0,
        "false_alarm": False,
        "final_rule_estimate": "base",
        "scenario_type": "change_detection",
    }


def _baseline_rows(scenario_ids: list[str], method: str = "no_adaptation") -> list[dict]:
    rows = []
    for index, scenario_id in enumerate(scenario_ids):
        row = _evaluator_row(scenario_id, index)
        row.update({"method": method, "final_rule_estimate": "base"})
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_chunk(root: Path, chunk_id: str, scenario_ids: list[str], start: int = 0) -> None:
    _write_jsonl(
        root / f"grid_real_decisions.{chunk_id}.jsonl",
        [_public_row(scenario_id) for scenario_id in scenario_ids],
    )
    _write_jsonl(
        root / f"grid_real_evaluator_only.{chunk_id}.jsonl",
        [_evaluator_row(scenario_id, start + index) for index, scenario_id in enumerate(scenario_ids)],
    )


def test_freeze_manifest_is_deterministic_and_exact():
    freeze = _load("grid_freeze_manifest")
    first = freeze.grid_freeze_manifest()
    second = freeze.grid_freeze_manifest()
    assert first == second
    assert set(first) == {
        "commit", "model_revision", "config_sha256",
        "scenario_count", "scenario_list_sha256",
    }
    assert first["scenario_count"] == 5500
    assert len(first["config_sha256"]) == 64
    assert len(first["scenario_list_sha256"]) == 64
    assert first["model_revision"] == "66285546d2b821cf421d4f5eb2576359d3770cd3"


def test_baseline_export_writes_public_summary_rows(tmp_path):
    export = _load("export_grid_baselines")
    from ruleshift.envs.grid.config import load_grid_config
    from ruleshift.evaluator.grid_runner import build_grid_scenarios

    config = load_grid_config(Path(__file__).resolve().parents[1] / "configs" / "grid_dev.yaml")
    scenarios = tuple(build_grid_scenarios(1, 1, config))
    out = export.export_grid_baselines(
        config, ["no_adaptation"], tmp_path / "baselines.jsonl", scenarios=scenarios
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == len(scenarios)
    assert {row["method"] for row in rows} == {"no_adaptation"}
    assert all("public_rows" not in row and "evaluator_rows" not in row for row in rows)


def test_merge_accepts_full_coverage(tmp_path):
    merge = _load("merge_grid_chunks")
    ids = _scenario_ids()
    _write_chunk(tmp_path, "c00", ids[:2750], 0)
    _write_chunk(tmp_path, "c01", ids[2750:], 2750)
    baselines = tmp_path / "baselines.jsonl"
    _write_jsonl(baselines, _baseline_rows(ids))
    payload = merge.merge_grid_chunks(tmp_path, baselines, n_boot=10)
    assert payload["scenarios"] == 5500
    assert payload["chunks"] == ["c00", "c01"]
    assert payload["resampling_unit"] == "scenario"
    assert "no_adaptation" in payload["paired_grid_vlm_minus_baseline"]


def test_merge_rejects_incomplete_coverage(tmp_path):
    merge = _load("merge_grid_chunks")
    ids = _scenario_ids()[:-1]
    _write_chunk(tmp_path, "c00", ids)
    baselines = tmp_path / "baselines.jsonl"
    _write_jsonl(baselines, _baseline_rows(ids))
    with pytest.raises(merge.MergeError, match="expected 5500"):
        merge.merge_grid_chunks(tmp_path, baselines, n_boot=10)


def test_merge_rejects_invalid_json_row(tmp_path):
    merge = _load("merge_grid_chunks")
    ids = _scenario_ids()
    _write_chunk(tmp_path, "c00", ids)
    decisions = tmp_path / "grid_real_decisions.c00.jsonl"
    rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    rows[7]["json_valid"] = False
    _write_jsonl(decisions, rows)
    baselines = tmp_path / "baselines.jsonl"
    _write_jsonl(baselines, _baseline_rows(ids))
    with pytest.raises(merge.MergeError, match="non-full-menu"):
        merge.merge_grid_chunks(tmp_path, baselines, n_boot=10)


def test_merge_rejects_duplicate_scenario_across_chunks(tmp_path):
    merge = _load("merge_grid_chunks")
    ids = _scenario_ids()
    _write_chunk(tmp_path, "c00", ids[:3000], 0)
    _write_chunk(tmp_path, "c01", ids[2999:], 2999)
    baselines = tmp_path / "baselines.jsonl"
    _write_jsonl(baselines, _baseline_rows(ids))
    with pytest.raises(merge.MergeError, match="more than one chunk"):
        merge.merge_grid_chunks(tmp_path, baselines, n_boot=10)


def test_merge_rejects_baseline_with_missing_scenario(tmp_path):
    merge = _load("merge_grid_chunks")
    ids = _scenario_ids()
    _write_chunk(tmp_path, "c00", ids)
    baselines = tmp_path / "baselines.jsonl"
    _write_jsonl(baselines, _baseline_rows(ids[:-1]))
    with pytest.raises(merge.MergeError, match="does not cover"):
        merge.merge_grid_chunks(tmp_path, baselines, n_boot=10)
