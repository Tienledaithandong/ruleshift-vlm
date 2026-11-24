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


def test_stage_grid_chunk_embeds_literal_map_index(tmp_path):
    module = _load("stage_grid_chunk")
    stage = module.build_stage(7, tmp_path / "m007")
    text = (stage / "run_grid_chunk.py").read_text(encoding="utf-8")
    assert "map_index = 7" in text
    assert 'os.environ.get("MAP_INDEX"' not in text
    assert 'os.environ["MAP_INDEX"]' not in text
    metadata = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"].endswith("-007")
    assert metadata["title"] == "RuleShift grid chunk 007"
    assert metadata["dataset_sources"] == ["tiennamlee/ruleshift-source"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_stage_grid_chunk_rejects_out_of_range_index(tmp_path):
    module = _load("stage_grid_chunk")
    with pytest.raises(ValueError):
        module.build_stage(100, tmp_path / "m100")