"""Regressions for running the intended checkout and the appropriate research stage."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ruleshift.agents.model_check import model_settings, readiness_errors
from ruleshift.evaluator.provenance import collect_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_can_precede_freeze_and_download_is_opt_in():
    model = {"hf_id": "org/model", "device": "cuda", "revision": "a" * 40}
    config = {"models": {"anchor": model}, "runtime": {
        "allow_download": False, "local_files_only": True, "cache_dir": "cache"
    }}
    settings = model_settings(config, "anchor", "cpu", False)
    draft = {"status": "DRAFT_NOT_FROZEN", "frozen_commit": None}
    assert settings["device"] == "cpu"
    assert settings["local_files_only"] and not settings["allow_download"]
    assert not readiness_errors(settings, "pilot", draft)
    assert readiness_errors(settings, "locked", draft)
    assert readiness_errors({**settings, "revision": "main"}, "pilot", draft)
    assert not model_settings(config, "anchor", "cpu", True)["local_files_only"]


def test_cli_uses_its_checkout_when_another_ruleshift_is_installed(tmp_path):
    foreign = tmp_path / "ruleshift"
    foreign.mkdir()
    (foreign / "__init__.py").write_text("raise RuntimeError('loaded the wrong checkout')")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/smoke_env.py"), "--seed", "17", "--decisions", "1"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, check=True,
    )


def test_staging_help_preserves_an_existing_snapshot(tmp_path):
    script = tmp_path / "scripts/stage_kaggle_dataset.py"
    script.parent.mkdir()
    shutil.copyfile(ROOT / "scripts" / script.name, script)
    marker = tmp_path / "kaggle_stage/ruleshift-source/marker"
    marker.parent.mkdir(parents=True)
    marker.touch()
    subprocess.run([sys.executable, str(script), "--help"], capture_output=True, check=True)
    assert marker.is_file()


def test_provenance_excludes_old_source_snapshots(tmp_path):
    for name in ("src/ruleshift/model.py", "kaggle_outputs/old/src/ruleshift/model.py",
                 "kaggle_stage/ruleshift-source/src/ruleshift/model.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    assert set(collect_hashes(tmp_path)) == {"src/ruleshift/model.py"}
