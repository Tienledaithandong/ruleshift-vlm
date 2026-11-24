import json
from pathlib import Path
import subprocess
import sys


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
