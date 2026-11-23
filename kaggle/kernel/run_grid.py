"""Kaggle GPU kernel: remote-only RuleShift-Grid pilot with a frozen VLM.

The repo is private, so the kernel copies a source snapshot from the private
Kaggle dataset tiennamlee/ruleshift-source instead of cloning GitHub. No token
is embedded anywhere. Kaggle extracts dataset archives into the input folder,
so the files appear with their original paths (src/, scripts/, configs/, ...).

This mirrors kaggle/kernel/run.py but runs the grid pilot scaffold. It is the
ONLY place allowed to set RULESHIFT_KAGGLE=1 and launch a CUDA pilot.

Runtime settings: GPU T4x2/P100, internet enabled.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/ruleshift-vlm")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


matches = sorted(Path("/kaggle/input").rglob("run_grid_real_pilot.py"))
if not matches:
    print("kaggle input tree:", [p.name for p in Path("/kaggle/input").iterdir()], flush=True)
    raise SystemExit("source dataset not found under /kaggle/input")
SOURCE = matches[0].parent.parent  # .../ruleshift-source
print("source dataset:", SOURCE, flush=True)

if PROJECT.exists():
    shutil.rmtree(PROJECT)
shutil.copytree(SOURCE, PROJECT)

# The snapshot must be traceable to the commit it was staged from.
marker = PROJECT / "SOURCE_COMMIT"
if not marker.exists():
    print("dataset root:", sorted(p.name for p in PROJECT.iterdir()), flush=True)
    raise SystemExit("SOURCE_COMMIT marker missing: source dataset is not traceable")
stamp = json.loads(marker.read_text(encoding="utf-8"))
commit = str(stamp.get("commit", ""))
print("source commit:", commit, stamp.get("short_commit"), flush=True)
print("source subject:", stamp.get("subject"), flush=True)
if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise SystemExit(f"SOURCE_COMMIT does not hold a full commit id: {commit!r}")
expected = os.environ.get("RULESHIFT_EXPECTED_COMMIT", "").strip()
if expected and expected != commit:
    raise SystemExit(
        f"source commit mismatch: dataset has {commit}, expected {expected}"
    )
print("expected commit check:", expected or "skipped (RULESHIFT_EXPECTED_COMMIT unset)", flush=True)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
# The dataset snapshot intentionally omits docs/ and research/, so the full
# pytest suite is not run here. The local repo CI already covers it.
run(["nvidia-smi"])

# Authorize the CUDA pilot ONLY inside this ephemeral kernel. The pilot's hard
# guard refuses any cuda device unless this variable is exactly "1".
os.environ["RULESHIFT_KAGGLE"] = "1"

report = "/kaggle/working/reports/grid_real_pilot.md"
out = "/kaggle/working/results/grid_real"
decisions = f"{out}/grid_real_decisions.jsonl"
evaluator = f"{out}/grid_real_evaluator_only.jsonl"

run(
    [
        sys.executable,
        "scripts/run_grid_real_pilot.py",
        "--model-key", "qwen2.5-vl-3b",
        "--device", "cuda",
        "--maps", "3",
        "--seeds", "1",
        "--max-episodes", "20",
        "--out", out,
        "--report", report,
    ]
)

print("GRID PILOT COMPLETE (scaffold, not a scientific finding)", flush=True)
for path in (report, decisions, evaluator):
    print(path, Path(path).exists(), flush=True)
