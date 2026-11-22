"""Kaggle GPU kernel: run the real offline pilot.

Before pushing:
1. Replace <KAGGLE_USERNAME> in kernel-metadata.json.
2. Add a Kaggle Secret named RULESHIFT_GITHUB_TOKEN (a read-only GitHub token),
   OR make the repository public and leave the token unset.
3. In Kaggle: Settings -> Accelerator -> GPU T4 x2 / P100.
4. Internet must be enabled (already set in kernel-metadata.json).

This script never falls back to the stub.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/ruleshift-vlm")
REPO = "https://github.com/Tienledaithandong/ruleshift-vlm.git"
TOKEN = os.environ.get("RULESHIFT_GITHUB_TOKEN")


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


if not PROJECT.exists():
    clone_url = REPO
    if TOKEN:
        # Token is never printed by run().
        clone_url = REPO.replace("https://", f"https://x-access-token:{TOKEN}@")
    run(["git", "clone", clone_url, str(PROJECT)])

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
run([sys.executable, "-m", "pytest", "-q"])
run([sys.executable, "scripts/nvidia_check.py"] if (PROJECT / "scripts/nvidia_check.py").exists()
    else ["nvidia-smi"])
run([
    sys.executable,
    "scripts/run_real_pilot.py",
    "--model-key", "qwen2.5-vl-3b",
    "--contexts", "81",
    "--device", "cuda",
    "--out", "/kaggle/working/results/real",
    "--report", "/kaggle/working/reports/real_pilot.md",
])

print("PILOT COMPLETE", flush=True)
for path in (
    "/kaggle/working/reports/real_pilot.md",
    "/kaggle/working/results/real/decisions.jsonl",
    "/kaggle/working/results/real/evaluator_only.jsonl",
):
    print(path, Path(path).exists(), flush=True)
