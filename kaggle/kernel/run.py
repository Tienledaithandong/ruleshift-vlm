"""Kaggle GPU kernel: real offline pilot using a private source dataset.

The repo is private, so the kernel copies a source snapshot from the private
Kaggle dataset tiennamlee/ruleshift-source instead of cloning GitHub. No token
is embedded anywhere. Kaggle extracts dataset archives into the input folder,
so the files appear with their original paths (src/, scripts/, configs/, ...).

Runtime settings: GPU T4x2/P100, internet enabled.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/ruleshift-vlm")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


matches = sorted(Path("/kaggle/input").rglob("run_real_pilot.py"))
if not matches:
    print("kaggle input tree:", [p.name for p in Path("/kaggle/input").iterdir()], flush=True)
    raise SystemExit("source dataset not found under /kaggle/input")
SOURCE = matches[0].parent.parent  # .../ruleshift-source
print("source dataset:", SOURCE, flush=True)

if PROJECT.exists():
    shutil.rmtree(PROJECT)
shutil.copytree(SOURCE, PROJECT)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

# Allow the one-time checkpoint download inside this ephemeral kernel copy.
runtime_config = PROJECT / "configs" / "models.yaml"
runtime_text = runtime_config.read_text()
runtime_text = runtime_text.replace("allow_download: false", "allow_download: true")
runtime_text = runtime_text.replace("local_files_only: true", "local_files_only: false")
runtime_config.write_text(runtime_text)

run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
# The dataset snapshot intentionally omits docs/ and research/, so the full
# pytest suite is not run here. The local repo CI already covers it.
run(["nvidia-smi"])
run(
    [
        sys.executable,
        "scripts/run_real_pilot.py",
        "--model-key", "qwen2.5-vl-3b",
        "--contexts", "81",
        "--device", "cuda",
        "--out", "/kaggle/working/results/real",
        "--report", "/kaggle/working/reports/real_pilot.md",
    ]
)

print("PILOT COMPLETE", flush=True)
for path in (
    "/kaggle/working/reports/real_pilot.md",
    "/kaggle/working/results/real/decisions.jsonl",
    "/kaggle/working/results/real/evaluator_only.jsonl",
):
    print(path, Path(path).exists(), flush=True)
