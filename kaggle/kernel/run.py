"""Kaggle GPU kernel: real offline pilot using a private source dataset.

The repo is private, so the kernel copies a source snapshot from the private
Kaggle dataset tiennamlee/ruleshift-source instead of cloning GitHub. No token
is embedded anywhere. Kaggle extracts dataset archives into the input folder,
so the files appear with their original paths (src/, scripts/, configs/, ...).

Runtime settings: GPU T4x2/P100, internet enabled.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT = Path("/kaggle/working/ruleshift-vlm")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    matches = [path for path in Path("/kaggle/input").rglob("scripts/run_real_pilot.py")
               if (path.parents[1] / "configs/runtime.yaml").is_file()]
    if len(matches) != 1:
        raise SystemExit(f"expected one RuleShift source dataset, found {len(matches)}")
    SOURCE = matches[0].parent.parent  # .../ruleshift-source
    print("source dataset:", SOURCE, flush=True)

    settings = yaml.safe_load((SOURCE / "configs/runtime.yaml").read_text(encoding="utf-8"))
    if not settings["execution"]["allow_remote_run"]:
        raise SystemExit("Remote pilot disabled in the source snapshot (configs/runtime.yaml)")
    shutil.copytree(SOURCE, PROJECT)

    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT / "src"))

    run([sys.executable, "-m", "pip", "install", "-e", ".[dev,vlm]"])
    # The dataset snapshot intentionally omits docs/ and research/, so the full
    # pytest suite is not run here. The local repo CI already covers it.
    run(["nvidia-smi"])
    run(
        [
            sys.executable,
            "scripts/run_real_pilot.py",
            "--allow-download",
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


if __name__ == "__main__":
    main()
