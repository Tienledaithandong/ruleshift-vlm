"""Assemble the private Kaggle source dataset for the offline pilot kernel.

The kernel (`kaggle/kernel/run.py`) expects the dataset root to contain the
repository layout (``src/``, ``scripts/run_real_pilot.py``, ``configs/``), so
this script copies only what the pilot needs into ``kaggle_stage/ruleshift-source``.
The staging directory is git-ignored.

Usage:
    python scripts/stage_kaggle_dataset.py
    python -m kaggle datasets version -p kaggle_stage/ruleshift-source -r zip \
        -m "fix prompt example"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "kaggle_stage" / "ruleshift-source"

#: Files and directories the pilot kernel imports or installs from.
INCLUDE = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "requirements.lock",
    "src",
    "scripts",
    "configs",
    "tests",
    "kaggle/kernel",
)

#: Caches and generated output that must never be uploaded.
IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.egg-info",
    "results",
    "reports",
    "kaggle_outputs",
    "kaggle_stage",
    ".venv",
    ".git",
)


def build_stage() -> Path:
    """Recreate the staging tree and return its root."""

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for relative in INCLUDE:
        source = ROOT / relative
        if not source.exists():
            raise SystemExit(f"missing required path: {relative}")
        target = STAGE / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORE)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    dataset_meta = {
        "title": "RuleShift source snapshot",
        "id": "tiennamlee/ruleshift-source",
        "licenses": [{"name": "MIT"}],
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, indent=2) + "\n", encoding="utf-8"
    )
    return STAGE


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    stage = build_stage()
    count = sum(1 for path in stage.rglob("*") if path.is_file())
    print(f"staged {count} files in {stage}")
    print(
        "next: python -m kaggle datasets version -p "
        f"{stage.relative_to(ROOT)} -r zip -m \"Update pilot source snapshot\""
    )


if __name__ == "__main__":
    main()
