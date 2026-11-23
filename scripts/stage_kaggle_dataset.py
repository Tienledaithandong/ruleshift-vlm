"""Assemble the private Kaggle source dataset for the offline pilot kernel.

The kernel (`kaggle/kernel/run.py`) expects the dataset root to contain the
repository layout (``src/``, ``scripts/run_real_pilot.py``, ``configs/``), so
this script copies only what the pilot needs into ``kaggle_stage/ruleshift-source``.
The staging directory is git-ignored.

Provenance: the snapshot is stamped with the commit it was taken from, both in
``dataset-metadata.json`` (the description shown on Kaggle) and in a
``SOURCE_COMMIT`` marker that the kernel prints and asserts. Staging refuses to
run on a dirty working tree, because the copied files would no longer match the
stamped commit.

Usage:
    python scripts/stage_kaggle_dataset.py
    python -m kaggle datasets version -p kaggle_stage/ruleshift-source -r zip \
        -m "snapshot <commit>"
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "kaggle_stage" / "ruleshift-source"

#: Files and directories the pilot kernel imports or installs from.
INCLUDE = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
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


def _git(*args: str) -> str:
    """Return stripped stdout of a git command run in the repository root."""

    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def snapshot_provenance() -> tuple[str, str, str]:
    """Return ``(commit, subject, repo_url)`` for the tree being staged."""

    try:
        commit = _git("rev-parse", "HEAD")
        subject = _git("log", "-1", "--pretty=%s")
        dirty = bool(_git("status", "--porcelain"))
        repo_url = _git("config", "--get", "remote.origin.url") or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise SystemExit(f"cannot read git provenance: {error}") from error
    if len(commit) != 40:
        raise SystemExit(f"unexpected commit id: {commit!r}")
    if dirty:
        raise SystemExit(
            "working tree is dirty; commit or stash first so the snapshot "
            "matches its stamped commit (see `git status --porcelain`)"
        )
    return commit, subject, repo_url


def build_stage() -> Path:
    """Recreate the staging tree and return its root."""

    commit, subject, repo_url = snapshot_provenance()
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
        "description": (
            f"Source snapshot of {repo_url} at commit {commit} ({subject}), "
            "staged from a clean working tree by "
            "scripts/stage_kaggle_dataset.py. The commit is repeated in the "
            "SOURCE_COMMIT marker, which kaggle/kernel/run.py prints and "
            "asserts before the pilot starts."
        ),
    }
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE / "SOURCE_COMMIT").write_text(
        json.dumps(
            {
                "commit": commit,
                "short_commit": commit[:7],
                "subject": subject,
                "repo_url": repo_url,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return STAGE


def main() -> None:
    stage = build_stage()
    count = sum(1 for path in stage.rglob("*") if path.is_file())
    commit = json.loads((stage / "SOURCE_COMMIT").read_text(encoding="utf-8"))
    print(f"staged {count} files in {stage}")
    print(f"snapshot commit: {commit['commit']} ({commit['short_commit']})")
    print(
        "next: python -m kaggle datasets version -p "
        f"{stage.relative_to(ROOT)} -r zip -m \"snapshot {commit['short_commit']}\""
    )


if __name__ == "__main__":
    main()
