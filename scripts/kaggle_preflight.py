"""Local-only preflight for a Kaggle GPU run. Makes NO network or API calls.

Usage: python scripts/kaggle_preflight.py [--strict]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

try:
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    messages: list[str] = []
    has_kaggle = importlib.util.find_spec("kaggle") is not None
    messages.append(f"INFO: kaggle package installed = {has_kaggle}")

    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    has_file = credential_path.exists()
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    messages.append(f"INFO: kaggle.json present = {has_file}")
    messages.append(f"INFO: KAGGLE_USERNAME/KAGGLE_KEY set = {has_env}")
    if not (has_file or has_env):
        messages.append("BLOCK: no Kaggle credentials found")

    cuda = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda = torch.cuda.is_available()
    messages.append(f"INFO: CUDA available = {cuda}")
    if not cuda:
        messages.append("BLOCK: no GPU visible on this machine")

    with (ROOT / "configs" / "runtime.yaml").open(encoding="utf-8") as handle:
        runtime = yaml.safe_load(handle)
    repo_url = str(runtime["repo_url"])
    if "PLACEHOLDER" in repo_url:
        messages.append("BLOCK: configs/runtime.yaml repo_url is still a placeholder")
    if runtime["execution"]["status"] != "SETUP_ONLY":
        messages.append("INFO: execution status changed from SETUP_ONLY")
    if not runtime["execution"]["allow_remote_run"]:
        messages.append("BLOCK: execution.allow_remote_run is false")

    for message in messages:
        print(message)
    no_go = any(message.startswith("BLOCK") for message in messages)
    print("GO" if not no_go else "NO-GO")
    if args.strict and no_go:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
