"""Check the submission machine for a Kaggle pilot. Makes no network calls.

Usage: python scripts/kaggle_preflight.py [--strict]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    messages: list[str] = []
    has_kaggle = importlib.util.find_spec("kaggle") is not None
    messages.append(f"INFO: kaggle package installed = {has_kaggle}")
    if not has_kaggle:
        messages.append("BLOCK: install the Kaggle CLI on the submission machine")

    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    has_file = credential_path.exists()
    has_env = bool(os.environ.get("KAGGLE_API_TOKEN") or
                   (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))
    messages.append(f"INFO: kaggle.json present = {has_file}")
    messages.append(f"INFO: credential environment variables present = {has_env}")
    if not (has_file or has_env):
        messages.append("BLOCK: no Kaggle credentials found")

    with (ROOT / "configs" / "runtime.yaml").open(encoding="utf-8") as handle:
        runtime = yaml.safe_load(handle)
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
