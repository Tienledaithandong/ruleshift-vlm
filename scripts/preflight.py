"""Preflight a real experiment run: model, hardware, config and freeze state.

Usage: python scripts/preflight.py [--strict] [--model-key qwen2.5-vl-3b]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

try:
    import ruleshift  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ruleshift.agents.model_check import check_model

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    models = load_yaml("configs/models.yaml")
    key = args.model_key or models["default"]
    model = models["models"][key]
    runtime = models["runtime"]
    model_id = model["hf_id"]

    readiness = check_model(
        model_id,
        local_files_only=runtime["local_files_only"],
        allow_download=runtime["allow_download"],
    )
    test_config = load_yaml("configs/preregistered_test.yaml")

    print(f"model_key={key} model_id={model_id} revision={model['revision']}")
    print(f"device_requested={model['device']} cuda_available={readiness.cuda_available}")
    print(f"local_checkpoint={readiness.local_checkpoint}")
    print(f"test_configuration={test_config['status']}")

    for message in readiness.messages:
        print(f"  {message}")

    if model["revision"] is None:
        print("  WARN: pin revision to a commit hash before a locked run")
    if test_config["status"] != "READY":
        print("  BLOCK: test configuration is not frozen")

    no_go = (
        not readiness.ready
        or model["revision"] is None
        or test_config["status"] != "READY"
    )
    print("GO" if not no_go else "NO-GO")
    if args.strict and no_go:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
