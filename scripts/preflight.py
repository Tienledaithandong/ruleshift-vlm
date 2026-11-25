"""Preflight a real experiment run: model, hardware, config and freeze state.

Usage: python scripts/preflight.py [--stage pilot|locked] [--strict]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from ruleshift.agents.model_check import check_model, model_settings, readiness_errors
from ruleshift.evaluator.provenance import implementation_digest

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--stage", choices=("pilot", "locked"), default="pilot")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true",
                        help="assess readiness for an explicitly enabled checkpoint download")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    models = load_yaml("configs/models.yaml")
    key = args.model_key or models["default"]
    if key not in models["models"]:
        parser.error(f"unknown model key: {key}")
    settings = model_settings(models, key, args.device, args.allow_download)
    readiness = check_model(**settings)
    test_config = load_yaml("configs/preregistered_test.yaml")

    print(f"stage={args.stage} model_key={key} model_id={settings['model_id']} revision={settings['revision']}")
    print(f"device_requested={settings['device']} cuda_available={readiness.cuda_available}")
    print(f"local_checkpoint={readiness.local_checkpoint}")
    print(f"test_configuration={test_config['status']}")

    for message in readiness.messages:
        print(f"  {message}")

    errors = readiness_errors(settings, args.stage, test_config)
    if args.stage == "locked" and not errors:
        if test_config["locked_run"]["implementation_sha256"] != implementation_digest():
            errors.append("current source or prompts differ from locked_run")
    for error in errors:
        print(f"  BLOCK: {error}")
    no_go = not readiness.ready or bool(errors)
    print("GO" if not no_go else "NO-GO")
    if args.strict and no_go:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
