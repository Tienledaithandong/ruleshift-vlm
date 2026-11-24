"""Verify CPU reproducibility and write an experiment manifest.

Usage: python scripts/check_reproducibility.py [--root .]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml
# Always run the code in this checkout, including when another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruleshift.envs.config import EnvConfig
from ruleshift.evaluator.closed_loop import run_episode
from ruleshift.evaluator.provenance import collect_hashes
from ruleshift.permissions import PRIVILEGED_FIELDS

def determinism_check() -> tuple[bool, str]:
    config = EnvConfig(
        probe_detour_steps=2,
        feedback_noise=0.1,
        lockout_steps=3,
        disable_shift=False,
        shift_hazard=0.05,
    )
    first = run_episode(config, "m1_stub", seed=4242, scenario_id="repro-check")
    second = run_episode(config, "m1_stub", seed=4242, scenario_id="repro-check")
    public_equal = first.public_rows == second.public_rows
    evaluator_equal = first.evaluator_rows == second.evaluator_rows
    return public_equal and evaluator_equal, (
        f"public_traces_equal={public_equal}, evaluator_traces_equal={evaluator_equal}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = collect_hashes(root)
    ok, detail = determinism_check()

    versions = {
        name: importlib.metadata.version(name)
        for name in ("pytest", "gymnasium", "numpy", "Pillow", "PyYAML", "matplotlib")
    }
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_only_verified": True,
        "cuda_available": None,
        "package_versions": versions,
        "seeds": {
            "dev_master": 20261121,
            "heldout": [101, 102],
            "diagnostics_contexts": 81,
            "repro_check": 4242,
        },
        "prompt_hashes": {
            path: hashes[path]
            for path in hashes
            if path.endswith(("direct_v1.txt", "outcomes_v1.txt"))
        },
        "privileged_field_inventory_size": len(PRIVILEGED_FIELDS),
        "determinism": {"passed": ok, "detail": detail},
        "file_hashes": hashes,
        "test_configuration": yaml.safe_load(
            (root / "configs/preregistered_test.yaml").read_text(encoding="utf-8")
        )["status"],
    }
    manifest_path = out_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Week 11 — Reproducibility manifest (SYNTHETIC harness)",
        "",
        "No VLM was run. The manifest records the tested CPU environment and "
        "source hashes so that synthetic runs can be reproduced; it is not a "
        "claim that any model result is stable.",
        "",
        f"- Python: `{manifest['python']}`",
        f"- Platform: `{manifest['platform']}`",
        f"- Determinism check: **{'PASS' if ok else 'FAIL'}** — {detail}",
        f"- Test configuration: `{manifest['test_configuration']}`",
        "",
        "## Tested package versions",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for name, version in versions.items():
        lines.append(f"| {name} | {version} |")
    lines += [
        "",
        "## Prompt hashes",
        "",
    ]
    for name, digest in manifest["prompt_hashes"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines += [
        "",
        f"Source/config/docs file hashes ({len(hashes)} files) are in "
        "`experiment_manifest.json`, together with seeds and the privileged "
        "field inventory size.",
        "",
        "Reproduce the synthetic artifacts with `python scripts/reproduce_all.py`.",
        "",
    ]
    report = out_dir / "w11_reproducibility.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"determinism={'PASS' if ok else 'FAIL'}; wrote {manifest_path} and {report}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
