"""Build a deterministic freeze manifest for the multi-week grid run.

The manifest is the provenance anchor for every chunk: it records the source
commit, pinned model revision, hashes of the configuration inputs, the exact
scenario count, and a hash of the canonical scenario list. It contains no
timestamp, so two runs on the same tree are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.envs.grid.config import load_grid_config
from ruleshift.evaluator.grid_runner import build_grid_scenarios

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "configs/models.yaml",
    "configs/grid_dev.yaml",
    "configs/preregistered_test.yaml",
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve(paths: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resolved.append(candidate)
    return sorted(resolved, key=lambda item: item.as_posix())


def _model_revision(models_path: Path, model_key: str) -> str:
    import yaml

    payload = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    models = payload.get("models") or {}
    if model_key not in models:
        raise KeyError(f"model key not found: {model_key!r}")
    revision = str(models[model_key].get("revision") or "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError(f"model revision is not a pinned 40-hex commit: {revision!r}")
    return revision


def _grid_config(paths: Sequence[Path]):
    import yaml

    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "width" in payload and "height" in payload:
            return load_grid_config(path)
    raise ValueError("no RuleShift grid config found in manifest inputs")


def _scenario_ids(scenarios: Sequence[Any]) -> list[str]:
    return sorted(str(scenario.scenario_id) for scenario in scenarios)


def grid_freeze_manifest(
    paths: Sequence[str | Path] = DEFAULT_PATHS,
    *,
    model_key: str = "qwen2.5-vl-3b",
) -> dict[str, Any]:
    """Return the deterministic manifest for the full 100x5 grid."""

    resolved = _resolve(paths)
    config = _grid_config(resolved)
    scenarios = build_grid_scenarios(100, 5, config)
    scenario_ids = _scenario_ids(scenarios)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario list contains duplicate ids")
    config_digest = hashlib.sha256()
    for path in resolved:
        config_digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        config_digest.update(b"\0")
        config_digest.update(path.read_bytes())
        config_digest.update(b"\0")
    models_path = next((path for path in resolved if path.name == "models.yaml"), None)
    if models_path is None:
        raise ValueError("configs/models.yaml is required")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "model_revision": _model_revision(models_path, model_key),
        "config_sha256": config_digest.hexdigest(),
        "scenario_count": len(scenario_ids),
        "scenario_list_sha256": _sha256_bytes(
            ("\n".join(scenario_ids) + "\n").encode("utf-8")
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="write JSON here instead of stdout")
    parser.add_argument("--model-key", default="qwen2.5-vl-3b")
    parser.add_argument("--path", action="append", dest="paths", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = grid_freeze_manifest(args.paths or DEFAULT_PATHS, model_key=args.model_key)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
