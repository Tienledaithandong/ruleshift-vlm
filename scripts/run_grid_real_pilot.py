"""Remote-only real frozen-VLM pilot scaffold for RuleShift-Grid.

This script is a SCAFFOLD, not a scientific finding. It exists so a genuine
frozen checkpoint can be exercised in an ephemeral, internet-enabled Kaggle
GPU kernel via ``kaggle/kernel/run_grid.py``. It never falls back to a stub and
never downloads a checkpoint locally.

HARD SAFETY GATE: a ``cuda`` device is refused unless ``RULESHIFT_KAGGLE=1`` is
set. The guard runs before torch/transformers are imported, so no local GPU job
can start by accident. Nothing here is run in local CI with a real backend.

Usage (Kaggle only):
    RULESHIFT_KAGGLE=1 python scripts/run_grid_real_pilot.py \
        --device cuda --maps 3 --seeds 1 --out results/grid_real \
        --report reports/grid_real_pilot.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Always prefer this checkout's src over any installed copy: the installed
# site-packages build does not carry the grid modules or prompt data files.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = "configs/models.yaml"
PREREG_CONFIG = "configs/preregistered_test.yaml"
DISCLAIMER = "REAL FROZEN VLM — SCAFFOLD RUN, NOT A SCIENTIFIC FINDING"
SUPPORTED_SCENARIOS = frozenset({"identification", "change_detection", "no_shift"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="qwen2.5-vl-3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maps", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="results/grid_real")
    parser.add_argument("--report", default="reports/grid_real_pilot.md")
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument(
        "--max-episodes", type=int, default=20,
        help="cap on the number of episodes actually executed (scaffold only)",
    )
    return parser.parse_args(argv)


def guard_local_cuda(device: str) -> None:
    """Hard barrier: never start a local GPU job or checkpoint download.

    This must be called before importing torch or transformers. The Kaggle
    kernel sets ``RULESHIFT_KAGGLE=1`` immediately before invoking this script.
    """

    if device.startswith("cuda") and os.environ.get("RULESHIFT_KAGGLE") != "1":
        print(
            "Refusing to run a CUDA pilot outside Kaggle. A real frozen-VLM run "
            "may only execute in the remote kernel, which sets RULESHIFT_KAGGLE=1. "
            "Do NOT run this locally: it would use a GPU and may download a "
            "checkpoint. Set RULESHIFT_KAGGLE=1 only inside the Kaggle kernel.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _markdown(payload: dict) -> str:
    metrics = payload["metrics"]
    lines = [
        f"# {DISCLAIMER}",
        "",
        "This is an instrumentation scaffold produced with a genuine frozen "
        "VLM in an ephemeral Kaggle kernel. It is **not** a benchmark result, a "
        "bound, a hypothesis test, or a scientific finding.",
        "",
        f"- Model: `{payload['model_id']}` (revision `{payload['revision']}`)",
        f"- Device: `{payload['device']}`; dtype: `{payload['dtype']}`",
        f"- Grid config: `{payload['config']}`",
        f"- Maps: {payload['maps']}; seeds: {payload['seeds']}; "
        f"max episodes: {payload['max_episodes']}",
        f"- Scenarios run: {payload['scenarios']}; episodes: {payload['episodes']}; "
        f"public decision rows: {payload['public_rows']}",
        f"- Preregistered test config status: `{payload['prereg_status']}` "
        "(read only; never written by this script)",
        "",
        "## Execution diagnostics (not evidence)",
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
        f"| JSON-valid decision rate | {metrics['json_valid_rate']:.4f} |",
        f"| Valid-action rate | {metrics['valid_action_rate']:.4f} |",
        f"| Mean output tokens per decision | {metrics['mean_output_tokens']:.2f} |",
        f"| Total output tokens | {metrics['total_output_tokens']} |",
        f"| Cache hit rate | {metrics['cache_hit_rate']:.4f} |",
        f"| Model calls (new responses) | {metrics['model_calls']} |",
        "",
        "The v10 lesson is honoured here: `max_new_tokens=2048` is never "
        "shrunk, because a truncated JSON response is counted as invalid.",
        "",
        "## Paired 95% CI (scenario-level bootstrap) — PLACEHOLDER",
        "",
        "| Comparison | Mean difference | Low | High | Scenarios |",
        "| --- | --- | --- | --- | --- |",
        "| grid_vlm vs public baselines | PENDING | PENDING | PENDING | PENDING |",
        "",
        "Paired CIs are intentionally left as placeholders. This scaffold runs "
        f"{payload['maps']} maps x {payload['seeds']} seeds "
        f"({payload['episodes']} episodes), far below the frozen 100x5 gate. "
        "Per the preregistered plan the resampling unit is the scenario (never "
        "the seed or frame), and a positive difference in a handful of episodes "
        "must not be treated as evidence.",
        "",
        "## Success rate is deliberately omitted",
        "",
        "A success rate from this run is not reported as evidence: the map and "
        "seed counts are far below the 100x5 gate and the test set is not "
        "frozen (`status: DRAFT_NOT_FROZEN`). Public decisions and privileged "
        "traces are written to separate files and must never be joined back "
        "into model input.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # HARD GATE first — nothing below runs for a local CUDA invocation, and no
    # torch/transformers import has happened yet.
    guard_local_cuda(args.device)

    # Imports that could touch heavy dependencies live after the gate.
    import yaml

    from ruleshift.agents.cache import ResponseCache
    from ruleshift.agents.grid_vlm import GridVLMConfig, GridVLMPolicy
    from ruleshift.agents.vlm_backend import TransformersVLMBackend
    from ruleshift.envs.grid.config import load_grid_config
    from ruleshift.envs.grid.grid_env import GridEnv
    from ruleshift.evaluator.grid_runner import (
        build_grid_scenarios,
        run_grid_episode,
    )
    from ruleshift.permissions import find_forbidden_fields

    on_kaggle = os.environ.get("RULESHIFT_KAGGLE") == "1"

    with (ROOT / MODELS_CONFIG).open(encoding="utf-8") as handle:
        models = yaml.safe_load(handle)
    if args.model_key not in models.get("models", {}):
        raise SystemExit(f"unknown model key: {args.model_key!r}")
    model = models["models"][args.model_key]
    runtime = models["runtime"]

    revision = str(model.get("revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"model revision is not a pinned 40-hex commit: {revision!r}")

    if not on_kaggle:
        # The committed config must stay safe and pinned. The Kaggle kernel
        # patches these two fields in its ephemeral copy, hence the skip.
        if runtime.get("local_files_only") is not True:
            raise SystemExit(
                f"{MODELS_CONFIG} must have local_files_only: true in the committed config"
            )
        config_device = model.get("device")
        if args.device != config_device:
            raise SystemExit(
                f"CLI device {args.device!r} disagrees with config device {config_device!r}"
            )

    # preregistered_test.yaml is READ ONLY — used for the status line in the
    # report; this script never writes it.
    with (ROOT / PREREG_CONFIG).open(encoding="utf-8") as handle:
        prereg = yaml.safe_load(handle)
    prereg_status = str(prereg.get("status", "UNKNOWN"))

    backend = TransformersVLMBackend(
        model_id=str(model["hf_id"]),
        revision=revision,
        device=args.device,
        dtype=str(model.get("dtype", "bfloat16")),
    )
    policy_config = GridVLMConfig(
        model_id=str(model["hf_id"]),
        revision=revision,
        decoding_seed=int(runtime.get("seed", 0)),
        max_new_tokens=2048,  # never shrink: truncation is counted as invalid JSON
    )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(out_dir / "response_cache.jsonl")
    policy = GridVLMPolicy(backend, policy_config, cache)

    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    grid_config = load_grid_config(config_path)
    scenarios = tuple(
        scenario
        for scenario in build_grid_scenarios(args.maps, args.seeds, grid_config)
        if scenario.scenario_type in SUPPORTED_SCENARIOS
    )[: args.max_episodes]

    episodes = []
    for scenario in scenarios:
        env = GridEnv(grid_config, observation_mode="visual")
        observation, info = env.reset(
            seed=scenario.run_seed, options={"scenario": scenario}
        )
        policy.name = "grid_vlm"
        episodes.append(
            run_grid_episode(env, policy, grid_config, scenario, observation, info)
        )

    public_rows = [row for episode in episodes for row in episode.public_rows]
    leaks = [
        (index, hits)
        for index, row in enumerate(public_rows)
        for hits in (find_forbidden_fields(row),)
        if hits
    ]
    if leaks:
        raise SystemExit(
            f"refusing to write public decisions: privileged fields leaked: {leaks[:3]}"
        )
    evaluator_rows = [row for episode in episodes for row in episode.evaluator_rows]

    decisions_path = out_dir / "grid_real_decisions.jsonl"
    evaluator_path = out_dir / "grid_real_evaluator_only.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in public_rows),
        encoding="utf-8",
    )
    evaluator_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in evaluator_rows),
        encoding="utf-8",
    )

    metrics = {
        "json_valid_rate": _mean([float(row.get("json_valid", False)) for row in public_rows]),
        "valid_action_rate": _mean([float(row.get("valid_action", False)) for row in public_rows]),
        "mean_output_tokens": _mean([float(row.get("output_tokens", 0)) for row in public_rows]),
        "total_output_tokens": int(sum(int(row.get("output_tokens", 0)) for row in public_rows)),
        "cache_hit_rate": _mean([float(row.get("cache_hit", False)) for row in public_rows]),
        "model_calls": int(policy.model_calls),
    }
    payload = {
        "disclaimer": DISCLAIMER,
        "model_id": str(model["hf_id"]),
        "revision": revision,
        "device": args.device,
        "dtype": str(model.get("dtype", "bfloat16")),
        "config": str(args.config),
        "maps": int(args.maps),
        "seeds": int(args.seeds),
        "max_episodes": int(args.max_episodes),
        "scenarios": len(scenarios),
        "episodes": len(episodes),
        "public_rows": len(public_rows),
        "prereg_status": prereg_status,
        "metrics": metrics,
    }

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(payload), encoding="utf-8")

    print(DISCLAIMER)
    print(f"episodes={len(episodes)} public_rows={len(public_rows)}")
    for path in (report_path, decisions_path, evaluator_path):
        print(path)


if __name__ == "__main__":
    main()
