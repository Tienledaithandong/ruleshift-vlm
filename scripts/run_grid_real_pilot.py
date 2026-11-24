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
        "--max-episodes", type=int, default=None,
        help="optional cap on episodes actually executed; default is no cap",
    )
    parser.add_argument("--map-offset", type=int, default=0)
    parser.add_argument(
        "--map-count", type=int, default=0,
        help="number of maps to run; 0 means every map from map-offset onward",
    )
    parser.add_argument("--chunk-id", default="default")
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


def behavioral_gate(public_rows: list[dict], episode_count: int) -> dict:
    """Return the P0 smoke gate; a constant-action policy must not scale."""

    actions = [str(row.get("chosen_action", "")) for row in public_rows]
    positions = [tuple(row.get("agent_position") or ()) for row in public_rows]
    steps = [int(row.get("env_steps", 0) or 0) for row in public_rows]
    moved = 0
    for index in range(1, len(positions)):
        if steps[index] > steps[index - 1] and positions[index] != positions[index - 1]:
            moved += 1
    full_menu = all(bool(row.get("json_valid", False)) for row in public_rows)
    fallback_count = sum(bool(row.get("fallback", False)) for row in public_rows)
    distinct_actions = sorted(set(action for action in actions if action))
    gate = {
        "episodes": int(episode_count),
        "full_menu_rate": _mean([float(row.get("json_valid", False)) for row in public_rows]),
        "fallback_count": int(fallback_count),
        "distinct_actions": distinct_actions,
        "moved_steps": int(moved),
    }
    gate["passed"] = bool(
        episode_count >= 2
        and bool(public_rows)
        and full_menu
        and fallback_count == 0
        and len(distinct_actions) >= 2
        and moved > 0
    )
    return gate


def select_chunk_scenarios(
    scenarios: tuple, map_offset: int, map_count: int
) -> tuple:
    """Select a contiguous map range without changing scenario order."""

    if map_offset < 0 or map_count < 0:
        raise ValueError("map_offset and map_count must be non-negative")
    if map_count == 0:
        return tuple(
            scenario for scenario in scenarios if int(scenario.map_seed) >= map_offset
        )
    end = map_offset + map_count
    return tuple(
        scenario
        for scenario in scenarios
        if map_offset <= int(scenario.map_seed) < end
    )


def chunk_artifact_paths(out_dir: Path, chunk_id: str) -> dict[str, Path]:
    """Return chunk-stamped decision/evaluator/cache paths."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", chunk_id):
        raise ValueError(f"invalid chunk_id: {chunk_id!r}")
    return {
        "decisions": out_dir / f"grid_real_decisions.{chunk_id}.jsonl",
        "evaluator": out_dir / f"grid_real_evaluator_only.{chunk_id}.jsonl",
        "cache": out_dir / f"response_cache.{chunk_id}.jsonl",
    }


def chunk_report_path(report_path: Path, chunk_id: str) -> Path:
    """Return a chunk-stamped report path beside the requested report."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", chunk_id):
        raise ValueError(f"invalid chunk_id: {chunk_id!r}")
    return report_path.with_name(
        f"{report_path.stem}.{chunk_id}{report_path.suffix}"
    )


def _markdown(payload: dict) -> str:
    metrics = payload["metrics"]
    gate = payload["gate"]
    gate_label = "PASS" if gate["passed"] else "FAIL"
    gate_actions = ", ".join(gate["distinct_actions"]) or "(none)"
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
        f"- Chunk: `{payload.get('chunk_id', 'default')}`; "
        f"map range: [{payload.get('map_offset', 0)}, "
        f"{payload.get('map_offset', 0) + payload.get('map_count', 0)})",
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
        f"| Fallback decision rate | {metrics.get('fallback_rate', 0.0):.4f} |",
        f"| Mean output tokens per decision | {metrics['mean_output_tokens']:.2f} |",
        f"| Total output tokens | {metrics['total_output_tokens']} |",
        f"| Cache hit rate | {metrics['cache_hit_rate']:.4f} |",
        f"| Cache hits | {metrics.get('cache_hits', 0)} |",
        f"| Model calls (new responses) | {metrics['model_calls']} |",
        "",
        "The v10 lesson is honoured here: `max_new_tokens=2048` is never "
        "shrunk, because a truncated JSON response is counted as invalid.",
        "",
        "## Behavioural gate (P0)",
        "",
        "A smoke run may inform a decision only if it exercises more than one "
        "action and contains no parser fallback. Passing this gate is still "
        "not a scientific finding; failing it blocks any larger run.",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Full-menu JSON valid | {gate['full_menu_rate']:.4f} |",
        f"| Parser fallbacks | {gate['fallback_count']} |",
        f"| Distinct actions | {gate_actions} |",
        f"| Moving steps | {gate['moved_steps']} |",
        f"| Verdict | **{gate_label}** |",
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
    artifact_paths = chunk_artifact_paths(out_dir, args.chunk_id)
    cache = ResponseCache(artifact_paths["cache"])
    policy = GridVLMPolicy(backend, policy_config, cache)

    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    grid_config = load_grid_config(config_path)
    scenarios = select_chunk_scenarios(
        build_grid_scenarios(args.maps, args.seeds, grid_config),
        args.map_offset,
        args.map_count,
    )
    if args.max_episodes is not None:
        scenarios = scenarios[: args.max_episodes]

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

    # The scenario id is evaluation metadata, not a model input. It is added
    # only after the episode so a public decision file can be merged by chunk.
    public_rows = [
        dict(row, scenario_id=episode.scenario_id)
        for episode in episodes
        for row in episode.public_rows
    ]
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

    artifact_paths["decisions"].write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in public_rows),
        encoding="utf-8",
    )
    artifact_paths["evaluator"].write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in evaluator_rows),
        encoding="utf-8",
    )

    gate = behavioral_gate(public_rows, len(episodes))
    metrics = {
        "json_valid_rate": _mean([float(row.get("json_valid", False)) for row in public_rows]),
        "valid_action_rate": _mean([float(row.get("valid_action", False)) for row in public_rows]),
        "fallback_rate": _mean([float(row.get("fallback", False)) for row in public_rows]),
        "mean_output_tokens": _mean([float(row.get("output_tokens", 0)) for row in public_rows]),
        "total_output_tokens": int(sum(int(row.get("output_tokens", 0)) for row in public_rows)),
        "cache_hit_rate": _mean([float(row.get("cache_hit", False)) for row in public_rows]),
        "cache_hits": int(policy.total_cache_hits),
        "model_calls": int(policy.total_model_calls),
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
        "max_episodes": args.max_episodes,
        "map_offset": int(args.map_offset),
        "map_count": int(args.map_count),
        "chunk_id": str(args.chunk_id),
        "scenarios": len(scenarios),
        "episodes": len(episodes),
        "public_rows": len(public_rows),
        "prereg_status": prereg_status,
        "metrics": metrics,
        "gate": gate,
    }

    report_path = chunk_report_path(ROOT / args.report, args.chunk_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown(payload), encoding="utf-8")

    print(DISCLAIMER)
    print(f"episodes={len(episodes)} public_rows={len(public_rows)}")
    print(f"behavioral_gate={gate['passed']} details={gate}")
    for path in (report_path, artifact_paths["decisions"], artifact_paths["evaluator"]):
        print(path)


if __name__ == "__main__":
    main()
