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


def _episode_groups(public_rows: list[dict]) -> list[list[dict]]:
    """Group consecutive public rows by scenario without reordering them."""

    groups: list[list[dict]] = []
    current: list[dict] = []
    current_id: object = object()
    for row in public_rows:
        scenario_id = row.get("scenario_id")
        if current and scenario_id != current_id:
            groups.append(current)
            current = []
        current.append(row)
        current_id = scenario_id
    if current:
        groups.append(current)
    return groups


def _belief_vector(row: dict) -> tuple[tuple[str, float], ...] | None:
    probs = row.get("hypothesis_probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    try:
        return tuple((str(key), round(float(value), 6)) for key, value in sorted(probs.items()))
    except (TypeError, ValueError):
        return None


def _last_public_outcome(row: dict) -> str | None:
    """Return the outcome of the most recent public step, if recorded."""

    history = str(row.get("public_history", "") or "")
    segments = [segment for segment in history.split(";") if segment.strip()]
    if not segments:
        return None
    match = re.search(r"Step\s+\d+:\s*([A-Z_]+).*?->\s*(\S+)", segments[-1])
    return match.group(2).lower() if match else None

def semantic_gate(
    public_rows: list[dict],
    episodes: list | tuple = (),
    evaluator_rows: list[dict] | tuple = (),
) -> dict:
    """Reject a schema-valid but behaviourally degenerate policy.

    This gate is deliberately separate from ``behavioral_gate``: the latter
    only checks serialization, fallbacks and action diversity. A policy can
    pass that gate while ignoring every observation, so this gate checks
    belief updates, blocked-action repetition, tool use and base-rule success.
    """

    groups = _episode_groups(public_rows)
    beliefs = [_belief_vector(row) for row in public_rows]
    belief_rows = sum(vector is not None for vector in beliefs)
    unique_beliefs = len({vector for vector in beliefs if vector is not None})

    moves = {"UP", "DOWN", "LEFT", "RIGHT"}
    blocked_total = 0
    blocked_repeats = 0
    for group in groups:
        for current, nxt in zip(group, group[1:]):
            action = str(current.get("chosen_action", ""))
            if action not in moves:
                continue
            try:
                advanced = int(nxt.get("env_steps", 0)) > int(current.get("env_steps", 0))
            except (TypeError, ValueError):
                continue
            if advanced and _last_public_outcome(nxt) == "blocked":
                blocked_total += 1
                if str(nxt.get("chosen_action", "")) == action:
                    blocked_repeats += 1
    blocked_repeat_rate = (
        blocked_repeats / blocked_total if blocked_total else 0.0
    )

    tool_available_rows = sum(
        any(
            action in ("INSPECT", "READ_SIGN")
            for action in (row.get("action_mask") or ())
        )
        for row in public_rows
    )
    tool_calls = sum(
        str(row.get("chosen_action", "")) in ("INSPECT", "READ_SIGN")
        for row in public_rows
    )
    guard_count = sum(bool(row.get("guard_applied", False)) for row in public_rows)

    truth = {
        str(row.get("scenario_id")): str(row.get("true_rule"))
        for row in evaluator_rows
        if row.get("scenario_id") is not None and row.get("true_rule") is not None
    }
    base_ids = {scenario_id for scenario_id, rule in truth.items() if rule == "base"}
    base_success = sum(
        1
        for episode in episodes
        if str(getattr(episode, "scenario_id", "")) in base_ids
        and bool(getattr(episode, "success", False))
    )
    base_episodes = sum(
        1 for episode in episodes
        if str(getattr(episode, "scenario_id", "")) in base_ids
    )

    prefixes = [
        tuple(str(row.get("chosen_action", "")) for row in group[:3])
        for group in groups
    ]
    unique_prefixes = len(set(prefixes))
    final_estimates = {
        str(row.get("final_rule_estimate"))
        for row in evaluator_rows
        if row.get("final_rule_estimate") is not None
    }

    reasons: list[str] = []
    if len(groups) < 2:
        reasons.append("fewer than two episodes")
    if belief_rows != len(public_rows) or not public_rows:
        reasons.append("missing hypothesis_probabilities on one or more rows")
    if unique_beliefs <= 1:
        reasons.append("belief vector never changed")
    if blocked_total and blocked_repeat_rate > 0.05:
        reasons.append("blocked action repeated more than 5% of the time")
    if tool_available_rows and tool_calls == 0:
        reasons.append("no INSPECT/READ_SIGN decision when a tool was available")
    if base_episodes and base_success == 0:
        reasons.append("zero success on base-rule episodes")

    return {
        "episodes": len(groups),
        "belief_rows": int(belief_rows),
        "unique_belief_vectors": int(unique_beliefs),
        "blocked_transitions": int(blocked_total),
        "blocked_repeats": int(blocked_repeats),
        "blocked_repeat_rate": float(blocked_repeat_rate),
        "tool_available_rows": int(tool_available_rows),
        "tool_calls": int(tool_calls),
        "guard_count": int(guard_count),
        "base_episodes": int(base_episodes),
        "base_success": int(base_success),
        "unique_final_estimates": int(len(final_estimates)),
        "unique_first_three_prefixes": int(unique_prefixes),
        "reasons": reasons,
        "passed": not reasons,
    }


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
    technical_label = (
        "PASS" if gate.get("technical_passed", gate["passed"]) else "FAIL"
    )
    combined_label = "PASS" if gate["passed"] else "FAIL"
    gate_actions = ", ".join(gate["distinct_actions"]) or "(none)"
    semantic = gate.get("semantic") or {}
    semantic_label = "PASS" if semantic.get("passed") else "FAIL"
    semantic_reasons = "; ".join(semantic.get("reasons", ())) or "(none)"
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
        f"| Technical verdict | **{technical_label}** |",
        f"| Combined with semantic gate | **{combined_label}** |",
        "",
        "## Semantic gate (P0)",
        "",
        "The technical gate above can pass on a fixed action script. These",
        "checks require the policy to use public evidence and to stop repeating",
        "an action that the environment already reported as blocked.",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| Distinct belief vectors | {semantic.get('unique_belief_vectors', 0)} |",
        f"| Belief rows / decision rows | {semantic.get('belief_rows', 0)} / {payload['public_rows']} |",
        f"| Blocked-action repeats / transitions | {semantic.get('blocked_repeats', 0)} / {semantic.get('blocked_transitions', 0)} |",
        f"| Blocked-action repeat rate | {semantic.get('blocked_repeat_rate', 0.0):.4f} |",
        f"| Tool decisions (INSPECT/READ_SIGN) | {semantic.get('tool_calls', 0)} |",
        f"| Guard activations | {semantic.get('guard_count', 0)} |",
        f"| Base-rule success | {semantic.get('base_success', 0)} / {semantic.get('base_episodes', 0)} |",
        f"| Unique first-three prefixes | {semantic.get('unique_first_three_prefixes', 0)} |",
        f"| Verdict | **{semantic_label}** |",
        f"| Reasons | {semantic_reasons} |",
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
        episode = run_grid_episode(env, policy, grid_config, scenario, observation, info)
        episodes.append(episode)
        last = episode.public_rows[-1] if episode.public_rows else {}
        print(
            f"episode {len(episodes)}/{len(scenarios)} {scenario.scenario_id} "
            f"steps={len(episode.public_rows)} json_valid={last.get('json_valid')} "
            f"fallback={last.get('fallback')}",
            flush=True,
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

    technical_gate = behavioral_gate(public_rows, len(episodes))
    semantic = semantic_gate(public_rows, episodes, evaluator_rows)
    gate = dict(technical_gate)
    gate["semantic"] = semantic
    gate["technical_passed"] = bool(technical_gate["passed"])
    gate["passed"] = bool(technical_gate["passed"] and semantic["passed"])
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
    print(f"behavioral_gate={technical_gate['passed']} details={technical_gate}")
    print(f"semantic_gate={semantic['passed']} details={semantic}")
    print(f"combined_gate={gate['passed']}")
    for path in (report_path, artifact_paths["decisions"], artifact_paths["evaluator"]):
        print(path)


if __name__ == "__main__":
    main()
