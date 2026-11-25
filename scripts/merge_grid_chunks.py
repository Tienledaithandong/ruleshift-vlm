"""Merge chunked real-VLM decisions with the exported public baselines.

The merge is intentionally strict: it refuses to silently drop invalid JSON,
parser fallbacks, duplicate scenarios, incomplete chunks, or a baseline that
does not cover every scenario. The scenario is the only resampling unit in the
paired intervals; seeds and frames are never resampled.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ruleshift.envs.grid.config import load_grid_config
from ruleshift.evaluator.grid_metrics import paired_grid_ci, summarize_grid
from ruleshift.evaluator.grid_runner import GridEpisode, build_grid_scenarios

FULL_GRID_SCENARIOS = 100 * 5 * 11


class MergeError(ValueError):
    """Raised when chunk or baseline artifacts cannot be merged safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise MergeError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _chunk_paths(chunks_dir: Path) -> dict[str, tuple[Path, Path]]:
    if not chunks_dir.is_dir():
        raise MergeError(f"chunk directory does not exist: {chunks_dir}")
    chunks: dict[str, tuple[Path, Path]] = {}
    for decisions in sorted(chunks_dir.glob("grid_real_decisions.*.jsonl")):
        chunk_id = decisions.name[len("grid_real_decisions.") : -len(".jsonl")]
        evaluator = chunks_dir / f"grid_real_evaluator_only.{chunk_id}.jsonl"
        if not evaluator.is_file():
            raise MergeError(f"chunk {chunk_id!r} is missing {evaluator.name}")
        chunks[chunk_id] = (decisions, evaluator)
    if not chunks:
        raise MergeError(f"no chunk decision files found in {chunks_dir}")
    return chunks


def _as_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _chunk_episodes(
    chunk_id: str,
    decisions_path: Path,
    evaluator_path: Path,
) -> list[GridEpisode]:
    decision_rows = _read_jsonl(decisions_path)
    evaluator_rows = _read_jsonl(evaluator_path)
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            raise MergeError(
                f"chunk {chunk_id!r} has a public row without scenario_id; "
                "rerun the chunk with a chunk-aware pilot"
            )
        by_scenario[scenario_id].append(row)
    evaluator_by_scenario: dict[str, dict[str, Any]] = {}
    for row in evaluator_rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            raise MergeError(f"chunk {chunk_id!r} has an evaluator row without scenario_id")
        if scenario_id in evaluator_by_scenario:
            raise MergeError(f"chunk {chunk_id!r} repeats evaluator scenario {scenario_id!r}")
        evaluator_by_scenario[scenario_id] = row
    if not by_scenario:
        raise MergeError(f"chunk {chunk_id!r} contains no public decisions")
    if set(by_scenario) != set(evaluator_by_scenario):
        missing = sorted(set(by_scenario) - set(evaluator_by_scenario))
        extra = sorted(set(evaluator_by_scenario) - set(by_scenario))
        raise MergeError(
            f"chunk {chunk_id!r} scenario mismatch: missing evaluator={missing[:3]}, "
            f"extra evaluator={extra[:3]}"
        )

    episodes: list[GridEpisode] = []
    for scenario_id, rows in by_scenario.items():
        invalid = [row for row in rows if row.get("json_valid") is not True]
        if invalid:
            raise MergeError(
                f"chunk {chunk_id!r} scenario {scenario_id!r} has "
                f"{len(invalid)} non-full-menu decision(s); refusing to merge"
            )
        fallback = [row for row in rows if row.get("fallback") is True]
        if fallback:
            raise MergeError(
                f"chunk {chunk_id!r} scenario {scenario_id!r} used parser fallback; "
                "refusing to merge"
            )
        evaluator = evaluator_by_scenario[scenario_id]
        final_rule = str(evaluator.get("final_rule_estimate") or "base")
        episodes.append(
            GridEpisode(
                scenario_id=scenario_id,
                run_seed=_as_int(evaluator.get("run_seed")),
                method="grid_vlm",
                success=bool(evaluator.get("success", False)),
                total_return=float(evaluator.get("total_return", 0.0)),
                env_steps=_as_int(evaluator.get("env_steps")),
                tool_units=_as_int(evaluator.get("tool_units")),
                advisor_calls=_as_int(evaluator.get("advisor_calls")),
                total_cost=float(
                    evaluator.get(
                        "total_cost",
                        _as_int(evaluator.get("env_steps"))
                        + 2 * _as_int(evaluator.get("advisor_calls")),
                    )
                ),
                adaptation_steps=evaluator.get("adaptation_steps"),
                false_alarm=bool(evaluator.get("false_alarm", False)),
                final_rule_estimate=final_rule,
                public_rows=tuple(rows),
                evaluator_rows=(evaluator,),
                scenario_type=str(evaluator.get("scenario_type", "change_detection")),
                shift_detected=bool(
                    evaluator.get("shift_detected", final_rule != "base")
                ),
                oracle_return=evaluator.get("oracle_return"),
                json_valid=True,
                valid_action=all(row.get("valid_action") is True for row in rows),
                output_tokens=sum(_as_int(row.get("output_tokens")) for row in rows),
                cache_hit=any(row.get("cache_hit") is True for row in rows),
                tool_calls_made=sum(
                    1
                    for row in rows
                    if str(row.get("chosen_action", "")).upper()
                    in {"INSPECT", "READ_SIGN"}
                ),
            )
        )
    return episodes


def _baseline_episodes(path: Path) -> dict[str, list[GridEpisode]]:
    grouped: dict[str, list[GridEpisode]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in _read_jsonl(path):
        method = str(row.get("method") or "")
        scenario_id = str(row.get("scenario_id") or "")
        if not method or not scenario_id:
            raise MergeError(f"{path}: baseline row is missing method or scenario_id")
        key = (method, scenario_id)
        if key in seen:
            raise MergeError(f"{path}: duplicate baseline {method!r}/{scenario_id!r}")
        seen.add(key)
        final_rule = str(row.get("final_rule_estimate") or "base")
        grouped[method].append(
            GridEpisode(
                scenario_id=scenario_id,
                run_seed=_as_int(row.get("run_seed")),
                method=method,
                success=bool(row.get("success", False)),
                total_return=float(row.get("total_return", 0.0)),
                env_steps=_as_int(row.get("env_steps")),
                tool_units=_as_int(row.get("tool_units")),
                advisor_calls=_as_int(row.get("advisor_calls")),
                total_cost=float(row.get("total_cost", 0.0)),
                adaptation_steps=row.get("adaptation_steps"),
                false_alarm=bool(row.get("false_alarm", False)),
                final_rule_estimate=final_rule,
                public_rows=(),
                evaluator_rows=(),
                scenario_type=str(row.get("scenario_type", "change_detection")),
                shift_detected=bool(row.get("shift_detected", final_rule != "base")),
                oracle_return=row.get("oracle_return"),
            )
        )
    if not grouped:
        raise MergeError(f"{path}: no baseline rows found")
    return dict(grouped)


def merge_grid_chunks(
    chunks_dir: str | Path,
    baseline_path: str | Path,
    *,
    expected_scenarios: int = FULL_GRID_SCENARIOS,
    expected_ids: set[str] | None = None,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Load all chunks, enforce coverage, and compute paired scenario CIs."""

    chunks_dir = Path(chunks_dir)
    baseline_path = Path(baseline_path)
    chunks = _chunk_paths(chunks_dir)
    episodes: list[GridEpisode] = []
    seen: set[str] = set()
    for chunk_id, (decisions, evaluator) in chunks.items():
        chunk_episodes = _chunk_episodes(chunk_id, decisions, evaluator)
        for episode in chunk_episodes:
            if episode.scenario_id in seen:
                raise MergeError(f"scenario appears in more than one chunk: {episode.scenario_id}")
            seen.add(episode.scenario_id)
        episodes.extend(chunk_episodes)
    if len(episodes) != expected_scenarios:
        raise MergeError(
            f"expected {expected_scenarios} scenarios, found {len(episodes)} "
            f"across {len(chunks)} chunk(s)"
        )
    if expected_ids is not None and seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise MergeError(
            "chunk scenario ids do not match the frozen grid: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    baselines = _baseline_episodes(baseline_path)
    expected_ids = {episode.scenario_id for episode in episodes}
    for method, method_episodes in baselines.items():
        ids = {episode.scenario_id for episode in method_episodes}
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise MergeError(
                f"baseline {method!r} does not cover the same scenarios: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    summaries = {"grid_vlm": summarize_grid(tuple(episodes))}
    paired: dict[str, dict[str, Any]] = {}
    for method, method_episodes in sorted(baselines.items()):
        summaries[method] = summarize_grid(tuple(method_episodes))
        paired[method] = asdict(
            paired_grid_ci(tuple(episodes), tuple(method_episodes), n_boot=n_boot)
        )
    return {
        "scenarios": len(episodes),
        "chunks": sorted(chunks),
        "methods": summaries,
        "paired_grid_vlm_minus_baseline": paired,
        "resampling_unit": "scenario",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="results/grid_chunks")
    parser.add_argument("--baselines", default="results/grid_baselines/baselines.jsonl")
    parser.add_argument("--out", default="results/grid_merged/summary.json")
    parser.add_argument("--config", default="configs/grid_dev.yaml")
    parser.add_argument("--expected-scenarios", type=int, default=FULL_GRID_SCENARIOS)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_grid_config(Path(args.config))
    expected_ids = {
        scenario.scenario_id for scenario in build_grid_scenarios(100, 5, config)
    }
    payload = merge_grid_chunks(
        args.chunks,
        args.baselines,
        expected_scenarios=args.expected_scenarios,
        expected_ids=expected_ids,
        n_boot=args.n_boot,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
