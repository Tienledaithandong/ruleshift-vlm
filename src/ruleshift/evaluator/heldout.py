"""Week 9 held-out evaluation scaffolding (SYNTHETIC while using the stub).

The scenario values here are deliberately disjoint from the development grid.
Nothing is frozen: `configs/preregistered_test.yaml` stays DRAFT_NOT_FROZEN
until a real frozen-VLM validation has been run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ruleshift.envs.config import EnvConfig
from ruleshift.evaluator.closed_loop import (
    ClosedLoopOutput,
    build_family_a_scenarios,
    run_closed_loop,
)
from ruleshift.evaluator.metrics_ci import PairedSummary

#: Method ids for the proposed method, in preference order. The ``_stub``
#: suffix only reflects which backend was injected, so a real-backend run uses
#: ``m1`` and the same code path still finds the proposed method.
PROPOSED_LABELS = ("m1", "m1_stub")
HELD_OUT_DETOUR = (1, 3)
HELD_OUT_NOISE = (0.05, 0.15)
HELD_OUT_LOCKOUT = (2, 5)
HELD_OUT_HAZARD = (0.02, 0.07)


def build_heldout_family_a() -> tuple[tuple[str, EnvConfig], ...]:
    """Build held-out scenarios with parameter values unseen in development."""

    scenarios: list[tuple[str, EnvConfig]] = []
    for detour in HELD_OUT_DETOUR:
        for noise in HELD_OUT_NOISE:
            for lockout in HELD_OUT_LOCKOUT:
                for index, hazard in enumerate(HELD_OUT_HAZARD):
                    scenario_id = (
                        f"heldA-d{detour}-n{noise}-k{lockout}-h{index}"
                    )
                    scenarios.append(
                        (
                            scenario_id,
                            EnvConfig(
                                probe_detour_steps=detour,
                                feedback_noise=noise,
                                lockout_steps=lockout,
                                shift_hazard=hazard,
                                disable_shift=False,
                                initial_mechanic="red",
                                scenario_id=scenario_id,
                            ),
                        )
                    )
    return tuple(scenarios)


@dataclass
class HeldOutResult:
    closed_loop: ClosedLoopOutput
    method_stats: dict[str, dict[str, float | int]]
    summaries: dict[tuple[str, str], PairedSummary]
    max_vlm_calls_per_episode: int = 6

    @property
    def episodes(self):
        return self.closed_loop.episodes


def proposed_label(methods: tuple[str, ...]) -> str | None:
    """Return the proposed-method id actually present in *methods*."""

    for candidate in PROPOSED_LABELS:
        if candidate in methods:
            return candidate
    return None


def run_heldout_evaluation(
    scenarios: tuple[tuple[str, EnvConfig], ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...] = (101,),
    backend: object | None = None,
    cache: object | None = None,
    max_vlm_calls_per_episode: int = 6,
) -> HeldOutResult:
    """Run paired held-out episodes and accounting for every method."""

    proposed = proposed_label(tuple(methods))
    comparisons: tuple[tuple[str, str], ...] = (
        tuple((proposed, method) for method in methods if method != proposed)
        if proposed is not None
        else ()
    )
    result = run_closed_loop(
        scenarios,
        methods,
        seeds,
        comparisons,
        backend=backend,
        cache=cache,
        max_vlm_calls_per_episode=max_vlm_calls_per_episode,
    )

    stats: dict[str, dict[str, float | int]] = {}
    for method in methods:
        episodes = [
            episode
            for (_, _, name), episode in result.episodes.items()
            if name == method
        ]
        decisions = [row for episode in episodes for row in episode.public_rows]
        total_return = sum(episode.total_return for episode in episodes)
        calls = sum(episode.vlm_calls for episode in episodes)
        tokens = sum(
            episode.input_tokens + episode.output_tokens for episode in episodes
        )
        stats[method] = {
            "episodes": len(episodes),
            "total_return": total_return,
            "mean_return": total_return / len(episodes),
            "steps": sum(episode.steps for episode in episodes),
            "probes": sum(episode.probes for episode in episodes),
            "vlm_calls": calls,
            "json_valid": sum(row["vlm_calls_used"] > 0 for row in decisions),
            "invalid_actions": sum(episode.invalid_actions for episode in episodes),
            "tokens": tokens,
            "latency_ms": sum(episode.latency_ms for episode in episodes),
            "call_cap_hits": sum(episode.call_cap_hits for episode in episodes),
        }
    return HeldOutResult(
        result, stats, result.summaries, max_vlm_calls_per_episode
    )


def write_result_files(
    result: HeldOutResult, results_dir: Path, report_dir: Path
) -> dict[str, Path]:
    """Write per-episode CSV, paired JSON and the Week 9 report."""

    results_dir = Path(results_dir)
    report_dir = Path(report_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "main_family_A.csv"
    json_path = results_dir / "main_paired.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario_id",
                "seed",
                "method",
                "return",
                "steps",
                "probes",
                "vlm_calls",
                "input_tokens",
                "output_tokens",
                "invalid_actions",
            ]
        )
        for (scenario_id, seed, method), episode in sorted(result.episodes.items()):
            writer.writerow(
                [
                    scenario_id,
                    seed,
                    method,
                    f"{episode.total_return:.6f}",
                    episode.steps,
                    episode.probes,
                    episode.vlm_calls,
                    episode.input_tokens,
                    episode.output_tokens,
                    episode.invalid_actions,
                ]
            )

    summary_rows = {
        f"{method}_minus_{baseline}": {
            "mean_difference": summary.mean_difference,
            "ci95_low": summary.low,
            "ci95_high": summary.high,
            "n_scenarios": summary.n_scenarios,
        }
        for (method, baseline), summary in result.summaries.items()
    }
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    lines = [
        "# Week 9 — Main held-out evaluation (SYNTHETIC)",
        "",
        "**SYNTHETIC: every model-side method uses the deterministic stub. "
        "No VLM was run.** These numbers exercise the held-out evaluation "
        "pipeline and cost accounting. They are not evidence about any model "
        "and must not be reported as results.",
        "",
        "Held-out scenario values are disjoint from the development grid; "
        "seeds are averaged within each scenario and the resampling unit is the "
        "scenario, never the frame or the seed. Every method is held to the same "
        "per-episode VLM call cap "
        f"({result.max_vlm_calls_per_episode}). "
        "`configs/preregistered_test.yaml` remains DRAFT_NOT_FROZEN because no "
        "real validation has been completed.",
        "",
        "## Paired differences (proposed minus baseline)",
        "",
        "| Comparison | Scenarios | Mean difference | 95% paired CI |",
        "|---|---:|---:|---|",
    ]
    for (method, baseline), summary in result.summaries.items():
        lines.append(
            f"| {method} − {baseline} | {summary.n_scenarios} | "
            f"{summary.mean_difference:+.3f} | [{summary.low:+.3f}, {summary.high:+.3f}] |"
        )

    lines += [
        "",
        "## Cost and call accounting",
        "",
        "| Method | Episodes | Mean return | Probes | VLM calls | Tokens "
        "| Invalid actions | Cap fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in result.method_stats.items():
        lines.append(
            f"| {method} | {stats['episodes']} | {stats['mean_return']:+.3f} | "
            f"{stats['probes']} | {stats['vlm_calls']} | {stats['tokens']} | "
            f"{stats['invalid_actions']} | {stats['call_cap_hits']} |"
        )

    negative = [
        key for key, summary in result.summaries.items() if summary.high <= 0.0
    ]
    lines += [
        "",
        "## Negative and inconclusive results",
        "",
        "All comparisons whose paired interval does not support the proposed "
        "method are reported here rather than omitted:",
        "",
    ]
    if negative:
        for method, baseline in negative:
            lines.append(f"- {method} − {baseline}: interval entirely at or below zero.")
    else:
        lines.append(
            "- None of the synthetic comparisons is entirely at or below zero; "
            "this is a property of the stub, not evidence of a real effect."
        )
    lines += [
        "",
        "A real held-out conclusion requires a frozen test configuration and a "
        "frozen-VLM checkpoint; both are still missing.",
        "",
    ]
    report_path = report_dir / "w9_main_results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "csv": csv_path,
        "paired_json": json_path,
        "report": report_path,
    }
