"""CPU-only base-preservation stress study for RuleShift-Grid.

The study is deliberately narrower than the full benchmark: every episode uses
the true ``base`` rule, with no shift.  It characterises two related outcomes:

* **task failure** -- the episode did not succeed (the original base gate);
* **misdiagnosis** -- the policy committed to a non-base rule.

Both are reported with episode-level Wilson intervals and a map-cluster
bootstrap interval.  The cluster interval resamples whole maps, so it does not
pretend that five seeds from one layout are five independent layouts.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ruleshift.envs.grid.config import GridConfig
from ruleshift.envs.grid.entities import GridScenario
from ruleshift.evaluator.grid_runner import build_grid_scenarios

__all__ = [
    "analyze_base_stress",
    "build_base_stress_scenarios",
    "map_cluster_bootstrap",
    "wilson_interval",
    "write_base_stress_report",
]


def build_base_stress_scenarios(
    n_maps: int, seeds: int, config: GridConfig
) -> tuple[GridScenario, ...]:
    """Return only the ``base`` identification scenarios from the grid builder."""

    if n_maps < 1 or seeds < 1:
        raise ValueError("n_maps and seeds must be positive")
    scenarios = tuple(
        scenario
        for scenario in build_grid_scenarios(n_maps, seeds, config)
        if scenario.scenario_type == "identification"
        and scenario.initial_rule == "base"
        and scenario.shift_rule is None
    )
    expected = int(n_maps) * int(seeds)
    if len(scenarios) != expected:
        raise ValueError(
            f"expected {expected} base scenarios, found {len(scenarios)}"
        )
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("base-stress scenarios must have unique ids")
    return scenarios


def wilson_interval(
    successes: int, n: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no SciPy dependency)."""

    if n < 1:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    p = successes / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == n else min(1.0, center + margin)
    return low, high


def _map_id(row: Mapping[str, Any]) -> str:
    return str(row["scenario_id"]).split("-")[0]


def map_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    event: Callable[[Mapping[str, Any]], bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval from resampling whole map clusters."""

    if not rows:
        raise ValueError("at least one row is required")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[_map_id(row)].append(1.0 if event(row) else 0.0)
    ids = sorted(groups)
    if len(ids) == 1:
        value = sum(groups[ids[0]]) / len(groups[ids[0]])
        return value, value
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(n_boot):
        sampled = [groups[ids[rng.randrange(len(ids))]] for _ in ids]
        values = [value for group in sampled for value in group]
        rates.append(sum(values) / len(values))
    rates.sort()
    low_index = max(0, int((alpha / 2.0) * n_boot))
    high_index = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot) - 1)
    return rates[low_index], rates[high_index]


def _rate_summary(
    rows: Sequence[Mapping[str, Any]],
    event: Callable[[Mapping[str, Any]], bool],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("cannot summarize an empty row set")
    events = sum(1 for row in rows if event(row))
    rate = events / n
    low, high = wilson_interval(events, n)
    cluster_low, cluster_high = map_cluster_bootstrap(
        rows, event, n_boot=n_boot, seed=seed
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_map_id(row)].append(1.0 if event(row) else 0.0)
    map_rates = {sum(values) / len(values) for values in grouped.values()}
    return {
        "episodes": n,
        "events": events,
        "rate": rate,
        "wilson_95": [low, high],
        "map_cluster_95": [cluster_low, cluster_high],
        "map_cluster_degenerate": len(map_rates) == 1,
    }


def _mcnemar_exact(only_method: int, only_baseline: int) -> float:
    n = only_method + only_baseline
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_method, only_baseline) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        scenario_id = str(row["scenario_id"])
        if scenario_id in indexed:
            raise ValueError(f"duplicate scenario row: {scenario_id}")
        indexed[scenario_id] = row
    return indexed


def _failure_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": str(row["scenario_id"]),
        "success": bool(row["success"]),
        "final_rule_estimate": row.get("final_rule_estimate"),
        "env_steps": int(row.get("env_steps", 0)),
        "tool_units": int(row.get("tool_units", 0)),
        "total_return": float(row.get("total_return", 0.0)),
    }


def analyze_base_stress(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    n_maps: int,
    seeds: int,
    n_boot: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Analyze base-only episodes without evaluating or changing any gate."""

    diag = _index(diagnostic_rows)
    baseline = _index(baseline_rows)
    if set(diag) != set(baseline):
        missing_diag = sorted(set(baseline) - set(diag))
        missing_base = sorted(set(diag) - set(baseline))
        raise ValueError(
            "diagnostic and baseline scenario ids differ: "
            f"missing diagnostic={missing_diag[:3]}, "
            f"missing baseline={missing_base[:3]}"
        )
    expected = int(n_maps) * int(seeds)
    if len(diag) != expected:
        raise ValueError(f"expected {expected} paired episodes, found {len(diag)}")

    diag_rows = [diag[key] for key in sorted(diag)]
    base_rows = [baseline[key] for key in sorted(diag)]

    def misdiagnosed(row: Mapping[str, Any]) -> bool:
        return row.get("final_rule_estimate") != "base"

    def failed(row: Mapping[str, Any]) -> bool:
        return not bool(row["success"])

    methods = {
        "diagnostic_belief": {
            "task_success": _rate_summary(
                diag_rows, lambda row: bool(row["success"]), n_boot=n_boot, seed=bootstrap_seed
            ),
            "task_failure": _rate_summary(
                diag_rows, failed, n_boot=n_boot, seed=bootstrap_seed
            ),
            "misdiagnosis": _rate_summary(
                diag_rows, misdiagnosed, n_boot=n_boot, seed=bootstrap_seed
            ),
            "env_steps_total": sum(int(row.get("env_steps", 0)) for row in diag_rows),
            "tool_units_total": sum(int(row.get("tool_units", 0)) for row in diag_rows),
            "final_estimate_counts": dict(
                sorted(Counter(str(row.get("final_rule_estimate")) for row in diag_rows).items())
            ),
        },
        "no_adaptation": {
            "task_success": _rate_summary(
                base_rows, lambda row: bool(row["success"]), n_boot=n_boot, seed=bootstrap_seed
            ),
            "task_failure": _rate_summary(
                base_rows, failed, n_boot=n_boot, seed=bootstrap_seed
            ),
            "misdiagnosis": _rate_summary(
                base_rows, misdiagnosed, n_boot=n_boot, seed=bootstrap_seed
            ),
            "env_steps_total": sum(int(row.get("env_steps", 0)) for row in base_rows),
            "tool_units_total": sum(int(row.get("tool_units", 0)) for row in base_rows),
            "final_estimate_counts": dict(
                sorted(Counter(str(row.get("final_rule_estimate")) for row in base_rows).items())
            ),
        },
    }

    ids = sorted(diag)
    both = sum(bool(diag[key]["success"]) and bool(baseline[key]["success"]) for key in ids)
    only_diag = sum(
        bool(diag[key]["success"]) and not bool(baseline[key]["success"]) for key in ids
    )
    only_base = sum(
        not bool(diag[key]["success"]) and bool(baseline[key]["success"]) for key in ids
    )
    neither = len(ids) - both - only_diag - only_base

    by_map: dict[str, dict[str, int]] = {}
    for map_id in sorted({_map_id(row) for row in diag_rows}, key=lambda x: int(x[1:])):
        diag_map = [row for row in diag_rows if _map_id(row) == map_id]
        base_map = [row for row in base_rows if _map_id(row) == map_id]
        by_map[map_id] = {
            "episodes": len(diag_map),
            "diagnostic_successes": sum(bool(row["success"]) for row in diag_map),
            "diagnostic_failures": sum(not bool(row["success"]) for row in diag_map),
            "diagnostic_misdiagnoses": sum(misdiagnosed(row) for row in diag_map),
            "baseline_successes": sum(bool(row["success"]) for row in base_map),
        }

    return {
        "study": "base_stress",
        "description": (
            "True-base identification episodes only; no shifts. "
            "Descriptive false-alarm characterization, not a gate evaluation."
        ),
        "gate_evaluation": False,
        "n_maps": int(n_maps),
        "seeds": int(seeds),
        "paired_episodes": len(ids),
        "methods": methods,
        "paired": {
            "both": both,
            "only_diagnostic": only_diag,
            "only_baseline": only_base,
            "neither": neither,
            "mcnemar_exact_p": _mcnemar_exact(only_diag, only_base),
        },
        "by_map": by_map,
        "diagnostic_failures": [
            _failure_row(row) for row in diag_rows if not bool(row["success"])
        ],
        "diagnostic_misdiagnoses": [
            _failure_row(row) for row in diag_rows if misdiagnosed(row)
        ],
        "diagnostic_misdiagnoses_with_success": [
            _failure_row(row)
            for row in diag_rows
            if misdiagnosed(row) and bool(row["success"])
        ],
        "bootstrap": {
            "n_boot": int(n_boot),
            "seed": int(bootstrap_seed),
            "unit": "map",
            "interval": "percentile 95%",
        },
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _ci_text(interval: Sequence[float]) -> str:
    return f"[{_pct(interval[0])}, {_pct(interval[1])}]"


def write_base_stress_report(payload: Mapping[str, Any], out_dir: Path) -> None:
    """Write the JSON payload and a compact Markdown summary."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    diag = payload["methods"]["diagnostic_belief"]
    base = payload["methods"]["no_adaptation"]
    paired = payload["paired"]
    lines = [
        "# Base-preservation stress study",
        "",
        "**Status: descriptive CPU study, not a gate evaluation or a preregistered result.**",
        "",
        "This is a narrow follow-up to the v0.3 10-map run. Every episode uses the",
        "true `base` rule, with no shift. It estimates how often the diagnostic",
        "policy fails the base task and how often it misdiagnoses `base` as another",
        "rule. It does not relax or replace the 100% base gate.",
        "",
        "## Scope",
        "",
        (
            f"- {payload['n_maps']} maps x {payload['seeds']} seeds = "
            f"{payload['paired_episodes']} paired base episodes per method."
        ),
        "- Methods: `diagnostic_belief` and `no_adaptation`.",
        "- CPU only; no frozen VLM, no GPU job, no shift episodes.",
        (
            f"- Raw: `{payload.get('raw_path', 'n/a')}`; SHA-256 "
            f"`{payload.get('raw_sha256', 'n/a')}`."
        ),
        "",
        "## Primary estimates",
        "",
        "| Method | Endpoint | Events | Rate | Wilson 95% | Map-cluster 95% |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in (("diagnostic_belief", diag), ("no_adaptation", base)):
        for endpoint in ("task_failure", "misdiagnosis"):
            record = metrics[endpoint]
            cluster = (
                "not estimable*"
                if record["map_cluster_degenerate"]
                else _ci_text(record["map_cluster_95"])
            )
            lines.append(
                f"| `{name}` | {endpoint} | {record['events']}/{record['episodes']} | "
                f"{_pct(record['rate'])} | {_ci_text(record['wilson_95'])} | "
                f"{cluster} |"
            )
    successful_misdiagnoses = payload["diagnostic_misdiagnoses_with_success"]
    lines.extend(
        [
            "",
            "`task_failure` is the complement of the original gate endpoint",
            "(`success == false`). `misdiagnosis` means the final estimate is not",
            "`base`. Wilson intervals treat episodes as independent; the map-cluster",
            "bootstrap resamples whole maps and is the more conservative interval.",
            "",
            (
                f"**{len(payload['diagnostic_misdiagnoses'])} misdiagnoses were observed; "
                f"{len(successful_misdiagnoses)} of them still reached the goal.** "
                "Task success and correct rule identification are therefore not "
                "interchangeable base-preservation endpoints."
            ),
            "",
            "## Paired episodes",
            "",
            f"- both succeed: {paired['both']}",
            f"- only diagnostic succeeds: {paired['only_diagnostic']}",
            f"- only baseline succeeds: {paired['only_baseline']}",
            f"- neither succeeds: {paired['neither']}",
            f"- exact McNemar p: {paired['mcnemar_exact_p']:.6g}",
            "",
        ]
    )
    paired_candidate = payload.get("paired_vs_v3")
    if isinstance(paired_candidate, Mapping):
        lines.extend(
            [
                "## Paired candidate vs v0.3",
                "",
                f"- both succeed: {paired_candidate['both_success']}",
                (
                    "- only the candidate succeeds: "
                    f"{paired_candidate['only_left_success']}"
                ),
                (
                    "- only v0.3 succeeds: "
                    f"{paired_candidate['only_right_success']}"
                ),
                f"- neither succeeds: {paired_candidate['neither_success']}",
                (
                    "- exact McNemar p: "
                    f"{paired_candidate['exact_mcnemar_p']:.6g}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Diagnostic failures",
            "",
        ]
    )
    failures = payload["diagnostic_failures"]
    if not failures:
        lines.append("None in this run.")
    else:
        lines.append("| Scenario | Estimate | Steps | Tool units |")
        lines.append("| --- | --- | ---: | ---: |")
        for row in failures:
            lines.append(
                f"| `{row['scenario_id']}` | `{row['final_rule_estimate']}` | "
                f"{row['env_steps']} | {row['tool_units']} |"
            )
    lines.extend(
        [
            "",
            "## Misdiagnoses that still succeeded",
            "",
        ]
    )
    if not successful_misdiagnoses:
        lines.append("None in this run.")
    else:
        lines.append("| Scenario | Estimate | Steps | Tool units |")
        lines.append("| --- | --- | ---: | ---: |")
        for row in successful_misdiagnoses:
            lines.append(
                f"| `{row['scenario_id']}` | `{row['final_rule_estimate']}` | "
                f"{row['env_steps']} | {row['tool_units']} |"
            )
    lines.extend(
        [
            "",
            "## Maps with diagnostic failures",
            "",
        ]
    )
    failing_maps = [
        (map_id, record)
        for map_id, record in payload["by_map"].items()
        if record["diagnostic_failures"] or record["diagnostic_misdiagnoses"]
    ]
    if not failing_maps:
        lines.append("No map had a diagnostic failure or misdiagnosis.")
    else:
        lines.append("| Map | Failures | Misdiagnoses | Episodes |")
        lines.append("| --- | ---: | ---: | ---: |")
        for map_id, record in failing_maps:
            lines.append(
                f"| `{map_id}` | {record['diagnostic_failures']} | "
                f"{record['diagnostic_misdiagnoses']} | {record['episodes']} |"
            )
    if any(
        metrics[endpoint]["map_cluster_degenerate"]
        for metrics in (diag, base)
        for endpoint in ("task_failure", "misdiagnosis")
    ):
        lines.extend(
            [
                "",
                "\\* The map-cluster interval is degenerate when every map has the",
                "same event rate; in that case use the Wilson interval instead.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This study estimates the base false-alarm/failure rate; it does not",
            "declare the base gate passed, change the benchmark, or authorize Phase 1.",
            "A new gate decision requires a separate preregistered protocol.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")