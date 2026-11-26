"""Audit legacy Grid result rows without re-running any policy.

The legacy Phase-0 exporter used task success for the identification gate and
used final_rule_estimate != base as a shift-detected proxy. This audit recovers
both fields and compares them with independent task and terminal-MAP measures.
It is development-only: no new episode is created and no result is promoted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PREFIX = re.compile(
    r"^m(?P<map>\d+)-s(?P<seed>\d+)-"
    r"(?P<type>identification|change_detection|no_shift)-"
    r"(?P<initial>[a-z_]+)-(?P<shift>None|[a-z_]+)$"
)


def parse_scenario_id(scenario_id: str) -> dict[str, Any] | None:
    match = PREFIX.match(scenario_id)
    if not match:
        return None
    shift = match.group("shift")
    return {
        "map_seed": int(match.group("map")),
        "run_seed": int(match.group("seed")),
        "scenario_type": match.group("type"),
        "initial_rule": match.group("initial"),
        "shift_rule": None if shift == "None" else shift,
    }


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _audit_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_success = sum(bool(row.get("success", False)) for row in rows)
    map_correct = 0
    old_joint = 0
    misdiagnosis = 0
    map_correct_task_failure = 0
    static_rows = 0
    static_task = 0
    static_map = 0
    static_joint = 0
    static_misdiagnosis = 0
    static_map_correct_task_failure = 0
    base_rows = []
    shift_flag_mismatch = 0
    shift_flag_missing = 0
    shift_proxy_true = 0
    zero_tool_hazard_sign_success = 0
    tool_units: list[int] = []
    malformed = 0
    ids: list[str] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        ids.append(scenario_id)
        scenario = parse_scenario_id(scenario_id)
        if scenario is None:
            malformed += 1
            continue
        initial = str(row.get("initial_rule") or scenario["initial_rule"])
        final = str(row.get("final_rule_estimate", ""))
        success = bool(row.get("success", False))
        correct = final == initial
        map_correct += int(correct)
        old_joint += int(success and correct)
        misdiagnosis += int(success and not correct)
        map_correct_task_failure += int((not success) and correct)
        if scenario["scenario_type"] == "identification":
            static_rows += 1
            static_task += int(success)
            static_map += int(correct)
            static_joint += int(success and correct)
            static_misdiagnosis += int(success and not correct)
            static_map_correct_task_failure += int((not success) and correct)
        tools = int(row.get("tool_units", 0) or 0)
        tool_units.append(tools)
        if scenario["scenario_type"] == "identification" and initial == "base":
            base_rows.append(
                {
                    "scenario_id": scenario_id,
                    "success": success,
                    "map_correct": correct,
                    "tools": tools,
                }
            )
        if scenario["scenario_type"] == "identification" and success and tools == 0:
            if initial in ("hazard_goal_swap", "sign_composition"):
                zero_tool_hazard_sign_success += 1
        if "shift_detected" in row:
            flag = bool(row["shift_detected"])
            proxy = final != "base"
            shift_proxy_true += int(proxy)
            shift_flag_mismatch += int(flag != proxy)
        else:
            shift_flag_missing += 1
    return {
        "rows": len(rows),
        "malformed_scenario_ids": malformed,
        "duplicate_scenario_ids": len(ids) - len(set(ids)),
        "task_successes": task_success,
        "task_success_rate": task_success / len(rows) if rows else None,
        "map_correct": map_correct,
        "map_accuracy_initial_rule_all_rows": map_correct / len(rows) if rows else None,
        "static_identification_rows": static_rows,
        "static_task_successes": static_task,
        "static_task_success_rate": _rate(static_task, static_rows),
        "static_map_correct": static_map,
        "static_map_accuracy": _rate(static_map, static_rows),
        "static_task_and_map_correct": static_joint,
        "static_successful_misdiagnosis": static_misdiagnosis,
        "static_map_correct_but_task_failure": static_map_correct_task_failure,
        "old_final_rule_matches_initial_requiring_success": old_joint,
        "successful_misdiagnosis": misdiagnosis,
        "map_correct_but_task_failure": map_correct_task_failure,
        "shift_proxy_true": shift_proxy_true,
        "shift_flag_mismatches": shift_flag_mismatch,
        "shift_flag_missing": shift_flag_missing,
        "base_identification_rows": len(base_rows),
        "base_task_successes": sum(item["success"] for item in base_rows),
        "base_map_correct": sum(item["map_correct"] for item in base_rows),
        "base_task_success_ids_failed": [
            item["scenario_id"] for item in base_rows if not item["success"]
        ],
        "zero_tool_hazard_or_sign_successes": zero_tool_hazard_sign_success,
        "tool_units_distribution": _distribution(tool_units),
        "commitment_status": "unavailable",
        "posterior_metrics": "unavailable_in_legacy_rows",
    }


def audit_rows(
    rows: list[dict[str, Any]],
    source: str,
    *,
    map_min: int | None = None,
    map_max: int | None = None,
) -> dict[str, Any]:
    filtered = []
    for row in rows:
        scenario = parse_scenario_id(str(row.get("scenario_id", "")))
        if scenario is not None:
            if map_min is not None and scenario["map_seed"] < map_min:
                continue
            if map_max is not None and scenario["map_seed"] > map_max:
                continue
        filtered.append(row)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_method[str(row.get("method", "unknown"))].append(row)
    return {
        "source": source,
        "map_min": map_min,
        "map_max": map_max,
        "rows": len(filtered),
        "methods": {
            method: _audit_method(method_rows)
            for method, method_rows in sorted(by_method.items())
        },
        "note": (
            "Development-only audit of existing rows; no policy was re-run. "
            "Old labels are not rewritten."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default=None)
    parser.add_argument("--map-min", type=int, default=None)
    parser.add_argument("--map-max", type=int, default=None)
    args = parser.parse_args()
    reports = []
    for value in args.inputs:
        path = Path(value)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reports.append(
            audit_rows(
                rows,
                str(path),
                map_min=args.map_min,
                map_max=args.map_max,
            )
        )
    payload = {"reports": reports}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
