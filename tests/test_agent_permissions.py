"""Static enforcement tests for the agent/evaluator permissions boundary.

The imports below are analyzed as *source text* by :mod:`ast`; no forbidden
module is ever imported by the test process itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ruleshift import permissions

AGENTS_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "ruleshift" / "agents"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agent_package_has_no_boundary_violations() -> None:
    assert AGENTS_DIR.is_dir(), "agents package must exist"
    assert permissions.scan_package(AGENTS_DIR) == []


@pytest.mark.parametrize(
    "source",
    [
        "import ruleshift.envs.gate_lock",
        "from ruleshift.evaluator import bayes_filter",
        "from ..envs.gate_lock import RuleShiftEnv",
        "from ..evaluator.exact_solver import ExactBayesSolver",
        "import envs",
        "from evaluator import metrics",
    ],
)
def test_forbidden_imports_are_detected(source: str) -> None:
    violations = permissions.scan_source_for_forbidden_imports(source, "probe.py")
    assert len(violations) == 1
    assert violations[0].kind == "forbidden_import"


@pytest.mark.parametrize(
    "source",
    [
        "from ruleshift.public_api import schemas",
        "from ruleshift.public_api.interface import PublicObservation",
        "from .consistency import update_belief",
        "from dataclasses import dataclass",
    ],
)
def test_safe_imports_are_not_flagged(source: str) -> None:
    assert permissions.scan_source_for_forbidden_imports(source, "probe.py") == []


@pytest.mark.parametrize(
    "source",
    [
        "def score(env, action):\n    return env.copy().step(action)\n",
        "def score(env, action):\n    return env.step(action)\n",
        "def score(self, action):\n    return self.simulator.step(action)\n",
    ],
)
def test_counterfactual_simulator_calls_are_detected(source: str) -> None:
    violations = permissions.scan_source_for_counterfactual_calls(source, "probe.py")
    assert len(violations) == 1
    assert violations[0].kind == "forbidden_simulator_call"


def test_arithmetic_on_public_predictions_is_allowed() -> None:
    source = """
from ruleshift.public_api.schemas import MACRO_ACTIONS

def value(b_hat, r_hat, c):
    return sum(b_hat[m] * r_hat[m] for m in b_hat) - c
"""
    assert permissions.scan_source(source, "selector.py") == []


def test_field_inventories_are_disjoint() -> None:
    assert permissions.PUBLIC_OBSERVATION_FIELDS.isdisjoint(permissions.PRIVILEGED_FIELDS)


def test_privileged_inventory_is_complete() -> None:
    required = {
        "mechanic_id",
        "true_rule",
        "shift_time",
        "shift_schedule",
        "transition_kernel",
        "oracle_belief",
        "q_star",
        "delta_probe",
    }
    assert required <= set(permissions.PRIVILEGED_FIELDS)


@pytest.mark.parametrize(
    "payload, expected_name",
    [
        ({"true_rule": "blue"}, "true_rule"),
        ({"nested": {"shift_time": 12}}, "shift_time"),
        ({"items": [{"oracle_belief": {}}]}, "oracle_belief"),
        ({"prompt": "the delta_probe label is positive"}, "delta_probe"),
    ],
)
def test_find_forbidden_fields_finds_nested_and_string_leaks(
    payload: dict, expected_name: str
) -> None:
    found = permissions.find_forbidden_fields(payload)
    assert any(name == expected_name for _, name in found)


def test_find_forbidden_fields_clean_payload() -> None:
    payload = {
        "public_text": "Gate ahead; red or blue may be active.",
        "available_actions": ["COMMIT_RED", "WAIT"],
        "public_costs": {"primitive_step": 0.05},
    }
    assert permissions.find_forbidden_fields(payload) == []


def test_scan_package_reports_violations_in_a_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad_agent.py"
    bad.write_text("from ruleshift.evaluator import exact_solver\n", encoding="utf-8")
    violations = permissions.scan_package(tmp_path)
    assert any(v.kind == "forbidden_import" for v in violations)
    assert any(v.path == str(bad) for v in violations)


def test_permissions_doc_lists_every_privileged_field() -> None:
    doc = (REPO_ROOT / "docs" / "agent_evaluator_permissions.md").read_text(
        encoding="utf-8"
    )
    missing = sorted(name for name in permissions.PRIVILEGED_FIELDS if name not in doc)
    assert missing == [], f"undocumented privileged fields: {missing}"
