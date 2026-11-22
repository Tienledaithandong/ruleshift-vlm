"""Static permissions contract for the RuleShift agent/evaluator boundary.

This module is deliberately dependency-free (standard library only) and is
imported by both the public schema validators and the CI permission tests. It
contains *inventories and static analysis helpers*, not a simulator.

Two distinct guarantees are expressed here:

1. A field inventory: which names may appear in agent-visible payloads and
   which names are evaluator-only.
2. A static import/call scan: agent source code must not import the environment
   or evaluator packages, and must not score counterfactual actions by calling
   ``env.copy().step(...)``.

Nothing in this module executes agent code; it only parses it with :mod:`ast`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Field inventory
# ---------------------------------------------------------------------------

#: Evaluator-only names. These must never appear in an agent-visible payload,
#: prompt, prompt template, or cache key/payload. This set is intentionally
#: over-inclusive: it is better to reject a harmless name than to leak a
#: privileged one. (See docs/agent_evaluator_permissions.md.)
PRIVILEGED_FIELDS: frozenset[str] = frozenset(
    {
        "mechanic_id",
        "true_rule",
        "true_rule_before_after",
        "active_mechanic",
        "shift_time",
        "shift_schedule",
        "shift_hazard",
        "latent_hazard",
        "hidden_rng",
        "hidden_state",
        "transition_kernel",
        "transition_probabilities",
        "oracle_belief",
        "bayes_belief",
        "q_star",
        "exact_q_values",
        "delta_probe",
        "actual_noise_outcome",
        "privileged_oracle_return",
        "scenario_generating_seed",
    }
)

#: Complete set of top-level keys allowed in an agent-visible observation.
#: Kept here so schemas and tests share a single source of truth.
PUBLIC_OBSERVATION_FIELDS: frozenset[str] = frozenset(
    {
        "observation_id",
        "public_obs_hash",
        "public_history_hash",
        "public_text",
        "available_actions",
        "candidate_rules",
        "public_costs",
    }
)

#: Names that, when appearing as an import path segment in agent source,
#: identify a forbidden package (the simulator or the privileged evaluator).
FORBIDDEN_MODULE_SEGMENTS: frozenset[str] = frozenset({"envs", "evaluator"})

#: Attribute receivers that make a direct ``<receiver>.step(...)`` call
#: suspicious inside the agents package. Counterfactual scoring is forbidden;
#: agents only receive already-executed outcomes.
_ENV_LIKE_TOKENS: frozenset[str] = frozenset(
    {"env", "environment", "sim", "simulator", "env_copy", "copied_env"}
)

_FIELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE))
    for name in sorted(PRIVILEGED_FIELDS)
)


class Violation(NamedTuple):
    """A single static boundary violation."""

    path: str
    line: int
    kind: str
    detail: str


# ---------------------------------------------------------------------------
# Payload scanning
# ---------------------------------------------------------------------------


def find_forbidden_fields(value: Any, path: str = "$") -> list[tuple[str, str]]:
    """Recursively find privileged field names in a JSON-like payload.

    Dict keys are matched exactly; string values are also scanned with word
    boundaries so that a privileged name smuggled into a prompt string or a
    cache key is still caught.

    Returns a list of ``(json_path, matched_name)`` pairs.
    """

    found: list[tuple[str, str]] = []

    def visit(item: Any, current: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                child = f"{current}.{key}"
                if isinstance(key, str) and key in PRIVILEGED_FIELDS:
                    found.append((child, key))
                visit(nested, child)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{current}[{index}]")
        elif isinstance(item, str):
            for name, pattern in _FIELD_PATTERNS:
                if pattern.search(item):
                    found.append((current, name))

    visit(value, path)
    return found


# ---------------------------------------------------------------------------
# Source scanning (AST based, never executes the scanned code)
# ---------------------------------------------------------------------------


def _module_is_forbidden(module: str | None) -> bool:
    if not module:
        return False
    segments = module.replace("/", ".").split(".")
    return any(segment in FORBIDDEN_MODULE_SEGMENTS for segment in segments)


def _looks_like_env(identifier: str) -> bool:
    lowered = identifier.lower()
    return lowered in _ENV_LIKE_TOKENS or "env" in lowered or lowered.startswith("sim")


def scan_source_for_forbidden_imports(
    source: str, path: str = "<memory>"
) -> list[Violation]:
    """Flag imports of ``envs``/``evaluator`` in agent source."""

    tree = ast.parse(source, filename=path)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_is_forbidden(alias.name):
                    violations.append(
                        Violation(
                            path,
                            node.lineno,
                            "forbidden_import",
                            f"import {alias.name}",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = [alias.name for alias in node.names]
            if _module_is_forbidden(module) or any(
                _module_is_forbidden(name) for name in imported
            ):
                target = module or ", ".join(imported)
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        "forbidden_import",
                        f"from {'.' * node.level}{target} import ...",
                    )
                )
    return violations


def scan_source_for_counterfactual_calls(
    source: str, path: str = "<memory>"
) -> list[Violation]:
    """Flag ``env.copy().step(...)`` and direct ``env.step(...)`` calls."""

    tree = ast.parse(source, filename=path)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "step":
            continue

        receiver = node.func.value
        detail: str | None = None
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "copy"
        ):
            detail = "counterfactual scoring via env.copy().step(...)"
        elif isinstance(receiver, ast.Name) and _looks_like_env(receiver.id):
            detail = f"direct simulator call via {receiver.id}.step(...)"
        elif isinstance(receiver, ast.Attribute) and _looks_like_env(receiver.attr):
            detail = f"direct simulator call via .{receiver.attr}.step(...)"

        if detail is not None:
            violations.append(
                Violation(path, node.lineno, "forbidden_simulator_call", detail)
            )
    return violations


def scan_source(source: str, path: str = "<memory>") -> list[Violation]:
    """Run every static source check that applies to agent code."""

    return sorted(
        [
            *scan_source_for_forbidden_imports(source, path),
            *scan_source_for_counterfactual_calls(source, path),
        ],
        key=lambda v: (v.path, v.line, v.kind, v.detail),
    )


def scan_package(root: Path) -> list[Violation]:
    """Scan every ``*.py`` file under *root* without importing any of them."""

    violations: list[Violation] = []
    root = Path(root)
    if not root.exists():
        return violations

    for source_file in sorted(root.rglob("*.py")):
        text = source_file.read_text(encoding="utf-8")
        try:
            violations.extend(scan_source(text, str(source_file)))
        except SyntaxError as exc:  # pragma: no cover - malformed agent code
            violations.append(
                Violation(str(source_file), exc.lineno or 0, "syntax_error", str(exc))
            )
    return violations
