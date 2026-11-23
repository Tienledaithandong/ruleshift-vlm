"""Tier B grid VLM prompt, parser, cache and stub policy (public-only).

This module is a SCAFFOLD for the offline harness: local CI exercises a
deterministic stub backend, the parser and the permission boundary, with no
model download and no GPU. It imports only the standard library, ``public_api``,
other ``agents`` modules and ``permissions`` -- never the simulator or the
privileged evaluator.

Observations may arrive either as a canonical public payload dict (what
``GridEnv.reset``/``step`` return) or as an attribute-accessible view (what the
runner passes to a policy). Every field read here goes through ``_field`` so
both shapes work without adding or hiding state.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruleshift.permissions import find_forbidden_fields
from ruleshift.public_api.grid_schemas import (
    GRID_PHYSICAL_ACTIONS,
    GRID_RULE_IDS,
    GridAction,
    validate_grid_vlm_output,
)

from .cache import ResponseCache
from .vlm_backend import ModelRequest, ModelResponse

GRID_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "grid_v1.txt"
GRID_PROMPT_VERSION = "grid_v1"

__all__ = [
    "GRID_PROMPT_PATH",
    "GRID_PROMPT_VERSION",
    "GridStubBackend",
    "GridVLMConfig",
    "GridVLMPolicy",
    "grid_candidate_hash",
    "parse_grid_output",
    "render_grid_prompt",
]


# ---------------------------------------------------------------------------
# Observation access (dict payloads and attribute views both work)
# ---------------------------------------------------------------------------


def _payload(observation: Any) -> Mapping[str, Any]:
    """Return the canonical public payload dict for *observation*, if any."""

    if isinstance(observation, Mapping):
        return observation
    to_payload = getattr(observation, "to_payload", None)
    if callable(to_payload):
        result = to_payload()
        if isinstance(result, Mapping):
            return result
    return {}


def _field(observation: Any, name: str, default: Any = None) -> Any:
    """Read a public field by attribute first, then by dict key."""

    if hasattr(observation, name):
        value = getattr(observation, name)
        if value is not None:
            return value
    payload = _payload(observation)
    try:
        value = payload[name]
    except (KeyError, TypeError):
        return default
    return value if value is not None else default


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _candidate_pairs(rules: Any) -> list[dict[str, str]]:
    """Normalize public candidate rules to ``rule_id``/``description`` rows."""

    pairs: list[dict[str, str]] = []
    for rule in _as_tuple(rules):
        if isinstance(rule, Mapping):
            pairs.append(
                {
                    "rule_id": str(rule.get("rule_id", "")),
                    "description": str(rule.get("description", "")),
                }
            )
        else:
            pairs.append(
                {
                    "rule_id": str(getattr(rule, "rule_id", "")),
                    "description": str(getattr(rule, "description", "")),
                }
            )
    return pairs


def _grid_size(observation: Any) -> tuple[int, int]:
    """Read ``(width, height)`` from the public map in either shape."""

    public_map = _field(observation, "public_map")
    if public_map is None:
        return (0, 0)
    try:
        return (int(_field(public_map, "width", 0)), int(_field(public_map, "height", 0)))
    except (TypeError, ValueError):
        return (0, 0)


# ---------------------------------------------------------------------------
# Prompt and candidate identity
# ---------------------------------------------------------------------------


def render_grid_prompt(observation: Any) -> str:
    """Fill the public-only grid template from an agent-visible observation.

    Only public fields are interpolated. The raw ``image`` is never passed into
    the prompt; at most its digest (``image_hash``) is readable on the request.
    """

    template = GRID_PROMPT_PATH.read_text(encoding="utf-8")
    candidates = [
        f"{row['rule_id']}: {row['description']}"
        for row in _candidate_pairs(_field(observation, "candidate_rules", ()))
    ]
    costs = _field(observation, "public_costs", {}) or {}
    cost_text = (
        ", ".join(f"{key}={value}" for key, value in sorted(costs.items()))
        if isinstance(costs, Mapping)
        else str(costs)
    )
    position = _as_tuple(_field(observation, "agent_position", (0, 0)))
    pos_x = position[0] if len(position) > 0 else 0
    pos_y = position[1] if len(position) > 1 else 0
    width, height = _grid_size(observation)
    return template.format(
        public_history=str(_field(observation, "public_history", "")),
        public_text=str(_field(observation, "public_text", "")),
        candidate_rules=", ".join(candidates),
        available_actions=", ".join(str(a) for a in _as_tuple(_field(observation, "available_actions", ()))),
        public_costs=cost_text,
        action_mask=", ".join(str(a) for a in _as_tuple(_field(observation, "action_mask", ()))),
        tool_budget=str(_field(observation, "tool_budget_remaining", 0)),
        position=f"({pos_x},{pos_y})",
        grid_size=f"({width},{height})",
    )


def grid_candidate_hash(observation: Any) -> str:
    """Hash only the public candidate-rule menu (stable across both shapes)."""

    rows = _candidate_pairs(_field(observation, "candidate_rules", ()))
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _is_coord(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    )


def _inspect_target_errors(
    target: Any,
    grid_size: Any,
    position: Any,
    inspect_range: Any,
) -> list[str]:
    """Range/bounds checks for an INSPECT target; never raises."""

    errors: list[str] = []
    if not _is_coord(grid_size) or min(grid_size) <= 0:
        return ["grid_size must be a positive (width, height) pair"]
    x, y = target
    if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
        errors.append("INSPECT target is outside the grid")
    if position is not None and inspect_range is not None:
        if not _is_coord(position):
            errors.append("position must be a two-integer pair")
        else:
            try:
                limit = int(inspect_range)
            except (TypeError, ValueError):
                errors.append("inspect_range must be a number")
            else:
                reach = abs(x - position[0]) + abs(y - position[1])
                if reach > limit:
                    errors.append("INSPECT target is outside inspect_range")
    return errors


def parse_grid_output(
    text: str,
    *,
    action_mask: Sequence[str],
    grid_size: Any,
    position: Any = None,
    inspect_range: Any = None,
) -> tuple[dict | None, list[str]]:
    """Parse one grid VLM decision; return ``(payload, errors)``, never raise."""

    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    try:
        payload = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        return None, [f"invalid JSON: {exc}"]

    try:
        errors = validate_grid_vlm_output(payload)
        if isinstance(payload, dict):
            action = payload.get("action")
            mask = tuple(action_mask) if action_mask is not None else ()
            if action not in mask:
                errors.append("action is not in action_mask")
            if action == "INSPECT":
                target = payload.get("tool_target")
                if _is_coord(target):
                    errors.extend(
                        _inspect_target_errors(target, grid_size, position, inspect_range)
                    )
        if errors:
            return None, errors
        return payload, []
    except Exception as exc:  # defensive: validation must never raise
        return None, [f"grid output validation error: {exc}"]


# ---------------------------------------------------------------------------
# Backend and policy
# ---------------------------------------------------------------------------


class GridStubBackend:
    """Deterministic stand-in for a grid VLM; never mistaken for evidence."""

    backend_id = "GRID_STUB_SYNTHETIC"

    def __init__(self, model_id: str = "stub-grid-v0") -> None:
        self.model_id = model_id
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        seed = int(request.fingerprint().removeprefix("sha256:")[:8], 16)
        rng = random.Random(seed ^ request.decoding_seed)
        # The stub cannot choose a meaningful INSPECT target, so it only emits
        # physical actions that are legal in this decision's mask.
        physical = [
            action
            for action in request.available_actions
            if action in GRID_PHYSICAL_ACTIONS
        ] or [GRID_PHYSICAL_ACTIONS[0]]
        payload = {
            "hypothesis_probabilities": {
                rule: 1.0 / len(GRID_RULE_IDS) for rule in GRID_RULE_IDS
            },
            "action": rng.choice(physical),
            "tool_target": None,
            "confidence": 0.5,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return ModelResponse(
            text=text,
            input_tokens=len(request.prompt.split()) + len(request.available_actions),
            output_tokens=len(text.split()),
            latency_ms=0.0,
            model_id=self.model_id,
            backend_id=self.backend_id,
        )


@dataclass(frozen=True)
class GridVLMConfig:
    """Public-only configuration for a grid VLM policy."""

    model_id: str
    revision: str | None = None
    decoding_seed: int = 0
    max_new_tokens: int = 1024


class GridVLMPolicy:
    """Cached, schema-validated grid policy over a VLM backend.

    ``advisor_calls`` stays at zero: Tier B is not priced by the Tier A
    ``lambda`` advisor comparison. All accounting is public and row-safe.
    """

    name = "grid_vlm"
    method = "grid_vlm"

    def __init__(
        self,
        backend: Any,
        config: GridVLMConfig,
        cache: ResponseCache | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.cache = cache if cache is not None else ResponseCache(None)
        self._last_row: dict[str, Any] = {}
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.advisor_calls = 0
        self.output_tokens = 0
        self.input_tokens = 0
        self.latency_ms = 0.0
        self.model_calls = 0
        self.cache_hits = 0
        self.last_output_tokens = 0
        self.last_input_tokens = 0
        self.last_latency_ms = 0.0
        self.last_cache_hit = False
        self.last_json_valid = True
        self.last_valid_action = True
        self.last_probabilities: dict[str, float] = {}
        self.last_tool_calls: tuple[Any, ...] = ()
        self.information_gain_bits = 0.0

    def reset(self, observation: Any, info: dict[str, Any]) -> None:
        self._reset_counters()
        self._last_row = {}

    def _prompt_version(self) -> str:
        # ModelRequest fingerprints prompt_version, so pinning the revision here
        # keeps it in the cache key while model_id stays bare for the backend.
        return f"{GRID_PROMPT_VERSION}|{self.config.revision or 'none'}"

    def choose(self, observation: Any, info: dict[str, Any]) -> GridAction:
        action_mask = tuple(str(a) for a in _as_tuple(_field(observation, "action_mask", ())))
        request = ModelRequest(
            model_id=self.config.model_id,
            prompt=render_grid_prompt(observation),
            image_hash=str(_field(observation, "image_hash", "") or ""),
            history_hash=str(_field(observation, "public_history_hash", "") or ""),
            available_actions=action_mask,
            candidate_hash=grid_candidate_hash(observation),
            decoding_seed=self.config.decoding_seed,
            max_new_tokens=self.config.max_new_tokens,
            prompt_version=self._prompt_version(),
        )

        cached = self.cache.get(request)
        self.last_cache_hit = cached is not None
        if cached is None:
            response = self.backend.generate(request)
            self.cache.put(request, response)
            self.model_calls += 1
        else:
            response = cached
            self.cache_hits += 1

        self.output_tokens += int(response.output_tokens)
        self.input_tokens += int(response.input_tokens)
        self.latency_ms += float(response.latency_ms)
        self.last_output_tokens = int(response.output_tokens)
        self.last_input_tokens = int(response.input_tokens)
        self.last_latency_ms = float(response.latency_ms)

        width, height = _grid_size(observation)
        position = _field(observation, "agent_position")
        costs = _field(observation, "public_costs", {}) or {}
        inspect_range = (
            costs.get("inspect_range") if isinstance(costs, Mapping) else None
        )
        payload, errors = parse_grid_output(
            response.text,
            action_mask=action_mask,
            grid_size=(width, height),
            position=tuple(position) if position is not None else None,
            inspect_range=inspect_range,
        )

        if errors or payload is None:
            self.last_json_valid = False
            self.last_valid_action = False
            self.last_probabilities = {}
            fallback = next(
                (action for action in GRID_PHYSICAL_ACTIONS if action in action_mask),
                GRID_PHYSICAL_ACTIONS[0],
            )
            return GridAction(fallback)

        self.last_json_valid = True
        self.last_valid_action = True
        self.last_probabilities = dict(payload.get("hypothesis_probabilities") or {})
        target = payload.get("tool_target")
        return GridAction(
            str(payload["action"]),
            tuple(target) if target is not None else None,
        )

    def record(
        self,
        action: GridAction,
        observation: Any,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        info = info or {}
        row: dict[str, Any] = {
            "observation_id": str(_field(observation, "observation_id", "") or ""),
            "public_obs_hash": str(_field(observation, "public_obs_hash", "") or ""),
            "public_history_hash": str(_field(observation, "public_history_hash", "") or ""),
            "chosen_action": action.kind,
            "tool_target": list(action.target) if action.target is not None else None,
            "actual_reward": float(reward),
            "env_steps": int(info.get("env_steps", 0)),
            "tool_budget_remaining": int(
                info.get(
                    "tool_budget_remaining",
                    _field(observation, "tool_budget_remaining", 0) or 0,
                )
            ),
            "advisor_calls": int(self.advisor_calls),
            "json_valid": bool(self.last_json_valid),
            "valid_action": bool(self.last_valid_action),
            "input_tokens": int(self.last_input_tokens),
            "output_tokens": int(self.last_output_tokens),
            "latency_ms": float(self.last_latency_ms),
            "model_calls": int(self.model_calls),
            "cache_hits": int(self.cache_hits),
            "model_id": str(self.config.model_id),
            "prompt_version": self._prompt_version(),
        }
        if find_forbidden_fields(row):
            # Defensive fallback: never store a row that leaked a privileged name.
            row = {
                "chosen_action": action.kind,
                "json_valid": bool(self.last_json_valid),
            }
        self._last_row = row

    def most_likely_rule(self) -> str:
        probabilities = self.last_probabilities
        if not probabilities:
            return "base"
        return max(
            GRID_RULE_IDS,
            key=lambda rule: float(probabilities.get(rule, 0.0)),
        )

    def shift_detected(self) -> bool:
        return self.most_likely_rule() != "base"

    def last_public_row(self) -> dict[str, Any]:
        """Return a copy of the last public decision row."""

        return dict(self._last_row)
