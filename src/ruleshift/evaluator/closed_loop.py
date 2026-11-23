"""Closed-loop Family A runner with one shared macro executor for all methods.

The VLM-side methods use the deterministic stub, so all outputs are SYNTHETIC.
What this runner verifies is the loop itself: identical action affordances and
step accounting, no privileged data in public traces, and paired scenario-level
confidence intervals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from ruleshift.agents.baselines import NeverProbePolicy, RandomProbePolicy
from ruleshift.agents.cache import ResponseCache
from ruleshift.agents.consistency import HistoryConsistencyUpdater
from ruleshift.agents.diva_style import DivaStyleSelector
from ruleshift.agents.structured_vlm import run_structured_prediction
from ruleshift.agents.value_selector import (
    ImmediateOnlySelector,
    PublicCostModel,
    ValueAwareSelector,
    public_macro_durations,
)
from ruleshift.agents.vlm_backend import StubVLMBackend
from ruleshift.envs.config import EnvConfig
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.metrics_ci import PairedSummary, summarize_paired
from ruleshift.public_api.interface import PublicObservation
from ruleshift.public_api.schemas import MACRO_ACTIONS_FROZEN

DEFAULT_VLM_CALL_CAP = 6
_RULES = ("red", "blue", "green")


@dataclass(frozen=True)
class Episode:
    scenario_id: str
    method: str
    total_return: float
    steps: int
    probes: int
    vlm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    invalid_actions: int
    public_rows: tuple[dict, ...]
    evaluator_rows: tuple[dict, ...]
    #: Decisions that fell back to WAIT because the VLM call cap was spent.
    call_cap_hits: int = 0


class _Method(Protocol):
    name: str

    def choose(self, observation: PublicObservation, info: dict) -> str: ...

    def record(self, action: str, outcome: str) -> None: ...


class _NeverProbeMethod:
    name = "never_probe"

    def __init__(self) -> None:
        self._policy = NeverProbePolicy()

    def choose(self, observation: PublicObservation, info: dict) -> str:
        return self._policy.choose(observation)

    def record(self, action: str, outcome: str) -> None:
        self._policy.record(action, outcome)


class _RandomProbeMethod:
    name = "random_probe"

    def __init__(self) -> None:
        self._policy: RandomProbePolicy | None = None

    def choose(self, observation: PublicObservation, info: dict) -> str:
        if self._policy is None:
            self._policy = RandomProbePolicy(observation.public_history_hash)
        return self._policy.choose(observation)

    def record(self, action: str, outcome: str) -> None:
        return None


class _StubMethod:
    def __init__(
        self,
        mode: str,
        backend: object | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.mode = mode
        self._backend = backend if backend is not None else StubVLMBackend()
        suffix = "_stub" if isinstance(self._backend, StubVLMBackend) else ""
        self.name = f"{mode}{suffix}"
        self._cache = cache if cache is not None else ResponseCache(None)
        self._updater = HistoryConsistencyUpdater(_RULES, learning_rate=0.5)
        self._belief = {rule: 1.0 / len(_RULES) for rule in _RULES}
        self.last_calls = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_latency_ms = 0.0

    @staticmethod
    def _cost(observation: PublicObservation, info: dict) -> PublicCostModel:
        """Signed cost model built from the public cost descriptions only."""

        costs = observation.public_costs
        probe_duration, commit_duration = public_macro_durations(costs)
        return PublicCostModel.from_public_costs(
            costs,
            probe_duration=probe_duration,
            commit_duration=commit_duration,
            remaining_horizon=max(
                1,
                int(info["max_env_steps"]) - int(info["physical_steps_elapsed"]),
            ),
            learning_rate=0.5,
        )

    def choose(self, observation: PublicObservation, info: dict) -> str:
        payload, response, errors = run_structured_prediction(
            observation, self._backend, self._cache
        )
        self.last_calls = 0 if errors and payload is None else 1
        self.last_input_tokens = response.input_tokens
        self.last_output_tokens = response.output_tokens
        self.last_latency_ms = response.latency_ms
        if payload is None:
            return "WAIT"
        cost = self._cost(observation, info)
        if self.mode == "direct":
            return payload["selected_action"]
        if self.mode == "diva":
            return DivaStyleSelector().choose(payload, self._belief, cost)
        if self.mode == "immediate":
            return ImmediateOnlySelector().choose(payload, self._belief, cost)
        return ValueAwareSelector().choose(payload, self._belief, cost)

    @property
    def belief(self) -> dict[str, float]:
        """Current public-only belief (read-only view for tests and logging)."""

        return dict(self._belief)

    def record(self, action: str, outcome: str) -> None:
        self._belief = self._updater.update(self._belief, action, outcome)


class _OracleMethod:
    """Privileged diagnostic: reads the true mechanic, never a main method."""

    name = "true_rule_oracle"
    is_privileged = True

    def __init__(self) -> None:
        self._env: RuleShiftEnv | None = None

    def attach_env(self, env: RuleShiftEnv) -> None:
        self._env = env

    def choose(self, observation: PublicObservation, info: dict) -> str:
        assert self._env is not None
        return f"COMMIT_{self._env.evaluator_state()['mechanic_id'].upper()}"

    def record(self, action: str, outcome: str) -> None:
        return None


#: Model-side method modes; the ``_stub`` suffix is cosmetic and derived from
#: whichever backend is injected, so both ``m1`` and ``m1_stub`` are accepted.
_STUB_MODES = ("direct", "m1", "diva", "immediate")


def _make_method(
    name: str,
    backend: object | None = None,
    cache: ResponseCache | None = None,
) -> _Method:
    if name == "never_probe":
        return _NeverProbeMethod()
    if name == "random_probe":
        return _RandomProbeMethod()
    if name == "true_rule_oracle":
        return _OracleMethod()
    mode = name.removesuffix("_stub")
    if mode in _STUB_MODES:
        return _StubMethod(mode, backend=backend, cache=cache)
    raise ValueError(f"unknown method: {name}")


def _last_outcome(public_text: str) -> str:
    match = re.search(r"last=(\w+)", public_text)
    return match.group(1) if match else "none"


def build_family_a_scenarios() -> tuple[tuple[str, EnvConfig], ...]:
    """A small paired dev grid covering both probe regimes and shift/no-shift."""

    scenarios: list[tuple[str, EnvConfig]] = []
    for detour in (0, 5):
        for noise in (0.0, 0.25):
            for lockout in (0, 8):
                for disable_shift, hazard in ((True, 0.0), (False, 0.1)):
                    scenario_id = f"famA-d{detour}-n{noise}-k{lockout}-s{int(not disable_shift)}"
                    scenarios.append(
                        (
                            scenario_id,
                            EnvConfig(
                                probe_detour_steps=detour,
                                feedback_noise=noise,
                                lockout_steps=lockout,
                                disable_shift=disable_shift,
                                shift_hazard=hazard,
                                initial_mechanic="red",
                                scenario_id=scenario_id,
                            ),
                        )
                    )
    return tuple(scenarios)


def _clamp_action(action: str, available_actions: Sequence[str]) -> tuple[str, bool]:
    """Clamp a predicted action to the environment's current public menu.

    Returns ``(action, invalid)``. An action that is not in ``available_actions``
    is replaced with ``WAIT`` and flagged invalid so the runner can count it as a
    neutral, budget-neutral decision instead of letting a cross-menu action leak
    into the environment's step loop.
    """

    if action in available_actions:
        return action, False
    return "WAIT", True


def run_episode(
    config: EnvConfig,
    method_name: str,
    seed: int,
    scenario_id: str,
    backend: object | None = None,
    cache: ResponseCache | None = None,
    max_vlm_calls_per_episode: int = DEFAULT_VLM_CALL_CAP,
) -> Episode:
    """Drive one full episode through the public macro interface.

    Once ``max_vlm_calls_per_episode`` model calls have been spent, further
    decisions use the documented deterministic ``WAIT`` fallback without
    calling the model, so every method stays inside the preregistered budget.
    """

    env = RuleShiftEnv(config=config)
    obs, info = env.reset(seed=seed)
    method = _make_method(method_name, backend=backend, cache=cache)
    if hasattr(method, "attach_env"):
        method.attach_env(env)

    public_rows: list[dict] = []
    evaluator_rows: list[dict] = []
    total = 0.0
    probes = 0
    invalid = 0
    calls = 0
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    cap_hits = 0

    for decision in range(config.max_env_steps):
        observation = env.public_view()
        if calls >= max_vlm_calls_per_episode:
            action = "WAIT"
            used_call = 0
            used_input = 0
            used_output = 0
            used_latency = 0.0
            cap_hits += 1
        else:
            action = method.choose(observation, info)
            used_call = int(getattr(method, "last_calls", 0))
            used_input = int(getattr(method, "last_input_tokens", 0))
            used_output = int(getattr(method, "last_output_tokens", 0))
            used_latency = float(getattr(method, "last_latency_ms", 0.0))
        # A method may emit an action that is valid on some families but not on
        # the family the environment advertises. Clamp to the public menu and
        # count the miss as an invalid (neutral) action rather than letting it
        # land as a silent WAIT with no decision credit.
        clamped_action, action_invalid = _clamp_action(action, observation.available_actions)
        if action_invalid:
            invalid += 1
        action = clamped_action
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        method.record(action, _last_outcome(env.public_view().public_text))

        public_rows.append(
            {
                "run_id": f"{scenario_id}-s{seed}",
                "episode_id": f"{scenario_id}-s{seed}",
                "scenario_id": scenario_id,
                "seed": seed,
                "method": method_name,
                "decision_id": decision,
                "public_obs_hash": observation.public_obs_hash,
                "public_history_hash": observation.public_history_hash,
                "available_actions": list(observation.available_actions),
                "chosen_action": action,
                "reward": reward,
                "physical_steps_elapsed": info["physical_steps_elapsed"],
                "vlm_calls_used": used_call,
                "input_tokens": used_input,
                "output_tokens": used_output,
                "latency_ms": used_latency,
                "cap_fallback": used_call == 0 and action == "WAIT" and cap_hits > 0,
            }
        )
        evaluator_rows.append(dict(env.evaluator_state()))
        if action.startswith("PROBE_"):
            probes += 1
        calls += used_call
        input_tokens += used_input
        output_tokens += used_output
        latency += used_latency
        if terminated or truncated:
            break

    return Episode(
        scenario_id=scenario_id,
        method=method_name,
        total_return=total,
        steps=int(info["physical_steps_elapsed"]),
        probes=probes,
        vlm_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency,
        invalid_actions=invalid,
        public_rows=tuple(public_rows),
        evaluator_rows=tuple(evaluator_rows),
        call_cap_hits=cap_hits,
    )


@dataclass
class ClosedLoopOutput:
    episodes: dict[tuple[str, int, str], Episode]
    summaries: dict[tuple[str, str], PairedSummary]

    def __getitem__(self, key: tuple[str, str]) -> PairedSummary:
        return self.summaries[key]


def _scenario_cluster_means(
    episodes: dict[tuple[str, int, str], Episode],
    methods: tuple[str, str],
) -> tuple[dict[str, list[float]], int, int]:
    """Per-scenario mean returns for two methods, paired on shared scenarios.

    The independent unit is the scenario: seeds are averaged *within* a
    scenario rather than treated as extra independent observations, which
    would shrink the paired interval for the wrong reason.
    """

    by_method: dict[str, dict[str, list[float]]] = {method: {} for method in methods}
    for (scenario_id, _seed, name), episode in episodes.items():
        if name in by_method:
            by_method[name].setdefault(scenario_id, []).append(episode.total_return)

    shared = sorted(set.intersection(*(set(v) for v in by_method.values()))) if all(
        by_method.values()
    ) else []
    means = {
        method: [
            sum(by_method[method][scenario_id]) / len(by_method[method][scenario_id])
            for scenario_id in shared
        ]
        for method in methods
    }
    n_episodes = sum(len(v) for v in by_method[methods[0]].values())
    return means, len(shared), n_episodes


def run_closed_loop(
    scenarios: tuple[tuple[str, EnvConfig], ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...] = (1,),
    comparisons: tuple[tuple[str, str], ...] = (("m1_stub", "never_probe"),),
    backend: object | None = None,
    cache: ResponseCache | None = None,
    max_vlm_calls_per_episode: int = DEFAULT_VLM_CALL_CAP,
) -> ClosedLoopOutput:
    """Run paired episodes and compute scenario-level paired summaries."""

    episodes: dict[tuple[str, int, str], Episode] = {}
    for scenario_id, config in scenarios:
        for seed in seeds:
            for method in methods:
                episodes[(scenario_id, seed, method)] = run_episode(
                    config,
                    method,
                    seed,
                    scenario_id,
                    backend=backend,
                    cache=cache,
                    max_vlm_calls_per_episode=max_vlm_calls_per_episode,
                )

    summaries: dict[tuple[str, str], PairedSummary] = {}
    for method, baseline in comparisons:
        if method not in methods or baseline not in methods:
            continue
        means, n_scenarios, n_episodes = _scenario_cluster_means(
            episodes, (method, baseline)
        )
        if n_scenarios == 0:
            continue
        summaries[(method, baseline)] = summarize_paired(
            means[method],
            means[baseline],
            seed=0,
            n_episodes=n_episodes,
        )
    return ClosedLoopOutput(episodes=episodes, summaries=summaries)


def write_episode_files(result: ClosedLoopOutput, output_dir: Path) -> tuple[Path, Path]:
    """Write public decisions and privileged traces to separate files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "episodes.jsonl"
    evaluator_path = output_dir / "evaluator_only.jsonl"
    with public_path.open("w", encoding="utf-8") as handle:
        for episode in result.episodes.values():
            for row in episode.public_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    with evaluator_path.open("w", encoding="utf-8") as handle:
        for episode in result.episodes.values():
            for row in episode.evaluator_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return public_path, evaluator_path


def write_report(result: ClosedLoopOutput, path: Path) -> Path:
    """Write the Week 7 report; all numbers here are synthetic."""

    lines = [
        "# Week 7 — Family A closed loop (SYNTHETIC)",
        "",
        "**SYNTHETIC: the model-side methods use a deterministic stub. No VLM "
        "was run.** This validates the loop, the cost accounting and the "
        "separation of public/evaluator traces — not model behaviour.",
        "",
        "All methods use the same public macro menu, the same macro executor "
        "and the same primitive-step budget. Costs are charged once in the "
        "shared reward stream.",
        "",
        "| Comparison | Scenarios | Mean paired difference | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for (method, baseline), summary in result.summaries.items():
        lines.append(
            f"| {method} − {baseline} | {summary.n_scenarios} | "
            f"{summary.mean_difference:+.3f} | [{summary.low:+.3f}, {summary.high:+.3f}] |"
        )
    lines += [
        "",
        " episodes.jsonl contains public decision rows only. "
        "evaluator_only.jsonl is a separate privileged trace and is never "
        "joined into model input.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
