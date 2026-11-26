"""P1 development decisions reached by genuine public Gate/Lock histories."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice, product
from pathlib import Path

from ruleshift.envs.config import EnvConfig, HAZARD_BY_CONDITION, load_dev_grid
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.exact_solver import ExactSolver, family_a_microtask, freeze_belief, macro_branches
from ruleshift.evaluator.offline_audit import OfflineContext
from ruleshift.public_api.schemas import FAMILY_A_ACTIONS


def build_history_contexts(
    count: int = 81, config_path: Path | None = None,
) -> tuple[OfflineContext, ...]:
    """Score short-horizon decisions after one to three scripted macro actions.

    Prefixes and public costs are fixed before sampling episode randomness.
    Every history uses the same noisy sensor and shift hazard. The evaluator
    conditions on public outcomes and marginalizes the unseen single shift.
    """
    dev = load_dev_grid(config_path)
    settings, grid = dev["environment"], dev["grid"]
    if dev["split"] != "dev" or settings["family"] != "gate_lock":
        raise ValueError("history contexts require the Family A development configuration")
    rules = tuple(settings["candidate_rules"])
    initial_rule = settings["seed_rule"]
    if set(rules) != {"red", "blue", "green"} or initial_rule not in rules:
        raise ValueError("history contexts require the three Gate/Lock card candidates")
    targets = tuple(rule for rule in rules if rule != initial_rule)
    noisy_rates = [rate for rate in grid["feedback_noise_rate"] if 0 < rate < 0.5]
    hazards = [HAZARD_BY_CONDITION[condition] for condition in grid["shift_condition"]]
    if not noisy_rates or not any(0 < hazard < 1 for hazard in hazards):
        raise ValueError("history contexts need informative noisy feedback and a stochastic shift")
    noise = min(noisy_rates)
    hazard = max(h for h in hazards if 0 < h < 1)
    detours, lockouts = grid["probe_detour_steps"], settings["lockout_steps_options"]
    if not detours or not lockouts:
        raise ValueError("history contexts need a station detour and lockout setting")

    first_actions = tuple(
        action for rule in rules for action in (f"PROBE_{rule.upper()}", f"COMMIT_{rule.upper()}")
    )
    suffix_actions = ("WAIT", "GO_TO_STATION", *first_actions)
    prefixes = (
        (first, *suffix) for length in range(3)
        for suffix in product(suffix_actions, repeat=length) for first in first_actions
    )
    capacity = len(first_actions) * sum(len(suffix_actions)**n for n in range(3))
    if not 1 <= count <= capacity:
        raise ValueError(f"history context count must be between 1 and {capacity}")

    width, height = settings["grid_size"]
    solvers: dict[tuple[int, int], ExactSolver] = {}
    contexts = []
    for index, prefix in enumerate(islice(prefixes, count)):
        # Each first action crosses all detour/budget pairs over consecutive
        # prefix blocks. Independently rotate the public rejection lockout.
        block, first_index = divmod(index, len(first_actions))
        detour = detours[(first_index + block) % len(detours)]
        budget_offset = (first_index + block // len(detours)) % 3
        lockout_index = (first_index + block + block // len(detours)) % len(lockouts)
        lockout = lockouts[-1 - lockout_index]
        scenario_id = f"dev-history-{index:04d}"
        config = EnvConfig(
            grid_width=width, grid_height=height, max_env_steps=settings["max_env_steps"],
            initial_mechanic=initial_rule, candidate_rules=rules, shift_targets=targets,
            step_cost=settings["step_cost"], delivery_reward=settings["delivery_reward"],
            probe_detour_steps=detour, lockout_steps=lockout,
            feedback_noise=noise, shift_hazard=hazard,
        )
        env = RuleShiftEnv(config)
        env.reset(options={"scenario_id": scenario_id, "master_seed": dev["seeds"]["master"]})
        microtask = replace(family_a_microtask(
            probe_detour_steps=detour, commit_travel_steps=width - 1,
            feedback_noise=noise, lockout_steps=lockout, shift_hazard=hazard,
            step_cost=config.step_cost, reward_success=config.delivery_reward,
        ), rules=rules, initial_rule=initial_rule, shift_targets=targets)
        belief = {(initial_rule, False): 1.0}
        position, remaining_lockout = env.geometry.start, 0
        prefix_events: list[tuple[str, str]] = []
        for action in prefix:
            _, _, _, truncated, info = env.step(action)
            if truncated:
                raise ValueError(f"prefix {prefix} exhausts configured primitive budget")
            public = env.public_view()
            observed = public.public_text.rsplit("last=", 1)[1]
            prefix_events.append((action, observed))
            branches = macro_branches(microtask, belief, action, position, remaining_lockout)
            branch = next(branch for branch in branches if branch.observation == observed)
            belief = branch.posterior
            position = branch.position
            remaining_lockout = branch.lockout_remaining

        evaluator_state = env.evaluator_state()
        # This is an explicitly budgeted local decision at the reached state.
        # The full episode may have more time than this diagnostic lookahead.
        horizon = min(6 + budget_offset, config.max_env_steps - info["physical_steps_elapsed"])
        # Disclose the generative rates, not the realized hidden rule/shift: the
        # exact Q* is then the Bayes-optimal benchmark for the agent's own
        # information, rather than an oracle-parameter optimum.
        sensor = (
            "Station tests are always correct."
            if noise <= 0.0
            else f"Station tests are correct with probability {1.0 - noise:.2f}; otherwise they flip."
        )
        switch = (
            "No switch can occur."
            if hazard <= 0.0
            else f"Each primitive step has probability {hazard:.2f} of that one-time switch."
        )
        public = replace(public, public_text=(
            f"The gate initially accepted {initial_rule} and can switch once without warning. "
            f"{switch} {sensor} Maximize total reward over the next "
            f"{horizon} primitive steps. {public.public_text}"
        ))
        solver_key = (detour, lockout)
        if solver_key not in solvers:
            solvers[solver_key] = ExactSolver(microtask, FAMILY_A_ACTIONS)
        solver = solvers[solver_key]
        values = dict(solver.q_values(freeze_belief(belief), horizon, position, remaining_lockout))
        contexts.append(OfflineContext(
            context_id=scenario_id, scenario_id=scenario_id, config=microtask,
            belief=belief, q_values=values, public_observation=public, horizon=horizon,
            delta_probe=solver.delta_probe(belief, horizon, position, remaining_lockout),
            prior=1.0,
            feedback_noise=noise,
            prefix=tuple(prefix_events),
            evaluator_meta={
                "design": "history",
                "initial_rule": initial_rule,
                "shift_hazard": hazard,
                "feedback_noise": noise,
                "true_rule": evaluator_state["active_mechanic"],
                "shift_time": evaluator_state["shift_time"],
                "steps_elapsed": info.get("physical_steps_elapsed"),
            },
        ))
    return tuple(contexts)
