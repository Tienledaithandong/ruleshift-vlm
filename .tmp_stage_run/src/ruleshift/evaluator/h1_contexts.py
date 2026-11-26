"""Balanced H1 development contexts with disclosed generative parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ruleshift.envs.config import EnvConfig, load_dev_grid
from ruleshift.envs.gate_lock import RuleShiftEnv
from ruleshift.evaluator.exact_solver import ExactSolver, freeze_belief
from ruleshift.evaluator.offline_audit import (
    ACTIONS,
    HORIZON,
    RULE_PAIRS,
    OfflineContext,
    _microtask_config,
)
from ruleshift.public_api.interface import PublicObservation

#: H1 diagnostic factors. Noise and prior are disclosed in the public text so
#: the exact label is the true Bayes-optimal action under the agent's
#: information, not an oracle-parameter optimum.
H1_PRIORS = (0.5, 0.8)
#: The dev grid stops at a five-step detour; H1 adds an expensive arm so the
#: negative-delta region is represented instead of being a rare corner.
H1_DETOURS = (0, 2, 5, 8)
#: Budget is part of the cost factor: a short horizon can make a probe too
#: expensive even when its information is valuable.
H1_HORIZONS = (12, HORIZON)


@dataclass(frozen=True)
class _H1Candidate:
    """Delta-labelled candidate; exact Q values are computed only if selected."""

    context: OfflineContext
    solver: ExactSolver


def _h1_description(
    initial: PublicObservation,
    pair: tuple[str, str],
    prior: float,
    noise: float,
    horizon: int,
    detour: int,
    lockout: int,
) -> str:
    if abs(prior - 0.5) < 1e-12:
        prior_text = (
            f"An unseen fair draw selected either {pair[0]} or {pair[1]} "
            "with equal probability."
        )
    else:
        prior_text = (
            f"An unseen draw selected {pair[0]} with probability {prior:.2f} "
            f"and {pair[1]} with probability {1.0 - prior:.2f}."
        )
    sensor = (
        "The station always reports correctly."
        if noise <= 0.0
        else (
            f"The station reports correctly with probability {1.0 - noise:.2f}; "
            "otherwise it flips the label."
        )
    )
    return (
        f"Controlled card lottery with a disclosed prior. {prior_text} "
        f"The rule stays fixed for this task. The station detour costs {detour} "
        f"extra primitive steps; a rejected commit locks the gate for {lockout} "
        f"steps. {sensor} You have {horizon} primitive steps to maximize total "
        f"delivery reward including step costs. {initial.public_text}"
    )


def _h1_context(
    *,
    detour: int,
    lockout: int,
    noise: float,
    prior: float,
    pair: tuple[str, str],
    horizon: int,
    environment: dict,
    solvers: dict[tuple[int, int, float], ExactSolver],
) -> _H1Candidate:
    env = RuleShiftEnv(config=EnvConfig(
        probe_detour_steps=detour,
        feedback_noise=noise,
        lockout_steps=lockout,
        disable_shift=True,
        max_env_steps=horizon,
        grid_width=environment["grid_size"][0],
        grid_height=environment["grid_size"][1],
        step_cost=environment["step_cost"],
        delivery_reward=environment["delivery_reward"],
    ))
    env.reset(seed=0)
    initial = env.public_view()
    description = _h1_description(initial, pair, prior, noise, horizon, detour, lockout)
    observation = replace(
        initial,
        public_text=description,
        candidate_rules=tuple(
            rule for rule in initial.candidate_rules if rule.rule_id in pair
        ),
    )
    key = (detour, lockout, noise)
    if key not in solvers:
        config = _microtask_config(
            detour, int(initial.public_costs["commit_travel"]), noise, lockout
        )
        config = replace(
            config,
            step_cost=environment["step_cost"],
            reward_success=environment["delivery_reward"],
        )
        solvers[key] = ExactSolver(config, ACTIONS)
    solver = solvers[key]
    belief = {(pair[0], False): prior, (pair[1], False): 1.0 - prior}
    scenario_id = (
        f"h1-d{detour}-l{lockout}-n{noise:.2f}-p{prior:.2f}-h{horizon}-"
        f"{pair[0]}-{pair[1]}"
    )
    context = OfflineContext(
        context_id=scenario_id,
        scenario_id=scenario_id,
        config=solver.config,
        belief=belief,
        delta_probe=solver.delta_probe(belief, horizon),
        q_values={},
        public_observation=observation,
        horizon=horizon,
        prior=prior,
        feedback_noise=noise,
        evaluator_meta={
            "design": "h1",
            "pair": list(pair),
            "detour": detour,
            "lockout": lockout,
            "horizon": horizon,
        },
    )
    return _H1Candidate(context=context, solver=solver)



def _select_h1_contexts(
    candidates: list[_H1Candidate], count: int
) -> tuple[OfflineContext, ...]:
    if count > len(candidates):
        raise ValueError(f"requested {count} H1 contexts but only {len(candidates)} exist")

    def materialize(
        rows: list[_H1Candidate], delta_bin: str
    ) -> tuple[OfflineContext, ...]:
        return tuple(
            replace(
                row.context,
                q_values=dict(
                    row.solver.q_values(
                        freeze_belief(row.context.belief), row.context.horizon
                    )
                ),
                evaluator_meta={**row.context.evaluator_meta, "delta_bin": delta_bin},
            )
            for row in rows
        )

    if count < 3:
        return materialize(
            sorted(candidates, key=lambda c: c.context.delta_probe)[:count], "marginal"
        )

    ordered = sorted(candidates, key=lambda c: c.context.delta_probe)
    third = count // 3
    middle_count = count - 2 * third
    negative = ordered[:third]
    positive = ordered[-third:]
    middle = sorted(
        ordered[third:len(ordered) - third],
        key=lambda c: abs(c.context.delta_probe),
    )[:middle_count]
    return (
        materialize(negative, "negative")
        + materialize(middle, "marginal")
        + materialize(positive, "positive")
    )


def build_h1_contexts(
    count: int = 81, config_path: Path | None = None
) -> tuple[OfflineContext, ...]:
    """Balanced H1 set: cost x disclosed noise x disclosed prior.

    The rule and the (absent) shift remain hidden. Selecting contexts by exact
    ``delta_probe`` ensures the pilot spans negative, marginal and positive
    decisions instead of inheriting the natural skew of one factor rotation.
    """

    dev = load_dev_grid(config_path)
    if dev["split"] != "dev" or dev["environment"]["family"] != "gate_lock":
        raise ValueError("H1 microtasks require the Family A dev configuration")
    settings, grid = dev["environment"], dev["grid"]
    detours = H1_DETOURS
    lockouts = tuple(int(value) for value in settings["lockout_steps_options"])
    noises = tuple(
        sorted({float(value) for value in grid["feedback_noise_rate"] if 0.0 <= value < 0.5})
    )
    if not detours or not lockouts or not noises:
        raise ValueError("H1 design needs detour, lockout and noise values")
    if len(lockouts) < 3 or len(detours) < 3:
        raise ValueError("H1 design needs at least three detour and lockout levels")

    if not 1 <= count <= 81:
        raise ValueError("H1 design supports between 1 and 81 decisions")

    # A fractional 81-cell design: detour 0 supplies the positive/marginal
    # region; detours 2, 5 and 8 supply the negative region. This is four
    # times cheaper than evaluating the full 4x3x3x2x2x3 factorial while
    # still covering every declared cost, noise, prior, horizon and pair.
    specs: list[tuple[int, int, float, float, tuple[str, str], int]] = []
    for noise in noises:
        for lockout in lockouts:
            for pair in RULE_PAIRS:
                specs.append((0, lockout, noise, 0.5, pair, 24))
    for noise in noises:
        for lockout in lockouts:
            for pair in RULE_PAIRS:
                specs.append((0, lockout, noise, 0.8, pair, 12))
    for detour in (2, 5, 8):
        for pair in RULE_PAIRS:
            for index, noise in enumerate(noises):
                offset = RULE_PAIRS.index(pair) * 3 + index
                specs.append((
                    detour,
                    lockouts[offset % len(lockouts)],
                    noise,
                    H1_PRIORS[(offset // 3) % len(H1_PRIORS)],
                    pair,
                    H1_HORIZONS[(offset // 6) % len(H1_HORIZONS)],
                ))

    candidates: list[_H1Candidate] = []
    solvers: dict[tuple[int, int, float], ExactSolver] = {}
    for detour, lockout, noise, prior, pair, horizon in specs:
        candidates.append(_h1_context(
            detour=detour,
            lockout=lockout,
            noise=noise,
            prior=prior,
            pair=pair,
            horizon=horizon,
            environment=settings,
            solvers=solvers,
        ))
    return _select_h1_contexts(candidates, count)
