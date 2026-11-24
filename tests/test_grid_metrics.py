from types import SimpleNamespace

from ruleshift.evaluator.grid_metrics import audit_grid, paired_grid_ci, summarize_grid
from ruleshift.evaluator.grid_reference import oracle_return
from ruleshift.evaluator.grid_runner import GridEpisode


def episode(scenario_id, seed, success, cost, false_alarm=False, adaptation=None,
            scenario_type="change_detection"):
    return GridEpisode(
        scenario_id=scenario_id, run_seed=seed, method="m", success=success,
        total_return=10.0 if success else 0.0, env_steps=cost,
        tool_units=0, advisor_calls=0, total_cost=float(cost),
        adaptation_steps=adaptation, false_alarm=false_alarm,
        scenario_type=scenario_type,
        final_rule_estimate="base", public_rows=(), evaluator_rows=(),
    )


def test_summary_counts_success_and_false_alarms():
    episodes = [
        episode("a", 0, True, 10),
        episode("a", 1, False, 20, false_alarm=True, scenario_type="no_shift"),
        episode("b", 0, True, 15, scenario_type="no_shift"),
    ]
    summary = summarize_grid(episodes)
    assert summary["n_episodes"] == 3
    assert abs(summary["success_rate"] - 2 / 3) < 1e-9
    assert abs(summary["false_alarm_rate"] - 0.5) < 1e-9


def test_paired_ci_resamples_scenarios_not_seeds():
    method = [
        episode("a", 0, True, 10), episode("a", 1, True, 12),
        episode("b", 0, True, 20), episode("b", 1, True, 22),
    ]
    baseline = [
        episode("a", 0, False, 30), episode("a", 1, False, 32),
        episode("b", 0, False, 40), episode("b", 1, False, 42),
    ]
    summary = paired_grid_ci(method, baseline, n_boot=200)
    assert summary.n_scenarios == 2
    assert summary.mean_difference < 0


def test_q_star_microtask_is_a_depth_limited_upper_bound():
    from ruleshift.agents.grid_belief import GridBelief
    from ruleshift.evaluator.grid_reference import q_star_microtask
    from ruleshift.public_api.grid_rules import BASE_RULE, PublicGridMap

    public = PublicGridMap(5, 5, frozenset(), (1, 1),
                           {"red": (1, 3), "blue": (3, 3), "green": (3, 1)},
                           None, None, None)
    belief = GridBelief(hypotheses=("base",))
    value = q_star_microtask(belief, public, (1, 1), horizon=2)
    assert 0.0 < value <= 11.0   # goal reward 10 minus at most two step costs


def test_summary_omits_unmeasured_keys_instead_of_nan():
    assert summarize_grid([]) == {"n_episodes": 0}
    summary = summarize_grid([episode("a", 0, True, 10)])
    for key in ("mean_adaptation_steps", "mean_regret", "brier_score",
                "expected_calibration_error", "mean_hazard_damage", "deaths",
                "tool_precision", "tool_recall", "mean_latency_ms"):
        assert key not in summary


def test_oracle_return_reads_the_optional_field():
    from dataclasses import replace

    plain = episode("a", 0, True, 10)
    assert oracle_return(plain) is None
    assert oracle_return(replace(plain, oracle_return=9.5)) == 9.5


def _with_fields(ep, **fields):
    """Duck-typed copy so a missing production field fails as an assertion."""

    data = dict(vars(ep))
    data.update(fields)
    return SimpleNamespace(**data)


def test_information_gain_uses_tool_calls_and_omits_unmeasured():
    unmeasured = episode("a", 0, True, 10)
    summary = summarize_grid([unmeasured])
    assert "information_gain_per_tool_call" not in summary
    assert "information_gain_per_total_cost" not in summary

    # 3 bits came from 2 tool calls; the other bit came from a move. The
    # per-tool metric must use only the tool-generated information.
    measured = _with_fields(
        unmeasured,
        information_gain_bits=4.0,
        tool_information_gain_bits=3.0,
        move_information_gain_bits=1.0,
        tool_calls_made=2,
        tool_units=4,
        public_rows=({"chosen_action": "UP"}, {"chosen_action": "INSPECT"}),
    )
    summary = summarize_grid([measured])
    assert abs(summary["information_gain_per_tool_call"] - 1.5) < 1e-9
    assert abs(summary["information_gain_per_total_cost"] - 4.0 / 10.0) < 1e-9
    assert summary["tool_information_gain_bits"] == 3.0
    assert summary["move_information_gain_bits"] == 1.0
    assert abs(summary["information_gain_per_decision"] - 2.0) < 1e-9


def test_move_only_information_gain_is_not_counted_as_tool_information():
    move_only = _with_fields(
        episode("a", 0, True, 10),
        information_gain_bits=2.0,
        tool_information_gain_bits=0.0,
        move_information_gain_bits=2.0,
        tool_calls_made=0,
        public_rows=({"chosen_action": "UP"}, {"chosen_action": "RIGHT"}),
    )
    summary = summarize_grid([move_only])
    assert "information_gain_per_tool_call" not in summary
    assert summary["tool_information_gain_bits"] == 0.0
    assert summary["move_information_gain_bits"] == 2.0
    assert abs(summary["information_gain_per_decision"] - 1.0) < 1e-9


def test_audit_grid_reports_every_structural_check():
    from ruleshift.envs.grid.config import GridConfig

    result = audit_grid(GridConfig(), n_maps=2)
    assert result["passed"] is True
    assert result["failures"] == []
    assert set(result["checks"]) == {
        "generator_solvability",
        "render_rule_independence",
        "public_observation_no_leak",
        "tool_accounting",
        "sign_partiality",
        "permissions_scan",
    }
    assert all(result["checks"].values())


def test_audit_generator_solvability_is_independent_of_the_generator_gate(monkeypatch):
    """A map the old gate wrongly accepted must still fail the audit."""

    from ruleshift.envs.grid.config import GridConfig
    from ruleshift.envs.grid.entities import GridMap, MapGenResult
    from ruleshift.public_api.grid_rules import PublicGridMap

    bad_map = PublicGridMap(
        width=5,
        height=5,
        walls=frozenset((x, y) for x in range(5) for y in range(1, 5)),
        start=(0, 0),
        markers={"red": (4, 0), "blue": (0, 4), "green": (3, 4)},
        key=None,
        door=None,
        signpost=(2, 0),
    )
    generated = MapGenResult(
        map=GridMap(public=bad_map, map_seed=0, door_required=False),
        attempts=1,
    )
    monkeypatch.setattr(
        "ruleshift.evaluator.grid_metrics.generate_solvable_map",
        lambda seed, config: generated,
    )
    # Simulate the old, tautological gate saying "yes" to every map.
    monkeypatch.setattr(
        "ruleshift.evaluator.grid_metrics.all_hypotheses_solvable",
        lambda *args, **kwargs: True,
        raising=False,
    )

    result = audit_grid(GridConfig(), n_maps=1)

    assert result["checks"]["generator_solvability"] is False
