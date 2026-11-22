"""History-only soft belief updates: no simulator, no evaluator, no oracle."""

from __future__ import annotations

import pytest

from ruleshift.agents.consistency import HistoryConsistencyUpdater


def test_accept_raises_probability_of_the_prob_color() -> None:
    updater = HistoryConsistencyUpdater(("red", "blue", "green"), learning_rate=0.5)
    belief = {"red": 0.5, "blue": 0.5, "green": 0.0}
    updated = updater.update(belief, "PROBE_RED", "accept")
    assert updated["red"] > belief["red"]
    assert updated["blue"] < belief["blue"]
    assert sum(updated.values()) == pytest.approx(1.0)


def test_reject_lowers_probability_of_the_prob_color() -> None:
    updater = HistoryConsistencyUpdater(("red", "blue", "green"), learning_rate=0.5)
    belief = {"red": 0.5, "blue": 0.25, "green": 0.25}
    updated = updater.update(belief, "PROBE_RED", "reject")
    assert updated["red"] < belief["red"]
    assert sum(updated.values()) == pytest.approx(1.0)


def test_commit_outcomes_update_the_same_way() -> None:
    updater = HistoryConsistencyUpdater(("red", "blue", "green"), learning_rate=0.5)
    belief = {"red": 0.5, "blue": 0.5, "green": 0.0}
    assert updater.update(belief, "COMMIT_RED", "accept")["red"] > 0.5
    assert updater.update(belief, "COMMIT_RED", "reject")["red"] < 0.5


def test_uninformative_actions_leave_belief_unchanged() -> None:
    updater = HistoryConsistencyUpdater(("red", "blue", "green"))
    belief = {"red": 0.5, "blue": 0.5, "green": 0.0}
    assert updater.update(belief, "WAIT", "none") == belief
    assert updater.update(belief, "GO_TO_STATION", "none") == belief


def test_zero_learning_rate_is_no_op_and_does_not_mutate_input() -> None:
    updater = HistoryConsistencyUpdater(("red", "blue", "green"), learning_rate=0.0)
    belief = {"red": 0.5, "blue": 0.5, "green": 0.0}
    assert updater.update(belief, "PROBE_RED", "accept") == belief
    assert belief == {"red": 0.5, "blue": 0.5, "green": 0.0}
