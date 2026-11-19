"""Tests for immutable, tracker-owned recovery decisions and budgets."""

from dataclasses import FrozenInstanceError, replace

import pytest

from rpent.harness.recovery import RecoveryPolicy, RecoveryTracker


def _tracker():
    return RecoveryTracker(
        RecoveryPolicy(
            local_replans_per_outcome=1,
            stage_replans_per_stage=1,
            remaining_stage_repartitions_per_run=1,
            max_total_physical_actions_per_run=2,
        )
    )


def test_tracker_selects_scope_and_escalates_only_after_consumption():
    tracker = _tracker()
    local = tracker.decide("outcome-1", "stage-1", False, 0)
    assert local.scope == "replan_outcome"
    tracker.consume(local, outcome_id="outcome-1", stage_id="stage-1")
    assert tracker.snapshot()["pending_recovery_action"] == {
        "scope": "replan_outcome",
        "outcome_id": "outcome-1",
        "stage_id": "stage-1",
    }

    tracker.record_physical_action()
    assert tracker.snapshot()["pending_recovery_action"] is None
    assert tracker.snapshot()["physical_actions_consumed"] == 1

    stage = tracker.decide("outcome-1", "stage-1", False, 0)
    assert stage.scope == "replan_stage"
    tracker.consume(stage, outcome_id="outcome-1", stage_id="stage-1")

    global_ = tracker.decide("outcome-1", "stage-1", False, 0)
    assert global_.scope == "repartition_remaining_stages"
    tracker.consume(global_, outcome_id="outcome-1", stage_id="stage-1")
    assert tracker.decide("outcome-1", "stage-1", False, 0).scope == "terminal_failure"


def test_decisions_are_immutable_and_cannot_be_forged_or_reused():
    tracker = _tracker()
    decision = tracker.decide("outcome-1", "stage-1", False, 0)

    with pytest.raises(FrozenInstanceError):
        decision.scope = "replan_stage"  # type: ignore[misc]
    with pytest.raises(ValueError, match="issued"):
        tracker.consume(
            replace(decision, scope="replan_stage"),
            outcome_id="outcome-1",
            stage_id="stage-1",
        )

    tracker.consume(decision, outcome_id="outcome-1", stage_id="stage-1")
    with pytest.raises(ValueError, match="already consumed"):
        tracker.consume(decision, outcome_id="outcome-1", stage_id="stage-1")


def test_consume_rejects_context_mismatch_and_terminal_scope():
    tracker = _tracker()
    decision = tracker.decide("outcome-1", "stage-1", False, 0)
    with pytest.raises(ValueError, match="does not match"):
        tracker.consume(decision, outcome_id="outcome-2", stage_id="stage-1")

    for _ in range(2):
        tracker.record_physical_action()
    terminal = tracker.decide("outcome-1", "stage-1", True, 1)
    assert terminal.scope == "terminal_failure"
    with pytest.raises(ValueError, match="cannot be consumed"):
        tracker.consume(terminal, outcome_id="outcome-1", stage_id="stage-1")


def test_safe_step_retry_and_action_budget_fail_closed():
    tracker = _tracker()
    retry = tracker.decide("outcome-1", "stage-1", True, 1)
    assert retry.scope == "retry_step"
    tracker.consume(retry, outcome_id="outcome-1", stage_id="stage-1")

    tracker.record_physical_action()
    tracker.record_physical_action()
    with pytest.raises(ValueError, match="exhausted"):
        tracker.record_physical_action()
    assert tracker.decide("outcome-1", "stage-1", True, 1).scope == "terminal_failure"


def test_budgets_are_scoped_per_outcome_and_stage():
    tracker = _tracker()
    first = tracker.decide("outcome-1", "stage-1", False, 0)
    tracker.consume(first, outcome_id="outcome-1", stage_id="stage-1")

    assert tracker.decide("outcome-2", "stage-1", False, 0).scope == "replan_outcome"
    assert tracker.decide("outcome-1", "stage-2", False, 0).scope == "replan_stage"


def test_cancel_discards_outstanding_decision_without_consuming_budget():
    tracker = _tracker()
    decision = tracker.decide("outcome-1", "stage-1", False, 0)
    snapshot = tracker.snapshot()

    assert snapshot["outstanding_decisions"] == [
        {
            "decision_id": decision.decision_id,
            "scope": "replan_outcome",
            "outcome_id": "outcome-1",
            "stage_id": "stage-1",
            "reason": "outcome recovery budget remains",
        }
    ]

    tracker.cancel(decision, outcome_id="outcome-1", stage_id="stage-1")

    assert tracker.snapshot()["outstanding_decisions"] == []
    assert tracker.remaining_for("outcome-1", "stage-1") == {
        "local_remaining": 1,
        "stage_remaining": 1,
        "repartition_remaining": 1,
    }
    with pytest.raises(ValueError, match="cancelled"):
        tracker.consume(decision, outcome_id="outcome-1", stage_id="stage-1")


def test_cancel_is_idempotent_for_the_same_owned_decision_and_context():
    tracker = _tracker()
    decision = tracker.decide("outcome-1", "stage-1", False, 0)

    tracker.cancel(decision, outcome_id="outcome-1", stage_id="stage-1")
    tracker.cancel(decision, outcome_id="outcome-1", stage_id="stage-1")

    assert tracker.snapshot()["outstanding_decisions"] == []


def test_cancel_rejects_foreign_mismatched_and_consumed_decisions():
    tracker = _tracker()
    foreign_tracker = _tracker()
    foreign = foreign_tracker.decide("outcome-1", "stage-1", False, 0)
    with pytest.raises(ValueError, match="issued"):
        tracker.cancel(foreign, outcome_id="outcome-1", stage_id="stage-1")

    decision = tracker.decide("outcome-1", "stage-1", False, 0)
    with pytest.raises(ValueError, match="does not match"):
        tracker.cancel(decision, outcome_id="outcome-2", stage_id="stage-1")
    assert len(tracker.snapshot()["outstanding_decisions"]) == 1

    tracker.consume(decision, outcome_id="outcome-1", stage_id="stage-1")
    with pytest.raises(ValueError, match="already consumed"):
        tracker.cancel(decision, outcome_id="outcome-1", stage_id="stage-1")
