"""Explicit semantic errors bypass motor retries without relaxing normal delegation."""

import pytest

from rpent.harness import Harness
from rpent.harness.lifecycle import consumes_physical_attempt
from rpent.planning.schemas import StepVerdict
from tests.test_harness_1011_control import SINK, act, machine, named, step
from tests.test_robocasa_preclose_review import action, call, setup


@pytest.mark.parametrize("kind", ["wrong_target", "wrong_destination", "wrong_allocation"])
def test_first_semantic_error_escalates_without_becoming_progress(kind):
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**SINK)])["ok"]
    act(state, 20, verdict="retry", failure_type=kind,
        raw={"grasp_status": {"retention_monitor_active": True}})
    verified = named(events, "step_verified")[-1]
    assert verified["progress_signal"] is False
    assert named(events, "control_mode_escalation_available")[-1]["reason"] == "task_conflict"


def test_semantic_error_overrides_progress_verdict_and_distance():
    state = machine()
    verdict = StepVerdict("progress", ("knife, not peeler",), "wrong object",
                          failure_type="wrong_target", metrics={"target_distance_m": 0.1})
    ledger = {"best_distance": 0.5}
    assert state._progress_signal(ledger, verdict, {"holding": True}) is False
    assert ledger["best_distance"] == 0.5


def test_normal_missed_grasp_escalates_after_first_physical_failure():
    events = []
    state = machine(events)
    state.register_plan("goal", [step(**SINK)])
    act(state, 20, verdict="retry", failure_type="grasp_missed")
    assert not named(events, "control_mode_escalation_available")
    act(state, 20, verdict="retry", failure_type="grasp_missed")
    escalation = named(events, "control_mode_escalation_available")[-1]
    assert escalation["reason"] == "failures"
    assert escalation["failed_verifications"] == 2
    assert not state.needs_replan


def task_review(toolkit, *, phase="grasp", verdict="conflict", **extra):
    observed = call(toolkit, "observe")
    return call(toolkit, "review_task_action", expected_step=observed["step"],
                phase=phase, verdict=verdict, expected_target="peeler",
                observed_target="knife" if verdict == "conflict" else "peeler",
                evidence="Current side and wrist views identify the selected target.", **extra)


def test_semantic_veto_survives_alignment_rejections_and_blocks_policy():
    toolkit, simulator, _ = setup(*[[action(close=True)]] * 4)
    task_review(toolkit, conflict_type="wrong_target")
    toolkit._preclose_rejections = 20
    for _ in range(3):
        result = call(toolkit, "continue_vla", max_chunks=1)
        assert result["stop_reason"] == "task_conflict_review_required"
    assert simulator.actions == []
    task_review(toolkit, verdict="aligned")
    assert toolkit.observe().get("task_action_review", {}).get("conflict") is None


def test_stale_review_cannot_raise_or_clear_a_veto():
    toolkit, _, _ = setup([action(close=True)])
    call(toolkit, "observe")
    result = toolkit.execute_tool("review_task_action", {
        "expected_step": 1, "phase": "grasp", "verdict": "conflict",
        "conflict_type": "wrong_target", "expected_target": "peeler",
        "observed_target": "knife", "evidence": "stale image",
    }).result
    assert "error" in result and "fresh" in result["error"]


def test_watched_release_stops_inside_chunk_before_opening():
    toolkit, simulator, policy = setup([action(close=True), action(dz=0.1)], [action(dz=1)])
    # A previously held object needs no acquisition review.
    toolkit._last_executed_gripper = "close"
    task_review(toolkit, phase="release", verdict="watch")
    result = call(toolkit, "continue_vla", max_chunks=1)
    assert result["stop_reason"] == "prerelease_review_required"
    assert result["task_action_review"]["release_watch"]["expected_target"] == "peeler"
    assert result["task_action_review"]["candidate"]["commanded_eef_delta_base_m"][2] == pytest.approx(0.005)
    assert len(simulator.actions) == 1 and simulator.actions[0][11] == 1
    task_review(toolkit, phase="release", verdict="aligned")
    call(toolkit, "continue_vla", max_chunks=1)
    assert simulator.actions[-1][11] < 0.5
    assert len(policy.inputs) == 1  # resume the saved candidate, never resample
    assert simulator.actions[-1][7] == pytest.approx(0.1)
    assert toolkit._operation.action_history_since()[-1]["control_source"] == "vla"
    assert not toolkit._task_release_watch


def test_unwatched_release_keeps_original_policy_execution():
    toolkit, simulator, _ = setup([action()])
    toolkit._last_executed_gripper = "close"
    result = call(toolkit, "continue_vla", max_chunks=1)
    assert result["executed_actions"] == 1
    assert simulator.actions[0][11] < 0.5


def test_uncertainty_cannot_erase_a_semantic_veto():
    toolkit, simulator, _ = setup([action(close=True)])
    task_review(toolkit, conflict_type="wrong_target")
    task_review(toolkit, verdict="uncertain")
    assert toolkit._current_task_conflict()["conflict_type"] == "wrong_target"
    assert call(toolkit, "continue_vla", max_chunks=1)["executed_actions"] == 0
    assert not simulator.actions


def test_zero_action_conflict_escalates_and_preserves_verification_and_budgets():
    events = []
    state = machine(events)
    state.register_plan("goal", [step(**SINK)])
    state.report_task_conflict({"phase": "grasp", "conflict_type": "wrong_target",
                                "expected_target": "peeler", "observed_target": "knife", "evidence": "fresh wrist"})
    assert state.can_replan
    ledger = next(iter(state._subtasks.values()))
    assert ledger["failures"] == {} and ledger["policy_steps"] == 0
    assert ledger["segments"] == 0 and ledger["fallbacks"] == 0
    assert named(events, "control_mode_escalation_available")[-1]["reason"] == "task_conflict"


@pytest.mark.parametrize("reason", ["prerelease_review_required", "task_conflict_review_required"])
def test_zero_action_review_pause_is_not_a_physical_attempt(reason):
    assert not consumes_physical_attempt(tool_name="continue_vla", execution_status="completed", result={
        "stop_reason": reason, "review_required": True, "executed_actions": 0,
        "requested_actions": 2, "cancelled_actions": 2, "transitions_used": 0,
        "environment": {"success": False, "terminated": False, "truncated": False},
    })


def test_release_conflict_blocks_opening_but_not_closed_transport():
    toolkit, simulator, _ = setup([action()])
    toolkit._last_executed_gripper = "close"
    task_review(toolkit, phase="release", verdict="conflict", conflict_type="wrong_allocation")
    with pytest.raises(ValueError, match="task conflict"):
        toolkit._check_task_gripper("open")
    toolkit._check_task_gripper("close")
    assert call(toolkit, "continue_vla", max_chunks=1)["executed_actions"] == 0
    assert not simulator.actions


def test_replanning_same_operation_does_not_remove_veto(tmp_path):
    toolkit, _, _ = setup([action(close=True)])
    state = machine()
    state.register_plan("goal", [step(**SINK, operation="select_tool", allowed_tools=["continue_vla"])])
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    harness.state = state
    toolkit.attach_harness(harness)
    task_review(toolkit, conflict_type="wrong_target")
    result = state.replan("repair", [step(**SINK, id="new", operation="select_tool", allowed_tools=["continue_vla"])])
    assert result.get("ok"), result
    assert toolkit._current_task_conflict()["conflict_type"] == "wrong_target"


def test_stale_release_candidate_is_not_executed_after_correction():
    toolkit, simulator, _ = setup([action()], [action()])
    toolkit._last_executed_gripper = "close"
    task_review(toolkit, phase="release", verdict="watch")
    call(toolkit, "continue_vla", max_chunks=1)
    task_review(toolkit, phase="release", verdict="aligned")
    # A confirmed physical change invalidates an approval, including a move at
    # the same apparent pose: the environment step is the freshness boundary.
    toolkit._operation._metrics["step_count"] += 1
    with pytest.raises(ValueError, match="stale"):
        toolkit._take_reviewed_release()
    assert not simulator.actions


def test_veto_preserves_safe_holding_and_open_gripper_recovery():
    toolkit, _, _ = setup([action()])
    toolkit._last_executed_gripper = "close"
    task_review(toolkit, conflict_type="wrong_target")
    toolkit._check_task_gripper("close")  # preserve an existing hold while returning the wrong item
    toolkit._last_executed_gripper = "open"
    with pytest.raises(ValueError, match="task conflict"):
        toolkit._check_task_gripper("close")
    task_review(toolkit, verdict="aligned")
    task_review(toolkit, phase="release", verdict="conflict", conflict_type="wrong_destination")
    toolkit._check_task_gripper("open")  # already released; open approach must remain possible
