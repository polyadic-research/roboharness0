"""The six review rules cannot preempt the preserved whole-task policy stream."""

import numpy as np
import pytest

from tests.test_harness_1011_control import act
from tests.test_robocasa_preclose_review import TASK, action
from tests.test_unmeasured_composite_policy_first import (
    INSTRUCTION,
    bound_toolkit,
    machine,
    whole,
)


@pytest.mark.parametrize("verdict,phase,kind", [
    ("conflict", "grasp", "wrong_target"),
    ("conflict", "release", "wrong_allocation"),
    ("watch", "release", None),
])
def test_task_reviews_cannot_arm_a_veto_or_release_watch_before_recovery(tmp_path, verdict, phase, kind):
    toolkit, simulator, policy, harness = bound_toolkit(tmp_path, [action(close=True), action()])
    toolkit.execute_tool("observe", {})
    arguments = {"expected_step": 0, "phase": phase, "verdict": verdict,
                 "expected_target": "peeler", "observed_target": "knife",
                 "evidence": "Public target assessment; no executed physical failures."}
    if kind:
        arguments["conflict_type"] = kind
    result = toolkit.execute_tool("review_task_action", arguments).result
    if verdict == "conflict":
        assert "error" not in result, result
        assert toolkit._current_task_conflict() is not None
        assert harness.state.escalated_operations()
    else:
        assert "whole-task" in result.get("error", "")
        assert toolkit._current_task_conflict() is None
        assert not harness.state.escalated_operations()
    assert toolkit._task_release_watch is None
    assert simulator.actions == [] and policy.inputs == []


def test_explicit_conflict_state_entry_opens_bounded_recovery_without_motion():
    state = machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    state.report_task_conflict({"phase": "grasp", "conflict_type": "wrong_target",
                                "expected_target": "peeler", "observed_target": "knife",
                                "evidence": "fresh public image explicitly identifies wrong target"})
    assert state._ledger(state.active_step).get("task_conflict")
    assert state.escalated_operations()
    assert state.needs_replan


def test_first_executed_semantic_failure_unlocks_recovery_without_an_unreviewed_veto():
    state = machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="wrong_target")
    assert not state.whole_task_policy_first_active
    assert state._ledger(state.active_step).get("task_conflict")
    assert state.escalated_operations()
    assert state.needs_replan
    assert state.replan("recover first verified failure", [whole(
        control_mode="takeover", allowed_tools=["move_eef_path"],
    )])["ok"]


def test_even_stale_review_state_cannot_cancel_or_substitute_protected_actions(tmp_path):
    chunks = [action(close=True), action()]
    toolkit, simulator, policy, harness = bound_toolkit(tmp_path, chunks)
    key = toolkit._task_scope()
    toolkit._task_conflicts[key] = {"phase": "grasp", "conflict_type": "wrong_target"}
    toolkit._task_release_watch = {"scope": key, "expected_target": "different plate"}
    toolkit._last_executed_gripper = "close"
    toolkit.execute_tool("observe", {})
    result = toolkit.execute_tool("continue_vla", {"max_chunks": 1}).result
    assert "error" not in result, result
    assert result["executed_actions"] == 2
    assert result["paused_actions"] == result["cancelled_actions"] == 0
    np.testing.assert_array_equal(simulator.actions, chunks)
    assert len(policy.inputs) == 1 and policy.inputs[0][1] == TASK
    assert harness.state.whole_task_policy_first_active


def test_first_physical_failure_unlocks_six_change_conflict_review(tmp_path):
    toolkit, _, _, harness = bound_toolkit(tmp_path, [action(close=True)])
    state = harness.state
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert state.whole_task_policy_first_active
    assert state.semantic_policy_replanning_allowed
    assert state.needs_replan
    toolkit.execute_tool("observe", {})
    result = toolkit.execute_tool("review_task_action", {
        "expected_step": 0, "phase": "grasp", "verdict": "conflict",
        "conflict_type": "wrong_target", "expected_target": "peeler", "observed_target": "knife",
        "evidence": "Fresh target conflict after the existing physical-failure recovery gate opened.",
    }).result
    assert "error" not in result, result
    assert toolkit._current_task_conflict()["conflict_type"] == "wrong_target"
    assert not state.whole_task_policy_first_active
