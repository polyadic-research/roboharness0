"""Focused tests for provider-independent Harness plan state mechanics."""

from __future__ import annotations

import pytest

from rpent.harness.state import (
    MAX_STEP_ATTEMPTS,
    PlanStateMachine,
    parse_steps,
    plan_input_schema,
)
from rpent.planning.schemas import ActionFeedback


def _action_step(
    step_id: str = "move",
    *,
    max_attempts: int = 2,
    stage_id: str | None = None,
    outcome_id: str | None = None,
) -> dict:
    step = {
        "id": step_id,
        "intent": f"execute {step_id}",
        "allowed_tools": ["move"],
        "max_attempts": max_attempts,
    }
    if stage_id is not None:
        step["stage_id"] = stage_id
    if outcome_id is not None:
        step["covers_outcomes"] = [outcome_id]
    return step


def _feedback(
    execution_id: str,
    *,
    environment_success: bool = False,
    environment_terminated: bool = False,
    environment_truncated: bool = False,
) -> ActionFeedback:
    return ActionFeedback(
        execution_id=execution_id,
        tool_name="move",
        physical=True,
        execution_status="completed",
        environment_success=environment_success,
        environment_terminated=environment_terminated,
        environment_truncated=environment_truncated,
    )


def _state(*, max_replans: int = 1) -> PlanStateMachine:
    return PlanStateMachine(max_replans=max_replans, physical_tools={"move"})


def test_parse_steps_preserves_stage_coverage_and_observation_type() -> None:
    steps = parse_steps(
        [
            {
                "id": "inspect",
                "intent": "confirm the target",
                "step_type": "observation",
                "allowed_tools": [],
                "covers_outcomes": ["outcome-1"],
                "stage_id": "grounding",
            }
        ],
        {"move"},
    )

    assert steps[0].step_type == "observation"
    assert steps[0].allowed_tools == ()
    assert steps[0].covers_outcomes == ("outcome-1",)
    assert steps[0].stage_id == "grounding"


def test_grouped_step_can_budget_a_bounded_multi_waypoint_sequence() -> None:
    steps = parse_steps([_action_step(max_attempts=50)], {"move"})

    assert steps[0].max_attempts == 50
    assert (
        plan_input_schema(require_goal=True)["properties"]["steps"]["items"][
            "properties"
        ]["max_attempts"]["maximum"]
        == MAX_STEP_ATTEMPTS
    )


def test_grouped_step_budget_remains_bounded() -> None:
    with pytest.raises(ValueError, match=f"between 1 and {MAX_STEP_ATTEMPTS}"):
        parse_steps([_action_step(max_attempts=MAX_STEP_ATTEMPTS + 1)], {"move"})


def test_uncertain_physical_verdict_keeps_same_execution_pending() -> None:
    state = _state()
    state.register_plan("move", [_action_step()])
    feedback = _feedback("exec-1")
    state.record_action_feedback(feedback, input_dict={})

    result = state.verify_step(
        "uncertain",
        ["target is occluded"],
        "cannot determine the effect",
        execution_id="exec-1",
        require_execution_id=True,
    )

    assert result["ok"] is True
    assert state.awaiting_verification is True
    assert state.pending_verification_execution_id == "exec-1"
    assert state.pending_execution_id == "exec-1"
    assert state.needs_replan is False
    assert state.active_step is not None and state.active_step.id == "move"
    assert state.last_feedback is feedback

    confirmed = state.verify_step(
        "pass",
        ["fresh view confirms arrival"],
        "effect confirmed",
        execution_id="exec-1",
        require_execution_id=True,
    )
    assert confirmed["ok"] is True
    assert state.completed is True
    assert state.pending_verification_execution_id is None


def test_authoritative_failure_clears_pending_and_requires_recovery() -> None:
    state = _state()
    state.register_plan("move", [_action_step()])
    state.record_action_feedback(_feedback("exec-1"), input_dict={})

    result = state.verify_step(
        "fail",
        ["fresh view confirms no motion"],
        "action failed",
        execution_id="exec-1",
        require_execution_id=True,
    )

    assert result["ok"] is True
    assert state.awaiting_verification is False
    assert state.pending_verification_execution_id is None
    assert state.needs_replan is True


def test_success_advances_only_current_step_even_when_environment_terminates() -> None:
    state = _state()
    state.register_plan(
        "move then audit",
        [
            _action_step(),
            {
                "id": "audit",
                "intent": "audit the final outcome",
                "step_type": "observation",
                "allowed_tools": [],
            },
        ],
    )
    state.record_action_feedback(
        _feedback("exec-1", environment_success=True), input_dict={}
    )

    state.verify_step("pass", ["move succeeded"], "action complete")

    assert state.environment_success is True
    assert state.completed is False
    assert state.active_step is not None and state.active_step.id == "audit"


def test_terminal_failure_does_not_set_environment_success() -> None:
    state = _state()
    state.register_plan("move", [_action_step()])
    state.record_action_feedback(
        _feedback("exec-1", environment_terminated=True), input_dict={}
    )

    assert state.environment_success is False


def test_failed_execution_cannot_set_environment_success() -> None:
    state = _state()
    state.register_plan("move", [_action_step()])
    state.record_action_feedback(
        ActionFeedback(
            execution_id="exec-1",
            tool_name="move",
            physical=True,
            execution_status="controller_error",
            error="transport failed",
            environment_success=True,
            environment_terminated=False,
            environment_truncated=False,
        ),
        input_dict={},
    )

    assert state.environment_success is False


def test_observation_step_can_be_rechecked_after_uncertain_evidence() -> None:
    state = _state()
    registered = state.register_plan(
        "inspect",
        [
            {
                "id": "inspect",
                "intent": "identify the target",
                "step_type": "observation",
                "allowed_tools": [],
            }
        ],
    )
    assert registered["ok"] is True

    uncertain = state.verify_observation(
        "uncertain", ["view is occluded"], "need another observation"
    )
    assert uncertain["ok"] is True
    assert state.awaiting_verification is True
    assert state.pending_execution_id is None
    assert state.needs_replan is False

    confirmed = state.verify_observation(
        "pass", ["fresh view identifies the target"], "target identified"
    )
    assert confirmed["ok"] is True
    assert state.completed is True


def test_register_next_stage_is_atomic_and_preserves_stage_binding() -> None:
    state = _state()
    state.register_plan("first", [_action_step("first")])
    state.record_action_feedback(_feedback("exec-1"), input_dict={})
    state.verify_step("pass", ["done"], "first complete")
    before = state.view_plan()

    invalid = state.register_next_stage(
        "second",
        [_action_step("second", stage_id="wrong", outcome_id="outcome-2")],
        stage_id="stage-2",
        stage_outcome_ids=("outcome-2",),
    )

    assert "error" in invalid
    assert state.view_plan() == before

    registered = state.register_next_stage(
        "second",
        [_action_step("second", stage_id="stage-2", outcome_id="outcome-2")],
        stage_id="stage-2",
        stage_outcome_ids=("outcome-2",),
    )
    assert registered["ok"] is True
    assert state.active_step is not None
    assert state.active_step.id == "second"
    assert state.active_step.stage_id == "stage-2"
    assert state.active_step.covers_outcomes == ("outcome-2",)


def test_staged_steps_allow_grouping_but_require_flattened_exact_order() -> None:
    state = _state()
    accepted = state.register_plan(
        "grouped stage",
        [
            {
                **_action_step("grouped", stage_id="stage-1"),
                "covers_outcomes": ["outcome-1", "outcome-2"],
            },
            _action_step("last", stage_id="stage-1", outcome_id="outcome-3"),
        ],
        stage_id="stage-1",
        stage_outcome_ids=("outcome-1", "outcome-2", "outcome-3"),
    )
    assert accepted["ok"] is True

    completed = _state()
    completed.register_plan("first", [_action_step("first")])
    completed.record_action_feedback(_feedback("exec-1"), input_dict={})
    completed.verify_step("pass", ["done"], "first complete")
    before = completed.view_plan()

    out_of_order = completed.register_next_stage(
        "second",
        [
            _action_step("second", stage_id="stage-2", outcome_id="outcome-2"),
            _action_step("first", stage_id="stage-2", outcome_id="outcome-1"),
        ],
        stage_id="stage-2",
        stage_outcome_ids=("outcome-1", "outcome-2"),
    )
    assert "exactly once and in order" in out_of_order["error"]
    assert completed.view_plan() == before

    missing = completed.register_next_stage(
        "second",
        [_action_step("first", stage_id="stage-2", outcome_id="outcome-1")],
        stage_id="stage-2",
        stage_outcome_ids=("outcome-1", "outcome-2"),
    )
    assert "exactly once and in order" in missing["error"]
    assert completed.view_plan() == before


def test_abandon_then_resume_retry_does_not_spend_replan_budget() -> None:
    state = _state(max_replans=0)
    state.register_plan("move", [_action_step(max_attempts=2)])
    state.record_action_feedback(_feedback("exec-1"), input_dict={})

    abandoned = state.abandon_pending_action(
        reason="verification service unavailable",
        evidence=["verifier timed out"],
    )
    assert abandoned["ok"] is True
    assert state.awaiting_verification is False
    assert state.pending_execution_id is None
    assert state.needs_replan is True

    resumed = state.resume_recovery_retry(reason="retry after a fresh observation")
    assert resumed["ok"] is True
    assert state.needs_replan is False
    assert state.replans_used == 0
    assert state.active_step is not None and state.active_step.id == "move"


def test_request_recovery_accepts_action_observation_pending_and_completed() -> None:
    action = _state()
    action.register_plan("move", [_action_step()])
    assert action.request_recovery(reason="blocked", evidence=["path blocked"])["ok"]
    assert action.needs_replan is True

    observation = _state()
    observation.register_plan(
        "inspect",
        [
            {
                "id": "inspect",
                "intent": "inspect target",
                "step_type": "observation",
                "allowed_tools": [],
            }
        ],
    )
    observation.verify_observation("uncertain", ["target occluded"], "need recovery")
    assert observation.request_recovery(
        reason="camera blocked", evidence=["occlusion persists"]
    )["ok"]
    assert observation.awaiting_verification is False
    assert observation.needs_replan is True

    pending = _state()
    pending.register_plan("move", [_action_step()])
    pending.record_action_feedback(_feedback("exec-1"), input_dict={})
    assert pending.request_recovery(
        reason="verifier failed", evidence=["verifier timeout"]
    )["ok"]
    assert pending.awaiting_verification is False
    assert pending.pending_execution_id is None
    assert pending.needs_replan is True

    completed = _state()
    completed.register_plan("move", [_action_step()])
    completed.record_action_feedback(_feedback("exec-1"), input_dict={})
    completed.verify_step("pass", ["done"], "complete")
    assert completed.request_recovery(
        reason="final audit failed", evidence=["predicate regressed"]
    )["ok"]
    assert completed.needs_replan is True
    assert completed.snapshot()["status"] == "needs_replan"


def test_request_recovery_invalid_calls_do_not_mutate_state() -> None:
    unplanned = _state()
    before_unplanned = unplanned.snapshot()
    rejected = unplanned.request_recovery(reason="blocked", evidence=["obstacle"])
    assert "no registered plan" in rejected["error"]
    assert unplanned.snapshot() == before_unplanned

    recovering = _state()
    recovering.register_plan("move", [_action_step()])
    recovering.request_recovery(reason="blocked", evidence=["obstacle"])
    before_repeated = recovering.snapshot()
    repeated = recovering.request_recovery(reason="again", evidence=["same obstacle"])
    assert "already required" in repeated["error"]
    assert recovering.snapshot() == before_repeated

    invalid_evidence = _state()
    invalid_evidence.register_plan("move", [_action_step()])
    before_invalid = invalid_evidence.snapshot()
    rejected_evidence = invalid_evidence.request_recovery(reason="blocked", evidence=[])
    assert "non-empty array" in rejected_evidence["error"]
    assert invalid_evidence.snapshot() == before_invalid

    bounded_retry = _state()
    bounded_retry.register_plan("move", [_action_step(max_attempts=2)])
    bounded_retry.record_action_feedback(_feedback("exec-1"), input_dict={})
    bounded_retry.verify_step("retry", ["retryable miss"], "retry locally")
    before_retry = bounded_retry.snapshot()
    rejected_retry = bounded_retry.request_recovery(
        reason="duplicate recovery", evidence=["retry already selected"]
    )
    assert "already in progress" in rejected_retry["error"]
    assert bounded_retry.snapshot() == before_retry


def test_retry_resume_rejects_exhausted_step_attempt_budget() -> None:
    state = _state()
    state.register_plan("move", [_action_step(max_attempts=1)])
    state.record_action_feedback(_feedback("exec-1"), input_dict={})
    state.abandon_pending_action(reason="execution exception", evidence=["timeout"])

    rejected = state.resume_recovery_retry(reason="try again")

    assert "attempt budget is exhausted" in rejected["error"]
    assert state.needs_replan is True


def test_replan_can_delegate_budget_and_atomically_validate_stage_steps() -> None:
    state = _state(max_replans=0)
    state.register_plan("move", [_action_step()])
    state.record_action_feedback(_feedback("exec-1"), input_dict={})
    state.verify_step("fail", ["move missed"], "repair required")
    before = state.view_plan()

    invalid = state.replan(
        "repair",
        [_action_step("repair", stage_id="wrong", outcome_id="outcome-1")],
        stage_id="stage-1",
        stage_outcome_ids=("outcome-1",),
        enforce_global_budget=False,
    )
    assert "error" in invalid
    assert state.view_plan() == before

    repaired = state.replan(
        "repair",
        [_action_step("repair", stage_id="stage-1", outcome_id="outcome-1")],
        stage_id="stage-1",
        stage_outcome_ids=("outcome-1",),
        enforce_global_budget=False,
    )
    assert repaired["ok"] is True
    assert state.active_step is not None and state.active_step.id == "repair"
