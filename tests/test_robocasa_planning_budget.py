"""Plan granularity and review-only pauses must preserve execution authority."""

from copy import deepcopy

import pytest

from robots.robocasa.prompts.system import system_prompt
from rpent.harness import Harness
from rpent.harness.gate import PlanSafetyGate
from rpent.harness.lifecycle import HarnessLifecycleState
from rpent.harness.session import HarnessSession
from rpent.harness.state import PlanStateMachine
from rpent.planning.feedback import action_feedback_from_result
from rpent.runtime import RuntimePlanGenerator
from tests.test_harness_session import Generator
from tests.test_plan_generation import FakePlanningRuntime
from tests.test_robocasa_preclose_review import action, call, review, setup
from tests.test_staged_plan_generation import _OutputRuntime, _staged_output


def pause_result():
    return {
        "stop_reason": "preclose_review_required", "review_required": True,
        "executed_actions": 0, "requested_actions": 5, "cancelled_actions": 5,
        "transitions_used": 0,
        "environment": {"success": False, "terminated": False, "truncated": False},
    }


@pytest.mark.parametrize("tool", ["smolvla_pick", "continue_vla", "execute_vla", "resume_vla"])
def test_review_only_pause_keeps_verification_pending_without_spending_an_attempt(tool):
    events = []
    state = PlanStateMachine(max_replans=1, physical_tools={tool},
                             record=lambda event, **payload: events.append((event, payload)))
    state.register_plan("Acquire the cup.", [{"id": "acquire", "intent": "Acquire the cup.",
                                             "allowed_tools": [tool], "max_attempts": 1}])
    feedback = action_feedback_from_result(execution_id="pause", tool_name=tool,
                                          physical=True, result=pause_result())
    state.record_action_feedback(feedback, input_dict={})
    assert state.snapshot()["active_step_attempts"] == 0
    assert state.awaiting_verification and state.pending_execution_id == "pause"
    assert not state.completed and not state.environment_success
    blocked = PlanSafetyGate(mode="plan-gated").evaluate(
        name=tool, input_dict={}, state=state, control_tools=set(),
    )
    assert not blocked.allowed
    assert "verify" in blocked.reason
    state.verify_step("progress", ["Wrist review pending; target not acquired."],
                      "Continue acquisition.", execution_id="pause", require_execution_id=True)
    assert not state.needs_replan and not state.awaiting_verification
    assert state.snapshot()["active_step_attempts"] == 0
    record = next(payload for event, payload in events if event == "verification_required")
    assert record["attempt_consumed"] is False


@pytest.mark.parametrize("updates", [
    {"executed_actions": 1, "cancelled_actions": 4, "transitions_used": 1},
    {"executed_actions": False}, {"executed_actions": "0"},
    {"requested_actions": 0, "cancelled_actions": 0}, {"cancelled_actions": 4},
    {"transitions_used": 1}, {"review_required": False},
    {"stop_reason": "max_chunks_exhausted"}, {"error": "controller fault"},
    {"environment": {"success": True, "terminated": False, "truncated": False}},
])
def test_non_review_or_unconfirmed_zero_action_calls_still_consume_attempts(updates):
    state = PlanStateMachine(max_replans=1, physical_tools={"smolvla_pick"})
    state.register_plan("Acquire the cup.", [{"id": "acquire", "intent": "Acquire the cup.",
                                             "allowed_tools": ["smolvla_pick"], "max_attempts": 1}])
    result = {**pause_result(), **updates}
    feedback = action_feedback_from_result(execution_id="call", tool_name="smolvla_pick",
                                          physical=True, result=result)
    state.record_action_feedback(feedback, input_dict={})
    assert state.snapshot()["active_step_attempts"] == 1


def test_other_physical_tool_cannot_claim_a_review_only_attempt_exemption():
    state = PlanStateMachine(max_replans=1, physical_tools={"move"})
    state.register_plan("Move.", [{"id": "move", "intent": "Move.",
                                  "allowed_tools": ["move"], "max_attempts": 1}])
    feedback = action_feedback_from_result(execution_id="move", tool_name="move",
                                          physical=True, result=pause_result())
    state.record_action_feedback(feedback, input_dict={})
    assert state.snapshot()["active_step_attempts"] == 1


def test_strict_session_review_pause_preserves_outcomes_and_blocks_continuation(tmp_path):
    class AcquisitionGenerator(Generator):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            result["steps"][0]["allowed_tools"] = ["smolvla_pick"]
            return result

    controller = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl",
                         physical_tools={"smolvla_pick"},
                         plan_generator=AcquisitionGenerator(count=1))
    session = HarnessSession(controller)
    assert session.start_instruction("Acquire the cup.", ["The cup is visible."])["ok"]
    feedback = action_feedback_from_result(execution_id="pause", tool_name="smolvla_pick",
                                          physical=True, result=pause_result())
    assert session.record_action_feedback(feedback)["ok"]
    assert controller.state.snapshot()["active_step_attempts"] == 0
    assert session.lifecycle_state == HarnessLifecycleState.VERIFICATION_REQUIRED
    assert not session.execution_status()["contract_success"]
    rejected = session.record_action_feedback(feedback)
    assert "error" in rejected
    assert controller.state.pending_execution_id == "pause"
    assert "error" in session.finish(status="success", summary="No acquisition has occurred.")


def test_one_preclose_correction_and_handoff_leave_a_default_attempt_for_acquisition(tmp_path):
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)], [action(close=True)])
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan("Acquire the cup.", [{
        "id": "acquire", "intent": "Acquire the cup.",
        "allowed_tools": ["smolvla_pick", "adjust_eef", "resume_vla"],
    }])["ok"]

    def progress():
        return call(toolkit, "verify_step", verdict="progress",
                    evidence=["A lift still needs verification."], summary="Continue acquisition.")

    paused = call(toolkit, "smolvla_pick", max_chunks=10)
    assert paused["executed_actions"] == 0
    assert harness.state.snapshot()["active_step_attempts"] == 0
    review(toolkit, between=False)
    progress()
    call(toolkit, "adjust_eef", expected_step=0, delta_position=[0, 0, 0.01], gripper="open")
    review(toolkit)
    progress()
    call(toolkit, "resume_vla")
    call(toolkit, "observe")
    progress()
    assert harness.state.active_step.max_attempts == 3
    assert harness.state.snapshot()["active_step_attempts"] == 2
    assert not harness.state.needs_replan
    assert not harness.state.completed and not toolkit._pick_progress.confirmed
    call(toolkit, "smolvla_pick", max_chunks=1)
    assert len(simulator.actions) == 3
    assert harness.state.snapshot()["active_step_attempts"] == 3


@pytest.mark.parametrize("request_type", ["initial", "replan"])
def test_atomic_plan_guidance_allows_one_complete_operation(request_type):
    runtime = FakePlanningRuntime(tool="continue_vla")
    generator = RuntimePlanGenerator(runtime, require_outcome_program=False)
    inputs = {"goal": "Fully slide the dishwasher rack out.",
              "observations": ["The rack is visible."], "constraints": [],
              "physical_tools": {"continue_vla"}}
    if request_type == "initial":
        result = generator.generate(**inputs)
    else:
        result = generator.generate_replan(**inputs, reason="More policy time is needed.",
                                           current_plan={"strategy": "Pull the rack.", "unfinished_steps": []})
    assert len(result["steps"]) == 1
    assert "single complete-operation step" in runtime.system_prompt
    assert "distinct dependencies" in runtime.system_prompt
    assert "4-12" not in runtime.system_prompt
    assert "never exceed 16" in runtime.system_prompt
    assert "zero-action pre-close review pause" in runtime.system_prompt


def test_granularity_guidance_preserves_all_staged_outcomes():
    runtime = _OutputRuntime(deepcopy(_staged_output()))
    result = RuntimePlanGenerator(runtime).generate(
        goal="1. open drawer; 2. place cup inside", observations=["Drawer and cup are visible."],
        constraints=[], physical_tools={"move"},
    )
    assert "single complete-operation step" in runtime.system_prompt
    assert result["required_outcome_ids"] == ["outcome-1", "outcome-2"]
    assert result["steps"][0]["covers_outcomes"] == ["outcome-1"]
    assert result["stages"][1]["outcome_ids"] == ["outcome-2"]


def test_executor_budget_guidance_explains_the_review_only_exception():
    budget = system_prompt()["BUDGET"]
    assert "zero-action pre-close review pause" in budget
    assert "still requires verification" in budget


def test_verified_retry_allows_replan_without_another_physical_action(tmp_path):
    runtime = FakePlanningRuntime(tool="move")
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "retry.jsonl",
                      physical_tools={"move"}, plan_generator=RuntimePlanGenerator(runtime))
    proposal = harness.create_plan("Move.", ["Target visible."])
    assert harness.register_plan(proposal_id=proposal["proposal_id"])["ok"]
    feedback = action_feedback_from_result(execution_id="miss", tool_name="move", physical=True,
                                          result={"success": False})
    harness.state.record_action_feedback(feedback, input_dict={})
    assert harness.verify_step("retry", ["Missed the target."], "Replan acquisition.")["ok"]
    assert not harness.state.awaiting_verification
    assert harness.state.snapshot()["can_replan"]
    replacement = harness.replan("Use a different approach.", ["Target remains visible."])
    assert replacement["proposal_id"]
    assert harness.register_plan(proposal_id=replacement["proposal_id"])["ok"]
    assert harness.state.replans_used == 1


def test_unverified_pause_cannot_replan_or_silently_move():
    state = PlanStateMachine(max_replans=1, physical_tools={"smolvla_pick"})
    state.register_plan("Acquire.", [{"id": "acquire", "intent": "Acquire.",
                                     "allowed_tools": ["smolvla_pick"], "max_attempts": 3}])
    state.record_action_feedback(action_feedback_from_result(
        execution_id="pause", tool_name="smolvla_pick", physical=True, result=pause_result()),
        input_dict={})
    assert not state.snapshot()["can_replan"]
    assert "error" in state.replan("Try again.", [{"id": "acquire", "intent": "Acquire.",
                                                   "allowed_tools": ["smolvla_pick"]}])
    assert state.pending_execution_id == "pause"
