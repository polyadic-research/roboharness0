"""1.2.3.4 opens existing LLM recovery at the first real VLA failure."""

import pytest

from robots.robocasa.prompts.system import system_prompt
from rpent.harness.gate import PlanSafetyGate
from rpent.harness.lifecycle import POLICY_FAILURE_TAKEOVER_GUIDANCE
from tests.test_harness_1011_control import SINK, act, machine, named, step
from tests.test_unmeasured_composite_policy_first import (
    INSTRUCTION,
    TASKS,
    whole,
)
from tests.test_unmeasured_composite_policy_first import (
    machine as whole_machine,
)


@pytest.mark.parametrize("mode", ["delegate", "monitor"])
def test_two_executed_policy_failures_allow_but_do_not_force_staged_takeover(mode):
    events = []
    state = machine(events, task="PickPlaceCounterToSink")
    policy_step = step(**{**SINK, "control_mode": mode}, allowed_tools=["continue_vla"])
    assert state.register_plan("goal", [policy_step])["ok"]
    act(state, 16, verdict="progress", tool="continue_vla")
    assert not state.escalated_operations()
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert not state.escalated_operations()
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    [escalation] = named(events, "control_mode_escalation_available")
    assert escalation["reason"] == "failures"
    assert escalation["failed_verifications"] == 2
    gate = PlanSafetyGate(mode="plan-gated")
    assert not state.needs_replan
    assert gate.evaluate(name="continue_vla", input_dict={}, state=state, control_tools=set()).allowed
    assert gate.evaluate(name="observe", input_dict={}, state=state, control_tools=set()).allowed
    assert state.replan("first verified VLA failure", [step(
        **{**SINK, "control_mode": "takeover"},
        allowed_tools=["move_eef_path", "continue_vla"],
    )])["ok"]


@pytest.mark.parametrize("task", TASKS)
def test_first_grounded_whole_task_failure_unlocks_recovery_for_all_ten_tasks(task):
    state = whole_machine(task)
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert state.whole_task_policy_first_active
    assert state.semantic_policy_replanning_allowed
    assert state.replan("semantic review first", [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert not state.whole_task_policy_first_active
    context = state.control_planning_context(include_capabilities=False)
    assert context["control_mode_escalations"][0]["failed_verifications"] == 2
    assert state.replan("recover the first failed operation", [whole(
        control_mode="takeover", allowed_tools=["move_eef_path"],
    )])["ok"]


@pytest.mark.parametrize("executed,verdict,failure_type,error", [
    (16, "progress", None, None),
    (16, "retry", "timeout", None),
    (0, "retry", "grasp_missed", None),
    (0, "fail", "grasp_missed", "rejected before motion"),
])
def test_non_failures_do_not_offer_early_policy_takeover(executed, verdict, failure_type, error):
    state = machine()
    assert state.register_plan("goal", [step(**SINK, allowed_tools=["continue_vla"])])["ok"]
    act(state, executed, verdict=verdict, tool="continue_vla", failure_type=failure_type, error=error)
    assert not state.escalated_operations()


def test_prompt_exposes_the_shared_first_failure_recovery_contract():
    assert POLICY_FAILURE_TAKEOVER_GUIDANCE in system_prompt()["CONTROL MODES"]


@pytest.mark.parametrize("executed,verdict,failure_type,error", [
    (16, "progress", None, None),
    (16, "retry", "timeout", None),
    (16, "retry", "uncertain", None),
    (16, "retry", "budget_exhausted", None),
    (0, "retry", "preclose_review_required", None),
    (0, "retry", "grasp_missed", None),
    (0, "fail", "grasp_missed", "rejected before motion"),
])
def test_unmeasured_whole_task_keeps_native_stream_until_a_real_failure(
    executed, verdict, failure_type, error,
):
    state = whole_machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    act(state, executed, verdict=verdict, tool="continue_vla", failure_type=failure_type, error=error)
    assert state.whole_task_policy_first_active
    assert not state.escalated_operations()
