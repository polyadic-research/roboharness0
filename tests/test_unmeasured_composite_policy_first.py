"""Unknown composite capability is not evidence for preemptive LLM control."""

import pytest

from robots.robocasa.capabilities import PRIVILEGED_PHYSICAL_TOOLS
from robots.robocasa.prompts.system import system_prompt
from rpent.harness.state import PlanStateMachine, parse_steps
from tests.test_harness_1011_control import act

TASKS = (
    "ArrangeTea", "BreadSelection", "CuttingToolSelection", "GarnishPancake",
    "GetToastedBread", "LoadDishwasher", "PortionHotDogs", "SetUpCuttingStation",
    "StackBowlsCabinet", "WeighIngredients",
)
TOOLS = {"continue_vla", "execute_vla", "smolvla_pick", "resume_vla", "adjust_eef",
         *PRIVILEGED_PHYSICAL_TOOLS}
INSTRUCTION = "Move both objects onto the board, then close the cabinet."


def whole(**changes):
    return {
        "id": "whole_task", "operation": "whole_task", "skill_row": "none",
        "tier": "none", "control_mode": "monitor",
        "intent": f'Execute the complete original task: "{INSTRUCTION}"',
        "allowed_tools": ["continue_vla", "execute_vla"], "max_attempts": 64,
        **changes,
    }


def machine(task="BreadSelection"):
    state = PlanStateMachine(max_replans=None, environment_horizon_only=True,
                             physical_tools=TOOLS, policy_family="xiaomi-mibot", task_name=task)
    state.set_policy_instruction(INSTRUCTION)
    return state


@pytest.mark.parametrize("task", TASKS)
def test_all_ten_unmeasured_tasks_require_one_whole_monitor_operation(task):
    state = machine(task)
    assert state.whole_task_policy_first_active
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    scope = state.control_planning_context()["task_execution_scope"]
    assert scope["whole_task_first"] and scope["tier"] == "none"
    assert scope["instruction"] == INSTRUCTION
    assert not scope["decomposition_allowed"]


@pytest.mark.parametrize("steps", [
    [whole(control_mode="takeover", allowed_tools=["move_eef_path"])],
    [whole(id="first"), whole(id="second", operation="other")],
    [whole(subtask="pick")],
    [whole(allowed_tools=["smolvla_pick"])],
    [whole(allowed_tools=["continue_vla", "adjust_eef"])],
])
def test_preemptive_takeover_decomposition_and_motor_corrections_are_rejected(steps):
    with pytest.raises(ValueError, match="whole-task"):
        parse_steps(steps, TOOLS, policy_family="xiaomi-mibot", task_name="BreadSelection")


def test_live_original_instruction_cannot_be_replaced_by_a_subtask():
    state = machine()
    result = state.register_plan(INSTRUCTION, [whole(intent='Execute "Move one object onto the board."')])
    assert "original instruction" in result["error"]


def test_review_replanning_cannot_change_the_whole_task_operation_identity():
    state = machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="timeout")
    result = state.replan("call window ended", [whole(operation="reset_operation")])
    assert "whole-task operation identity" in result["error"]


def test_frozen_task_scope_does_not_reload_the_registry_at_each_control_step(monkeypatch):
    state = machine()

    def no_registry_io(_task):
        pytest.fail("task metadata is already frozen; do not reread it per action")

    monkeypatch.setattr("robots.robocasa.capabilities.is_composite_task", no_registry_io)
    for _ in range(20):
        assert state.whole_task_policy_first_active


def test_call_limits_and_long_waits_never_unlock_unknown_task_takeover():
    state = machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    for _ in range(20):
        act(state, 160, verdict="retry", tool="continue_vla", failure_type="timeout")
        assert state.replan("same task at call boundary", [whole()])["ok"]
    assert not state.escalated_operations()
    assert state.whole_task_policy_first_active
    assert "error" in state.replan("no capability row", [whole(control_mode="takeover",
                                                              allowed_tools=["move_eef_path"])])


def test_first_grounded_physical_failure_allows_existing_recovery_flow():
    state = machine()
    assert state.register_plan(INSTRUCTION, [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert state.semantic_policy_replanning_allowed
    assert not state.escalated_operations()
    assert state.replan("semantic review", [whole()])["ok"]
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert not state.whole_task_policy_first_active
    assert state.escalated_operations()
    assert state.replan("observed recovery", [whole(control_mode="takeover",
                                                   allowed_tools=["move_eef_path"])])["ok"]


@pytest.mark.parametrize("task", ["SteamInMicrowave", "CategorizeCondiments", "OpenFridge"])
def test_measured_composites_and_atomic_tasks_keep_the_existing_routing(task):
    assert not machine(task).whole_task_policy_first_active


def test_prompt_exposes_unknown_composite_exception_without_fabricating_a_tier():
    text = " ".join(system_prompt().values())
    assert "unmeasured composite" in text
    assert "whole-task" in text
    assert "120-step" in text


def bound_toolkit(tmp_path, *chunks):
    from rpent.harness import Harness
    from tests.test_robocasa_preclose_review import TASK, setup

    toolkit, simulator, policy = setup(*chunks, actions_per_policy_request=len(chunks[0]))
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl", policy_family="xiaomi-mibot",
                      task_name="CuttingToolSelection", max_replans=None)
    toolkit.attach_harness(harness)
    plan = whole(intent=f'Execute the complete original task: "{TASK}"')
    assert harness.state.register_plan(TASK, [plan])["ok"]
    return toolkit, simulator, policy, harness


def test_whole_task_policy_stream_matches_unwrapped_policy_across_observations(tmp_path):
    import numpy as np

    from tests.test_robocasa_preclose_review import TASK, action, setup

    chunks = [[action(dz=-1), action(close=True)], [action(close=True), action()]]
    reference, reference_sim, reference_policy = setup(*chunks, actions_per_policy_request=2)
    toolkit, simulator, policy, harness = bound_toolkit(tmp_path, *chunks)
    # Even an armed acquisition review cannot replace the whole-task stream.
    toolkit._preclose_active = True
    for index in range(2):
        reference._operation.execute(max_actions=2, instruction=TASK)
        toolkit.execute_tool("observe", {})
        result = toolkit.execute_tool("continue_vla", {"max_chunks": 1}).result
        assert "error" not in result, result
        assert result["executed_actions"] == 2
        assert result["paused_actions"] == result["cancelled_actions"] == 0
        assert harness.state.verify_step("progress", ["same native action stream"],
                                         "whole task remains active")["ok"]
        assert len(policy.inputs) == index + 1
    np.testing.assert_array_equal(simulator.actions, reference_sim.actions)
    assert {instruction for _, instruction in policy.inputs} == {TASK}
    for (actual, _), (expected, _) in zip(policy.inputs, reference_policy.inputs):
        np.testing.assert_array_equal(actual, expected)


def test_subtask_instruction_is_refused_before_model_or_robot_execution(tmp_path):
    from tests.test_robocasa_preclose_review import action

    toolkit, simulator, policy, _ = bound_toolkit(tmp_path, [action(), action(close=True)])
    result = toolkit.execute_tool("continue_vla", {
        "instruction": "Only acquire the tool.", "intervene": True, "max_chunks": 1,
    }).result
    assert "original instruction" in result["error"]
    assert simulator.actions == [] and policy.inputs == []


@pytest.mark.parametrize("intervene", [False, True])
def test_authorized_semantic_review_can_explicitly_switch_language_without_renaming(tmp_path, intervene):
    from tests.test_robocasa_preclose_review import TASK, action

    toolkit, _, policy, harness = bound_toolkit(tmp_path, [action(), action()])
    state = harness.state
    act(state, 16, verdict="retry", tool="continue_vla", failure_type="grasp_missed")
    assert state.semantic_policy_replanning_allowed and not state.escalated_operations()
    assert state.replan("Fresh evidence identifies the remaining original goal", [
        whole(intent=f'Execute the complete original task: "{TASK}"'),
    ])["ok"]
    selected = "Finish the remaining placement onto the original board."
    result = toolkit.execute_tool("continue_vla", {
        "max_chunks": 1, "intervene": intervene, "instruction": selected,
    }).result
    assert "error" not in result, result
    assert policy.inputs[-1][1] == (selected if intervene else TASK)
    assert state.active_step.operation == "whole_task"
    assert state._ledger(state.active_step)["failures"]["monitor"] == 1
    text = " ".join(system_prompt().values())
    assert "same registered operation" in text
    assert "Internal phase bookkeeping alone never requires this language switch" in text
