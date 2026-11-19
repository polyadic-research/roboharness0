"""Regression cases for direct gripper review, fault locks and action budgets."""

import pytest

from robots.robocasa.binding import make_robocasa_binding
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.episode.controller import (
    EpisodeController,
    EpisodeExecutionError,
    EpisodeFaultedError,
)
from rpent.tools.vla import VLAOperation
from tests.test_robocasa_motion_handoff import MotionController
from tests.test_robocasa_preclose_review import (
    Policy,
    Simulator,
    action,
    call,
    review,
    setup,
)


@pytest.mark.parametrize("opening_tool", ["servo_eef", "adjust_eef"])
def test_direct_reopening_requires_new_wrist_review(opening_tool):
    toolkit, simulator, _ = setup(
        [action(close=True)], [action(close=True)], [action(close=True)],
    )
    call(toolkit, "smolvla_pick")
    assert review(toolkit)["approved"]
    call(toolkit, "smolvla_pick")
    if opening_tool == "servo_eef":
        call(toolkit, opening_tool, target_position=[0.2, 0.1, 0.5],
             target_quaternion=[0, 0, 0, 1], gripper="open")
        next_tool, arguments = "continue_vla", {"max_chunks": 1}
    else:
        observed = call(toolkit, "observe")
        call(toolkit, opening_tool, expected_step=observed["step"],
             delta_position=[0, 0, 0.01], gripper="open")
        next_tool, arguments = "resume_vla", {}
    observed = call(toolkit, "observe")
    assert observed["grasp_review"]["required"]
    before = len(simulator.actions)
    result = call(toolkit, next_tool, **arguments)
    assert result["stop_reason"] == "preclose_review_required"
    assert result["executed_actions"] == 0
    assert len(simulator.actions) == before


def test_direct_closure_consumes_visual_approval():
    toolkit, _, _ = setup([action(close=True)])
    call(toolkit, "smolvla_pick")
    assert review(toolkit)["approved"]
    call(toolkit, "servo_eef", target_position=[0.2, 0.1, 0.5],
         target_quaternion=[0, 0, 0, 1], gripper="close")
    observed = call(toolkit, "observe")
    assert observed["grasp_review"]["approved_step"] is None
    result = toolkit.execute_tool("review_grasp", {
        "expected_step": observed["step"], "object_between_fingers": True,
        "narrow_axis_aligned": True, "evidence": "The fingers have already closed.",
    }).result
    assert "error" in result


def test_ten_direct_open_actions_expire_visual_approval():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    assert review(toolkit)["approved"]
    for _ in range(10):
        call(toolkit, "servo_eef", target_position=[0.2, 0.1, 0.5],
             target_quaternion=[0, 0, 0, 1], gripper="open")
    result = call(toolkit, "continue_vla", max_chunks=1)
    assert result["stop_reason"] == "preclose_review_required"
    assert len(simulator.actions) == 10
    assert all(value[11] < 0.5 for value in simulator.actions)


@pytest.mark.parametrize("formal", [True, False])
def test_direct_monitor_error_locks_future_execution_and_preserves_progress(formal):
    simulator = Simulator()
    controller = (
        EpisodeController(simulator, Policy([]), make_robocasa_binding())
        if formal else MotionController()
    )
    initial = simulator.observation if formal else controller.observation
    guard_calls = 0

    def failing_guard(_action, _transition):
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            raise ValueError("invalid proprioception")
        return None

    operation = VLAOperation(controller, initial_observation=initial,
                             instruction="Move the cup.", capture_action_history=True,
                             transition_guard=failing_guard)
    commands = tuple(action() for _ in range(3))
    with pytest.raises(EpisodeExecutionError, match="guard failed") as raised:
        operation.execute_actions(commands)
    assert raised.value.confirmed_actions == 2
    assert operation.metrics["step_count"] == 2
    record = operation.action_history_since()[0]
    assert record["status"] == "faulted"
    assert record["step_ids"] == [1, 2]
    assert record["executed_actions"] == 2 and record["cancelled_actions"] == 1
    assert len(record["actions"]) == 3
    for execute in (lambda: operation.execute_actions(commands), operation.execute):
        with pytest.raises(EpisodeFaultedError):
            execute()
    if formal:
        with pytest.raises(EpisodeFaultedError):
            controller.execute_action(action())
    assert guard_calls == 2


def budget_setup(*, budget, horizon=None):
    simulator = Simulator()
    policy = Policy([[action()] * 5])
    controller = EpisodeController(simulator, policy, make_robocasa_binding())
    toolkit = RoboCasaToolkit(controller, initial_observation=simulator.observation,
                             instruction="Move the cup.", actions_per_policy_request=5,
                             max_transitions_per_tool_call=budget, episode_horizon=horizon)
    return toolkit, simulator, policy


@pytest.mark.parametrize("budget", [1, 2])
@pytest.mark.parametrize("requested", [None, 2, 10])
def test_diagnostic_policy_tool_caps_the_request_before_inference(budget, requested):
    toolkit, simulator, policy = budget_setup(budget=budget)
    result = call(toolkit, "execute_vla", max_actions=requested)
    expected = min(budget, 5, requested if requested is not None else 5)
    assert result["executed_actions"] == expected
    assert len(simulator.actions) == expected
    assert len(policy.inputs) == 1


@pytest.mark.parametrize("budget", [1, 2])
def test_direct_servo_caps_control_transitions(budget):
    toolkit, simulator, _ = budget_setup(budget=budget)
    result = call(toolkit, "servo_eef", target_position=[0.4, 0.1, 0.5],
                  target_quaternion=[0, 0, 0, 1], gripper="open", max_steps=10)
    assert result["executed_actions"] == budget
    assert len(simulator.actions) == budget
    assert not result["target_reached"]


@pytest.mark.parametrize("tool", ["execute_vla", "servo_eef"])
def test_direct_tools_respect_remaining_episode_horizon(tool):
    toolkit, simulator, policy = budget_setup(budget=10, horizon=1)
    arguments = {} if tool == "execute_vla" else {
        "target_position": [0.4, 0.1, 0.5],
        "target_quaternion": [0, 0, 0, 1], "gripper": "open", "max_steps": 10,
    }
    result = call(toolkit, tool, **arguments)
    assert result["executed_actions"] == 1
    assert result["episode_budget"]["exhausted"]
    assert len(simulator.actions) == 1
    requests = len(policy.inputs)
    rejected = toolkit.execute_tool(tool, arguments).result
    assert "remaining episode action budget" in rejected["error"]
    assert len(simulator.actions) == 1 and len(policy.inputs) == requests
