"""Direct base/torso commands share the robot and Harness execution boundary."""

import numpy as np
import pytest

from robots.robocasa.toolkit import RoboCasaToolkit
from tests.test_robocasa_motion_handoff import MotionController, call


class BodyController(MotionController):
    def execute_action(self, action, *, _action_index=0):
        value = super().execute_action(action, _action_index=_action_index)
        self.observation["state"][:3] += [action[0] * 0.05, action[1] * 0.05,
                                          action[3] * 0.01]
        return value


def setup(**kwargs):
    controller = BodyController()
    toolkit = RoboCasaToolkit(
        controller, initial_observation=controller.observation,
        instruction="Reach the observed cup.", forward_step_guidance=True,
        episode_horizon=kwargs.pop("episode_horizon", 100), **kwargs,
    )
    return toolkit, controller


def arguments(tool, step=0, **kwargs):
    return {"expected_step": step, "gripper": "open",
            **({"velocity": [0.2, -0.1, 0.15]} if tool == "move_base"
               else {"command": 0.2}), **kwargs}


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_native_dimensions_mode_stop_history_and_measured_feedback(tool):
    toolkit, controller = setup()
    toolkit.observe()
    result = call(toolkit, tool, **arguments(tool))
    assert result["executed_actions"] == result["transitions_used"] == 5
    assert result["policy_instruction"] == "Reach the observed cup."
    assert controller.policy_inputs == []
    actions = np.array(controller.actions)
    assert np.all(actions[:, 4] == 1)  # desired arm target tracks the mobile body
    assert np.all(actions[:, 5:11] == 0)
    assert np.all(actions[:, 11] == -1)
    np.testing.assert_array_equal(actions[-1, :4], 0)
    if tool == "move_base":
        np.testing.assert_allclose(actions[:-1, :3], [[0.2, -0.1, 0.15]] * 4)
        assert np.all(actions[:, 3] == 0)
    else:
        assert np.all(actions[:, :3] == 0)
        np.testing.assert_allclose(actions[:-1, 3], 0.2)
    motion = result["body_motion"]
    np.testing.assert_allclose(motion["measured_base_delta_position"],
                               controller.observation["state"][:3])
    assert "not a torso joint measurement" in motion["evidence_scope"]
    assert result["motion_review"]["observe_required"]
    assert result["motion_review"]["resume_required"]
    history = result["action_history"]["records"][-1]
    assert history["control_source"] == "direct"
    assert history["step_ids"] == list(range(1, 6))
    np.testing.assert_allclose(history["actions"], actions)
    assert result["environment"]["success"] is False


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_observe_verification_and_short_fresh_handoff(tool, tmp_path):
    from rpent.harness import Harness

    toolkit, controller = setup()
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan("Reach the cup.", [{
        "id": "reach", "intent": "Reach the cup.", "max_attempts": 4,
        "allowed_tools": [tool, "resume_vla", "continue_vla"],
    }])["ok"]
    toolkit.observe()
    call(toolkit, tool, **arguments(tool))
    assert harness.state.snapshot()["active_step_attempts"] == 1
    assert "error" in toolkit.execute_tool("resume_vla", {}).result
    toolkit.observe()
    assert "error" in toolkit.execute_tool("resume_vla", {}).result
    call(toolkit, "verify_step", verdict="progress", evidence=["Body moved; goal pending."],
         summary="Resume the current policy.")
    call(toolkit, "resume_vla", max_actions=2)
    assert controller.policy_inputs[-1][1] == "Reach the observed cup."
    assert controller.policy_inputs[-1][2] == 2
    assert "error" in toolkit.execute_tool(tool, arguments(tool, step=7)).result


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_requires_exact_fresh_observation(tool):
    toolkit, controller = setup()
    assert "observe" in toolkit.execute_tool(tool, arguments(tool)).result["error"]
    toolkit.observe()
    assert "expected_step" in toolkit.execute_tool(tool, arguments(tool, step=1)).result["error"]
    assert controller.actions == []


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
@pytest.mark.parametrize("overrides", [
    {"expected_step": True}, {"max_steps": True}, {"max_steps": 1},
    {"max_steps": 26}, {"max_steps": 2.5}, {"gripper": "invalid"},
])
def test_bad_arguments_do_not_execute_or_arm_handoff(tool, overrides):
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool(tool, arguments(tool, **overrides)).result
    assert controller.actions == []
    assert not toolkit._motion_review()["resume_required"]


@pytest.mark.parametrize("tool,overrides", [
    ("move_base", {"velocity": [0, 0, 0]}),
    ("move_base", {"velocity": [1.01, 0, 0]}),
    ("move_base", {"velocity": [float("nan"), 0, 0]}),
    ("move_base", {"velocity": [0.2, 0]}),
    ("adjust_torso", {"command": 0}), ("adjust_torso", {"command": 1.01}),
    ("adjust_torso", {"command": float("inf")}),
    ("adjust_torso", {"command": True}), ("adjust_torso", {"command": "0.2"}),
])
def test_invalid_native_commands_fail_closed(tool, overrides):
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool(tool, arguments(tool, **overrides)).result
    assert controller.actions == []


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_call_budget_horizon_and_early_terminal(tool):
    toolkit, controller = setup(max_transitions_per_tool_call=3, episode_horizon=2)
    toolkit.observe()
    result = call(toolkit, tool, **arguments(tool, max_steps=25))
    assert result["executed_actions"] == result["transition_budget"] == 2
    np.testing.assert_array_equal(controller.actions[-1][:4], 0)
    toolkit.observe()
    assert "error" in toolkit.execute_tool(tool, arguments(tool, step=2)).result
    toolkit, controller = setup()
    controller.success = True
    toolkit.observe()
    result = call(toolkit, tool, **arguments(tool))
    assert result["executed_actions"] == 1
    assert result["cancelled_actions"] == 4
    assert not result["body_motion"]["stop_command_executed"]


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_keep_requires_confirmed_gripper_and_preserves_close(tool):
    toolkit, controller = setup()
    toolkit.observe()
    args = arguments(tool, gripper="keep")
    assert "gripper" in toolkit.execute_tool(tool, args).result["error"]
    action = np.zeros(12, dtype=np.float32)
    action[11] = 1
    toolkit._operation.execute_actions((action,))
    toolkit.observe()
    result = call(toolkit, tool, **arguments(tool, step=1, gripper="keep"))
    assert all(row[11] == 1 for row in controller.actions[-5:])
    assert result["body_motion"]["gripper_source"] == "last_confirmed_command"


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_preclose_movement_requires_open_and_invalidates_review(tool):
    toolkit, controller = setup()
    toolkit._preclose_active = True
    toolkit._preclose_review_enabled = True
    toolkit.observe()
    toolkit.review_grasp(expected_step=0, object_between_fingers=True,
                         narrow_axis_aligned=True, evidence="Observed aligned wrist pose.")
    assert "open" in toolkit.execute_tool(tool, arguments(tool, gripper="close")).result["error"]
    assert controller.actions == []
    call(toolkit, tool, **arguments(tool))
    assert toolkit._preclose_approval is None
    assert toolkit._preclose_pending


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_partial_fault_requires_observation_and_retains_confirmed_history(tool):
    from rpent.episode.controller import EpisodeExecutionError

    toolkit, controller = setup()
    toolkit.observe()
    original = controller.execute_action

    def fault(action, *, _action_index=0):
        if controller.step:
            raise EpisodeExecutionError("probe fault", confirmed_transitions=(), requested_actions=1,
                                        attempted_action_index=_action_index)
        return original(action, _action_index=_action_index)

    controller.execute_action = fault
    result = toolkit.execute_tool(tool, arguments(tool)).result
    assert "error" in result
    assert result["motion_review"]["observe_required"]
    record = result["action_history"]["records"][-1]
    assert record["status"] == "faulted" and record["step_ids"] == [1]
    assert record["executed_actions"] == 1


def test_tools_are_registered_for_strict_harness_and_planning(tmp_path):
    from robots.robocasa.prompts.system import system_prompt
    from tests.test_toolkit_harness_session import RecordingSession

    toolkit, _ = setup()
    session = RecordingSession()
    toolkit.attach_harness_session(session, tmp_path)
    specs = {row["name"]: row for row in toolkit.get_tools_spec()}
    for tool in ("move_base", "adjust_torso"):
        assert tool in session.controller.physical_tools
        assert tool in specs
        assert tool in system_prompt()["ACTION"]
        assert "expected_step" in specs[tool]["input_schema"]["required"]


def test_all_three_correction_tool_descriptions_reject_productive_target_contact():
    toolkit, _ = setup()
    specs = {row["name"]: row for row in toolkit.get_tools_spec()}
    for tool in ("adjust_eef", "move_base", "adjust_torso"):
        description = specs[tool]["description"]
        assert "Do not use for productive contact with the intended target" in description
        assert "task-direction progress and typical-step timing" in description


@pytest.mark.parametrize("tool", ["move_base", "adjust_torso"])
def test_grasp_loss_cancels_remaining_direct_actions(tool):
    from tests.test_robocasa_grasp_lost import setup as grasp_setup

    toolkit, simulator, _ = grasp_setup([0.0] * 5)
    toolkit.observe()
    result = call(toolkit, tool, **arguments(tool, step=2, gripper="keep"))
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1 and result["cancelled_actions"] == 4
    assert simulator.step_id == 3
    assert result["grasp_events"][0]["step"] == 3
