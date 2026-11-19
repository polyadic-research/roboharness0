"""Grounded local corrections must hand back to fresh, bounded policy inference."""

import numpy as np
import pytest

from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.episode.contracts import ChunkExecutionResult, StepResult
from tests.test_robocasa_toolkit import _pose_observation

TASK = "Open the stand mixer head."


class MotionController:
    def __init__(self):
        self.step = 0
        self.actions = []
        self.policy_inputs = []
        self.success = False
        self.truncated = False
        self.observation = _pose_observation(
            position=(0.2, 0.1, 0.5), gripper_qpos=(0.035, -0.035),
        )

    def execute_action(self, action, *, _action_index=0):
        del _action_index
        self.actions.append(action.copy())
        self.step += 1
        self.observation = {**self.observation, "state": self.observation["state"].copy()}
        self.observation["state"][7:10] += action[5:8] * 0.05
        return StepResult(
            episode_id="test-motion", step_id=self.step,
            observation=self.observation, reward=0, success=self.success,
            terminated=False, truncated=self.truncated,
        )

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        self.policy_inputs.append((observation["state"].copy(), instruction, max_actions))
        actions = tuple(np.zeros(12, dtype=np.float32) for _ in range(max_actions))
        transitions = []
        for action in actions:
            transition = self.execute_action(action)
            transitions.append(transition)
            if transition.terminal:
                break
        return ChunkExecutionResult(tuple(transitions), len(actions), actions=actions)


def setup(*, horizon=100, budget=250, actions_per_request=5):
    controller = MotionController()
    toolkit = RoboCasaToolkit(
        controller, initial_observation=controller.observation, instruction=TASK,
        forward_step_guidance=True, actions_per_policy_request=actions_per_request,
        episode_horizon=horizon, max_transitions_per_tool_call=budget,
    )
    return toolkit, controller


def call(toolkit, name, **kwargs):
    result = toolkit.execute_tool(name, kwargs).result
    assert "error" not in result, result
    return result


def adjust(toolkit, **kwargs):
    return call(toolkit, "adjust_eef", expected_step=toolkit.observe()["step"],
                delta_position=[0, 0, 0.02], gripper="open", **kwargs)


def test_correction_is_measured_bounded_and_does_not_change_language():
    toolkit, controller = setup()
    result = adjust(toolkit)
    assert result["executed_actions"] == 1
    assert result["target_reached"]
    assert result["policy_instruction"] == TASK
    np.testing.assert_allclose(result["motion_adjustment"]["measured_delta_position"],
                               [0, 0, 0.02], atol=1e-7)
    assert result["environment"]["success"] is False
    assert result["motion_review"]["observe_required"]
    assert result["motion_review"]["resume_required"]
    assert all(r["control_source"] == "direct" for r in result["action_history"]["records"])
    assert controller.policy_inputs == []


@pytest.mark.parametrize("tool", ["execute_vla", "continue_vla", "smolvla_pick", "servo_eef"])
def test_adjustment_cannot_be_overwritten_without_observation_and_short_handoff(tool):
    toolkit, controller = setup()
    adjust(toolkit)
    arguments = ({"target_position": [0.2, 0.1, 0.52],
                  "target_quaternion": [0, 0, 0, 1], "gripper": "open"}
                 if tool == "servo_eef" else {})
    rejected = toolkit.execute_tool(tool, arguments).result
    assert "observe" in rejected["error"]
    toolkit.observe()
    rejected = toolkit.execute_tool(tool, arguments).result
    assert "resume_vla" in rejected["error"]
    assert len(controller.actions) == 1


def test_fresh_short_handoff_keeps_instruction_and_requires_a_progress_review():
    toolkit, controller = setup(actions_per_request=None)
    adjust(toolkit)
    rejected = toolkit.execute_tool("resume_vla", {}).result
    assert "observe" in rejected["error"]
    fresh = toolkit.observe()
    result = call(toolkit, "resume_vla", max_actions=10)
    assert result["executed_actions"] == 10
    assert result["instruction"] == TASK
    assert result["intervened"] is False
    np.testing.assert_array_equal(controller.policy_inputs[0][0], fresh["state"]["proprioception"])
    assert result["motion_review"]["observe_required"]
    rejected = toolkit.execute_tool("continue_vla", {"max_chunks": 2}).result
    assert "observe" in rejected["error"]
    toolkit.observe()
    normal = call(toolkit, "continue_vla", max_chunks=2)
    assert normal["executed_actions"] > 10
    assert {entry[1] for entry in controller.policy_inputs} == {TASK}


def test_configured_policy_chunk_limit_is_preserved_on_handoff():
    toolkit, controller = setup()
    adjust(toolkit)
    toolkit.observe()
    result = call(toolkit, "resume_vla", max_actions=10)
    assert result["executed_actions"] == 5
    assert controller.policy_inputs[-1][2] == 5


@pytest.mark.parametrize("arguments", [
    {"expected_step": 1}, {"expected_step": True},
    {"delta_position": [0.04, 0.04, 0]}, {"delta_position": [float("nan"), 0, 0]},
    {"rotation_degrees": [0, 0, 16]}, {"rotation_degrees": [12, 12, 0]},
    {"max_steps": 26}, {"max_steps": False}, {"gripper": "invalid"},
])
def test_invalid_correction_does_not_execute_or_require_handoff(arguments):
    toolkit, controller = setup()
    toolkit.observe()
    kwargs = {"expected_step": 0, "delta_position": [0, 0, 0.02], "gripper": "open"}
    kwargs.update(arguments)
    result = toolkit.execute_tool("adjust_eef", kwargs).result
    assert "error" in result
    assert controller.actions == []
    call(toolkit, "execute_vla")
    assert controller.policy_inputs[-1][1] == TASK


def test_correction_requires_current_observation_even_when_step_is_known():
    toolkit, controller = setup()
    rejected = toolkit.execute_tool("adjust_eef", {
        "expected_step": 0, "delta_position": [0, 0, 0.02], "gripper": "open",
    }).result
    assert "observe" in rejected["error"]
    assert controller.actions == []
    toolkit.observe()
    call(toolkit, "execute_vla")
    rejected = toolkit.execute_tool("adjust_eef", {
        "expected_step": 5, "delta_position": [0, 0, 0.02], "gripper": "open",
    }).result
    assert "observe" in rejected["error"]


def test_rotation_vector_is_in_base_frame_and_composes_with_current_orientation():
    toolkit, controller = setup()
    half = np.sqrt(0.5)
    controller.observation["state"][10:14] = [half, 0, 0, half]
    result = adjust(toolkit, rotation_degrees=[0, 0, 10], max_steps=1)
    target = result["motion_adjustment"]["target_quaternion"]
    expected = [half * np.cos(np.deg2rad(5)), half * np.sin(np.deg2rad(5)),
                half * np.sin(np.deg2rad(5)), half * np.cos(np.deg2rad(5))]
    np.testing.assert_allclose(target, expected, atol=1e-7)
    np.testing.assert_allclose(controller.actions[0][8:11], [0, 0, np.deg2rad(10) / 0.5],
                               atol=1e-6)


def test_keep_gripper_uses_last_confirmed_command_instead_of_opening_or_suffix():
    toolkit, controller = setup()
    call(toolkit, "execute_vla")  # zero command means open, even at a narrow measured width
    controller.observation["state"][14:16] = [0.01, -0.01]
    toolkit.observe()
    result = call(toolkit, "adjust_eef", expected_step=5, delta_position=[0, 0, 0.02])
    assert result["motion_adjustment"]["gripper_command"] == "open"
    assert controller.actions[-1][11] == -1


def test_keep_without_a_confirmed_gripper_command_requires_explicit_choice():
    toolkit, controller = setup()
    toolkit.observe()
    rejected = toolkit.execute_tool("adjust_eef", {
        "expected_step": 0, "delta_position": [0, 0, 0.02],
    }).result
    assert "gripper" in rejected["error"]
    assert controller.actions == []


@pytest.mark.parametrize("max_actions", [0, 11, True, "5"])
def test_invalid_handoff_cannot_consume_the_pending_adjustment(max_actions):
    toolkit, controller = setup()
    adjust(toolkit)
    toolkit.observe()
    assert "error" in toolkit.execute_tool("resume_vla", {"max_actions": max_actions}).result
    result = call(toolkit, "resume_vla")
    assert result["executed_actions"] == 5
    assert len(controller.policy_inputs) == 1


def test_handoff_requires_an_actual_adjustment():
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool("resume_vla", {}).result
    assert controller.policy_inputs == []


def test_direct_adjustment_respects_call_budget_and_terminal_transition():
    toolkit, controller = setup(budget=2)
    result = adjust(toolkit, rotation_degrees=[0, 0, 10], max_steps=25)
    assert result["executed_actions"] == 2
    assert result["transition_budget"] == 2
    toolkit, controller = setup()
    controller.truncated = True
    result = adjust(toolkit, rotation_degrees=[0, 0, 10], max_steps=25)
    assert result["executed_actions"] == 1
    toolkit.observe()
    rejected = toolkit.execute_tool("resume_vla", {}).result
    assert "error" in rejected
    assert controller.policy_inputs == []


def test_motion_tools_are_subject_to_the_same_plan_and_verification_gates(tmp_path):
    from rpent.harness import Harness

    toolkit, controller = setup()
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan(TASK, [{
        "id": "lift", "intent": "lift head", "max_attempts": 8,
        "allowed_tools": ["adjust_eef", "resume_vla", "continue_vla"],
    }])["ok"]
    result = adjust(toolkit)
    assert result["environment"]["success"] is False
    toolkit.observe()
    rejected = toolkit.execute_tool("resume_vla", {}).result
    assert "error" in rejected
    assert controller.policy_inputs == []
    assert call(toolkit, "verify_step", verdict="progress",
                evidence=["Observed a small adjustment; target still needs lifting."],
                summary="Check fresh policy motion next.")["ok"]
    result = call(toolkit, "resume_vla")
    assert result["executed_actions"] == 5


def test_prompt_teaches_grounded_adjust_observe_resume_then_review():
    from robots.robocasa.prompts.system import system_prompt

    prompt = system_prompt()["ACTION"]
    assert "adjust_eef" in prompt
    assert "resume_vla" in prompt
    assert "rotation_degrees" in prompt
    assert "target slides back or disengages" in prompt
    assert "target_reached" in prompt
    assert "simulator internals" in prompt
