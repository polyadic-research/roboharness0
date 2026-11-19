"""The formal controller pauses a grasp before sending its first close action."""

import numpy as np
import pytest

from robots.robocasa.binding import make_robocasa_binding
from robots.robocasa.outcomes import RoboCasaSkillOutcomeAdapter
from robots.robocasa.prompts.system import system_prompt
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.episode.contracts import StepResult
from rpent.episode.controller import EpisodeController
from tests.test_robocasa_toolkit import _pose_observation

TASK = "Pick up the cup and place it on the counter."


def action(*, close=False, dz=0):
    value = np.zeros(12, dtype=np.float32)
    value[7] = dz
    value[11] = 1 if close else -1
    return value


class Simulator:
    def __init__(self):
        self.actions = []
        self.observation = _pose_observation(
            position=(0.2, 0.1, 0.5), gripper_qpos=(0.04, -0.04),
        )
        for camera in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
            self.observation[camera] = np.zeros((256, 256, 3), dtype=np.uint8)

    def step(self, value):
        self.actions.append(value.copy())
        self.observation = {**self.observation, "state": self.observation["state"].copy()}
        self.observation["state"][7:10] += value[5:8] * 0.05
        self.observation["state"][14:16] = (0.01, -0.01) if value[11] >= 0.5 else (0.04, -0.04)
        return StepResult("preclose", len(self.actions), self.observation, 0, False, False, False)


class Policy:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.inputs = []

    def predict(self, observation, instruction, *, max_actions=None):
        self.inputs.append((observation["observation.state"].copy(), instruction))
        return np.stack(self.chunks.pop(0))[:max_actions]


def setup(*chunks, actions_per_policy_request=5):
    simulator = Simulator()
    policy = Policy(chunks)
    controller = EpisodeController(simulator, policy, make_robocasa_binding())
    toolkit = RoboCasaToolkit(
        controller, initial_observation=simulator.observation, instruction=TASK,
        forward_step_guidance=True, actions_per_policy_request=actions_per_policy_request,
    )
    return toolkit, simulator, policy


def call(toolkit, name, **kwargs):
    if name == "smolvla_pick":
        kwargs.setdefault("max_chunks", 1)
    result = toolkit.execute_tool(name, kwargs).result
    assert "error" not in result, result
    return result


def review(toolkit, *, between=True, narrow=True):
    observed = call(toolkit, "observe")
    return call(
        toolkit, "review_grasp", expected_step=observed["step"],
        object_between_fingers=between, narrow_axis_aligned=narrow,
        evidence="Current robot0_eye_in_hand shows the cup between the fingers across its narrow side.",
    )


def test_first_close_is_paused_inside_chunk_and_is_not_a_failed_grasp():
    toolkit, simulator, policy = setup([action(dz=-1), action(close=True), action(close=True)])
    result = call(toolkit, "smolvla_pick")
    assert result["stop_reason"] == "preclose_review_required"
    assert result["failure_reason"] is None
    assert result["executed_actions"] == 1
    assert result["paused_actions"] == 2 and result["cancelled_actions"] == 0
    assert simulator.actions[0][11] == -1
    assert len(policy.inputs) == 1
    assert toolkit._consecutive_closed_empty == 0
    history = result["action_history"]["records"][0]
    assert history["executed_actions"] == 1 and history["actions"][1][11] == 1
    outcome = RoboCasaSkillOutcomeAdapter().classify(tool_name="smolvla_pick", result=result, observation=None)
    assert outcome.status == "progress"


def test_close_at_start_returns_current_images_without_advancing():
    toolkit, simulator, _ = setup([action(close=True)])
    result = call(toolkit, "smolvla_pick")
    assert result["executed_actions"] == 0
    assert simulator.actions == []
    observed = call(toolkit, "observe")
    assert observed["step"] == 0
    assert observed["grasp_review"]["required"]
    assert observed["_image_wrist_bytes"]


def test_approved_review_resumes_original_suffix_and_keeps_original_language():
    toolkit, simulator, policy = setup([action(dz=-1), action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick", instruction="Review only: align the fingers.")
    assert review(toolkit)["approved"]
    result = call(toolkit, "smolvla_pick")
    assert simulator.actions[-1][11] == 1
    assert len(policy.inputs) == 1
    assert {instruction for _, instruction in policy.inputs} == {TASK}
    assert result["environment"]["success"] is False


def test_approved_review_preserves_every_original_action_without_predicting():
    original = [action(dz=-0.4), action(close=True, dz=0.2), action(close=True, dz=0.6)]
    toolkit, simulator, policy = setup(original, [action(dz=-1)])
    paused = call(toolkit, "smolvla_pick")
    assert paused["executed_actions"] == 1
    assert review(toolkit)["approved"]
    resumed = call(toolkit, "continue_vla", max_chunks=1)
    np.testing.assert_array_equal(np.stack(simulator.actions), np.stack(original))
    assert len(policy.inputs) == 1
    assert paused["paused_actions"] == 2 and paused["cancelled_actions"] == 0
    assert resumed["resumed_actions"] == 2


def test_cs04_recorded_chunk_keeps_the_original_step_219_close():
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/robocasa/cs04_preclose_chunk.json").read_text())
    original = np.asarray(fixture["original_chunk"], dtype=np.float32)
    replacement = np.asarray(fixture["old_replacement_chunk"], dtype=np.float32)
    toolkit, simulator, policy = setup(original, replacement, actions_per_policy_request=16)
    paused = call(toolkit, "smolvla_pick")
    assert paused["executed_actions"] == 10 and paused["paused_actions"] == 6
    assert paused["cancelled_actions"] == 0
    review(toolkit)
    resumed = call(toolkit, "continue_vla", max_chunks=1)
    assert resumed["resumed_actions"] == 6 and len(policy.inputs) == 1
    np.testing.assert_array_equal(simulator.actions, original)
    assert simulator.actions[10][11] == np.float32(0.99609375)
    assert replacement[0, 11] == -1


def test_repeated_unreviewed_pause_does_not_predict_again_or_move():
    toolkit, simulator, policy = setup([action(close=True)], [action(dz=-1)])
    call(toolkit, "smolvla_pick")
    repeated = call(toolkit, "continue_vla", max_chunks=1)
    assert repeated["stop_reason"] == "preclose_review_required"
    assert simulator.actions == []
    assert len(policy.inputs) == 1
    assert repeated["paused_actions"] == 1 and repeated["cancelled_actions"] == 0


def test_rejected_review_discards_original_suffix_before_fallback_prediction():
    toolkit, simulator, policy = setup([action(close=True, dz=0.8)], [action(dz=-0.2)])
    call(toolkit, "smolvla_pick")
    assert not review(toolkit, between=False)["approved"]
    result = call(toolkit, "continue_vla", max_chunks=1)
    assert len(policy.inputs) == 2
    assert result["resumed_actions"] == 0
    np.testing.assert_array_equal(simulator.actions, [action(dz=-0.2)])


def test_operation_change_discards_paused_policy_actions():
    toolkit, simulator, policy = setup([action(close=True, dz=0.8)], [action(dz=-0.2)])
    call(toolkit, "smolvla_pick")
    review(toolkit)
    toolkit._reset_grasp_operation()
    result = call(toolkit, "continue_vla", max_chunks=1)
    assert len(policy.inputs) == 2 and result["resumed_actions"] == 0
    np.testing.assert_array_equal(simulator.actions, [action(dz=-0.2)])


@pytest.mark.parametrize("between,narrow", [(False, True), (True, False), (False, False)])
def test_rejected_alignment_keeps_gripper_open_during_correction(between, narrow):
    toolkit, simulator, policy = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    assert not review(toolkit, between=between, narrow=narrow)["approved"]
    denied = toolkit.execute_tool("adjust_eef", {
        "expected_step": 0, "delta_position": [0, 0, 0.01], "gripper": "close",
    }).result
    assert "open" in denied["error"]
    call(toolkit, "adjust_eef", expected_step=0, delta_position=[0, 0, 0.01], gripper="open")
    assert all(value[11] < 0.5 for value in simulator.actions)
    assert review(toolkit)["approved"]
    result = call(toolkit, "resume_vla")
    assert result["handoff"]["fresh_policy_inference"]
    assert policy.inputs[-1][1] == TASK
    np.testing.assert_allclose(policy.inputs[-1][0][9], 0.51, atol=1e-6)
    assert simulator.actions[-1][11] == 1


def test_observation_alone_cannot_approve_closure_or_direct_close():
    toolkit, simulator, policy = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    observed = call(toolkit, "observe")
    denied = toolkit.execute_tool("servo_eef", {
        "target_position": [0.2, 0.1, 0.5], "target_quaternion": [0, 0, 0, 1], "gripper": "close",
    }).result
    assert "review_grasp" in denied["error"]
    result = call(toolkit, "continue_vla", max_chunks=2)
    assert result["stop_reason"] == "preclose_review_required"
    assert simulator.actions == []
    assert len(policy.inputs) == 1
    assert observed["grasp_review"]["required"]


@pytest.mark.parametrize("kwargs", [
    {"expected_step": 1}, {"expected_step": True},
    {"object_between_fingers": "true"}, {"narrow_axis_aligned": None}, {"evidence": " "},
])
def test_invalid_or_stale_review_cannot_authorize_closure(kwargs):
    toolkit, simulator, _ = setup([action(close=True)])
    call(toolkit, "smolvla_pick")
    call(toolkit, "observe")
    args = {"expected_step": 0, "object_between_fingers": True,
            "narrow_axis_aligned": True, "evidence": "Wrist view at step 0."}
    args.update(kwargs)
    result = toolkit.execute_tool("review_grasp", args).result
    assert "error" in result
    assert simulator.actions == []
    assert toolkit.observe()["grasp_review"]["required"]


def test_correction_invalidates_previous_visual_approval():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    review(toolkit)
    call(toolkit, "adjust_eef", expected_step=0, delta_position=[0, 0, 0.01], gripper="open")
    call(toolkit, "observe")
    result = call(toolkit, "resume_vla")
    assert result["stop_reason"] == "preclose_review_required"
    assert all(value[11] < 0.5 for value in simulator.actions)
    # A zero-action handoff must still allow another correction/handoff cycle.
    review(toolkit)
    assert toolkit.observe()["motion_review"]["resume_required"] is False


def test_pose_drift_invalidates_pending_actions_and_requires_another_review():
    toolkit, simulator, _ = setup(
        [action(close=True)], [action(dz=0.4), action(close=True)],
        [action(close=True), action(), action(close=True)],
    )
    call(toolkit, "smolvla_pick")
    review(toolkit)
    simulator.observation["state"][9] += 0.02
    drifted = call(toolkit, "continue_vla", max_chunks=1)
    assert drifted["stop_reason"] == "preclose_review_required"
    assert len(simulator.actions) == 1
    assert drifted["resumed_actions"] == 0


def test_reopening_in_a_resumed_suffix_requires_another_review():
    toolkit, simulator, policy = setup([action(close=True), action(), action(close=True)])
    call(toolkit, "smolvla_pick")
    review(toolkit)
    reopened = call(toolkit, "continue_vla", max_chunks=1)
    assert reopened["stop_reason"] == "preclose_review_required"
    assert [value[11] for value in simulator.actions] == [1, -1]
    assert len(policy.inputs) == 1
    review(toolkit)
    call(toolkit, "continue_vla", max_chunks=1)
    assert [value[11] for value in simulator.actions] == [1, -1, 1]
    assert len(policy.inputs) == 1


def test_non_grasp_articulation_does_not_require_between_fingers_review():
    toolkit, simulator, _ = setup([action(close=True)])
    call(toolkit, "continue_vla", max_chunks=1)
    assert len(simulator.actions) == 1


def test_pause_during_empty_grasp_recovery_preserves_retry_and_descent():
    toolkit, simulator, _ = setup(
        [action(dz=-1), action(close=True)], [action(close=True), action(close=True, dz=1.2)],
    )
    toolkit._consecutive_closed_empty = 1
    paused = call(toolkit, "smolvla_pick")
    assert paused["stop_reason"] == "preclose_review_required"
    assert paused["retry_allowed"]
    assert paused["diagnostics"]["descent_confirmed"]
    assert toolkit._consecutive_closed_empty == 1
    review(toolkit)
    resumed = call(toolkit, "smolvla_pick", max_chunks=2)
    assert resumed["diagnostics"]["descent_confirmed"]
    assert resumed["success"]
    assert toolkit._consecutive_closed_empty == 0
    assert len(simulator.actions) == 4


def test_visual_approval_expires_after_ten_open_actions():
    toolkit, simulator, _ = setup(
        [action()] * 5, [action()] * 5, [action(close=True)],
    )
    review(toolkit)
    result = call(toolkit, "continue_vla", max_chunks=3)
    assert result["stop_reason"] == "preclose_review_required"
    assert result["executed_actions"] == 10
    assert all(value[11] < 0.5 for value in simulator.actions)


def test_visual_review_cannot_bypass_harness_verification(tmp_path):
    from rpent.harness import Harness

    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan(TASK, [{
        "id": "grasp", "intent": "Acquire the cup.", "max_attempts": 8,
        "allowed_tools": ["smolvla_pick", "adjust_eef", "resume_vla"],
    }])["ok"]
    call(toolkit, "smolvla_pick")
    review(toolkit)
    blocked = toolkit.execute_tool("smolvla_pick", {}).result
    assert "error" in blocked
    assert simulator.actions == []
    assert not harness.state.completed
    assert call(toolkit, "verify_step", verdict="progress",
                evidence=["Wrist alignment reviewed; grasp still unconfirmed."],
                summary="Continue acquisition.")["ok"]
    call(toolkit, "smolvla_pick")
    assert simulator.actions[-1][11] == 1


def test_prompt_requires_wrist_review_before_close_and_original_task_wording():
    prompt = system_prompt()["ACTION"]
    assert "preclose_review_required" in prompt
    assert "review_grasp" in prompt
    assert "robot0_eye_in_hand" in prompt
    assert "narrow" in prompt
    assert "Short tasks retain it verbatim" in prompt
    assert "intervene=true when entering a new registered long-task operation" in prompt
    assert "Internal phase bookkeeping alone never requires this language switch" in prompt


def test_two_rejections_allow_policy_close_without_claiming_visual_approval():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    first = review(toolkit, between=False)
    assert not first["approved"] and not first["policy_close_allowed"]
    assert call(toolkit, "smolvla_pick")["executed_actions"] == 0
    second = review(toolkit, between=False)
    assert not second["approved"] and second["policy_close_allowed"]
    assert second["grasp_review"]["consecutive_rejections"] == 2
    direct = toolkit.execute_tool("servo_eef", {
        "target_position": [0.2, 0.1, 0.5],
        "target_quaternion": [0, 0, 0, 1], "gripper": "close",
    }).result
    assert "review_grasp" in direct["error"]
    assert simulator.actions == []
    closed = call(toolkit, "smolvla_pick")
    assert closed["executed_actions"] == 1
    assert len(simulator.actions) == 1 and simulator.actions[0][11] == 1
    assert not toolkit._pick_progress.confirmed
    assert not closed["environment"]["success"]
    assert toolkit._grasp_review_status()["consecutive_rejections"] == 0


def test_reopening_starts_a_new_rejection_budget():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)],
                                 [action(), action(close=True)])
    call(toolkit, "smolvla_pick")
    review(toolkit, between=False)
    review(toolkit, between=False)
    call(toolkit, "smolvla_pick")
    reopened = call(toolkit, "execute_vla", max_actions=2)
    assert reopened["stop_reason"] == "preclose_review_required"
    assert len(simulator.actions) == 2
    assert not review(toolkit, between=False)["policy_close_allowed"]


def test_correction_does_not_reset_rejections_for_the_pending_closure():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    review(toolkit, between=False)
    call(toolkit, "adjust_eef", expected_step=0, delta_position=[0, 0, 0.01], gripper="open")
    assert review(toolkit, narrow=False)["policy_close_allowed"]
    resumed = call(toolkit, "resume_vla")
    assert resumed["executed_actions"] == 1


def fixed_control_review(toolkit):
    observed = call(toolkit, "observe")
    return call(
        toolkit, "review_grasp", expected_step=observed["step"],
        object_between_fingers=False, narrow_axis_aligned=False,
        closure_target="fixed_control",
        evidence="Wrist view shows a faucet lever beside the left finger; nothing is being picked up.",
    )


def test_fixed_control_closures_pass_without_alignment_review():
    """1.2.1: D08 policy closed on a faucet lever; seven alignment reviews all rejected."""
    cycle = [action(close=True), action(), action(close=True), action(), action(close=True)]
    toolkit, simulator, _ = setup([action(close=True)], cycle)
    assert call(toolkit, "smolvla_pick")["stop_reason"] == "preclose_review_required"
    result = fixed_control_review(toolkit)
    assert result["closure_target"] == "fixed_control"
    assert result["policy_close_allowed"] and not result["approved"]
    assert result["grasp_review"]["at_fixed_control"]
    assert not result["grasp_review"]["required"]
    resumed = call(toolkit, "continue_vla", max_chunks=1)
    assert resumed.get("stop_reason") != "preclose_review_required"
    assert [value[11] for value in simulator.actions] == [1, -1, 1, -1, 1]
    assert not toolkit.observe()["grasp_review"]["required"]


def test_fixed_control_exemption_ends_away_from_the_control():
    # Policy actions are clipped to one unit (5 cm here), so leave in two steps.
    toolkit, simulator, _ = setup([action(close=True)], [action(dz=1), action(dz=1), action(close=True)])
    call(toolkit, "smolvla_pick")
    fixed_control_review(toolkit)
    moved = call(toolkit, "continue_vla", max_chunks=1)
    assert moved["stop_reason"] == "preclose_review_required"
    assert len(simulator.actions) == 2 and all(value[11] < 0.5 for value in simulator.actions)


def test_fixed_control_exemption_ends_when_the_base_moves():
    class BaseSimulator(Simulator):
        def step(self, value):
            result = super().step(value)
            self.observation["state"][0] += value[0] * 0.05
            return result

    simulator = BaseSimulator()
    policy = Policy([[action(close=True)], [np.r_[1, np.zeros(10), -1].astype(np.float32), action(close=True)]])
    controller = EpisodeController(simulator, policy, make_robocasa_binding())
    toolkit = RoboCasaToolkit(
        controller, initial_observation=simulator.observation, instruction=TASK,
        forward_step_guidance=True, actions_per_policy_request=5,
    )
    call(toolkit, "smolvla_pick")
    fixed_control_review(toolkit)
    moved = call(toolkit, "continue_vla", max_chunks=1)
    assert moved["stop_reason"] == "preclose_review_required"
    assert len(simulator.actions) == 1


def test_fixed_control_exemption_ends_with_the_operation():
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    fixed_control_review(toolkit)
    toolkit._reset_grasp_operation()
    assert call(toolkit, "smolvla_pick")["stop_reason"] == "preclose_review_required"
    assert simulator.actions == []


def test_fixed_control_permits_a_direct_close_at_the_control():
    toolkit, simulator, _ = setup([action(close=True)])
    call(toolkit, "smolvla_pick")
    fixed_control_review(toolkit)
    result = toolkit.execute_tool("servo_eef", {
        "target_position": [0.2, 0.1, 0.5], "target_quaternion": [0, 0, 0, 1], "gripper": "close",
    }).result
    assert "review_grasp" not in str(result.get("error", ""))


def test_movable_object_review_is_unchanged_by_default():
    toolkit, _, _ = setup([action(close=True)])
    call(toolkit, "smolvla_pick")
    result = review(toolkit, between=False)
    assert result["closure_target"] == "movable_object"
    assert not result["policy_close_allowed"]
    assert toolkit._fixed_control_zones == []


def test_unknown_closure_target_is_rejected():
    toolkit, simulator, _ = setup([action(close=True)])
    call(toolkit, "smolvla_pick")
    call(toolkit, "observe")
    result = toolkit.execute_tool("review_grasp", {
        "expected_step": 0, "object_between_fingers": False, "narrow_axis_aligned": False,
        "evidence": "Wrist view at step 0.", "closure_target": "fixture",
    }).result
    assert "error" in result
    assert toolkit._fixed_control_zones == []
    assert simulator.actions == []


def test_prompt_and_tool_describe_fixed_control_closures():
    prompt = system_prompt()["ACTION"]
    assert 'closure_target="fixed_control"' in prompt
    assert "lifted or carried" in prompt
    toolkit, _, _ = setup([action(close=True)])
    spec, _ = toolkit._tools["review_grasp"]
    assert spec["input_schema"]["properties"]["closure_target"]["enum"] == ["movable_object", "fixed_control"]
    assert "closure_target" not in spec["input_schema"]["required"]


# --- 1.2.2: an unseen target is not a misaligned one ------------------------------------


def test_unseen_target_lets_the_policy_close_without_a_correction():
    """1.2.1 full eval D05: two 'occluded' rejections lifted the hand, the later grasp slipped."""
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    observed = call(toolkit, "observe")
    result = call(
        toolkit, "review_grasp", expected_step=observed["step"], object_between_fingers=False,
        narrow_axis_aligned=False, target_visible=False,
        evidence="Wrist view at the source: the cube is hidden under the finger housing even after zoom.",
    )
    assert not result["approved"] and result["policy_close_allowed"] and result["target_visible"] is False
    assert "do not correct the pose" in result["next_action"]
    assert result["release_basis"] == "target_not_visible"
    assert result["grasp_review"]["policy_close_allowed"]  # the two-rejection release, without a correction
    direct = toolkit.execute_tool("servo_eef", {
        "target_position": [0.2, 0.1, 0.5], "target_quaternion": [0, 0, 0, 1], "gripper": "close",
    }).result
    assert "review_grasp" in direct["error"]  # only the policy may close
    closed = call(toolkit, "smolvla_pick")
    assert closed["executed_actions"] == 1 and simulator.actions[-1][11] == 1
    assert not toolkit._pick_progress.confirmed


def test_visible_review_is_unchanged_and_target_visible_must_be_boolean():
    toolkit, _, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    rejected = review(toolkit, between=False)
    assert not rejected["approved"] and not rejected["policy_close_allowed"]
    observed = call(toolkit, "observe")
    bad = toolkit.execute_tool("review_grasp", {
        "expected_step": observed["step"], "object_between_fingers": False, "narrow_axis_aligned": False,
        "evidence": "unclear", "target_visible": "no",
    }).result
    assert "target_visible" in bad["error"]


def test_prompt_and_tool_separate_unseen_from_misaligned():
    prompt = system_prompt()["ACTION"]
    assert "target_visible=false" in prompt
    assert "treat an occluded or uncertain check as false" not in prompt
    assert "A grasp_lost result means the held object left the hand" in prompt
    toolkit, _, _ = setup([action(close=True)])
    spec, _ = toolkit._tools["review_grasp"]
    visible = spec["input_schema"]["properties"]["target_visible"]
    assert visible["type"] == "boolean" and visible["default"] is True
    assert "target_visible" not in spec["input_schema"]["required"]
    assert "unseen target is not misaligned" in spec["description"]
