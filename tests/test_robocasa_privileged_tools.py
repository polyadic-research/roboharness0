"""Public-state privileged controls: bounds, interruption and audit contracts."""

import io

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.toolkit import RoboCasaToolkit, _rotate_base_quaternion
from rpent.episode.contracts import StepResult
from tests.test_robocasa_motion_handoff import MotionController, call


class PublicController(MotionController):
    def __init__(self, *, stuck=False, terminal=None, width=None, torso_gain=0.05, torso_max=None):
        super().__init__()
        self.stuck, self.terminal, self.width = stuck, terminal, width
        self.torso_gain, self.torso_max = torso_gain, torso_max
        self.observation["state"][3:7] = [0, 0, 0, 1]

    def execute_action(self, action, *, _action_index=0):
        before = self.observation["state"].copy()
        super().execute_action(action, _action_index=_action_index)
        state = self.observation["state"]
        yaw = 2 * np.arctan2(before[5], before[6])
        if self.stuck:
            state[:] = before
        else:
            state[:2] += np.array([[np.cos(yaw), -np.sin(yaw)],
                                   [np.sin(yaw), np.cos(yaw)]]) @ (action[:2] * 0.05)
            state[3:7] = _rotate_base_quaternion(before[3:7], np.array([0, 0, action[2] * 0.1]))
            # Pinned Omron center rides on the torso; relative EEF z cancels.
            state[2] += action[3] * self.torso_gain
            if self.torso_max is not None:
                state[2] = min(state[2], self.torso_max)
            state[10:14] = _rotate_base_quaternion(before[10:14], action[8:11] * 0.5)
        if self.width is not None:
            state[14:16] = [self.width / 2, -self.width / 2]
        return StepResult(episode_id="test-motion", step_id=self.step,
                          observation=self.observation, reward=0,
                          success=self.terminal == "success", terminated=self.terminal == "terminated",
                          truncated=self.terminal == "truncated")


def setup(**kwargs):
    controller = PublicController(**{k: kwargs.pop(k) for k in ("stuck", "terminal", "width", "torso_gain", "torso_max")
                                     if k in kwargs})
    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Move the observed cup.", episode_horizon=kwargs.pop("horizon", 500),
                              **kwargs)
    return toolkit, controller


def args(tool, **overrides):
    values = {"navigate_to": {"target": {"relative": [0.2, 0, 20]}, "gripper": "open"},
              "move_eef_path": {"waypoints": [{"position": [0.2, 0.1, 0.6]}], "gripper": "open"},
              "adjust_torso_to": {"delta_m": 0.1}}[tool]
    return {"expected_step": 0, **values, **overrides}


PHYSICAL = ["navigate_to", "move_eef_path", "adjust_torso_to"]
# Plan-step control fields (1.0.10 C): privileged tools need takeover/transition.
TAKEOVER_FIELDS = {"skill_row": "none", "tier": "none", "control_mode": "takeover"}


def controller_z():
    return float(PublicController().observation["state"][2])


@pytest.mark.parametrize("tool", PHYSICAL)
def test_registered_physical_tools_require_exact_fresh_observation(tool):
    toolkit, controller = setup()
    assert tool in toolkit.PHYSICAL_TOOLS
    assert "observe" in toolkit.execute_tool(tool, args(tool)).result["error"]
    toolkit.observe()
    assert "expected_step" in toolkit.execute_tool(tool, args(tool, expected_step=1)).result["error"]
    assert controller.actions == []


@pytest.mark.parametrize("tool,override", [
    ("navigate_to", {"target": {"relative": [3.01, 0, 0]}}),
    ("navigate_to", {"target": {"relative": [0, 0, 180.1]}}),
    ("navigate_to", {"max_steps": 201}),
    ("navigate_to", {"target": {"relative": [0, 0, 0], "point": [1, 0, 0]}}),
    ("navigate_to", {"target": {"point": [4, 0, 0], "standoff_m": 0.2}}),
    ("move_eef_path", {"waypoints": []}),
    ("move_eef_path", {"waypoints": [{"position": [0.2, 0.1, 0.5]}] * 9}),
    ("move_eef_path", {"waypoints": [{"position": [0.2, 0.1, 0.801]}]}),
    ("move_eef_path", {"waypoints": [{"position": [0.2, 0.1, 0.6], "rotation_degrees": [0, 0, 91]}]}),
    ("move_eef_path", {"waypoints": [{"position": [0.2, 0.1, 0.6]}, {"position": [0.2, 0.1, 0.901]}]}),
    ("move_eef_path", {"max_steps": 151}),
    ("adjust_torso_to", {"delta_m": 2}), ("adjust_torso_to", {"max_steps": 51}),
    ("adjust_torso_to", {"delta_m": float("nan")}),
])
def test_over_limit_rejected_before_any_action(tool, override):
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool(tool, args(tool, **override)).result
    assert controller.actions == []


def seed_closed(toolkit, controller):
    action = np.zeros(12, np.float32)
    action[11] = 1
    toolkit._operation.execute_actions((action,))
    toolkit._pick_progress.retention_monitor_active = True
    controller.observation["state"][14:16] = [0.01, -0.01]
    toolkit._pick_progress.final_opening = 0.02
    toolkit.observe()


@pytest.mark.parametrize("tool", PHYSICAL)
@pytest.mark.parametrize("terminal,reason", [("success", "success"), ("terminated", "terminated"), ("truncated", "terminated")])
def test_environment_interrupts_immediately(tool, terminal, reason):
    toolkit, controller = setup(terminal=terminal)
    if tool == "adjust_torso_to":
        toolkit._operation.execute_actions((np.zeros(12, np.float32),))
        # Seed only command history; do not run another transition after terminal.
        toolkit._operation._metrics.update(simulator_success=False, terminated=False, truncated=False)
    toolkit.observe()
    result = call(toolkit, tool, **args(tool, expected_step=controller.step))
    assert result["interrupt_reason"] == reason
    assert result["interrupted"] and result["executed_actions"] == 1


@pytest.mark.parametrize("width", [0.001, 0.04])
@pytest.mark.parametrize("tool", PHYSICAL)
def test_grasp_lost_cancels_motion_and_suffix(tool, width):
    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    controller.width = width
    result = call(toolkit, tool, **args(tool, expected_step=1, **({"gripper": "keep"} if tool != "adjust_torso_to" else {})))
    assert result["interrupted"] and result["interrupt_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1


def test_stalled_base_and_deviating_path_stop_and_leave_zero_command():
    toolkit, controller = setup(stuck=True)
    toolkit.observe()
    result = call(toolkit, "navigate_to", **args("navigate_to"))
    assert result["interrupt_reason"] == "base_stalled"
    np.testing.assert_array_equal(controller.actions[-1][:11], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    toolkit, controller = setup(stuck=True)
    toolkit.observe()
    points = [{"position": [0.2, 0.1, 0.7]}, {"position": [0.3, 0.1, 0.7]}]
    result = call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=points))
    assert result["interrupt_reason"] == "deviation"
    assert len(result["waypoint_results"]) == 1
    assert not result["waypoint_results"][0]["target_reached"]
    assert np.all(controller.actions[-1][5:11] == 0)


@pytest.mark.parametrize("tool", PHYSICAL)
def test_budget_return_history_and_step_accounting(tool):
    toolkit, controller = setup(horizon=4)
    if tool == "adjust_torso_to":
        toolkit._operation.execute_actions((np.zeros(12, np.float32),))
    toolkit.observe()
    result = call(toolkit, tool, **args(tool, expected_step=controller.step, max_steps=3))
    assert result["executed_actions"] <= 3
    assert result["interrupt_reason"] == "budget"
    for field in ("interrupted", "measured_before", "measured_after", "target_reached", "control_mode"):
        assert field in result
    records = [r for r in result["action_history"]["records"] if r["control_source"] == tool]
    assert sum(r["executed_actions"] for r in records) == result["executed_actions"]
    assert result["episode_budget"]["steps_used"] == controller.step


def test_holding_keeps_the_policy_carry_speed_with_a_gentler_ramp_and_turn():
    from robots.robocasa.privileged_tools import (
        BASE_COMMAND_MAX,
        BASE_COMMAND_MIN,
        BASE_COMMAND_RAMP,
        BASE_TURN_COMMAND_MAX,
        HOLD_BASE_COMMAND_MAX,
        HOLD_BASE_COMMAND_RAMP,
        HOLD_BASE_TURN_COMMAND_MAX,
    )

    def travel(holding):
        toolkit, controller = setup()
        if holding:
            seed_closed(toolkit, controller)
        else:
            toolkit.observe()
        start = len(controller.actions)
        result = call(toolkit, "navigate_to", **args("navigate_to", expected_step=controller.step,
                                                      target={"relative": [1.2, 0, 90]},
                                                      gripper="keep" if holding else "open"))
        assert result["target_reached"] and result["holding_speed_limited"] is holding
        body = np.array(controller.actions[start:])
        body = body[body[:, 4] > 0][:-1]  # body-mode steps before the final zero stop
        return np.linalg.norm(body[:, :2], axis=1), np.abs(body[:, 2])

    free_travel, free_turn = travel(False)
    hold_travel, hold_turn = travel(True)
    for speeds, cap, ramp in ((free_travel, BASE_COMMAND_MAX, BASE_COMMAND_RAMP),
                              (hold_travel, HOLD_BASE_COMMAND_MAX, HOLD_BASE_COMMAND_RAMP)):
        moving = speeds[speeds > 1e-9]
        # Every travel starts at the friction floor or one ramp step above it,
        # accelerates by at most the ramp per step and reaches the policy's carry level.
        assert moving[0] == pytest.approx(min(cap, max(BASE_COMMAND_MIN, ramp)))
        assert np.max(np.diff(moving)) <= ramp + 1e-7
        assert np.max(moving) == pytest.approx(cap)
    assert np.max(hold_travel) == pytest.approx(np.max(free_travel))  # holding is no slower than free travel
    assert np.max(hold_turn) == pytest.approx(HOLD_BASE_TURN_COMMAND_MAX)
    assert np.max(free_turn) == pytest.approx(BASE_TURN_COMMAND_MAX)
    assert np.max(np.diff(hold_turn[hold_turn > 1e-9])) <= HOLD_BASE_COMMAND_RAMP + 1e-7


def test_holding_reduces_the_eef_step():
    from robots.robocasa.privileged_tools import HOLD_EEF_STEP_M

    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    result = call(toolkit, "move_eef_path", **args("move_eef_path", expected_step=1, gripper="keep"))
    assert result["holding_speed_limited"]
    actions = np.array(controller.actions[1:])
    assert np.max(np.linalg.norm(actions[:, 5:8] * 0.05, axis=1)) <= HOLD_EEF_STEP_M + 1e-7


@pytest.mark.parametrize("tool", ["navigate_to", "move_eef_path"])
def test_unreviewed_close_rejected(tool):
    toolkit, controller = setup()
    toolkit._preclose_active = True
    toolkit._preclose_review_enabled = True
    toolkit.observe()
    assert "review_grasp" in toolkit.execute_tool(tool, args(tool, gripper="close")).result["error"]
    assert controller.actions == []


def test_path_reaches_three_waypoints_then_closes_only_at_the_reviewed_pose():
    toolkit, controller = setup()
    toolkit._operation.execute_actions((np.zeros(12, np.float32),))
    toolkit.observe()
    points = [{"position": [0.2, 0.1, z]} for z in (0.55, 0.6, 0.65)]
    # 1.0.11 A1: closing an open gripper needs a review of the closing pose,
    # with or without an active policy pick.
    assert "review_grasp" in toolkit.execute_tool(
        "move_eef_path", args("move_eef_path", expected_step=1, waypoints=points, gripper="close")).result["error"]
    result = call(toolkit, "move_eef_path", **args("move_eef_path", expected_step=1, waypoints=points, gripper="open"))
    assert result["target_reached"] and not result["interrupted"]
    assert len(result["waypoint_results"]) == 3
    assert all(r["target_reached"] for r in result["waypoint_results"])
    assert all(a[11] < 0.5 for a in controller.actions[1:])
    toolkit.observe()
    review = call(toolkit, "review_grasp", expected_step=controller.step, object_between_fingers=True,
                  narrow_axis_aligned=True, evidence="Handle centred between both fingers.")
    assert review["approved"] and "move_eef_path" in review["next_action"]
    start = controller.step
    result = call(toolkit, "move_eef_path", **args("move_eef_path", expected_step=start,
                                                    waypoints=[points[-1]], gripper="close"))
    assert result["gripper_command_executed"] and controller.actions[-1][11] >= 0.5
    assert all(a[11] >= 0.5 for a in controller.actions[start:])


def test_point_navigation_uses_initial_base_frame_and_standoff():
    toolkit, controller = setup()
    controller.observation["state"][:2] = [2, 3]
    controller.observation["state"][3:7] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    toolkit.observe()
    result = call(toolkit, "navigate_to", **args("navigate_to", target={"point": [0.4, 0, 0.8], "standoff_m": 0.1}))
    assert result["target_reached"]
    np.testing.assert_allclose(result["target_base_position_world"], [2, 3.3], atol=1e-6)
    assert np.all(np.array(controller.actions)[:, 5:11] == 0)


def test_standoff_can_request_backing_away_from_a_nearby_point():
    toolkit, controller = setup()
    toolkit.observe()
    result = call(toolkit, "navigate_to", **args("navigate_to", target={"point": [0.1, 0, 0.8], "standoff_m": 0.3}))
    assert result["target_reached"]
    np.testing.assert_allclose(result["target_base_position_world"], [-0.2, 0], atol=1e-6)


def test_navigation_final_heading_survives_small_coast_at_position_boundary():
    toolkit, controller = setup()
    toolkit.observe()
    original = controller.execute_action
    drifted = False
    drift_index = None

    def drift(action, *, _action_index=0):
        nonlocal drifted, drift_index
        transition = original(action, _action_index=_action_index)
        # Measured inertia can cross the entry threshold just after arrival.
        if not drifted and action[2] and controller.observation["state"][0] > 0.18:
            controller.observation["state"][0] = 0.1799
            drifted = True
            drift_index = len(controller.actions) - 1
        return transition

    controller.execute_action = drift
    result = call(toolkit, "navigate_to", **args("navigate_to"))
    assert drift_index is not None
    assert controller.actions[drift_index + 1][2] > 0  # remain in final heading alignment
    assert result["heading_error_rad"] < np.deg2rad(3)


def test_zoom_cannot_reveal_the_center_image_hidden_from_llm_context():
    toolkit, controller = setup()
    toolkit.observe()
    result = toolkit.execute_tool("zoom", {"camera_id": "robot0_agentview_center", "box": [0, 0, 1, 1]}).result
    assert "error" in result
    assert controller.actions == []


def test_navigation_holds_public_center_when_yaw_pivot_is_offset():
    toolkit, controller = setup()
    toolkit.observe()
    original = controller.execute_action

    def offset_center(action, *, _action_index=0):
        before = controller.observation["state"].copy()
        transition = original(action, _action_index=_action_index)
        yaw = 2 * np.arctan2(before[5], before[6])
        change = float(action[2]) * 0.1
        offset = np.array([0.21 * (np.cos(change) - 1), 0.21 * np.sin(change)])
        transition.observation["state"][:2] += np.array([[np.cos(yaw), -np.sin(yaw)],
                                                         [np.sin(yaw), np.cos(yaw)]]) @ offset
        return transition

    controller.execute_action = offset_center
    result = call(toolkit, "navigate_to", **args("navigate_to", target={"relative": [0.2, 0, 90]}))
    assert result["target_reached"]
    assert result["position_error_m"] <= 0.02
    assert result["heading_error_rad"] <= np.deg2rad(3)
    assert np.all(np.array(controller.actions)[:, 5:11] == 0)


def test_torso_closes_the_loop_on_public_base_height_with_zero_arm_deltas():
    # Gain measured on SDS: about 4.65 mm of public base-center lift per step
    # per unit native torso command.
    toolkit, controller = setup(torso_gain=0.00465)
    toolkit._operation.execute_actions((np.zeros(12, np.float32),))
    toolkit.observe()
    start_z = controller.observation["state"][2]
    result = call(toolkit, "adjust_torso_to", **args("adjust_torso_to", expected_step=1))
    assert result["target_reached"] and result["interrupt_reason"] is None
    assert result["torso_measurement"] == "public_base_center_z"
    assert abs(controller.observation["state"][2] - start_z - 0.1) <= 0.003
    assert abs(result["measured_delta_m"] - 0.1) <= 0.003
    assert result["transitions_used"] <= 50
    assert abs(result["measured_eef_delta_z_m"]) < 1e-6
    actions = np.array(controller.actions[1:])
    assert np.all(actions[:, 5:11] == 0) and np.all(actions[:, :3] == 0)
    assert np.max(np.abs(actions[:, 3])) <= 1.0


def prime(toolkit):
    # adjust_torso_to keeps the confirmed gripper command, so it needs one.
    toolkit._operation.execute_actions((np.zeros(12, np.float32),))
    toolkit.observe()
    return 1


def test_torso_reports_remaining_travel_when_one_call_cannot_finish():
    toolkit, controller = setup(torso_gain=0.00465)
    step = prime(toolkit)
    result = call(toolkit, "adjust_torso_to", **args("adjust_torso_to", expected_step=step, delta_m=0.3))
    assert not result["target_reached"] and result["interrupt_reason"] == "budget"
    assert 0.15 < result["measured_delta_m"] < 0.3
    assert abs(result["measured_delta_m"] + result["remaining_delta_m"] - 0.3) < 1e-6


def test_torso_stops_without_measured_lift_at_the_joint_limit():
    toolkit, controller = setup(torso_gain=0.00465, torso_max=controller_z() + 0.02)
    step = prime(toolkit)
    result = call(toolkit, "adjust_torso_to", **args("adjust_torso_to", expected_step=step, delta_m=0.2))
    assert not result["target_reached"]
    assert result["interrupt_reason"] == "deviation" and result["torso_limit_suspected"]
    assert result["transitions_used"] < 20


def test_torso_lowers_with_reduced_command_while_holding():
    toolkit, controller = setup(torso_gain=0.00465)
    seed_closed(toolkit, controller)
    toolkit.observe()
    step = controller.step
    result = call(toolkit, "adjust_torso_to", **args("adjust_torso_to", expected_step=step, delta_m=-0.05))
    assert result["holding_speed_limited"]
    commands = np.array(controller.actions[step + 1:])[:, 3]
    assert np.max(np.abs(commands)) <= 0.5
    assert result["target_reached"]


@pytest.mark.parametrize("tool", ["navigate_to", "adjust_torso_to"])
def test_body_motion_first_reanchors_arm_goal_with_zero_arm_mode_step(tool):
    toolkit, controller = setup(torso_gain=0.00465)
    step = prime(toolkit)
    result = call(toolkit, tool, **args(tool, expected_step=step))
    first = np.array(controller.actions[step])
    assert first[4] == -1 and np.all(first[:4] == 0) and np.all(first[5:11] == 0)
    assert result["arm_goal_sync_step"]
    assert all(action[4] == 1 for action in controller.actions[step + 1:])


@pytest.mark.parametrize("tool", PHYSICAL)
def test_privileged_motion_hands_back_through_continue_vla_not_resume_vla(tool):
    # pi05-style toolkit: forward step guidance allows instruction intervention.
    toolkit, controller = setup(torso_gain=0.00465, forward_step_guidance=True, actions_per_policy_request=5)
    step = prime(toolkit)
    call(toolkit, tool, **args(tool, expected_step=step))
    review = toolkit._motion_review()
    assert review["observe_required"] and not review["resume_required"]
    # 1.0.11 C3: the error names the handback path, before or after observe.
    for _ in range(2):
        error = toolkit.execute_tool("resume_vla", {}).result["error"]
        assert f"after {tool}" in error and "continue_vla" in error and "intervene=true" in error
        toolkit.observe()
    toolkit.observe()
    before = controller.step
    result = call(toolkit, "continue_vla", instruction="Place the bowl in the fridge.",
                  intervene=True, max_chunks=1)
    assert result["executed_actions"] > 0 and controller.step > before


def test_zoom_is_exact_crop_enlargement_without_render_or_actions():
    toolkit, controller = setup()
    camera = "robot0_agentview_left"
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    controller.observation[camera] = pixels
    toolkit._operation.observation[camera] = pixels
    observed = toolkit.observe()
    result = call(toolkit, "zoom", camera_id=camera, box=[1, 2, 5, 8], scale=4)
    original = Image.open(io.BytesIO(observed["_image_bytes"]))
    expected = np.array(original.crop((2, 1, 8, 5)).resize((24, 16), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(result["_image_bytes"]))), expected)
    assert result["source_step"] == 0 and controller.actions == []
    assert toolkit.evaluation_metrics["step_count"] == 0


@pytest.mark.parametrize("box", [[-1, 0, 2, 2], [0, 0, 999, 999], [2, 2, 1, 1], [0.5, 0, 1, 1]])
def test_zoom_rejects_out_of_bounds_box(box):
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool("zoom", {"camera_id": "robot0_agentview_left", "box": box}).result
    assert controller.actions == []


def test_privileged_call_uses_harness_attempt_and_verification_gate(tmp_path):
    from rpent.harness import Harness

    toolkit, controller = setup()
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan("Move.", [{"id": "motion", "intent": "Move.",
                                                "max_attempts": 3, "allowed_tools": PHYSICAL,
                                                **TAKEOVER_FIELDS}])["ok"]
    toolkit.observe()
    call(toolkit, "navigate_to", **args("navigate_to"))
    assert harness.state.snapshot()["active_step_attempts"] == 1
    toolkit.observe()
    assert "error" in toolkit.execute_tool("move_eef_path", args("move_eef_path", expected_step=controller.step)).result


@pytest.mark.parametrize("tool", PHYSICAL)
@pytest.mark.parametrize("override", [{"expected_step": True}, {"max_steps": True}, {"max_steps": 1}, {"max_steps": 2.5}])
def test_invalid_common_arguments_do_not_arm_motion(tool, override):
    toolkit, controller = setup()
    toolkit.observe()
    assert "error" in toolkit.execute_tool(tool, args(tool, **override)).result
    assert controller.actions == []
    assert not toolkit._motion_observe_required


def test_reviewed_close_must_be_reviewed_again_after_path_pose_drift():
    toolkit, controller = setup()
    toolkit._preclose_active = True
    toolkit._preclose_review_enabled = True
    toolkit.observe()
    toolkit.review_grasp(expected_step=0, object_between_fingers=True, narrow_axis_aligned=True,
                         evidence="Aligned current wrist view.")
    result = call(toolkit, "move_eef_path", **args("move_eef_path", gripper="close"))
    assert result["review_required"] and result["stop_reason"] == "preclose_review_required"
    assert not result["gripper_command_executed"]
    assert all(a[11] < 0.5 for a in controller.actions)


def test_path_partial_fault_preserves_source_and_requires_fresh_observe():
    from rpent.episode.controller import EpisodeExecutionError

    toolkit, controller = setup()
    toolkit.observe()
    original = controller.execute_action

    def fail(action, *, _action_index=0):
        if controller.step:
            raise EpisodeExecutionError("probe fault", confirmed_transitions=(), requested_actions=1,
                                        attempted_action_index=_action_index)
        return original(action, _action_index=_action_index)

    controller.execute_action = fail
    result = toolkit.execute_tool("move_eef_path", args("move_eef_path")).result
    assert "error" in result and result["motion_review"]["observe_required"]
    records = result["action_history"]["records"]
    assert all(r["control_source"] == "move_eef_path" for r in records)
    assert sum(r["executed_actions"] for r in records) == 1


def test_zoom_rejects_unobserved_and_stale_sources_and_unknown_camera():
    toolkit, controller = setup()
    values = {"camera_id": "robot0_agentview_left", "box": [0, 0, 1, 1]}
    assert "observe" in toolkit.execute_tool("zoom", values).result["error"]
    toolkit.observe()
    assert "error" in toolkit.execute_tool("zoom", {**values, "camera_id": "hidden_camera"}).result
    call(toolkit, "navigate_to", **args("navigate_to"))
    assert "observe" in toolkit.execute_tool("zoom", values).result["error"]


def test_read_only_zoom_preserves_pending_harness_verification(tmp_path):
    from rpent.harness import Harness

    toolkit, controller = setup()
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan("Move.", [{"id": "move", "intent": "Move.",
                                                "max_attempts": 2, "allowed_tools": ["navigate_to"],
                                                **TAKEOVER_FIELDS}])["ok"]
    toolkit.observe()
    call(toolkit, "navigate_to", **args("navigate_to"))
    toolkit.observe()
    step = controller.step
    call(toolkit, "zoom", camera_id="robot0_agentview_left", box=[0, 0, 1, 1])
    assert harness.state.awaiting_verification
    assert harness.state.snapshot()["active_step_attempts"] == 1
    assert controller.step == step


def test_final_zero_step_measurement_is_used_for_reported_convergence():
    toolkit, controller = setup()
    toolkit.observe()
    original = controller.execute_action

    def coast(action, *, _action_index=0):
        transition = original(action, _action_index=_action_index)
        if not np.any(action[:4]) and not np.any(action[5:11]):
            transition.observation["state"][0] += 0.05
        return transition

    controller.execute_action = coast
    result = call(toolkit, "navigate_to", **args("navigate_to"))
    assert not result["target_reached"]
    assert result["position_error_m"] > 0.02
    assert result["measured_after"]["base_position"] == controller.observation["state"][:3].tolist()
