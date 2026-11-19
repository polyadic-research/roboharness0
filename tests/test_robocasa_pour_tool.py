"""1.0.12 pour_into: rim-pivot tilt of a held container over a receiving point."""

import numpy as np
import pytest

from robots.robocasa import control_modes, pour_tool
from robots.robocasa.toolkit import _quaternion_error_rotvec
from tests.test_robocasa_motion_handoff import call
from tests.test_robocasa_privileged_tools import PublicController, seed_closed, setup

TARGET = [0.40, 0.45, 0.30]  # receiving surface centre, robot base frame


def held(**kwargs):
    toolkit, controller = setup(**kwargs)
    seed_closed(toolkit, controller)
    return toolkit, controller


def pour(toolkit, controller, **overrides):
    return call(toolkit, "pour_into", expected_step=controller.step, target=TARGET, **overrides)


def test_pour_into_is_a_privileged_physical_tool_in_the_shared_vocabulary():
    toolkit, _ = setup()
    assert "pour_into" in toolkit.PHYSICAL_TOOLS
    assert "pour_into" in control_modes.PRIVILEGED_TOOLS
    for tools in control_modes.PRIVILEGED_TOOLS_BY_MODE.values():
        assert "pour_into" in tools
    schema = toolkit._tools["pour_into"][0]["input_schema"]
    assert schema["required"] == ["expected_step", "target"]
    assert schema["properties"]["max_steps"]["maximum"] == pour_tool.POUR_CEILING


def test_requires_fresh_observe_and_a_held_container_before_any_action():
    toolkit, controller = setup()
    assert "observe" in toolkit.execute_tool("pour_into", {"expected_step": 0, "target": TARGET}).result["error"]
    toolkit.observe()
    # Open gripper, no confirmed closure: refused before any action.
    result = toolkit.execute_tool("pour_into", {"expected_step": 0, "target": TARGET}).result
    assert "held container" in result["error"]
    assert controller.actions == []


@pytest.mark.parametrize("override", [
    {"target": [0.4, 0.45]}, {"step_degrees": 3}, {"max_degrees": 130}, {"clearance_m": 0.5},
    {"container_depth_m": 0.01}, {"rim": [1, 2]}, {"max_steps": 251}, {"max_steps": 50}, {"return_upright": "yes"},
])
def test_invalid_arguments_rejected_before_any_action(override):
    toolkit, controller = held()
    before = len(controller.actions)
    result = toolkit.execute_tool("pour_into", {"expected_step": controller.step, "target": TARGET, **override}).result
    assert "error" in result
    assert len(controller.actions) == before


def test_geometry_puts_the_pour_rim_on_the_target_side_away_from_the_handle():
    toolkit, controller = held()
    state = toolkit._operation.observation["state"]
    g = pour_tool.plan_geometry(state, np.array(TARGET), rim=None, container_center=None,
                                container_depth_m=0.08, clearance_m=None)
    # Identity hand orientation: the handle axis is +x, the body sits 0.20 m ahead.
    np.testing.assert_allclose(g["handle_dir"], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(g["center"], state[7:10] + [0.20, 0, 0.05], atol=1e-9)
    np.testing.assert_allclose(g["pour_dir"], [0, 1, 0], atol=1e-6)  # target is to the left
    np.testing.assert_allclose(g["rim"], g["center"] + [0, 0.11, -0.01], atol=1e-6)
    np.testing.assert_allclose(g["rim_target"], [TARGET[0], TARGET[1], TARGET[2] + 0.12], atol=1e-9)
    # +theta about the axis raises the body relative to the rim.
    body = pour_tool._axis_angle_mat(g["axis"], np.radians(30)) @ (-g["pour_dir"])
    assert body[2] > 0
    # A target on the far side (beyond the body, along the handle axis) pours straight over the far rim.
    g_far = pour_tool.plan_geometry(state, state[7:10] + np.array([0.6, 0.0, -0.2]), rim=None, container_center=None,
                                    container_depth_m=0.08, clearance_m=None)
    assert not g_far["pour_dir_adjusted_from_handle"]
    np.testing.assert_allclose(g_far["pour_dir"], [1, 0, 0], atol=1e-6)
    # A target back toward the hand is pushed 40 degrees around the rim, away from the gripper.
    g2 = pour_tool.plan_geometry(state, state[7:10] + np.array([-0.3, 0.03, -0.2]), rim=None, container_center=None,
                                 container_depth_m=0.08, clearance_m=None)
    assert g2["pour_dir_adjusted_from_handle"]
    assert np.degrees(np.arccos(g2["pour_dir"] @ -g2["handle_dir"])) == pytest.approx(40, abs=1e-6)
    assert g2["pour_dir"][1] > 0  # on the side nearer the target


def test_pour_aligns_the_rim_tilts_to_max_and_returns_upright():
    toolkit, controller = held(horizon=1000)
    start = toolkit._operation.observation["state"].copy()
    result = pour(toolkit, controller)
    assert result["target_reached"], result
    assert result["stage"] == "done"
    assert result["tilt_reached_deg"] >= 85 and result["tilt_target_deg"] == 90  # 1.2.1 default cap
    assert result["upright_error_rad"] <= 0.08
    assert result["align"]["aligned"]
    names = [p["phase"] for p in result["phases"]]
    assert "translate" in names and "descend" in names  # the rim already clears the target: no lift phase
    assert any(n.startswith("tilt-") for n in names) and names[-1] == "return-0"
    # The hand ends where the rim sits on the target with the default clearance.
    rim_offset = np.array([0.20, 0.11, 0.04])
    expected_eef = np.array(TARGET) + [0, 0, 0.12] - rim_offset
    final = toolkit._operation.observation["state"]
    np.testing.assert_allclose(final[7:10], expected_eef, atol=0.012)
    assert float(np.linalg.norm(_quaternion_error_rotvec(final[10:14], start[10:14]))) <= 0.08
    # Rotation really happened: some actions carried a rotation command about the pour axis.
    rotations = np.array([a[8:11] for a in controller.actions])
    assert np.abs(rotations).max() > 0
    assert result["geometry"]["rim_source"] == "model" and result["geometry"]["center_source"] == "model"
    assert result["executed_actions"] == len(controller.actions) - 1  # seed_closed executed one action
    assert "fresh images" in result["verification"]
    assert result["contact_suspected"] is False and result["contact_phases"] == []


def test_tight_budget_pauses_tilted_and_the_next_call_finishes_the_pour():
    toolkit, controller = held(horizon=1000)
    full = pour(toolkit, controller)["executed_actions"]
    toolkit, controller = held(horizon=1000)
    start = toolkit._operation.observation["state"].copy()
    first = pour(toolkit, controller, max_steps=max(150, int(full * 0.6)))
    assert not first["target_reached"] and first["continue_required"]
    assert first["stage"] in {"align", "tilting", "tilted", "returning"}
    assert first["stop_reason"] in {"continue_pour", "alignment_incomplete"}
    assert "call pour_into again" in first["next"]
    assert first["executed_actions"] <= max(150, int(full * 0.6))
    # A different target while tilted is refused; the same target resumes without re-fitting the rim.
    toolkit.observe()
    if first["tilt_reached_deg"] >= 5:
        result = toolkit.execute_tool("pour_into", {"expected_step": controller.step, "target": [0.1, 0.9, 0.3]}).result
        assert "in progress" in result["error"]
    second = pour(toolkit, controller)
    assert second["resumed"] and second["call_index"] == 2
    assert second["target_reached"], second
    assert second["stage"] == "done" and not second["continue_required"]
    final = toolkit._operation.observation["state"]
    assert float(np.linalg.norm(_quaternion_error_rotvec(final[10:14], start[10:14]))) <= 0.08
    assert first["executed_actions"] + second["executed_actions"] <= full + 12


def test_given_rim_and_center_are_reported_but_the_cookware_model_is_used():
    # 1.2.1: back_projected container points were 2.4-6.5 cm off in D10 runs and moved the real pour.
    toolkit, controller = held(horizon=1000)
    state = toolkit._operation.observation["state"]
    center = (state[7:10] + [0.25, 0.0, 0.03]).tolist()
    rim = (np.array(center) + [0, 0.14, 0.02]).tolist()
    result = pour(toolkit, controller, rim=rim, container_center=center, clearance_m=0.06)
    assert result["target_reached"], result
    geometry = result["geometry"]
    assert geometry["rim_source"] == "model" and geometry["center_source"] == "model"
    ignored = geometry["ignored_inputs"]
    assert ignored["container_center"] == pytest.approx(center) and ignored["rim"] == pytest.approx(rim)
    assert ignored["container_center_minus_model_m"] == pytest.approx(
        float(np.linalg.norm(np.array(center) - np.array(geometry["container_center"]))))
    assert ignored["rim_minus_model_m"] > 0.01
    # The hand ends where the modelled rim sits on the target, as without the given points.
    rim_offset = np.array([0.20, 0.11, 0.04])
    expected_eef = np.array(TARGET) + [0, 0, 0.06] - rim_offset
    np.testing.assert_allclose(toolkit._operation.observation["state"][7:10], expected_eef, atol=0.012)


def test_grasp_loss_interrupts_immediately():
    toolkit, controller = held()
    controller.width = 0.001
    result = pour(toolkit, controller)
    assert result["interrupted"] and result["interrupt_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1
    assert not result["target_reached"]


def test_terminal_success_stops_the_pour():
    toolkit, controller = held()
    controller.terminal = "success"
    result = pour(toolkit, controller)
    assert result["interrupt_reason"] == "success" and result["executed_actions"] == 1


def test_stuck_arm_reports_alignment_incomplete_without_tilting():
    toolkit, controller = held(stuck=True)
    result = pour(toolkit, controller)
    assert result["stage"] == "align"
    assert result["tilt_reached_deg"] == 0
    assert result["interrupt_reason"] == "deviation"
    assert not result["target_reached"] and not result["continue_required"]


def test_short_calls_are_refused_and_return_upright_rights_a_tilted_container():
    toolkit, controller = held(horizon=1000)
    result = toolkit.execute_tool("pour_into", {"expected_step": controller.step, "target": TARGET, "max_steps": 149}).result
    assert "150" in result["error"] and len(controller.actions) == 1
    # Upright container: nothing to return.
    result = toolkit.execute_tool("pour_into", {"expected_step": controller.step, "target": TARGET, "return_upright": True}).result
    assert "nothing to return" in result["error"] and len(controller.actions) == 1
    # Interrupt a pour mid-tilt with a tight budget, then right it explicitly.
    start = toolkit._operation.observation["state"].copy()
    full = pour(toolkit, controller)["executed_actions"]
    toolkit, controller = held(horizon=1000)
    first = pour(toolkit, controller, max_steps=max(150, int(full * 0.6)))
    assert first["stage"] in {"tilting", "tilted", "returning"} and first["tilt_reached_deg"] >= 5
    toolkit.observe()
    second = pour(toolkit, controller, return_upright=True)
    assert second["return_upright"] and second["stage"] == "done" and second["target_reached"]
    assert not any(p["phase"].startswith("tilt-") for p in second["phases"])
    assert second["upright_error_rad"] <= 0.08
    final = toolkit._operation.observation["state"]
    assert float(np.linalg.norm(_quaternion_error_rotvec(final[10:14], start[10:14]))) <= 0.08


class _StuckAfter(PublicController):
    """Freeze the arm after a number of actions, as a container blocked by the receiver."""

    def __init__(self, after, **kwargs):
        super().__init__(**kwargs)
        self.after, self.count = after, 0

    def execute_action(self, action, *, _action_index=0):
        self.count += 1
        self.stuck = self.after is not None and self.count > self.after
        return super().execute_action(action, _action_index=_action_index)


def _held_with(controller, horizon=2000):
    from robots.robocasa.toolkit import RoboCasaToolkit

    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Move the observed cup.", episode_horizon=horizon)
    seed_closed(toolkit, controller)
    return toolkit


def test_a_blocked_return_keeps_the_pour_and_return_upright_finishes_it():
    toolkit, controller = held(horizon=2000)
    first_return = None
    for phase_row in pour(toolkit, controller)["phases"]:
        if phase_row["phase"].startswith("return-"):
            break
        first_return = (first_return or 0) + phase_row["steps"]
    controller = _StuckAfter(first_return + 3)  # +1 seed action, then stuck early in the return
    toolkit = _held_with(controller)
    blocked = pour(toolkit, controller)
    assert "error" not in blocked
    assert blocked["stage"] == "returning" and blocked["interrupt_reason"] == "deviation"
    assert not blocked["target_reached"] and blocked["executed_actions"] > 0
    assert "return_upright=true" in blocked["next"]
    controller.after = None  # contact cleared
    toolkit.observe()
    righted = pour(toolkit, controller, return_upright=True)
    assert righted["stage"] == "done" and righted["upright_error_rad"] <= 0.08
    assert not any(p["phase"].startswith("tilt-") for p in righted["phases"])
    assert toolkit._tools["pour_into"][1].__self__._pour_state is None


def test_hand_moved_between_calls_reanchors_the_pivot_on_the_rim():
    toolkit, controller = held(horizon=2000)
    first = pour(toolkit, controller, max_steps=150)
    assert first["stage"] in {"tilting", "tilted", "returning"} and first["tilt_reached_deg"] >= 5
    state = toolkit._operation.observation["state"]
    call(toolkit, "move_eef_path", expected_step=toolkit.observe()["step"],
         waypoints=[{"position": (state[7:10] + [0.05, 0.0, 0.0]).tolist()}], gripper="keep", max_steps=60)
    toolkit.observe()
    second = pour(toolkit, controller)
    assert second["resumed"] and second["stage"] == "done"
    expected = np.array(TARGET) + [0, 0, 0.12] - np.array([0.20, 0.11, 0.04]) + [0.05, 0.0, 0.0]
    np.testing.assert_allclose(toolkit._operation.observation["state"][7:10], expected, atol=0.015)


def test_a_wrist_yaw_is_not_a_tilt_and_a_level_pour_restarts_after_other_motion():
    toolkit, controller = held(horizon=2000)
    far = [0.40, 1.60, 0.30]  # alignment alone takes more than 150 steps
    first = call(toolkit, "pour_into", expected_step=controller.step, target=far, max_steps=150)
    assert first["stage"] == "align" and first["continue_required"]
    state = toolkit._operation.observation["state"]
    call(toolkit, "move_eef_path", expected_step=toolkit.observe()["step"],
         waypoints=[{"position": state[7:10].tolist(), "rotation_degrees": [0, 0, 20]}], gripper="keep", max_steps=60)
    toolkit.observe()
    fresh = pour(toolkit, controller)  # a different target is not refused as a pour in progress
    assert not fresh["resumed"] and fresh["call_index"] == 1


def test_pour_into_guard_errors_are_refusals_not_faults():
    toolkit, controller = held()
    for arguments in ({"max_steps": 100}, {"return_upright": True}):
        result = toolkit.execute_tool("pour_into", {"expected_step": controller.step, "target": TARGET, **arguments}).result
        assert "error" in result and result.get("refused") is True, result


# --- 1.2.1 place_container: set held cookware down at a freshly grounded destination --------------


def place(toolkit, controller, target, **overrides):
    toolkit.observe()
    return call(toolkit, "place_container", expected_step=controller.step, target=target, **overrides)


def fingers_follow_commands(controller):
    """The public test controller keeps the finger width; widen it on an open command like the simulator."""
    original = controller.execute_action

    def execute(action, **kwargs):
        result = original(action, **kwargs)
        if action[11] < 0:
            controller.observation["state"][14:16] = [0.04, -0.04]
        return result

    controller.execute_action = execute


def test_place_container_is_a_privileged_physical_tool_in_the_shared_vocabulary():
    toolkit, _ = setup()
    assert "place_container" in toolkit.PHYSICAL_TOOLS
    assert "place_container" in control_modes.PRIVILEGED_TOOLS
    for tools in control_modes.PRIVILEGED_TOOLS_BY_MODE.values():
        assert "place_container" in tools
    schema = toolkit._tools["place_container"][0]["input_schema"]
    assert schema["required"] == ["expected_step", "target"]
    assert schema["properties"]["release"]["default"] is True


def test_place_puts_the_model_centre_over_the_target_lowers_releases_and_retreats():
    toolkit, controller = held(horizon=1000)
    state = toolkit._operation.observation["state"].copy()
    center, _ = pour_tool.model_center(state)
    # Destination 15 cm away and 12 cm below the held centre, like a burner after a policy lift.
    target = (center + [0.15, -0.10, -0.12]).tolist()
    fingers_follow_commands(controller)
    result = place(toolkit, controller, target)
    assert result["target_reached"] and result["released"], result
    assert result["gripper_settle"]["release_confirmed"]
    names = [p["phase"] for p in result["phases"]]
    assert names == ["translate", "lower", "retreat", "back_off"]  # the centre already clears target z + 0.10
    geometry = result["geometry"]
    assert geometry["center_xy_error_m"] <= 0.01 and geometry["center_source"] == "model"
    placed = np.array(geometry["model_center_placed"])
    assert abs(placed[2] - target[2]) <= 0.012  # no support in the test controller: lowered to the surface point
    assert result["gripper_settle"]["command"] == "open"
    assert controller.actions[-1][11] < 0  # gripper left open
    final = toolkit._operation.observation["state"]
    # Hand ends 10 cm above where the centre was put down and 15 cm back toward the robot.
    raised = placed - (center - state[7:10]) + [0, 0, 0.10]
    back = -np.array([raised[0], raised[1], 0.0])
    np.testing.assert_allclose(final[7:10], raised + 0.15 * back / np.linalg.norm(back), atol=0.015)
    assert float(np.linalg.norm(_quaternion_error_rotvec(final[10:14], state[10:14]))) <= 0.05


def test_place_without_release_keeps_the_gripper_closed_and_lifts_to_clearance_first():
    toolkit, controller = held(horizon=1000)
    center, _ = pour_tool.model_center(toolkit._operation.observation["state"])
    result = place(toolkit, controller, (center + [0.05, 0.05, -0.05]).tolist(), release=False)
    assert result["target_reached"] and not result["released"] and "gripper_settle" not in result
    assert [p["phase"] for p in result["phases"]] == ["lift", "translate", "lower"]
    assert controller.actions[-1][11] > 0


def test_place_needs_a_held_container_and_waits_for_an_upright_pour():
    toolkit, controller = setup()
    toolkit.observe()
    result = toolkit.execute_tool("place_container", {"expected_step": 0, "target": TARGET}).result
    assert "held container" in result["error"] and result.get("refused") is True
    assert controller.actions == []
    toolkit, controller = held(horizon=1000)
    full = pour(toolkit, controller)["executed_actions"]
    toolkit, controller = held(horizon=1000)
    first = pour(toolkit, controller, max_steps=max(150, int(full * 0.6)))
    toolkit.observe()
    if first["tilt_reached_deg"] >= 5:
        before = len(controller.actions)
        result = toolkit.execute_tool("place_container", {"expected_step": controller.step, "target": TARGET}).result
        assert "pour is in progress" in result["error"] and result.get("refused") is True
        assert len(controller.actions) == before


@pytest.mark.parametrize("override", [
    {"target": [0.4, 0.45]}, {"clearance_m": 0.5}, {"release": "yes"}, {"max_steps": 10}, {"max_steps": 251},
])
def test_place_invalid_arguments_rejected_before_any_action(override):
    toolkit, controller = held()
    before = len(controller.actions)
    result = toolkit.execute_tool("place_container", {"expected_step": controller.step, "target": TARGET, **override}).result
    assert "error" in result
    assert len(controller.actions) == before


def test_the_held_centre_keeps_its_side_of_the_hand_after_a_sideways_carry():
    # 1.2.1 fork of D10: with the hand over the plate the "far side from the robot" rule flipped
    # and put the pan centre 40 cm off; the side is fixed while the container stays in the hand.
    from types import SimpleNamespace

    yaw = np.arctan2(-0.49, 0.87)  # hand x axis along the D10 handle at the grasp
    quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    closure = np.zeros(16)
    closure[7:10], closure[10:14], closure[14:16] = [0.43, 0.05, 0.40], quaternion, [0.008, -0.008]
    over_plate = closure.copy()
    over_plate[7:10] = [0.25, 0.62, 0.33]  # carried sideways, same orientation
    true_offset = pour_tool.model_center(closure)[0] - closure[7:10]
    flipped = pour_tool.model_center(over_plate)[0] - over_plate[7:10]
    np.testing.assert_allclose(flipped[:2], -true_offset[:2], atol=1e-9)  # the old rule at the plate

    history = []
    operation = SimpleNamespace(metrics={"step_count": 98}, action_history_since=lambda: history)
    progress = SimpleNamespace(cycle=2, closure_state=closure)
    handlers = SimpleNamespace(tk=SimpleNamespace(_pick_progress=progress, _operation=operation),
                               _held_container=None)
    center, source = pour_tool.held_model_center(handlers, over_plate, "place_container")
    assert source == "grasp_closure_pose"
    np.testing.assert_allclose(center, over_plate[7:10] + true_offset, atol=1e-9)

    # First 1.2.1 D10 run: each back_project restarted the grasp watch (cycle +1, closure pose
    # cleared) while the pan stayed in the hand. The model made at the pour start must survive.
    handlers._held_container, progress.closure_state = None, None
    operation.metrics["step_count"] = 130
    _, source = pour_tool.held_model_center(handlers, closure, "pour_into")
    assert source == "pour_into_start_pose"
    held = [1.0] * 12  # a closed-gripper command (native value >= 0.5)
    history.append({"step_ids": [131, 132], "actions": [held, held]})
    progress.cycle, progress.closure_state = 4, over_plate  # restarted, late closure pose over the plate
    operation.metrics["step_count"] = 680
    later, source = pour_tool.held_model_center(handlers, over_plate, "place_container")
    assert source == "pour_into_start_pose"
    np.testing.assert_allclose(later, over_plate[7:10] + true_offset, atol=1e-9)

    # An executed open command ends the hold: the next model starts over.
    opened = [1.0] * 11 + [-1.0]
    history.append({"step_ids": [681], "actions": [opened]})
    progress.closure_state = None
    _, source = pour_tool.held_model_center(handlers, over_plate, "place_container")
    assert source == "place_container_start_pose"
    # A measured open hand also ends it, even without a recorded command.
    history.clear()
    handlers._held_container = {"step": 0, "center_e": np.zeros(3), "source": "pour_into_start_pose"}
    wide = over_plate.copy()
    wide[14:16] = [0.04, -0.04]
    _, source = pour_tool.held_model_center(handlers, wide, "place_container")
    assert source == "place_container_start_pose"
