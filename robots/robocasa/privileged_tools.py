"""Larger bounded controls using only existing images and public 16D state."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from robots.robocasa import control_modes
from robots.robocasa.toolkit import (
    _CAMERAS,
    _POSITION_ACTION_SCALE_M,
    _POSITION_TOLERANCE_M,
    _ROTATION_ACTION_SCALE_RAD,
    _ROTATION_TOLERANCE_RAD,
    _finite_vector,
    _merge_physical_result,
    _png_bytes,
    _quaternion_error_rotvec,
    _robocasa_state,
    _rotate_base_quaternion,
)
from rpent.harness.lifecycle import (
    GRASP_STABLE_CLOSE_STEPS,
    GRASP_STABLE_WIDTH_TOLERANCE_M,
)

PRIVILEGED_PHYSICAL_TOOLS = frozenset(control_modes.PRIVILEGED_TOOLS)
READ_ONLY_PERCEPTION_TOOLS = frozenset(control_modes.ALLOWLISTABLE_READ_ONLY_TOOLS)
BASE_POSITION_TOLERANCE_M = 0.02
BASE_HEADING_TOLERANCE_RAD = np.deg2rad(3)
# Base commands are velocity-like and saturate near 1.0. Measured in the
# 1.0.11 full run (D06/D07 transitions.parquet, 2026-10-11): |command| 0.3
# moves the public base centre about 2.3 mm per step, 0.6 about 9 mm, 0.7-1.0
# about 27 mm and above 1.0 about 33 mm. pi0.5 carries a filled bowl at a
# mean 16-19 mm per step with p90 commands of 1.0 and no acceleration ramp
# (D06 handback: 0 -> 1.04 in one step), so the former 0.3 holding cap made a
# privileged carry about ten times slower than the policy's own (D07: 0.66 m
# in 450 steps, then pi0.5 2.9 m in 153 steps) with no slip-risk basis: the
# 1.0.5 drops were fast arm lifts, not base travel. Travel now shares the
# policy's carry level; holding limits acceleration and the turn rate instead,
# which load a held object through the arm lever.
BASE_COMMAND_MAX = 1.0
BASE_COMMAND_MIN = 0.3  # static-friction floor; below it the base barely moves
HOLD_BASE_COMMAND_MAX = 1.0
BASE_TURN_COMMAND_MAX = 0.6
HOLD_BASE_TURN_COMMAND_MAX = 0.5  # pi0.5 carry yaw p90: D06 0.50, D07 0.19
BASE_COMMAND_RAMP = 0.5  # per-step command increase above the floor: free, two steps to full
HOLD_BASE_COMMAND_RAMP = 0.25  # holding: four steps to full
EEF_STEP_M = 0.02
HOLD_EEF_STEP_M = 0.005
EEF_ROTATION_STEP_RAD = 0.15
HOLD_EEF_ROTATION_STEP_RAD = 0.04
DEVIATION_POSITION_M = 0.06
DEVIATION_ROTATION_RAD = 0.35
DEVIATION_PROGRESS_M = 0.001
DEVIATION_PROGRESS_RAD = 0.003
DEVIATION_STEPS = 8
# While holding, a loaded arm tracks slowly (1.0.11 E2: 22 of 27 deviation
# stops were still closing in at 0.5-0.9 mm/step). Judge progress over a longer
# window at lower rates, and when the arm covers less than HOLD_LAG_RATIO of
# the commanded motion over a window, double the per-step command (up to
# HOLD_LOAD_LEVELS times, never above the free-motion caps): in delta mode the
# command is the controller's only lever against the load.
HOLD_DEVIATION_WINDOW = 16
HOLD_DEVIATION_PROGRESS_M = 0.00025
HOLD_DEVIATION_PROGRESS_RAD = 0.001
HOLD_LAG_RATIO = 0.5
HOLD_LOAD_LEVELS = 2
# A held object sags below a stalled setpoint by its load (E2 pan: 4.1-4.5 mm);
# a waypoint within this distance whose progress has stopped is settled.
HOLD_SAG_TOLERANCE_M = 0.01
BASE_STALL_POSITION_M = 0.0005
BASE_STALL_HEADING_RAD = 0.001
BASE_STALL_STEPS = 12
TORSO_FULL_TRAVEL_M = 0.34  # pinned Omron XML range [0,0.34], not a live joint read
# Measured on the pinned SDS stack (B smoke attempt09): the public base-center
# height rises about 0.93 mm per control step at native torso command 0.2,
# i.e. a velocity-like response of about 4.65 mm per step per unit command.
TORSO_RESPONSE_M_PER_COMMAND = 0.00465
TORSO_COMMAND_MAX = 1.0
HOLD_TORSO_COMMAND_MAX = 0.5
TORSO_GAIN = 0.8
TORSO_TOLERANCE_M = 0.003
TORSO_STALL_M = 0.0001
TORSO_STALL_STEPS = 6
# A gripper change at the end of a path keeps commanding until the measured
# opening is stable (the grasp detector's own stable-close window) or this many
# steps pass (1.0.11 A3; one step left a pan handle 0.078 m open in D10).
GRIPPER_SETTLE_MAX_STEPS = 15
# Fingers that met an object and then creep slowly (a compliant handle, 1.0.11
# P1: about 0.7 mm/step after closing at 5-8 mm/step) count as settled once the
# last GRIPPER_CONTACT_STEPS width changes are each at most this much.
GRIPPER_CONTACT_RATE_M = 0.001
GRIPPER_CONTACT_STEPS = 3
GRIPPER_EMPTY_OPENING_M = 0.002
GRIPPER_OPEN_OPENING_M = 0.06


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _cap(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * min(1, maximum / max(norm, 1e-12))


def _yaw(state: np.ndarray) -> float:
    q = state[3:7]
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("navigation requires a valid public base quaternion")
    x, y, z, w = q / norm
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def _base_command_limits(holding: bool) -> tuple[float, float, float]:
    """(travel cap, turn cap, per-step ramp) for free or loaded base motion."""
    if holding:
        return HOLD_BASE_COMMAND_MAX, HOLD_BASE_TURN_COMMAND_MAX, HOLD_BASE_COMMAND_RAMP
    return BASE_COMMAND_MAX, BASE_TURN_COMMAND_MAX, BASE_COMMAND_RAMP


def _base_translation(error_world: np.ndarray, yaw: float, cap: float) -> np.ndarray:
    """Hold a public base-center target without assuming the yaw pivot is there."""
    body_error = np.array([[np.cos(yaw), np.sin(yaw)],
                           [-np.sin(yaw), np.cos(yaw)]]) @ error_world
    command = np.zeros(2)
    axis = int(np.argmax(np.abs(body_error)))
    if abs(body_error[axis]) > BASE_POSITION_TOLERANCE_M * 0.3:
        # One translation axis at a time keeps the vector speed bounded and
        # retains enough effort to overcome the native static-friction region.
        command[axis] = np.sign(body_error[axis]) * min(cap, BASE_COMMAND_MIN + abs(body_error[axis]) * 2)
    return command


def _holonomic_translation(error_world: np.ndarray, yaw: float, cap: float) -> np.ndarray:
    """Straight base-frame translation toward a public base-center target.

    The dominant axis keeps at least the static-friction command; the vector
    norm stays within the speed cap.
    """
    body_error = np.array([[np.cos(yaw), np.sin(yaw)],
                           [-np.sin(yaw), np.cos(yaw)]]) @ error_world
    major = float(np.max(np.abs(body_error)))
    if major <= 1e-9:
        return np.zeros(2)
    magnitude = min(cap, max(BASE_COMMAND_MIN, float(np.linalg.norm(body_error)) * 2))
    command = _cap(body_error / major * magnitude, cap)
    axis = int(np.argmax(np.abs(body_error)))
    floor = min(BASE_COMMAND_MIN, cap)
    if abs(command[axis]) < floor:
        # With the holding cap at the friction floor, a diagonal shared out by
        # the norm cap left both axes below it and the base stalled; keep the
        # dominant axis at the floor and trim the other within the cap.
        other = 1 - axis
        command[axis] = np.sign(body_error[axis]) * floor
        command[other] = np.sign(body_error[other]) * min(
            abs(command[other]), float(np.sqrt(max(cap * cap - floor * floor, 0.0))))
    return command


def _direction_label(vector: np.ndarray) -> str:
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    return ("forward" if abs(angle) <= 45 else "backward" if abs(angle) >= 135
            else "left" if angle > 0 else "right")


def _stall_report(phase: str, action: np.ndarray) -> dict[str, Any]:
    """Which commanded motion the base failed to make (1.0.11 C1)."""
    translation = np.asarray(action[:2], dtype=float)
    blocked = "translation" if np.linalg.norm(translation) > 1e-6 else "rotation"
    report: dict[str, Any] = {
        "phase": phase, "blocked": blocked,
        "commanded_base_velocity": [float(v) for v in action[:3]], "command_frame": "current_robot_base",
    }
    if blocked == "translation":
        report["blocked_direction"] = _direction_label(translation)
    if abs(float(action[2])) > 1e-6:
        report["blocked_turn"] = "left" if action[2] > 0 else "right"
    return report


def _measured(state: np.ndarray) -> dict[str, Any]:
    return {"eef_position": state[7:10].tolist(), "eef_quaternion": state[10:14].tolist(),
            "eef_frame": "robot_base", "base_position": state[:3].tolist(),
            "base_quaternion": state[3:7].tolist(), "base_frame": "world",
            "gripper_opening": float(np.abs(state[14:16]).sum())}


class _Motion:
    """One budgeted physical call, with history tagged even on partial faults."""

    def __init__(self, toolkit, tool: str, expected_step: int, max_steps: int, ceiling: int):
        self.tk, self.tool = toolkit, tool
        step = int(toolkit._operation.metrics["step_count"])
        if isinstance(expected_step, bool) or not isinstance(expected_step, int):
            raise ValueError("expected_step must be an integer")
        if expected_step != step or toolkit._last_observed_step != step:
            raise ValueError(f"{tool} requires a fresh observe and its exact expected_step")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 2 <= max_steps <= ceiling:
            raise ValueError(f"max_steps must be an integer between 2 and {ceiling}")
        self.limit = toolkit._motion_action_budget(max_steps)
        if self.limit < 2:
            raise RuntimeError("motion needs two remaining actions including a zero stop command")
        self.before = _robocasa_state(toolkit._operation.observation).copy()
        self.result = {"requested_actions": 0, "executed_actions": 0, "cancelled_actions": 0}
        self.reason = None
        # Holding limits speed on the retention watch, or on public evidence of
        # a closure stopped by an object (a restored state has no grasp history).
        opening = float(np.abs(self.before[14:16]).sum())
        if toolkit._pick_progress and toolkit._pick_progress.retention_monitor_active:
            self.holding_basis = "retention_watch"
        elif (toolkit._last_gripper_command() == "close"
              and GRIPPER_EMPTY_OPENING_M < opening < GRIPPER_OPEN_OPENING_M):
            self.holding_basis = "closed_on_object_width"
        else:
            self.holding_basis = None
        self.holding = self.holding_basis is not None
        self.armed = False
        self.stop_executed = False
        self.goal_synced = False

    @property
    def state(self) -> np.ndarray:
        return _robocasa_state(self.tk._operation.observation).copy()

    @property
    def available(self) -> int:
        return self.limit - int(self.result["executed_actions"])

    def action(self, gripper: str, *, body=False) -> np.ndarray:
        action = np.zeros(12, dtype=np.float32)
        action[4] = 1 if body else -1
        action[11] = -1 if gripper == "open" else 1
        return action

    def execute(self, action: np.ndarray) -> None:
        if not self.armed:
            self.tk._motion_observe_required = True
            # Privileged motion supersedes the short unchanged-language
            # correction handoff: the policy re-enters through continue_vla
            # with the next operation's frozen sentence, never resume_vla.
            self.tk._resume_after_adjustment = False
            self.tk._precontact_grounding = None
            self.tk._burner_control_anchor = None
            self.armed = True
        cursor = self.tk._operation.action_history_count
        try:
            step = self.tk._operation.execute_actions((action,))
        finally:
            # VLAOperation has no source-label argument. Keep this adaptation
            # local to B instead of changing the shared operation API/files.
            for record in self.tk._operation._action_history[cursor:]:
                record["control_source"] = self.tool
        self.tk._update_pick_progress(step)
        self.result = _merge_physical_result(self.result, step)
        environment = step.get("environment", {})
        if environment.get("success"):
            self.reason = "success"
        elif environment.get("terminated") or environment.get("truncated"):
            self.reason = "terminated"
        elif step.get("stop_reason") == "grasp_lost" or step.get("grasp_events"):
            self.reason = "grasp_lost"
        elif (self.holding and action[11] > 0
              and float(np.abs(self.state[14:16]).sum()) <= GRIPPER_EMPTY_OPENING_M):
            # The grasp watch arms only after a confirmed lift; a held object
            # that slips out of a closing gripper ends the motion here too.
            self.reason = "grasp_lost"
            self.result["closed_empty"] = True

    def sync_arm_goal(self, gripper: str) -> None:
        """Send one zero arm-mode step before body-mode motion.

        Robosuite's OSC reset_goal stores a world-frame goal while base-frame
        input reads it as base-relative, so a first body-mode step after an
        episode start or state restore throws the arm toward a wrong target.
        One zero-delta arm-mode ("achieved") step re-anchors the goal to the
        current base-relative pose; body-mode steps then carry it with the base.
        """
        self.execute(self.action(gripper))
        self.goal_synced = True

    def finish(self, gripper: str, *, body: bool, reached: bool, **extra) -> dict[str, Any]:
        # Terminal success/loss must stop immediately. For nonterminal stops,
        # reserve and record one zero-increment command in the same budget.
        if self.reason not in {"success", "terminated", "grasp_lost"} and self.available:
            self.execute(self.action(gripper, body=body))
            self.stop_executed = True
        self.tk._preclose_approval = None
        if self.tk._preclose_active and not self.tk._preclose_closing:
            self.tk._preclose_pending = True
        return {**self.result, "transitions_used": self.result["executed_actions"],
                "transition_budget": self.limit, "interrupted": self.reason is not None,
                "interrupt_reason": self.reason, "control_mode": None,
                "measured_before": _measured(self.before), "measured_after": _measured(self.state),
                "target_reached": bool(reached and self.reason is None),
                "stop_command_executed": self.stop_executed,
                "arm_goal_sync_step": self.goal_synced,
                "holding_speed_limited": self.holding, "holding_basis": self.holding_basis,
                "policy_instruction": self.tk._active_policy_instruction, **extra}


class PrivilegedTools:
    def __init__(self, toolkit):
        self.tk = toolkit
        # Last target and outcome per tool, for the near-repeat warning (B4).
        self._last_targets: dict[str, tuple[np.ndarray, bool]] = {}

    def _repeat_warning(self, tool: str, target: np.ndarray) -> str | None:
        """Warn when an unreached target is resent almost unchanged (1.0.11 B4)."""
        previous = self._last_targets.get(tool)
        if previous is None or previous[1] or previous[0].shape != target.shape:
            return None
        # Positions in metres; unit-vector or quaternion parts compared as angles.
        size = 2 if tool == "navigate_to" else 3
        position_change = float(np.linalg.norm(target[:size] - previous[0][:size]))
        direction_change = 0.0
        if tool == "navigate_to":
            direction_change = abs(_angle(float(np.arctan2(target[3], target[2]) - np.arctan2(previous[0][3], previous[0][2]))))
        elif tool == "move_eef_path":
            direction_change = float(np.linalg.norm(_quaternion_error_rotvec(previous[0][3:7], target[3:7])))
        if position_change < control_modes.REPEAT_POSITION_M and direction_change < np.deg2rad(control_modes.REPEAT_ROTATION_DEG):
            return (f"{tool} repeats the previous unreached target almost unchanged; change the approach "
                    "(route, waypoint, height or orientation) from fresh evidence instead of resending it")
        return None

    def _remember(self, tool: str, target: np.ndarray, reached: bool) -> None:
        self._last_targets[tool] = (target.copy(), bool(reached))

    def navigate_to(self, expected_step: int, target: dict, *, gripper="keep", max_steps=200, turn_first=False):
        motion = _Motion(self.tk, "navigate_to", expected_step, max_steps, 200)
        yaw = _yaw(motion.before)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        if not isinstance(target, dict):
            raise ValueError("target must be exactly one relative or point target")
        if set(target) == {"relative"}:
            relative = _finite_vector(target["relative"], size=3, label="relative")
            delta, heading = relative[:2], yaw + np.deg2rad(relative[2])
            if abs(relative[2]) > 180:
                raise ValueError("relative angle must be <=180 degrees")
        elif set(target) == {"point", "standoff_m"}:
            point = _finite_vector(target["point"], size=3, label="point")
            standoff = _number(target["standoff_m"], "standoff_m")
            distance = float(np.linalg.norm(point[:2]))
            if standoff < 0 or distance <= 1e-8:
                raise ValueError("standoff_m must be nonnegative and point must have nonzero planar distance")
            delta = point[:2] * (1 - standoff / distance)
            heading = yaw + np.arctan2(point[1], point[0])
        else:
            raise ValueError("target must be exactly one relative or point/standoff_m target")
        if np.linalg.norm(delta) > 3 + 1e-9:
            raise ValueError("target displacement must be <=3 metres")
        if not isinstance(turn_first, bool):
            raise ValueError("turn_first must be a boolean")
        position = motion.before[:2] + rotation @ delta
        gripper, source = self.tk._resolve_gripper_command(gripper)
        self.tk._require_reviewed_direct_close(gripper)
        repeat = self._repeat_warning("navigate_to", np.array([*position, np.cos(heading), np.sin(heading)]))
        stalled = 0
        stall: dict[str, Any] | None = None
        reached = False
        phase = "travel"
        turn_anchor = None
        hold_yaw = yaw
        motion.sync_arm_goal(gripper)
        travel_cap, turn_cap, ramp = _base_command_limits(motion.holding)
        previous = np.zeros(3)
        while not motion.reason and motion.available > 1:
            state = motion.state
            error = position - state[:2]
            distance = float(np.linalg.norm(error))
            current_yaw = _yaw(state)
            heading_error = _angle(heading - current_yaw)
            reached = distance <= BASE_POSITION_TOLERANCE_M and abs(heading_error) <= BASE_HEADING_TOLERANCE_RAD
            if reached:
                break
            if distance <= BASE_POSITION_TOLERANCE_M:
                phase = "align"
            action = motion.action(gripper, body=True)
            # Accelerate from the friction floor by at most the ramp per step;
            # deceleration is immediate, so a stop or a stall never waits on it.
            cap = min(travel_cap, max(BASE_COMMAND_MIN, float(np.linalg.norm(previous[:2])) + ramp))
            spin = min(turn_cap, max(BASE_COMMAND_MIN, abs(float(previous[2])) + ramp))
            if phase == "travel" and not turn_first:
                # The base is omnidirectional: translate straight in the base
                # frame while holding the start heading, then align (1.0.11
                # C1). Turning first in a narrow gap hit the stools in D07.
                action[:2] = _holonomic_translation(error, current_yaw, cap)
                drift = _angle(hold_yaw - current_yaw)
                if abs(drift) > BASE_HEADING_TOLERANCE_RAD:
                    action[2] = np.sign(drift) * min(spin, max(BASE_COMMAND_MIN, abs(drift) * 1.5))
            else:
                turn = _angle(float(np.arctan2(error[1], error[0])) - current_yaw) if phase == "travel" else heading_error
                if abs(turn) > BASE_HEADING_TOLERANCE_RAD:
                    action[2] = np.sign(turn) * min(spin, max(BASE_COMMAND_MIN, abs(turn) * 1.5))
                    if phase == "travel":
                        if turn_anchor is None:
                            turn_anchor = state[:2].copy()
                        action[:2] = _base_translation(turn_anchor - state[:2], current_yaw, cap)
                    else:
                        action[:2] = _base_translation(error, current_yaw, cap)
                elif phase == "travel":
                    action[0] = min(cap, max(BASE_COMMAND_MIN, distance * 2))
                    turn_anchor = None
                else:
                    action[:2] = _base_translation(error, current_yaw, cap)
            previous = action[:3].copy()
            motion.execute(action)
            if motion.reason:
                break
            after = motion.state
            moved = np.linalg.norm(after[:2] - state[:2])
            turned = abs(_angle(_yaw(after) - current_yaw))
            stalled = stalled + 1 if moved < BASE_STALL_POSITION_M and turned < BASE_STALL_HEADING_RAD else 0
            if stalled >= BASE_STALL_STEPS:
                motion.reason = "base_stalled"
                stall = _stall_report(phase, action)
                break
        state = motion.state
        reached = np.linalg.norm(position - state[:2]) <= BASE_POSITION_TOLERANCE_M and abs(_angle(heading - _yaw(state))) <= BASE_HEADING_TOLERANCE_RAD
        if not reached and motion.reason is None:
            motion.reason = "budget"
        result = motion.finish(gripper, body=True, reached=reached,
                               target_base_position_world=position.tolist(), target_heading_rad=_angle(heading),
                               gripper_source=source, travel_mode="turn_first" if turn_first else "holonomic",
                               **({"base_stall": stall} if stall else {}),
                               **({"repeat_warning": repeat} if repeat else {}))
        after = motion.state
        distance = float(np.linalg.norm(position - after[:2]))
        heading_error = abs(_angle(heading - _yaw(after)))
        result.update(position_error_m=distance, heading_error_rad=heading_error,
                      target_reached=bool(distance <= BASE_POSITION_TOLERANCE_M
                                          and heading_error <= BASE_HEADING_TOLERANCE_RAD and motion.reason is None))
        self._remember("navigate_to", np.array([*position, np.cos(heading), np.sin(heading)]), result["target_reached"])
        return result

    def move_eef_path(self, expected_step: int, waypoints: list[dict], *, gripper="keep", max_steps=150):
        motion = _Motion(self.tk, "move_eef_path", expected_step, max_steps, 150)
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 8:
            raise ValueError("waypoints must contain 1 to 8 points")
        targets = []
        previous = motion.before[7:10]
        quaternion = motion.before[10:14].copy()
        if np.linalg.norm(quaternion) <= 1e-8:
            raise ValueError("path requires a valid public EEF quaternion")
        quaternion /= np.linalg.norm(quaternion)
        for waypoint in waypoints:
            if not isinstance(waypoint, dict) or set(waypoint) - {"position", "rotation_degrees"} or "position" not in waypoint:
                raise ValueError("each waypoint needs position and optional rotation_degrees")
            position = _finite_vector(waypoint["position"], size=3, label="position")
            degrees = _finite_vector(waypoint.get("rotation_degrees", [0, 0, 0]), size=3, label="rotation_degrees")
            if np.linalg.norm(position - previous) > 0.30 + 1e-9 or np.linalg.norm(degrees) > 90 + 1e-9:
                raise ValueError("each path segment must be <=0.30 metres and <=90 degrees")
            quaternion = _rotate_base_quaternion(quaternion, np.deg2rad(degrees))
            targets.append((position, quaternion.copy()))
            previous = position
        final_gripper, source = self.tk._resolve_gripper_command(gripper)
        self.tk._require_reviewed_direct_close(final_gripper)
        signature = np.concatenate((targets[-1][0], targets[-1][1]))
        repeat = self._repeat_warning("move_eef_path", signature)
        # Motion preserves the confirmed command. With no history, an explicit
        # command may initialize open; a requested close waits for the endpoint.
        try:
            moving_gripper, _ = self.tk._resolve_gripper_command("keep")
        except ValueError:
            moving_gripper = "open"
        results = []
        holding = motion.holding
        window = HOLD_DEVIATION_WINDOW if holding else DEVIATION_STEPS
        rate_m = HOLD_DEVIATION_PROGRESS_M if holding else DEVIATION_PROGRESS_M
        rate_rad = HOLD_DEVIATION_PROGRESS_RAD if holding else DEVIATION_PROGRESS_RAD
        # Load compensation persists across waypoints while the load works
        # against the same gravity; a descent starts again from the hold limit.
        load_level = rot_level = 0
        for index, (position, quaternion) in enumerate(targets):
            start_step = int(self.tk._operation.metrics["step_count"])
            mismatch_run = 0
            reached = False
            settled_under_load = False
            start = motion.state
            if position[2] < start[9] - _POSITION_TOLERANCE_M:
                load_level = 0
            errors = [(float(np.linalg.norm(position - start[7:10])),
                       float(np.linalg.norm(_quaternion_error_rotvec(start[10:14], quaternion))))]
            commanded: list[tuple[float, float]] = []
            since_change = 0
            while motion.available > 1:
                state = motion.state
                pos_error = position - state[7:10]
                rot_error = _quaternion_error_rotvec(state[10:14], quaternion)
                reached = np.linalg.norm(pos_error) <= _POSITION_TOLERANCE_M and np.linalg.norm(rot_error) <= _ROTATION_TOLERANCE_RAD
                if reached:
                    break
                if holding:
                    step_m = min(EEF_STEP_M, HOLD_EEF_STEP_M * 2 ** load_level)
                    step_rad = min(EEF_ROTATION_STEP_RAD, HOLD_EEF_ROTATION_STEP_RAD * 2 ** rot_level)
                else:
                    step_m, step_rad = EEF_STEP_M, EEF_ROTATION_STEP_RAD
                increment = _cap(pos_error, step_m)
                rot_increment = _cap(rot_error, step_rad)
                action = motion.action(moving_gripper)
                action[5:8] = increment / _POSITION_ACTION_SCALE_M
                action[8:11] = rot_increment / _ROTATION_ACTION_SCALE_RAD
                motion.execute(action)
                if motion.reason:
                    break
                after = motion.state
                errors.append((float(np.linalg.norm(position - after[7:10])),
                               float(np.linalg.norm(_quaternion_error_rotvec(after[10:14], quaternion)))))
                commanded.append((float(np.linalg.norm(increment)), float(np.linalg.norm(rot_increment))))
                since_change += 1
                # A large error against the current bounded setpoint, sustained,
                # means contact or a block.
                mismatch = np.linalg.norm(state[7:10] + increment - after[7:10])
                rotation_mismatch = np.linalg.norm(_quaternion_error_rotvec(after[10:14], _rotate_base_quaternion(state[10:14], rot_increment)))
                mismatch_run = mismatch_run + 1 if mismatch > DEVIATION_POSITION_M or rotation_mismatch > DEVIATION_ROTATION_RAD else 0
                if mismatch_run >= DEVIATION_STEPS:
                    motion.reason = "deviation"
                    break
                if since_change < window:
                    continue
                # A distant point alone is not deviation: over the last window,
                # some still-unfinished component (position or rotation) must
                # have closed in at the minimum rate.
                (old_m, old_rad), (new_m, new_rad) = errors[-window - 1], errors[-1]
                pos_active = new_m > _POSITION_TOLERANCE_M
                rot_active = new_rad > _ROTATION_TOLERANCE_RAD
                gain_m, gain_rad = old_m - new_m, old_rad - new_rad
                pos_lag = holding and pos_active and gain_m < HOLD_LAG_RATIO * sum(c[0] for c in commanded[-window:])
                rot_lag = holding and rot_active and gain_rad < HOLD_LAG_RATIO * sum(c[1] for c in commanded[-window:])
                if (pos_lag and load_level < HOLD_LOAD_LEVELS) or (rot_lag and rot_level < HOLD_LOAD_LEVELS):
                    load_level += int(pos_lag and load_level < HOLD_LOAD_LEVELS)
                    rot_level += int(rot_lag and rot_level < HOLD_LOAD_LEVELS)
                    since_change = 0
                    continue
                if (pos_active or rot_active) and not (
                    (pos_active and gain_m >= window * rate_m) or (rot_active and gain_rad >= window * rate_rad)
                ):
                    # 2: the commanded increment shrinks to the remaining error,
                    # so a sagging load stops short of the target at any level.
                    if holding and not rot_active and new_m <= HOLD_SAG_TOLERANCE_M:
                        settled_under_load = True
                        break
                    motion.reason = "deviation"
                    break
            after = motion.state
            reached = settled_under_load or (
                np.linalg.norm(position - after[7:10]) <= _POSITION_TOLERANCE_M
                and np.linalg.norm(_quaternion_error_rotvec(after[10:14], quaternion)) <= _ROTATION_TOLERANCE_RAD
            )
            if not reached and motion.reason is None:
                motion.reason = "budget"
            results.append({"index": index, "start_step": start_step,
                            "end_step": int(self.tk._operation.metrics["step_count"]),
                            "target_position": position.tolist(), "target_quaternion": quaternion.tolist(),
                            "target_reached": bool(reached), "interrupt_reason": motion.reason,
                            "status": "reached" if reached else "timeout" if motion.reason == "budget" else "interrupted",
                            "position_error_m": float(np.linalg.norm(position - after[7:10])),
                            "rotation_error_rad": float(np.linalg.norm(_quaternion_error_rotvec(after[10:14], quaternion))),
                            "load_level": load_level, "rotation_load_level": rot_level,
                            "settled_under_load": settled_under_load})
            if motion.reason:
                break
        reached = len(results) == len(targets) and all(row["target_reached"] for row in results)
        final_command = moving_gripper
        review_required = False
        if reached and motion.reason is None:
            try:
                self.tk._require_reviewed_direct_close(final_gripper)
                final_command = final_gripper
            except ValueError:
                review_required = True
        settle = None
        if final_command != moving_gripper:
            settle = self._settle_gripper(motion, final_command)
        result = motion.finish(final_command, body=False, reached=reached,
                               waypoint_results=results, gripper_source=source,
                               gripper_command_executed=reached and not review_required and motion.reason is None,
                               **({"gripper_settle": settle} if settle else {}),
                               **({"repeat_warning": repeat} if repeat else {}),
                               load_compensation={"level": load_level, "rotation_level": rot_level,
                                                  "max_levels": HOLD_LOAD_LEVELS,
                                                  "step_cap_m": min(EEF_STEP_M, HOLD_EEF_STEP_M * 2 ** load_level)
                                                  if holding else EEF_STEP_M,
                                                  "deviation_window_steps": window})
        after = motion.state
        if settle is not None:
            settle["opening_after_stop_m"] = float(np.abs(after[14:16]).sum())
        tolerance = HOLD_SAG_TOLERANCE_M if results and results[-1]["settled_under_load"] else _POSITION_TOLERANCE_M
        result["target_reached"] = bool(reached and motion.reason is None
                                        and np.linalg.norm(targets[-1][0] - after[7:10]) <= tolerance
                                        and np.linalg.norm(_quaternion_error_rotvec(after[10:14], targets[-1][1])) <= _ROTATION_TOLERANCE_RAD)
        if review_required:
            result.update(stop_reason="preclose_review_required", review_required=True,
                          gripper_command_executed=False)
        self._remember("move_eef_path", signature, result["target_reached"])
        return result

    def pour_into(self, expected_step: int, target: list, *, rim=None, container_center=None,
                  container_depth_m=0.08, clearance_m=None, step_degrees=10, max_degrees=90,
                  return_upright=False, max_steps=250):
        """Tilt the held container about its rim over a receiving point (1.0.12; robots.robocasa.pour_tool)."""
        from robots.robocasa.pour_tool import pour_into

        return pour_into(self, expected_step, target, rim=rim, container_center=container_center,
                         container_depth_m=container_depth_m, clearance_m=clearance_m,
                         step_degrees=step_degrees, max_degrees=max_degrees, return_upright=return_upright,
                         max_steps=max_steps)

    def place_container(self, expected_step: int, target: list, *, clearance_m=0.10, release=True, max_steps=250):
        """Set held handle cookware down with its model centre over a destination (1.2.1; robots.robocasa.pour_tool)."""
        from robots.robocasa.pour_tool import place_container

        return place_container(self, expected_step, target, clearance_m=clearance_m, release=release,
                               max_steps=max_steps)

    def _settle_gripper(self, motion: _Motion, command: str) -> dict[str, Any]:
        """Hold a gripper change until the measured opening settles (1.0.11 A3).

        Arm deltas stay zero. Stops after GRASP_STABLE_CLOSE_STEPS readings
        within GRASP_STABLE_WIDTH_TOLERANCE_M, after GRIPPER_SETTLE_MAX_STEPS,
        on an interrupt, or with one action left for the final stop command.
        """
        widths = [float(np.abs(motion.state[14:16]).sum())]
        stable = False
        rule = None
        while motion.available > 1 and len(widths) <= GRIPPER_SETTLE_MAX_STEPS and motion.reason is None:
            motion.execute(motion.action(command))
            widths.append(float(np.abs(motion.state[14:16]).sum()))
            window = widths[-GRASP_STABLE_CLOSE_STEPS:]
            if (len(widths) > GRASP_STABLE_CLOSE_STEPS
                    and max(window) - min(window) <= GRASP_STABLE_WIDTH_TOLERANCE_M + 1e-9):
                stable, rule = True, "stable_window"
            elif (command == "close" and len(widths) > GRIPPER_CONTACT_STEPS
                  and widths[-1] > GRIPPER_EMPTY_OPENING_M
                  and widths[0] - widths[-1] >= GRASP_STABLE_WIDTH_TOLERANCE_M
                  and all(abs(a - b) <= GRIPPER_CONTACT_RATE_M + 1e-9
                          for a, b in zip(widths[-GRIPPER_CONTACT_STEPS - 1:-1], widths[-GRIPPER_CONTACT_STEPS:]))):
                stable, rule = True, "contact_rate"
            if stable:
                break
        before, after = widths[0], widths[-1]
        report: dict[str, Any] = {
            "command": command, "steps": len(widths) - 1, "max_steps": GRIPPER_SETTLE_MAX_STEPS,
            "opening_before_m": before, "opening_after_m": after, "stable": stable, "settle_rule": rule,
            "evidence_scope": "finger width only; object identity, contact and retention need fresh images",
        }
        if command == "close":
            report["closed_empty"] = stable and after <= GRIPPER_EMPTY_OPENING_M
            report["closure_confirmed"] = bool(
                stable and after > GRIPPER_EMPTY_OPENING_M and before - after >= GRASP_STABLE_WIDTH_TOLERANCE_M
            )
        else:
            report["release_confirmed"] = bool(after >= GRIPPER_OPEN_OPENING_M or (stable and after > before))
        return report

    def adjust_torso_to(self, expected_step: int, delta_m: float, *, max_steps=50):
        motion = _Motion(self.tk, "adjust_torso_to", expected_step, max_steps, 50)
        delta = _number(delta_m, "delta_m")
        if abs(delta) > TORSO_FULL_TRAVEL_M + 1e-9:
            raise ValueError("delta_m must not exceed the 0.34 metre full torso travel")
        gripper, source = self.tk._resolve_gripper_command("keep")
        self.tk._require_reviewed_direct_close(gripper)
        # The public base center rides on the torso (pinned MobileRobot
        # observables / Omron XML), so its world z measures lift travel while
        # base-relative EEF z does not. Close the loop on that public height;
        # never read the hidden torso joint.
        start_z = float(motion.before[2])
        target_z = start_z + delta
        cap = HOLD_TORSO_COMMAND_MAX if motion.holding else TORSO_COMMAND_MAX
        motion.sync_arm_goal(gripper)
        stalled = 0
        reached = abs(target_z - float(motion.state[2])) <= TORSO_TOLERANCE_M
        while not reached and not motion.reason and motion.available > 1:
            z = float(motion.state[2])
            command = float(np.clip(TORSO_GAIN * (target_z - z) / TORSO_RESPONSE_M_PER_COMMAND, -cap, cap))
            action = motion.action(gripper, body=True)
            action[3] = command
            motion.execute(action)
            if motion.reason:
                break
            after = float(motion.state[2])
            reached = abs(target_z - after) <= TORSO_TOLERANCE_M
            # No measurable lift under a real command: joint limit or blocked.
            stalled = stalled + 1 if abs(after - z) < TORSO_STALL_M and abs(command) >= 0.1 else 0
            if stalled >= TORSO_STALL_STEPS:
                motion.reason = "deviation"
        if not reached and motion.reason is None:
            motion.reason = "budget"
        result = motion.finish(gripper, body=True, reached=reached,
                               torso_measurement="public_base_center_z",
                               torso_control="closed_loop_public_height",
                               requested_delta_m=delta, target_base_z_m=target_z,
                               gripper_source=source)
        final_z = float(motion.state[2])
        result.update(measured_delta_m=final_z - start_z, remaining_delta_m=target_z - final_z,
                      torso_limit_suspected=motion.reason == "deviation",
                      target_reached=bool(abs(target_z - final_z) <= TORSO_TOLERANCE_M
                                          and motion.reason is None),
                      measured_eef_delta_z_m=float(motion.state[9] - motion.before[9]))
        return result

    def zoom(self, camera_id: str, box: list[int], *, scale=4):
        step = int(self.tk._operation.metrics["step_count"])
        if self.tk._last_observed_step != step:
            raise ValueError("zoom requires observe of the same current source image")
        if camera_id not in _CAMERAS:
            raise ValueError("camera_id must identify an image already exposed by observe")
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
            raise ValueError("box must contain four integer pixel coordinates")
        scale = _number(scale, "scale")
        if scale <= 0:
            raise ValueError("scale must be positive")
        # Match observe's original pixel encoding; no perception_source call or
        # rendering, depth, new frame, or higher-resolution sensor acquisition.
        original_bytes = _png_bytes(self.tk._operation.observation[camera_id])
        with Image.open(io.BytesIO(original_bytes)) as original:
            r0, c0, r1, c1 = box
            if not 0 <= r0 < r1 <= original.height or not 0 <= c0 < c1 <= original.width:
                raise ValueError("box must be nonempty and within the source image")
            size = (round((c1 - c0) * scale), round((r1 - r0) * scale))
            if min(size) < 1 or size[0] * size[1] > 16_777_216:
                raise ValueError("scaled crop must contain 1 to 16777216 pixels")
            crop = original.crop((c0, r0, c1, r1)).resize(size, Image.Resampling.NEAREST)
            output = io.BytesIO()
            crop.save(output, format="PNG")
        # ``step`` lets image archives bind the crop to its observation; the
        # hidden derivation record marks it as a crop, not a camera frame
        # (1.0.11 C4). Neither changes what the model sees.
        derivation = {"kind": "zoom", "source_camera": camera_id, "source_step": step, "box": list(box),
                      "scale": scale, "source_size": [original.width, original.height],
                      "output_size": list(size)}
        return {"step": step, "source_step": step, "camera_id": camera_id, "box": box, "scale": scale,
                "source_size": [original.width, original.height], "output_size": list(size),
                "executed_actions": 0, "_image_bytes": output.getvalue(),
                "_image_camera_ids": {camera_id: "_image_bytes"},
                "_image_derivation": {"_image_bytes": derivation},
                "evidence_scope": "nearest-neighbour enlargement of the same observed pixels"}


def register_privileged_tools(toolkit) -> None:
    handlers = PrivilegedTools(toolkit)
    vector = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    gripper = {"type": "string", "enum": ["keep", "open", "close"], "default": "keep"}
    definitions = [
        ("navigate_to", {
            "target": {"oneOf": [
                {"type": "object", "properties": {"relative": vector}, "required": ["relative"], "additionalProperties": False},
                {"type": "object", "properties": {"point": vector, "standoff_m": {"type": "number", "minimum": 0}},
                 "required": ["point", "standoff_m"], "additionalProperties": False}]}, "gripper": gripper,
             "turn_first": {"type": "boolean", "default": False}},
         ["target"], 200,
         "Closed-loop base navigation from a fresh observe at expected_step. target is relative [dx_m,dy_m,dyaw_deg] in the initial base frame (<=3 m, <=180 degrees), or a public back_project point [x,y,z] and standoff_m in that frame (the base stops standoff_m short of the point, facing it). The base is omnidirectional: by default it translates straight in its own frame while holding its heading, then turns to the final heading in place; turn_first=true restores turn, advance, align. Zero arm deltas. Preserves the chosen gripper command; closing an open gripper requires the wrist review. No automatic obstacle avoidance: on base_stalled, base_stall names the blocked phase and direction. A near-repeat of an unreached target returns repeat_warning."),
        ("move_eef_path", {
            "waypoints": {"type": "array", "minItems": 1, "maxItems": 8,
                          "items": {"type": "object", "properties": {"position": vector, "rotation_degrees": vector},
                                    "required": ["position"], "additionalProperties": False}}, "gripper": gripper},
         ["waypoints"], 150,
         "Track 1-8 public robot-base-frame absolute EEF positions from a fresh observe at expected_step. Each segment <=0.30 m; optional rotation_degrees is a base-axis axis-angle increment from the previous waypoint, norm <=90 degrees. Execute gripper only after the last point: closing an open gripper needs a current review_grasp approval of the pose where it closes, so approach with the gripper open, observe, review, then call with gripper=close and the current position as the only waypoint; keeping a closed gripper needs no review. A gripper change is held for up to 15 steps until the measured opening is stable; gripper_settle reports the openings and closure_confirmed/closed_empty or release_confirmed (finger width only). Stops on sustained deviation; records every reached/interrupted/timed-out waypoint. A near-repeat of an unreached target returns repeat_warning."),
        ("pour_into", {
            "target": vector, "rim": vector, "container_center": vector,
            "container_depth_m": {"type": "number", "minimum": 0.02, "maximum": 0.3, "default": 0.08},
            "clearance_m": {"type": "number", "minimum": 0.02, "maximum": 0.3},
            "step_degrees": {"type": "number", "minimum": 5, "maximum": 20, "default": 10},
            "max_degrees": {"type": "number", "minimum": 20, "maximum": 120, "default": 90},
            "return_upright": {"type": "boolean", "default": False}},
         ["target"], 250,
         "Pour or tilt the held container over a receiving point, from a fresh observe at expected_step, keeping the gripper closed. "
         "target is a public back_project point [x,y,z] on the receiving surface (the plate or container centre, robot base frame). "
         "The tool lifts the container's pour rim to target z + clearance_m (default container_depth_m + 0.04), moves the rim over the "
         "target, rotates the container about that rim in step_degrees increments up to max_degrees (default 90; contents leave by then, "
         "and tilting further swings the hand into the receiver) while tracking the measured angle, "
         "dwells, returns upright and stops. Default geometry for handle-held cookware: body centre 0.20 m "
         "along the handle axis and 5 cm above the hand, rim radius 0.11 m, pour rim 1 cm below the centre and at least 40 degrees around from the handle. "
         "The tool always locates the held container from the hand pose with this model: back_projected container points fall on its "
         "surface, not its centre, so container_center and rim are only reported against the model, never used; pass only target. Always call it with the "
         "default max_steps (250; below 150 is refused): the whole pour from the grasp pose takes about 450 steps, so short calls only "
         "fragment it and spend the segment quota on alignment. One call rarely "
         "holds the whole pour: the result reports stage (align, tilting, tilted, returning, done) and continue_required; while the "
         "container is tilted the tool keeps the pour geometry, so observe, resolve verification and call pour_into again with the "
         "same target until stage is done (a different target is refused until the container is upright); return_upright=true skips "
         "further tilting and only brings a tilted container back upright, for a restore step after the segment ended mid-pour. Stops on grasp loss, "
         "repeated deviation or budget. target_reached means only that the tilt reached max_degrees and the container is upright "
         "again: verify from fresh images of the receiver whether the contents landed, then place the container."),
        ("place_container", {
            "target": vector,
            "clearance_m": {"type": "number", "minimum": 0.03, "maximum": 0.3, "default": 0.10},
            "release": {"type": "boolean", "default": True}},
         ["target"], 250,
         "Set held handle cookware (a pan or pot held by its handle) down at a destination, from a fresh observe at "
         "expected_step. target is a public back_project point [x,y,z] at the centre of the destination surface "
         "(for example a burner), robot base frame. Ground it afresh from current images when you take over: a "
         "policy may move a container while lifting it, so the hand pose at the handover is not where the "
         "container stood. The tool locates the container centre from the hand pose with pour_into's handle model, "
         "lifts it to target z + clearance_m, moves the centre over the target, lowers until the support stops the "
         "hand and, with release=true (default), opens the gripper, retreats 10 cm up and backs off 15 cm toward the "
         "robot. The result reports the "
         "model centre's xy error to the target (center_xy_error_m) and whether the support stopped the descent "
         "(support_contact). Refused while a pour_into is still tilted. Centres are model estimates: verify from "
         "fresh images that the container rests on the destination and the gripper is clear."),
        ("adjust_torso_to", {"delta_m": {"type": "number", "minimum": -TORSO_FULL_TRAVEL_M, "maximum": TORSO_FULL_TRAVEL_M}},
         ["delta_m"], 50,
         "Closed-loop relative torso lift, positive up, from a fresh observe at expected_step; abs(delta_m)<=0.34 m full travel. Measured on the public base-center height, which rides on the torso (base-relative EEF z does not change). One call moves about 0.2 m at most (0.1 m while holding); call again for the reported remaining_delta_m. Stops within 3 mm, on no measured lift (torso_limit_suspected) or budget. Zero base velocity and arm deltas; confirmed gripper command retained."),
    ]
    for name, properties, required, ceiling, description in definitions:
        toolkit.add_tool(name, {"name": name, "description": description +
                               " Includes a final zero command within max_steps and episode/call budgets; body motions start with one zero arm-mode step that re-anchors the arm goal. Terminal success, termination and grasp loss stop immediately. Afterward observe and resolve Harness verification; hand back to the policy with continue_vla (intervene=true with the next operation's frozen sentence), not resume_vla. Full-speed base travel covers roughly 25 mm per step (less over the start ramp and the last 0.5 m); size max_steps from the distance. Holding keeps that travel speed with a gentler acceleration ramp and turn rate; arm and torso moves use smaller steps while holding. Returned motion is advisory, not object/contact or task-success evidence.",
                               "input_schema": {"type": "object", "properties": {
                                   "expected_step": {"type": "integer", "minimum": 0}, **properties,
                                   "max_steps": {"type": "integer", "minimum": 2, "maximum": ceiling, "default": ceiling}},
                                   "required": ["expected_step", *required], "additionalProperties": False}},
                         getattr(handlers, name))
    toolkit.add_tool("zoom", {"name": "zoom", "description":
                             "Crop box [row0,col0,row1,col1] from the same currently observed camera image and enlarge its existing pixels (nearest neighbour, scale default 4). Requires observe; no new render, simulation step, depth or action. Pixel bounds must be inside that image.",
                             "input_schema": {"type": "object", "properties": {
                                 "camera_id": {"type": "string", "enum": list(_CAMERAS)},
                                 "box": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0}},
                                 "scale": {"type": "number", "exclusiveMinimum": 0, "default": 4}},
                                 "required": ["camera_id", "box"], "additionalProperties": False}}, handlers.zoom)
