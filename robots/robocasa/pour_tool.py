"""pour_into: tilt a held container over a receiving point (1.0.12).

The 1.0.11 full run (D10) and the scripted fork probe of 2026-10-11 showed the
split: with the pour rim held about container depth + 2 cm above the plate and
the container rotated about that rim, the vegetable landed on the plate and the
task succeeded; rotating about the end effector at the same angles (what the
LLM composed from move_eef_path waypoints) dropped it on the counter. This tool
owns that geometry. The LLM supplies only the receiving point from back_project.

1.2.1 (D10 step-130 fork sweep, 2026-10-13): the container is always located by
the handle model below. back_projected container points land on its surface,
not its centre, and were 2.4-6.5 cm off in the 1.2.0 runs (the model 0.4-0.9
cm), shifting the real pour by that much; given rim/container_center are now
only reported next to the model. Tilting stops at 90 degrees by default: the
contents left the pan at 85-98 degrees of pan tilt in every cell, and going on
to 106-110 degrees swung the gripper into the plate (moved 0.8-4 cm), after
which the landed vegetable slid 1.5-3.6 cm further.

Public state only: EEF pose, gripper opening, base pose. No simulator internals.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from robots.robocasa.toolkit import (
    _OPEN_GRIPPER_THRESHOLD,
    _POSITION_ACTION_SCALE_M,
    _POSITION_TOLERANCE_M,
    _ROTATION_ACTION_SCALE_RAD,
    _ROTATION_TOLERANCE_RAD,
    _finite_vector,
    _quaternion_error_rotvec,
    _rotate_base_quaternion,
)

# Handle-held cookware model, measured on the D10 pan at the pi0.5 handoff pose
# (fork probe 2026-10-11): body centre 0.20 m along the handle axis, 5 cm above
# the EEF site; rim radius 0.11 m; depth 8 cm. A pan hanging from its handle
# sags toward the pour side, so the pour rim is modelled 1 cm below the centre;
# with the default clearance (depth + 4 cm) the probe-measured pan clears the
# plate, where depth + 2 cm grazed and pushed it (fork validation 2026-10-12).
DEFAULT_CENTER_ALONG_HANDLE_M = 0.20
DEFAULT_CENTER_ABOVE_EEF_M = 0.05
DEFAULT_RIM_RADIUS_M = 0.11
DEFAULT_RIM_ABOVE_CENTER_M = -0.01
DEFAULT_CONTAINER_DEPTH_M = 0.08
CLEARANCE_MARGIN_M = 0.04
MIN_HANDLE_ANGLE_DEG = 40.0  # pour rim at least this far around from the handle (and gripper)
LIFT_MARGIN_M = 0.03
SEGMENT_M = 0.25
POUR_CEILING = 250
POUR_MIN_STEPS = 150  # shorter calls fragment the pour; D10 run of 2026-10-12 used nine 50-step calls and lost the segment quota mid-return
DWELL_STEPS = 8
RETURN_STEP_DEG = 20.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        raise ValueError("degenerate direction")
    return v / n


def _horizontal(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[1], 0.0])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _unit(np.asarray(q, dtype=float))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_mat(axis: np.ndarray, theta: float) -> np.ndarray:
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _rotate_in_plane(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])


def _tilt_angle(q_now: np.ndarray, q_start: np.ndarray) -> float:
    """Measured rotation (degrees) of the EEF since the pour started."""
    return math.degrees(float(np.linalg.norm(_quaternion_error_rotvec(q_start, q_now))))


def _tilt_about(q_now: np.ndarray, q_start: np.ndarray, axis: np.ndarray) -> float:
    """Rotation (degrees) about the pour axis only; a wrist yaw is not a tilt."""
    return math.degrees(abs(float(_quaternion_error_rotvec(q_start, q_now) @ np.asarray(axis))))


def model_center(state: np.ndarray, tool: str = "pour_into") -> tuple[np.ndarray, np.ndarray]:
    """The handle model's container centre and horizontal handle direction (away from the robot)."""
    eef, q = state[7:10].copy(), state[10:14].copy()
    handle = _quat_to_mat(q)[:, 0]  # Panda hand local x: the axis a handle lies along between the fingers
    try:
        handle_h = _unit(_horizontal(handle))
    except ValueError as exc:
        raise ValueError(f"{tool} cannot resolve the handle axis: the hand points straight up or down") from exc
    # The container body is on the far side of the hand from the robot.
    away = _horizontal(eef)
    if np.linalg.norm(away) > 1e-6 and float(handle_h @ _unit(away)) < 0:
        handle_h = -handle_h
    return eef + DEFAULT_CENTER_ALONG_HANDLE_M * handle_h + np.array([0, 0, DEFAULT_CENTER_ABOVE_EEF_M]), handle_h


def _gripper_opened_since(toolkit, step: int) -> bool:
    """Whether an executed command after step opened the gripper (native value below 0.5)."""
    for record in toolkit._operation.action_history_since():
        for step_id, action in zip(record.get("step_ids", []), record.get("actions", [])):
            if step_id > step and float(action[11]) < 0.5:
                return True
    return False


def held_model_center(handlers, state: np.ndarray, tool: str) -> tuple[np.ndarray, str]:
    """The held container's model centre, with its side of the hand fixed for the whole grasp.

    model_center's "far side of the hand from the robot" holds where the container
    was grasped, but not after the hand has carried it sideways: in the 1.2.1 fork
    of D10, with the hand over the plate it flipped and put the centre 40 cm off.
    So the centre is taken in the hand frame where it was first modelled while the
    container is held, else at the grasp closure the grasp watch kept, else here,
    and is carried with the hand from there. The first model stays valid until an
    executed command opens the gripper. It is not tied to the grasp watch's cycle:
    in the first 1.2.1 D10 run every back_project restarted that cycle (and cleared
    its closure pose) while the pan stayed in the hand, so the model was rebuilt
    over the plate, flipped, and the first two placements failed.
    """
    tk = handlers.tk
    cached = getattr(handlers, "_held_container", None)
    if (cached is not None and float(np.abs(state[14:16]).sum()) < _OPEN_GRIPPER_THRESHOLD
            and not _gripper_opened_since(tk, cached["step"])):
        return state[7:10] + _quat_to_mat(state[10:14]) @ cached["center_e"], cached["source"]
    closure = getattr(getattr(tk, "_pick_progress", None), "closure_state", None)
    if closure is not None:
        reference, source = np.asarray(closure, dtype=float), "grasp_closure_pose"
    else:
        reference, source = np.asarray(state, dtype=float), f"{tool}_start_pose"
    center, _ = model_center(reference, tool)
    center_e = _quat_to_mat(reference[10:14]).T @ (center - reference[7:10])
    handlers._held_container = {"step": int(tk._operation.metrics["step_count"]), "center_e": center_e,
                                "source": source}
    return state[7:10] + _quat_to_mat(state[10:14]) @ center_e, source


def plan_geometry(state: np.ndarray, target: np.ndarray, *, rim, container_center,
                  container_depth_m: float, clearance_m) -> dict[str, Any]:
    """Pour geometry in the robot base frame from public state and the given points."""
    eef, q = state[7:10].copy(), state[10:14].copy()
    R = _quat_to_mat(q)
    center, handle_h = model_center(state)
    if container_center is not None:
        center = np.asarray(container_center, dtype=float)
        try:  # the handle runs from the hand toward the body centre
            handle_h = _unit(_horizontal(center - eef))
        except ValueError:
            pass
    to_target = _horizontal(target - center)
    if np.linalg.norm(to_target) < 0.02:
        raise ValueError("target lies under the container centre; give the receiving container's centre, "
                         "not a point under the held container")
    pour_dir = _unit(to_target)
    # The hand (and handle) sit on the -handle_h side of the body centre; keep the
    # pour rim at least MIN_HANDLE_ANGLE_DEG around from that side so the contents
    # clear the gripper. The far side is the safest.
    hand_side = -handle_h
    handle_angle = math.degrees(math.acos(max(-1.0, min(1.0, float(pour_dir @ hand_side)))))
    adjusted = False
    if handle_angle < MIN_HANDLE_ANGLE_DEG:
        side = 1.0 if float(np.cross(hand_side, pour_dir)[2]) >= 0 else -1.0
        pour_dir = _unit(_rotate_in_plane(hand_side, side * math.radians(MIN_HANDLE_ANGLE_DEG)))
        adjusted = True
    if rim is not None:
        rim_point = np.asarray(rim, dtype=float)
    else:
        rim_point = center + DEFAULT_RIM_RADIUS_M * pour_dir + np.array([0, 0, DEFAULT_RIM_ABOVE_CENTER_M])
    clearance = float(clearance_m) if clearance_m is not None else float(container_depth_m) + CLEARANCE_MARGIN_M
    rim_target = np.array([target[0], target[1], target[2] + clearance])
    axis = _unit(np.cross(np.array([0, 0, 1.0]), pour_dir))
    if (_axis_angle_mat(axis, math.radians(30)) @ (-pour_dir))[2] < 0:
        axis = -axis  # +theta must raise the body relative to the rim
    return {
        "eef": eef, "quaternion": q, "rotation": R,
        "handle_dir": handle_h, "center": center, "pour_dir": pour_dir,
        "hand_side_angle_deg": handle_angle, "pour_dir_adjusted_from_handle": adjusted,
        "rim": rim_point, "rim_in_eef_frame": R.T @ (rim_point - eef),
        "center_in_eef_frame": R.T @ (center - eef),
        "rim_target": rim_target, "clearance_m": clearance, "axis": axis,
        "rim_source": "given" if rim is not None else "model",
        "center_source": "given" if container_center is not None else "model",
    }


class _Tracker:
    """Servo the EEF to a pose with move_eef_path's holding-aware step and deviation rules."""

    def __init__(self, motion, consts):
        self.m, self.c = motion, consts
        self.load_level = self.rot_level = 0

    def track(self, position: np.ndarray, quaternion: np.ndarray, *, max_actions: int, gripper: str = "close") -> str:
        c, m = self.c, self.m
        holding = m.holding
        window = c.HOLD_DEVIATION_WINDOW if holding else c.DEVIATION_STEPS
        rate_m = c.HOLD_DEVIATION_PROGRESS_M if holding else c.DEVIATION_PROGRESS_M
        rate_rad = c.HOLD_DEVIATION_PROGRESS_RAD if holding else c.DEVIATION_PROGRESS_RAD
        start = m.state
        if position[2] < start[9] - _POSITION_TOLERANCE_M:
            self.load_level = 0
        errors = [(float(np.linalg.norm(position - start[7:10])),
                   float(np.linalg.norm(_quaternion_error_rotvec(start[10:14], quaternion))))]
        commanded: list[tuple[float, float]] = []
        since_change = mismatch_run = executed = 0
        while m.available > 1 and executed < max_actions:
            state = m.state
            pos_error = position - state[7:10]
            rot_error = _quaternion_error_rotvec(state[10:14], quaternion)
            if np.linalg.norm(pos_error) <= _POSITION_TOLERANCE_M and np.linalg.norm(rot_error) <= _ROTATION_TOLERANCE_RAD:
                return "reached"
            if holding:
                step_m = min(c.EEF_STEP_M, c.HOLD_EEF_STEP_M * 2 ** self.load_level)
                step_rad = min(c.EEF_ROTATION_STEP_RAD, c.HOLD_EEF_ROTATION_STEP_RAD * 2 ** self.rot_level)
            else:
                step_m, step_rad = c.EEF_STEP_M, c.EEF_ROTATION_STEP_RAD
            increment = c._cap(pos_error, step_m)
            rot_increment = c._cap(rot_error, step_rad)
            action = m.action(gripper)
            action[5:8] = increment / _POSITION_ACTION_SCALE_M
            action[8:11] = rot_increment / _ROTATION_ACTION_SCALE_RAD
            m.execute(action)
            executed += 1
            if m.reason:
                return m.reason
            after = m.state
            errors.append((float(np.linalg.norm(position - after[7:10])),
                           float(np.linalg.norm(_quaternion_error_rotvec(after[10:14], quaternion)))))
            commanded.append((float(np.linalg.norm(increment)), float(np.linalg.norm(rot_increment))))
            since_change += 1
            mismatch = np.linalg.norm(state[7:10] + increment - after[7:10])
            rotation_mismatch = np.linalg.norm(_quaternion_error_rotvec(
                after[10:14], _rotate_base_quaternion(state[10:14], rot_increment)))
            mismatch_run = mismatch_run + 1 if mismatch > c.DEVIATION_POSITION_M or rotation_mismatch > c.DEVIATION_ROTATION_RAD else 0
            if mismatch_run >= c.DEVIATION_STEPS:
                return "deviation"
            if since_change < window:
                continue
            (old_m, old_rad), (new_m, new_rad) = errors[-window - 1], errors[-1]
            pos_active = new_m > _POSITION_TOLERANCE_M
            rot_active = new_rad > _ROTATION_TOLERANCE_RAD
            gain_m, gain_rad = old_m - new_m, old_rad - new_rad
            pos_lag = holding and pos_active and gain_m < c.HOLD_LAG_RATIO * sum(x[0] for x in commanded[-window:])
            rot_lag = holding and rot_active and gain_rad < c.HOLD_LAG_RATIO * sum(x[1] for x in commanded[-window:])
            if (pos_lag and self.load_level < c.HOLD_LOAD_LEVELS) or (rot_lag and self.rot_level < c.HOLD_LOAD_LEVELS):
                self.load_level += int(pos_lag and self.load_level < c.HOLD_LOAD_LEVELS)
                self.rot_level += int(rot_lag and self.rot_level < c.HOLD_LOAD_LEVELS)
                since_change = 0
                continue
            if (pos_active or rot_active) and not (
                (pos_active and gain_m >= window * rate_m) or (rot_active and gain_rad >= window * rate_rad)
            ):
                if holding and not rot_active and new_m <= c.HOLD_SAG_TOLERANCE_M:
                    return "settled_under_load"
                return "deviation"
        state = m.state
        if (np.linalg.norm(position - state[7:10]) <= _POSITION_TOLERANCE_M
                and np.linalg.norm(_quaternion_error_rotvec(state[10:14], quaternion)) <= _ROTATION_TOLERANCE_RAD):
            return "reached"
        return "budget"


def pour_into(handlers, expected_step: int, target: list[float], *, rim=None, container_center=None,
              container_depth_m: float = DEFAULT_CONTAINER_DEPTH_M, clearance_m=None,
              step_degrees: float = 10.0, max_degrees: float = 90.0, return_upright: bool = False,
              max_steps: int = POUR_CEILING) -> dict[str, Any]:
    """Resumable pour: align, tilt about the rim, dwell, return upright, across as many calls as the budget needs.

    return_upright=True skips any further tilting and brings a tilted container back
    upright along the remembered arc (for a restore step after an interrupted pour).
    rim and container_center are accepted but not used: the geometry always comes
    from the handle model, and the result reports the given points against it.
    """
    from robots.robocasa import privileged_tools as consts

    tk = handlers.tk
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < POUR_MIN_STEPS:
        raise ValueError(f"pour_into needs at least {POUR_MIN_STEPS} steps per call (default {POUR_CEILING}); "
                         "short calls fragment the pour and spend the segment quota on alignment")
    if not isinstance(return_upright, bool):
        raise ValueError("return_upright must be a boolean")
    target = _finite_vector(target, size=3, label="target")
    rim_v = None if rim is None else _finite_vector(rim, size=3, label="rim")
    center_v = None if container_center is None else _finite_vector(container_center, size=3, label="container_center")
    for name, value, low, high in (("container_depth_m", container_depth_m, 0.02, 0.30),
                                   ("clearance_m", clearance_m, 0.02, 0.30),
                                   ("step_degrees", step_degrees, 5.0, 20.0),
                                   ("max_degrees", max_degrees, 20.0, 120.0)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be a number between {low} and {high}")
    motion = consts._Motion(tk, "pour_into", expected_step, max_steps, POUR_CEILING)
    try:
        command, source = tk._resolve_gripper_command("keep")
    except ValueError:
        command, source = None, None
    if command != "close" or not motion.holding:
        raise ValueError("pour_into needs a held container: the gripper must be closed on an object "
                         "(measured opening between 2 mm and 6 cm); grasp first, then pour")

    # A pour in progress (container still tilted) keeps its geometry: the rim
    # model assumes a level container and must not be re-fitted mid-tilt.
    pending = getattr(handlers, "_pour_state", None)
    resumed = False
    moved_between = False
    if pending is not None:
        tilted = _tilt_about(motion.before[10:14], pending["q_start"], pending["axis"]) >= 5.0
        # Another tool or the policy moved the robot since the last pour call.
        moved_between = pending.get("end_step") not in (None, int(tk._operation.metrics["step_count"]))
        if moved_between and not tilted:
            pending = handlers._pour_state = None  # level again: start a fresh pour from this pose
    if pending is not None:
        same_target = float(np.linalg.norm(pending["target"] - target)) <= 0.02
        if tilted and not same_target:
            raise ValueError("a pour is in progress with a different target; call pour_into with the same "
                             "target to finish it (it returns the container upright) before changing targets")
        if tilted or (same_target and pending.get("stage") in {"align", "tilting"}):
            resumed = True
    if return_upright and not resumed:
        raise ValueError("return_upright applies to a tilted container from an interrupted pour_into; "
                         "the container is upright (or no pour is remembered), so there is nothing to return")
    if resumed:
        st = pending
        geometry = st["geometry"]
        if moved_between and st["pivot"] is not None:
            # The rim rides with the hand: re-anchor the pivot where the rim is now,
            # keeping the start orientation and lever (the arc is unchanged).
            s = motion.before
            st["pivot"] = s[7:10] + _quat_to_mat(s[10:14]) @ st["rim_e"]
        if return_upright and st["stage"] in {"align", "tilting", "tilted"}:
            st["stage"] = "returning" if st["pivot"] is not None else "done"
    else:
        held_center, center_reference = held_model_center(handlers, motion.before, "pour_into")
        geometry = plan_geometry(motion.before, target, rim=None, container_center=held_center,
                                 container_depth_m=float(container_depth_m), clearance_m=clearance_m)
        geometry["center_source"] = "model"
        geometry["center_reference"] = center_reference
        ignored = {}
        if center_v is not None:
            ignored["container_center"] = center_v.tolist()
            ignored["container_center_minus_model_m"] = float(np.linalg.norm(center_v - geometry["center"]))
        if rim_v is not None:
            ignored["rim"] = rim_v.tolist()
            ignored["rim_minus_model_m"] = float(np.linalg.norm(rim_v - geometry["rim"]))
        geometry["ignored_inputs"] = ignored
        st = {"target": target.copy(), "geometry": geometry, "q_start": geometry["quaternion"].copy(),
              "rim_e": geometry["rim_in_eef_frame"], "axis": geometry["axis"], "rim_target": geometry["rim_target"],
              "max_degrees": float(max_degrees), "stage": "align", "tilt_reached": 0.0,
              "pivot": None, "lever": None, "calls": 0}
    st["calls"] += 1
    st["max_degrees"] = float(max_degrees)
    handlers._pour_state = st
    rim_e, axis, rim_target, q_start = st["rim_e"], st["axis"], st["rim_target"], st["q_start"]
    tracker = _Tracker(motion, consts)
    phases: list[dict[str, Any]] = []
    align_report = None

    def rim_now() -> np.ndarray:
        s = motion.state
        return s[7:10] + _quat_to_mat(s[10:14]) @ rim_e

    def phase(name: str, position: np.ndarray, quaternion: np.ndarray, max_actions: int) -> str:
        before = int(motion.result["executed_actions"])
        outcome = tracker.track(position, quaternion, max_actions=max(0, max_actions))
        phases.append({"phase": name, "outcome": outcome, "steps": int(motion.result["executed_actions"]) - before,
                       "position_error_m": float(np.linalg.norm(position - motion.state[7:10])),
                       "rotation_error_rad": float(np.linalg.norm(_quaternion_error_rotvec(motion.state[10:14], quaternion)))})
        return outcome

    def terminal(outcome: str) -> bool:
        return outcome in {"success", "terminated", "grasp_lost"}

    def go(name: str, position: np.ndarray, quaternion: np.ndarray) -> str:
        """Translate in <=SEGMENT_M segments toward position within the remaining budget."""
        while True:
            s = motion.state
            delta = position - s[7:10]
            distance = float(np.linalg.norm(delta))
            if motion.available <= 1:
                return "budget"
            point = position if distance <= SEGMENT_M else s[7:10] + delta / distance * SEGMENT_M
            outcome = phase(name, point, quaternion, motion.available - 1)
            if outcome not in {"reached", "settled_under_load"} or distance <= SEGMENT_M:
                return outcome

    try:
        if st["stage"] == "align":
            r = rim_now()
            lift = max(0.0, rim_target[2] + LIFT_MARGIN_M - r[2])
            outcome = "reached"
            if lift > _POSITION_TOLERANCE_M:
                outcome = go("lift", motion.state[7:10] + np.array([0, 0, lift]), q_start)
            if not terminal(outcome) and outcome != "budget":
                r = rim_now()
                outcome = go("translate", motion.state[7:10] + np.array([rim_target[0] - r[0], rim_target[1] - r[1], 0.0]), q_start)
            if not terminal(outcome) and outcome != "budget":
                r = rim_now()
                outcome = go("descend", motion.state[7:10] + np.array([0, 0, rim_target[2] - r[2]]), q_start)
            r = rim_now()
            aligned = float(np.linalg.norm(r[:2] - rim_target[:2])) <= 0.03 and abs(r[2] - rim_target[2]) <= 0.03
            align_report = {"rim_xy_error_m": float(np.linalg.norm(r[:2] - rim_target[:2])),
                            "rim_z_error_m": float(r[2] - rim_target[2]), "aligned": bool(aligned)}
            if terminal(outcome):
                raise StopIteration(outcome)
            if not aligned:
                if outcome not in {"budget"}:
                    motion.reason = motion.reason or "deviation"
                raise StopIteration(outcome)
            st["pivot"] = rim_now()
            st["lever"] = motion.state[7:10] - st["pivot"]
            st["q_start"] = motion.state[10:14].copy()
            q_start = st["q_start"]
            st["stage"] = "tilting"

        pivot, lever = st["pivot"], st["lever"]
        if st["stage"] == "tilting":
            deviations = 0
            theta = _tilt_angle(motion.state[10:14], q_start)
            while theta < float(max_degrees) - 1e-9:
                if motion.available <= 4:
                    raise StopIteration("budget")
                measured = _tilt_angle(motion.state[10:14], q_start)
                theta = min(float(max_degrees), max(theta, measured) + float(step_degrees))
                Rk = _axis_angle_mat(axis, math.radians(theta))
                outcome = phase(f"tilt-{int(round(theta))}", pivot + Rk @ lever,
                                _rotate_base_quaternion(q_start, axis * math.radians(theta)), motion.available - 1)
                st["tilt_reached"] = max(st["tilt_reached"], _tilt_angle(motion.state[10:14], q_start))
                if terminal(outcome):
                    raise StopIteration(outcome)
                if outcome == "deviation":
                    deviations += 1
                    if deviations >= 2:
                        motion.reason = "deviation"
                        raise StopIteration(outcome)
                    theta -= float(step_degrees)  # retry the same increment once
                    continue
                if outcome == "budget":
                    raise StopIteration(outcome)
            st["stage"] = "tilted"

        if st["stage"] == "tilted":
            for _ in range(DWELL_STEPS):
                if motion.available <= 2:
                    raise StopIteration("budget")
                motion.execute(motion.action("close"))
                if motion.reason:
                    raise StopIteration(motion.reason)
            st["stage"] = "returning"

        if st["stage"] == "returning":
            current = _tilt_angle(motion.state[10:14], q_start)
            angles = list(np.arange(max(current - RETURN_STEP_DEG, 0.0), -1e-9, -RETURN_STEP_DEG)) + [0.0]
            deviations = 0
            index = 0
            while index < len(angles):
                angle = angles[index]
                if motion.available <= 2:
                    raise StopIteration("budget")
                Rk = _axis_angle_mat(axis, math.radians(angle))
                outcome = phase(f"return-{int(round(angle))}", pivot + Rk @ lever,
                                _rotate_base_quaternion(q_start, axis * math.radians(angle)), motion.available - 1)
                if terminal(outcome) or outcome == "budget":
                    raise StopIteration(outcome)
                if outcome == "deviation":
                    deviations += 1
                    if deviations >= 2:  # blocked, e.g. the container touches the receiver
                        motion.reason = motion.reason or "deviation"
                        raise StopIteration(outcome)
                    continue  # retry the same angle once
                index += 1
            # Done only when upright; otherwise the next call keeps returning.
            upright = float(np.linalg.norm(_quaternion_error_rotvec(motion.state[10:14], q_start))) <= _ROTATION_TOLERANCE_RAD
            st["stage"] = "done" if upright else "returning"
    except StopIteration:
        pass

    state = motion.state
    upright_error = float(np.linalg.norm(_quaternion_error_rotvec(state[10:14], q_start)))
    if st["stage"] == "done" and upright_error > _ROTATION_TOLERANCE_RAD:
        st["stage"] = "returning"  # never report done for a container that is still tilted
    stage = st["stage"]
    contact_phases = [p["phase"] for p in phases
                      if p["phase"].startswith(("tilt-", "return-")) and p["outcome"] in {"settled_under_load", "deviation"}]
    done = stage == "done" and upright_error <= _ROTATION_TOLERANCE_RAD
    reached = done and motion.reason is None and (return_upright or st["tilt_reached"] >= float(max_degrees) - 5.0)
    if done or motion.reason in {"success", "terminated", "grasp_lost"}:
        handlers._pour_state = None
    continue_required = not done and motion.reason is None
    result = motion.finish("close", body=False, reached=reached,
                           gripper_source=source, phases=phases, stage=stage, resumed=resumed, call_index=st["calls"],
                           return_upright=return_upright,
                           continue_required=continue_required,
                           contact_suspected=bool(contact_phases), contact_phases=contact_phases,
                           tilt_target_deg=float(max_degrees), tilt_reached_deg=float(st["tilt_reached"]),
                           upright_error_rad=upright_error, align=align_report,
                           geometry={
                               "target": target.tolist(), "rim_target": np.asarray(rim_target).tolist(),
                               "rim": np.asarray(geometry["rim"]).tolist(), "rim_source": geometry["rim_source"],
                               "container_center": np.asarray(geometry["center"]).tolist(), "center_source": geometry["center_source"],
                               "pour_direction": np.asarray(geometry["pour_dir"]).tolist(), "rotation_axis": np.asarray(axis).tolist(),
                               "handle_direction": np.asarray(geometry["handle_dir"]).tolist(),
                               "hand_side_angle_deg": geometry["hand_side_angle_deg"],
                               "pour_direction_adjusted_from_handle": geometry["pour_dir_adjusted_from_handle"],
                               "clearance_m": geometry["clearance_m"], "step_degrees": float(step_degrees),
                               "ignored_inputs": geometry.get("ignored_inputs", {}),
                               "center_reference": geometry.get("center_reference")},
                           verification="target_reached means the tilt reached max_degrees and the container is "
                                        "upright again; whether the contents landed in the receiver must be "
                                        "verified from fresh images of the receiver; contact_suspected means the "
                                        "tilt lagged or deviated, as when the container touches the receiver, so "
                                        "check that the receiver did not move")
    st["end_step"] = int(tk._operation.metrics["step_count"])
    if continue_required:
        result["stop_reason"] = {"align": "alignment_incomplete"}.get(stage, "continue_pour")
    if not done and handlers._pour_state is st:
        if motion.reason == "deviation" and stage != "align":
            hint = ("with return_upright=true: the motion was blocked while the container is tilted, "
                    "so bring it upright before anything else")
        else:
            hint = {"align": "to finish aligning the rim over the target", "tilting": "to continue the tilt",
                    "tilted": "to dwell and return upright"}.get(stage, "to finish returning upright")
        result["next"] = "observe, resolve verification, then call pour_into again with the same target " + hint
    return result


PLACE_CLEARANCE_M = 0.10  # container centre height above the destination while it travels
PLACE_RETREAT_M = 0.10
PLACE_BACK_OFF_M = 0.15  # then toward the robot: object checks such as RoboCasa's gripper_obj_far use 0.25 m


def place_container(handlers, expected_step: int, target: list[float], *, clearance_m: float = PLACE_CLEARANCE_M,
                    release: bool = True, max_steps: int = POUR_CEILING) -> dict[str, Any]:
    """Set held handle cookware down with the model centre over a destination point (1.2.1).

    A policy may move a container while lifting it: in the 1.2.0 D10 runs the pan
    was 1.4 cm from its burner when the gripper closed and 9.5 cm away when control
    was handed over, and putting the hand back to the handover pose left it 8.4 cm
    off. The LLM grounds the destination afresh with back_project; this tool puts
    the handle model's centre over it: lift to target z + clearance_m, translate,
    lower until the support stops the hand, then (release) open, retreat 10 cm up
    and back off 15 cm toward the robot.
    """
    from robots.robocasa import privileged_tools as consts

    tk = handlers.tk
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 30 <= max_steps <= POUR_CEILING:
        raise ValueError(f"max_steps must be an integer between 30 and {POUR_CEILING}")
    if not isinstance(release, bool):
        raise ValueError("release must be a boolean")
    if (isinstance(clearance_m, bool) or not isinstance(clearance_m, (int, float))
            or not math.isfinite(clearance_m) or not 0.03 <= clearance_m <= 0.30):
        raise ValueError("clearance_m must be a number between 0.03 and 0.3")
    target = _finite_vector(target, size=3, label="target")
    motion = consts._Motion(tk, "place_container", expected_step, max_steps, POUR_CEILING)
    try:
        command, source = tk._resolve_gripper_command("keep")
    except ValueError:
        command, source = None, None
    if command != "close" or not motion.holding:
        raise ValueError("place_container needs a held container: the gripper must be closed on an object "
                         "(measured opening between 2 mm and 6 cm)")
    pending = getattr(handlers, "_pour_state", None)
    if pending is not None and _tilt_about(motion.before[10:14], pending["q_start"], pending["axis"]) >= 5.0:
        raise ValueError("a pour is in progress: call pour_into with its target (return_upright=true to stop "
                         "tilting) until the container is upright, then place it")
    start = motion.before
    q0 = start[10:14].copy()
    center0, center_reference = held_model_center(handlers, start, "place_container")
    center_e = _quat_to_mat(q0).T @ (center0 - start[7:10])
    tracker = _Tracker(motion, consts)
    phases: list[dict[str, Any]] = []
    moving = {"reached", "settled_under_load"}

    def center_now() -> np.ndarray:
        s = motion.state
        return s[7:10] + _quat_to_mat(s[10:14]) @ center_e

    def go(name: str, delta: np.ndarray, gripper: str = "close") -> str:
        """Move the hand by delta in <=SEGMENT_M segments at the start orientation."""
        goal = motion.state[7:10] + delta
        while True:
            s = motion.state
            remaining = goal - s[7:10]
            distance = float(np.linalg.norm(remaining))
            if motion.available <= 1:
                return "budget"
            point = goal if distance <= SEGMENT_M else s[7:10] + remaining / distance * SEGMENT_M
            before = int(motion.result["executed_actions"])
            outcome = tracker.track(point, q0, max_actions=motion.available - 1, gripper=gripper)
            phases.append({"phase": name, "outcome": outcome, "steps": int(motion.result["executed_actions"]) - before,
                           "position_error_m": float(np.linalg.norm(point - motion.state[7:10]))})
            if outcome not in moving or distance <= SEGMENT_M:
                return outcome

    outcome, support, settle, placed = "reached", False, None, None
    lift = float(target[2] + clearance_m - center_now()[2])
    if lift > _POSITION_TOLERANCE_M:
        outcome = go("lift", np.array([0.0, 0.0, lift]))
    if outcome in moving and motion.reason is None:
        c = center_now()
        outcome = go("translate", np.array([target[0] - c[0], target[1] - c[1], 0.0]))
    if outcome in moving and motion.reason is None:
        # The support stops the hand before the model centre reaches the surface point.
        outcome = go("lower", np.array([0.0, 0.0, float(target[2] - center_now()[2])]))
        support = outcome in {"settled_under_load", "deviation"}
        placed = center_now()
    lowered = placed is not None and motion.reason is None and (support or outcome == "reached")
    if release and lowered:
        settle = handlers._settle_gripper(motion, "open")
        if motion.reason is None:
            go("retreat", np.array([0.0, 0.0, PLACE_RETREAT_M]), gripper="open")
        toward_robot = -_horizontal(motion.state[7:10])
        if motion.reason is None and np.linalg.norm(toward_robot) > 1e-6:
            go("back_off", _unit(toward_robot) * PLACE_BACK_OFF_M, gripper="open")
    released = bool(settle and settle.get("release_confirmed"))  # task success may end the retreat
    if released:
        handlers._pour_state = handlers._held_container = None
    result = motion.finish("open" if settle else "close", body=False,
                           reached=released if release else lowered,
                           gripper_source=source, phases=phases, support_contact=support, released=released,
                           **({"gripper_settle": settle} if settle else {}),
                           geometry={
                               "target": target.tolist(), "clearance_m": float(clearance_m),
                               "model_center_start": center0.tolist(),
                               "model_center_placed": None if placed is None else placed.tolist(),
                               "center_xy_error_m": None if placed is None else float(np.linalg.norm(placed[:2] - target[:2])),
                               "center_above_target_m": None if placed is None else float(placed[2] - target[2]),
                               "center_source": "model", "center_reference": center_reference},
                           verification="centres are the handle model's estimate from the hand pose; verify from "
                                        "fresh images that the container rests on the destination and that the "
                                        "gripper is clear of it")
    if not lowered:
        result["next"] = ("observe and re-ground the destination; the container was not lowered onto it "
                          "(budget or an obstruction), so call place_container again with a fresh target")
    return result
