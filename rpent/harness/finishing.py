"""Read-only public motion features for bounded finishing; no success authority."""

from __future__ import annotations

import math

from rpent.harness.lifecycle import (
    FINISHING_PROGRESS_M,
    FINISHING_TOOLS,
    FINISHING_WIDTH_DRIFT_M,
)


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in value
    ):
        return None
    return value


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def finishing_motion(feedback):
    """Require measured endpoint progress, fixed base and explicit release feedback.

    Public width/retention feedback is a budget prerequisite, not object truth.
    """
    raw = feedback.raw_result
    if feedback.tool_name not in FINISHING_TOOLS:
        return {}
    before, after = raw.get("measured_before"), raw.get("measured_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    positions = [_vector(v.get("eef_position"), 3) for v in [before, after]]
    bases = [_vector(v.get("base_position"), 3) for v in [before, after]]
    quats = [_vector(v.get("base_quaternion"), 4) for v in [before, after]]
    widths = [v.get("gripper_opening") for v in [before, after]]
    if any(v is None for v in positions + bases + quats):
        return {}
    if not all(
        isinstance(w, (int, float))
        and not isinstance(w, bool)
        and math.isfinite(w)
        and w >= 0
        for w in widths
    ):
        return {}
    norms = [math.sqrt(sum(x * x for x in q)) for q in quats]
    if min(norms) <= 1e-8:
        return {}
    qdot = abs(sum(x * y for x, y in zip(*quats))) / (norms[0] * norms[1])
    fixed = _distance(*bases) <= 0.005 and 2 * math.acos(min(1, qdot)) <= 0.02
    waypoints = raw.get("waypoint_results") or []
    target = (
        _vector(waypoints[-1].get("target_position"), 3)
        if isinstance(waypoints, list) and waypoints and isinstance(waypoints[-1], dict)
        else None
    )
    moving = bool(
        target
        and fixed
        and _distance(positions[0], target) - _distance(positions[1], target)
        >= FINISHING_PROGRESS_M
    )
    settle = raw.get("gripper_settle") or {}
    release = bool(
        isinstance(settle, dict)
        and settle.get("command") == "open"
        and settle.get("stable") is True
        and settle.get("release_confirmed") is True
        and raw.get("gripper_command_executed") is True
        and widths[1] >= 0.06
    )
    events = raw.get("grasp_events") or []
    unsafe = bool(
        feedback.error
        or raw.get("review_required")
        or raw.get("interrupt_reason") not in {None, "budget"}
        or any(
            not isinstance(e, dict) or e.get("release_commanded") is not True
            for e in events
        )
    )
    # A controlled opening may produce release-labelled retention events. Width
    # alone never converts an unexpected loss into a deliberate release.
    if events and not release:
        unsafe = True
    environment = raw.get("environment") or {}
    if not isinstance(environment, dict):
        return {}
    unsafe |= any(
        environment.get(k) is True for k in ["success", "terminated", "truncated"]
    )
    return {
        "moving": moving,
        "stable_hold": bool(
            0.002 < min(widths)
            and max(widths) < 0.06
            and abs(widths[1] - widths[0]) <= FINISHING_WIDTH_DRIFT_M
        ),
        "open": widths[1] >= 0.06,
        "released": release,
        "unsafe": unsafe,
    }
