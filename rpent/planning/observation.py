"""Normalize environment/tool results into planner-facing observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rpent.planning.schemas import ObservationSummary

_IMAGE_PATH_KEYS = (
    "image_path",
    "image_cam_path",
    "image_wrist_path",
    "image_cam_hi_path",
    "image_wrist_hi_path",
)


def observation_from_tool_result(
    result: Mapping[str, Any],
) -> ObservationSummary | None:
    """Build a compact observation when ``result`` contains a robot state.

    The adapter intentionally uses only normal tool output.  It never loads
    simulator internals or derives privileged object coordinates.
    """
    step_value = result.get("step", result.get("step_idx"))
    state = result.get("state")
    if step_value is None or not isinstance(state, Mapping):
        return None
    try:
        step = int(step_value)
    except (TypeError, ValueError):
        return None
    if step < 0:
        return None

    eef_position = _float_tuple(state.get("robot0_eef_pos"), length=3)
    proprioception = _float_values(state.get("proprioception"))
    gripper = _gripper_opening(state.get("robot0_gripper_qpos"))
    visible_objects = tuple(
        {"name": str(name)} for name in state.get("object_names", ()) if name
    )
    image_paths = tuple(
        str(result[key]) for key in _IMAGE_PATH_KEYS if result.get(key)
    )
    log = result.get("log")
    last_action = None
    last_action_result = None
    if isinstance(log, Mapping):
        if isinstance(log.get("command"), Mapping):
            last_action = dict(log["command"])
        if isinstance(log.get("result"), Mapping):
            last_action_result = dict(log["result"])
    environment_success, environment_terminated, environment_truncated = (
        environment_status_from_tool_result(result)
    )

    return ObservationSummary(
        step=step,
        image_paths=image_paths,
        task_instruction=_optional_string(result.get("task_language")),
        eef_position=eef_position,
        proprioception=proprioception,
        gripper_opening=gripper,
        holding_object=_optional_string(state.get("obj_of_interest")),
        visible_objects=visible_objects,
        last_action=last_action,
        last_action_result=last_action_result,
        environment_success=environment_success,
        environment_terminated=environment_terminated,
        environment_truncated=environment_truncated,
    )


def environment_status_from_tool_result(
    result: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Normalize success, termination, and truncation at the tool boundary."""
    environment = result.get("environment")
    if isinstance(environment, Mapping):
        return (
            _strict_boolean(environment.get("success")),
            _strict_boolean(environment.get("terminated")),
            _strict_boolean(environment.get("truncated")),
        )
    # Compatibility for state artifacts produced before the generic envelope.
    # The old LIBERO success flag did not distinguish generic termination from
    # time-limit truncation, but its companion artifact fields did.
    success = _strict_boolean(result.get("libero_terminated"))
    truncated = _strict_boolean(result.get("episode_truncated"))
    terminated = (
        _strict_boolean(result.get("episode_done"))
        and not success
        and not truncated
    )
    return success, terminated, truncated


def _strict_boolean(value: Any) -> bool:
    """Accept JSON booleans only so truthy malformed values fail closed."""
    return value if isinstance(value, bool) else False


def _float_tuple(value: Any, *, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _float_values(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if any(isinstance(item, bool) for item in value):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(item) for item in values) else None


def _gripper_opening(value: Any) -> float | None:
    values = _float_tuple(value, length=2)
    if values is None:
        return None
    # LIBERO stores mirrored finger joints (+q, -q).  Their separation is the
    # physically meaningful opening and remains non-negative.
    return abs(values[0] - values[1])


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
