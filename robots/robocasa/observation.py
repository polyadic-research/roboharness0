"""Exact PandaOmron observation assembly for the RoboCasa365 runtime contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

OBSERVATION_SCHEMA = "robocasa-pandaomron-observation-v2"
OVERVIEW_CAMERA_KEY = "robot0_agentview_center"
CAMERA_SHAPES = {
    OVERVIEW_CAMERA_KEY: (512, 768, 3),
    "robot0_agentview_left": (256, 256, 3),
    "robot0_agentview_right": (256, 256, 3),
    "robot0_eye_in_hand": (256, 256, 3),
}
CAMERA_KEYS = tuple(CAMERA_SHAPES)

# The released checkpoint consumes this exact 16D order.  Keep the named
# slices next to the assembler so audits do not need to infer semantics from
# concatenation code.
STATE_SLICES = {
    "base_position": slice(0, 3),
    "base_quaternion": slice(3, 7),
    "relative_end_effector_position": slice(7, 10),
    "relative_end_effector_quaternion": slice(10, 14),
    "gripper_qpos": slice(14, 16),
}

_STATE_COMPONENTS = (
    ("state.base_position", (3,), "base_position"),
    ("state.base_rotation", (4,), "base_quaternion"),
    (
        "state.end_effector_position_relative",
        (3,),
        "relative_end_effector_position",
    ),
    (
        "state.end_effector_rotation_relative",
        (4,),
        "relative_end_effector_quaternion",
    ),
    ("state.gripper_qpos", (2,), "gripper_qpos"),
)
_UPSTREAM_CAMERA_KEYS = {
    OVERVIEW_CAMERA_KEY: f"video.{OVERVIEW_CAMERA_KEY}",
    "robot0_agentview_left": "video.robot0_agentview_left",
    "robot0_agentview_right": "video.robot0_agentview_right",
    "robot0_eye_in_hand": "video.robot0_eye_in_hand",
}


def _state_component(
    observation: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    if key not in observation:
        raise ValueError(f"RoboCasa observation is missing {key}")
    value = np.asarray(observation[key])
    if value.shape != shape:
        raise ValueError(f"RoboCasa {label} has shape {value.shape}, expected {shape}")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"RoboCasa {label} must contain finite numeric values")
    return np.array(value, dtype=np.float32, copy=True)


def assemble_robocasa_observation(
    upstream: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Copy one upright upstream Gym observation into RPent-native keys."""

    components = [
        _state_component(upstream, key, shape, label)
        for key, shape, label in _STATE_COMPONENTS
    ]
    state = np.concatenate(components).astype(np.float32, copy=False)
    if state.shape != (16,):
        raise ValueError(f"RoboCasa state has shape {state.shape}, expected (16,)")

    result: dict[str, np.ndarray] = {"state": np.array(state, copy=True)}
    for native_key in CAMERA_KEYS:
        expected_shape = CAMERA_SHAPES[native_key]
        upstream_key = _UPSTREAM_CAMERA_KEYS[native_key]
        if upstream_key not in upstream:
            raise ValueError(f"RoboCasa observation is missing {upstream_key}")
        image = np.asarray(upstream[upstream_key])
        if image.shape != expected_shape:
            raise ValueError(
                f"RoboCasa {native_key} has shape {image.shape}, "
                f"expected {expected_shape} HWC"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"RoboCasa {native_key} must use uint8 pixels")
        # RoboCasaGymEnv already flips MuJoCo's raw framebuffer into upright
        # HWC orientation.  Preserve that orientation and take ownership.
        result[native_key] = np.array(image, dtype=np.uint8, copy=True)
    return result


def validate_robocasa_observation(observation: Mapping[str, Any]) -> None:
    """Fail closed unless a native observation exactly matches the contract."""

    expected_keys = {"state", *CAMERA_KEYS}
    if set(observation) != expected_keys:
        raise ValueError(
            "RoboCasa observation keys mismatch: "
            f"expected {sorted(expected_keys)}, got {sorted(observation)}"
        )
    state = np.asarray(observation["state"])
    if state.shape != (16,) or state.dtype != np.float32:
        raise ValueError("RoboCasa state must have shape (16,) and dtype float32")
    if not np.isfinite(state).all():
        raise ValueError("RoboCasa state must contain finite values")
    for key in CAMERA_KEYS:
        expected_shape = CAMERA_SHAPES[key]
        image = np.asarray(observation[key])
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise ValueError(
                f"RoboCasa {key} must have shape {expected_shape} HWC and dtype uint8"
            )


def copy_robocasa_observation(
    observation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate and return an owned copy of a native observation."""

    validate_robocasa_observation(observation)
    return {key: np.array(value, copy=True) for key, value in observation.items()}


__all__ = [
    "CAMERA_KEYS",
    "CAMERA_SHAPES",
    "OBSERVATION_SCHEMA",
    "OVERVIEW_CAMERA_KEY",
    "STATE_SLICES",
    "assemble_robocasa_observation",
    "copy_robocasa_observation",
    "validate_robocasa_observation",
]
