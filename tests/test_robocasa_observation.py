from __future__ import annotations

import numpy as np
import pytest

from robots.robocasa.observation import (
    CAMERA_KEYS,
    CAMERA_SHAPES,
    OBSERVATION_SCHEMA,
    STATE_SLICES,
    assemble_robocasa_observation,
    validate_robocasa_observation,
)


def _upstream_observation() -> dict:
    left = np.zeros((256, 256, 3), dtype=np.uint8)
    left[0, :, :] = 11
    left[-1, :, :] = 22
    right = np.full((256, 256, 3), 33, dtype=np.uint8)
    wrist = np.full((256, 256, 3), 44, dtype=np.uint8)
    center = np.full((512, 768, 3), 55, dtype=np.uint8)
    return {
        "state.base_position": np.array([0, 1, 2], dtype=np.float64),
        "state.base_rotation": np.array([3, 4, 5, 6], dtype=np.float64),
        "state.end_effector_position_relative": np.array([7, 8, 9], dtype=np.float64),
        "state.end_effector_rotation_relative": np.array(
            [10, 11, 12, 13], dtype=np.float64
        ),
        "state.gripper_qpos": np.array([14, 15], dtype=np.float64),
        "video.robot0_agentview_left": left,
        "video.robot0_agentview_right": right,
        "video.robot0_eye_in_hand": wrist,
        "video.robot0_agentview_center": center,
    }


def test_assembly_locks_state_order_shapes_dtypes_and_camera_orientation() -> None:
    upstream = _upstream_observation()

    observation = assemble_robocasa_observation(upstream)

    assert tuple(observation) == ("state", *CAMERA_KEYS)
    assert observation["state"].shape == (16,)
    assert observation["state"].dtype == np.float32
    np.testing.assert_array_equal(observation["state"], np.arange(16))
    assert STATE_SLICES == {
        "base_position": slice(0, 3),
        "base_quaternion": slice(3, 7),
        "relative_end_effector_position": slice(7, 10),
        "relative_end_effector_quaternion": slice(10, 14),
        "gripper_qpos": slice(14, 16),
    }
    assert OBSERVATION_SCHEMA == "robocasa-pandaomron-observation-v2"
    for key, shape in CAMERA_SHAPES.items():
        assert observation[key].shape == shape
        assert observation[key].dtype == np.uint8
    np.testing.assert_array_equal(
        observation["robot0_agentview_left"][0],
        upstream["video.robot0_agentview_left"][0],
    )
    np.testing.assert_array_equal(
        observation["robot0_agentview_left"][-1],
        upstream["video.robot0_agentview_left"][-1],
    )


def test_assembly_owns_copies_unaffected_by_later_upstream_mutation() -> None:
    upstream = _upstream_observation()
    observation = assemble_robocasa_observation(upstream)

    upstream["state.base_position"][:] = 99
    upstream["video.robot0_agentview_left"][:] = 99
    upstream["video.robot0_agentview_center"][:] = 99

    np.testing.assert_array_equal(observation["state"][:3], [0, 1, 2])
    assert int(observation["robot0_agentview_left"][0, 0, 0]) == 11
    assert not np.shares_memory(
        observation["robot0_agentview_left"],
        upstream["video.robot0_agentview_left"],
    )
    assert int(observation["robot0_agentview_center"][0, 0, 0]) == 55
    assert not np.shares_memory(
        observation["robot0_agentview_center"],
        upstream["video.robot0_agentview_center"],
    )


def test_assembly_ignores_global_depth_without_changing_policy_contract() -> None:
    upstream = _upstream_observation()
    upstream.update(
        {
            "video.robot0_agentview_center_depth": np.ones(
                (512, 768), dtype=np.float32
            ),
            "video.robot0_agentview_left_depth": np.ones(
                (256, 256, 1), dtype=np.float32
            ),
            "video.robot0_agentview_right_depth": np.ones(
                (256, 256, 1), dtype=np.float32
            ),
            "video.robot0_eye_in_hand_depth": np.ones(
                (256, 256, 1), dtype=np.float32
            ),
        }
    )

    observation = assemble_robocasa_observation(upstream)

    assert tuple(observation) == ("state", *CAMERA_KEYS)
    assert not any("depth" in key for key in observation)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("state.base_position", np.zeros(2), "base_position.*shape"),
        (
            "state.end_effector_position_relative",
            np.array([0, np.nan, 0]),
            "finite",
        ),
        (
            "video.robot0_agentview_right",
            np.zeros((3, 256, 256), dtype=np.uint8),
            "robot0_agentview_right.*shape",
        ),
        (
            "video.robot0_eye_in_hand",
            np.zeros((256, 256, 3), dtype=np.float32),
            "robot0_eye_in_hand.*uint8",
        ),
        (
            "video.robot0_agentview_center",
            np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_agentview_center.*shape",
        ),
        (
            "video.robot0_agentview_center",
            np.zeros((512, 768, 3), dtype=np.float32),
            "robot0_agentview_center.*uint8",
        ),
    ],
)
def test_assembly_rejects_schema_drift(
    key: str, value: np.ndarray, message: str
) -> None:
    upstream = _upstream_observation()
    upstream[key] = value

    with pytest.raises(ValueError, match=message):
        assemble_robocasa_observation(upstream)


def test_native_validator_rejects_missing_or_non_owned_schema() -> None:
    observation = assemble_robocasa_observation(_upstream_observation())
    validate_robocasa_observation(observation)

    missing = dict(observation)
    missing.pop("robot0_eye_in_hand")
    with pytest.raises(ValueError, match="keys"):
        validate_robocasa_observation(missing)
