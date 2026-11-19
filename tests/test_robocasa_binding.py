from __future__ import annotations

import numpy as np
import pytest

from robots.robocasa.binding import make_robocasa_binding
from rpent.episode.binding import FeatureValidationError


def _native() -> dict[str, np.ndarray]:
    return {
        "state": np.zeros(16, dtype=np.float32),
        "robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_agentview_center": np.zeros((512, 768, 3), dtype=np.uint8),
    }


def test_robocasa_binding_matches_pinned_checkpoint_camera_order() -> None:
    binding = make_robocasa_binding()

    assert [(feature.native_key, feature.policy_key) for feature in binding.inputs] == [
        ("state", "observation.state"),
        ("robot0_agentview_left", "observation.images.camera1"),
        ("robot0_agentview_right", "observation.images.camera2"),
        ("robot0_eye_in_hand", "observation.images.camera3"),
    ]
    assert binding.inputs[0].shape == (16,)
    assert binding.inputs[0].dtype == np.dtype("float32")
    assert all(feature.shape == (256, 256, 3) for feature in binding.inputs[1:])
    assert all(feature.dtype == np.dtype("uint8") for feature in binding.inputs[1:])
    assert binding.action_shape == (12,)
    assert binding.action_dtype == np.dtype("float32")
    assert (binding.action_low, binding.action_high) == (-1.0, 1.0)
    assert binding.clip_actions_to_bounds is True
    assert binding.allowed_missing_policy_inputs == frozenset()




def test_binding_ignores_display_only_overview_camera() -> None:
    mapped = make_robocasa_binding().bind_observation(_native())

    assert tuple(mapped) == (
        "observation.state",
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("state", np.zeros(6, dtype=np.float32)),
        ("state", np.zeros(8, dtype=np.float32)),
        ("state", np.full(16, np.nan, dtype=np.float32)),
        ("robot0_agentview_left", np.zeros((3, 256, 256), dtype=np.uint8)),
        ("robot0_agentview_right", np.zeros((256, 256, 3), dtype=np.float32)),
    ],
)
def test_binding_rejects_wrong_native_shapes_dtypes_and_values(key, value) -> None:
    observation = _native()
    observation[key] = value
    with pytest.raises(FeatureValidationError):
        make_robocasa_binding().bind_observation(observation)


def test_binding_rejects_missing_camera() -> None:
    observation = _native()
    del observation["robot0_eye_in_hand"]
    with pytest.raises(FeatureValidationError, match="robot0_eye_in_hand.*missing"):
        make_robocasa_binding().bind_observation(observation)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 7), dtype=np.float32),
        np.zeros((2, 12), dtype=np.float64),
        np.full((2, 12), np.nan, dtype=np.float32),
        np.zeros((12,), dtype=np.float32),
    ],
)
def test_binding_rejects_invalid_action_chunks(actions) -> None:
    with pytest.raises(FeatureValidationError):
        make_robocasa_binding().validate_action_chunk(actions)


def test_binding_saturates_actions_at_upstream_controller_bounds() -> None:
    actions = np.zeros((2, 12), dtype=np.float32)
    actions[0, 0] = -1.0334665
    actions[1, 11] = 1.0604779

    adapted = make_robocasa_binding().validate_action_chunk(actions)

    assert float(adapted.min()) == -1.0
    assert float(adapted.max()) == 1.0
