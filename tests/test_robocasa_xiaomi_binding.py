from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.xiaomi_binding import (
    XIAOMI_CAMERA_KEYS,
    build_xiaomi_messages,
    center_crop,
    make_xiaomi_policy_binding,
    native_state_to_xiaomi14,
    pack_xiaomi_state,
    quat_xyzw_to_axis_angle,
    sample_history,
)


def test_native_state_is_reordered_to_official_ee_first_14d() -> None:
    native = np.array(
        [
            1,
            2,
            3,
            0,
            0,
            0,
            1,
            4,
            5,
            6,
            0,
            0,
            0,
            1,
            0.02,
            -0.02,
        ],
        dtype=np.float32,
    )

    result = native_state_to_xiaomi14(native)

    np.testing.assert_allclose(
        result,
        np.array(
            [4, 5, 6, 0, 0, 0, 0.02, -0.02, 1, 2, 3, 0, 0, 0],
            dtype=np.float32,
        ),
        atol=1e-7,
        rtol=0,
    )
    assert result.dtype == np.float32


def test_xyzw_quaternion_matches_known_axis_angle_and_canonicalizes_sign() -> None:
    half = np.sqrt(0.5)
    expected = np.array([0, 0, np.pi / 2], dtype=np.float32)

    np.testing.assert_allclose(
        quat_xyzw_to_axis_angle([0, 0, half, half]), expected, atol=1e-6
    )
    np.testing.assert_allclose(
        quat_xyzw_to_axis_angle([0, 0, -half, -half]), expected, atol=1e-6
    )


@pytest.mark.parametrize(
    "state", [np.zeros(15, dtype=np.float32), np.full(16, np.nan, dtype=np.float32)]
)
def test_native_state_rejects_invalid_shape_or_non_finite_values(state: np.ndarray) -> None:
    with pytest.raises(ValueError, match="16D|finite"):
        native_state_to_xiaomi14(state)


def test_history_samples_interval_two_and_repeats_earliest() -> None:
    history = [np.array([value], dtype=np.float32) for value in range(7)]

    sampled = sample_history(history, length=4, interval=2)
    initial = sample_history([np.array([9], dtype=np.float32)], length=4, interval=2)

    np.testing.assert_array_equal(sampled[:, 0], [0, 2, 4, 6])
    np.testing.assert_array_equal(initial[:, 0], [9, 9, 9, 9])


def test_history_rejects_empty_and_invalid_sampling_parameters() -> None:
    with pytest.raises(ValueError, match="empty"):
        sample_history([], length=4, interval=2)
    with pytest.raises(ValueError, match="positive"):
        sample_history([np.zeros(1)], length=0, interval=2)


def test_state_history_is_packed_into_leading_14_of_60_dimensions() -> None:
    history = np.arange(56, dtype=np.float32).reshape(4, 14)

    packed = pack_xiaomi_state(history)

    assert packed.shape == (1, 4, 60)
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[0, :, :14], history)
    np.testing.assert_array_equal(packed[0, :, 14:], 0)


def test_state_packing_rejects_wrong_shape_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match=r"\(4, 14\)"):
        pack_xiaomi_state(np.zeros((3, 14), dtype=np.float32))
    invalid = np.zeros((4, 14), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        pack_xiaomi_state(invalid)


def test_center_crop_matches_official_integer_crop_and_bilinear_resize() -> None:
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    expected = Image.fromarray(image).crop((0, 0, 11, 9)).resize(
        (12, 10), Image.Resampling.BILINEAR
    )

    result = center_crop(image, 0.95)

    assert isinstance(result, Image.Image)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_messages_preserve_left_right_wrist_order_and_official_template() -> None:
    assert XIAOMI_CAMERA_KEYS == (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    image_history = {
        key: np.full((4, 8, 8, 3), index, dtype=np.uint8)
        for index, key in enumerate(XIAOMI_CAMERA_KEYS)
    }

    messages = build_xiaomi_messages(image_history, "KettleBoiling")

    user_content = messages[0]["content"]
    videos = [item for item in user_content if item["type"] == "video"]
    assert [int(np.asarray(item["video"][0])[0, 0, 0]) for item in videos] == [
        0,
        1,
        2,
    ]
    assert user_content[-1] == {
        "type": "text",
        "text": "\n\nGenerate robot actions for the task:\nKettleBoiling /no_cot",
    }
    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "<cot></cot>"}],
    }


def test_messages_reject_missing_camera_or_invalid_instruction() -> None:
    history = {
        key: np.zeros((4, 8, 8, 3), dtype=np.uint8) for key in XIAOMI_CAMERA_KEYS
    }
    del history[XIAOMI_CAMERA_KEYS[-1]]
    with pytest.raises(ValueError, match="robot0_eye_in_hand"):
        build_xiaomi_messages(history, "KettleBoiling")
    with pytest.raises(ValueError, match="instruction"):
        build_xiaomi_messages(
            {key: np.zeros((4, 8, 8, 3), dtype=np.uint8) for key in XIAOMI_CAMERA_KEYS},
            "",
        )


def test_xiaomi_policy_binding_preserves_native_keys_and_12d_actions() -> None:
    binding = make_xiaomi_policy_binding()
    observation = {
        "state": np.zeros(16, dtype=np.float32),
        **{
            key: np.zeros((256, 256, 3), dtype=np.uint8)
            for key in XIAOMI_CAMERA_KEYS
        },
    }

    bound = binding.bind_observation(observation)
    actions = binding.validate_action_chunk(np.zeros((16, 12), dtype=np.float32))

    assert bound.keys() == observation.keys()
    assert actions.shape == (16, 12)



def test_no_cot_suffix_is_added_once_at_message_boundary():
    import numpy as np

    from robots.robocasa.xiaomi_binding import XIAOMI_CAMERA_KEYS, build_xiaomi_messages

    instruction = "Slide the dishwasher rack."
    messages = build_xiaomi_messages({
        camera: np.zeros((4, 8, 8, 3), dtype=np.uint8) for camera in XIAOMI_CAMERA_KEYS
    }, instruction)
    text = messages[0]["content"][-1]["text"]
    assert text.count(" /no_cot") == 1
    assert text.endswith(instruction + " /no_cot")
