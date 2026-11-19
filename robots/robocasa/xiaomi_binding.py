"""Official Xiaomi MiBoT input binding for native RoboCasa observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from rpent.episode.binding import FeatureBinding, PolicyBinding
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_POLICY_SPEC

XIAOMI_CAMERA_KEYS = XIAOMI_MIBOT_POLICY_SPEC.input_camera_ids


def xiaomi_action_chunk_to_rpent(actions: Any) -> np.ndarray:
    """Convert Xiaomi's EE-first actions to RPent's base-first contract."""
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 12:
        raise ValueError(f"expected an action chunk with shape (N, 12), got {value.shape}")
    return np.ascontiguousarray(
        np.concatenate((value[:, 7:11], value[:, 11:12], value[:, 0:7]), axis=1),
        dtype=np.float32,
    )


def make_xiaomi_policy_binding() -> PolicyBinding:
    """Keep native RoboCasa keys intact at the Xiaomi policy boundary."""
    image_shape = (256, 256, 3)
    return PolicyBinding(
        name="robocasa-xiaomi-mibot-v1",
        inputs=(
            FeatureBinding("state", "state", (16,), np.float32),
            *(
                FeatureBinding(key, key, image_shape, np.uint8)
                for key in XIAOMI_CAMERA_KEYS
            ),
        ),
        action_shape=(12,),
        action_dtype=np.float32,
        action_low=-1.0,
        action_high=1.0,
        clip_actions_to_bounds=True,
    )


def quat_xyzw_to_axis_angle(quaternion: Any) -> np.ndarray:
    """Convert an XYZW quaternion using Xiaomi's canonical sign convention."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite XYZW values")
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    value = value / norm
    if value[3] < 0:
        value = -value
    xyz = value[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(value[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def native_state_to_xiaomi14(state: Any) -> np.ndarray:
    """Reorder RPent's native 16D state into Xiaomi's EE-first 14D state."""
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (16,):
        raise ValueError(f"expected a 16D RoboCasa state, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("RoboCasa state must contain only finite values")
    return np.concatenate(
        [
            value[7:10],
            quat_xyzw_to_axis_angle(value[10:14]),
            value[14:16],
            value[0:3],
            quat_xyzw_to_axis_angle(value[3:7]),
        ]
    ).astype(np.float32)


def sample_history(history: Sequence[Any], *, length: int, interval: int) -> np.ndarray:
    """Sample oldest-to-newest frames, repeating the earliest when necessary."""
    items = list(history)
    if not items:
        raise ValueError("observation history cannot be empty")
    if length < 1 or interval < 1:
        raise ValueError("history length and interval must be positive")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(np.stack([items[index] for index in indices], axis=0))


def center_crop(image: Any, crop_ratio: float = XIAOMI_MIBOT_POLICY_SPEC.crop_ratio) -> Image.Image:
    """Apply Xiaomi's integer center crop and resize to the original resolution."""
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1]")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("camera image must be an HWC uint8 RGB array")
    pil_image = Image.fromarray(value)
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def pack_xiaomi_state(history14: Any) -> np.ndarray:
    """Zero-pad four official 14D states into the model's 60D container."""
    value = np.asarray(history14, dtype=np.float32)
    if value.shape != (4, 14):
        raise ValueError(f"state history must have shape (4, 14), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("state history must contain only finite values")
    packed = np.zeros((1, 4, 60), dtype=np.float32)
    packed[0, :, :14] = value
    return packed


def build_xiaomi_messages(
    image_history: Mapping[str, Any],
    instruction: str,
    *,
    crop_ratio: float = XIAOMI_MIBOT_POLICY_SPEC.crop_ratio,
) -> list[dict[str, Any]]:
    """Build the official left/right/wrist message without rewriting the task."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    videos: dict[str, list[Image.Image]] = {}
    for key in XIAOMI_CAMERA_KEYS:
        if key not in image_history:
            raise ValueError(f"required camera {key!r} is missing")
        frames = np.asarray(image_history[key])
        if frames.ndim != 4 or frames.shape[0] != 4:
            raise ValueError(f"camera {key!r} history must have four HWC frames")
        videos[key] = [center_crop(frame, crop_ratio) for frame in frames]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Left camera: "},
                {"type": "video", "video": videos[XIAOMI_CAMERA_KEYS[0]]},
                {"type": "text", "text": "\nRight camera: "},
                {"type": "video", "video": videos[XIAOMI_CAMERA_KEYS[1]]},
                {"type": "text", "text": "\nWrist camera: "},
                {"type": "video", "video": videos[XIAOMI_CAMERA_KEYS[2]]},
                {
                    "type": "text",
                    "text": (
                        "\n\nGenerate robot actions for the task:\n"
                        f"{instruction}{XIAOMI_MIBOT_POLICY_SPEC.instruction_suffix}"
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<cot></cot>"}],
        },
    ]


__all__ = [
    "XIAOMI_CAMERA_KEYS",
    "build_xiaomi_messages",
    "center_crop",
    "make_xiaomi_policy_binding",
    "native_state_to_xiaomi14",
    "pack_xiaomi_state",
    "quat_xyzw_to_axis_angle",
    "sample_history",
    "xiaomi_action_chunk_to_rpent",
]
