"""Match the pinned OpenPI RoboCasa evaluator's image, state and action order."""

from dataclasses import replace

import numpy as np
from PIL import Image


def make_pi05_binding():
    from robots.robocasa.binding import make_robocasa_binding

    original = make_robocasa_binding()
    return replace(
        original,
        name="robocasa-pi05-v1",
        inputs=tuple(
            replace(feature, policy_key=feature.native_key)
            for feature in original.inputs
        ),
    )


def prepare_pi05_observation(observation, instruction):
    cameras = {"robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"}
    if set(observation) != {"state", *cameras}:
        raise ValueError("Pi0.5 observation keys do not match the binding")
    state = np.asarray(observation["state"])
    if (
        state.shape != (16,)
        or state.dtype != np.float32
        or not np.isfinite(state).all()
    ):
        raise ValueError("Pi0.5 state must be a finite float32 vector of length 16")
    for camera in cameras:
        image = np.asarray(observation[camera])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError(f"Pi0.5 camera {camera} must be 256x256 uint8 RGB")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Pi0.5 requires a natural-language instruction")
    # RPent: base xyz/quat, relative eef xyz/quat, gripper.
    # OpenPI: relative eef xyz/quat, base xyz/quat, gripper.
    result = {
        "observation/state": np.concatenate((state[7:14], state[:7], state[14:16])),
        "prompt": instruction,
    }
    for native, target in (
        ("robot0_agentview_left", "image"),
        ("robot0_agentview_right", "right_image"),
        ("robot0_eye_in_hand", "wrist_image"),
    ):
        # The contract supplies square 256px upright RGB images. This is exactly
        # upstream resize_with_pad's PIL bilinear path; no extra image flip.
        result[f"observation/{target}"] = np.asarray(
            Image.fromarray(observation[native]).resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        )
    return result


def pi05_actions_to_native(actions):
    values = validate_pi05_action_chunk(actions)
    # Upstream convert_action: eef xyz/rotation, gripper, base, mode.
    return values[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]


def validate_pi05_action_chunk(actions):
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12 or not len(values):
        raise ValueError("Pi0.5 must return a nonempty (N, 12) action chunk")
    if not np.isfinite(values).all():
        raise ValueError("Pi0.5 actions must be finite")
    return np.clip(values, -1.0, 1.0)
