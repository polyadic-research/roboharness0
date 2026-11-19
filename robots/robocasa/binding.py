"""Exact RoboCasa365 simulator-to-SmolVLA feature binding."""

from __future__ import annotations

import numpy as np

from rpent.episode.binding import FeatureBinding, PolicyBinding

CHECKPOINT_CHUNK_SIZE = 50
CHECKPOINT_N_ACTION_STEPS = 50
DEFAULT_ACTIONS_PER_POLICY_REQUEST = 25


def make_robocasa_binding() -> PolicyBinding:
    """Return the immutable contract qualified against the pinned checkpoint."""
    image = (256, 256, 3)
    return PolicyBinding(
        name="robocasa-smolvla-v1",
        inputs=(
            FeatureBinding("state", "observation.state", (16,), np.float32),
            FeatureBinding(
                "robot0_agentview_left",
                "observation.images.camera1",
                image,
                np.uint8,
            ),
            FeatureBinding(
                "robot0_agentview_right",
                "observation.images.camera2",
                image,
                np.uint8,
            ),
            FeatureBinding(
                "robot0_eye_in_hand",
                "observation.images.camera3",
                image,
                np.uint8,
            ),
        ),
        action_shape=(12,),
        action_dtype=np.float32,
        action_low=-1.0,
        action_high=1.0,
        allowed_missing_policy_inputs=frozenset(),
        clip_actions_to_bounds=True,
    )


__all__ = [
    "CHECKPOINT_CHUNK_SIZE",
    "CHECKPOINT_N_ACTION_STEPS",
    "DEFAULT_ACTIONS_PER_POLICY_REQUEST",
    "make_robocasa_binding",
]
