"""RoboCasa365 display camera declaration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rpent.recording.contracts import CameraDescriptor
from rpent.recording.display import camera_frame, ordered_frames


class RoboCasaDisplayAdapter:
    def __init__(
        self,
        *,
        width: int = 256,
        height: int = 256,
        overview_width: int = 768,
        overview_height: int = 512,
    ) -> None:
        self._descriptors = (
            CameraDescriptor(
                "robot0_agentview_center",
                "Overview",
                "primary",
                overview_width,
                overview_height,
                3,
                20.0,
            ),
            CameraDescriptor(
                "robot0_agentview_left",
                "Policy Left",
                None,
                width,
                height,
                3,
                20.0,
            ),
            CameraDescriptor(
                "robot0_agentview_right",
                "Policy Right",
                None,
                width,
                height,
                3,
                20.0,
            ),
            CameraDescriptor(
                "robot0_eye_in_hand",
                "Wrist",
                "wrist",
                width,
                height,
                3,
                20.0,
            ),
        )

    @property
    def descriptors(self) -> tuple[CameraDescriptor, ...]:
        return self._descriptors

    def frames(self, observation: Mapping[str, Any]) -> Mapping[str, bytes]:
        return ordered_frames(self.descriptors, observation)

    def frame(self, camera_id: str, observation: Mapping[str, Any]) -> bytes:
        return camera_frame(self.descriptors, camera_id, observation)


__all__ = ["RoboCasaDisplayAdapter"]
