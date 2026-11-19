"""Retain the initial view before planning, independently of robot transitions."""

import hashlib
import logging
from pathlib import Path

from rpent.recording.media import atomic_write_bytes, camera_storage_component

logger = logging.getLogger(__name__)


def capture_initial(recorder, display, observation):
    if recorder.profile.mode == "none":
        return
    selected = set(recorder.profile.camera_ids)
    for camera in display.descriptors:
        if recorder.profile.mode == "selected" and camera.camera_id not in selected:
            continue
        try:
            data = display.frame(camera.camera_id, observation)
            relative = Path(
                "narration_initial", camera_storage_component(camera.camera_id) + ".png"
            )
            atomic_write_bytes(recorder.root / relative, data)
            recorder.record(
                "narration_initial_frame",
                camera_id=camera.camera_id,
                relative_path=relative.as_posix(),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        except Exception as exc:
            logger.warning(
                "Initial narration frame unavailable (%s): %s",
                camera.camera_id,
                type(exc).__name__,
            )
