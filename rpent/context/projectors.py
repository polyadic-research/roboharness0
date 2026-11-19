"""Built-in context projectors for tool results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any

from rpent.context.policy import ContextPolicy, ContextProjection, ProjectedImage
from rpent.planning.observation import observation_from_tool_result
from rpent.recording.media import camera_storage_component

_IMAGE_FIELDS = (
    ("_image_bytes", "agent"),
    ("_image_cam_bytes", "camera"),
    ("_image_right_bytes", "camera"),
    ("_image_wrist_bytes", "wrist"),
)
# ``_image_derivation`` marks derived images (zoom crops) for archives only.
_HIDDEN_IMAGE_FIELDS = ("_image_center_bytes", "_image_camera_ids", "_image_derivation")

_ARTIFACT_FIELDS = (
    "image_path",
    "image_cam_path",
    "image_wrist_path",
    "image_cam_hi_path",
    "image_wrist_hi_path",
    "world_map",
    "wrist_world_map",
    "world_map_hi",
    "wrist_world_map_hi",
)


class RawToolContextV1:
    """Compatibility projector matching the original ToolResult behavior."""

    id = "raw-tool-result-v1"

    def project_tool_result(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
        max_text_chars: int,
    ) -> ContextProjection:
        del tool_name
        text_result, images = _extract_images(result)
        return ContextProjection(
            text=_render_json(text_result, max_text_chars=max_text_chars),
            images=images,
        )


class EpisodeCompactContextV1:
    """Compact stateful episode results while preserving inspection outputs.

    Results without a normal robot observation fall back to the raw projector,
    so geometry, segmentation, file, and Harness control tools keep their full
    task-specific payload.
    """

    id = "episode-compact-v1"

    def __init__(self) -> None:
        self._fallback = RawToolContextV1()

    def project_tool_result(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
        max_text_chars: int,
    ) -> ContextProjection:
        observation = observation_from_tool_result(result)
        if observation is None:
            return self._fallback.project_tool_result(
                tool_name=tool_name,
                result=result,
                max_text_chars=max_text_chars,
            )

        text_result, images = _extract_images(result)
        log = text_result.get("log")
        action_result: Any = None
        if isinstance(log, dict):
            action_result = log.get("result")
        observation_payload = asdict(observation)
        # Keep command/result and named artifact paths in one place only.  Do
        # not expose ``obj_of_interest`` as a holding oracle to the model.
        for duplicate_or_private in (
            "image_paths",
            "holding_object",
            "last_action",
            "last_action_result",
            "environment_success",
            "environment_terminated",
            "environment_truncated",
        ):
            observation_payload.pop(duplicate_or_private, None)
        payload = {
            "tool": tool_name,
            "status": "error" if text_result.get("error") else "completed",
            "error": text_result.get("error"),
            "observation": observation_payload,
            "environment": {
                "success": observation.environment_success,
                "terminated": observation.environment_terminated,
                "truncated": observation.environment_truncated,
                "base_task_success": bool(
                    text_result.get("base_task_success", False)
                ),
                "termination_ignored": bool(
                    text_result.get(
                        "environment_termination_ignored",
                        False,
                    )
                ),
            },
            "action_result": action_result,
            "action_history": text_result.get("action_history"),
            "observation_refs": text_result.get("observation_refs"),
            "artifacts": {
                key: text_result[key]
                for key in _ARTIFACT_FIELDS
                if text_result.get(key) is not None
            },
            "harness": text_result.get("harness"),
            "episode_budget": text_result.get("episode_budget"),
        }
        if "libero_terminated" in text_result:
            # Compatibility for the unchanged LIBERO prompt. Keep the alias
            # derived from normalized evidence so a contradictory legacy field
            # cannot override the generic environment envelope.
            payload["libero_terminated"] = observation.environment_success
        payload = {key: value for key, value in payload.items() if value is not None}
        return ContextProjection(
            text=_render_json(payload, max_text_chars=max_text_chars),
            images=images,
        )


class LiberoCompactContextV1(EpisodeCompactContextV1):
    """Legacy LIBERO identity backed by the generic compact projector."""

    id = "libero-compact-v1"


def _extract_images(
    result: dict[str, Any],
) -> tuple[dict[str, Any], tuple[ProjectedImage, ...]]:
    text_result = dict(result)
    camera_ids = text_result.get("_image_camera_ids")
    cameras_by_field = {
        field: camera_id
        for camera_id, field in camera_ids.items()
        if isinstance(camera_id, str) and camera_id and isinstance(field, str)
    } if isinstance(camera_ids, Mapping) else {}
    hidden_camera_files = {
        f"{camera_storage_component(camera_id)}.png"
        for field, camera_id in cameras_by_field.items()
        if field in _HIDDEN_IMAGE_FIELDS
    }
    refs = text_result.get("observation_refs")
    if hidden_camera_files and isinstance(refs, list):
        text_result["observation_refs"] = [
            ref for ref in refs
            if not isinstance(ref, str)
            or PurePosixPath(ref).name not in hidden_camera_files
        ]
    images: list[ProjectedImage] = []
    for field, role in _IMAGE_FIELDS:
        value = text_result.pop(field, None)
        if isinstance(value, bytes) and value:
            images.append(ProjectedImage(
                data=value, role=role, camera_id=cameras_by_field.get(field),
            ))
    for field in _HIDDEN_IMAGE_FIELDS:
        text_result.pop(field, None)
    return text_result, tuple(images)


def _render_json(value: Any, *, max_text_chars: int) -> str:
    if isinstance(value, dict) and "action_history" in value:
        # Numeric command history is bounded per physical call/page upstream.
        # Keep every row and valid JSON instead of silently cutting a sequence.
        return json.dumps(value, separators=(",", ":"), default=str)
    text = json.dumps(value, indent=2, default=str)
    if len(text) > max_text_chars:
        return text[:max_text_chars] + "\n[truncated]"
    return text


ContextPolicyFactory = Callable[[], ContextPolicy]

_CONTEXT_POLICIES: dict[str, ContextPolicyFactory] = {
    RawToolContextV1.id: RawToolContextV1,
    EpisodeCompactContextV1.id: EpisodeCompactContextV1,
    LiberoCompactContextV1.id: LiberoCompactContextV1,
}


def list_context_policy_ids() -> tuple[str, ...]:
    """Return built-in projector ids in stable order."""
    return tuple(_CONTEXT_POLICIES)


def resolve_context_policy(policy_id: str) -> ContextPolicy:
    """Build one stateless context projector by stable id."""
    try:
        factory = _CONTEXT_POLICIES[policy_id]
    except KeyError as exc:
        choices = ", ".join(_CONTEXT_POLICIES)
        raise ValueError(
            f"unknown context policy {policy_id!r}; choose one of: {choices}"
        ) from exc
    return factory()
