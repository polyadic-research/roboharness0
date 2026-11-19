"""Validated JSON contracts shared by recording, storage, and UI layers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from numbers import Integral, Real
from pathlib import PurePosixPath
from typing import Any, Literal

from rpent.evaluation.results import EpisodeResult

RECORDING_SCHEMA_VERSION = 1
JSON_SAFE_INTEGER_MAX = (1 << 53) - 1
RecordingMode = Literal["none", "primary", "selected", "all"]
ArtifactStatus = Literal["complete", "degraded", "partial", "unavailable"]
GENERIC_ARTIFACT_ISSUE = "recording artifact issue"
PUBLIC_ARTIFACT_ISSUES = frozenset(
    {
        GENERIC_ARTIFACT_ISSUE,
        "Parquet finalization failed",
        "artifact metadata unavailable",
        "artifacts.json is missing",
        "camera extraction failed",
        "camera index finalization failed",
        "camera segment write failed",
        "camera video finalization failed",
        "event journal write failed",
        "ignored incomplete events.jsonl tail during recovery",
        "ignored incomplete health.jsonl tail during recovery",
        "ignored incomplete transitions.jsonl tail at finalize",
        "ignored incomplete transitions.jsonl tail during recovery",
        "live camera cleanup failed",
        "live camera write failed",
        "notification delivery failed",
        "qualification projection unavailable",
        "qualification write failed",
        "recorder poisoned",
        "recorder was unavailable before finalization",
        "recording close failed",
        "recording discovery unavailable",
        "recording finalization failed",
        "retained camera frame unavailable",
        "transition event journal write failed",
        "transition event recovered",
        "transition event repair failed",
        "transition event unavailable",
        "transition journal write failed",
    }
)


def sanitize_artifact_issues(issues: Iterable[str]) -> tuple[str, ...]:
    """Map persisted issue text to stable public categories in source order."""
    sanitized: list[str] = []
    for issue in issues:
        _text(issue, "artifact issue")
        public_issue = (
            issue if issue in PUBLIC_ARTIFACT_ISSUES else GENERIC_ARTIFACT_ISSUE
        )
        if public_issue not in sanitized:
            sanitized.append(public_issue)
    return tuple(sanitized)


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _index(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
        or value > JSON_SAFE_INTEGER_MAX
    ):
        raise ValueError(
            f"{name} must be a non-negative JSON safe integer "
            f"(at most {JSON_SAFE_INTEGER_MAX})"
        )
    return int(value)


def _non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")
    return float(value)


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number):
            return number
        raise ValueError(f"{name} must contain only finite numbers")
    if isinstance(value, list):
        return [_json_value(item, name) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{name} must contain only string object keys")
        return {key: _json_value(item, name) for key, item in value.items()}
    raise ValueError(f"{name} must contain only strict JSON values")


def _from_fields(cls: type[Any], payload: Any, label: str) -> Any:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {label} fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CameraDescriptor:
    camera_id: str
    label: str
    role: str | None
    width: int
    height: int
    channels: int
    fps: float
    media_type: str = "rgb"

    def __post_init__(self) -> None:
        _text(self.camera_id, "camera_id")
        if len(self.camera_id.encode("utf-8")) > 512:
            raise ValueError("camera_id must be at most 512 UTF-8 bytes")
        _text(self.label, "label")
        if self.role is not None:
            _text(self.role, "role")
        for name in ("width", "height", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, Real)
            or not math.isfinite(float(self.fps))
            or self.fps <= 0
        ):
            raise ValueError("fps must be a positive finite number")
        object.__setattr__(self, "fps", float(self.fps))
        _text(self.media_type, "media_type")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> CameraDescriptor:
        return _from_fields(cls, payload, "camera descriptor")


@dataclass(frozen=True, slots=True)
class RecordingProfile:
    mode: RecordingMode
    camera_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"none", "primary", "selected", "all"}:
            raise ValueError(f"unknown recording profile: {self.mode}")
        if not isinstance(self.camera_ids, tuple):
            raise ValueError("camera_ids must be a tuple")
        for camera_id in self.camera_ids:
            _text(camera_id, "camera_id")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("selected camera IDs must be unique")
        if self.mode == "selected" and not self.camera_ids:
            raise ValueError("selected recording requires camera IDs")
        if self.mode != "selected" and self.camera_ids:
            raise ValueError("camera IDs are valid only for selected recording")

    @classmethod
    def validate(
        cls,
        mode: str,
        camera_ids: tuple[str, ...],
        cameras: tuple[CameraDescriptor, ...],
    ) -> RecordingProfile:
        allowed = {"none", "primary", "selected", "all"}
        if mode not in allowed:
            raise ValueError(f"unknown recording profile: {mode}")
        available = tuple(camera.camera_id for camera in cameras)
        if len(set(available)) != len(available):
            raise ValueError("camera IDs must be unique")
        from rpent.recording import media

        storage_keys = tuple(media.camera_storage_component(item) for item in available)
        if len(set(storage_keys)) != len(storage_keys):
            raise ValueError("camera storage keys must be unique")
        primary = tuple(
            camera.camera_id for camera in cameras if camera.role == "primary"
        )
        if cameras and len(primary) != 1:
            raise ValueError("display cameras require exactly one primary")
        selected = tuple(camera_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("selected camera IDs must be unique")
        if not cameras and mode != "none":
            raise ValueError("a zero-camera Simulator supports only none")
        if mode == "selected" and not selected:
            raise ValueError("selected recording requires camera IDs")
        unknown = set(selected).difference(available)
        if unknown:
            raise ValueError(f"unknown camera IDs: {sorted(unknown)}")
        if mode != "selected" and selected:
            raise ValueError("camera IDs are valid only for selected recording")
        return cls(mode=mode, camera_ids=selected)  # type: ignore[arg-type]

    def to_payload(self) -> dict[str, Any]:
        return {"mode": self.mode, "camera_ids": list(self.camera_ids)}

    def retained_camera_ids(
        self, cameras: tuple[CameraDescriptor, ...]
    ) -> tuple[str, ...]:
        """Resolve the durable camera set after validating the display contract."""
        validated = type(self).validate(self.mode, self.camera_ids, cameras)
        if validated.mode == "none":
            return ()
        if validated.mode == "all":
            return tuple(camera.camera_id for camera in cameras)
        if validated.mode == "selected":
            return validated.camera_ids
        return tuple(camera.camera_id for camera in cameras if camera.role == "primary")

    @classmethod
    def from_payload(cls, payload: Any) -> RecordingProfile:
        if not isinstance(payload, dict):
            raise ValueError("recording profile must be a JSON object")
        try:
            camera_ids = payload.get("camera_ids", [])
            if not isinstance(camera_ids, list):
                raise ValueError("camera_ids must be a JSON array")
            return cls(payload["mode"], tuple(camera_ids))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid recording profile fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ArtifactHealth:
    status: ArtifactStatus
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"complete", "degraded", "partial", "unavailable"}:
            raise ValueError(f"invalid artifact health: {self.status}")
        if not isinstance(self.issues, tuple):
            raise ValueError("issues must be a tuple")
        object.__setattr__(self, "issues", sanitize_artifact_issues(self.issues))

    def to_payload(self) -> dict[str, Any]:
        return {"status": self.status, "issues": list(self.issues)}

    @classmethod
    def from_payload(cls, payload: Any) -> ArtifactHealth:
        if not isinstance(payload, dict):
            raise ValueError("artifact health must be a JSON object")
        try:
            issues = payload.get("issues", [])
            if not isinstance(issues, list):
                raise ValueError("issues must be a JSON array")
            return cls(payload["status"], tuple(issues))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid artifact health fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class QualificationOutcome:
    episode_result: EpisodeResult
    artifact_health: ArtifactHealth
    require_complete_artifacts: bool
    accepted: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_result, EpisodeResult):
            raise ValueError("episode_result must be an EpisodeResult")
        if not isinstance(self.artifact_health, ArtifactHealth):
            raise ValueError("artifact_health must be ArtifactHealth")
        if not isinstance(self.require_complete_artifacts, bool):
            raise ValueError("require_complete_artifacts must be a boolean")
        object.__setattr__(
            self,
            "accepted",
            not self.require_complete_artifacts
            or self.artifact_health.status == "complete",
        )

    @classmethod
    def evaluate(
        cls,
        episode_result: EpisodeResult,
        artifact_health: ArtifactHealth,
        require_complete_artifacts: bool,
    ) -> QualificationOutcome:
        return cls(episode_result, artifact_health, require_complete_artifacts)

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_result": self.episode_result.to_payload(),
            "artifact_health": self.artifact_health.to_payload(),
            "require_complete_artifacts": self.require_complete_artifacts,
            "accepted": self.accepted,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> QualificationOutcome:
        if not isinstance(payload, dict):
            raise ValueError("qualification outcome must be a JSON object")
        try:
            outcome = cls.evaluate(
                EpisodeResult.from_payload(payload["episode_result"]),
                ArtifactHealth.from_payload(payload["artifact_health"]),
                payload["require_complete_artifacts"],
            )
            accepted = payload["accepted"]
        except KeyError as exc:
            raise ValueError(
                f"qualification outcome is missing {exc.args[0]!r}"
            ) from exc
        if accepted is not outcome.accepted:
            raise ValueError("qualification accepted value does not match policy")
        return outcome


@dataclass(frozen=True, slots=True)
class FrameReference:
    camera_id: str
    frame_index: int
    step_id: int
    timestamp_s: float
    relative_path: str

    def __post_init__(self) -> None:
        _text(self.camera_id, "camera_id")
        object.__setattr__(self, "frame_index", _index(self.frame_index, "frame_index"))
        object.__setattr__(self, "step_id", _index(self.step_id, "step_id"))
        object.__setattr__(
            self,
            "timestamp_s",
            _non_negative_number(self.timestamp_s, "timestamp_s"),
        )
        _text(self.relative_path, "relative_path")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within the artifact root")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> FrameReference:
        return _from_fields(cls, payload, "frame reference")


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    case_id: str
    attempt_id: str
    episode_id: str
    step_id: int
    frame_index: int
    timestamp_s: float
    state: dict[str, Any]
    action: list[float]
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    operation_id: str | None
    execution_id: str | None
    frames: tuple[FrameReference, ...]

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.attempt_id, "attempt_id")
        _text(self.episode_id, "episode_id")
        object.__setattr__(self, "step_id", _index(self.step_id, "step_id"))
        object.__setattr__(self, "frame_index", _index(self.frame_index, "frame_index"))
        object.__setattr__(
            self,
            "timestamp_s",
            _non_negative_number(self.timestamp_s, "timestamp_s"),
        )
        if not isinstance(self.state, dict):
            raise ValueError("state must be a JSON object")
        object.__setattr__(self, "state", _json_value(self.state, "state"))
        if not isinstance(self.action, list):
            raise ValueError("action must be a list")
        action = _json_value(self.action, "action")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in action
        ):
            raise ValueError("action must contain only numbers")
        object.__setattr__(self, "action", [float(value) for value in action])
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, Real)
            or not math.isfinite(float(self.reward))
        ):
            raise ValueError("reward must be a finite number")
        object.__setattr__(self, "reward", float(self.reward))
        for name in ("success", "terminated", "truncated"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("operation_id", "execution_id"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if not isinstance(self.frames, tuple) or not all(
            isinstance(frame, FrameReference) for frame in self.frames
        ):
            raise ValueError("frames must contain FrameReference values")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [frame.to_payload() for frame in self.frames]
        return payload

    @classmethod
    def from_payload(cls, payload: Any) -> TransitionRecord:
        if not isinstance(payload, dict):
            raise ValueError("transition record must be a JSON object")
        values = dict(payload)
        try:
            values["frames"] = tuple(
                FrameReference.from_payload(item) for item in values["frames"]
            )
        except KeyError as exc:
            raise ValueError("transition record is missing 'frames'") from exc
        return _from_fields(cls, values, "transition record")


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    schema_version: int
    evaluation_id: str
    case_id: str
    attempt_id: str
    sequence: int
    recorded_at: str
    event_type: str
    step_id: int | None
    operation_id: str | None
    execution_id: str | None
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or type(self.schema_version) is not int
        ):
            raise ValueError("schema_version must be a builtin integer")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if self.schema_version != RECORDING_SCHEMA_VERSION:
            raise ValueError("unsupported recording schema_version")
        for name in (
            "evaluation_id",
            "case_id",
            "attempt_id",
            "recorded_at",
            "event_type",
        ):
            _text(getattr(self, name), name)
        try:
            recorded_at = datetime.fromisoformat(
                self.recorded_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                "recorded_at must be a timezone-aware UTC ISO timestamp"
            ) from exc
        if recorded_at.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be a timezone-aware UTC ISO timestamp")
        object.__setattr__(self, "sequence", _index(self.sequence, "sequence"))
        if self.step_id is not None:
            object.__setattr__(self, "step_id", _index(self.step_id, "step_id"))
        for name in ("operation_id", "execution_id"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a JSON object")
        object.__setattr__(self, "payload", _json_value(self.payload, "payload"))

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> EventEnvelope:
        return _from_fields(cls, payload, "event envelope")
