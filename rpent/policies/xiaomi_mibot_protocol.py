"""Versioned, Transformers-free RPC contract for Xiaomi MiBoT."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from rpent.policies.xiaomi_mibot_catalog import XiaomiMiBoTCheckpoint

XIAOMI_MIBOT_PROTOCOL_VERSION = 1
XIAOMI_MIBOT_WIRE_OPERATIONS = ("health", "load", "reset", "predict", "close")

_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _version(value: Any) -> None:
    if value != XIAOMI_MIBOT_PROTOCOL_VERSION:
        raise ValueError(
            "protocol_version mismatch: "
            f"expected {XIAOMI_MIBOT_PROTOCOL_VERSION}, actual {value!r}"
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _checkpoint_payload(checkpoint: XiaomiMiBoTCheckpoint) -> dict[str, str]:
    return checkpoint.to_payload()


def _checkpoint_from_payload(payload: Any) -> XiaomiMiBoTCheckpoint:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    try:
        return XiaomiMiBoTCheckpoint(
            repo_id=payload["repo_id"],
            revision=payload["revision"],
            binding_name=payload["binding_name"],
            robot_type=payload["robot_type"],
        )
    except KeyError as error:
        raise ValueError(f"checkpoint missing {error.args[0]}") from error


def _validate_checkpoint(
    actual: XiaomiMiBoTCheckpoint, expected: XiaomiMiBoTCheckpoint
) -> None:
    if actual != expected:
        raise ValueError("checkpoint identity does not match request")


def _validate_observation(observation: Any) -> dict[str, np.ndarray]:
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    required = {"state", *_CAMERAS}
    if set(observation) != required:
        missing = sorted(required - set(observation))
        extra = sorted(set(observation) - required)
        raise ValueError(f"observation keys mismatch: missing={missing}, extra={extra}")
    state = np.asarray(observation["state"])
    if state.shape != (16,) or state.dtype != np.dtype(np.float32):
        raise ValueError("state must have shape (16,) and dtype float32")
    if not np.isfinite(state).all():
        raise ValueError("state must contain finite values")
    normalized = {"state": state}
    camera_shape = None
    for key in _CAMERAS:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be an HWC uint8 RGB camera image")
        if camera_shape is None:
            camera_shape = image.shape
        elif image.shape != camera_shape:
            raise ValueError("all camera images must have the same shape")
        normalized[key] = image
    return normalized


def _validate_history_updates(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("history_updates must be a sequence")
    updates = []
    previous = None
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("history update must be a mapping")
        transition_id = _identifier(item.get("transition_id"), "transition_id")
        if previous is not None and transition_id <= previous:
            raise ValueError("history_updates transition_id must increase")
        terminal = item.get("terminal", False)
        if not isinstance(terminal, bool):
            raise ValueError("history update terminal must be boolean")
        updates.append(
            {
                "transition_id": transition_id,
                "observation": _validate_observation(item.get("observation")),
                "terminal": terminal,
            }
        )
        previous = transition_id
    return tuple(updates)


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTLoadRequest:
    checkpoint: XiaomiMiBoTCheckpoint
    binding_identity: str
    device: str
    dtype: str
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        if not isinstance(self.checkpoint, XiaomiMiBoTCheckpoint):
            raise ValueError("checkpoint must be a XiaomiMiBoTCheckpoint")
        _digest(self.binding_identity, "binding_identity")
        _text(self.device, "device")
        _text(self.dtype, "dtype")

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "checkpoint_identity": self.checkpoint.identity_digest,
            "binding_identity": self.binding_identity,
            "device": self.device,
            "dtype": self.dtype,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTLoadRequest:
        _version(payload.get("protocol_version"))
        request = cls(
            _checkpoint_from_payload(payload.get("checkpoint")),
            payload.get("binding_identity"),
            payload.get("device"),
            payload.get("dtype"),
        )
        if payload.get("checkpoint_identity") != request.checkpoint.identity_digest:
            raise ValueError("checkpoint_identity does not match load request")
        return request


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTHealthResponse:
    checkpoint: XiaomiMiBoTCheckpoint
    checkpoint_identity: str
    binding_identity: str
    artifact_identity: str
    remote_code_revision: str
    device: str
    dtype: str
    server_version: str
    loaded: bool
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _digest(self.checkpoint_identity, "checkpoint_identity")
        _digest(self.binding_identity, "binding_identity")
        _digest(self.artifact_identity, "artifact_identity")
        _text(self.remote_code_revision, "remote_code_revision")
        _text(self.device, "device")
        _text(self.dtype, "dtype")
        _text(self.server_version, "server_version")
        if not isinstance(self.loaded, bool):
            raise ValueError("loaded must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "checkpoint_identity": self.checkpoint_identity,
            "binding_identity": self.binding_identity,
            "artifact_identity": self.artifact_identity,
            "remote_code_revision": self.remote_code_revision,
            "device": self.device,
            "dtype": self.dtype,
            "server_version": self.server_version,
            "loaded": self.loaded,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTHealthResponse:
        _version(payload.get("protocol_version"))
        return cls(
            checkpoint=_checkpoint_from_payload(payload.get("checkpoint")),
            checkpoint_identity=payload.get("checkpoint_identity"),
            binding_identity=payload.get("binding_identity"),
            artifact_identity=payload.get("artifact_identity"),
            remote_code_revision=payload.get("remote_code_revision"),
            device=payload.get("device"),
            dtype=payload.get("dtype"),
            server_version=payload.get("server_version"),
            loaded=payload.get("loaded"),
        )

    def validate_for(self, request: XiaomiMiBoTLoadRequest) -> None:
        _validate_checkpoint(self.checkpoint, request.checkpoint)
        if self.checkpoint_identity != request.checkpoint.identity_digest:
            raise ValueError("checkpoint_identity does not match load request")
        for field in ("binding_identity", "device", "dtype"):
            if getattr(self, field) != getattr(request, field):
                raise ValueError(f"{field} does not match load request")
        if self.remote_code_revision != request.checkpoint.revision:
            raise ValueError("remote_code_revision does not match load request")


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTResetRequest:
    checkpoint: XiaomiMiBoTCheckpoint
    episode_id: str
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION
    seed: int | None = None

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _text(self.episode_id, "episode_id")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be absent or a uint32 integer")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "episode_id": self.episode_id,
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTResetRequest:
        _version(payload.get("protocol_version"))
        return cls(
            _checkpoint_from_payload(payload.get("checkpoint")),
            payload.get("episode_id"),
            seed=payload.get("seed"),
        )


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTResetResponse(XiaomiMiBoTResetRequest):
    rng_state: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = XiaomiMiBoTResetRequest.to_payload(self)
        if self.rng_state is not None:
            payload["rng_state"] = dict(self.rng_state)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTResetResponse:
        request = XiaomiMiBoTResetRequest.from_payload(payload)
        return cls(request.checkpoint, request.episode_id,
                   seed=request.seed, rng_state=payload.get("rng_state"))

    def validate_for(self, request: XiaomiMiBoTResetRequest) -> None:
        _validate_checkpoint(self.checkpoint, request.checkpoint)
        if self.episode_id != request.episode_id:
            raise ValueError("episode_id does not match reset request")
        if self.seed != request.seed:
            raise ValueError("RNG seed does not match reset request")
        if request.seed is not None:
            from rpent.policies.xiaomi_mibot_rng import validate_rng_evidence

            validate_rng_evidence(self.rng_state, request.seed)


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTPredictRequest:
    checkpoint: XiaomiMiBoTCheckpoint
    episode_id: str
    request_id: int
    transition_id: int
    instruction: str
    observation: Mapping[str, np.ndarray]
    terminal: bool = False
    history_updates: tuple[Mapping[str, Any], ...] = ()
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _text(self.episode_id, "episode_id")
        _text(self.instruction, "instruction")
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        object.__setattr__(
            self, "transition_id", _identifier(self.transition_id, "transition_id")
        )
        if not isinstance(self.terminal, bool):
            raise ValueError("terminal must be boolean")
        object.__setattr__(self, "observation", _validate_observation(self.observation))
        object.__setattr__(
            self, "history_updates", _validate_history_updates(self.history_updates)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "transition_id": self.transition_id,
            "instruction": self.instruction,
            "observation": dict(self.observation),
            "terminal": self.terminal,
            "history_updates": [dict(item) for item in self.history_updates],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTPredictRequest:
        _version(payload.get("protocol_version"))
        return cls(
            checkpoint=_checkpoint_from_payload(payload.get("checkpoint")),
            episode_id=payload.get("episode_id"),
            request_id=payload.get("request_id"),
            transition_id=payload.get("transition_id"),
            instruction=payload.get("instruction"),
            observation=payload.get("observation"),
            terminal=payload.get("terminal", False),
            history_updates=payload.get("history_updates", ()),
        )


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTPredictResponse:
    checkpoint: XiaomiMiBoTCheckpoint
    episode_id: str
    request_id: int
    actions: np.ndarray
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _text(self.episode_id, "episode_id")
        object.__setattr__(self, "request_id", _identifier(self.request_id, "request_id"))
        actions = np.asarray(self.actions)
        if actions.shape != (16, 12) or actions.dtype != np.dtype(np.float32):
            raise ValueError("actions must have shape (16, 12) and dtype float32")
        if not np.isfinite(actions).all():
            raise ValueError("actions must contain finite values")
        object.__setattr__(self, "actions", actions)

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "actions": self.actions,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTPredictResponse:
        _version(payload.get("protocol_version"))
        return cls(
            _checkpoint_from_payload(payload.get("checkpoint")),
            payload.get("episode_id"),
            payload.get("request_id"),
            payload.get("actions"),
        )

    def validate_for(self, request: XiaomiMiBoTPredictRequest) -> None:
        _validate_checkpoint(self.checkpoint, request.checkpoint)
        if self.episode_id != request.episode_id:
            raise ValueError("episode_id does not match predict request")
        if self.request_id != request.request_id:
            raise ValueError("request_id does not match predict request")


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTCloseRequest:
    checkpoint: XiaomiMiBoTCheckpoint
    episode_id: str | None
    protocol_version: int = XIAOMI_MIBOT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        if self.episode_id is not None:
            _text(self.episode_id, "episode_id")

    def to_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "checkpoint": _checkpoint_payload(self.checkpoint),
            "episode_id": self.episode_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> XiaomiMiBoTCloseRequest:
        _version(payload.get("protocol_version"))
        return cls(
            _checkpoint_from_payload(payload.get("checkpoint")),
            payload.get("episode_id"),
        )


__all__ = [
    "XIAOMI_MIBOT_PROTOCOL_VERSION",
    "XIAOMI_MIBOT_WIRE_OPERATIONS",
    "XiaomiMiBoTCloseRequest",
    "XiaomiMiBoTHealthResponse",
    "XiaomiMiBoTLoadRequest",
    "XiaomiMiBoTPredictRequest",
    "XiaomiMiBoTPredictResponse",
    "XiaomiMiBoTResetRequest",
    "XiaomiMiBoTResetResponse",
]
