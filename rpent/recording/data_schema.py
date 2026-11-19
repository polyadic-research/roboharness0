"""Versioned, simulator-described Case data metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from rpent.recording.contracts import CameraDescriptor


@dataclass(frozen=True, slots=True)
class CameraSchema:
    id: str
    label: str
    role: str | None
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class DimensionSchema:
    index: int
    name: str
    label: str
    group: str
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class VectorSchema:
    shape: tuple[int, ...]
    dimensions: tuple[DimensionSchema, ...]


@dataclass(frozen=True, slots=True)
class SignalSchema:
    id: str
    label: str
    kind: str
    authoritative: bool


@dataclass(frozen=True, slots=True)
class CaseDataSchema:
    schema_version: int
    cameras: tuple[CameraSchema, ...]
    action: VectorSchema
    state: VectorSchema
    signals: tuple[SignalSchema, ...]

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SIGNALS = (
    SignalSchema("environment.success", "Simulator success", "boolean", True),
    SignalSchema("contract.success", "Contract success", "boolean", True),
    SignalSchema("reward", "Reward", "number", True),
    SignalSchema("environment.terminated", "Terminated", "boolean", True),
    SignalSchema("environment.truncated", "Truncated", "boolean", True),
)


def dimensions(
    prefix: str,
    count: int,
    declared: Iterable[tuple[str, str, str, str | None]] = (),
) -> tuple[DimensionSchema, ...]:
    supplied = tuple(declared)
    if supplied and len(supplied) != count:
        raise ValueError(f"{prefix} dimensions do not match shape")
    values = supplied or tuple(
        (f"{prefix}[{index}]", f"{prefix}[{index}]", prefix, None)
        for index in range(count)
    )
    return tuple(
        DimensionSchema(index, name, label, group, unit)
        for index, (name, label, group, unit) in enumerate(values)
    )


def build_data_schema(
    *,
    cameras: tuple[CameraDescriptor, ...],
    action_shape: tuple[int, ...],
    state_shape: tuple[int, ...],
    action_dimensions: Iterable[tuple[str, str, str, str | None]] = (),
    state_dimensions: Iterable[tuple[str, str, str, str | None]] = (),
    signals: tuple[SignalSchema, ...] = DEFAULT_SIGNALS,
) -> CaseDataSchema:
    for name, shape in (("action", action_shape), ("state", state_shape)):
        if len(shape) > 1 or any(not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError(f"{name} shape must describe one vector")
    action_count = action_shape[0] if action_shape else 0
    state_count = state_shape[0] if state_shape else 0
    return CaseDataSchema(
        schema_version=1,
        cameras=tuple(
            CameraSchema(
                item.camera_id, item.label, item.role, item.width, item.height, item.fps
            )
            for item in cameras
        ),
        action=VectorSchema(
            action_shape,
            dimensions("action", action_count, action_dimensions),
        ),
        state=VectorSchema(
            state_shape,
            dimensions("state", state_count, state_dimensions),
        ),
        signals=signals,
    )


__all__ = [
    "CaseDataSchema",
    "DimensionSchema",
    "SignalSchema",
    "build_data_schema",
    "dimensions",
]
