"""Strict version-one terminal Case display snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from rpent.evaluation.results import EpisodeResult
from rpent.ui.schemas import (
    ArtifactDescriptor,
    ArtifactHealthView,
    AttemptSummaryView,
    CameraDescriptorView,
    CaseDataSchemaView,
    CaseDetail,
    ExecutionState,
    RecordingProfileView,
)


class SnapshotRebuilding(RuntimeError):
    """A bounded background Case snapshot rebuild is in progress."""

    code = "case_snapshot_rebuilding"


class SnapshotUnavailable(RuntimeError):
    """The last bounded Case snapshot rebuild failed."""

    code = "case_snapshot_unavailable"


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    evaluation_id: str
    manifest_id: str
    case_id: str
    attempt_id: str


class _StrictSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CaseDetailSnapshotV1(_StrictSnapshot):
    case_id: str
    ordinal: int
    task_name: str
    task_index: int | None
    episode_index: int
    seed: int
    execution_state: ExecutionState
    attempt_id: str
    episode_result: dict[str, JsonValue]
    artifact_health: ArtifactHealthView | None
    current_step: int | None
    started_at: str | None
    finished_at: str | None
    run_summary: AttemptSummaryView | None
    evaluation_id: str
    manifest_id: str
    last_sequence: int
    case: dict[str, JsonValue]
    artifacts: tuple[ArtifactDescriptor, ...]
    recording_profile: RecordingProfileView | None
    cameras: tuple[CameraDescriptorView, ...]
    retained_camera_ids: tuple[str, ...]
    data_schema: CaseDataSchemaView | None
    playback_artifact_id: str | None


class CaseDetailSnapshotEnvelopeV1(_StrictSnapshot):
    schema_version: Literal[1]
    detail: CaseDetailSnapshotV1


_STABLE_FIELDS = frozenset(CaseDetailSnapshotV1.model_fields)


def encode_case_snapshot(detail: CaseDetail) -> dict[str, JsonValue]:
    """Encode only stable terminal display facts."""
    if not isinstance(detail, CaseDetail):
        raise TypeError("detail must be a CaseDetail")
    snapshot = CaseDetailSnapshotV1.model_validate(
        detail.model_dump(include=_STABLE_FIELDS)
    )
    envelope = CaseDetailSnapshotEnvelopeV1(schema_version=1, detail=snapshot)
    return envelope.model_dump(mode="json")


def decode_case_snapshot(
    payload: Mapping[str, object],
    expected_identity: SnapshotIdentity,
    accepted_result: EpisodeResult,
) -> CaseDetail:
    """Strictly decode a snapshot and bind it to retained result authority."""
    if not isinstance(payload, Mapping):
        raise TypeError("snapshot envelope must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Case snapshot version")
    envelope = CaseDetailSnapshotEnvelopeV1.model_validate_json(
        json.dumps(dict(payload), separators=(",", ":"), sort_keys=True)
    )
    detail = envelope.detail
    if not isinstance(expected_identity, SnapshotIdentity):
        raise TypeError("expected_identity must be a SnapshotIdentity")
    if not isinstance(accepted_result, EpisodeResult):
        raise TypeError("accepted_result must be an EpisodeResult")
    actual_identity = SnapshotIdentity(
        detail.evaluation_id,
        detail.manifest_id,
        detail.case_id,
        detail.attempt_id,
    )
    if actual_identity != expected_identity:
        raise ValueError("Case snapshot identity mismatch")
    if (
        accepted_result.evaluation_id != expected_identity.evaluation_id
        or accepted_result.manifest_id != expected_identity.manifest_id
        or accepted_result.case_id != expected_identity.case_id
        or accepted_result.attempt_id != expected_identity.attempt_id
    ):
        raise ValueError("accepted EpisodeResult identity mismatch")
    if detail.episode_result != accepted_result.to_payload():
        raise ValueError("snapshot does not match accepted EpisodeResult")
    return CaseDetail(**detail.model_dump())


__all__ = [
    "CaseDetailSnapshotEnvelopeV1",
    "CaseDetailSnapshotV1",
    "SnapshotIdentity",
    "SnapshotRebuilding",
    "SnapshotUnavailable",
    "decode_case_snapshot",
    "encode_case_snapshot",
]
