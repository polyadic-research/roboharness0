"""Strict version-one operation display snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from rpent.ui.case_snapshot import SnapshotIdentity
from rpent.ui.schemas import OperationPage


class _StrictSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class OperationSnapshotEnvelopeV1(_StrictSnapshot):
    schema_version: Literal[1]
    manifest_id: str
    page: OperationPage


def encode_operation_snapshot(
    page: OperationPage, manifest_id: str
) -> dict[str, JsonValue]:
    """Encode an operation page with the manifest binding required for replay."""
    envelope = OperationSnapshotEnvelopeV1(
        schema_version=1, manifest_id=manifest_id, page=page
    )
    return envelope.model_dump(mode="json")


def decode_operation_snapshot(
    payload: Mapping[str, object],
    expected_identity: SnapshotIdentity,
    final_sequence: int,
) -> OperationPage:
    """Decode and authorize an operation snapshot against retained identity."""
    envelope = OperationSnapshotEnvelopeV1.model_validate_json(
        json.dumps(dict(payload), separators=(",", ":"), sort_keys=True)
    )
    actual = SnapshotIdentity(
        envelope.page.evaluation_id,
        envelope.manifest_id,
        envelope.page.case_id,
        envelope.page.attempt_id,
    )
    if actual != expected_identity:
        raise ValueError("Operation snapshot identity mismatch")
    if envelope.page.last_sequence != final_sequence:
        raise ValueError("Operation snapshot sequence mismatch")
    return envelope.page


__all__ = [
    "OperationSnapshotEnvelopeV1",
    "decode_operation_snapshot",
    "encode_operation_snapshot",
]
