from __future__ import annotations

import pytest

from rpent.ui.case_snapshot import SnapshotIdentity
from rpent.ui.operation_snapshot import (
    decode_operation_snapshot,
    encode_operation_snapshot,
)
from rpent.ui.schemas import OperationPage


def _identity() -> SnapshotIdentity:
    return SnapshotIdentity("evaluation-1", "manifest-1", "case-1", "attempt-1")


def _page(*, last_sequence: int) -> OperationPage:
    return OperationPage(
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        last_sequence=last_sequence,
        incomplete_tail=False,
        first_step_id=1,
        latest_step_id=2,
        operations=(),
    )


def test_operation_snapshot_round_trip_binds_identity_and_sequence():
    page = _page(last_sequence=12)
    payload = encode_operation_snapshot(page, "manifest-1")
    assert decode_operation_snapshot(
        payload,
        SnapshotIdentity("evaluation-1", "manifest-1", "case-1", "attempt-1"),
        12,
    ) == page


@pytest.mark.parametrize("field", ["evaluation_id", "manifest_id", "case_id", "attempt_id"])
def test_operation_snapshot_rejects_identity_mismatch(field: str):
    payload = encode_operation_snapshot(_page(last_sequence=12), "manifest-1")
    if field == "manifest_id":
        payload["manifest_id"] = "other"
    else:
        payload["page"][field] = "other"
    with pytest.raises(ValueError, match="identity mismatch"):
        decode_operation_snapshot(payload, _identity(), 12)


def test_operation_snapshot_rejects_final_sequence_mismatch():
    with pytest.raises(ValueError, match="sequence mismatch"):
        decode_operation_snapshot(
            encode_operation_snapshot(_page(last_sequence=11), "manifest-1"),
            _identity(),
            12,
        )
