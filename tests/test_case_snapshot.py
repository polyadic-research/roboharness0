from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.case_snapshot import (
    SnapshotIdentity,
    decode_case_snapshot,
    encode_case_snapshot,
)
from tests.test_case_runtime import _registration
from tests.test_episode_coordinator import _manifest, _result


def _terminal(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-1")
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    store.mark_terminal_pending(result)
    detail = store.get_entry(manifest.evaluation_id, case.case_id).value  # type: ignore[union-attr]
    store.publish_terminal(detail, result)
    terminal = store.get_entry(manifest.evaluation_id, case.case_id).value  # type: ignore[union-attr]
    identity = SnapshotIdentity(
        manifest.evaluation_id, manifest.manifest_id, case.case_id, "attempt-1"
    )
    return terminal, result, identity


def test_v1_snapshot_round_trip_excludes_request_decorations_and_event_media(
    tmp_path,
) -> None:
    detail, result, identity = _terminal(tmp_path)
    decorated = detail.model_copy(
        update={
            "available_commands": ("rerun",),
            "attempts_total": 257,
            "attempts_truncated": True,
        }
    )

    payload = encode_case_snapshot(decorated)
    encoded = repr(payload)
    decoded = decode_case_snapshot(payload, identity, result)

    assert set(payload) == {"schema_version", "detail"}
    assert payload["schema_version"] == 1
    assert "available_commands" not in payload["detail"]
    assert "attempts" not in payload["detail"]
    assert "attempts_total" not in payload["detail"]
    assert "events" not in encoded
    assert "image_bytes" not in encoded
    assert decoded.available_commands == ()
    assert decoded.attempts == ()
    assert decoded.attempts_total == 0
    assert decoded.model_dump(exclude={"available_commands", "attempts", "attempts_total", "attempts_truncated"}) == detail.model_dump(exclude={"available_commands", "attempts", "attempts_total", "attempts_truncated"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=True),
        lambda payload: payload.update(schema_version="1"),
        lambda payload: payload["detail"].update(last_sequence="0"),
        lambda payload: payload["detail"].update(evaluation_id="foreign"),
    ],
)
def test_v1_parser_rejects_extra_fields_coercion_and_identity_drift(
    tmp_path, mutation
) -> None:
    detail, result, identity = _terminal(tmp_path)
    payload = deepcopy(encode_case_snapshot(detail))
    mutation(payload)

    with pytest.raises((TypeError, ValueError, ValidationError)):
        decode_case_snapshot(payload, identity, result)


def test_snapshot_terminal_result_must_equal_accepted_episode_result(tmp_path) -> None:
    detail, result, identity = _terminal(tmp_path)
    payload = deepcopy(encode_case_snapshot(detail))
    payload["detail"]["episode_result"]["contract_success"] = not result.contract_success

    with pytest.raises(ValueError, match="accepted EpisodeResult"):
        decode_case_snapshot(payload, identity, result)


def test_unknown_snapshot_version_is_rejected_without_guessing(tmp_path) -> None:
    detail, result, identity = _terminal(tmp_path)
    payload = deepcopy(encode_case_snapshot(detail))
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="version"):
        decode_case_snapshot(payload, identity, result)
