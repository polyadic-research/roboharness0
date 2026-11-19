from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import rpent.recording.media as recording_media
from rpent.evaluation.results import EpisodeResult
from rpent.recording.contracts import (
    GENERIC_ARTIFACT_ISSUE,
    JSON_SAFE_INTEGER_MAX,
    ArtifactHealth,
    CameraDescriptor,
    EventEnvelope,
    FrameReference,
    QualificationOutcome,
    RecordingProfile,
    TransitionRecord,
    sanitize_artifact_issues,
)


def _episode_result() -> EpisodeResult:
    return EpisodeResult(
        manifest_id="manifest-1",
        case_id="case-1",
        status="success",
        environment="libero",
        preset_name="libero_10",
        task_name="task-1",
        task_index=0,
        seed=0,
        episode_index=0,
        worker_pid=123,
        worker_exit_code=0,
        started_at="2026-08-06T12:00:00Z",
        finished_at="2026-08-06T12:01:00Z",
        artifact_dir="cases/case-1",
        simulator_success=True,
        contract_success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id="repo/model",
        checkpoint_revision="revision",
        binding_name="binding",
    )


def _camera(camera_id: str = "agent", role: str | None = "primary") -> CameraDescriptor:
    return CameraDescriptor(camera_id, camera_id.title(), role, 256, 256, 3, 20.0)


def test_camera_set_requires_one_primary() -> None:
    cameras = (
        CameraDescriptor("agent", "Agent", "primary", 256, 256, 3, 20.0),
        CameraDescriptor("wrist", "Wrist", "wrist", 256, 256, 3, 20.0),
    )
    assert RecordingProfile.validate("all", (), cameras) == RecordingProfile("all")


@pytest.mark.parametrize(
    "cameras",
    [
        (_camera(role=None),),
        (_camera(), _camera("wrist")),
    ],
)
def test_camera_set_rejects_missing_or_multiple_primary(
    cameras: tuple[CameraDescriptor, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly one primary"):
        RecordingProfile.validate("all", (), cameras)


def test_camera_set_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="camera IDs must be unique"):
        RecordingProfile.validate("all", (), (_camera(), _camera(role="wrist")))


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 0), ("height", -1), ("channels", 0), ("fps", 0.0)],
)
def test_camera_rejects_invalid_dimensions_or_fps(
    field: str, value: int | float
) -> None:
    values = {
        "camera_id": "agent",
        "label": "Agent",
        "role": "primary",
        "width": 256,
        "height": 256,
        "channels": 3,
        "fps": 20.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        CameraDescriptor(**values)


def test_camera_id_enforces_utf8_storage_boundary() -> None:
    boundary = "视" * 170 + "ab"
    assert len(boundary.encode("utf-8")) == 512
    assert _camera(boundary).camera_id == boundary

    with pytest.raises(ValueError, match="512 UTF-8 bytes"):
        _camera(boundary + "c")
    with pytest.raises(ValueError, match="512 UTF-8 bytes"):
        _camera("a" * 513)


def test_recording_profile_rejects_colliding_camera_storage_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recording_media, "camera_storage_component", lambda _camera_id: "same-key"
    )
    cameras = (_camera("agent"), _camera("wrist", role="secondary"))

    with pytest.raises(ValueError, match="storage keys must be unique"):
        RecordingProfile.validate("all", (), cameras)


def test_selected_profile_rejects_unknown_camera() -> None:
    camera = CameraDescriptor("agent", "Agent", "primary", 256, 256, 3, 20.0)
    with pytest.raises(ValueError, match="unknown camera"):
        RecordingProfile.validate("selected", ("missing",), (camera,))


def test_recording_profile_rejects_invalid_camera_selection() -> None:
    camera = _camera()
    with pytest.raises(ValueError, match="must be unique"):
        RecordingProfile.validate("selected", ("agent", "agent"), (camera,))
    with pytest.raises(ValueError, match="requires camera IDs"):
        RecordingProfile.validate("selected", (), (camera,))
    with pytest.raises(ValueError, match="only for selected"):
        RecordingProfile.validate("all", ("agent",), (camera,))


def test_recording_profile_constructor_enforces_selection_invariants() -> None:
    with pytest.raises(ValueError, match="requires camera IDs"):
        RecordingProfile("selected")
    with pytest.raises(ValueError, match="must be unique"):
        RecordingProfile("selected", ("agent", "agent"))
    with pytest.raises(ValueError, match="only for selected"):
        RecordingProfile("all", ("agent",))


@pytest.mark.parametrize("camera_ids", [("agent",), "agent", {"agent": True}])
def test_recording_profile_payload_requires_camera_id_array(camera_ids: object) -> None:
    with pytest.raises(ValueError, match="camera_ids must be a JSON array"):
        RecordingProfile.from_payload({"mode": "selected", "camera_ids": camera_ids})


def test_zero_camera_simulator_accepts_only_none() -> None:
    assert RecordingProfile.validate("none", (), ()) == RecordingProfile("none")
    with pytest.raises(ValueError, match="zero-camera"):
        RecordingProfile.validate("primary", (), ())


@pytest.mark.parametrize(
    ("frame_index", "step_id", "field"),
    [(-1, 0, "frame_index"), (0, -1, "step_id")],
)
def test_frame_reference_rejects_negative_indexes(
    frame_index: int, step_id: int, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        FrameReference("agent", frame_index, step_id, 0.0, "agent/0.png")


@pytest.mark.parametrize("field", ["frame_index", "step_id"])
def test_frame_reference_rejects_indexes_above_json_safe_integer(field: str) -> None:
    values = {"frame_index": 0, "step_id": 0}
    values[field] = JSON_SAFE_INTEGER_MAX + 1
    with pytest.raises(ValueError, match="JSON safe integer"):
        FrameReference("agent", **values, timestamp_s=0.0, relative_path="agent/0.png")


@pytest.mark.parametrize("relative_path", ["/tmp/frame.png", "camera/../frame.png"])
def test_frame_reference_rejects_unsafe_relative_path(relative_path: str) -> None:
    with pytest.raises(ValueError, match="relative_path"):
        FrameReference("agent", 0, 0, 0.0, relative_path)


def test_transition_rejects_negative_indexes() -> None:
    with pytest.raises(ValueError, match="step_id"):
        _transition(step_id=-1)
    with pytest.raises(ValueError, match="frame_index"):
        _transition(frame_index=-1)


@pytest.mark.parametrize("field", ["frame_index", "step_id"])
def test_transition_rejects_indexes_above_json_safe_integer(field: str) -> None:
    with pytest.raises(ValueError, match="JSON safe integer"):
        _transition(**{field: JSON_SAFE_INTEGER_MAX + 1})


def _transition(**overrides: object) -> TransitionRecord:
    values = {
        "case_id": "case-1",
        "attempt_id": "attempt-1",
        "episode_id": "episode-1",
        "step_id": 0,
        "frame_index": 0,
        "timestamp_s": 0.05,
        "state": {"joint": [0.0, 1.0]},
        "action": [0.1, 0.2],
        "reward": 1.0,
        "success": True,
        "terminated": True,
        "truncated": False,
        "operation_id": "operation-1",
        "execution_id": "execution-1",
        "frames": (
            FrameReference("agent", 0, 0, 0.05, "cameras/agent/segments/0.png"),
        ),
    }
    values.update(overrides)
    return TransitionRecord(**values)


@pytest.mark.parametrize("field", ["case_id", "attempt_id"])
def test_transition_requires_case_and_attempt_identity(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _transition(**{field: ""})


def test_transition_payload_contains_case_and_attempt_identity() -> None:
    payload = _transition().to_payload()

    assert payload["case_id"] == "case-1"
    assert payload["attempt_id"] == "attempt-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", {"image": b"png"}),
        ("state", {"labels": {"one", "two"}}),
        ("state", {1: "non-string key"}),
        ("state", {"joint": [float("nan")]}),
        ("action", ["bad"]),
        ("action", [float("inf")]),
        ("reward", float("-inf")),
    ],
)
def test_transition_rejects_non_strict_json(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _transition(**{field: value})


@pytest.mark.parametrize(
    "payload",
    [
        {"binary": b"png"},
        {"values": {1, 2}},
        {1: "non-string key"},
        {"value": float("nan")},
    ],
)
def test_event_envelope_rejects_non_strict_json(payload: object) -> None:
    with pytest.raises(ValueError, match="payload"):
        _event(payload=payload)


def test_contracts_copy_json_inputs_and_payload_snapshots() -> None:
    state = {"joint": [0.0]}
    action = [0.1]
    event_payload = {"nested": {"values": [1]}}
    transition = _transition(state=state, action=action)
    event = _event(payload=event_payload)

    state["joint"].append(1.0)
    action.append(0.2)
    event_payload["nested"]["values"].append(2)
    transition_payload = transition.to_payload()
    envelope_payload = event.to_payload()
    transition_payload["state"]["joint"].append(3.0)
    envelope_payload["payload"]["nested"]["values"].append(3)

    assert transition.state == {"joint": [0.0]}
    assert transition.action == [0.1]
    assert event.payload == {"nested": {"values": [1]}}
    assert transition.to_payload()["state"] == {"joint": [0.0]}
    assert event.to_payload()["payload"] == {"nested": {"values": [1]}}


def test_numpy_json_numbers_are_normalized_to_python_numbers() -> None:
    transition = _transition(
        step_id=np.int64(1),
        timestamp_s=np.float32(0.25),
        state={"count": np.int64(2), "position": [np.float32(0.5)]},
        action=[np.float32(0.75)],
        reward=np.float32(1.0),
    )

    payload = transition.to_payload()
    assert type(payload["step_id"]) is int
    assert type(payload["timestamp_s"]) is float
    assert type(payload["state"]["count"]) is int
    assert type(payload["state"]["position"][0]) is float
    assert type(payload["action"][0]) is float
    assert type(payload["reward"]) is float


@pytest.mark.parametrize("schema_version", [True, 1.0, np.int64(1)])
def test_event_envelope_requires_builtin_integer_schema_version(
    schema_version: object,
) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _event(schema_version=schema_version)


def _event(**overrides: object) -> EventEnvelope:
    values = {
        "schema_version": 1,
        "evaluation_id": "evaluation-1",
        "case_id": "case-1",
        "attempt_id": "attempt-1",
        "sequence": 1,
        "recorded_at": "2026-08-06T12:00:00.000Z",
        "event_type": "transition_recorded",
        "step_id": 0,
        "operation_id": "operation-1",
        "execution_id": "execution-1",
        "payload": {"transition": _transition().to_payload()},
    }
    values.update(overrides)
    return EventEnvelope(**values)


@pytest.mark.parametrize("field", ["sequence", "step_id"])
def test_event_envelope_rejects_indexes_above_json_safe_integer(field: str) -> None:
    with pytest.raises(ValueError, match="JSON safe integer"):
        _event(**{field: JSON_SAFE_INTEGER_MAX + 1})


def test_event_envelope_accepts_json_safe_integer_boundary() -> None:
    event = _event(
        sequence=JSON_SAFE_INTEGER_MAX,
        step_id=JSON_SAFE_INTEGER_MAX,
    )
    assert event.sequence == JSON_SAFE_INTEGER_MAX
    assert event.step_id == JSON_SAFE_INTEGER_MAX


@pytest.mark.parametrize(
    "recorded_at",
    [
        "2026-08-06T12:00:00",
        "2026-08-06T12:00:00+10:00",
        "not-a-date",
    ],
)
def test_event_envelope_requires_timezone_aware_utc_iso(recorded_at: str) -> None:
    with pytest.raises(ValueError, match="recorded_at"):
        _event(recorded_at=recorded_at)


@pytest.mark.parametrize(
    "recorded_at",
    ["2026-08-06T12:00:00Z", "2026-08-06T12:00:00+00:00"],
)
def test_event_envelope_accepts_utc_iso(recorded_at: str) -> None:
    assert _event(recorded_at=recorded_at).recorded_at == recorded_at


def test_every_contract_round_trips_through_payload() -> None:
    result = _episode_result()
    contracts = (
        _camera(),
        RecordingProfile("selected", ("agent",)),
        ArtifactHealth("degraded", ("camera encoder failed",)),
        QualificationOutcome.evaluate(result, ArtifactHealth("complete"), True),
        EventEnvelope(
            schema_version=1,
            evaluation_id="evaluation-1",
            case_id="case-1",
            attempt_id="attempt-1",
            sequence=1,
            recorded_at="2026-08-06T12:00:00.000Z",
            event_type="transition_recorded",
            step_id=0,
            operation_id="operation-1",
            execution_id="execution-1",
            payload={"transition": _transition().to_payload()},
        ),
        FrameReference("agent", 0, 0, 0.05, "cameras/agent/segments/0.png"),
        _transition(),
    )
    for contract in contracts:
        payload = contract.to_payload()
        strict_payload = json.loads(json.dumps(payload, allow_nan=False))
        assert type(contract).from_payload(strict_payload) == contract


def test_complete_artifact_requirement_rejects_degraded_without_rewriting_result() -> (
    None
):
    result = _episode_result()
    outcome = QualificationOutcome.evaluate(
        result,
        ArtifactHealth("degraded", ("camera encoder failed",)),
        require_complete_artifacts=True,
    )

    assert not outcome.accepted
    assert outcome.episode_result is result
    assert outcome.episode_result.status == "success"
    assert replace(outcome.episode_result) == result


@pytest.mark.parametrize("issues", ["encoder failed", ("encoder failed",)])
def test_artifact_health_payload_requires_issues_array(issues: object) -> None:
    with pytest.raises(ValueError, match="issues must be a JSON array"):
        ArtifactHealth.from_payload({"status": "degraded", "issues": issues})


def test_artifact_health_sanitizes_legacy_issues_and_deduplicates_in_order() -> None:
    secret = "/Users/alice/private/model-key"

    health = ArtifactHealth.from_payload(
        {
            "status": "degraded",
            "issues": [
                "camera extraction failed",
                secret,
                "old exception text",
                "camera extraction failed",
                "notification delivery failed",
            ],
        }
    )

    assert health.issues == (
        "camera extraction failed",
        GENERIC_ARTIFACT_ISSUE,
        "notification delivery failed",
    )
    assert sanitize_artifact_issues(health.issues) == health.issues
    assert ArtifactHealth("degraded", (secret,)).issues == (GENERIC_ARTIFACT_ISSUE,)
