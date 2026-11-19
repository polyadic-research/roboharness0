from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rpent.evaluation.attempt_registration import AttemptRegistration, register_attempt
from rpent.recording.contracts import EventEnvelope
from rpent.ui.app import create_evaluation_app
from rpent.ui.case_runtime import CaseRuntimeStore, JournalRead, RecordingDeclaration
from rpent.ui.case_snapshot import SnapshotRebuilding, SnapshotUnavailable
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.operation_projection import OperationProjection
from rpent.ui.repository import (
    MAX_RECORDING_DECLARATION_BYTES,
    CurrentAttemptIdentity,
    EvaluationRepository,
)
from rpent.ui.schemas import (
    ArtifactDescriptor,
    CaseDataSchemaView,
    DataVectorView,
    OperationPage,
    RecordingProfileView,
)
from tests.test_episode_coordinator import _manifest, _result
from tests.test_ui_repository import _write_evaluation


def _registration(manifest, case, attempt_id: str = "attempt-1"):
    return AttemptRegistration(
        schema_version=1,
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        registered_at="2026-08-13T00:00:00+00:00",
    )


def _recording() -> RecordingDeclaration:
    empty = DataVectorView(shape=(0,), dimensions=())
    return RecordingDeclaration(
        profile=RecordingProfileView(mode="none"),
        cameras=(),
        retained_camera_ids=(),
        data_schema=CaseDataSchemaView(
            schema_version=1,
            cameras=(),
            action=empty,
            state=empty,
            signals=(),
        ),
    )


def _event(manifest, case, sequence: int, *, attempt_id: str = "attempt-1"):
    return EventEnvelope(
        schema_version=1,
        evaluation_id=manifest.evaluation_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        sequence=sequence,
        recorded_at=f"2026-08-13T00:00:{sequence:02d}+00:00",
        event_type="transition_recorded",
        step_id=sequence - 1,
        operation_id=None,
        execution_id=None,
        payload={},
    )


def _runtime_with_attempt():
    store, registration, recording, _manifest_value, _case = _runtime_context()
    return store, registration, recording


def _runtime_context():
    manifest = _manifest("/tmp/runtime-operation-test", task_count=1)
    case = manifest.cases[0]
    registration = _registration(manifest, case)
    recording = _recording()
    store = CaseRuntimeStore()
    store.start_attempt(registration, manifest, case)
    return store, registration, recording, manifest, case


def _operation_events(registration):
    common = {
        "schema_version": 1,
        "evaluation_id": registration.evaluation_id,
        "case_id": registration.case_id,
        "attempt_id": registration.attempt_id,
        "operation_id": "operation-1",
        "execution_id": "execution-1",
    }
    return (
        EventEnvelope(
            **common,
            sequence=1,
            recorded_at="2026-08-13T00:00:01+00:00",
            event_type="operation_started",
            step_id=None,
            payload={"tool": "move_to_pose"},
        ),
        EventEnvelope(
            **common,
            sequence=2,
            recorded_at="2026-08-13T00:00:02+00:00",
            event_type="transition_recorded",
            step_id=10,
            payload={},
        ),
        EventEnvelope(
            **common,
            sequence=3,
            recorded_at="2026-08-13T00:00:03+00:00",
            event_type="operation_completed",
            step_id=None,
            payload={
                "tool": "move_to_pose",
                "first_step_id": 10,
                "last_step_id": 10,
                "executed_actions": 1,
                "result": {"ok": True},
            },
        ),
        EventEnvelope(
            **common,
            sequence=4,
            recorded_at="2026-08-13T00:00:04+00:00",
            event_type="action_camera_clip_available",
            step_id=None,
            payload={"camera_id": "front", "artifact_id": "front-clip"},
        ),
        EventEnvelope(
            **common,
            sequence=5,
            recorded_at="2026-08-13T00:00:05+00:00",
            event_type="action_camera_clips_ready",
            step_id=None,
            payload={
                "clips": [{"camera_id": "front", "artifact_id": "front-clip"}]
            },
        ),
    )


def test_live_operations_match_journal_projection():
    store, registration, recording = _runtime_with_attempt()
    events = _operation_events(registration)
    initial = store.get_entry(registration.evaluation_id, registration.case_id)
    assert initial is not None
    for event in events:
        assert store.apply_event(event, recording).disposition == "applied"
    expected = OperationProjection()
    for event in events:
        expected.consume(event.to_payload())
    assert store.get_operations(
        registration.evaluation_id, registration.case_id, registration.attempt_id
    ).operations == expected.summaries()
    assert initial.operations.summaries() == ()


def test_get_entry_cannot_expose_retained_operation_state_for_mutation():
    store, registration, _recording_declaration = _runtime_with_attempt()
    entry = store.get_entry(registration.evaluation_id, registration.case_id)
    assert entry is not None
    assert store.get_entry(registration.evaluation_id, registration.case_id) == entry

    entry.operations.consume(_operation_events(registration)[0].to_payload())

    assert entry.operations.summaries()
    assert store.get_entry(registration.evaluation_id, registration.case_id) != entry
    assert store.get_case(
        registration.evaluation_id, registration.case_id
    ).last_sequence == 0
    assert store.get_operations(
        registration.evaluation_id, registration.case_id
    ).operations == ()


def test_duplicate_and_gap_do_not_partially_advance_operations():
    store, registration, recording = _runtime_with_attempt()
    first, second = _operation_events(registration)[:2]
    store.apply_event(first, recording)
    before = store.get_operations(registration.evaluation_id, registration.case_id)
    assert store.apply_event(first, recording).disposition == "duplicate"
    assert store.apply_event(
        replace(second, sequence=99), recording
    ).disposition == "repairing"
    assert store.get_operations(registration.evaluation_id, registration.case_id) == before


def test_duplicate_clip_ready_advances_authoritative_live_artifacts() -> None:
    store, registration, recording = _runtime_with_attempt()
    events = _operation_events(registration)
    for event in events:
        store.apply_event(event, recording)
    authoritative = replace(
        recording,
        artifacts=(
            ArtifactDescriptor(
                artifact_id="front-clip",
                kind="action_camera_clip",
                relative_path="front.mp4",
                available=True,
                byte_size=10,
            ),
        ),
    )
    before = store.get_entry(registration.evaluation_id, registration.case_id)
    assert before is not None

    reduction = store.apply_event(events[-1], authoritative)

    assert reduction.disposition == "applied"
    after = store.get_entry(registration.evaluation_id, registration.case_id)
    assert after is not None
    assert after.last_sequence == before.last_sequence
    assert after.runtime_revision > before.runtime_revision
    assert [
        clip.artifact_id
        for clip in store.get_operations(
            registration.evaluation_id,
            registration.case_id,
            registration.attempt_id,
        ).operations[0].camera_clips
    ] == ["front-clip"]


def test_repair_replays_when_clip_authority_refreshes_midflight() -> None:
    manifest = _manifest("/tmp/runtime-operation-repair", task_count=1)
    case = manifest.cases[0]
    registration = _registration(manifest, case)
    recording = _recording()
    events = _operation_events(registration)
    entered = threading.Event()
    release = threading.Event()
    reads = 0

    def read_journal(_identity, *, through_sequence):
        nonlocal reads
        reads += 1
        assert through_sequence == len(events)
        if reads == 1:
            entered.set()
            assert release.wait(timeout=2)
        return JournalRead(events=events, high_water=len(events))

    store = CaseRuntimeStore(
        journal_reader=read_journal,
        high_water_reader=lambda _identity: len(events),
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    store.start_attempt(registration, manifest, case)
    identity = CurrentAttemptIdentity(
        registration.evaluation_id,
        registration.manifest_id,
        registration.case_id,
        registration.attempt_id,
    )
    store.prime_recording_declaration(identity, recording)
    store.observe_high_water(identity, len(events))
    assert entered.wait(timeout=1)
    authoritative = replace(
        recording,
        artifacts=(
            ArtifactDescriptor(
                artifact_id="front-clip",
                kind="action_camera_clip",
                relative_path="front.mp4",
                available=True,
                byte_size=10,
            ),
        ),
    )
    store.prime_recording_declaration(identity, authoritative)
    assert store.apply_event(events[-1], authoritative).disposition == "repairing"
    release.set()

    assert store.wait_for_repairs(timeout=2)
    assert reads >= 2
    assert [
        clip.artifact_id
        for clip in store.get_operations(
            registration.evaluation_id,
            registration.case_id,
            registration.attempt_id,
        ).operations[0].camera_clips
    ] == ["front-clip"]
    store.close()


def test_finalize_operations_uses_terminal_artifact_authority_without_mutation():
    store, registration, recording, manifest, case = _runtime_context()
    for event in _operation_events(registration):
        store.apply_event(event, recording)
    store.mark_terminal_pending(
        _result(manifest, case, attempt_id=registration.attempt_id)
    )
    before = store.get_entry(registration.evaluation_id, registration.case_id)
    assert before is not None
    retained_page = store.get_operations(
        registration.evaluation_id, registration.case_id
    )
    detail = before.value.model_copy(
        update={
            "artifacts": (
                ArtifactDescriptor(
                    artifact_id="front-clip",
                    kind="action_camera_clip",
                    relative_path="front.mp4",
                    available=True,
                    byte_size=10,
                ),
                ArtifactDescriptor(
                    artifact_id="zero-byte",
                    kind="action_camera_clip",
                    relative_path="empty.mp4",
                    available=True,
                    byte_size=0,
                ),
                ArtifactDescriptor(
                    artifact_id="unavailable",
                    kind="action_camera_clip",
                    relative_path="missing.mp4",
                    available=False,
                    byte_size=10,
                ),
                ArtifactDescriptor(
                    artifact_id="wrong-kind",
                    kind="action_clip",
                    relative_path="action.mp4",
                    available=True,
                    byte_size=10,
                ),
            )
        }
    )

    page = store.finalize_operations(detail)

    assert page.last_sequence == detail.last_sequence
    assert [clip.artifact_id for clip in page.operations[0].camera_clips] == [
        "front-clip"
    ]
    after = store.get_entry(registration.evaluation_id, registration.case_id)
    assert after is not None
    assert after.runtime_revision == before.runtime_revision
    assert store.get_operations(
        registration.evaluation_id, registration.case_id
    ) == retained_page
    assert before.phase == "terminal_pending"


def test_finalize_operations_requires_terminal_pending_phase():
    store, registration, _recording_declaration = _runtime_with_attempt()
    entry = store.get_entry(registration.evaluation_id, registration.case_id)
    assert entry is not None

    with pytest.raises(RuntimeError, match="terminal-pending"):
        store.finalize_operations(entry.value)


def test_finalize_operations_rejects_final_sequence_mismatch():
    store, registration, recording, manifest, case = _runtime_context()
    for event in _operation_events(registration):
        store.apply_event(event, recording)
    store.mark_terminal_pending(
        _result(manifest, case, attempt_id=registration.attempt_id)
    )
    entry = store.get_entry(registration.evaluation_id, registration.case_id)
    assert entry is not None

    with pytest.raises(ValueError, match="sequence mismatch"):
        store.finalize_operations(
            entry.value.model_copy(
                update={"last_sequence": entry.last_sequence + 1}
            )
        )


def test_finalize_operations_rejects_noncurrent_identity():
    store, registration, _recording_declaration = _runtime_with_attempt()
    entry = store.get_entry(registration.evaluation_id, registration.case_id)
    assert entry is not None

    with pytest.raises(ValueError, match="identity mismatch"):
        store.finalize_operations(
            entry.value.model_copy(update={"attempt_id": "attempt-other"})
        )


@pytest.mark.parametrize(
    ("last_sequence", "incomplete_tail"),
    ((4, False), (6, False), (5, True)),
    ids=("shorter", "longer", "incomplete-terminal-tail"),
)
def test_operation_rebuild_rejects_journal_without_exact_terminal_authority(
    last_sequence: int,
    incomplete_tail: bool,
):
    page = OperationPage(
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        last_sequence=last_sequence,
        incomplete_tail=incomplete_tail,
        first_step_id=None,
        latest_step_id=None,
        operations=(),
    )

    def missing(*_args, **_kwargs):
        raise FileNotFoundError

    store = CaseRuntimeStore(
        operation_snapshot_loader=missing,
        operation_snapshot_reconstructor=lambda *_args, **_kwargs: page,
        operation_sequence_reader=lambda *_args, **_kwargs: 5,
    )

    with pytest.raises(SnapshotRebuilding):
        store.get_operations("evaluation-1", "case-1", "attempt-1")
    assert store.wait_for_snapshots(timeout=2)
    with pytest.raises(SnapshotUnavailable):
        store.get_operations("evaluation-1", "case-1", "attempt-1")
    store.close()


def test_gap_repair_replaces_case_and_operation_projections_together():
    manifest = _manifest("/tmp/runtime-operation-repair-test", task_count=1)
    case = manifest.cases[0]
    registration = _registration(manifest, case)
    recording = _recording()
    events = _operation_events(registration)
    store = CaseRuntimeStore(
        journal_reader=lambda _identity, **_kwargs: JournalRead(events, len(events)),
        high_water_reader=lambda _identity: len(events),
        update_publisher=lambda _update: None,
    )
    store.start_attempt(registration, manifest, case)
    store.prime_recording_declaration(
        CurrentAttemptIdentity(
            registration.evaluation_id,
            registration.manifest_id,
            registration.case_id,
            registration.attempt_id,
        ),
        recording,
    )

    assert store.apply_event(replace(events[0], sequence=99), recording).disposition == "repairing"
    assert store.wait_for_repairs(timeout=2)
    expected = OperationProjection()
    for event in events:
        expected.consume(event.to_payload())
    case_detail = store.get_case(registration.evaluation_id, registration.case_id)
    operations = store.get_operations(registration.evaluation_id, registration.case_id)

    assert case_detail.last_sequence == operations.last_sequence == len(events)
    assert operations.operations == expected.summaries()


def test_runtime_entry_accepts_recursively_frozen_data_schema(tmp_path) -> None:
    expected = _recording().data_schema
    manifest = _manifest(
        tmp_path,
        task_count=1,
        contracts={"data_schema": expected.model_dump(mode="json")},
    )
    frozen = manifest.contracts["data_schema"]
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["action"], MappingProxyType)
    case = manifest.cases[0]

    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    entry = store.get_entry(manifest.evaluation_id, case.case_id)

    assert entry is not None
    assert entry.value.data_schema == expected


def test_runtime_entry_and_case_detail_are_immutable(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None
    with pytest.raises(FrozenInstanceError):
        entry.runtime_revision = 9  # type: ignore[misc]
    with pytest.raises(Exception):
        entry.value.last_sequence = 9


def test_concurrent_adjacent_sequences_are_linearizable_without_lost_update(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    barrier = threading.Barrier(3)

    def apply(sequence: int):
        barrier.wait()
        return store.apply_event(_event(manifest, case, sequence), _recording())

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(apply, 1)
        second = executor.submit(apply, 2)
        barrier.wait()
        dispositions = {first.result().disposition, second.result().disposition}

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None
    if "repairing" in dispositions:
        assert entry.last_sequence in {0, 1}
        assert entry.phase == "repairing"
    else:
        assert dispositions == {"applied"}
        assert entry.last_sequence == 2
        assert entry.value.current_step == 1


def test_duplicate_gap_repairing_stale_attempt_and_cross_case_dispositions(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path, task_count=2)
    case, other = manifest.cases
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    store.start_attempt(
        _registration(manifest, other, "attempt-other"), manifest, other
    )

    applied = store.apply_event(_event(manifest, case, 1), _recording())
    duplicate = store.apply_event(_event(manifest, case, 1), _recording())
    gap = store.apply_event(_event(manifest, case, 3), _recording())
    repairing = store.apply_event(_event(manifest, case, 2), _recording())
    repairing_gap = store.apply_event(_event(manifest, case, 4), _recording())
    stale = store.apply_event(
        _event(manifest, case, 2, attempt_id="attempt-stale"), _recording()
    )

    assert applied.disposition == "applied"
    assert applied.update is not None
    assert duplicate.disposition == "duplicate"
    assert duplicate.update is None
    assert gap.disposition == "repairing"
    assert gap.sequence_gap is True
    assert repairing.disposition == "repairing"
    assert repairing.sequence_gap is False
    assert repairing_gap.disposition == "repairing"
    assert repairing_gap.sequence_gap is True
    assert stale.disposition == "rejected"
    untouched = store.get_entry(manifest.evaluation_id, other.case_id)
    assert untouched is not None and untouched.last_sequence == 0


def test_runtime_revision_is_per_case_monotonic_across_attempt_promotion(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()

    started = store.start_attempt(_registration(manifest, case), manifest, case)
    event = store.apply_event(_event(manifest, case, 1), _recording())
    successor = store.start_attempt(
        _registration(manifest, case, "attempt-2"), manifest, case
    )

    assert [started.runtime_revision, event.update.runtime_revision, successor.runtime_revision] == [1, 2, 3]  # type: ignore[union-attr]
    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.attempt_epoch == 2


def test_terminal_publication_racing_late_event_wins_permanently(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    store.apply_event(_event(manifest, case, 1), _recording())
    pending = store.mark_terminal_pending(
        _result(manifest, case, attempt_id="attempt-1")
    )
    detail = store.get_entry(manifest.evaluation_id, case.case_id).value  # type: ignore[union-attr]
    result = _result(manifest, case, attempt_id="attempt-1")
    sealed = store.publish_terminal(detail, result)
    late = store.apply_event(_event(manifest, case, 2), _recording())

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert pending.runtime_revision < sealed.runtime_revision
    assert late.disposition == "rejected"
    assert entry is not None and entry.phase == "sealed"
    assert entry.runtime_revision == sealed.runtime_revision


def test_runtime_metrics_count_sealed_entries_separately_from_active_entries(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path, task_count=2)
    store = CaseRuntimeStore()
    for case in manifest.cases:
        store.start_attempt(_registration(manifest, case), manifest, case)
    terminal_case = manifest.cases[0]
    result = _result(manifest, terminal_case, attempt_id="attempt-1")
    store.mark_terminal_pending(result)
    entry = store.get_entry(manifest.evaluation_id, terminal_case.case_id)
    assert entry is not None
    store.publish_terminal(entry.value, result)

    metrics = store.snapshot()

    assert metrics["active_entries"] == 1
    assert metrics["terminal_entries"] == 1


def test_recording_bootstrap_absent_returns_retryable_without_advancing_sequence(
    tmp_path,
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    registration = register_attempt(
        manifest, case, "attempt-1", "2026-08-13T00:00:00+00:00"
    )
    repository = EvaluationRepository((root.parent,))
    store = CaseRuntimeStore()
    store.start_attempt(registration, manifest, case)
    hub = EvaluationEventHub(secret=b"runtime-test-secret")
    app = create_evaluation_app(
        repository=repository,
        service=SimpleNamespace(),
        event_hub=hub,
        notification_token="notification-secret",
        case_runtime_store=store,
    )
    client = TestClient(
        app,
        headers={"X-RPent-Notification-Token": "notification-secret"},
        raise_server_exceptions=False,
    )
    event = _event(manifest, case, 1)

    unavailable = client.post("/api/internal/events", json=event.to_payload())
    entry = store.get_entry(manifest.evaluation_id, case.case_id)

    assert unavailable.status_code == 503
    assert unavailable.headers["Retry-After"] == "1"
    assert unavailable.json() == {"detail": "recording_declaration_unavailable"}
    assert entry is not None and entry.last_sequence == 0

    recording_path = (
        root
        / "cases"
        / case.case_id
        / "attempts"
        / "attempt-1"
        / "recording.json"
    )
    recording_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_id": manifest.evaluation_id,
                "case_id": case.case_id,
                "attempt_id": "attempt-1",
                "episode_id": case.case_id,
                "profile": {"mode": "none", "camera_ids": []},
                "cameras": [],
                "retained_camera_ids": [],
            }
        ),
        encoding="utf-8",
    )
    cursor = hub.cursor(manifest.evaluation_id)

    assert client.post("/api/internal/events", json=event.to_payload()).status_code == 204
    assert client.post("/api/internal/events", json=event.to_payload()).status_code == 204
    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.last_sequence == 1
    assert len(hub.read(cursor, evaluation_id=manifest.evaluation_id).events) == 1


def test_clip_ready_ingress_refreshes_live_artifact_authority_without_get_scan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    registration = register_attempt(
        manifest, case, "attempt-1", "2026-08-13T00:00:00+00:00"
    )
    attempt = root / "cases" / case.case_id / "attempts" / "attempt-1"
    recording_path = attempt / "recording.json"
    recording = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": "attempt-1",
        "episode_id": case.case_id,
        "profile": {"mode": "all", "camera_ids": []},
        "cameras": [
            {
                "camera_id": "front",
                "label": "Front",
                "role": "primary",
                "width": 2,
                "height": 2,
                "channels": 3,
                "fps": 20.0,
                "media_type": "rgb",
            }
        ],
        "retained_camera_ids": ["front"],
        "artifacts": [],
    }
    recording_path.write_text(json.dumps(recording), encoding="utf-8")
    repository = EvaluationRepository((root.parent,))
    store = CaseRuntimeStore()
    store.start_attempt(registration, manifest, case)
    app = create_evaluation_app(
        repository=repository,
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"live-artifact-secret"),
        notification_token="notification-secret",
        case_runtime_store=store,
    )
    client = TestClient(
        app,
        headers={"X-RPent-Notification-Token": "notification-secret"},
    )
    events = _operation_events(registration)
    for event in events[:-1]:
        assert client.post(
            "/api/internal/events", json=event.to_payload()
        ).status_code == 204
    assert store.get_operations(
        manifest.evaluation_id, case.case_id, registration.attempt_id
    ).operations[0].camera_clips == ()

    clip = attempt / "action_videos" / "operation-1" / "front.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"retained-clip")
    recording["artifacts"] = [
        {
            "artifact_id": "front-clip",
            "kind": "action_camera_clip",
            "relative_path": clip.relative_to(attempt).as_posix(),
        }
    ]
    recording_path.write_text(json.dumps(recording), encoding="utf-8")
    assert client.post(
        "/api/internal/events", json=events[-1].to_payload()
    ).status_code == 204

    monkeypatch.setattr(
        repository,
        "_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live operations GET scanned artifacts")
        ),
    )
    response = client.get(
        f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}"
        f"/operations?attempt_id={registration.attempt_id}"
    )

    assert response.status_code == 200
    assert response.json()["operations"][0]["camera_clips"] == [
        {"camera_id": "front", "artifact_id": "front-clip"}
    ]
    detail = store.get_case(manifest.evaluation_id, case.case_id)
    assert [artifact.artifact_id for artifact in detail.artifacts] == ["front-clip"]
    assert detail.artifacts[0].available is True
    assert detail.artifacts[0].byte_size == len(b"retained-clip")


def test_recording_declaration_cache_is_single_flight(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return _recording()

    async def run():
        key = (manifest.evaluation_id, case.case_id, "attempt-1")
        return await asyncio.gather(
            store.recording_declaration(key, load),
            store.recording_declaration(key, load),
        )

    declarations = asyncio.run(run())

    assert calls == 1
    assert declarations == [_recording(), _recording()]


def test_active_case_get_reads_only_runtime_snapshot(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore()
    store.start_attempt(_registration(manifest, case), manifest, case)

    class Repository:
        def get_case(self, *_args, **_kwargs):
            raise AssertionError("active GET must not reconstruct repository state")

    app = create_evaluation_app(
        repository=Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"runtime-get-secret"),
        case_runtime_store=store,
    )
    client = TestClient(app)

    response = client.get(
        f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}"
    )

    assert response.status_code == 200
    assert response.json()["attempt_id"] == "attempt-1"
    assert response.json()["last_sequence"] == 0


def test_recording_bootstrap_rejects_identity_mismatch_and_byte_overflow(
    tmp_path,
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    register_attempt(
        manifest, case, "attempt-1", "2026-08-13T00:00:00+00:00"
    )
    repository = EvaluationRepository((root.parent,))
    identity = CurrentAttemptIdentity(
        manifest.evaluation_id, manifest.manifest_id, case.case_id, "attempt-1"
    )
    path = root / "cases" / case.case_id / "attempts" / "attempt-1" / "recording.json"
    payload = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": "foreign",
        "episode_id": case.case_id,
        "profile": {"mode": "none", "camera_ids": []},
        "cameras": [],
        "retained_camera_ids": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attempt_id identity mismatch"):
        repository.load_recording_declaration(identity)

    path.write_bytes(b"{" + b" " * MAX_RECORDING_DECLARATION_BYTES + b"}")
    with pytest.raises(ValueError, match="byte limit"):
        repository.load_recording_declaration(identity)
