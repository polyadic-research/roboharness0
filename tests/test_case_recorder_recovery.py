from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

import rpent.recording.media as media
from rpent.recording.contracts import RecordingProfile
from rpent.recording.recorder import CaseRecorder
from tests.test_case_recorder import (
    FakeDisplayAdapter,
    _camera,
    _recorder,
    _result,
    _step,
)


def _fail_next_append_after_prefix(
    monkeypatch: pytest.MonkeyPatch, *, marker: bytes | None = None
) -> None:
    real_write_all = media._write_all
    failed = False

    def partial_then_fail(stream, value: bytes) -> None:
        nonlocal failed
        if not failed and (marker is None or marker in value):
            failed = True
            stream.write(value[:7])
            stream.flush()
            raise OSError("partial append failed")
        real_write_all(stream, value)

    monkeypatch.setattr(media, "_write_all", partial_then_fail)


def test_write_all_handles_short_writes() -> None:
    class ShortWriter(BytesIO):
        def write(self, value: bytes) -> int:
            return super().write(value[:3])

    stream = ShortWriter()

    media._write_all(stream, b"complete-record")

    assert stream.getvalue() == b"complete-record"


def test_append_reports_both_original_and_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"

    def partial_failure(stream, value: bytes) -> None:
        stream.write(value[:3])
        raise OSError("original append failure")

    def rollback_failure(stream, original_size: int, *, created: bool) -> None:
        raise OSError("rollback failure")

    monkeypatch.setattr(media, "_write_all", partial_failure)
    monkeypatch.setattr(media, "_rollback_append", rollback_failure)

    with pytest.raises(media.JsonlRollbackError, match="cannot continue") as caught:
        media.append_jsonl(path, {"event": "broken"})

    assert "original append failure" in str(caught.value.append_error)
    assert "rollback failure" in str(caught.value.rollback_error)


def test_partial_event_append_rolls_back_and_same_instance_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record("before")
    _fail_next_append_after_prefix(monkeypatch)

    with pytest.raises(OSError, match="partial append failed"):
        recorder.record("partial")
    recorder.record("after")
    recorder.close()

    reopened = _recorder(tmp_path)
    assert reopened.snapshot()["event_count"] == 2
    rows, incomplete = media.read_complete_jsonl(tmp_path / "events.jsonl")
    assert not incomplete
    assert [row["event_type"] for row in rows] == ["before", "after"]


def test_new_event_parent_fsync_failure_rolls_back_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_fsync_directory = media._fsync_directory
    failed = False

    def fail_first_parent_fsync(path: Path) -> None:
        nonlocal failed
        if path == tmp_path and not failed:
            failed = True
            raise OSError("parent fsync failed")
        real_fsync_directory(path)

    monkeypatch.setattr(media, "_fsync_directory", fail_first_parent_fsync)

    with pytest.raises(OSError, match="parent fsync failed"):
        recorder.record("not_committed")
    assert recorder.snapshot()["event_count"] == 0
    assert not (tmp_path / "events.jsonl").exists()

    assert recorder.record("committed")["sequence"] == 1
    recorder.close()
    reopened = _recorder(tmp_path)
    rows, incomplete = media.read_complete_jsonl(tmp_path / "events.jsonl")
    assert not incomplete
    assert [row["sequence"] for row in rows] == [1]
    assert reopened.snapshot()["event_count"] == 1


def test_rollback_failure_poisons_recorder_until_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_write_all = media._write_all
    real_rollback = media._rollback_append
    write_failed = False
    rollback_failed = False

    def fail_one_write(stream, value: bytes) -> None:
        nonlocal write_failed
        if not write_failed:
            write_failed = True
            stream.write(value[:5])
            raise OSError("partial event")
        real_write_all(stream, value)

    def fail_one_rollback(stream, original_size: int, *, created: bool) -> None:
        nonlocal rollback_failed
        if not rollback_failed:
            rollback_failed = True
            raise OSError("truncate failed")
        real_rollback(stream, original_size, created=created)

    monkeypatch.setattr(media, "_write_all", fail_one_write)
    monkeypatch.setattr(media, "_rollback_append", fail_one_rollback)

    with pytest.raises(media.JsonlRollbackError, match="cannot continue"):
        recorder.record("poisons")

    snapshot = recorder.snapshot()
    assert snapshot["health"]["status"] == "degraded"
    assert any("poison" in issue for issue in snapshot["health"]["issues"])
    with pytest.raises(RuntimeError, match="poison"):
        recorder.record("rejected")
    with pytest.raises(RuntimeError, match="poison"):
        recorder.record_transition(
            action=np.zeros(1),
            result=_step(),
            timestamp_s=0.0,
            operation_id=None,
            execution_id=None,
        )
    with pytest.raises(RuntimeError, match="poison"):
        recorder.finalize(_result(tmp_path))
    assert not (tmp_path / "artifacts.json").exists()
    recorder.close()

    reopened = _recorder(tmp_path)
    assert reopened.record("recovered")["sequence"] == 1


def test_recovery_stops_after_first_unrecoverable_synthesis_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    for step_id in range(3):
        recorder.record_transition(
            action=np.array([step_id], dtype=np.float32),
            result=_step(step_id),
            timestamp_s=step_id * 0.1,
            operation_id=f"operation-{step_id}",
            execution_id=f"execution-{step_id}",
        )
    recorder.close()
    (tmp_path / "events.jsonl").unlink()
    media.append_jsonl(
        tmp_path / "health.jsonl",
        {
            "evaluation_id": "evaluation-1",
            "case_id": "case-1",
            "attempt_id": "attempt-1",
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": "transition event recovered",
        },
    )

    real_write_all = media._write_all
    real_rollback = media._rollback_append
    write_failed = False
    rollback_failed = False

    def fail_one_write(stream, value: bytes) -> None:
        nonlocal write_failed
        if not write_failed:
            write_failed = True
            stream.write(value[:5])
            raise OSError("partial repair event")
        real_write_all(stream, value)

    def fail_one_rollback(stream, original_size: int, *, created: bool) -> None:
        nonlocal rollback_failed
        if not rollback_failed:
            rollback_failed = True
            raise OSError("repair truncate failed")
        real_rollback(stream, original_size, created=created)

    monkeypatch.setattr(media, "_write_all", fail_one_write)
    monkeypatch.setattr(media, "_rollback_append", fail_one_rollback)

    with pytest.raises(media.JsonlRollbackError, match="cannot continue"):
        _recorder(tmp_path)

    damaged = (tmp_path / "events.jsonl").read_bytes()
    assert damaged and b"\n" not in damaged

    recovered = _recorder(tmp_path)
    rows, incomplete = media.read_complete_jsonl(tmp_path / "events.jsonl")

    assert not incomplete
    assert recovered.snapshot()["event_count"] == 3
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["step_id"] for row in rows] == [0, 1, 2]


def test_partial_transition_append_rolls_back_and_same_instance_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    _fail_next_append_after_prefix(monkeypatch)

    with pytest.raises(OSError, match="partial append failed"):
        recorder.record_transition(
            action=np.ones(1),
            result=_step(1),
            timestamp_s=0.1,
            operation_id=None,
            execution_id=None,
        )
    recorder.record_transition(
        action=np.ones(1),
        result=_step(1),
        timestamp_s=0.1,
        operation_id=None,
        execution_id=None,
    )
    recorder.close()

    reopened = _recorder(tmp_path)
    assert reopened.snapshot()["transition_count"] == 2
    rows, incomplete = media.read_complete_jsonl(
        tmp_path / "trajectory/transitions.jsonl"
    )
    assert not incomplete
    assert [row["step_id"] for row in rows] == [0, 1]


@pytest.mark.parametrize("retry", ("record", "close", "finalize"))
def test_partial_health_append_is_retried_and_survives_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: str
) -> None:
    recorder = _recorder(tmp_path)
    _fail_next_append_after_prefix(monkeypatch, marker=b'"issue"')
    recorder.notify = lambda _event: (_ for _ in ()).throw(OSError("hub failed"))

    recorder.record("causes_health_issue")
    recorder.notify = None
    if retry == "record":
        recorder.record("retries_health")
        recorder.close()
    elif retry == "finalize":
        monkeypatch.setattr(
            media,
            "write_parquet",
            lambda path, _records: path.write_bytes(b"PAR1fixture"),
        )
        recorder.finalize(_result(tmp_path))
    else:
        recorder.close()

    reopened = _recorder(tmp_path)
    assert any(
        "notification delivery failed" in issue
        for issue in reopened.snapshot()["health"]["issues"]
    )


def test_recovery_ignores_incomplete_jsonl_tails_and_continues_sequences(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record("started")
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(2),
        timestamp_s=0.1,
        operation_id=None,
        execution_id=None,
    )
    recorder.close()
    with (tmp_path / "events.jsonl").open("ab") as stream:
        stream.write(b'{"sequence":')
    with (tmp_path / "trajectory/transitions.jsonl").open("ab") as stream:
        stream.write(b'{"step_id":')

    recovered = _recorder(tmp_path)
    event = recovered.record("resumed")

    assert event["sequence"] == 3
    assert recovered.snapshot()["transition_count"] == 1
    assert any(
        "incomplete" in issue for issue in recovered.snapshot()["health"]["issues"]
    )
    recovered.close()

    reopened = _recorder(tmp_path)
    assert reopened.record("continued")["sequence"] == 4
    assert reopened.snapshot()["transition_count"] == 1


def test_display_failure_keeps_confirmed_transition_metadata(tmp_path: Path) -> None:
    camera = _camera()

    class BrokenDisplay(FakeDisplayAdapter):
        def frames(self, observation: dict[str, object]) -> dict[str, bytes]:
            raise ValueError("encoder failed")

        def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
            raise ValueError("encoder failed")

    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=BrokenDisplay((camera,)),
        profile=RecordingProfile("all"),
    )

    recorder.record_transition(
        action=np.ones(2),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert row["action"] == [1.0, 1.0]
    assert row["frames"] == []
    assert recorder.snapshot()["health"]["status"] == "degraded"


def test_one_camera_disk_failure_does_not_lose_other_camera_or_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_atomic_write = media.atomic_write_bytes

    def fail_wrist(path: Path, value: bytes) -> None:
        if (
            media.camera_storage_component("wrist") in path.parts
            and "segments" in path.parts
        ):
            raise OSError("disk full")
        real_atomic_write(path, value)

    monkeypatch.setattr(media, "atomic_write_bytes", fail_wrist)
    recorder = _recorder(tmp_path)

    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert [frame["camera_id"] for frame in row["frames"]] == ["agent"]
    assert (tmp_path / row["frames"][0]["relative_path"]).exists()
    assert recorder.snapshot()["transition_count"] == 1


def test_finalizer_fault_still_atomically_publishes_degraded_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    def broken_video(*_args, **_kwargs) -> None:
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(media, "finalize_video", broken_video)
    manifest = recorder.finalize(_result(tmp_path))

    assert manifest["health"]["status"] == "degraded"
    assert "camera video finalization failed" in manifest["health"]["issues"]
    assert json.loads((tmp_path / "artifacts.json").read_text()) == manifest
    assert (
        json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())["step_id"]
        == 0
    )


def test_finalizer_health_rollback_poison_blocks_artifacts_until_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    real_write_all = media._write_all
    real_rollback = media._rollback_append
    health_write_failed = False
    rollback_failed = False

    def fail_health_write(stream, value: bytes) -> None:
        nonlocal health_write_failed
        if b'"issue"' in value and not health_write_failed:
            health_write_failed = True
            stream.write(value[:5])
            raise OSError("partial health")
        real_write_all(stream, value)

    def fail_health_rollback(stream, original_size: int, *, created: bool) -> None:
        nonlocal rollback_failed
        if not rollback_failed:
            rollback_failed = True
            raise OSError("health truncate failed")
        real_rollback(stream, original_size, created=created)

    monkeypatch.setattr(media, "_write_all", fail_health_write)
    monkeypatch.setattr(media, "_rollback_append", fail_health_rollback)
    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parquet failed")),
    )

    with pytest.raises(RuntimeError, match="poison"):
        recorder.finalize(_result(tmp_path))

    assert not (tmp_path / "artifacts.json").exists()
    assert recorder.snapshot()["health"]["status"] == "degraded"
    assert any("poison" in issue for issue in recorder.snapshot()["health"]["issues"])
    recorder.close()

    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )
    reopened = _recorder(tmp_path)
    manifest = reopened.finalize(_result(tmp_path))

    assert manifest == json.loads((tmp_path / "artifacts.json").read_text())


def test_parquet_success_receives_all_rows_and_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    for step_id in (0, 1):
        recorder.record_transition(
            action=np.array([step_id], dtype=np.float32),
            result=_step(step_id),
            timestamp_s=step_id * 0.05,
            operation_id=None,
            execution_id=None,
        )
    captured: dict[str, object] = {}

    def successful_parquet(path: Path, records) -> None:
        captured["path"] = path
        captured["records"] = list(records)
        path.write_bytes(b"PAR1fixture")

    monkeypatch.setattr(media, "write_parquet", successful_parquet)

    manifest = recorder.finalize(_result(tmp_path))

    assert captured["path"] == tmp_path / "trajectory/transitions.parquet"
    assert [row["step_id"] for row in captured["records"]] == [0, 1]
    assert [row["action"] for row in captured["records"]] == [[0.0], [1.0]]
    assert manifest["health"]["status"] == "complete"
    assert {
        "kind": "transitions_parquet",
        "relative_path": "trajectory/transitions.parquet",
    } in manifest["artifacts"]


def test_missing_pyarrow_is_explicitly_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    def missing_pyarrow(*_args, **_kwargs) -> None:
        raise RuntimeError("Parquet finalization requires pyarrow")

    monkeypatch.setattr(media, "write_parquet", missing_pyarrow)
    manifest = recorder.finalize(_result(tmp_path))

    assert manifest["health"]["status"] == "degraded"
    assert "Parquet finalization failed" in manifest["health"]["issues"]
    assert not any(
        artifact["kind"] == "transitions_parquet" for artifact in manifest["artifacts"]
    )


def test_real_parquet_is_readable_when_pyarrow_is_installed(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.array([3.0], dtype=np.float32),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    manifest = recorder.finalize(_result(tmp_path))
    table = parquet.read_table(tmp_path / "trajectory/transitions.parquet")

    assert table.to_pylist()[0]["action"] == [3.0]
    assert any(
        artifact["kind"] == "transitions_parquet" for artifact in manifest["artifacts"]
    )


def test_transition_validates_action_and_timestamp_before_writing(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)

    with pytest.raises(ValueError, match="action"):
        recorder.record_transition(
            action=np.array([np.nan]),
            result=_step(),
            timestamp_s=0.0,
            operation_id=None,
            execution_id=None,
        )
    with pytest.raises(ValueError, match="timestamp"):
        recorder.record_transition(
            action=np.zeros(1),
            result=_step(),
            timestamp_s=-1.0,
            operation_id=None,
            execution_id=None,
        )
    assert not (tmp_path / "trajectory/transitions.jsonl").exists()


def test_invalid_step_result_does_not_publish_camera_files(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    invalid = _step()
    object.__setattr__(invalid, "reward", float("nan"))

    with pytest.raises(ValueError, match="reward"):
        recorder.record_transition(
            action=np.zeros(1),
            result=invalid,
            timestamp_s=0.0,
            operation_id=None,
            execution_id=None,
        )

    assert not (tmp_path / "live").exists()
    assert not (tmp_path / "trajectory/cameras").exists()


def test_valid_json_without_final_newline_is_repaired_before_append(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record("started")
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    recorder.close()
    events_path = tmp_path / "events.jsonl"
    transitions_path = tmp_path / "trajectory/transitions.jsonl"
    events_path.write_bytes(events_path.read_bytes().rstrip(b"\n"))
    transitions_path.write_bytes(transitions_path.read_bytes().rstrip(b"\n"))

    recovered = _recorder(tmp_path)
    recovered.record_transition(
        action=np.ones(1),
        result=_step(1),
        timestamp_s=0.1,
        operation_id=None,
        execution_id=None,
    )
    recovered.record("continued")
    recovered.close()

    reopened = _recorder(tmp_path)
    assert reopened.snapshot()["transition_count"] == 2
    assert reopened.snapshot()["event_count"] == 4


def test_camera_and_notification_health_survive_reopen(tmp_path: Path) -> None:
    class BrokenCamera(FakeDisplayAdapter):
        def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
            if camera_id == "wrist":
                raise ValueError("camera unavailable")
            return super().frame(camera_id, observation)

    def broken_notify(_event) -> None:
        raise OSError("hub unavailable")

    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=BrokenCamera((_camera(), _camera("wrist", "wrist"))),
        profile=RecordingProfile("all"),
        notify=broken_notify,
    )
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    recorder.close()

    reopened = _recorder(tmp_path)
    issues = reopened.snapshot()["health"]["issues"]
    assert reopened.snapshot()["health"]["status"] == "degraded"
    assert "camera extraction failed" in issues
    assert "notification delivery failed" in issues


def test_recovery_detects_missing_referenced_segment(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    (tmp_path / row["frames"][0]["relative_path"]).unlink()
    recorder.close()

    reopened = _recorder(tmp_path)

    assert reopened.snapshot()["health"]["status"] == "degraded"
    assert (
        "retained camera frame unavailable" in reopened.snapshot()["health"]["issues"]
    )


def test_recovery_infers_missing_retained_camera_frame_without_health_journal(
    tmp_path: Path,
) -> None:
    class BrokenCamera(FakeDisplayAdapter):
        def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
            if camera_id == "wrist":
                raise ValueError("camera unavailable")
            return super().frame(camera_id, observation)

    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=BrokenCamera((_camera(), _camera("wrist", "wrist"))),
        profile=RecordingProfile("all"),
    )
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    recorder.close()
    (tmp_path / "health.jsonl").unlink()

    reopened = _recorder(tmp_path)

    assert (
        "retained camera frame unavailable" in reopened.snapshot()["health"]["issues"]
    )


def test_record_event_failure_does_not_consume_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_append = media.append_jsonl

    def fail_events(path: Path, value) -> None:
        if path.name == "events.jsonl":
            raise OSError("event disk failed")
        real_append(path, value)

    monkeypatch.setattr(media, "append_jsonl", fail_events)
    with pytest.raises(OSError, match="event disk failed"):
        recorder.record("not_durable")
    assert recorder.snapshot()["event_count"] == 0

    monkeypatch.setattr(media, "append_jsonl", real_append)
    assert recorder.record("durable")["sequence"] == 1
    recorder.close()

    reopened = _recorder(tmp_path)
    assert any(
        "event journal write failed" in issue
        for issue in reopened.snapshot()["health"]["issues"]
    )


def test_recovery_rejects_durable_event_sequence_gap(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record("one")
    recorder.record("two")
    recorder.close()
    events_path = tmp_path / "events.jsonl"
    rows, _ = media.read_complete_jsonl(events_path)
    rows[1]["sequence"] = 3
    media.replace_jsonl(events_path, rows)

    with pytest.raises(ValueError, match="contiguous"):
        _recorder(tmp_path)


def test_health_journal_failure_does_not_recurse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_append = media.append_jsonl

    def fail_health(path: Path, value) -> None:
        if path.name == "health.jsonl":
            raise OSError("health disk failed")
        real_append(path, value)

    monkeypatch.setattr(media, "append_jsonl", fail_health)
    recorder.notify = lambda _event: (_ for _ in ()).throw(OSError("hub failed"))

    recorder.record("durable_event")

    assert recorder.snapshot()["health"]["status"] == "degraded"
    assert len(recorder.snapshot()["health"]["issues"]) == 1


def test_recovery_synthesizes_transition_event_without_sequence_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_append = media.append_jsonl

    def fail_events(path: Path, value) -> None:
        if path.name == "events.jsonl":
            raise OSError("event disk failed")
        real_append(path, value)

    monkeypatch.setattr(media, "append_jsonl", fail_events)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id="operation-1",
        execution_id="execution-1",
    )
    assert recorder.snapshot()["event_count"] == 0
    recorder.close()
    monkeypatch.setattr(media, "append_jsonl", real_append)

    recovered = _recorder(tmp_path)

    assert recovered.snapshot()["event_count"] == 1
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["sequence"] == 1
    assert event["step_id"] == 0
    assert event["operation_id"] == "operation-1"
    assert event["execution_id"] == "execution-1"
    assert recovered.record("after_repair")["sequence"] == 2


def test_finalized_recovery_reports_missing_event_without_mutating_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)
    real_append = media.append_jsonl

    def fail_events(path: Path, value) -> None:
        if path.name == "events.jsonl":
            raise OSError("event disk failed")
        real_append(path, value)

    monkeypatch.setattr(media, "append_jsonl", fail_events)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    monkeypatch.setattr(media, "append_jsonl", real_append)
    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )
    recorder.finalize(_result(tmp_path))
    events_path = tmp_path / "events.jsonl"
    events_before = events_path.read_bytes() if events_path.exists() else None
    artifacts_before = (tmp_path / "artifacts.json").read_bytes()

    reopened = _recorder(tmp_path)

    assert reopened.snapshot()["event_count"] == 1
    assert json.loads(events_before)["event_type"] == "episode_video_available"
    assert "transition event unavailable" in reopened.snapshot()["health"]["issues"]
    events_after = events_path.read_bytes() if events_path.exists() else None
    assert events_after == events_before
    assert (tmp_path / "artifacts.json").read_bytes() == artifacts_before


def test_new_journal_directories_and_file_entries_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = media.os.fsync

    def tracking_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(media.os, "fsync", tracking_fsync)

    media.append_jsonl(tmp_path / "one/two/events.jsonl", {"event": "created"})

    assert len(calls) >= 4


def test_atomic_replace_fsyncs_new_ancestor_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = media.os.fsync

    def tracking_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(media.os, "fsync", tracking_fsync)

    media.atomic_write_bytes(tmp_path / "one/two/value.bin", b"value")

    assert len(calls) >= 4


@pytest.mark.parametrize(
    "camera_id", ("/absolute", "../escape", "nested/camera", "相机/腕")
)
def test_camera_storage_component_is_one_safe_stable_path_part(camera_id: str) -> None:
    component = media.camera_storage_component(camera_id)

    assert component == media.camera_storage_component(camera_id)
    assert component.startswith("camera-sha256-")
    assert len(component) == 78
    assert all(character in "0123456789abcdef" for character in component[14:])
    assert "/" not in component
    assert component not in {".", ".."}


def test_512_byte_camera_id_writes_live_and_retained_frames(tmp_path: Path) -> None:
    camera_id = "视" * 170 + "ab"
    camera = _camera(camera_id)
    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=FakeDisplayAdapter((camera,)),
        profile=RecordingProfile("all"),
    )
    result = _step()
    result.observation[camera_id] = np.zeros((2, 2, 3), dtype=np.uint8)

    recorder.record_transition(
        action=np.zeros(1),
        result=result,
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    component = media.camera_storage_component(camera_id)
    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert (tmp_path / "live" / f"{component}.png").is_file()
    assert (tmp_path / row["frames"][0]["relative_path"]).is_file()


def test_persisted_health_uses_stable_categories_for_sensitive_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "/Users/alice/private/model-key"

    class BrokenDisplay(FakeDisplayAdapter):
        def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
            if camera_id == secret:
                raise OSError(36, "File name too long", secret)
            return super().frame(camera_id, observation)

    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=BrokenDisplay((_camera(secret), _camera("safe", None))),
        profile=RecordingProfile("all"),
    )
    result = _step()
    result.observation["safe"] = np.zeros((2, 2, 3), dtype=np.uint8)

    def fail_frame_write(_path: Path, _value: bytes) -> None:
        raise OSError(36, "File name too long", secret)

    monkeypatch.setattr(media, "atomic_write_bytes", fail_frame_write)
    recorder.record_transition(
        action=np.zeros(1),
        result=result,
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    health = json.dumps(recorder.snapshot()["health"])
    journal = (tmp_path / "health.jsonl").read_text()
    assert secret not in health + journal
    assert "File name too long" not in health + journal
    assert set(recorder.snapshot()["health"]["issues"]) == {
        "camera extraction failed",
        "live camera write failed",
        "camera segment write failed",
    }


def test_reopen_sanitizes_legacy_health_journal_and_refinalized_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "/Users/alice/private/model-key"
    recorder = _recorder(tmp_path)
    recorder.close()
    media.append_jsonl(
        tmp_path / "health.jsonl",
        {
            "evaluation_id": "evaluation-1",
            "case_id": "case-1",
            "attempt_id": "attempt-1",
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": f"camera {secret} failed: [Errno 36] File name too long",
        },
    )
    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )

    reopened = _recorder(tmp_path)
    snapshot = json.dumps(reopened.snapshot())
    manifest = json.dumps(reopened.finalize(_result(tmp_path)))
    journal = (tmp_path / "health.jsonl").read_text()
    finalized = (tmp_path / "artifacts.json").read_text()

    assert secret in journal
    assert secret not in snapshot + manifest + finalized
    assert "File name too long" not in snapshot + manifest + finalized
    assert "recording artifact issue" in reopened.snapshot()["health"]["issues"]


@pytest.mark.parametrize(
    ("stored_status", "expected_status"),
    [
        ("complete", "degraded"),
        ("degraded", "degraded"),
        ("partial", "partial"),
        ("unavailable", "unavailable"),
    ],
)
def test_reopen_sanitizes_legacy_final_manifest_health_and_joins_severity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_status: str,
    expected_status: str,
) -> None:
    secret = "/Users/alice/private/model-key"
    recorder = _recorder(tmp_path)

    def write_parquet(path: Path, _records) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PAR1fixture")

    monkeypatch.setattr(media, "write_parquet", write_parquet)
    result = _result(tmp_path)
    recorder.finalize(result)
    path = tmp_path / "artifacts.json"
    payload = json.loads(path.read_text())
    payload["health"] = {"status": stored_status, "issues": [secret]}
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    path.chmod(0o400)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("finalized reopen must not write")

    monkeypatch.setattr(media, "append_jsonl", unexpected_write)
    monkeypatch.setattr(media, "replace_jsonl", unexpected_write)
    monkeypatch.setattr(media, "atomic_write_json", unexpected_write)

    reopened = _recorder(tmp_path)
    manifest = reopened.finalize(result)

    assert manifest["health"]["status"] == expected_status
    assert "recording artifact issue" in manifest["health"]["issues"]
    assert path.read_bytes() == original


def test_camera_storage_components_are_case_fold_injective(tmp_path: Path) -> None:
    camera_ids = ("aaa", "aaG")
    components = tuple(media.camera_storage_component(item) for item in camera_ids)
    assert len({item.casefold() for item in components}) == 2
    cameras = (_camera("aaa"), _camera("aaG", None))
    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=FakeDisplayAdapter(cameras),
        profile=RecordingProfile("all"),
    )
    result = _step()
    result.observation["aaa"] = np.zeros((2, 2, 3), dtype=np.uint8)
    result.observation["aaG"] = np.full((2, 2, 3), 255, dtype=np.uint8)

    recorder.record_transition(
        action=np.zeros(1),
        result=result,
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    paths = {frame["camera_id"]: frame["relative_path"] for frame in row["frames"]}
    assert paths["aaa"].casefold() != paths["aaG"].casefold()
    assert (tmp_path / paths["aaa"]).read_bytes() != (
        tmp_path / paths["aaG"]
    ).read_bytes()


@pytest.mark.parametrize("camera_id", ("/absolute", "../../../escape"))
def test_native_camera_id_cannot_escape_recording_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, camera_id: str
) -> None:
    real_atomic_write = media.atomic_write_bytes

    def guarded_write(path: Path, value: bytes) -> None:
        assert path.resolve().is_relative_to(tmp_path.resolve())
        real_atomic_write(path, value)

    monkeypatch.setattr(media, "atomic_write_bytes", guarded_write)
    camera = _camera(camera_id)
    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=FakeDisplayAdapter((camera,)),
        profile=RecordingProfile("all"),
    )
    result = _step()
    result.observation[camera_id] = np.zeros((2, 2, 3), dtype=np.uint8)

    recorder.record_transition(
        action=np.zeros(1),
        result=result,
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert row["frames"][0]["camera_id"] == camera_id
    assert (tmp_path / row["frames"][0]["relative_path"]).exists()


def test_safe_camera_paths_keep_native_id_in_manifest_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    camera_id = "相机/腕"
    component = media.camera_storage_component(camera_id)
    camera = _camera(camera_id)
    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=FakeDisplayAdapter((camera,)),
        profile=RecordingProfile("all"),
    )
    result = _step()
    result.observation[camera_id] = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder.record_transition(
        action=np.zeros(1),
        result=result,
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )

    manifest = recorder.finalize(_result(tmp_path))
    index = json.loads(
        (tmp_path / "trajectory/cameras" / component / "index.json").read_text()
    )

    assert manifest["cameras"][0]["camera_id"] == camera_id
    assert index["camera"]["camera_id"] == camera_id
    assert all(
        component in artifact["relative_path"]
        for artifact in manifest["artifacts"]
        if artifact["kind"].startswith("camera_")
    )


def test_finalized_manifest_rejects_mismatched_embedded_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        media,
        "write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )
    recorder = _recorder(tmp_path)
    recorder.finalize(_result(tmp_path))
    artifacts_path = tmp_path / "artifacts.json"
    manifest = json.loads(artifacts_path.read_text())
    manifest["episode_result"]["case_id"] = "different-case"
    media.atomic_write_json(artifacts_path, manifest)

    with pytest.raises(ValueError, match="result identity"):
        _recorder(tmp_path)
