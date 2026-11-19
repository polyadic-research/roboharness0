from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rpent.recording.media as media
from rpent.episode.contracts import StepResult
from rpent.evaluation.results import EpisodeResult
from rpent.recording.contracts import CameraDescriptor, RecordingProfile
from rpent.recording.display import png_bytes
from rpent.recording.recorder import CaseRecorder


class FakeDisplayAdapter:
    def __init__(self, descriptors: tuple[CameraDescriptor, ...]) -> None:
        self.descriptors = descriptors

    def frames(self, observation: dict[str, object]) -> dict[str, bytes]:
        return {
            descriptor.camera_id: self.frame(descriptor.camera_id, observation)
            for descriptor in self.descriptors
        }

    def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
        return png_bytes(observation[camera_id])


def _camera(camera_id: str = "agent", role: str | None = "primary") -> CameraDescriptor:
    return CameraDescriptor(camera_id, camera_id.title(), role, 2, 2, 3, 20.0)


def _live(root: Path, camera_id: str) -> Path:
    return root / "live" / f"{media.camera_storage_component(camera_id)}.png"


def _camera_root(root: Path, camera_id: str) -> Path:
    return root / "trajectory/cameras" / media.camera_storage_component(camera_id)


def _step(step_id: int = 0) -> StepResult:
    return StepResult(
        episode_id="episode-1",
        step_id=step_id,
        observation={
            "state": np.array([1.0, 2.0], dtype=np.float32),
            "agent": np.full((2, 2, 3), step_id + 1, dtype=np.uint8),
            "wrist": np.full((2, 2, 3), step_id + 2, dtype=np.uint8),
        },
        reward=0.25,
        success=False,
        terminated=False,
        truncated=False,
    )


def _result(root: Path) -> EpisodeResult:
    return EpisodeResult(
        evaluation_id="evaluation-1",
        manifest_id="manifest-1",
        case_id="case-1",
        attempt_id="attempt-1",
        status="terminal_failure",
        environment="test",
        preset_name="test",
        task_name="task",
        task_index=0,
        seed=1,
        episode_index=0,
        worker_pid=os.getpid(),
        worker_exit_code=0,
        started_at="2026-08-06T00:00:00+00:00",
        finished_at="2026-08-06T00:00:01+00:00",
        artifact_dir=str(root),
        simulator_success=False,
        contract_success=False,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id="repo",
        checkpoint_revision="revision",
        binding_name="binding",
    )


def _recorder(
    root: Path,
    *,
    profile: RecordingProfile = RecordingProfile("all"),
    notify=None,
) -> CaseRecorder:
    return CaseRecorder(
        root=root,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=FakeDisplayAdapter((_camera(), _camera("wrist", "wrist"))),
        profile=profile,
        notify=notify,
    )


def test_transition_is_durable_before_notification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced: list[int] = []
    real_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracking_fsync)

    def notify(event) -> None:
        transition_rows = (
            (tmp_path / "trajectory/transitions.jsonl").read_text().splitlines()
        )
        event_rows = (tmp_path / "events.jsonl").read_text().splitlines()
        assert len(transition_rows) == len(event_rows) == 1
        assert len(fsynced) >= 2
        assert event.sequence == 1

    recorder = _recorder(tmp_path, notify=notify)
    event = recorder.record_transition(
        action=np.zeros(7, dtype=np.float32),
        result=_step(),
        timestamp_s=0.05,
        operation_id="operation-1",
        execution_id="execution-1",
    )

    assert event.sequence == 1
    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert row["state"] == {"state": [1.0, 2.0]}
    assert [frame["camera_id"] for frame in row["frames"]] == ["agent", "wrist"]
    assert _live(tmp_path, "agent").read_bytes().startswith(b"\x89PNG")


def test_record_is_planning_recorder_compatible_and_json_strict(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)

    payload = recorder.record("planning_started", values=np.array([1, 2]))

    assert payload["event_type"] == "planning_started"
    assert payload["payload"] == {"values": [1, 2]}
    with pytest.raises(ValueError, match="finite"):
        recorder.record("bad", value=float("nan"))


def test_transition_rejects_wrong_identity_and_non_monotonic_steps(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(1),
        timestamp_s=0.1,
        operation_id=None,
        execution_id=None,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.record_transition(
            action=np.zeros(1),
            result=_step(1),
            timestamp_s=0.2,
            operation_id=None,
            execution_id=None,
        )
    with pytest.raises(ValueError, match="episode_id"):
        recorder.record_transition(
            action=np.zeros(1),
            result=StepResult("other", 2, {}, 0.0, False, False, False),
            timestamp_s=0.2,
            operation_id=None,
            execution_id=None,
        )


def test_profile_controls_segments_but_not_live_frames(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, profile=RecordingProfile("primary"))
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    assert _live(tmp_path, "agent").exists()
    assert _live(tmp_path, "wrist").exists()
    assert (_camera_root(tmp_path, "agent") / "segments/00000000.png").exists()
    assert not (_camera_root(tmp_path, "wrist") / "segments").exists()


def test_one_bad_camera_does_not_drop_healthy_camera(tmp_path: Path) -> None:
    class PartlyBrokenDisplay(FakeDisplayAdapter):
        def frames(self, observation: dict[str, object]) -> dict[str, bytes]:
            raise AssertionError("recorder must isolate extraction per camera")

        def frame(self, camera_id: str, observation: dict[str, object]) -> bytes:
            if camera_id == "wrist":
                raise ValueError("camera missing")
            return super().frame(camera_id, observation)

    recorder = CaseRecorder(
        root=tmp_path,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=PartlyBrokenDisplay((_camera(), _camera("wrist", "wrist"))),
        profile=RecordingProfile("all"),
    )

    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    row = json.loads((tmp_path / "trajectory/transitions.jsonl").read_text())
    assert [frame["camera_id"] for frame in row["frames"]] == ["agent"]
    assert _live(tmp_path, "agent").exists()
    assert not _live(tmp_path, "wrist").exists()
    assert (_camera_root(tmp_path, "agent") / "segments/00000000.png").exists()
    assert recorder.snapshot()["health"]["status"] == "degraded"
    assert "camera extraction failed" in recorder.snapshot()["health"]["issues"]


def test_notification_failure_degrades_health_without_aborting(tmp_path: Path) -> None:
    def broken_notify(_event) -> None:
        raise OSError("hub offline")

    recorder = _recorder(tmp_path, notify=broken_notify)
    event = recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    assert event.sequence == 1
    snapshot = recorder.snapshot()
    assert snapshot["health"]["status"] == "degraded"
    assert any("notification" in issue for issue in snapshot["health"]["issues"])


def test_finalize_writes_manifest_indexes_and_removes_non_retained_live(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path, profile=RecordingProfile("primary"))
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    manifest = recorder.finalize(_result(tmp_path))

    assert manifest == json.loads((tmp_path / "artifacts.json").read_text())
    assert (_camera_root(tmp_path, "agent") / "index.json").exists()
    assert (_camera_root(tmp_path, "agent") / "video.mp4").exists()
    assert _live(tmp_path, "agent").exists()
    assert not _live(tmp_path, "wrist").exists()
    if (tmp_path / "trajectory/transitions.parquet").exists():
        assert manifest["health"]["status"] == "complete"
    else:
        assert manifest["health"]["status"] == "degraded"
        assert any("Parquet" in issue for issue in manifest["health"]["issues"])

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event_type"] == "episode_video_available"
        and event["payload"]["artifact_id"]
        == f"{media.camera_storage_component('agent')}-video"
        for event in events
    )


def test_finalize_declares_nonempty_action_clips(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, profile=RecordingProfile("none"))
    clips = tmp_path / "action_videos"
    clips.mkdir()
    (clips / "step_03_move_gripper.mp4").write_bytes(b"video")
    (clips / "step_04_empty.mp4").touch()

    manifest = recorder.finalize(_result(tmp_path))

    assert {
        (item.get("artifact_id"), item["kind"], item["relative_path"])
        for item in manifest["artifacts"]
    } >= {
        (
            "action-step-3",
            "action_clip",
            "action_videos/step_03_move_gripper.mp4",
        )
    }
    assert all(
        item.get("artifact_id") != "action-step-4" for item in manifest["artifacts"]
    )


def test_none_profile_is_live_only_until_terminal(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, profile=RecordingProfile("none"))
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    assert _live(tmp_path, "agent").exists()
    assert _live(tmp_path, "wrist").exists()
    assert not (tmp_path / "trajectory/cameras").exists()

    manifest = recorder.finalize(_result(tmp_path))

    assert not _live(tmp_path, "agent").exists()
    assert not _live(tmp_path, "wrist").exists()
    assert not (tmp_path / "trajectory/cameras").exists()
    assert not any(
        artifact["kind"].startswith("camera_") for artifact in manifest["artifacts"]
    )


def test_selected_profile_retains_exact_camera_set(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path, profile=RecordingProfile("selected", ("wrist",)))
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )

    assert _live(tmp_path, "agent").exists()
    assert _live(tmp_path, "wrist").exists()
    assert not _camera_root(tmp_path, "agent").exists()
    assert (_camera_root(tmp_path, "wrist") / "segments/00000000.png").exists()

    recorder.finalize(_result(tmp_path))

    assert not _live(tmp_path, "agent").exists()
    assert _live(tmp_path, "wrist").exists()
    assert not (_camera_root(tmp_path, "agent") / "video.mp4").exists()
    assert (_camera_root(tmp_path, "wrist") / "video.mp4").exists()


def test_only_one_writer_can_own_an_attempt(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already has an active writer"):
            _recorder(tmp_path)
    finally:
        recorder.close()


def test_transition_event_exposes_timeline_references(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)

    event = recorder.record_transition(
        action=np.zeros(1),
        result=_step(4),
        timestamp_s=0.2,
        operation_id="operation-4",
        execution_id="execution-4",
    )

    assert event.step_id == 4
    assert event.operation_id == "operation-4"
    assert event.execution_id == "execution-4"
    assert event.payload["transition"]["step_id"] == 4


def test_finalize_is_idempotent_and_reopened_attempt_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def successful_parquet(path: Path, _records) -> None:
        path.write_bytes(b"PAR1fixture")

    monkeypatch.setattr("rpent.recording.media.write_parquet", successful_parquet)
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(),
        timestamp_s=0.0,
        operation_id=None,
        execution_id=None,
    )
    result = _result(tmp_path)

    manifest = recorder.finalize(result)

    assert recorder.finalize(result) is manifest
    discovery = tmp_path / "recording.json"
    sentinel = b'{"sentinel":"finalized-read-only"}\n'
    discovery.write_bytes(sentinel)
    discovery.chmod(0o400)
    reopened = _recorder(tmp_path)
    assert discovery.read_bytes() == sentinel
    assert reopened.snapshot() == manifest | {
        "episode_result": manifest["episode_result"],
        "artifacts": manifest["artifacts"],
    }
    with pytest.raises(RuntimeError, match="finalized"):
        reopened.record("too_late")
    with pytest.raises(RuntimeError, match="finalized"):
        reopened.record_transition(
            action=np.zeros(1),
            result=_step(1),
            timestamp_s=0.1,
            operation_id=None,
            execution_id=None,
        )
    assert reopened.finalize(result) == manifest
    with pytest.raises(ValueError, match="result"):
        reopened.finalize(replace(result, finished_at="2026-08-06T00:00:02+00:00"))


def test_finalized_attempt_rejects_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rpent.recording.media.write_parquet",
        lambda path, _records: path.write_bytes(b"PAR1fixture"),
    )
    recorder = _recorder(tmp_path)
    recorder.finalize(_result(tmp_path))

    with pytest.raises(ValueError, match="identity"):
        CaseRecorder(
            root=tmp_path,
            evaluation_id="other-evaluation",
            case_id="case-1",
            attempt_id="attempt-1",
            episode_id="episode-1",
            display=FakeDisplayAdapter((_camera(), _camera("wrist", "wrist"))),
            profile=RecordingProfile("all"),
        )
