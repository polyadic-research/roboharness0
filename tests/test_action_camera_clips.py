from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from rpent.recording import media
from rpent.ui.operation_projection import OperationProjection
from tests.test_case_recorder import _recorder, _result, _step


def test_action_camera_storage_identity_fixed_vector() -> None:
    assert media.operation_storage_component("operation-1") == (
        "operation-sha256-"
        "0430270410180e96528795b8ac4682648dd43971c541bb6a2040a84f506b618e"
    )
    assert media.action_camera_artifact_id("operation-1", "front") == (
        "action-camera-sha256-"
        "f71d764c899da0eff4bae9e71f740d6ebeb0ace21779d2216861c04ddef25404"
    )


def test_action_camera_storage_identity_accepts_unicode_and_long_ids() -> None:
    operation_id = "操作/🤖/" + "x" * 4096
    camera_id = "前方/📷/" + "y" * 4096

    operation_component = media.operation_storage_component(operation_id)
    artifact_id = media.action_camera_artifact_id(operation_id, camera_id)

    assert re.fullmatch(r"operation-sha256-[0-9a-f]{64}", operation_component)
    assert re.fullmatch(r"action-camera-sha256-[0-9a-f]{64}", artifact_id)
    assert "/" not in operation_component
    assert operation_component == media.operation_storage_component(operation_id)
    assert artifact_id == media.action_camera_artifact_id(operation_id, camera_id)


def _png(path: Path, value: int) -> Path:
    imageio.imwrite(path, np.full((4, 4, 3), value, dtype=np.uint8))
    return path


def test_finalize_validated_video_decodes_exact_expected_frame_count(
    tmp_path: Path,
) -> None:
    frames = (_png(tmp_path / "one.png", 1), _png(tmp_path / "two.png", 2))
    target = tmp_path / "clips" / "clip.mp4"

    media.finalize_validated_video(target, frames, fps=12.0, expected_frames=2)

    assert target.stat().st_size > 0
    assert media.video_frame_count(target) == 2
    assert not tuple(target.parent.glob(f".{target.name}.*.mp4"))


def test_finalize_validated_video_rejects_wrong_decoded_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = (_png(tmp_path / "one.png", 1), _png(tmp_path / "two.png", 2))
    target = tmp_path / "clip.mp4"
    monkeypatch.setattr(media, "video_frame_count", lambda _path: 1)

    with pytest.raises(ValueError, match="decoded frame count"):
        media.finalize_validated_video(target, frames, fps=12.0, expected_frames=2)

    assert not target.exists()
    assert not tuple(tmp_path.glob(f".{target.name}.*.mp4"))


def test_recorder_finalization_publishes_one_clip_per_operation_camera(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    for step_id in (0, 1):
        recorder.record_transition(
            action=np.zeros(1),
            result=_step(step_id),
            timestamp_s=step_id / 20,
            operation_id="operation-1",
            execution_id="execution-1",
        )

    manifest = recorder.finalize(_result(tmp_path))

    clips = [
        item for item in manifest["artifacts"]
        if item["kind"] == "action_camera_clip"
    ]
    assert {item["artifact_id"] for item in clips} == {
        media.action_camera_artifact_id("operation-1", "agent"),
        media.action_camera_artifact_id("operation-1", "wrist"),
    }
    events, incomplete = media.read_complete_jsonl(tmp_path / "events.jsonl")
    assert not incomplete
    available = [
        event for event in events
        if event["event_type"] == "action_camera_clip_available"
    ]
    assert [event["operation_id"] for event in available] == [
        "operation-1", "operation-1"
    ]
    assert {event["payload"]["camera_id"] for event in available} == {
        "agent", "wrist"
    }

    reopened = _recorder(tmp_path)
    assert reopened.finalize(_result(tmp_path))["artifacts"] == manifest["artifacts"]


def test_operation_clip_submission_does_not_wait_for_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(0),
        timestamp_s=0,
        operation_id="operation-1",
        execution_id="execution-1",
    )

    def blocked(*_args, **_kwargs) -> None:
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(media, "finalize_validated_video", blocked)
    recorder.submit_operation_clips("operation-1", (0,))

    assert entered.wait(5)
    release.set()
    recorder.finalize(_result(tmp_path))


def test_operation_clip_worker_publishes_complete_batches_fifo(tmp_path: Path) -> None:
    observed = []
    recorder = _recorder(tmp_path, notify=observed.append)
    for step_id, operation_id in ((0, "operation-1"), (1, "operation-2")):
        recorder.record_transition(
            action=np.zeros(1),
            result=_step(step_id),
            timestamp_s=step_id / 20,
            operation_id=operation_id,
            execution_id=f"execution-{step_id + 1}",
        )
        recorder.submit_operation_clips(operation_id, (step_id,))

    recorder.finalize(_result(tmp_path))

    batches = [
        event.operation_id
        for event in observed
        if event.event_type == "action_camera_clips_ready"
    ]
    assert batches == ["operation-1", "operation-2"]
    for operation_id in batches:
        events = [event for event in observed if event.operation_id == operation_id]
        assert [event.event_type for event in events[-3:]] == [
            "action_camera_clip_available",
            "action_camera_clip_available",
            "action_camera_clips_ready",
        ]


def test_operation_clip_batch_is_declared_before_case_finalization(
    tmp_path: Path,
) -> None:
    ready = threading.Event()

    def notify(event) -> None:
        if event.event_type == "action_camera_clips_ready":
            ready.set()

    recorder = _recorder(tmp_path, notify=notify)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(0),
        timestamp_s=0,
        operation_id="operation-1",
        execution_id="execution-1",
    )
    recorder.submit_operation_clips("operation-1", (0,))

    assert ready.wait(5)
    discovery = json.loads((tmp_path / "recording.json").read_text())
    assert [item["kind"] for item in discovery["artifacts"]] == [
        "action_camera_clip",
        "action_camera_clip",
    ]
    assert not (tmp_path / "artifacts.json").exists()
    recorder.close()


def test_recorder_replaces_truncated_deterministic_regular_target(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(0),
        timestamp_s=0,
        operation_id="operation-1",
        execution_id="execution-1",
    )
    target = (
        tmp_path / "action_videos"
        / media.operation_storage_component("operation-1")
        / f"{media.camera_storage_component('agent')}.mp4"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated")

    manifest = recorder.finalize(_result(tmp_path))

    assert media.video_frame_count(target) == 1
    assert manifest["health"]["status"] == "degraded"


def test_recorder_does_not_replace_symlink_action_camera_target(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.record_transition(
        action=np.zeros(1),
        result=_step(0),
        timestamp_s=0,
        operation_id="operation-1",
        execution_id="execution-1",
    )
    target = (
        tmp_path / "action_videos"
        / media.operation_storage_component("operation-1")
        / f"{media.camera_storage_component('agent')}.mp4"
    )
    target.parent.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside.write_bytes(b"outside")
    target.symlink_to(outside)

    manifest = recorder.finalize(_result(tmp_path))

    assert target.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert media.action_camera_artifact_id("operation-1", "agent") not in {
        item.get("artifact_id") for item in manifest["artifacts"]
    }


def _operation_event(sequence: int, event_type: str, **values: object) -> dict:
    payload = values.pop("payload", {})
    return {
        "sequence": sequence,
        "event_type": event_type,
        "operation_id": values.pop("operation_id", "operation-1"),
        "execution_id": "execution-1",
        "recorded_at": f"2026-08-14T00:00:0{sequence}+00:00",
        "payload": payload,
        **values,
    }


def test_operation_projection_attaches_only_valid_explicit_camera_clips() -> None:
    projection = OperationProjection()
    projection.consume(_operation_event(1, "operation_started", payload={"tool": "act"}))
    projection.consume(_operation_event(2, "transition_recorded", step_id=3))
    projection.consume(_operation_event(
        3,
        "action_camera_clip_available",
        payload={"camera_id": "all", "artifact_id": "artifact-front"},
    ))
    projection.consume(_operation_event(
        4,
        "action_camera_clip_available",
        payload={"camera_id": "all", "artifact_id": "artifact-front"},
    ))
    projection.consume(_operation_event(
        5,
        "action_camera_clips_ready",
        payload={"clips": [{"camera_id": "all", "artifact_id": "artifact-front"}]},
    ))

    summary = projection.summaries(frozenset({"artifact-front"}))[0]

    assert summary.has_physical_transitions is True
    assert [clip.model_dump() for clip in summary.camera_clips] == [
        {"camera_id": "all", "artifact_id": "artifact-front"}
    ]


def test_operation_projection_waits_for_complete_camera_batch() -> None:
    projection = OperationProjection()
    projection.consume(
        _operation_event(1, "operation_started", payload={"tool": "act"})
    )
    projection.consume(_operation_event(2, "transition_recorded", step_id=3))
    projection.consume(
        _operation_event(
            3,
            "action_camera_clip_available",
            payload={"camera_id": "front", "artifact_id": "front-clip"},
        )
    )

    assert projection.summaries(frozenset({"front-clip"}))[0].camera_clips == ()

    projection.consume(
        _operation_event(
            4,
            "action_camera_clips_ready",
            payload={
                "clips": [
                    {"camera_id": "front", "artifact_id": "front-clip"}
                ]
            },
        )
    )
    assert [
        clip.model_dump()
        for clip in projection.summaries(frozenset({"front-clip"}))[0].camera_clips
    ] == [{"camera_id": "front", "artifact_id": "front-clip"}]


def test_operation_projection_keeps_transition_truth_when_clips_are_filtered() -> None:
    projection = OperationProjection()
    projection.consume(_operation_event(1, "operation_started", payload={"tool": "act"}))
    projection.consume(_operation_event(2, "transition_recorded", step_id=3))
    projection.consume(_operation_event(
        3,
        "action_camera_clip_available",
        payload={"camera_id": "front", "artifact_id": "missing"},
    ))

    summary = projection.summaries(frozenset())[0]

    assert summary.has_physical_transitions is True
    assert summary.camera_clips == ()


def test_operation_projection_invalidates_camera_and_artifact_conflicts() -> None:
    projection = OperationProjection()
    projection.consume(_operation_event(1, "operation_started", payload={"tool": "act"}))
    projection.consume(_operation_event(2, "transition_recorded", step_id=3))
    projection.consume(_operation_event(3, "operation_started", operation_id="operation-2", payload={"tool": "act"}))
    projection.consume(_operation_event(4, "transition_recorded", operation_id="operation-2", step_id=4))
    projection.consume(_operation_event(5, "action_camera_clip_available", payload={"camera_id": "front", "artifact_id": "shared"}))
    projection.consume(_operation_event(6, "action_camera_clip_available", operation_id="operation-2", payload={"camera_id": "wrist", "artifact_id": "shared"}))
    projection.consume(_operation_event(7, "action_camera_clip_available", payload={"camera_id": "front", "artifact_id": "other"}))

    summaries = projection.summaries(frozenset({"shared", "other"}))

    assert [summary.camera_clips for summary in summaries] == [(), ()]
