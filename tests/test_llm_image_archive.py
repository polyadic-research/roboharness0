"""Archive the actual model image payload, after transport history selection."""

import asyncio
import base64
import hashlib
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from rpent.recording.llm_images import LLMImageArchive


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(stream, format="PNG")
    return stream.getvalue()


def body(images):
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Inspect the current scene."},
            {"role": "tool", "tool_call_id": "observe-7", "content": '{"step":7}'},
            {"role": "user", "content": [
                {"type": "text", "text": "\n".join(f"Image {i + 1} camera: {c}" for i, c in enumerate(cameras))},
                *[{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}} for raw in images],
            ]},
        ],
    }


def test_wire_images_are_exact_and_indexed_on_every_request(tmp_path):
    archive = LLMImageArchive(tmp_path)
    pixels = [png("red"), png("green"), png("blue")]
    for _ in range(2):
        request = httpx.Request("POST", "http://gateway.invalid/v1/chat/completions",
                                json=body(pixels), headers={"Authorization": "Bearer do-not-save", "X-Codex-Session-ID": "session-1"})
        asyncio.run(archive.on_request(request))
        asyncio.run(archive.on_response(httpx.Response(200, request=request)))
    rows = [json.loads(line) for line in (tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["request_id"] != rows[1]["request_id"]
    assert len(list((tmp_path / "llm_inputs/images").iterdir())) == 3
    assert [item["camera_id"] for item in rows[0]["images"]] == [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    ]
    for row in rows:
        assert row["capture_stage"] == "http_request_body"
        assert row["session_id"] == "session-1"
        for image, raw in zip(row["images"], pixels, strict=True):
            assert (tmp_path / image["relative_path"]).read_bytes() == raw
            assert image["sha256"] == hashlib.sha256(raw).hexdigest()
            assert image["width"] == image["height"] == 256
            assert image["observation_step"] == 7
        snapshot = (tmp_path / row["request_file"]).read_text()
        assert "data:image" not in snapshot and "do-not-save" not in snapshot
        assert json.loads(snapshot)["input"]["messages"][0]["content"] == "Inspect the current scene."
    statuses = [json.loads(s) for s in (tmp_path / "llm_inputs/responses.jsonl").read_text().splitlines()]
    assert len(statuses) == 2 and all(s["status_code"] == 200 for s in statuses)


def test_cleanup_and_non_image_requests_do_not_save_extra_images(tmp_path):
    archive = LLMImageArchive(tmp_path)
    asyncio.run(archive.on_request(httpx.Request("DELETE", "http://gateway.invalid/v1/codex/sessions/a")))
    assert not (tmp_path / "llm_inputs").exists()
    request = httpx.Request("POST", "http://gateway.invalid/v1/chat/completions", json=body([]))
    asyncio.run(archive.on_request(request))
    rows = [json.loads(s) for s in (tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["images"] == []
    assert not (tmp_path / "llm_inputs/images").exists()


def test_model_input_archives_only_images_left_in_selected_history(tmp_path):
    archive = LLMImageArchive(tmp_path)
    raw = png("yellow")
    messages = [ModelRequest(parts=[UserPromptPart(content=[
        'Current state {"step":12}\nImage 1 camera: robot0_agentview_left',
        BinaryContent(data=raw, media_type="image/png"),
    ])])]
    archive.capture_model_request(messages, model="test-model")
    row = json.loads((tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()[0])
    assert row["capture_stage"] == "provider_model_input"
    assert len(row["images"]) == 1
    assert row["images"][0]["camera_id"] == "robot0_agentview_left"
    assert row["images"][0]["observation_step"] == 12
    assert (tmp_path / row["images"][0]["relative_path"]).read_bytes() == raw


def test_model_input_maps_tool_return_labels_to_image_attachment(tmp_path):
    archive = LLMImageArchive(tmp_path)
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    pixels = [png(c) for c in ["red", "green", "blue"]]
    # Pydantic splits ToolReturn.return_value and ToolReturn.content into
    # separate parts; provider input must still retain their correspondence.
    messages = [ModelRequest(parts=[
        ToolReturnPart(tool_name="observe", tool_call_id="call-1", content='{"step":9}\n' +
                       "\n".join(f"Image {i + 1} camera: {c}" for i, c in enumerate(cameras))),
        UserPromptPart(content=[BinaryContent(data=p, media_type="image/png") for p in pixels]),
    ])]
    archive.capture_model_request(messages, model="test-model")
    row = json.loads((tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()[0])
    assert [p["camera_id"] for p in row["images"]] == cameras
    assert all(p["observation_step"] == 9 for p in row["images"])


def test_archive_failure_prevents_unlogged_model_request(tmp_path, monkeypatch):
    archive = LLMImageArchive(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(archive, "_write_image", fail)
    request = httpx.Request("POST", "http://gateway.invalid/v1/chat/completions", json=body([png("black")]))
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(archive.on_request(request))
    assert not (tmp_path / "llm_inputs/requests.jsonl").exists()


def test_pruning_keeps_complete_current_three_view_observation(tmp_path):
    from rpent.runtime.api import _prune_history_images

    old = [png("black") for _ in range(3)]
    current = [png(c) + b"padding" * 300_000 for c in ["red", "green", "blue"]]
    labels = "\n".join(f"Image {i + 1} camera: {camera}" for i, camera in enumerate(
        ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]))
    history = [ModelRequest(parts=[UserPromptPart(content=[labels, *[
        BinaryContent(data=raw, media_type="image/png") for raw in group]])]) for group in [old, current]]
    selected = _prune_history_images(history)
    archive = LLMImageArchive(tmp_path)
    archive.capture_model_request(selected, model="test-model")
    row = json.loads((tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()[0])
    assert len(row["images"]) == 3
    assert [image["camera_id"] for image in row["images"]] == [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    ]
    assert [image["sha256"] for image in row["images"]] == [hashlib.sha256(raw).hexdigest() for raw in current]




def test_pure_transition_recording_does_not_retain_unseen_camera_frames(tmp_path):
    import numpy as np
    from test_case_recorder import FakeDisplayAdapter, _camera, _step

    from rpent.recording import CaseRecorder
    from rpent.recording.contracts import RecordingProfile

    recorder = CaseRecorder(root=tmp_path, evaluation_id="eval", case_id="case", attempt_id="attempt",
                            episode_id="episode-1", display=FakeDisplayAdapter((_camera(),)), profile=RecordingProfile("none"),
                            llm_inputs_only=True)
    recorder.record_transition(action=np.zeros(7, dtype=np.float32), result=_step(1),
                               timestamp_s=0.05, operation_id=None, execution_id=None)
    assert not (tmp_path / "live").exists()
    assert not (tmp_path / "trajectory/cameras").exists()
    assert (tmp_path / "trajectory/transitions.jsonl").is_file()
    recorder.close()
