from __future__ import annotations

from pathlib import Path

import pytest

from rpent.recording.live_events import (
    record_action_clip_available,
    record_episode_video_available,
    record_reasoning_delta,
    record_tool_call,
    record_tool_result,
    record_usage_updated,
)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record(self, event: str, **payload: object) -> dict[str, object]:
        self.events.append((event, payload))
        return {"event_type": event, **payload}


def test_records_canonical_reasoning_and_usage_events() -> None:
    recorder = Recorder()

    record_reasoning_delta(recorder, "checking the gripper")
    record_usage_updated(
        recorder,
        input_tokens=120,
        output_tokens=34,
        cached_input_tokens=20,
        reasoning_tokens=8,
        tool_calls=2,
    )

    assert recorder.events == [
        ("reasoning_delta", {"text": "checking the gripper"}),
        (
            "usage_updated",
            {
                "input_tokens": 120,
                "output_tokens": 34,
                "cached_input_tokens": 20,
                "reasoning_tokens": 8,
                "tool_calls": 2,
            },
        ),
    ]


def test_tool_events_are_json_safe_and_bounded() -> None:
    recorder = Recorder()

    record_tool_call(
        recorder,
        tool="move_gripper",
        arguments={"target": Path("/tmp/private"), "samples": list(range(5000))},
        call_id="call-1",
    )
    record_tool_result(
        recorder,
        tool="move_gripper",
        result={"ok": True, "secret_path": Path("/tmp/private")},
        call_id="call-1",
    )

    call = recorder.events[0]
    result = recorder.events[1]
    assert call[0] == "tool_call"
    assert call[1]["tool"] == "move_gripper"
    assert call[1]["call_id"] == "call-1"
    assert len(str(call[1]["arguments"])) < 20_000
    assert "/tmp/private" not in str(call[1])
    assert result == (
        "tool_result",
        {
            "tool": "move_gripper",
            "result": {"ok": True, "secret_path": "[path]"},
            "is_error": False,
            "call_id": "call-1",
        },
    )


def test_media_events_only_publish_artifact_ids() -> None:
    recorder = Recorder()

    record_action_clip_available(recorder, "action-step-3", step=3)
    record_episode_video_available(recorder, "primary-video")

    assert recorder.events == [
        ("action_clip_available", {"artifact_id": "action-step-3", "step": 3}),
        ("episode_video_available", {"artifact_id": "primary-video"}),
    ]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_usage_rejects_invalid_counters(value: object) -> None:
    with pytest.raises(ValueError):
        record_usage_updated(Recorder(), input_tokens=value)  # type: ignore[arg-type]


def test_rejects_unbounded_reasoning_and_invalid_artifact_ids() -> None:
    with pytest.raises(ValueError):
        record_reasoning_delta(Recorder(), "x" * 32_001)
    with pytest.raises(ValueError):
        record_episode_video_available(Recorder(), "../../episode.mp4")


def test_none_recorder_is_a_noop() -> None:
    record_reasoning_delta(None, "safe")
    record_usage_updated(None, input_tokens=1)
