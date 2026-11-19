from __future__ import annotations

import json

import pytest

from rpent.evaluation.startup_log import (
    StartupEvent,
    append_startup_event,
    read_startup_events,
    record_worker_startup_event,
    render_startup_console,
)


def event(name: str, **kwargs) -> StartupEvent:
    return StartupEvent(
        timestamp="2026-08-17T07:01:02+00:00",
        evaluation_id="evaluation-1",
        phase="simulator_starting",
        event=name,
        **kwargs,
    )


def test_startup_journal_preserves_order_and_renders_safe_progress(tmp_path):
    path = tmp_path / "startup.jsonl"
    append_startup_event(path, event("coordinator_constructed"))
    append_startup_event(
        path,
        event(
            "worker_spawned",
            case_id="case-1",
            attempt_id="attempt-1",
            worker_pid=1234,
            duration_ms=18,
        ),
    )

    events = read_startup_events(path)

    assert [item.event for item in events] == [
        "coordinator_constructed",
        "worker_spawned",
    ]
    assert render_startup_console(events).splitlines() == [
        "[07:01:02] simulator_starting · coordinator_constructed",
        "[07:01:02] simulator_starting · worker_spawned · pid=1234 · 18ms",
    ]


def test_startup_journal_ignores_one_truncated_final_record(tmp_path):
    path = tmp_path / "startup.jsonl"
    append_startup_event(path, event("worker_spawn_started"))
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1')

    assert [item.event for item in read_startup_events(path)] == [
        "worker_spawn_started"
    ]


def test_startup_event_rejects_unknown_or_sensitive_fields(tmp_path):
    path = tmp_path / "startup.jsonl"
    append_startup_event(path, event("worker_command_ready"))
    payload = json.loads(path.read_text().splitlines()[0])

    assert set(payload) == {
        "schema_version",
        "timestamp",
        "evaluation_id",
        "phase",
        "event",
    }
    assert "command" not in payload
    with pytest.raises(ValueError, match="event"):
        event("contains spaces")


def test_missing_startup_journal_is_empty(tmp_path):
    assert read_startup_events(tmp_path / "missing.jsonl") == ()


def test_worker_startup_event_preserves_duration(monkeypatch, tmp_path):
    path = tmp_path / "startup.jsonl"
    monkeypatch.setenv("RPENT_STARTUP_LOG_PATH", str(path))
    monkeypatch.setenv("RPENT_EVALUATION_ID", "evaluation-1")
    monkeypatch.setenv("RPENT_STARTUP_CASE_ID", "case-1")
    monkeypatch.setenv("RPENT_STARTUP_ATTEMPT_ID", "attempt-1")

    assert record_worker_startup_event(
        "scene_initialization", "completed", duration_ms=1234
    )

    recorded = read_startup_events(path)
    assert len(recorded) == 1
    assert recorded[0].phase == "scene_initialization"
    assert recorded[0].duration_ms == 1234
