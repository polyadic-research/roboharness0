"""Bounded, append-only diagnostics for evaluation startup."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024
MAX_JOURNAL_BYTES = 1024 * 1024
MAX_RECORDS = 4096
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_KEYS = {
    "schema_version",
    "timestamp",
    "evaluation_id",
    "case_id",
    "attempt_id",
    "phase",
    "event",
    "duration_ms",
    "worker_pid",
    "error_type",
}


@dataclass(frozen=True, slots=True)
class StartupEvent:
    timestamp: str
    evaluation_id: str
    phase: str
    event: str
    case_id: str | None = None
    attempt_id: str | None = None
    duration_ms: int | None = None
    worker_pid: int | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
        if not self.evaluation_id:
            raise ValueError("evaluation_id must not be empty")
        for field_name in ("phase", "event"):
            if not _NAME.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be lowercase snake_case")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.worker_pid is not None and self.worker_pid <= 0:
            raise ValueError("worker_pid must be positive")
        if self.error_type is not None and not _NAME.fullmatch(self.error_type):
            raise ValueError("error_type must be lowercase snake_case")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            key: value for key, value in asdict(self).items() if value is not None
        }
        return {"schema_version": SCHEMA_VERSION, **payload}


def append_startup_event(path: Path, event: StartupEvent) -> None:
    """Append one complete JSON record with restrictive file permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (
        json.dumps(
            event.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(record) > MAX_RECORD_BYTES:
        raise ValueError("startup event exceeds maximum record size")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, record) != len(record):
            raise OSError("incomplete startup journal write")
    finally:
        os.close(descriptor)


def read_startup_events(path: Path) -> tuple[StartupEvent, ...]:
    """Read complete, valid records within fixed memory and record limits."""
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_JOURNAL_BYTES + 1)
    except FileNotFoundError:
        return ()
    if len(data) > MAX_JOURNAL_BYTES:
        data = data[-MAX_JOURNAL_BYTES:]
        first_newline = data.find(b"\n")
        data = data[first_newline + 1 :] if first_newline >= 0 else b""
    lines = data.splitlines(keepends=True)
    events: list[StartupEvent] = []
    for line in lines[-MAX_RECORDS:]:
        if not line.endswith(b"\n") or len(line) > MAX_RECORD_BYTES:
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict) or set(payload) - _ALLOWED_KEYS:
                continue
            if payload.pop("schema_version", None) != SCHEMA_VERSION:
                continue
            events.append(StartupEvent(**payload))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return tuple(events)


def render_startup_console(events: Iterable[StartupEvent]) -> str:
    lines: list[str] = []
    for item in events:
        timestamp = datetime.fromisoformat(item.timestamp.replace("Z", "+00:00"))
        parts = [f"[{timestamp:%H:%M:%S}] {item.phase}", item.event]
        if item.worker_pid is not None:
            parts.append(f"pid={item.worker_pid}")
        if item.duration_ms is not None:
            parts.append(f"{item.duration_ms}ms")
        if item.error_type is not None:
            parts.append(f"error={item.error_type}")
        lines.append(" · ".join(parts))
    return "\n".join(lines)


def record_startup_event(
    root: str | Path,
    evaluation_id: str,
    phase: str,
    event: str,
    **fields: Any,
) -> bool:
    """Best-effort diagnostic emission; never alter evaluation outcomes."""
    try:
        startup_event = StartupEvent(
            timestamp=datetime.now().astimezone().isoformat(),
            evaluation_id=evaluation_id,
            phase=phase,
            event=event,
            **fields,
        )
        append_startup_event(
            Path(root) / "startup.jsonl",
            startup_event,
        )
        logging.getLogger("rpent.evaluation.startup").info(
            "evaluation_startup %s",
            json.dumps(
                startup_event.to_payload(), sort_keys=True, separators=(",", ":")
            ),
        )
        return True
    except (OSError, TypeError, ValueError):
        return False


def record_worker_startup_event(
    phase: str,
    event: str,
    *,
    duration_ms: int | None = None,
) -> bool:
    """Emit using coordinator-provided, non-secret worker metadata."""
    path = os.environ.get("RPENT_STARTUP_LOG_PATH")
    evaluation_id = os.environ.get("RPENT_EVALUATION_ID")
    if not path or not evaluation_id:
        return False
    try:
        startup_event = StartupEvent(
            timestamp=datetime.now().astimezone().isoformat(),
            evaluation_id=evaluation_id,
            phase=phase,
            event=event,
            case_id=os.environ.get("RPENT_STARTUP_CASE_ID"),
            attempt_id=os.environ.get("RPENT_STARTUP_ATTEMPT_ID"),
            duration_ms=duration_ms,
            worker_pid=os.getpid(),
        )
        append_startup_event(Path(path), startup_event)
        logging.getLogger("rpent.evaluation.startup").info(
            "evaluation_startup %s",
            json.dumps(
                startup_event.to_payload(), sort_keys=True, separators=(",", ":")
            ),
        )
        return True
    except (OSError, TypeError, ValueError):
        return False
