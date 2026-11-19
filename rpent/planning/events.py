"""Append-only event recording for reproducible planning experiments."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


class EventRecorder(Protocol):
    """Minimal recorder interface used by the runtime, toolkit, and harness."""

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        """Persist one event and return its JSON-safe representation."""
        ...


class NullEventRecorder:
    """Recorder that preserves call sites without producing artifacts."""

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        return {"event": event, **payload}


class JsonlEventRecorder:
    """Thread-safe append-only JSONL recorder with monotonic sequence numbers."""

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        self.path = Path(path)
        self.episode_id = episode_id
        self.context = dict(context or {})
        self._sequence = 0
        self._lock = threading.Lock()

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        if not event.strip():
            raise ValueError("event must be non-empty")
        with self._lock:
            self._sequence += 1
            record = {
                "event_id": uuid4().hex,
                "episode_id": self.episode_id,
                "sequence": self._sequence,
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **self.context,
                **payload,
            }
            safe = _json_safe(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n"
                )
            return safe


def _json_safe(value: Any) -> Any:
    """Convert an event payload without embedding binary observation data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {
            "_binary_omitted": True,
            "bytes": len(value),
            "prefix_hex": value[:8].hex(),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)
