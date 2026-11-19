"""Canonical, bounded events consumed by the unified Case monitor."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

_MAX_TEXT = 32_000
_MAX_STRING = 4_000
_MAX_ITEMS = 64
_MAX_DEPTH = 6
_MAX_JSON_BYTES = 16_000
_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}\Z")


def record_reasoning_delta(
    recorder: Any | None,
    text: str,
    *,
    content_kind: Literal["reasoning", "message"] = "reasoning",
) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reasoning text must be non-empty")
    if len(text) > _MAX_TEXT:
        raise ValueError(f"reasoning text exceeds {_MAX_TEXT} characters")
    if content_kind not in {"reasoning", "message"}:
        raise ValueError("reasoning content_kind must be reasoning or message")
    payload = {"text": text}
    if content_kind == "message":
        payload["content_kind"] = content_kind
    _record(recorder, "reasoning_delta", **payload)


def record_tool_call(
    recorder: Any | None,
    *,
    tool: str,
    arguments: Any,
    call_id: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "tool": _name(tool, "tool"),
        "arguments": _bounded_json(arguments),
    }
    if call_id is not None:
        payload["call_id"] = _name(call_id, "call_id")
    _record(recorder, "tool_call", **payload)


def record_tool_result(
    recorder: Any | None,
    *,
    tool: str,
    result: Any,
    is_error: bool = False,
    call_id: str | None = None,
) -> None:
    if not isinstance(is_error, bool):
        raise ValueError("is_error must be a boolean")
    payload: dict[str, Any] = {
        "tool": _name(tool, "tool"),
        "result": _bounded_json(result),
        "is_error": is_error,
    }
    if call_id is not None:
        payload["call_id"] = _name(call_id, "call_id")
    _record(recorder, "tool_result", **payload)


def record_usage_updated(
    recorder: Any | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    tool_calls: int = 0,
) -> None:
    payload = {
        "input_tokens": _counter(input_tokens, "input_tokens"),
        "output_tokens": _counter(output_tokens, "output_tokens"),
        "cached_input_tokens": _counter(cached_input_tokens, "cached_input_tokens"),
        "reasoning_tokens": _counter(reasoning_tokens, "reasoning_tokens"),
        "tool_calls": _counter(tool_calls, "tool_calls"),
    }
    _record(recorder, "usage_updated", **payload)


def record_action_clip_available(
    recorder: Any | None, artifact_id: str, *, step: int
) -> None:
    _record(
        recorder,
        "action_clip_available",
        artifact_id=_artifact(artifact_id),
        step=_counter(step, "step"),
    )


def record_episode_video_available(recorder: Any | None, artifact_id: str) -> None:
    _record(
        recorder,
        "episode_video_available",
        artifact_id=_artifact(artifact_id),
    )


def _record(recorder: Any | None, event: str, **payload: Any) -> None:
    if recorder is not None:
        try:
            recorder.record(event, **payload)
        except Exception:
            # Live projection is observational and must not interrupt execution.
            return


def _counter(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    selected = int(value)
    if selected < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return selected


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a non-empty string up to 256 characters")
    return value


def _artifact(value: Any) -> str:
    selected = _name(value, "artifact_id")
    if _ARTIFACT_ID.fullmatch(selected) is None:
        raise ValueError("artifact_id has an invalid format")
    return selected


def _bounded_json(value: Any) -> Any:
    selected = _project(value, depth=0)
    encoded = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= _MAX_JSON_BYTES:
        return selected
    return {
        "_truncated": True,
        "preview": encoded[: _MAX_JSON_BYTES // 2],
    }


def _project(value: Any, *, depth: int) -> Any:
    if depth >= _MAX_DEPTH:
        return "[max-depth]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "[non-finite-number]"
        return number
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "…"
    if isinstance(value, (Path, bytes, bytearray, memoryview)):
        return "[path]" if isinstance(value, Path) else f"[binary:{len(value)} bytes]"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        projected = {
            str(key)[:256]: _project(item, depth=depth + 1)
            for key, item in items[:_MAX_ITEMS]
        }
        if len(items) > _MAX_ITEMS:
            projected["_truncated_items"] = len(items) - _MAX_ITEMS
        return projected
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        projected = [_project(item, depth=depth + 1) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            projected.append({"_truncated_items": len(items) - _MAX_ITEMS})
        return projected
    return f"[{type(value).__name__}]"


__all__ = [
    "record_action_clip_available",
    "record_episode_video_available",
    "record_reasoning_delta",
    "record_tool_call",
    "record_tool_result",
    "record_usage_updated",
]
