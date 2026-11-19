"""Best-effort timing; recording faults must not alter robot execution."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _emit(recorder: Any, event: str, payload: dict) -> None:
    if recorder is None:
        return
    try:
        recorder.record(event, **payload)
    except Exception as exc:
        logger.warning("Narration timing unavailable: %s", type(exc).__name__)


@contextmanager
def phase_span(
    recorder: Any,
    phase: str,
    *,
    clock=time.monotonic,
    wall_clock=time.time,
    reset_at=None,
    **metadata,
):
    """Bracket an actual request, including failure/cancellation, without content."""
    if recorder is None or not callable(getattr(recorder, "record", None)):
        yield
        return
    started = clock()
    payload = {
        "span_id": uuid4().hex,
        "phase": phase,
        "started_wall_s": wall_clock(),
        **metadata,
    }
    if reset_at is not None:
        payload["started_s"] = started - reset_at
    _emit(recorder, "narration_phase_started", payload)
    status = "completed"
    try:
        yield
    except BaseException as exc:
        status = (
            "cancelled"
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
            else "error"
        )
        raise
    finally:
        ended = clock()
        payload = {
            **payload,
            "ended_wall_s": wall_clock(),
            "duration_s": max(0, ended - started),
            "status": status,
        }
        if reset_at is not None:
            payload["ended_s"] = ended - reset_at
        _emit(recorder, "narration_phase_finished", payload)


def observer_phase(observer: Any, phase: str, **metadata):
    callback = getattr(observer, "phase", None)
    return callback(phase, **metadata) if callable(callback) else nullcontext()
