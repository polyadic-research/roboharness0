"""Bounded, reconnectable SSE over durable projections and Event Hub wakes."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from fastapi.responses import StreamingResponse

from rpent.recording.contracts import EventEnvelope
from rpent.recording.notifications import (
    NotificationTooLargeError,
    validate_notification,
)
from rpent.ui.case_snapshot import SnapshotRebuilding
from rpent.ui.event_hub import EvaluationEventHub

DEFAULT_HEARTBEAT_INTERVAL = 15.0
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_REPOSITORY_WORKERS = 4
DEFAULT_REPOSITORY_QUEUE = 32
DEFAULT_REPOSITORY_TIMEOUT = 2.0
DEFAULT_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
SSE_PAGE_MAX_EVENTS = 64
SSE_PAGE_MAX_BYTES = 256 * 1024


class SSECapacityError(RuntimeError):
    """The bounded SSE server cannot admit more work."""


class SSEPayloadTooLargeError(ValueError):
    """An authoritative snapshot exceeds its explicit response budget."""


class SSEConnection:
    """One reserved stream slot, released exactly once."""

    def __init__(self, manager: SSEManager) -> None:
        self._manager = manager
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release_connection()


class SSEManager:
    """Application-owned bounds for streams and repository filesystem reads."""

    def __init__(
        self,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        repository_workers: int = DEFAULT_REPOSITORY_WORKERS,
        repository_queue: int = DEFAULT_REPOSITORY_QUEUE,
        repository_timeout_s: float = DEFAULT_REPOSITORY_TIMEOUT,
        snapshot_max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
    ) -> None:
        self.max_connections = _positive_integer(max_connections, "max_connections")
        self.repository_workers = _positive_integer(
            repository_workers, "repository_workers"
        )
        self.repository_queue = _non_negative_integer(
            repository_queue, "repository_queue"
        )
        self.repository_timeout_s = _positive_number(
            repository_timeout_s, "repository_timeout_s"
        )
        self.snapshot_max_bytes = _positive_integer(
            snapshot_max_bytes, "snapshot_max_bytes"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.repository_workers,
            thread_name_prefix="rpent-sse-repository",
        )
        self._repository_slots = threading.BoundedSemaphore(
            self.repository_workers + self.repository_queue
        )
        self._lock = threading.Lock()
        self._connections = 0
        self._rejected_connections = 0
        self._submitted_repository_calls = 0
        self._repository_idle = threading.Event()
        self._repository_idle.set()
        self._closed = False

    @property
    def active_connections(self) -> int:
        with self._lock:
            return self._connections

    @property
    def submitted_repository_calls(self) -> int:
        with self._lock:
            return self._submitted_repository_calls

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def reserve(self) -> SSEConnection:
        with self._lock:
            if self._closed:
                self._rejected_connections += 1
                raise SSECapacityError("SSE manager is closed")
            if self._connections >= self.max_connections:
                self._rejected_connections += 1
                raise SSECapacityError("SSE connection limit reached")
            self._connections += 1
        return SSEConnection(self)

    def snapshot(self) -> dict[str, int | bool]:
        """Return coherent in-memory connection diagnostics."""
        with self._lock:
            return {
                "capacity": self.max_connections,
                "active": self._connections,
                "rejected": self._rejected_connections,
                "saturated": int(
                    self._closed or self._connections >= self.max_connections
                ),
                "closed": self._closed,
            }

    async def repository_call(self, function: Callable[..., Any], *args, **kwargs):
        if self.closed:
            raise SSECapacityError("SSE manager is closed")
        if not self._repository_slots.acquire(blocking=False):
            raise SSECapacityError("SSE repository capacity is full")
        if self.closed:
            self._repository_slots.release()
            raise SSECapacityError("SSE manager is closed")
        with self._lock:
            if self._closed:
                self._repository_slots.release()
                raise SSECapacityError("SSE manager is closed")
            try:
                concurrent_future = self._executor.submit(
                    partial(function, *args, **kwargs)
                )
            except BaseException:
                self._repository_slots.release()
                raise
            self._submitted_repository_calls += 1
            self._repository_idle.clear()

        def completed(_future) -> None:
            self._repository_slots.release()
            with self._lock:
                self._submitted_repository_calls -= 1
                if self._submitted_repository_calls == 0:
                    self._repository_idle.set()

        concurrent_future.add_done_callback(completed)
        loop = asyncio.get_running_loop()
        future = asyncio.wrap_future(concurrent_future, loop=loop)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.repository_timeout_s,
            )
        except TimeoutError as exc:
            if concurrent_future.done() and not concurrent_future.cancelled():
                underlying = concurrent_future.exception()
                if isinstance(underlying, TimeoutError):
                    raise underlying
            raise SSECapacityError("SSE repository call timed out") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._repository_idle.wait(0.1):
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _release_connection(self) -> None:
        with self._lock:
            if self._connections:
                self._connections -= 1


def encode_sse(
    name: str,
    payload: dict[str, Any],
    *,
    event_id: str | None = None,
    snapshot_max_bytes: int = DEFAULT_SNAPSHOT_MAX_BYTES,
) -> str:
    """Encode one strict-JSON, media-free SSE frame."""
    if not isinstance(name, str) or not name or "\r" in name or "\n" in name:
        raise ValueError("invalid SSE event name")
    if event_id is not None and ("\r" in event_id or "\n" in event_id):
        raise ValueError("invalid SSE event ID")
    selected_snapshot_limit = _positive_integer(
        snapshot_max_bytes, "snapshot_max_bytes"
    )
    data = _validated_public_json(payload)
    if name == "snapshot" and len(data.encode()) > selected_snapshot_limit:
        raise SSEPayloadTooLargeError("SSE snapshot exceeds serialized byte limit")
    identifier = "" if event_id is None else f"id: {event_id}\n"
    return f"{identifier}event: {name}\ndata: {data}\n\n"


def _overload_frame() -> str:
    return "retry: 2000\n" + encode_sse("overload", {"retry_after": 2})


async def evaluation_event_stream(
    request: Any,
    repository: Any,
    hub: EvaluationEventHub,
    manager: SSEManager,
    evaluation_id: str,
    *,
    cursor: str,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> AsyncIterator[str]:
    inner = _evaluation_event_stream_impl(
        request,
        repository,
        hub,
        manager,
        evaluation_id,
        cursor=cursor,
        heartbeat_interval=heartbeat_interval,
        poll_interval=poll_interval,
    )
    try:
        async for frame in inner:
            yield frame
    except SSECapacityError:
        if manager.closed:
            return
        yield _overload_frame()
    finally:
        await inner.aclose()


async def _evaluation_event_stream_impl(
    request: Any,
    repository: Any,
    hub: EvaluationEventHub,
    manager: SSEManager,
    evaluation_id: str,
    *,
    cursor: str,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> AsyncIterator[str]:
    """Read bounded Hub pages only from the browser-acknowledged position."""
    _stream_limits(heartbeat_interval, poll_interval)
    subscription = hub.open_async_subscription(evaluation_id)
    try:
        scan_cursor = cursor
        projection = await manager.repository_call(
            repository.get_evaluation, evaluation_id
        )
        projection_payload = projection.model_dump()
        last_emit = time.monotonic()
    except BaseException:
        subscription.close()
        raise
    try:
        while not manager.closed and not await request.is_disconnected():
            batch = hub.read(
                scan_cursor,
                evaluation_id=evaluation_id,
                max_events=SSE_PAGE_MAX_EVENTS,
                max_bytes=SSE_PAGE_MAX_BYTES,
            )
            if batch.expired:
                scan_cursor = batch.next_cursor
                yield await _evaluation_snapshot(
                    manager,
                    repository,
                    evaluation_id,
                    scan_cursor,
                    batch.expired_reason or "cursor_expired",
                )
                last_emit = time.monotonic()
                continue
            scan_cursor = batch.next_cursor
            if batch.events:
                for index, event in enumerate(batch.events):
                    validate_notification(event)
                    yield encode_sse(
                        "evaluation_event",
                        event.to_payload(),
                        event_id=(
                            scan_cursor if index == len(batch.events) - 1 else None
                        ),
                    )
                    last_emit = time.monotonic()
                continue

            current = (
                await manager.repository_call(repository.get_evaluation, evaluation_id)
            ).model_dump()
            if current != projection_payload:
                projection_payload = current
                yield await _evaluation_snapshot(
                    manager,
                    repository,
                    evaluation_id,
                    scan_cursor,
                    "projection_changed",
                )
                last_emit = time.monotonic()
                continue
            await subscription.wait(poll_interval)
            if time.monotonic() - last_emit >= heartbeat_interval:
                yield encode_sse(
                    "heartbeat",
                    {"cursor": scan_cursor},
                    event_id=scan_cursor,
                )
                last_emit = time.monotonic()
    finally:
        subscription.close()


async def case_event_stream(
    request: Any,
    repository: Any,
    hub: EvaluationEventHub,
    manager: SSEManager,
    evaluation_id: str,
    case_id: str,
    *,
    since: int | None = None,
    cursor: str | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    case_reader: Any | None = None,
) -> AsyncIterator[str]:
    inner = _case_event_stream_impl(
        request,
        repository,
        hub,
        manager,
        evaluation_id,
        case_id,
        since=since,
        cursor=cursor,
        heartbeat_interval=heartbeat_interval,
        poll_interval=poll_interval,
        case_reader=case_reader,
    )
    try:
        async for frame in inner:
            yield frame
    except SSECapacityError:
        if manager.closed:
            return
        yield _overload_frame()
    finally:
        await inner.aclose()


async def _case_event_stream_impl(
    request: Any,
    repository: Any,
    hub: EvaluationEventHub,
    manager: SSEManager,
    evaluation_id: str,
    case_id: str,
    *,
    since: int | None = None,
    cursor: str | None = None,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    case_reader: Any | None = None,
) -> AsyncIterator[str]:
    """Stream one Case attempt with its journal as the lossless authority."""
    _stream_limits(heartbeat_interval, poll_interval)
    if since is not None and (
        isinstance(since, bool) or not isinstance(since, int) or since < 0
    ):
        raise ValueError("since must be a non-negative integer")
    subscription = hub.open_async_subscription(evaluation_id)
    selected_case_reader = repository.get_case if case_reader is None else case_reader
    try:
        while True:
            try:
                current = await manager.repository_call(
                    selected_case_reader, evaluation_id, case_id
                )
                break
            except SnapshotRebuilding:
                if manager.closed or await request.is_disconnected():
                    return
                await subscription.wait(poll_interval)
        position = (
            hub.case_position(cursor, evaluation_id, case_id)
            if cursor is not None
            else None
        )
        reset_attempt = (
            position is not None and position.attempt_id != current.attempt_id
        )
        sequence = (
            0
            if reset_attempt
            else position.sequence
            if position is not None
            else 0
            if since is None
            else since
        )
        projection_payload = current.model_dump()
        last_emit = time.monotonic()
    except BaseException:
        subscription.close()
        raise
    try:
        if reset_attempt:
            yield _case_snapshot(
                manager,
                hub,
                evaluation_id,
                case_id,
                projection_payload,
                current.attempt_id,
                0,
                "attempt_changed",
            )
            last_emit = time.monotonic()

        while not manager.closed and not await request.is_disconnected():
            try:
                current = await manager.repository_call(
                    selected_case_reader, evaluation_id, case_id
                )
            except SnapshotRebuilding:
                await subscription.wait(poll_interval)
                continue
            current_payload = current.model_dump()
            if current.attempt_id != projection_payload.get("attempt_id"):
                sequence = 0
                projection_payload = current_payload
                yield _case_snapshot(
                    manager,
                    hub,
                    evaluation_id,
                    case_id,
                    projection_payload,
                    current.attempt_id,
                    sequence,
                    "attempt_changed",
                )
                last_emit = time.monotonic()
            elif current_payload != projection_payload:
                projection_payload = current_payload
                yield _case_snapshot(
                    manager,
                    hub,
                    evaluation_id,
                    case_id,
                    projection_payload,
                    current.attempt_id,
                    sequence,
                    "projection_changed",
                )
                last_emit = time.monotonic()

            emitted = False
            repairing = False
            while True:
                page = await manager.repository_call(
                    repository.events_since,
                    evaluation_id,
                    case_id,
                    since=sequence,
                )
                try:
                    latest = await manager.repository_call(
                        selected_case_reader, evaluation_id, case_id
                    )
                except SnapshotRebuilding:
                    repairing = True
                    break
                latest_payload = latest.model_dump()
                if latest.attempt_id != current.attempt_id:
                    current = latest
                    sequence = 0
                    projection_payload = latest_payload
                    yield _case_snapshot(
                        manager,
                        hub,
                        evaluation_id,
                        case_id,
                        projection_payload,
                        current.attempt_id,
                        sequence,
                        "attempt_changed",
                    )
                    emitted = True
                    last_emit = time.monotonic()
                    break
                if latest_payload != projection_payload:
                    current = latest
                    projection_payload = latest_payload
                    yield _case_snapshot(
                        manager,
                        hub,
                        evaluation_id,
                        case_id,
                        projection_payload,
                        current.attempt_id,
                        sequence,
                        "projection_changed",
                    )
                    emitted = True
                    last_emit = time.monotonic()
                    break
                if page.attempt_id != current.attempt_id:
                    break
                for payload in page.events:
                    event = EventEnvelope.from_payload(payload)
                    validate_notification(event)
                    if (
                        event.attempt_id != current.attempt_id
                        or event.sequence <= sequence
                    ):
                        continue
                    if event.sequence != sequence + 1:
                        raise ValueError("Case event journal sequence mismatch")
                    sequence = event.sequence
                    yield encode_sse(
                        "case_event",
                        event.to_payload(),
                        event_id=hub.case_cursor(
                            evaluation_id, case_id, event.attempt_id, sequence
                        ),
                    )
                    emitted = True
                    last_emit = time.monotonic()
                if not page.has_more:
                    break
            if repairing:
                await subscription.wait(poll_interval)
                continue
            if emitted:
                continue

            await subscription.wait(poll_interval)
            if time.monotonic() - last_emit >= heartbeat_interval:
                heartbeat_cursor = hub.case_cursor(
                    evaluation_id, case_id, current.attempt_id, sequence
                )
                yield encode_sse(
                    "heartbeat",
                    {
                        "attempt_id": current.attempt_id,
                        "last_sequence": sequence,
                    },
                    event_id=heartbeat_cursor,
                )
                last_emit = time.monotonic()
    finally:
        subscription.close()


async def managed_stream(
    stream: AsyncIterator[str], connection: SSEConnection
) -> AsyncIterator[str]:
    """Release a route reservation on normal, exceptional, or cancelled exit."""
    try:
        async for frame in stream:
            yield frame
    finally:
        await stream.aclose()
        connection.release()


class ReservedStreamingResponse(StreamingResponse):
    """Release an admitted SSE slot across the complete ASGI response call."""

    def __init__(self, content, connection: SSEConnection, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self._connection = connection

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            try:
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except BaseException:
                        pass
            finally:
                self._connection.release()
            raise
        try:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            self._connection.release()


async def _evaluation_snapshot(
    manager: SSEManager,
    repository: Any,
    evaluation_id: str,
    cursor: str,
    reason: str,
) -> str:
    evaluation = await manager.repository_call(
        repository.get_evaluation,
        evaluation_id,
    )
    checkpoints = await manager.repository_call(
        repository.event_checkpoint,
        evaluation_id,
    )
    return encode_sse(
        "snapshot",
        {
            "cursor": cursor,
            "reset_reason": reason,
            "evaluation": evaluation.model_dump(),
            "last_sequences": checkpoints,
        },
        event_id=cursor,
        snapshot_max_bytes=manager.snapshot_max_bytes,
    )


def _case_snapshot(
    manager: SSEManager,
    hub: EvaluationEventHub,
    evaluation_id: str,
    case_id: str,
    case: dict[str, Any],
    attempt_id: str | None,
    sequence: int,
    reason: str,
) -> str:
    return encode_sse(
        "snapshot",
        {
            "evaluation_id": evaluation_id,
            "case_id": case_id,
            "attempt_id": attempt_id,
            "last_sequence": sequence,
            "reset_reason": reason,
            "case": case,
        },
        event_id=hub.case_cursor(evaluation_id, case_id, attempt_id, sequence),
        snapshot_max_bytes=manager.snapshot_max_bytes,
    )


def _stream_limits(heartbeat: float, poll: float) -> None:
    for value, name in ((heartbeat, "heartbeat_interval"), (poll, "poll_interval")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")


def _validated_public_json(payload: dict[str, Any]) -> str:
    synthetic = False
    try:
        event = EventEnvelope.from_payload(payload)
    except ValueError:
        synthetic = True
        safe_payload = json.loads(
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        )
        event = EventEnvelope(
            schema_version=1,
            evaluation_id="sse",
            case_id="sse",
            attempt_id="sse",
            sequence=0,
            recorded_at="1970-01-01T00:00:00+00:00",
            event_type="sse_payload",
            step_id=None,
            operation_id=None,
            execution_id=None,
            payload=safe_payload,
        )
    try:
        validate_notification(event)
    except NotificationTooLargeError:
        if not synthetic:
            raise
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_REPOSITORY_TIMEOUT",
    "DEFAULT_SNAPSHOT_MAX_BYTES",
    "SSECapacityError",
    "SSEConnection",
    "SSEManager",
    "SSEPayloadTooLargeError",
    "ReservedStreamingResponse",
    "case_event_stream",
    "encode_sse",
    "evaluation_event_stream",
    "managed_stream",
]
