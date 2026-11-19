"""Bounded in-memory routing for durable evaluation notifications."""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import math
import secrets
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from rpent.recording.contracts import EventEnvelope
from rpent.recording.notifications import validate_notification
from rpent.ui.case_runtime import RuntimeUpdate


@dataclass(frozen=True, slots=True)
class AttemptSequence:
    attempt_id: str | None
    sequence: int


@dataclass(frozen=True, slots=True)
class EventBatch:
    events: tuple[EventEnvelope, ...]
    next_cursor: str
    expired: bool
    last_sequences: Mapping[str, AttemptSequence]
    expired_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Entry:
    offset: int
    event: EventEnvelope


@dataclass(frozen=True, slots=True)
class _Cursor:
    offset: int
    epoch: bytes
    scope: bytes | None


@dataclass(frozen=True, slots=True)
class _CurrentAttempt:
    attempt_id: str
    sequence: int
    recorded_at: datetime


@dataclass(slots=True)
class _AsyncListener:
    evaluation_id: str
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[None]
    wake_scheduled: bool = False


class AsyncHubSubscription:
    """One coalesced wake queue registered with an Evaluation Event Hub."""

    def __init__(
        self,
        hub: EvaluationEventHub,
        subscriber_id: int,
        queue: asyncio.Queue[None],
    ) -> None:
        self._hub = hub
        self._subscriber_id = subscriber_id
        self._queue = queue
        self._closed = False

    @property
    def pending_wakes(self) -> int:
        return self._queue.qsize()

    async def wait(self, timeout: float | None = None) -> bool:
        if self._closed:
            return False
        try:
            if timeout is None:
                await self._queue.get()
            else:
                await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return False
        self._hub._ack_async_wake(self._subscriber_id)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub._remove_async_subscription(self._subscriber_id)


class EvaluationEventHub:
    """Keep bounded arrival and de-duplication windows for notifications."""

    def __init__(
        self,
        capacity: int = 10_000,
        *,
        dedupe_capacity: int | None = None,
        secret: bytes | None = None,
    ) -> None:
        self._capacity = _positive_integer(capacity, "capacity")
        self._dedupe_capacity = _positive_integer(
            capacity if dedupe_capacity is None else dedupe_capacity,
            "dedupe_capacity",
        )
        if secret is not None and (not isinstance(secret, bytes) or not secret):
            raise ValueError("secret must be non-empty bytes")
        # Server assembly may inject a stable secret; the process-local default
        # deliberately expires cursors after an uncoordinated restart.
        self._secret = secret or secrets.token_bytes(32)
        self._epoch = secrets.token_bytes(8)
        self._entries: deque[_Entry] = deque(maxlen=self._capacity)
        self._attempt_highwaters: OrderedDict[tuple[str, str, str], int] = OrderedDict()
        self._current_attempts: OrderedDict[tuple[str, str], _CurrentAttempt] = (
            OrderedDict()
        )
        self._runtime_revisions: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._offset = 0
        self._condition = threading.Condition()
        self._async_subscribers: dict[int, _AsyncListener] = {}
        self._next_async_subscriber_id = 0

    @property
    def async_subscriber_count(self) -> int:
        with self._condition:
            return len(self._async_subscribers)

    def open_async_subscription(
        self,
        evaluation_id: str,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> AsyncHubSubscription:
        self._scope(evaluation_id)
        selected_loop = asyncio.get_running_loop() if loop is None else loop
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        with self._condition:
            self._next_async_subscriber_id += 1
            subscriber_id = self._next_async_subscriber_id
            self._async_subscribers[subscriber_id] = _AsyncListener(
                evaluation_id=evaluation_id,
                loop=selected_loop,
                queue=queue,
            )
        return AsyncHubSubscription(self, subscriber_id, queue)

    def cursor(self, evaluation_id: str | None = None) -> str:
        scope = self._scope(evaluation_id)
        with self._condition:
            return self._encode_cursor(self._offset, scope)

    def case_cursor(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
        sequence: int,
    ) -> str:
        """Issue an opaque durable-journal cursor scoped to one Case attempt."""
        _non_empty_text(evaluation_id, "evaluation_id")
        _non_empty_text(case_id, "case_id")
        if attempt_id is not None:
            _non_empty_text(attempt_id, "attempt_id")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        payload = json.dumps(
            {
                "attempt_id": attempt_id,
                "case_id": case_id,
                "evaluation_id": evaluation_id,
                "sequence": sequence,
                "version": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.digest(self._secret, b"case-cursor:" + payload, "sha256")[:16]
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode()

    def case_position(
        self, cursor: str, evaluation_id: str, case_id: str
    ) -> AttemptSequence:
        """Validate and decode a Case cursor for the requested resource scope."""
        _non_empty_text(evaluation_id, "evaluation_id")
        _non_empty_text(case_id, "case_id")
        if not isinstance(cursor, str) or not cursor or len(cursor) > 4096:
            raise ValueError("invalid case cursor")
        try:
            padding = "=" * (-len(cursor) % 4)
            token = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("invalid case cursor") from exc
        canonical = base64.urlsafe_b64encode(token).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(cursor, canonical) or len(token) <= 16:
            raise ValueError("invalid case cursor encoding")
        payload, signature = token[:-16], token[-16:]
        expected = hmac.digest(self._secret, b"case-cursor:" + payload, "sha256")[:16]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid case cursor signature")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid case cursor payload") from exc
        if not isinstance(decoded, dict) or set(decoded) != {
            "attempt_id",
            "case_id",
            "evaluation_id",
            "sequence",
            "version",
        }:
            raise ValueError("invalid case cursor payload")
        canonical_payload = json.dumps(
            decoded, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if payload != canonical_payload:
            raise ValueError("invalid case cursor payload")
        attempt_id = decoded["attempt_id"]
        sequence = decoded["sequence"]
        if (
            decoded["version"] != 1
            or not isinstance(decoded["evaluation_id"], str)
            or not isinstance(decoded["case_id"], str)
            or (attempt_id is not None and not isinstance(attempt_id, str))
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError("invalid case cursor payload")
        if decoded["evaluation_id"] != evaluation_id or decoded["case_id"] != case_id:
            raise ValueError("case cursor scope does not match")
        if attempt_id is not None:
            _non_empty_text(attempt_id, "attempt_id")
        return AttemptSequence(attempt_id, sequence)

    def publish(self, event: EventEnvelope) -> bool:
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be an EventEnvelope")
        validate_notification(event)
        retained = EventEnvelope.from_payload(event.to_payload())
        attempt_key = (
            retained.evaluation_id,
            retained.case_id,
            retained.attempt_id,
        )
        case_key = (retained.evaluation_id, retained.case_id)
        recorded_at = datetime.fromisoformat(
            retained.recorded_at.replace("Z", "+00:00")
        )
        listeners: tuple[
            tuple[int, asyncio.AbstractEventLoop, asyncio.Queue[None]], ...
        ]
        with self._condition:
            current = self._current_attempts.get(case_key)
            if (
                current is not None
                and current.attempt_id == retained.attempt_id
                and retained.sequence <= current.sequence
            ):
                self._current_attempts.move_to_end(case_key)
                if attempt_key in self._attempt_highwaters:
                    self._attempt_highwaters.move_to_end(attempt_key)
                return False
            highwater = self._attempt_highwaters.get(attempt_key)
            if highwater is not None and retained.sequence <= highwater:
                self._attempt_highwaters.move_to_end(attempt_key)
                return False
            known_attempt = highwater is not None
            self._remember_attempt(attempt_key, retained.sequence)
            if (
                current is None
                or current.attempt_id == retained.attempt_id
                or (not known_attempt and recorded_at > current.recorded_at)
            ):
                self._remember_current(
                    case_key,
                    _CurrentAttempt(
                        retained.attempt_id, retained.sequence, recorded_at
                    ),
                )
            self._offset += 1
            self._entries.append(_Entry(self._offset, retained))
            pending = []
            for subscriber_id, listener in self._async_subscribers.items():
                if (
                    listener.evaluation_id == retained.evaluation_id
                    and not listener.wake_scheduled
                ):
                    listener.wake_scheduled = True
                    pending.append((subscriber_id, listener.loop, listener.queue))
            listeners = tuple(pending)
            self._condition.notify_all()
        for subscriber_id, loop, queue in listeners:
            try:
                loop.call_soon_threadsafe(_coalesced_wake, queue)
            except RuntimeError:
                with self._condition:
                    listener = self._async_subscribers.get(subscriber_id)
                    if listener is not None:
                        listener.wake_scheduled = False
                        self._async_subscribers.pop(subscriber_id, None)
        return True

    def publish_runtime_update(self, update: RuntimeUpdate) -> bool:
        """Publish one monotonic reducer update and wake on lifecycle changes."""
        if not isinstance(update, RuntimeUpdate):
            raise TypeError("update must be a RuntimeUpdate")
        if update.runtime_revision < 1:
            raise ValueError("runtime_revision must be positive")
        case_key = (update.evaluation_id, update.case_id)
        with self._condition:
            revision = self._runtime_revisions.get(case_key)
            if revision is not None and update.runtime_revision <= revision:
                self._runtime_revisions.move_to_end(case_key)
                return False
            self._runtime_revisions[case_key] = update.runtime_revision
            self._runtime_revisions.move_to_end(case_key)
            while len(self._runtime_revisions) > self._dedupe_capacity:
                self._runtime_revisions.popitem(last=False)
        published = False
        if update.event is not None:
            if (
                update.event.evaluation_id != update.evaluation_id
                or update.event.case_id != update.case_id
                or update.event.attempt_id != update.attempt_id
            ):
                raise ValueError("RuntimeUpdate event identity mismatch")
            published = self.publish(update.event)
        if not published:
            self._wake_evaluation(update.evaluation_id)
        return True

    def _wake_evaluation(self, evaluation_id: str) -> None:
        pending: list[tuple[int, asyncio.AbstractEventLoop, asyncio.Queue[None]]] = []
        with self._condition:
            for subscriber_id, listener in self._async_subscribers.items():
                if (
                    listener.evaluation_id == evaluation_id
                    and not listener.wake_scheduled
                ):
                    listener.wake_scheduled = True
                    pending.append((subscriber_id, listener.loop, listener.queue))
            self._condition.notify_all()
        for subscriber_id, loop, queue in pending:
            try:
                loop.call_soon_threadsafe(_coalesced_wake, queue)
            except RuntimeError:
                with self._condition:
                    listener = self._async_subscribers.get(subscriber_id)
                    if listener is not None:
                        listener.wake_scheduled = False
                        self._async_subscribers.pop(subscriber_id, None)

    def read(
        self,
        cursor: str,
        *,
        evaluation_id: str,
        max_events: int | None = None,
        max_bytes: int | None = None,
    ) -> EventBatch:
        selected_max_events = _optional_positive_integer(max_events, "max_events")
        selected_max_bytes = _optional_positive_integer(max_bytes, "max_bytes")
        scope = self._scope(evaluation_id)
        assert scope is not None
        decoded = self._decode_cursor(cursor)
        self._validate_scope(decoded, scope)
        with self._condition:
            return self._read_locked(
                decoded,
                evaluation_id,
                scope,
                max_events=selected_max_events,
                max_bytes=selected_max_bytes,
            )

    def subscribe(
        self,
        cursor: str,
        *,
        evaluation_id: str,
        timeout: float = 15.0,
        max_events: int | None = None,
        max_bytes: int | None = None,
    ) -> EventBatch:
        """Short-poll until a matching event, expiration, or finite timeout."""
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(timeout)
            or timeout < 0
            or timeout > 3600
        ):
            raise ValueError("timeout must be a builtin number between 0 and 3600")
        selected_max_events = _optional_positive_integer(max_events, "max_events")
        selected_max_bytes = _optional_positive_integer(max_bytes, "max_bytes")
        scope = self._scope(evaluation_id)
        assert scope is not None
        decoded = self._decode_cursor(cursor)
        self._validate_scope(decoded, scope)
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                batch = self._read_locked(
                    decoded,
                    evaluation_id,
                    scope,
                    max_events=selected_max_events,
                    max_bytes=selected_max_bytes,
                )
                if batch.events or batch.expired:
                    return batch
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    return batch
                self._condition.wait(remaining)

    def _read_locked(
        self,
        decoded: _Cursor,
        evaluation_id: str,
        scope: bytes,
        *,
        max_events: int | None,
        max_bytes: int | None,
    ) -> EventBatch:
        if decoded.epoch != self._epoch:
            return self._expired_batch(evaluation_id, scope, "hub_restarted")
        if decoded.offset > self._offset:
            raise ValueError("cursor is ahead of the Event Hub")
        oldest = self._entries[0].offset if self._entries else self._offset + 1
        if decoded.offset < oldest - 1:
            return self._expired_batch(evaluation_id, scope, "capacity")
        events: list[EventEnvelope] = []
        used_bytes = 0
        scanned_offset = decoded.offset
        for entry in self._entries:
            if entry.offset <= decoded.offset:
                continue
            if entry.event.evaluation_id != evaluation_id:
                scanned_offset = entry.offset
                continue
            if max_events is not None and len(events) == max_events:
                break
            event_bytes = _event_bytes(entry.event)
            if max_bytes is not None and used_bytes + event_bytes > max_bytes:
                if not events:
                    raise ValueError("event exceeds max_bytes")
                break
            events.append(EventEnvelope.from_payload(entry.event.to_payload()))
            used_bytes += event_bytes
            scanned_offset = entry.offset
        else:
            scanned_offset = self._offset
        return EventBatch(
            events=tuple(events),
            next_cursor=self._encode_cursor(scanned_offset, scope),
            expired=False,
            last_sequences=self._snapshot(evaluation_id),
        )

    def _expired_batch(
        self, evaluation_id: str, scope: bytes, reason: str
    ) -> EventBatch:
        return EventBatch(
            events=(),
            next_cursor=self._encode_cursor(self._offset, scope),
            expired=True,
            last_sequences=self._snapshot(evaluation_id),
            expired_reason=reason,
        )

    def _snapshot(self, evaluation_id: str) -> Mapping[str, AttemptSequence]:
        return MappingProxyType(
            {
                case_id: AttemptSequence(sequence.attempt_id, sequence.sequence)
                for (selected, case_id), sequence in self._current_attempts.items()
                if selected == evaluation_id
            }
        )

    def _remember_attempt(self, key: tuple[str, str, str], sequence: int) -> None:
        self._attempt_highwaters[key] = sequence
        self._attempt_highwaters.move_to_end(key)
        if len(self._attempt_highwaters) > self._dedupe_capacity:
            self._attempt_highwaters.popitem(last=False)

    def _remember_current(
        self, key: tuple[str, str], sequence: _CurrentAttempt
    ) -> None:
        self._current_attempts[key] = sequence
        self._current_attempts.move_to_end(key)
        if len(self._current_attempts) > self._dedupe_capacity:
            self._current_attempts.popitem(last=False)

    def _remove_async_subscription(self, subscriber_id: int) -> None:
        with self._condition:
            self._async_subscribers.pop(subscriber_id, None)

    def _ack_async_wake(self, subscriber_id: int) -> None:
        with self._condition:
            listener = self._async_subscribers.get(subscriber_id)
            if listener is not None:
                listener.wake_scheduled = False

    def _scope(self, evaluation_id: str | None) -> bytes | None:
        if evaluation_id is None:
            return None
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            raise ValueError("evaluation_id must be non-empty")
        return hmac.digest(
            self._secret, b"cursor-scope:" + evaluation_id.encode(), "sha256"
        )[:16]

    @staticmethod
    def _validate_scope(decoded: _Cursor, expected: bytes) -> None:
        if decoded.scope is not None and not hmac.compare_digest(
            decoded.scope, expected
        ):
            raise ValueError("cursor evaluation scope does not match")

    def _encode_cursor(self, offset: int, scope: bytes | None) -> str:
        packed = struct.pack(">Q", offset)
        nonce = secrets.token_bytes(8)
        mask = hmac.digest(self._secret, b"cursor-mask:" + nonce, "sha256")[:8]
        concealed = bytes(left ^ right for left, right in zip(packed, mask))
        payload = (
            b"\x02"
            + self._epoch
            + (b"\x00" + bytes(16) if scope is None else b"\x01" + scope)
            + nonce
            + concealed
        )
        signature = hmac.digest(self._secret, payload, "sha256")[:16]
        return (
            base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")
        )

    def _decode_cursor(self, cursor: str) -> _Cursor:
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("invalid cursor")
        try:
            padding = "=" * (-len(cursor) % 4)
            token = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ValueError("invalid cursor") from exc
        canonical = base64.urlsafe_b64encode(token).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(cursor, canonical):
            raise ValueError("invalid cursor encoding")
        if len(token) != 58 or token[0] != 2:
            raise ValueError("invalid cursor")
        payload, signature = token[:42], token[42:]
        expected = hmac.digest(self._secret, payload, "sha256")[:16]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid cursor signature")
        scoped = payload[9]
        if scoped not in {0, 1} or (scoped == 0 and any(payload[10:26])):
            raise ValueError("invalid cursor scope")
        nonce, concealed = payload[26:34], payload[34:42]
        mask = hmac.digest(self._secret, b"cursor-mask:" + nonce, "sha256")[:8]
        packed = bytes(left ^ right for left, right in zip(concealed, mask))
        return _Cursor(
            offset=struct.unpack(">Q", packed)[0],
            epoch=payload[1:9],
            scope=payload[10:26] if scoped else None,
        )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_integer(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _event_bytes(event: EventEnvelope) -> int:
    return len(
        json.dumps(
            event.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _coalesced_wake(queue: asyncio.Queue[None]) -> None:
    if not queue.full():
        queue.put_nowait(None)


def _non_empty_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


__all__ = [
    "AsyncHubSubscription",
    "AttemptSequence",
    "EvaluationEventHub",
    "EventBatch",
]
