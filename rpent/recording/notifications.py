"""Best-effort delivery of already-durable recording notifications."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from rpent.recording.contracts import EventEnvelope

NOTIFICATION_MAX_BYTES = 64 * 1024
_MEDIA_FIELDS = frozenset(
    {
        "image_bytes",
        "frame_bytes",
        "image_data",
        "data_url",
        "base64",
        "png_bytes",
        "jpeg_bytes",
    }
)


class NotificationTooLargeError(ValueError):
    """A notification exceeds the fixed serialized envelope budget."""


class NoopNotificationSink:
    def __call__(self, event: EventEnvelope) -> bool:
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be an EventEnvelope")
        return True


AsyncPost = Callable[..., Awaitable[httpx.Response]]


@dataclass(frozen=True, slots=True)
class NotificationTarget:
    instance_id: str
    origin: str
    notification_token: str | None


TargetResolver = Callable[[], NotificationTarget]


@dataclass(frozen=True, slots=True)
class _Delivery:
    event: EventEnvelope


class HttpNotificationSink:
    """Route deliveries through a fixed, bounded set of daemon workers."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        target: TargetResolver | None = None,
        post: AsyncPost | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        notification_token: str | None = None,
        workers: int = 2,
        queue_capacity: int | None = None,
    ) -> None:
        if post is not None and transport is not None:
            raise ValueError("provide post or transport, not both")
        if target is not None and (
            endpoint is not None or notification_token is not None
        ):
            raise ValueError("provide target or endpoint credentials, not both")
        if target is not None and not callable(target):
            raise TypeError("target must be callable")
        workers = _positive_integer(workers, "workers")
        capacity = _positive_integer(
            workers if queue_capacity is None else queue_capacity,
            "queue_capacity",
        )
        self.endpoint = None if endpoint is None else _http_origin(endpoint)
        if target is None:
            if self.endpoint is None:
                raise ValueError("endpoint or target is required")
            if notification_token is not None:
                _notification_token(notification_token)

            def target() -> NotificationTarget:
                return NotificationTarget(
                    "legacy", self.endpoint or "", notification_token
                )
        self._target = target
        self._post = post
        self._transport = transport
        self._timeout = httpx.Timeout(0.25, connect=0.1)
        self._deliveries: queue.Queue[_Delivery | None] = queue.Queue(capacity)
        self._close_lock = threading.Lock()
        self._closed = False
        self._stop_deliveries = 0
        prefix = f"rpent-event-notification-{id(self):x}"
        self._workers = tuple(
            threading.Thread(
                target=self._worker,
                name=f"{prefix}-{index}",
                daemon=True,
            )
            for index in range(workers)
        )
        for worker in self._workers:
            worker.start()

    def __call__(self, event: EventEnvelope) -> bool:
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be an EventEnvelope")
        try:
            validate_notification(event)
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        except (NotificationTooLargeError, ValueError):
            return False
        else:
            return False
        with self._close_lock:
            if self._closed:
                return False
            try:
                self._deliveries.put_nowait(
                    _Delivery(EventEnvelope.from_payload(event.to_payload()))
                )
            except queue.Full:
                return False
        return True

    def close(self) -> None:
        """Boundedly stop idle delivery workers; safe to retry or repeat."""
        deadline = time.monotonic() + 1.0
        with self._close_lock:
            self._closed = True
            while self._stop_deliveries < len(self._workers):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    self._deliveries.put(None, timeout=remaining)
                except queue.Full:
                    break
                self._stop_deliveries += 1
        for worker in self._workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(timeout=remaining)

    def _worker(self) -> None:
        while True:
            delivery = self._deliveries.get()
            try:
                if delivery is None:
                    return
                try:
                    asyncio.run(self._send(delivery))
                except BaseException:
                    pass
            finally:
                self._deliveries.task_done()

    async def _send(self, delivery: _Delivery) -> bool:
        return await asyncio.wait_for(self._request(delivery), timeout=0.25)

    async def _request(self, delivery: _Delivery) -> bool:
        target = self._target()
        if not isinstance(target, NotificationTarget):
            raise TypeError("target must return a NotificationTarget")
        origin = _http_origin(target.origin)
        token = (
            None
            if target.notification_token is None
            else _notification_token(target.notification_token)
        )
        response: httpx.Response | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
            trust_env=False,
        ) as client:
            try:
                post = self._post or client.post
                response = await post(
                    f"{origin}/api/internal/events",
                    json=delivery.event.to_payload(),
                    headers=(
                        {"X-RPent-Notification-Token": token}
                        if token is not None
                        else None
                    ),
                    timeout=self._timeout,
                )
                await response.aread()
                response.raise_for_status()
                return True
            finally:
                if response is not None:
                    await response.aclose()


def notification_sink(
    endpoint: Any = None,
    notification_token: Any = None,
    *,
    descriptor_path: Any = None,
    expected_runs_root: Any = None,
) -> HttpNotificationSink | NoopNotificationSink:
    if descriptor_path is not None:
        def resolve_target() -> NotificationTarget:
            from rpent.dashboard.attachment import DashboardDescriptor

            descriptor = DashboardDescriptor.load(
                descriptor_path,
                expected_runs_root=expected_runs_root,
                expected_descriptor_path=descriptor_path,
            )
            return NotificationTarget(
                descriptor.instance_id,
                descriptor.origin,
                descriptor.notification_token,
            )

        return HttpNotificationSink(target=resolve_target)
    if not isinstance(endpoint, str) or not endpoint.strip():
        return NoopNotificationSink()
    if not isinstance(notification_token, str) or not notification_token:
        return NoopNotificationSink()
    return HttpNotificationSink(endpoint, notification_token=notification_token)


def _notification_token(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError("notification_token must be a non-empty header value")
    return value


def validate_notification(event: EventEnvelope) -> int:
    """Reject media-bearing or oversized envelopes before network/SSE ingress."""
    if not isinstance(event, EventEnvelope):
        raise TypeError("event must be an EventEnvelope")
    _reject_media_fields(event.payload)
    try:
        encoded = json.dumps(
            event.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("notification must be strict JSON") from exc
    if len(encoded) > NOTIFICATION_MAX_BYTES:
        raise NotificationTooLargeError("notification exceeds 64 KiB")
    return len(encoded)


def _reject_media_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in _MEDIA_FIELDS:
                raise ValueError(f"notification media field is forbidden: {key}")
            _reject_media_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_media_fields(item)


def _http_origin(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty HTTP origin")
    selected = endpoint.strip()
    parsed = urlsplit(selected)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint must be a valid HTTP origin") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint must be an HTTP origin without path or credentials")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}" + (f":{port}" if port is not None else "")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "HttpNotificationSink",
    "NOTIFICATION_MAX_BYTES",
    "NoopNotificationSink",
    "NotificationTarget",
    "NotificationTooLargeError",
    "notification_sink",
    "validate_notification",
]
