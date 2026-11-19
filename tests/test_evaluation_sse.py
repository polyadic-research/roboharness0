from __future__ import annotations

import asyncio
import functools
import inspect
import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from rpent.recording.contracts import EventEnvelope
from rpent.ui.app import create_evaluation_app
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.schemas import EventPage
from rpent.ui.sse import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_SNAPSHOT_MAX_BYTES,
    ReservedStreamingResponse,
    SSECapacityError,
    SSEManager,
    SSEPayloadTooLargeError,
    encode_sse,
)
from rpent.ui.sse import (
    case_event_stream as _case_event_stream,
)
from rpent.ui.sse import (
    evaluation_event_stream as _evaluation_event_stream,
)


def _async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _managed(function, *args, **kwargs):
    manager = SSEManager(repository_workers=2, repository_queue=2)
    stream = function(*args[:3], manager, *args[3:], **kwargs)

    async def body():
        try:
            async for frame in stream:
                yield frame
        finally:
            await stream.aclose()
            manager.close()

    return body()


def evaluation_event_stream(*args, **kwargs):
    return _managed(_evaluation_event_stream, *args, **kwargs)


def case_event_stream(*args, **kwargs):
    return _managed(_case_event_stream, *args, **kwargs)


def test_sse_routes_are_long_lived_by_default() -> None:
    from rpent.ui.routes.cases import stream_case
    from rpent.ui.routes.evaluations import stream_evaluation

    assert inspect.signature(stream_evaluation).parameters["timeout"].default is None
    assert inspect.signature(stream_case).parameters["timeout"].default is None


def test_default_fallback_reconciliation_is_one_second() -> None:
    assert DEFAULT_POLL_INTERVAL == 1.0


def test_large_authoritative_snapshot_is_not_limited_like_a_notification() -> None:
    frame = encode_sse("snapshot", {"evaluation": {"manifest": "x" * 70_000}})

    assert frame.startswith("event: snapshot\n")
    assert len(frame) > 70_000


def test_authoritative_snapshot_has_independent_16_mibibyte_limit() -> None:
    with pytest.raises(SSEPayloadTooLargeError, match="snapshot"):
        encode_sse(
            "snapshot",
            {"evaluation": "x" * (DEFAULT_SNAPSHOT_MAX_BYTES + 1)},
        )


class _Request:
    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


@dataclass
class _Case:
    attempt_id: str | None
    execution_state: str

    def model_dump(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "execution_state": self.execution_state,
        }


class _Repository:
    def __init__(self) -> None:
        self.attempt_id = "attempt-1"
        self.execution_state = "running"
        self.events: list[dict] = []
        self.evaluation = SimpleNamespace(
            model_dump=lambda: {"execution_state": self.execution_state}
        )

    def get_case(self, evaluation_id: str, case_id: str) -> _Case:
        assert (evaluation_id, case_id) == ("evaluation-1", "case-1")
        return _Case(self.attempt_id, self.execution_state)

    def events_since(
        self, evaluation_id: str, case_id: str, *, since: int
    ) -> EventPage:
        self.get_case(evaluation_id, case_id)
        selected = tuple(event for event in self.events if event["sequence"] > since)
        return EventPage(
            evaluation_id=evaluation_id,
            case_id=case_id,
            attempt_id=self.attempt_id,
            since=since,
            last_sequence=selected[-1]["sequence"] if selected else since,
            incomplete_tail=False,
            has_more=False,
            events=selected,
        )

    def get_evaluation(self, evaluation_id: str):
        assert evaluation_id == "evaluation-1"
        return self.evaluation

    def event_checkpoint(self, evaluation_id: str) -> dict:
        self.get_evaluation(evaluation_id)
        return {
            "case-1": {
                "attempt_id": self.attempt_id,
                "sequence": len(self.events),
            }
        }


def _event(
    sequence: int,
    *,
    attempt_id: str = "attempt-1",
    event_type: str = "execution_state",
    payload: dict | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        schema_version=1,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id=attempt_id,
        sequence=sequence,
        recorded_at=f"2026-08-07T00:00:{sequence % 60:02d}+00:00",
        event_type=event_type,
        step_id=None,
        operation_id=None,
        execution_id=None,
        payload={} if payload is None else payload,
    )


def _decode(frame: str) -> tuple[str, str | None, dict]:
    fields = {}
    for line in frame.splitlines():
        name, _, value = line.partition(": ")
        fields[name] = value
    return fields["event"], fields.get("id"), json.loads(fields["data"])


async def _next(generator):
    return await asyncio.wait_for(anext(generator), timeout=1)


@_async_test
async def test_case_catches_up_since_then_delivers_live_once() -> None:
    repository = _Repository()
    repository.events = [_event(sequence).to_payload() for sequence in range(1, 6)]
    hub = EvaluationEventHub(secret=b"test-secret")
    request = _Request()
    stream = case_event_stream(
        request,
        repository,
        hub,
        "evaluation-1",
        "case-1",
        since=4,
        heartbeat_interval=0.05,
        poll_interval=0.01,
    )
    try:
        name, _, payload = _decode(await _next(stream))
        assert (name, payload["sequence"]) == ("case_event", 5)

        # The durable poll must advance even when best-effort notification fails.
        repository.events.append(_event(6).to_payload())
        while True:
            name, _, payload = _decode(await _next(stream))
            if name == "case_event":
                break
        assert payload["sequence"] == 6

        repository.events.append(_event(7).to_payload())
        hub.publish(_event(6))
        hub.publish(_event(7))
        while True:
            name, _, payload = _decode(await _next(stream))
            if name == "case_event":
                break
        assert payload["sequence"] == 7
    finally:
        await stream.aclose()


@_async_test
async def test_case_attempt_change_snapshots_then_restarts_at_one() -> None:
    repository = _Repository()
    repository.events = [_event(1).to_payload()]
    hub = EvaluationEventHub(secret=b"test-secret")
    request = _Request()
    cursor = hub.case_cursor("evaluation-1", "case-1", "attempt-1", 1)
    stream = case_event_stream(
        request,
        repository,
        hub,
        "evaluation-1",
        "case-1",
        cursor=cursor,
        heartbeat_interval=0.05,
        poll_interval=0.01,
    )
    try:
        repository.attempt_id = "attempt-2"
        repository.events = [_event(1, attempt_id="attempt-2").to_payload()]
        hub.publish(_event(1, attempt_id="attempt-2"))
        assert _decode(await _next(stream))[0] == "snapshot"
        name, _, payload = _decode(await _next(stream))
        assert (name, payload["attempt_id"], payload["sequence"]) == (
            "case_event",
            "attempt-2",
            1,
        )
    finally:
        await stream.aclose()


@_async_test
async def test_expired_evaluation_cursor_snapshots_durable_checkpoints() -> None:
    repository = _Repository()
    repository.events = [_event(sequence).to_payload() for sequence in range(1, 4)]
    hub = EvaluationEventHub(capacity=1, secret=b"test-secret")
    expired = hub.cursor("evaluation-1")
    hub.publish(_event(1))
    hub.publish(_event(2))
    request = _Request()
    stream = evaluation_event_stream(
        request,
        repository,
        hub,
        "evaluation-1",
        cursor=expired,
        heartbeat_interval=0.05,
        poll_interval=0.01,
    )
    try:
        name, fresh_cursor, payload = _decode(await _next(stream))
        assert name == "snapshot"
        assert fresh_cursor
        assert payload["last_sequences"]["case-1"] == {
            "attempt_id": "attempt-1",
            "sequence": 3,
        }

        hub.publish(_event(3))
        while True:
            name, _, payload = _decode(await _next(stream))
            if name == "evaluation_event":
                break
        assert payload["sequence"] == 3
    finally:
        await stream.aclose()


@_async_test
async def test_evaluation_polls_authoritative_terminal_state_without_notification() -> (
    None
):
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    stream = evaluation_event_stream(
        _Request(),
        repository,
        hub,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
        heartbeat_interval=1,
        poll_interval=0.01,
    )
    try:
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.03)
        repository.execution_state = "completed"
        name, _, payload = _decode(await asyncio.wait_for(pending, 1))
        assert name == "snapshot"
        assert payload["evaluation"]["execution_state"] == "completed"
    finally:
        await stream.aclose()


@_async_test
async def test_coalesced_thumbnail_wake_cannot_drop_terminal_hub_event() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(capacity=64, secret=b"test-secret")
    stream = evaluation_event_stream(
        _Request(),
        repository,
        hub,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
        heartbeat_interval=1,
        poll_interval=0.005,
    )
    try:
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.02)
        hub.publish(_event(1, event_type="thumbnail_available"))
        hub.publish(_event(2, event_type="terminal"))

        first = _decode(await asyncio.wait_for(pending, 1))[2]
        second = _decode(await _next(stream))[2]
        assert [
            (first["sequence"], first["event_type"]),
            (second["sequence"], second["event_type"]),
        ] == [
            (1, "thumbnail_available"),
            (2, "terminal"),
        ]
    finally:
        await stream.aclose()


@_async_test
async def test_case_polls_authoritative_terminal_state_without_notification() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    stream = case_event_stream(
        _Request(),
        repository,
        hub,
        "evaluation-1",
        "case-1",
        since=0,
        heartbeat_interval=1,
        poll_interval=0.01,
    )
    try:
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.03)
        repository.execution_state = "completed"
        name, _, payload = _decode(await asyncio.wait_for(pending, 1))
        assert name == "snapshot"
        assert payload["case"]["execution_state"] == "completed"
    finally:
        await stream.aclose()


@_async_test
async def test_case_projection_snapshot_keeps_cursor_then_delivers_same_round_events() -> (
    None
):
    class ProjectionRepository(_Repository):
        def get_case(self, evaluation_id: str, case_id: str):
            assert (evaluation_id, case_id) == ("evaluation-1", "case-1")
            payload = {
                "attempt_id": self.attempt_id,
                "execution_state": self.execution_state,
                "last_sequence": len(self.events),
            }
            return SimpleNamespace(
                attempt_id=self.attempt_id,
                model_dump=lambda: payload,
            )

    repository = ProjectionRepository()
    repository.events = [_event(sequence).to_payload() for sequence in range(1, 5)]
    stream = case_event_stream(
        _Request(),
        repository,
        EvaluationEventHub(secret=b"test-secret"),
        "evaluation-1",
        "case-1",
        since=4,
        heartbeat_interval=1,
        poll_interval=0.005,
    )
    try:
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.02)
        repository.execution_state = "completed"
        repository.events.extend(
            _event(sequence).to_payload() for sequence in range(5, 10)
        )

        name, _, snapshot = _decode(await asyncio.wait_for(pending, 1))
        assert name == "snapshot"
        assert snapshot["last_sequence"] == 4
        assert snapshot["case"]["last_sequence"] == 9
        delivered = []
        while len(delivered) < 5:
            event_name, _, payload = _decode(await _next(stream))
            if event_name == "case_event":
                delivered.append(payload["sequence"])
        assert delivered == [5, 6, 7, 8, 9]
    finally:
        await stream.aclose()


@_async_test
async def test_case_slow_subscriber_uses_bounded_queue_and_durable_resync() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(capacity=2, secret=b"test-secret")
    request = _Request()
    stream = case_event_stream(
        request,
        repository,
        hub,
        "evaluation-1",
        "case-1",
        since=0,
        heartbeat_interval=1,
        poll_interval=0.005,
    )
    try:
        first = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        for sequence in range(1, 21):
            event = _event(sequence)
            repository.events.append(event.to_payload())
            hub.publish(event)
        observed = []
        observed.append(_decode(await asyncio.wait_for(first, 1))[2]["sequence"])
        while observed[-1] < 20:
            name, _, payload = _decode(await _next(stream))
            if name == "case_event":
                observed.append(payload["sequence"])
        assert observed == list(range(1, 21))
    finally:
        await stream.aclose()


@_async_test
async def test_disconnect_stops_subscription_tasks_without_waiting_for_heartbeat() -> (
    None
):
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    request = _Request()
    before = set(asyncio.all_tasks())
    stream = evaluation_event_stream(
        request,
        repository,
        hub,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
        heartbeat_interval=60,
        poll_interval=0.01,
    )
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.03)
    request.disconnected = True
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, 1)
    await stream.aclose()
    await asyncio.sleep(0.03)
    leaked = [task for task in asyncio.all_tasks() - before if not task.done()]
    assert leaked == []


@_async_test
async def test_repository_reads_do_not_block_the_event_loop() -> None:
    class SlowRepository(_Repository):
        def get_evaluation(self, evaluation_id: str):
            time.sleep(0.08)
            return super().get_evaluation(evaluation_id)

    repository = SlowRepository()
    hub = EvaluationEventHub(secret=b"test-secret")
    stream = evaluation_event_stream(
        _Request(),
        repository,
        hub,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
        heartbeat_interval=1,
        poll_interval=0.01,
    )
    pending = asyncio.create_task(anext(stream))
    started = time.monotonic()
    await asyncio.sleep(0.01)
    elapsed = time.monotonic() - started
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await stream.aclose()

    assert elapsed < 0.04


@_async_test
async def test_sse_refuses_media_bytes_even_if_repository_is_compromised() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    # Bypass Hub ingress to exercise the final SSE boundary through the journal.
    repository.events = [_event(1, payload={"image_bytes": "secret"}).to_payload()]
    stream = case_event_stream(
        _Request(),
        repository,
        hub,
        "evaluation-1",
        "case-1",
        since=0,
        heartbeat_interval=0.01,
        poll_interval=0.005,
    )
    with pytest.raises(ValueError, match="media field"):
        await _next(stream)
    await stream.aclose()


@_async_test
async def test_evaluation_snapshot_refuses_repository_media_bytes() -> None:
    repository = _Repository()
    repository.evaluation = SimpleNamespace(
        model_dump=lambda: {"nested": {"image_bytes": "secret"}}
    )
    hub = EvaluationEventHub(capacity=1, secret=b"test-secret")
    expired = hub.cursor("evaluation-1")
    hub.publish(_event(1))
    hub.publish(_event(2))
    stream = evaluation_event_stream(
        _Request(), repository, hub, "evaluation-1", cursor=expired
    )
    with pytest.raises(ValueError, match="media field"):
        await _next(stream)
    await stream.aclose()


@_async_test
async def test_expired_snapshot_works_with_one_repository_slot() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(capacity=1, secret=b"test-secret")
    expired = hub.cursor("evaluation-1")
    hub.publish(_event(1))
    hub.publish(_event(2))
    manager = SSEManager(repository_workers=1, repository_queue=0)
    stream = _evaluation_event_stream(
        _Request(),
        repository,
        hub,
        manager,
        "evaluation-1",
        cursor=expired,
    )
    try:
        frame = await _next(stream)
        assert "event: snapshot\n" in frame
        assert "event: overload\n" not in frame
    finally:
        await stream.aclose()
        manager.close()


@_async_test
async def test_manager_close_race_ends_stream_without_false_overload() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)

    class ClosingRequest(_Request):
        async def is_disconnected(self) -> bool:
            manager.close()
            return False

    stream = _evaluation_event_stream(
        ClosingRequest(),
        repository,
        hub,
        manager,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
    )
    try:
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        await stream.aclose()


@_async_test
async def test_established_stream_emits_one_overload_frame_then_closes() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(1)

    occupying = asyncio.create_task(manager.repository_call(blocked))
    stream = _evaluation_event_stream(
        _Request(),
        repository,
        hub,
        manager,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
    )
    try:
        while not started.is_set():
            await asyncio.sleep(0.005)
        frame = await asyncio.wait_for(anext(stream), 0.2)
        assert "event: overload\n" in frame
        assert "retry: 2000\n" in frame
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    finally:
        await stream.aclose()
        release.set()
        await occupying
        manager.close()
    assert hub.async_subscriber_count == 0


def test_manager_enforces_global_connection_limit_and_releases() -> None:
    manager = SSEManager(max_connections=2)
    first = manager.reserve()
    second = manager.reserve()

    with pytest.raises(SSECapacityError, match="limit"):
        manager.reserve()
    first.release()
    replacement = manager.reserve()
    replacement.release()
    second.release()
    assert manager.active_connections == 0
    manager.close()


@_async_test
async def test_repository_executor_submission_is_bounded_and_dedicated() -> None:
    manager = SSEManager(repository_workers=2, repository_queue=2)
    preexisting_sse_threads = {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("rpent-sse-repository")
    }
    release = threading.Event()
    started = 0
    lock = threading.Lock()

    def blocked():
        nonlocal started
        with lock:
            started += 1
        release.wait(1)

    tasks = [asyncio.create_task(manager.repository_call(blocked)) for _ in range(12)]
    await asyncio.sleep(0.05)
    assert started == 2
    assert manager.submitted_repository_calls == 4
    assert not any(
        thread.name.startswith("asyncio")
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
    )
    owned_threads = {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("rpent-sse-repository")
    } - preexisting_sse_threads

    for task in tasks:
        task.cancel()
    release.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    manager.close()
    assert not any(thread.is_alive() for thread in owned_threads)


@_async_test
async def test_full_repository_executor_rejects_immediately() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=0)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(1)

    running = asyncio.create_task(manager.repository_call(blocked))
    try:
        while not started.is_set():
            await asyncio.sleep(0.005)
        before = time.monotonic()
        with pytest.raises(SSECapacityError, match="capacity"):
            await asyncio.wait_for(manager.repository_call(lambda: None), 0.1)
        assert time.monotonic() - before < 0.1
        assert manager.submitted_repository_calls == 1
    finally:
        release.set()
        await running
        manager.close()


@_async_test
async def test_repository_deadline_keeps_underlying_work_counted() -> None:
    manager = SSEManager(
        repository_workers=1,
        repository_queue=0,
        repository_timeout_s=0.02,
    )
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(1)

    try:
        with pytest.raises(SSECapacityError, match="timed out"):
            await manager.repository_call(blocked)
        assert started.is_set()
        assert manager.submitted_repository_calls == 1
        with pytest.raises(SSECapacityError, match="capacity"):
            await manager.repository_call(lambda: None)
    finally:
        release.set()
        deadline = time.monotonic() + 1
        while manager.submitted_repository_calls and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        manager.close()
    assert manager.submitted_repository_calls == 0


@_async_test
async def test_repository_failure_releases_manager_submission_slot() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=0)

    def fail():
        raise OSError("unavailable")

    with pytest.raises(OSError, match="unavailable"):
        await manager.repository_call(fail)
    assert manager.submitted_repository_calls == 0
    assert await manager.repository_call(lambda: "ok") == "ok"
    manager.close()


@_async_test
async def test_repository_timeout_error_is_not_a_capacity_deadline() -> None:
    manager = SSEManager(
        repository_workers=1,
        repository_queue=0,
        repository_timeout_s=1,
    )

    def fail() -> None:
        raise TimeoutError("repository-originated timeout")

    try:
        with pytest.raises(TimeoutError, match="repository-originated timeout") as exc:
            await manager.repository_call(fail)
        assert type(exc.value) is TimeoutError
        assert manager.submitted_repository_calls == 0
    finally:
        manager.close()


@_async_test
async def test_manager_shutdown_rejects_repository_waiters_without_queue_growth() -> (
    None
):
    manager = SSEManager(repository_workers=1, repository_queue=0)
    release = threading.Event()
    running = asyncio.create_task(manager.repository_call(release.wait, 1))
    await asyncio.sleep(0.02)
    waiters = [
        asyncio.create_task(manager.repository_call(lambda: None)) for _ in range(5)
    ]
    await asyncio.sleep(0.02)

    manager.close()
    results = await asyncio.wait_for(
        asyncio.gather(*waiters, return_exceptions=True), 0.3
    )
    assert all(isinstance(result, SSECapacityError) for result in results)
    assert manager.submitted_repository_calls == 1
    release.set()
    await running


def test_manager_shutdown_rejects_new_connections_and_is_idempotent() -> None:
    manager = SSEManager()
    connection = manager.reserve()

    manager.close()
    manager.close()
    with pytest.raises(SSECapacityError, match="closed"):
        manager.reserve()
    connection.release()
    assert manager.active_connections == 0


def test_app_shutdown_closes_owned_sse_manager() -> None:
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"test-secret"),
        notification_token="notification",
    )
    manager = app.state.sse_manager

    with TestClient(app):
        assert manager.closed is False
    assert manager.closed is True


def test_app_shutdown_does_not_close_injected_sse_manager() -> None:
    manager = SSEManager()
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"test-secret"),
        notification_token="notification",
        sse_manager=manager,
    )

    with TestClient(app):
        pass
    assert manager.closed is False
    manager.close()


def test_app_shutdown_closes_owned_read_gates() -> None:
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"test-secret"),
        notification_token="notification",
    )
    repository_gate = app.state.repository_read_gate
    media_gate = app.state.media_read_gate

    with TestClient(app):
        assert repository_gate.snapshot().closed is False
        assert media_gate.snapshot().closed is False

    assert repository_gate.snapshot().closed is True
    assert media_gate.snapshot().closed is True


def test_app_shutdown_does_not_close_injected_read_gates() -> None:
    from rpent.ui.io_gate import RepositoryReadGate

    repository_gate = RepositoryReadGate()
    media_gate = RepositoryReadGate(workers=4, pending=4)
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"test-secret"),
        notification_token="notification",
        repository_read_gate=repository_gate,
        media_read_gate=media_gate,
    )

    with TestClient(app):
        pass

    assert repository_gate.snapshot().closed is False
    assert media_gate.snapshot().closed is False
    repository_gate.close()
    media_gate.close()


def test_repository_limiter_is_stable_across_event_loops_with_contention() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=1)
    barrier = threading.Barrier(2)
    results = []

    def worker(value: int) -> None:
        barrier.wait()
        results.append(asyncio.run(manager.repository_call(lambda: value)))

    threads = [threading.Thread(target=worker, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [1, 2]
    manager.close()


def test_cancelled_call_releases_slot_after_originating_loop_is_closed() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=0)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(2)

    async def cancel_from_first_loop() -> None:
        task = asyncio.create_task(manager.repository_call(blocked))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(cancel_from_first_loop())
    assert manager.submitted_repository_calls == 1
    release.set()
    deadline = time.monotonic() + 1
    while manager.submitted_repository_calls and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.submitted_repository_calls == 0
    assert asyncio.run(manager.repository_call(lambda: "second-loop")) == "second-loop"
    manager.close()


def test_close_cancels_queued_future_and_returns_all_repository_slots() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=1)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(2)

    async def submit_then_close_loop() -> None:
        running = asyncio.create_task(manager.repository_call(blocked))
        while not started.is_set():
            await asyncio.sleep(0.01)
        queued = asyncio.create_task(manager.repository_call(lambda: "queued"))
        while manager.submitted_repository_calls != 2:
            await asyncio.sleep(0.01)
        running.cancel()
        queued.cancel()
        await asyncio.gather(running, queued, return_exceptions=True)

    asyncio.run(submit_then_close_loop())
    manager.close()
    deadline = time.monotonic() + 1
    while manager.submitted_repository_calls > 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.submitted_repository_calls == 1

    release.set()
    deadline = time.monotonic() + 1
    while manager.submitted_repository_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.submitted_repository_calls == 0


def test_repository_submit_failure_rolls_back_capacity() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=0)
    submit = manager._executor.submit

    def reject(*args, **kwargs):
        raise RuntimeError("executor rejected submission")

    manager._executor.submit = reject
    with pytest.raises(RuntimeError, match="rejected"):
        asyncio.run(manager.repository_call(lambda: None))
    assert manager.submitted_repository_calls == 0

    manager._executor.submit = submit
    assert asyncio.run(manager.repository_call(lambda: "recovered")) == "recovered"
    manager.close()


@_async_test
async def test_reserved_streaming_response_releases_on_response_start_failure() -> None:
    manager = SSEManager()
    response = ReservedStreamingResponse(iter(["frame"]), manager.reserve())

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("send failed")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }
    with pytest.raises(Exception):
        await response(scope, receive, send)
    assert manager.active_connections == 0
    manager.close()


@_async_test
async def test_reserved_response_preserves_transport_error_if_close_raises() -> None:
    class CloseFailureStream:
        def __init__(self) -> None:
            self.sent = False
            self.close_calls = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return "frame"

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError("stream close failed")

    manager = SSEManager()
    stream = CloseFailureStream()
    response = ReservedStreamingResponse(stream, manager.reserve())

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("send failed")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }
    try:
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert stream.close_calls == 1
        assert manager.active_connections == 0
    finally:
        manager.close()


@_async_test
async def test_reserved_streaming_response_releases_on_body_failure_and_completion() -> (
    None
):
    manager = SSEManager()

    async def fail():
        raise OSError("first frame failed")
        yield "unreachable"

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        return None

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }
    response = ReservedStreamingResponse(fail(), manager.reserve())
    with pytest.raises(Exception):
        await response(scope, receive, send)
    assert manager.active_connections == 0

    response = ReservedStreamingResponse(iter(["ok"]), manager.reserve())
    await response(scope, receive, send)
    assert manager.active_connections == 0
    manager.close()


@_async_test
async def test_reserved_streaming_response_releases_on_cancellation() -> None:
    manager = SSEManager()

    async def content():
        await asyncio.Event().wait()
        yield "unreachable"

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        return None

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }
    response = ReservedStreamingResponse(content(), manager.reserve())
    task = asyncio.create_task(response(scope, receive, send))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.active_connections == 0
    manager.close()


def test_route_connection_cap_covers_finite_stream_and_releases_after_body() -> None:
    manager = SSEManager(max_connections=1)
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"test-secret"),
        notification_token="notification",
        sse_manager=manager,
    )
    held = manager.reserve()
    client = TestClient(app)

    assert (
        client.get("/api/evaluations/evaluation-1/stream?timeout=0").status_code == 503
    )
    held.release()
    for _ in range(20):
        assert (
            client.get("/api/evaluations/evaluation-1/stream?timeout=0").status_code
            == 200
        )
        assert manager.active_connections == 0
    manager.close()


def test_route_repository_capacity_returns_retry_after_two() -> None:
    manager = SSEManager(repository_workers=1, repository_queue=0)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(1)

    worker = threading.Thread(
        target=lambda: asyncio.run(manager.repository_call(blocked))
    )
    worker.start()
    assert started.wait(1)
    try:
        app = create_evaluation_app(
            repository=_Repository(),
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"test-secret"),
            notification_token="notification",
            sse_manager=manager,
        )

        response = TestClient(app).get("/api/evaluations/evaluation-1/stream?timeout=0")

        assert response.status_code == 503
        assert response.headers["retry-after"] == "2"
    finally:
        release.set()
        worker.join(1)
        manager.close()


def test_finite_evaluation_stream_returns_one_bounded_page() -> None:
    manager = SSEManager()
    hub = EvaluationEventHub(capacity=100, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")
    for sequence in range(1, 81):
        hub.publish(_event(sequence))
    app = create_evaluation_app(
        repository=_Repository(),
        service=SimpleNamespace(),
        event_hub=hub,
        notification_token="notification",
        sse_manager=manager,
    )

    response = TestClient(app).get(
        "/api/evaluations/evaluation-1/stream",
        params={"cursor": cursor, "timeout": 0},
    )

    assert response.status_code == 200
    assert response.text.count("event: evaluation_event") == 64
    manager.close()


@_async_test
async def test_manager_shutdown_ends_stream_and_unregisters_subscription() -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)
    stream = _evaluation_event_stream(
        _Request(),
        repository,
        hub,
        manager,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
        heartbeat_interval=60,
        poll_interval=0.01,
    )
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.03)
    assert hub.async_subscriber_count == 1

    manager.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, 1)
    await stream.aclose()
    assert hub.async_subscriber_count == 0


@pytest.mark.parametrize("stream_kind", ["evaluation", "case"])
@_async_test
async def test_public_stream_close_unregisters_inner_subscription(
    stream_kind: str,
) -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)
    if stream_kind == "evaluation":
        cursor = hub.cursor("evaluation-1")
        hub.publish(_event(1))
        stream = _evaluation_event_stream(
            _Request(),
            repository,
            hub,
            manager,
            "evaluation-1",
            cursor=cursor,
        )
    else:
        repository.events = [_event(1).to_payload()]
        stream = _case_event_stream(
            _Request(),
            repository,
            hub,
            manager,
            "evaluation-1",
            "case-1",
            since=0,
        )

    try:
        await _next(stream)
        assert hub.async_subscriber_count == 1
        await stream.aclose()
        assert hub.async_subscriber_count == 0
    finally:
        manager.close()


@pytest.mark.parametrize("stream_kind", ["evaluation", "case"])
@_async_test
async def test_response_send_failure_unregisters_inner_subscription(
    stream_kind: str,
) -> None:
    repository = _Repository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)
    if stream_kind == "evaluation":
        cursor = hub.cursor("evaluation-1")
        hub.publish(_event(1))
        stream = _evaluation_event_stream(
            _Request(),
            repository,
            hub,
            manager,
            "evaluation-1",
            cursor=cursor,
        )
    else:
        repository.events = [_event(1).to_payload()]
        stream = _case_event_stream(
            _Request(),
            repository,
            hub,
            manager,
            "evaluation-1",
            "case-1",
            since=0,
        )
    response = ReservedStreamingResponse(
        stream,
        manager.reserve(),
        media_type="text/event-stream",
    )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("body send failed")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }
    try:
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert manager.active_connections == 0
        assert hub.async_subscriber_count == 0
    finally:
        manager.close()


@_async_test
async def test_initial_repository_failure_unregisters_hub_subscription() -> None:
    class BrokenRepository(_Repository):
        def get_evaluation(self, evaluation_id: str):
            raise OSError("projection unavailable")

    repository = BrokenRepository()
    hub = EvaluationEventHub(secret=b"test-secret")
    manager = SSEManager(repository_workers=1, repository_queue=0)
    stream = _evaluation_event_stream(
        _Request(),
        repository,
        hub,
        manager,
        "evaluation-1",
        cursor=hub.cursor("evaluation-1"),
    )

    with pytest.raises(OSError, match="projection unavailable"):
        await anext(stream)
    await stream.aclose()
    assert hub.async_subscriber_count == 0
    manager.close()
