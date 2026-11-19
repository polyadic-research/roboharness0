from __future__ import annotations

import asyncio
import functools
import threading
import time

import pytest

from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout, RepositoryReadGate


def _async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@_async_test
async def test_same_key_callers_share_one_underlying_result() -> None:
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()
    sentinel = object()
    calls = 0
    lock = threading.Lock()

    def blocked():
        nonlocal calls
        with lock:
            calls += 1
        started.set()
        assert release.wait(1)
        return sentinel

    tasks = [
        asyncio.create_task(gate.run(("case", "e1", "c1"), blocked)) for _ in range(50)
    ]
    try:
        await _wait_until(started.is_set)
        release.set()
        results = await asyncio.gather(*tasks)
        snapshot = gate.snapshot()

        assert calls == 1
        assert all(result is sentinel for result in results)
        assert snapshot.shared_callers == 49
        assert snapshot.completed == 1
        assert snapshot.flights == 0
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        gate.close()


@_async_test
async def test_different_keys_use_no_more_than_eight_workers() -> None:
    gate = RepositoryReadGate(workers=8, pending=16, default_timeout_s=1)
    release = threading.Event()
    running = 0
    maximum = 0
    lock = threading.Lock()

    def blocked(value: int) -> int:
        nonlocal maximum, running
        with lock:
            running += 1
            maximum = max(maximum, running)
        try:
            assert release.wait(1)
            return value
        finally:
            with lock:
                running -= 1

    tasks = [
        asyncio.create_task(
            gate.run(("evaluation", index), functools.partial(blocked, index))
        )
        for index in range(12)
    ]
    try:
        await _wait_until(lambda: gate.snapshot().running == 8)
        snapshot = gate.snapshot()
        assert snapshot.running == 8
        assert snapshot.queued == 4
        assert snapshot.flights == 12
        assert maximum == 8
        release.set()
        assert await asyncio.gather(*tasks) == list(range(12))
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        gate.close()


@_async_test
async def test_distinct_key_admission_is_immediately_bounded_at_twenty_four() -> None:
    gate = RepositoryReadGate(workers=8, pending=16, default_timeout_s=1)
    release = threading.Event()

    def blocked() -> None:
        assert release.wait(1)

    tasks = [
        asyncio.create_task(gate.run(("distinct", index), blocked))
        for index in range(24)
    ]
    try:
        await _wait_until(lambda: gate.snapshot().flights == 24)
        snapshot = gate.snapshot()
        assert snapshot.running == 8
        assert snapshot.queued == 16
        with pytest.raises(ReadGateBusy):
            await gate.run(("distinct", 24), blocked)
        assert gate.snapshot().rejected == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        gate.close()


@_async_test
async def test_snapshot_marks_rejection_and_one_second_at_capacity_as_saturated() -> (
    None
):
    gate = RepositoryReadGate(workers=1, pending=0, default_timeout_s=2)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(2)

    running = asyncio.create_task(gate.run(("full",), blocked))
    try:
        await _wait_until(started.is_set)
        assert gate.snapshot().saturated == 0
        with pytest.raises(ReadGateBusy):
            await gate.run(("rejected",), lambda: None)
        assert gate.snapshot().saturated == 1
        await asyncio.sleep(1.01)
        snapshot = gate.snapshot()
        assert snapshot.capacity_seconds >= 1
        assert snapshot.saturated == 1
    finally:
        release.set()
        await running
        gate.close()


@_async_test
async def test_timeout_keeps_the_underlying_slot_until_sync_work_finishes() -> None:
    gate = RepositoryReadGate(workers=1, pending=0, default_timeout_s=0.02)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(1)

    try:
        with pytest.raises(ReadGateTimeout):
            await gate.run(("case", "slow"), blocked)
        assert started.is_set()
        snapshot = gate.snapshot()
        assert snapshot.running == 1
        assert snapshot.flights == 1
        assert snapshot.timed_out == 1
        with pytest.raises(ReadGateBusy):
            await gate.run(("case", "different"), lambda: None)

        release.set()
        await _wait_until(lambda: gate.snapshot().flights == 0)
        assert await gate.run(("case", "different"), lambda: "finished") == "finished"
    finally:
        release.set()
        gate.close()


@_async_test
async def test_underlying_timeout_error_is_not_a_caller_deadline() -> None:
    gate = RepositoryReadGate(workers=1, pending=0, default_timeout_s=1)

    def fail() -> None:
        raise TimeoutError("repository-originated timeout")

    try:
        with pytest.raises(TimeoutError, match="repository-originated timeout") as exc:
            await gate.run(("case", "source-timeout"), fail)
        assert type(exc.value) is TimeoutError
        assert gate.snapshot().timed_out == 0
    finally:
        gate.close()


def test_loop_teardown_does_not_release_running_work_admission() -> None:
    gate = RepositoryReadGate(workers=1, pending=0, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(1)

    loop = asyncio.new_event_loop()
    loop.create_task(gate.run(("first-loop",), blocked))
    try:
        loop.run_until_complete(_wait_until(started.is_set))
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        loop.close()

    try:
        with pytest.raises(ReadGateBusy):
            asyncio.run(
                gate.run(("second-loop",), lambda: "should not run", timeout_s=0.02)
            )
    finally:
        release.set()
        deadline = time.monotonic() + 1
        while gate.snapshot().flights and time.monotonic() < deadline:
            time.sleep(0.005)
        gate.close()


def test_loop_teardown_preserves_same_key_single_flight() -> None:
    gate = RepositoryReadGate(workers=2, pending=0, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocked() -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        assert release.wait(1)
        return "shared"

    first_loop = asyncio.new_event_loop()
    first_loop.create_task(gate.run(("same-key",), blocked))
    try:
        first_loop.run_until_complete(_wait_until(started.is_set))
        pending = asyncio.all_tasks(first_loop)
        for task in pending:
            task.cancel()
        first_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        first_loop.close()

    async def join_from_second_loop() -> None:
        joined = asyncio.create_task(gate.run(("same-key",), blocked, timeout_s=0.5))
        try:
            await _wait_until(lambda: calls > 1 or gate.snapshot().shared_callers == 1)
            assert calls == 1
            assert gate.snapshot().shared_callers == 1
            release.set()
            assert await joined == "shared"
        finally:
            release.set()
            await asyncio.gather(joined, return_exceptions=True)

    try:
        asyncio.run(join_from_second_loop())
        assert gate.snapshot().completed == 1
        assert gate.snapshot().flights == 0
    finally:
        release.set()
        gate.close()


@_async_test
async def test_timed_out_key_remains_single_flight_for_retry() -> None:
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=0.02)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked() -> str:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(1)
        return "shared"

    try:
        with pytest.raises(ReadGateTimeout):
            await gate.run(("case", "slow"), blocked)
        retry = asyncio.create_task(gate.run(("case", "slow"), blocked, timeout_s=0.5))
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
        assert await retry == "shared"
        assert gate.snapshot().shared_callers == 1
    finally:
        release.set()
        gate.close()


@_async_test
async def test_cancelled_waiter_does_not_cancel_or_duplicate_sync_work() -> None:
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked() -> str:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(1)
        return "shared"

    first = asyncio.create_task(gate.run(("case", "cancelled"), blocked))
    try:
        await _wait_until(started.is_set)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        retry = asyncio.create_task(gate.run(("case", "cancelled"), blocked))
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
        assert await retry == "shared"
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        gate.close()


@_async_test
async def test_close_rejects_new_work_without_waiting_for_stuck_work() -> None:
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(1)

    task = asyncio.create_task(gate.run(("stuck",), blocked))
    try:
        await _wait_until(started.is_set)
        before = time.monotonic()
        gate.close()
        assert time.monotonic() - before < 0.2
        assert gate.snapshot().closed is True
        with pytest.raises(ReadGateBusy, match="closed"):
            await gate.run(("new",), lambda: None)
    finally:
        release.set()
        await task
        gate.close()


@_async_test
async def test_close_returns_cancelled_queue_capacity_to_the_snapshot() -> None:
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    started = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(1)

    running = asyncio.create_task(gate.run(("running",), blocked))
    queued = asyncio.create_task(gate.run(("queued",), lambda: None))
    try:
        await _wait_until(lambda: gate.snapshot().queued == 1)
        gate.close()
        await asyncio.gather(queued, return_exceptions=True)
        await _wait_until(lambda: gate.snapshot().flights == 1)
        snapshot = gate.snapshot()
        assert snapshot.queued == 0
        assert snapshot.running == 1
    finally:
        release.set()
        await asyncio.gather(running, queued, return_exceptions=True)
        gate.close()
