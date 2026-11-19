"""Bounded single-flight execution for Dashboard filesystem projections."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

_T = TypeVar("_T")
_SATURATION_SECONDS = 1.0


class ReadGateBusy(RuntimeError):
    """The bounded read executor cannot admit a distinct key."""

    def __init__(self, key: Hashable, *, closed: bool = False) -> None:
        self.key = key
        super().__init__(
            "repository read gate is closed"
            if closed
            else "repository read gate is busy"
        )


class ReadGateTimeout(TimeoutError):
    """One caller's deadline elapsed while bounded work continues."""

    def __init__(self, key: Hashable) -> None:
        self.key = key
        super().__init__("repository read timed out")


@dataclass(frozen=True, slots=True)
class ReadGateSnapshot:
    """Bounded counters and gauges safe for in-memory diagnostics."""

    workers: int
    pending: int
    capacity: int
    running: int
    queued: int
    flights: int
    shared_callers: int
    rejected: int
    timed_out: int
    completed: int
    capacity_seconds: float
    saturated: int
    closed: bool


@dataclass(slots=True)
class _Invocation:
    function: Callable[[], Any]
    state: str = "queued"


class RepositoryReadGate:
    """Run at most one bounded executor task for each semantic query key."""

    def __init__(
        self,
        *,
        workers: int = 8,
        pending: int = 16,
        default_timeout_s: float = 2.0,
        thread_name_prefix: str = "rpent-ui-read",
    ) -> None:
        self.workers = _positive_integer(workers, "workers")
        self.pending = _non_negative_integer(pending, "pending")
        self.default_timeout_s = _positive_number(
            default_timeout_s, "default_timeout_s"
        )
        self.capacity = self.workers + self.pending
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._flight_lock = threading.Lock()
        self._flights: dict[Hashable, Future[Any]] = {}
        self._stats_lock = threading.Lock()
        self._running = 0
        self._queued = 0
        self._flight_count = 0
        self._shared_callers = 0
        self._rejected = 0
        self._timed_out = 0
        self._completed = 0
        self._capacity_since: float | None = None
        self._last_rejected_at: float | None = None
        self._closed = False

    async def run(
        self,
        key: Hashable,
        function: Callable[[], _T],
        *,
        timeout_s: float | None = None,
    ) -> _T:
        """Run or join one keyed call without cancelling its underlying work."""
        hash(key)
        selected_timeout = (
            self.default_timeout_s
            if timeout_s is None
            else _non_negative_number(timeout_s, "timeout_s")
        )
        new_flight: tuple[_Invocation, Future[Any]] | None = None
        with self._flight_lock:
            concurrent_future = self._flights.get(key)
            if concurrent_future is None:
                with self._stats_lock:
                    closed = self._closed
                if closed or not self._slots.acquire(blocking=False):
                    self._record_rejection()
                    raise ReadGateBusy(key, closed=closed)
                with self._stats_lock:
                    if self._closed:
                        self._slots.release()
                        self._record_rejection_locked()
                        raise ReadGateBusy(key, closed=True)
                    self._queued += 1
                    self._flight_count += 1
                    if self._flight_count == self.capacity:
                        self._capacity_since = time.monotonic()
                invocation = _Invocation(function)
                try:
                    concurrent_future = self._executor.submit(
                        self._run_sync, invocation
                    )
                except BaseException:
                    self._release_invocation(invocation)
                    raise
                self._flights[key] = concurrent_future
                new_flight = invocation, concurrent_future
            else:
                with self._stats_lock:
                    self._shared_callers += 1
        if new_flight is not None:
            invocation, concurrent_future = new_flight
            concurrent_future.add_done_callback(
                lambda future: self._finish_invocation(key, invocation, future)
            )
        loop = asyncio.get_running_loop()
        future = asyncio.wrap_future(concurrent_future, loop=loop)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=selected_timeout,
            )
        except TimeoutError as exc:
            if concurrent_future.done() and not concurrent_future.cancelled():
                underlying = concurrent_future.exception()
                if isinstance(underlying, TimeoutError):
                    raise underlying
            with self._stats_lock:
                self._timed_out += 1
            raise ReadGateTimeout(key) from exc

    def snapshot(self) -> ReadGateSnapshot:
        """Return lock-bounded gauges without touching the repository."""
        now = time.monotonic()
        with self._stats_lock:
            capacity_seconds = (
                0.0
                if self._capacity_since is None
                else max(0.0, now - self._capacity_since)
            )
            recently_rejected = (
                self._last_rejected_at is not None
                and now - self._last_rejected_at < _SATURATION_SECONDS
            )
            saturated = recently_rejected or capacity_seconds >= _SATURATION_SECONDS
            return ReadGateSnapshot(
                workers=self.workers,
                pending=self.pending,
                capacity=self.capacity,
                running=self._running,
                queued=self._queued,
                flights=self._flight_count,
                shared_callers=self._shared_callers,
                rejected=self._rejected,
                timed_out=self._timed_out,
                completed=self._completed,
                capacity_seconds=capacity_seconds,
                saturated=int(saturated),
                closed=self._closed,
            )

    def close(self) -> None:
        """Reject new calls and request queued cancellation without blocking."""
        with self._stats_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _finish_invocation(
        self,
        key: Hashable,
        invocation: _Invocation,
        future: Future[Any],
    ) -> None:
        with self._flight_lock:
            if self._flights.get(key) is future:
                del self._flights[key]
        self._release_invocation(invocation)

    def _release_invocation(self, invocation: _Invocation) -> None:
        with self._stats_lock:
            if invocation.state == "queued":
                invocation.state = "cancelled"
                self._queued -= 1
            self._flight_count -= 1
            if self._flight_count < self.capacity:
                self._capacity_since = None
        self._slots.release()

    def _run_sync(self, invocation: _Invocation) -> _T:
        with self._stats_lock:
            invocation.state = "running"
            self._queued -= 1
            self._running += 1
        try:
            return invocation.function()
        finally:
            with self._stats_lock:
                invocation.state = "completed"
                self._running -= 1
                self._completed += 1

    def _record_rejection(self) -> None:
        with self._stats_lock:
            self._record_rejection_locked()

    def _record_rejection_locked(self) -> None:
        self._rejected += 1
        self._last_rejected_at = time.monotonic()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_number(value: float, name: str) -> float:
    selected = _non_negative_number(value, name)
    if selected == 0:
        raise ValueError(f"{name} must be a finite positive number")
    return selected


def _non_negative_number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


__all__ = [
    "ReadGateBusy",
    "ReadGateSnapshot",
    "ReadGateTimeout",
    "RepositoryReadGate",
]
