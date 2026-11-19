"""Shared episode liveness budget, including nested model-only planning."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock


class EpisodeBudgetExceeded(RuntimeError):
    """An episode exhausted its independent model or wall-time budget."""


_ACTIVE_BUDGET: ContextVar[EpisodeBudget | None] = ContextVar("episode_budget", default=None)


class EpisodeBudget:
    """Charge observable requests atomically and preserve one monotonic deadline."""

    def __init__(self, *, max_requests: int, max_seconds: float,
                 clock: Callable[[], float] = time.monotonic):
        if type(max_requests) is not int or max_requests < 1 or max_seconds <= 0:
            raise ValueError("episode budgets must be positive")
        self.max_requests = max_requests
        self.max_seconds = max_seconds
        self._clock = clock
        self._started: float | None = None
        self._requests = 0
        self._exhausted_by: str | None = None
        self._lock = Lock()

    def start(self) -> None:
        """Start once at model execution, never reset for nested solves."""
        with self._lock:
            if self._started is None:
                self._started = self._clock()

    def _check_locked(self) -> None:
        if self._started is not None and self._clock() - self._started >= self.max_seconds:
            self._exhausted_by = self._exhausted_by or "wall_time"
        if self._exhausted_by:
            raise EpisodeBudgetExceeded(f"episode_budget_exhausted: {self._exhausted_by}")

    def check(self) -> None:
        """Reject dispatch after exhaustion; permit tools from the last allowed response."""
        with self._lock:
            self._check_locked()

    def reserve_request(self) -> None:
        """Charge before issuing a model request, including failed requests."""
        self.start()
        with self._lock:
            self._check_locked()
            if self._requests >= self.max_requests:
                self._exhausted_by = "llm_requests"
                self._check_locked()
            self._requests += 1

    def expire(self) -> None:
        """Record the deadline when an outstanding asynchronous request is cancelled."""
        with self._lock:
            self._exhausted_by = self._exhausted_by or "wall_time"

    @property
    def remaining_seconds(self) -> float:
        self.start()
        with self._lock:
            return max(0.0, self.max_seconds - (self._clock() - self._started))

    def snapshot(self) -> dict:
        """Return audit metadata without mutating or resetting the budget."""
        with self._lock:
            return {"max_llm_requests": self.max_requests, "max_seconds": self.max_seconds,
                    "llm_requests": self._requests, "exhausted_by": self._exhausted_by,
                    "elapsed_seconds": 0.0 if self._started is None else self._clock() - self._started,
                    "request_scope": "observable API requests; native SDK provider requests are opaque"}


def current_budget(toolkit=None) -> EpisodeBudget | None:
    """Use the parent episode budget for a nested planning toolkit."""
    return _ACTIVE_BUDGET.get() or getattr(toolkit, "episode_budget", None)


@contextmanager
def budget_scope(budget: EpisodeBudget | None):
    """Propagate the same budget through synchronous tools and nested planning."""
    token = _ACTIVE_BUDGET.set(budget)
    try:
        yield
    finally:
        _ACTIVE_BUDGET.reset(token)
