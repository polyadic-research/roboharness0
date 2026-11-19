"""Thread-safe cooperative control for evaluation execution."""

from __future__ import annotations

import threading


class EvaluationCancelled(RuntimeError):
    """Raised when cooperative evaluation work must stop."""


class CancellationToken:
    """A one-way cancellation signal retaining its first reason."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str) -> bool:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be a non-empty string")
        with self._lock:
            if self._reason is not None:
                return False
            self._reason = reason
            self._event.set()
            return True

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise EvaluationCancelled(self.reason or "evaluation cancelled")


__all__ = ["CancellationToken", "EvaluationCancelled"]
