"""Transition recording adapters kept outside simulator and policy sessions."""

from __future__ import annotations

import math
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np

from rpent.episode.contracts import StepResult
from rpent.recording.narrated.instrumentation import phase_span


class TransitionObserver(Protocol):
    """Observe only actions accepted or rejected by the simulator boundary."""

    def confirmed(self, action: np.ndarray, result: StepResult) -> None: ...

    def failed(
        self, action: np.ndarray, action_index: int, error: Exception
    ) -> None: ...


class NullTransitionObserver:
    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        del action, result

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        del action, action_index, error


NULL_TRANSITION_OBSERVER = NullTransitionObserver()


class CompositeTransitionObserver:
    """Fan out one authoritative transition to independent observers."""

    def __init__(self, *observers: TransitionObserver) -> None:
        if not observers:
            raise ValueError("at least one transition observer is required")
        self._observers = tuple(observers)

    def reset(self) -> None:
        for observer in self._observers:
            reset = getattr(observer, "reset", None)
            if callable(reset):
                reset()

    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        self._fan_out("confirmed", action, result)

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        self._fan_out("failed", action, action_index, error)

    def _fan_out(self, method: str, *args: Any) -> None:
        first_error = None
        for observer in self._observers:
            try:
                getattr(observer, method)(*args)
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    @contextmanager
    def operation(
        self, operation_id: str, execution_id: str
    ) -> Iterator[TransitionOperation]:
        current = TransitionOperation(operation_id, execution_id)
        with ExitStack() as stack:
            for observer in self._observers:
                operation = getattr(observer, "operation", None)
                if callable(operation):
                    entered = stack.enter_context(operation(operation_id, execution_id))
                    if not current.step_ids:
                        current = entered
            yield current

    def submit_operation(self, operation_id: str, step_ids: tuple[int, ...]) -> None:
        for observer in self._observers:
            submit = getattr(observer, "submit_operation", None)
            if callable(submit):
                submit(operation_id, step_ids)

    @contextmanager
    def phase(self, phase: str, **metadata: Any):
        with ExitStack() as stack:
            for observer in self._observers:
                callback = getattr(observer, "phase", None)
                if callable(callback):
                    stack.enter_context(callback(phase, **metadata))
            yield


@dataclass(slots=True)
class TransitionOperation:
    operation_id: str
    execution_id: str
    step_ids: list[int] = field(default_factory=list)


class RecorderTransitionObserver:
    """Adapt confirmed controller transitions to a CaseRecorder."""

    def __init__(
        self,
        recorder: Any,
        *,
        monotonic=time.monotonic,
        reset_at: float | None = None,
    ) -> None:
        self._recorder = recorder
        self._monotonic = monotonic
        self._explicit_reset_at = reset_at is not None
        self._reset_at = (
            self._validate_reset_at(reset_at) if reset_at is not None else monotonic()
        )
        self._operation: ContextVar[TransitionOperation | None] = ContextVar(
            f"transition_operation_{id(self)}", default=None
        )

    @staticmethod
    def _validate_reset_at(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("reset_at must be a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("reset_at must be a finite number")
        return normalized

    def reset(self, reset_at: float | None = None) -> None:
        if reset_at is not None:
            self._reset_at = self._validate_reset_at(reset_at)
            self._explicit_reset_at = True
        elif not self._explicit_reset_at:
            self._reset_at = self._monotonic()

    @contextmanager
    def operation(
        self, operation_id: str, execution_id: str
    ) -> Iterator[TransitionOperation]:
        current = TransitionOperation(operation_id, execution_id)
        token = self._operation.set(current)
        try:
            yield current
        finally:
            self._operation.reset(token)

    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        operation = self._operation.get()
        if operation is not None:
            operation.step_ids.append(result.step_id)
        self._recorder.record_transition(
            action=action,
            result=result,
            timestamp_s=self._monotonic() - self._reset_at,
            operation_id=operation.operation_id if operation else None,
            execution_id=operation.execution_id if operation else None,
        )

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        operation = self._operation.get()
        self._recorder.record_event(
            "transition_failed",
            action=action,
            action_index=action_index,
            error_type=type(error).__name__,
            error_message=str(error),
            operation_id=operation.operation_id if operation else None,
            execution_id=operation.execution_id if operation else None,
        )

    def submit_operation(self, operation_id: str, step_ids: tuple[int, ...]) -> None:
        """Submit one completed operation without coupling Toolkit to recording."""
        submit = getattr(self._recorder, "submit_operation_clips", None)
        if callable(submit):
            submit(operation_id, step_ids)

    def phase(self, phase: str, **metadata: Any):
        operation = self._operation.get()
        return phase_span(
            self._recorder,
            phase,
            clock=self._monotonic,
            reset_at=self._reset_at,
            operation_id=operation.operation_id if operation else None,
            execution_id=operation.execution_id if operation else None,
            **metadata,
        )


@contextmanager
def operation_scope(
    observer: TransitionObserver,
    operation_id: str,
    execution_id: str,
) -> Iterator[TransitionOperation]:
    operation = getattr(observer, "operation", None)
    if callable(operation):
        with operation(operation_id, execution_id) as current:
            yield current
        return
    yield TransitionOperation(operation_id, execution_id)


__all__ = [
    "CompositeTransitionObserver",
    "NULL_TRANSITION_OBSERVER",
    "NullTransitionObserver",
    "RecorderTransitionObserver",
    "TransitionObserver",
    "TransitionOperation",
    "operation_scope",
]
