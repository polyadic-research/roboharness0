"""Execute policy action chunks with per-transition terminal checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from rpent.episode.binding import PolicyBinding
from rpent.episode.contracts import (
    POLICY_REVIEW_PAUSE_REASON,
    ChunkExecutionResult,
    PolicySession,
    SimulatorSession,
    StepResult,
)
from rpent.episode.recording import NULL_TRANSITION_OBSERVER, TransitionObserver
from rpent.recording.narrated.instrumentation import observer_phase


class EpisodeTerminalError(RuntimeError):
    """Raised when an action is requested after a terminal transition."""


class EpisodeFaultedError(RuntimeError):
    """Raised when an action is requested after an execution fault."""


class EpisodeExecutionError(RuntimeError):
    """Preserve confirmed progress when execution fails closed."""

    def __init__(
        self,
        message: str,
        *,
        confirmed_transitions: tuple[StepResult, ...],
        requested_actions: int,
        attempted_action_index: int | None,
        observer_error: Exception | None = None,
        confirmed_action_vectors: tuple[Any, ...] = (),
        proposed_actions: tuple[Any, ...] = (),
    ) -> None:
        super().__init__(message)
        self.confirmed_transitions = confirmed_transitions
        self.requested_actions = requested_actions
        self.attempted_action_index = attempted_action_index
        self.observer_error = observer_error
        self.confirmed_action_vectors = confirmed_action_vectors
        self.proposed_actions = proposed_actions

    @property
    def confirmed_actions(self) -> int:
        """Return the number of transitions confirmed before the fault."""

        return len(self.confirmed_transitions)


class TransitionObserverError(EpisodeExecutionError):
    """A transition was confirmed physically but could not be observed safely."""


@dataclass
class _PausedPolicyChunk:
    actions: tuple[Any, ...]
    observation: Mapping[str, Any]
    instruction: str


class EpisodeController:
    """Apply policy chunks one action at a time until the episode terminates."""

    def __init__(
        self,
        simulator: SimulatorSession,
        policy: PolicySession,
        binding: PolicyBinding,
        transition_observer: TransitionObserver | None = None,
    ) -> None:
        self._simulator = simulator
        self._policy = policy
        self._binding = binding
        self._transition_observer: TransitionObserver = NULL_TRANSITION_OBSERVER
        self._observer_installed = False
        self._terminal = False
        self._faulted = False
        self._action_attempted = False
        self._paused_policy_chunk: _PausedPolicyChunk | None = None
        self._policy_transition_guard: Callable[[Any, StepResult], str | None] | None = None
        self._policy_action_guard: Callable[[Mapping[str, Any], Any], str | None] | None = None
        if transition_observer is not None:
            self.set_transition_observer(transition_observer)

    @property
    def transition_observer(self) -> TransitionObserver:
        return self._transition_observer

    def mark_faulted(self) -> None:
        """Latch a tool-owned execution monitor failure at the same fault boundary."""
        self._faulted = True
        self.discard_pending_policy_actions()

    def discard_pending_policy_actions(self) -> None:
        """Invalidate a retained suffix after rejection or a changed operation."""
        self._paused_policy_chunk = None

    def _retain_policy_actions(self, actions, observation, instruction) -> None:
        self._paused_policy_chunk = _PausedPolicyChunk(
            tuple(action.copy() for action in actions),
            deepcopy(self._binding.bind_observation(observation)), instruction,
        )

    def set_policy_transition_guard(
        self, guard: Callable[[Any, StepResult], str | None],
    ) -> None:
        """Install a tool-owned interruption check without replacing recording."""
        if self._action_attempted or self._policy_transition_guard is not None:
            raise RuntimeError("policy transition guard must be installed before execution")
        if not callable(guard):
            raise TypeError("policy transition guard must be callable")
        self._policy_transition_guard = guard

    def set_policy_action_guard(
        self, guard: Callable[[Mapping[str, Any], Any], str | None],
    ) -> None:
        """Install an environment check before sending each policy action."""
        if self._action_attempted or self._policy_action_guard is not None:
            raise RuntimeError("policy action guard must be installed once before execution")
        self._policy_action_guard = guard

    def set_transition_observer(self, observer: TransitionObserver) -> None:
        """Install an observer before the first attempted action."""
        if self._action_attempted:
            raise RuntimeError(
                "cannot replace transition observer after action attempt"
            )
        if self._observer_installed:
            raise RuntimeError("a transition observer is already installed")
        reset = getattr(observer, "reset", None)
        if callable(reset):
            reset()
        self._transition_observer = observer
        self._observer_installed = True

    def execute_action(
        self, action: Any, *, _action_index: int = 0, _control_source: str = "direct"
    ) -> StepResult:
        """Execute one action through the authoritative transition boundary."""
        if self._terminal:
            raise EpisodeTerminalError("cannot act after a terminal transition")
        if self._faulted:
            raise EpisodeFaultedError("cannot act after an execution fault")

        if _control_source != "vla":
            self.discard_pending_policy_actions()
        sent_action = action.copy()
        self._action_attempted = True
        try:
            with observer_phase(
                self._transition_observer, "robot", control_source=_control_source
            ):
                transition = self._simulator.step(action)
        except Exception as exc:
            self._faulted = True
            observer_error = None
            try:
                self._transition_observer.failed(sent_action.copy(), _action_index, exc)
            except Exception as observer_exc:
                observer_error = observer_exc
            raise EpisodeExecutionError(
                "simulator failed while executing action",
                confirmed_transitions=(),
                requested_actions=1,
                attempted_action_index=_action_index,
                observer_error=observer_error,
                proposed_actions=(sent_action.copy(),),
            ) from exc
        try:
            self._transition_observer.confirmed(sent_action.copy(), transition)
        except Exception as exc:
            self._faulted = True
            self._terminal = transition.terminal
            raise TransitionObserverError(
                "transition observer failed after simulator confirmation",
                confirmed_transitions=(transition,),
                requested_actions=1,
                attempted_action_index=_action_index,
                observer_error=exc,
                confirmed_action_vectors=(sent_action.copy(),),
                proposed_actions=(sent_action.copy(),),
            ) from exc
        self._terminal = transition.terminal
        return transition

    def execute_policy_chunk(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> ChunkExecutionResult:
        """Resume an unchanged reviewed chunk, or predict and execute a new one."""

        if self._terminal:
            raise EpisodeTerminalError("cannot act after a terminal transition")
        if self._faulted:
            raise EpisodeFaultedError("cannot act after an execution fault")

        normalized_max_actions = _validate_max_actions(max_actions)

        policy_observation = self._binding.bind_observation(observation)
        pending = self._paused_policy_chunk
        self.discard_pending_policy_actions()
        resume = (
            pending is not None and pending.instruction == instruction
            and pending.observation.keys() == policy_observation.keys()
            and all(np.array_equal(pending.observation[k], policy_observation[k])
                    for k in policy_observation)
        )
        reserved_tail = ()
        if resume:
            limit = len(pending.actions) if normalized_max_actions is None else normalized_max_actions
            actions = tuple(action.copy() for action in pending.actions[:limit])
            reserved_tail = pending.actions[limit:]
        else:
            with observer_phase(self._transition_observer, "vla", instruction=instruction):
                predicted_actions = self._policy.predict(
                    policy_observation,
                    instruction,
                    max_actions=normalized_max_actions,
                )
            actions = self._binding.validate_action_chunk(predicted_actions)
        requested_actions = len(actions)
        if (
            normalized_max_actions is not None
            and requested_actions > normalized_max_actions
        ):
            self._faulted = True
            raise EpisodeExecutionError(
                f"policy returned {requested_actions} actions, "
                f"exceeding max_actions={normalized_max_actions}",
                confirmed_transitions=(),
                requested_actions=requested_actions,
                attempted_action_index=None,
            )

        transitions: list[StepResult] = []
        stop_reason = None
        current_observation = observation
        for action_index, action in enumerate(actions):
            if self._policy_action_guard is not None:
                try:
                    stop_reason = self._policy_action_guard(current_observation, action.copy())
                except Exception as exc:
                    self._faulted = True
                    raise EpisodeExecutionError(
                        "policy action guard failed before simulator execution",
                        confirmed_transitions=tuple(transitions),
                        requested_actions=requested_actions,
                        attempted_action_index=None,
                        confirmed_action_vectors=tuple(a.copy() for a in actions[:len(transitions)]),
                        proposed_actions=tuple(a.copy() for a in actions),
                    ) from exc
                if stop_reason is not None:
                    break
            try:
                transition = self.execute_action(
                    action, _action_index=action_index, _control_source="vla"
                )
            except TransitionObserverError as exc:
                raise TransitionObserverError(
                    "transition observer failed after simulator confirmation",
                    confirmed_transitions=(
                        *transitions,
                        *exc.confirmed_transitions,
                    ),
                    requested_actions=requested_actions,
                    attempted_action_index=action_index,
                    observer_error=exc.observer_error,
                    confirmed_action_vectors=tuple(
                        a.copy() for a in actions[:len(transitions) + len(exc.confirmed_transitions)]
                    ),
                    proposed_actions=tuple(a.copy() for a in actions),
                ) from exc.__cause__
            except EpisodeExecutionError as exc:
                raise EpisodeExecutionError(
                    "simulator failed while executing "
                    f"action index {action_index} of {requested_actions}",
                    confirmed_transitions=tuple(transitions),
                    requested_actions=requested_actions,
                    attempted_action_index=action_index,
                    observer_error=exc.observer_error,
                    confirmed_action_vectors=tuple(a.copy() for a in actions[:len(transitions)]),
                    proposed_actions=tuple(a.copy() for a in actions),
                ) from exc.__cause__
            transitions.append(transition)
            current_observation = transition.observation
            if self._policy_transition_guard is not None:
                try:
                    stop_reason = self._policy_transition_guard(action.copy(), transition)
                except Exception as exc:
                    self._faulted = True
                    raise EpisodeExecutionError(
                        "policy transition guard failed after simulator confirmation",
                        confirmed_transitions=tuple(transitions),
                        requested_actions=requested_actions,
                        attempted_action_index=action_index,
                        confirmed_action_vectors=tuple(a.copy() for a in actions[:len(transitions)]),
                        proposed_actions=tuple(a.copy() for a in actions),
                    ) from exc
            if transition.terminal:
                break
            if stop_reason is not None:
                break

        paused_actions = 0
        if stop_reason == POLICY_REVIEW_PAUSE_REASON:
            paused_actions = requested_actions - len(transitions)
            self._retain_policy_actions(
                (*actions[len(transitions):], *reserved_tail), current_observation, instruction,
            )
        elif reserved_tail and not stop_reason and not self._terminal:
            self._retain_policy_actions(reserved_tail, current_observation, instruction)

        return ChunkExecutionResult(
            tuple(transitions),
            requested_actions,
            actions=tuple(action.copy() for action in actions[: len(transitions)]),
            proposed_actions=tuple(action.copy() for action in actions),
            stop_reason=stop_reason,
            paused_actions=paused_actions,
            resumed_actions=len(transitions) if resume else 0,
        )


def _validate_max_actions(max_actions: int | None) -> int | None:
    if max_actions is None:
        return None
    if (
        isinstance(max_actions, bool)
        or not isinstance(max_actions, Integral)
        or max_actions <= 0
    ):
        raise ValueError("max_actions must be a positive integer")
    return int(max_actions)
