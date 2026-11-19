"""Policy-neutral explicit action-chunk operation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from rpent.episode.contracts import ChunkExecutionResult, StepResult
from rpent.episode.controller import (
    EpisodeController,
    EpisodeExecutionError,
    EpisodeFaultedError,
)


class VLAOperation:
    def __init__(
        self,
        controller: EpisodeController,
        *,
        initial_observation: Mapping[str, Any],
        instruction: str,
        capture_action_history: bool = False,
        transition_guard: Callable[[Any, StepResult], str | None] | None = None,
    ) -> None:
        self._controller = controller
        self._observation = dict(initial_observation)
        self._instruction = instruction
        self._last_observations: tuple[dict[str, Any], ...] = ()
        self._last_actions: tuple[Any, ...] = ()
        self._last_transitions: tuple[StepResult, ...] = ()
        self._transition_guard = transition_guard
        self._transition_guard_faulted = False
        install_guard = getattr(controller, "set_policy_transition_guard", None)
        self.live_transition_guard = transition_guard is not None and callable(install_guard)
        if self.live_transition_guard:
            install_guard(transition_guard)
        self._capture_action_history = capture_action_history
        self._action_history: list[dict[str, Any]] = []
        self._metrics: dict[str, int | bool] = {
            "step_count": 0,
            "action_count": 0,
            "requested_actions": 0,
            "cancelled_actions": 0,
            "paused_actions": 0,
            "resumed_actions": 0,
            "simulator_success": False,
            "terminated": False,
            "truncated": False,
        }

    @property
    def observation(self) -> dict[str, Any]:
        return self._observation

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def metrics(self) -> dict[str, int | bool]:
        return dict(self._metrics)

    @property
    def last_observations(self) -> tuple[dict[str, Any], ...]:
        return self._last_observations

    @property
    def last_actions(self) -> tuple[Any, ...]:
        return self._last_actions

    @property
    def last_transitions(self) -> tuple[StepResult, ...]:
        return self._last_transitions

    @property
    def action_history_count(self) -> int:
        return len(self._action_history)

    def action_history_since(self, cursor: int = 0) -> list[dict[str, Any]]:
        """Copy recorded command chunks without consuming them."""
        return deepcopy(self._action_history[cursor:])

    def discard_pending_policy_actions(self) -> None:
        """Discard a retained chunk without moving or resetting the policy."""
        discard = getattr(self._controller, "discard_pending_policy_actions", None)
        if callable(discard):
            discard()

    def execute(
        self,
        max_actions: int | None = None,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        if self._transition_guard_faulted:
            raise EpisodeFaultedError("cannot act after a transition guard fault")
        effective_instruction = self._instruction if instruction is None else instruction
        try:
            chunk = self._controller.execute_policy_chunk(
                self._observation, effective_instruction, max_actions=max_actions,
            )
        except EpisodeExecutionError as exc:
            self._record_chunk(
                ChunkExecutionResult(
                    exc.confirmed_transitions, exc.requested_actions,
                    actions=exc.confirmed_action_vectors, proposed_actions=exc.proposed_actions,
                ),
                control_source="vla", instruction=effective_instruction, error=exc,
            )
            raise
        return self._record_chunk(chunk, control_source="vla", instruction=effective_instruction)

    def execute_reviewed_policy_action(self, action: Any, instruction: str) -> dict[str, Any]:
        """Resume the exact approved VLA candidate, preserving provenance and guards."""
        return self.execute_actions((action,), _control_source="vla", _instruction=instruction)

    def execute_actions(self, actions: tuple[Any, ...], *,
                        _control_source: str = "direct", _instruction: str | None = None) -> dict[str, Any]:
        """Execute an explicit bounded action sequence through the same controller."""
        if self._transition_guard_faulted:
            raise EpisodeFaultedError("cannot act after a transition guard fault")
        transitions = []
        stop_reason = None
        for action_index, action in enumerate(actions):
            try:
                transition = (
                    self._controller.execute_action(action, _action_index=action_index)
                    if _control_source == "direct" else
                    self._controller.execute_action(
                        action, _action_index=action_index, _control_source=_control_source,
                    )
                )
            except EpisodeExecutionError as exc:
                confirmed = (*transitions, *exc.confirmed_transitions)
                self._record_chunk(
                    ChunkExecutionResult(
                        confirmed, len(actions),
                        actions=tuple(a.copy() for a in actions[:len(confirmed)]),
                        proposed_actions=tuple(a.copy() for a in actions),
                    ), control_source=_control_source, instruction=_instruction, error=exc,
                )
                raise
            transitions.append(transition)
            if self._transition_guard is not None:
                try:
                    stop_reason = self._transition_guard(action.copy(), transition)
                except Exception as exc:
                    self._transition_guard_faulted = True
                    mark_faulted = getattr(self._controller, "mark_faulted", None)
                    if callable(mark_faulted):
                        mark_faulted()
                    error = EpisodeExecutionError(
                        "direct transition guard failed after simulator confirmation",
                        confirmed_transitions=tuple(transitions), requested_actions=len(actions),
                        attempted_action_index=action_index,
                        confirmed_action_vectors=tuple(a.copy() for a in actions[:len(transitions)]),
                        proposed_actions=tuple(a.copy() for a in actions),
                    )
                    self._record_chunk(
                        ChunkExecutionResult(
                            tuple(transitions), len(actions), actions=error.confirmed_action_vectors,
                            proposed_actions=error.proposed_actions,
                        ), control_source=_control_source, instruction=_instruction, error=error,
                    )
                    raise error from exc
            if transition.terminal:
                break
            if stop_reason is not None:
                break
        return self._record_chunk(
            ChunkExecutionResult(
                tuple(transitions),
                len(actions),
                actions=tuple(action.copy() for action in actions[: len(transitions)]),
                proposed_actions=tuple(action.copy() for action in actions),
                stop_reason=stop_reason,
            ), control_source=_control_source, instruction=_instruction,
        )

    def _record_chunk(
        self, chunk: ChunkExecutionResult, *, control_source: str,
        instruction: str | None = None, error: EpisodeExecutionError | None = None,
    ) -> dict[str, Any]:
        self._last_observations = tuple(
            dict(transition.observation) for transition in chunk.transitions
        )
        self._last_transitions = chunk.transitions
        self._last_actions = tuple(action.copy() for action in chunk.actions)
        if self._capture_action_history:
            commands = chunk.proposed_actions or chunk.actions
            self._action_history.append({
                "chunk_index": len(self._action_history),
                "control_source": control_source,
                "instruction": instruction,
                "episode_id": chunk.final.episode_id if chunk.final is not None else None,
                "step_ids": [t.step_id for t in chunk.transitions],
                "actions": [a.tolist() for a in commands],
                "actions_scope": (
                    "validated_chunk" if chunk.proposed_actions else
                    "confirmed_only" if chunk.actions else "unavailable"
                ),
                "requested_actions": chunk.requested_actions,
                "executed_actions": chunk.executed_actions,
                "cancelled_actions": chunk.cancelled_actions,
                "paused_actions": chunk.paused_actions,
                "resumed_actions": chunk.resumed_actions,
                "status": "faulted" if error is not None else "completed",
                "attempted_action_index": error.attempted_action_index if error is not None else None,
                "stop_reason": chunk.stop_reason,
            })
        final = chunk.final
        if final is not None:
            self._observation = dict(final.observation)
            self._metrics.update(
                step_count=final.step_id,
                simulator_success=final.success,
                terminated=final.terminated,
                truncated=final.truncated,
            )
            environment = {
                "episode_id": final.episode_id,
                "step_id": final.step_id,
                "success": final.success,
                "terminated": final.terminated,
                "truncated": final.truncated,
            }
        else:
            environment = None
        self._metrics["action_count"] = (
            int(self._metrics["action_count"]) + chunk.executed_actions
        )
        self._metrics["requested_actions"] = (
            int(self._metrics["requested_actions"]) + chunk.requested_actions
        )
        self._metrics["cancelled_actions"] = (
            int(self._metrics["cancelled_actions"]) + chunk.cancelled_actions
        )
        for key in ("paused_actions", "resumed_actions"):
            self._metrics[key] = int(self._metrics[key]) + getattr(chunk, key)
        result: dict[str, Any] = {
            "requested_actions": chunk.requested_actions,
            "executed_actions": chunk.executed_actions,
            "cancelled_actions": chunk.cancelled_actions,
            "paused_actions": chunk.paused_actions,
            "resumed_actions": chunk.resumed_actions,
        }
        if environment is not None:
            result["environment"] = environment
        if chunk.stop_reason is not None:
            result["stop_reason"] = chunk.stop_reason
        return result


__all__ = ["VLAOperation"]
