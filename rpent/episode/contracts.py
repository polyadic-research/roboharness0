"""Simulator, policy, and transition contracts for one episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

POLICY_REVIEW_PAUSE_REASON = "preclose_review_required"


@dataclass(frozen=True, slots=True)
class StepResult:
    """One authoritative transition returned by a simulator."""

    episode_id: str
    step_id: int
    observation: dict[str, Any]
    reward: float
    success: bool
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")

    @property
    def terminal(self) -> bool:
        """Return whether no more actions may be applied to this episode."""
        return self.success or self.terminated or self.truncated


class SimulatorSession(Protocol):
    """Minimal interface implemented by an RPent simulator backend."""

    @property
    def episode_id(self) -> str:
        """Return the identifier of the active episode."""
        raise NotImplementedError

    def reset(self, task: str, seed: int) -> dict[str, Any]:
        """Start one episode and return its initial observation."""
        raise NotImplementedError

    def observe(self) -> dict[str, Any]:
        """Return the current observation without advancing the simulator."""
        raise NotImplementedError

    def step(self, action: np.ndarray) -> StepResult:
        """Apply one action and return its authoritative transition."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources owned by this simulator session."""
        raise NotImplementedError


class PolicySession(Protocol):
    """Minimal interface implemented by an RPent policy backend."""

    def reset(self, episode_id: str) -> None:
        """Clear policy state for a newly started episode."""
        raise NotImplementedError

    def predict(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> np.ndarray:
        """Return an explicit action chunk for the current observation."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources owned by this policy session."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ChunkExecutionResult:
    """Executed transitions and the unexecuted suffix of an action chunk."""

    transitions: tuple[StepResult, ...]
    requested_actions: int
    actions: tuple[Any, ...] = ()
    proposed_actions: tuple[Any, ...] = ()
    stop_reason: str | None = None
    paused_actions: int = 0
    resumed_actions: int = 0

    def __post_init__(self) -> None:
        if self.requested_actions < len(self.transitions):
            raise ValueError(
                "requested_actions must be at least the number of transitions"
            )
        if self.actions and len(self.actions) != len(self.transitions):
            raise ValueError("actions must match the confirmed transitions")
        if self.proposed_actions and len(self.proposed_actions) != self.requested_actions:
            raise ValueError("proposed_actions must match requested_actions")
        if not 0 <= self.paused_actions <= self.requested_actions - len(self.transitions):
            raise ValueError("paused_actions must belong to the unexecuted suffix")
        if not 0 <= self.resumed_actions <= len(self.transitions):
            raise ValueError("resumed_actions must belong to confirmed transitions")

    @property
    def executed_actions(self) -> int:
        """Return the number of actions actually sent to the simulator."""
        return len(self.transitions)

    @property
    def cancelled_actions(self) -> int:
        """Return the number of actions omitted from the requested chunk."""
        return self.requested_actions - self.executed_actions - self.paused_actions

    @property
    def final(self) -> StepResult | None:
        """Return the final transition, if the chunk executed any actions."""
        return self.transitions[-1] if self.transitions else None
