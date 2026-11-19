"""Scoped, fail-closed recovery budget ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal
from uuid import uuid4

RecoveryScope = Literal[
    "retry_step",
    "replan_outcome",
    "replan_stage",
    "repartition_remaining_stages",
    "terminal_failure",
]


@dataclass(frozen=True)
class RecoveryPolicy:
    local_replans_per_outcome: int = 1
    stage_replans_per_stage: int = 1
    remaining_stage_repartitions_per_run: int = 1
    max_total_physical_actions_per_run: int = 100

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in asdict(self).values()
        ):
            raise ValueError("recovery policy budgets must be non-negative integers")


@dataclass(frozen=True)
class RecoveryDecision:
    scope: RecoveryScope
    outcome_id: str
    stage_id: str
    reason: str
    decision_id: str = field(repr=False, compare=True)


class RecoveryTracker:
    """Issue and consume immutable recovery decisions against owned budgets."""

    def __init__(self, policy: RecoveryPolicy) -> None:
        self.policy = policy
        self._outcome_replans: dict[str, int] = {}
        self._stage_replans: dict[str, int] = {}
        self._repartitions = 0
        self._physical_actions = 0
        self._issued: dict[str, RecoveryDecision] = {}
        self._consumed: set[str] = set()
        self._cancelled: set[str] = set()
        self._pending_recovery_action: RecoveryDecision | None = None

    @property
    def pending_recovery_action(self) -> RecoveryDecision | None:
        return self._pending_recovery_action

    def record_physical_action(self) -> None:
        if self._physical_actions >= self.policy.max_total_physical_actions_per_run:
            raise ValueError("physical action budget exhausted")
        self._physical_actions += 1
        self._pending_recovery_action = None

    def decide(
        self,
        outcome_id: str,
        stage_id: str,
        safe_step_retry: bool,
        step_attempts_remaining: int,
    ) -> RecoveryDecision:
        if not outcome_id.strip() or not stage_id.strip():
            raise ValueError("recovery outcome_id and stage_id must be non-empty")
        if step_attempts_remaining < 0:
            raise ValueError("step_attempts_remaining must be >= 0")
        if self._physical_actions >= self.policy.max_total_physical_actions_per_run:
            return self._issue(
                "terminal_failure", outcome_id, stage_id, "run action budget exhausted"
            )
        if safe_step_retry and step_attempts_remaining > 0:
            return self._issue(
                "retry_step", outcome_id, stage_id, "safe step retry remains"
            )
        if (
            self._outcome_replans.get(outcome_id, 0)
            < self.policy.local_replans_per_outcome
        ):
            return self._issue(
                "replan_outcome",
                outcome_id,
                stage_id,
                "outcome recovery budget remains",
            )
        if self._stage_replans.get(stage_id, 0) < self.policy.stage_replans_per_stage:
            return self._issue(
                "replan_stage", outcome_id, stage_id, "stage recovery budget remains"
            )
        if self._repartitions < self.policy.remaining_stage_repartitions_per_run:
            return self._issue(
                "repartition_remaining_stages",
                outcome_id,
                stage_id,
                "run repartition budget remains",
            )
        return self._issue(
            "terminal_failure", outcome_id, stage_id, "all recovery budgets exhausted"
        )

    def consume(
        self,
        decision: RecoveryDecision,
        *,
        outcome_id: str,
        stage_id: str,
    ) -> None:
        if not isinstance(decision, RecoveryDecision):
            raise TypeError(
                "consume requires a RecoveryDecision issued by this tracker"
            )
        issued = self._issued.get(decision.decision_id)
        if issued is None or issued != decision:
            raise ValueError("recovery decision was not issued by this tracker")
        if decision.decision_id in self._cancelled:
            raise ValueError("recovery decision was cancelled")
        if decision.decision_id in self._consumed:
            raise ValueError("recovery decision was already consumed")
        if decision.outcome_id != outcome_id or decision.stage_id != stage_id:
            raise ValueError("recovery decision does not match outcome/stage context")
        if decision.scope == "terminal_failure":
            raise ValueError("terminal recovery decisions cannot be consumed")
        if not self._scope_is_available(decision):
            raise ValueError("recovery budget exhausted before decision consumption")

        if decision.scope == "replan_outcome":
            self._outcome_replans[outcome_id] = (
                self._outcome_replans.get(outcome_id, 0) + 1
            )
        elif decision.scope == "replan_stage":
            self._stage_replans[stage_id] = self._stage_replans.get(stage_id, 0) + 1
        elif decision.scope == "repartition_remaining_stages":
            self._repartitions += 1
        self._consumed.add(decision.decision_id)
        self._pending_recovery_action = decision

    def cancel(
        self,
        decision: RecoveryDecision,
        *,
        outcome_id: str,
        stage_id: str,
    ) -> None:
        """Cancel an unconsumed decision; repeating the same cancellation is a no-op."""
        if not isinstance(decision, RecoveryDecision):
            raise TypeError("cancel requires a RecoveryDecision issued by this tracker")
        issued = self._issued.get(decision.decision_id)
        if issued is None or issued != decision:
            raise ValueError("recovery decision was not issued by this tracker")
        if decision.outcome_id != outcome_id or decision.stage_id != stage_id:
            raise ValueError("recovery decision does not match outcome/stage context")
        if decision.decision_id in self._consumed:
            raise ValueError("recovery decision was already consumed")
        if decision.decision_id in self._cancelled:
            return
        self._cancelled.add(decision.decision_id)

    def snapshot(self) -> dict[str, object]:
        return {
            "policy": asdict(self.policy),
            "outcome_replans_consumed": dict(self._outcome_replans),
            "stage_replans_consumed": dict(self._stage_replans),
            "remaining_stage_repartitions_consumed": self._repartitions,
            "physical_actions_consumed": self._physical_actions,
            "pending_recovery_action": (
                {
                    "scope": self._pending_recovery_action.scope,
                    "outcome_id": self._pending_recovery_action.outcome_id,
                    "stage_id": self._pending_recovery_action.stage_id,
                }
                if self._pending_recovery_action is not None
                else None
            ),
            "outstanding_decisions": [
                {
                    "decision_id": decision.decision_id,
                    "scope": decision.scope,
                    "outcome_id": decision.outcome_id,
                    "stage_id": decision.stage_id,
                    "reason": decision.reason,
                }
                for decision_id, decision in self._issued.items()
                if decision_id not in self._consumed
                and decision_id not in self._cancelled
            ],
        }

    def remaining_for(self, outcome_id: str, stage_id: str) -> dict[str, int]:
        return {
            "local_remaining": max(
                0,
                self.policy.local_replans_per_outcome
                - self._outcome_replans.get(outcome_id, 0),
            ),
            "stage_remaining": max(
                0,
                self.policy.stage_replans_per_stage
                - self._stage_replans.get(stage_id, 0),
            ),
            "repartition_remaining": max(
                0,
                self.policy.remaining_stage_repartitions_per_run - self._repartitions,
            ),
        }

    def _issue(
        self,
        scope: RecoveryScope,
        outcome_id: str,
        stage_id: str,
        reason: str,
    ) -> RecoveryDecision:
        decision = RecoveryDecision(scope, outcome_id, stage_id, reason, uuid4().hex)
        self._issued[decision.decision_id] = decision
        return decision

    def _scope_is_available(self, decision: RecoveryDecision) -> bool:
        if self._physical_actions >= self.policy.max_total_physical_actions_per_run:
            return False
        if decision.scope == "retry_step":
            return True
        if decision.scope == "replan_outcome":
            return (
                self._outcome_replans.get(decision.outcome_id, 0)
                < self.policy.local_replans_per_outcome
            )
        if decision.scope == "replan_stage":
            return (
                self._stage_replans.get(decision.stage_id, 0)
                < self.policy.stage_replans_per_stage
            )
        if decision.scope == "repartition_remaining_stages":
            return self._repartitions < self.policy.remaining_stage_repartitions_per_run
        return False


__all__ = ["RecoveryDecision", "RecoveryPolicy", "RecoveryScope", "RecoveryTracker"]
