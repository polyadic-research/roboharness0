"""Immutable outcome definitions and evidence-backed execution state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

OutcomeLifecycle = Literal["event", "persistent", "terminal"]
OutcomeState = Literal[
    "pending",
    "ready",
    "in_progress",
    "progressing",
    "satisfied",
    "failed",
    "blocked",
    "invalidated",
    "recovery_ready",
]

_OUTCOME_LIFECYCLES = {"event", "persistent", "terminal"}
_REORDER_REASONS = {
    "temporarily_unobservable",
    "risk_reduction",
    "environment_change",
}


def _parse_capture_time(value: str) -> datetime:
    captured = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("evidence capture time must be timezone-aware")
    return captured
_TRANSITIONS: dict[OutcomeState, set[OutcomeState]] = {
    "pending": {"ready"},
    "ready": {"in_progress", "blocked"},
    "in_progress": {"progressing", "satisfied", "failed", "blocked"},
    "progressing": {"progressing", "satisfied", "failed", "blocked"},
    "satisfied": {"invalidated"},
    "failed": {"recovery_ready"},
    "blocked": {"ready"},
    "invalidated": {"recovery_ready", "satisfied"},
    "recovery_ready": {"in_progress", "blocked"},
}


@dataclass(frozen=True)
class OutcomeEvidence:
    execution_id: str | None
    source: str
    summary: str
    confidence: float | None = None
    observation_id: str | None = None
    captured_at: str | None = None
    verifier: str | None = None
    verifier_version: str | None = None
    predicate_result: str | None = None
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("evidence source must be non-empty")
        if not self.summary.strip():
            raise ValueError("evidence summary must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be between 0 and 1")


@dataclass(frozen=True)
class AtomicOutcome:
    id: str
    instruction_index: int
    description: str
    success_predicate: str
    depends_on: tuple[str, ...]
    lifecycle: OutcomeLifecycle


@dataclass
class _OutcomeRuntime:
    status: OutcomeState = "pending"
    evidence: list[OutcomeEvidence] = field(default_factory=list)
    invalidations: list[str] = field(default_factory=list)
    final_verified: bool = False


@dataclass(frozen=True)
class ReorderRequest:
    selected_outcome_id: str
    deferred_outcome_ids: tuple[str, ...]
    reason: Literal["temporarily_unobservable", "risk_reduction", "environment_change"]
    evidence: tuple[str, ...]


class OutcomeGraph:
    """Track a fixed outcome contract without allowing definition mutation."""

    def __init__(
        self,
        outcomes: list[AtomicOutcome],
        *,
        emit: Callable[..., None] | None = None,
    ) -> None:
        outcome_ids = [item.id for item in outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("duplicate outcome id")
        ordered = tuple(sorted(outcomes, key=lambda value: value.instruction_index))
        self._definitions = ordered
        self._outcomes = {item.id: item for item in ordered}
        self._runtime = {item.id: _OutcomeRuntime() for item in ordered}
        self._order = tuple(item.id for item in ordered)
        self._active_outcome_id: str | None = None
        self._reorders: list[ReorderRequest] = []
        self._illegal_reorder_count = 0
        self._emit = emit or (lambda event, **payload: None)
        self._validate()
        self._refresh_ready()

    @classmethod
    def from_payload(
        cls,
        payload: list[dict[str, Any]],
        *,
        emit: Callable[..., None] | None = None,
    ) -> "OutcomeGraph":
        if not isinstance(payload, list) or not payload:
            raise ValueError("outcomes must be a non-empty array")
        outcomes = [
            AtomicOutcome(
                id=str(item.get("id", "")).strip(),
                instruction_index=int(item.get("instruction_index", -1)),
                description=str(item.get("description", "")).strip(),
                success_predicate=str(item.get("success_predicate", "")).strip(),
                depends_on=tuple(
                    str(value).strip() for value in item.get("depends_on", ())
                ),
                lifecycle=item.get("lifecycle", ""),
            )
            for item in payload
        ]
        return cls(outcomes, emit=emit)

    @property
    def definitions(self) -> tuple[AtomicOutcome, ...]:
        return self._definitions

    @property
    def outcome_ids(self) -> tuple[str, ...]:
        return self._order

    @property
    def ready_ids(self) -> tuple[str, ...]:
        return tuple(
            outcome_id
            for outcome_id in self._order
            if self._runtime[outcome_id].status in {"ready", "recovery_ready"}
            and all(
                self._runtime[dependency_id].status == "satisfied"
                for dependency_id in self._outcomes[outcome_id].depends_on
            )
        )

    @property
    def active_outcome_id(self) -> str | None:
        return self._active_outcome_id

    def recovery_ids_for(self, stage_outcome_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Return unresolved stage outcomes and broken prerequisites in order."""
        self._require_known(stage_outcome_ids, context="recovery")
        selected: set[str] = set()

        def include_broken_prerequisites(outcome_id: str) -> None:
            for dependency_id in self._outcomes[outcome_id].depends_on:
                dependency_state = self._runtime[dependency_id].status
                if dependency_state in {"failed", "invalidated", "recovery_ready"}:
                    selected.add(dependency_id)
                if dependency_state != "satisfied":
                    include_broken_prerequisites(dependency_id)

        for outcome_id in stage_outcome_ids:
            if self._runtime[outcome_id].status != "satisfied":
                selected.add(outcome_id)
            include_broken_prerequisites(outcome_id)
        return tuple(outcome_id for outcome_id in self._order if outcome_id in selected)

    def snapshot(self) -> dict[str, Any]:
        counts = dict.fromkeys(_TRANSITIONS, 0)
        outcomes: list[dict[str, Any]] = []
        for outcome_id in self._order:
            definition = self._outcomes[outcome_id]
            runtime = self._runtime[outcome_id]
            counts[runtime.status] += 1
            outcomes.append(
                {
                    **asdict(definition),
                    "status": runtime.status,
                    "evidence": [asdict(item) for item in runtime.evidence],
                    "invalidations": list(runtime.invalidations),
                    "final_verified": runtime.final_verified,
                }
            )
        return {
            "outcomes": outcomes,
            "ready_ids": list(self.ready_ids),
            "active_outcome_id": self._active_outcome_id,
            "reorders": [asdict(item) for item in self._reorders],
            "illegal_reorder_count": self._illegal_reorder_count,
            "counts": counts,
        }

    def select(self, outcome_id: str, reorder: ReorderRequest | None = None) -> None:
        self._require_known((outcome_id,))
        if self._active_outcome_id is not None:
            raise ValueError(f"outcome already in progress: {self._active_outcome_id}")
        ready_ids = self.ready_ids
        if outcome_id not in ready_ids:
            raise ValueError(f"outcome is not ready: {outcome_id}")
        earliest = ready_ids[0]
        if outcome_id != earliest:
            self._validate_reorder(outcome_id, ready_ids, reorder)
            assert reorder is not None
            self._reorders.append(reorder)
        self._transition(outcome_id, "in_progress", reason="outcome selected")
        self._active_outcome_id = outcome_id

    def progress(self, outcome_id: str, evidence: OutcomeEvidence) -> None:
        evidence = self._require_evidence(evidence)
        runtime = self._runtime_for_active(outcome_id)
        runtime.evidence.append(evidence)
        self._transition(
            outcome_id, "progressing", reason=evidence.summary, evidence=evidence
        )

    def satisfy(self, outcome_id: str, evidence: OutcomeEvidence) -> None:
        evidence = self._require_success_evidence(evidence)
        runtime = self._runtime_for_active(outcome_id)
        runtime.evidence.append(evidence)
        self._transition(
            outcome_id, "satisfied", reason=evidence.summary, evidence=evidence
        )
        runtime.final_verified = self._outcomes[outcome_id].lifecycle == "event"
        self._active_outcome_id = None
        self._refresh_ready()

    def fail(self, outcome_id: str, evidence: OutcomeEvidence) -> None:
        evidence = self._require_evidence(evidence)
        runtime = self._runtime_for_active(outcome_id)
        runtime.evidence.append(evidence)
        self._transition(
            outcome_id, "failed", reason=evidence.summary, evidence=evidence
        )
        runtime.final_verified = False
        self._active_outcome_id = None

    def invalidate(self, outcome_id: str, *, reason: str) -> None:
        self._require_known((outcome_id,))
        if not reason.strip():
            raise ValueError("invalidation reason must be non-empty")
        outcome = self._outcomes[outcome_id]
        runtime = self._runtime[outcome_id]
        if outcome.lifecycle == "event":
            raise ValueError("event outcomes cannot be invalidated")
        if runtime.status != "satisfied":
            raise ValueError(
                f"only satisfied outcomes can be invalidated: {outcome_id}"
            )
        runtime.invalidations.append(reason)
        self._transition(outcome_id, "invalidated", reason=reason)
        runtime.final_verified = False
        self._block_direct_dependents(outcome_id, reason=reason)

    def invalidate_affected(
        self, outcome_ids: tuple[str, ...], *, reason: str
    ) -> tuple[str, ...]:
        """Invalidate satisfied durable outcomes and return them in contract order."""
        self._require_known(outcome_ids, context="affected")
        selected = set(outcome_ids)
        invalidated: list[str] = []
        for outcome_id in self._order:
            outcome = self._outcomes[outcome_id]
            if (
                outcome_id in selected
                and self._runtime[outcome_id].status == "satisfied"
                and outcome.lifecycle != "event"
            ):
                self.invalidate(outcome_id, reason=reason)
                invalidated.append(outcome_id)
        self._refresh_ready()
        return tuple(invalidated)

    def mark_final_verified(self, outcome_id: str, evidence: OutcomeEvidence) -> None:
        evidence = self._require_evidence(evidence)
        self._require_known((outcome_id,))
        outcome = self._outcomes[outcome_id]
        runtime = self._runtime[outcome_id]
        if outcome.lifecycle == "event":
            raise ValueError("event outcomes do not require final verification")
        if runtime.status not in {"satisfied", "invalidated"}:
            raise ValueError(f"outcome cannot be final-verified: {outcome_id}")
        if not evidence.authoritative:
            raise ValueError("final verification requires authoritative evidence")
        if evidence.predicate_result != "satisfied":
            raise ValueError("final verification requires a satisfied predicate result")
        if (
            not evidence.observation_id
            or not evidence.captured_at
            or not evidence.verifier
        ):
            raise ValueError("final verification requires fresh observation provenance")
        previous_captures = tuple(
            _parse_capture_time(item.captured_at)
            for item in runtime.evidence
            if item.observation_id == evidence.observation_id and item.captured_at
        )
        captured_at = _parse_capture_time(evidence.captured_at)
        if previous_captures and captured_at <= max(previous_captures):
            raise ValueError("final verification evidence must be freshly captured")
        runtime.evidence.append(evidence)
        if runtime.status == "invalidated":
            self._transition(
                outcome_id, "satisfied", reason=evidence.summary, evidence=evidence
            )
        runtime.final_verified = True
        self._refresh_ready()

    def reject_final_verification(
        self, outcome_id: str, evidence: OutcomeEvidence
    ) -> None:
        """Record authoritative negative final evidence and require recovery."""
        evidence = self._require_evidence(evidence)
        self._require_known((outcome_id,))
        runtime = self._runtime[outcome_id]
        outcome = self._outcomes[outcome_id]
        if outcome.lifecycle == "event" or runtime.status != "satisfied":
            raise ValueError(f"outcome cannot fail final verification: {outcome_id}")
        if not evidence.authoritative or evidence.predicate_result != "not_satisfied":
            raise ValueError("final rejection requires authoritative negative evidence")
        if (
            not evidence.observation_id
            or not evidence.captured_at
            or not evidence.verifier
        ):
            raise ValueError("final rejection requires observation provenance")
        runtime.evidence.append(evidence)
        runtime.invalidations.append(evidence.summary)
        self._transition(
            outcome_id, "invalidated", reason=evidence.summary, evidence=evidence
        )
        runtime.final_verified = False
        self._block_direct_dependents(outcome_id, reason=evidence.summary)
        self._transition(
            outcome_id, "recovery_ready", reason="final verification failed"
        )
        self._refresh_ready()

    def prepare_recovery(self, outcome_ids: tuple[str, ...], *, reason: str) -> None:
        self._require_known(outcome_ids, context="recovery")
        if not reason.strip():
            raise ValueError("recovery reason must be non-empty")
        for outcome_id in outcome_ids:
            runtime = self._runtime[outcome_id]
            if runtime.status in {"failed", "invalidated"}:
                self._transition(outcome_id, "recovery_ready", reason=reason)
        self._refresh_ready()

    def suspend_active_for_recovery(self, *, reason: str) -> str | None:
        """Block an unrelated active outcome while an earlier outcome recovers."""
        if not reason.strip():
            raise ValueError("recovery suspension reason must be non-empty")
        outcome_id = self._active_outcome_id
        if outcome_id is None:
            return None
        self._transition(outcome_id, "blocked", reason=reason)
        self._active_outcome_id = None
        return outcome_id

    def final_audit(self) -> dict[str, Any]:
        satisfied = [
            outcome_id
            for outcome_id in self._order
            if self._runtime[outcome_id].status == "satisfied"
        ]
        finally_valid = [
            outcome_id
            for outcome_id in self._order
            if self._runtime[outcome_id].final_verified
        ]
        unresolved = [
            outcome_id
            for outcome_id in self._order
            if self._runtime[outcome_id].status != "satisfied"
            or not self._runtime[outcome_id].final_verified
        ]
        total = len(self._order)
        return {
            "contract_success": not unresolved,
            "atomic_completion_rate": len(satisfied) / total,
            "final_validation_rate": len(finally_valid) / total,
            "illegal_reorder_count": self._illegal_reorder_count,
            "unresolved_ids": unresolved,
        }

    def _validate(self) -> None:
        indices = [item.instruction_index for item in self._definitions]
        if len(indices) != len(set(indices)):
            raise ValueError("outcome instruction_index values must be unique")
        known = set(self._order)
        for outcome in self._definitions:
            if not outcome.id:
                raise ValueError("outcome id must be non-empty")
            if outcome.instruction_index < 1:
                raise ValueError("outcome instruction_index must be positive")
            if not outcome.description:
                raise ValueError(f"outcome description must be non-empty: {outcome.id}")
            if not outcome.success_predicate:
                raise ValueError(
                    f"outcome success_predicate must be non-empty: {outcome.id}"
                )
            if outcome.lifecycle not in _OUTCOME_LIFECYCLES:
                raise ValueError(f"invalid outcome lifecycle: {outcome.lifecycle}")
            if len(outcome.depends_on) != len(set(outcome.depends_on)):
                raise ValueError(f"duplicate dependency for {outcome.id}")
            unknown = sorted(set(outcome.depends_on) - known)
            if unknown:
                raise ValueError(f"unknown dependency for {outcome.id}: {unknown}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(outcome_id: str) -> None:
            if outcome_id in visiting:
                raise ValueError("outcome dependency graph contains a cycle")
            if outcome_id in visited:
                return
            visiting.add(outcome_id)
            for dependency_id in self._outcomes[outcome_id].depends_on:
                visit(dependency_id)
            visiting.remove(outcome_id)
            visited.add(outcome_id)

        for outcome_id in self._order:
            visit(outcome_id)

    def _refresh_ready(self) -> None:
        for outcome_id in self._order:
            runtime = self._runtime[outcome_id]
            if runtime.status not in {"pending", "blocked"}:
                continue
            dependencies_satisfied = all(
                self._runtime[dependency_id].status == "satisfied"
                for dependency_id in self._outcomes[outcome_id].depends_on
            )
            if dependencies_satisfied and runtime.status != "ready":
                self._transition(outcome_id, "ready", reason="dependencies satisfied")

    def _block_direct_dependents(self, outcome_id: str, *, reason: str) -> None:
        for candidate_id in self._order:
            candidate = self._outcomes[candidate_id]
            runtime = self._runtime[candidate_id]
            if outcome_id not in candidate.depends_on:
                continue
            if runtime.status in {"ready", "in_progress"}:
                self._transition(candidate_id, "blocked", reason=reason)
                if self._active_outcome_id == candidate_id:
                    self._active_outcome_id = None

    def _runtime_for_active(self, outcome_id: str) -> _OutcomeRuntime:
        self._require_known((outcome_id,))
        runtime = self._runtime[outcome_id]
        if self._active_outcome_id != outcome_id or runtime.status not in {
            "in_progress",
            "progressing",
        }:
            raise ValueError(f"outcome is not in progress: {outcome_id}")
        return runtime

    def _validate_reorder(
        self,
        outcome_id: str,
        ready_ids: tuple[str, ...],
        reorder: ReorderRequest | None,
    ) -> None:
        if reorder is None or reorder.selected_outcome_id != outcome_id:
            self._illegal_reorder_count += 1
            raise ValueError("out-of-order selection requires a reorder request")
        selected_index = self._outcomes[outcome_id].instruction_index
        expected_deferred = tuple(
            candidate_id
            for candidate_id in ready_ids
            if self._outcomes[candidate_id].instruction_index < selected_index
        )
        if (
            reorder.deferred_outcome_ids != expected_deferred
            or reorder.reason not in _REORDER_REASONS
            or not reorder.evidence
            or any(not item.strip() for item in reorder.evidence)
        ):
            self._illegal_reorder_count += 1
            raise ValueError(
                "reorder must name every earlier ready outcome and include evidence"
            )

    def _require_known(
        self, outcome_ids: tuple[str, ...], *, context: str = ""
    ) -> None:
        unknown = sorted(set(outcome_ids) - set(self._order))
        if unknown:
            prefix = f"{context} " if context else ""
            raise ValueError(f"unknown {prefix}outcome IDs: {unknown}")

    @staticmethod
    def _require_evidence(evidence: OutcomeEvidence) -> OutcomeEvidence:
        if not isinstance(evidence, OutcomeEvidence):
            raise TypeError("evidence must be OutcomeEvidence")
        return evidence

    @classmethod
    def _require_success_evidence(cls, evidence: OutcomeEvidence) -> OutcomeEvidence:
        evidence = cls._require_evidence(evidence)
        if not evidence.authoritative:
            raise ValueError("satisfaction requires authoritative evidence")
        if evidence.predicate_result != "satisfied":
            raise ValueError("satisfaction requires a satisfied predicate result")
        if (
            not evidence.observation_id
            or not evidence.captured_at
            or not evidence.verifier
        ):
            raise ValueError("satisfaction requires observation provenance")
        return evidence

    def _transition(
        self,
        outcome_id: str,
        to_state: OutcomeState,
        *,
        reason: str,
        evidence: OutcomeEvidence | None = None,
    ) -> None:
        runtime = self._runtime[outcome_id]
        from_state = runtime.status
        if to_state not in _TRANSITIONS[from_state]:
            raise ValueError(f"illegal outcome transition: {from_state} -> {to_state}")
        runtime.status = to_state
        payload = {
            "entity_type": "outcome",
            "entity_id": outcome_id,
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
            "execution_id": evidence.execution_id if evidence else None,
            "evidence": asdict(evidence) if evidence else None,
        }
        self._emit("outcome_transitioned", **payload)
        semantic_event = {
            "ready": "outcome_ready",
            "progressing": "outcome_progressed",
            "satisfied": "outcome_satisfied",
            "failed": "outcome_failed",
            "blocked": "outcome_blocked",
            "invalidated": "outcome_invalidated",
            "recovery_ready": "outcome_recovery_ready",
        }.get(to_state)
        if semantic_event is not None:
            self._emit(semantic_event, **payload)


__all__ = [
    "AtomicOutcome",
    "OutcomeEvidence",
    "OutcomeGraph",
    "OutcomeLifecycle",
    "OutcomeState",
    "ReorderRequest",
]
