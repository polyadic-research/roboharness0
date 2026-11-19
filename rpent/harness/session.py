"""Authoritative external session protocol for outcome-aware Harness runs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from rpent.harness.controller import HarnessController
from rpent.harness.lifecycle import (
    HarnessLifecycleState,
    LifecycleEvent,
    allowed_operations,
    transition,
)
from rpent.harness.outcomes import AtomicOutcome, OutcomeEvidence, OutcomeGraph
from rpent.harness.recovery import RecoveryPolicy, RecoveryTracker
from rpent.harness.stages import StageRuntime
from rpent.harness.verification import (
    PredicateVerificationRequest,
    PredicateVerificationResult,
    VerificationCoordinator,
)
from rpent.planning.outcome_program import (
    CompiledStage,
    OutcomeProgram,
    compile_outcome_program,
)
from rpent.planning.schemas import ActionFeedback, PlanStep


class HarnessSession:
    """Own the external lifecycle while composing controller-owned domains."""

    def __init__(
        self,
        controller: HarnessController,
        *,
        verification_coordinator: VerificationCoordinator | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        self.controller = controller
        self.controller.verification_coordinator = verification_coordinator
        self.controller.recovery_tracker = RecoveryTracker(
            recovery_policy or RecoveryPolicy()
        )
        self._lifecycle_state: HarnessLifecycleState | None = None
        self._pending_final_outcome_ids: tuple[str, ...] = ()
        self._recovery_outcome_ids: tuple[str, ...] = ()
        self._pending_action_lease: tuple[str, str] | None = None
        self._finished: dict[str, Any] | None = None

    @property
    def lifecycle_state(self) -> HarnessLifecycleState | None:
        return self._lifecycle_state

    def start_instruction(
        self,
        goal: str,
        observations: list[str],
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compile and register one immutable instruction contract atomically."""
        if (
            self._lifecycle_state is not None
            or self.controller.outcome_graph is not None
        ):
            return self._error("session instruction already started")
        generator = self.controller.plan_generator
        if generator is None:
            return self._error("plan generator is not configured")
        try:
            generated = generator.generate(
                goal=goal,
                observations=observations,
                constraints=self.controller.planning_constraints(list(constraints or ())),
                physical_tools=self.controller.physical_tools,
            )
            program = _compile_generated(goal, generated)
            graph = _graph_from_program(program)
            stages = StageRuntime(program.stages)
            first_stage = stages.active
            steps = self.controller.state.parse_steps(generated.get("steps"))
            _validate_stage_steps(steps, first_stage.id, first_stage.outcome_ids)
            first_outcome_id = steps[0].covers_outcomes[0]
            if first_outcome_id not in graph.ready_ids:
                raise ValueError("first strict step does not cover a ready outcome")
            graph.select(first_outcome_id)
        except Exception as exc:
            return self._error(f"instruction compilation failed: {exc}")

        checkpoint = self._state_checkpoint()
        try:
            registered = self.controller.state.register_plan(
                goal,
                generated["steps"],
                stage_id=first_stage.id,
                stage_outcome_ids=first_stage.outcome_ids,
            )
        except Exception as exc:
            self._restore_state(checkpoint)
            return self._error(f"instruction registration failed safely: {exc}")
        if not registered.get("ok"):
            self._restore_state(checkpoint)
            return self._error(str(registered.get("error", "plan registration failed")))

        self.controller.outcome_program = program
        self.controller.outcome_graph = graph
        self.controller.stage_runtime = stages
        self.controller._strict_goal = goal
        self.controller._strict_strategy = str(generated.get("strategy", "")).strip()
        self.controller._strict_session_active = True
        self._lifecycle_state = HarnessLifecycleState.ACTION_REQUIRED
        return {
            "ok": True,
            "contract_id": program.contract_id,
            **self.execution_status(),
        }

    def execution_status(self) -> dict[str, Any]:
        state = self._lifecycle_state
        if state is None:
            return {"state": None, "allowed_operations": []}
        recovery_action_pending = bool(
            self.controller.recovery_tracker is not None
            and self.controller.recovery_tracker.pending_recovery_action is not None
        )
        payload: dict[str, Any] = {
            "state": state.value,
            "allowed_operations": list(
                allowed_operations(
                    state,
                    recovery_action_pending=recovery_action_pending,
                )
            ),
        }
        graph = self.controller.outcome_graph
        active = self.controller.state.active_step
        if state is HarnessLifecycleState.ACTION_REQUIRED and active is not None:
            payload.update(
                step_id=active.id,
                outcome_id=graph.active_outcome_id if graph else None,
                allowed_tools=list(active.allowed_tools),
            )
        elif state is HarnessLifecycleState.VERIFICATION_REQUIRED:
            if self._pending_final_outcome_ids:
                payload.update(
                    required_operation="verify_outcomes",
                    outcome_ids=list(self._pending_final_outcome_ids),
                    requires_fresh_observation=True,
                )
            elif active is not None and active.step_type == "observation":
                payload.update(
                    required_operation="verify_observation",
                    outcome_id=graph.active_outcome_id if graph else None,
                )
            else:
                payload.update(
                    required_operation="submit_observation",
                    outcome_id=graph.active_outcome_id if graph else None,
                    execution_id=self.controller.state.pending_execution_id,
                )
        if graph is not None:
            payload.update(graph.final_audit())
        return payload

    def snapshot(self) -> dict[str, Any]:
        """Return one immutable-by-copy audit view of the strict session."""
        return {
            **self.controller.strict_snapshot(),
            **self.execution_status(),
            "finished": deepcopy(self._finished),
        }

    def record_action_feedback(self, feedback: ActionFeedback) -> dict[str, Any]:
        rejected = self._require_operation("configured_physical_tool")
        if rejected is not None:
            return rejected
        active = self.controller.state.active_step
        if not isinstance(feedback, ActionFeedback) or not feedback.physical:
            return self._error("physical ActionFeedback is required")
        if active is None or feedback.tool_name not in active.allowed_tools:
            return self._error("feedback tool is not allowed by the active step")
        leased = self._pending_action_lease
        if leased is not None and leased != (feedback.tool_name, feedback.execution_id):
            return self._error(
                "feedback does not match the pending physical action lease"
            )
        candidate_tracker = deepcopy(self._tracker())
        candidate_graph = deepcopy(self._graph())
        checkpoint = self._state_checkpoint()
        try:
            affected = tuple(feedback.raw_result.get("affected_outcome_ids", ()))
            unknown = sorted(set(affected) - set(candidate_graph.outcome_ids))
            if unknown:
                raise ValueError(f"unknown affected outcome IDs: {unknown}")
            if leased is None:
                candidate_tracker.record_physical_action()
            invalidated: tuple[str, ...] = ()
            if affected:
                invalidated = candidate_graph.invalidate_affected(
                    affected,
                    reason=f"affected by physical execution {feedback.execution_id}",
                )
            invalidation_requires_recovery = bool(invalidated)
            if invalidation_requires_recovery:
                candidate_graph.suspend_active_for_recovery(
                    reason="an affected durable outcome must recover first"
                )
            self.controller.state.record_action_feedback(feedback, input_dict={})
            failed = (
                feedback.execution_status != "completed"
                or feedback.outcome.status
                in {
                    "retryable_failure",
                    "failure",
                }
                or invalidation_requires_recovery
            )
            if failed:
                internal = self.controller.state.abandon_pending_action(
                    reason=feedback.error or feedback.outcome.status,
                    evidence=list(feedback.outcome.evidence)
                    or ["physical action failed"],
                    failure_type=feedback.failure_type or feedback.outcome.failure_type,
                )
                if not internal.get("ok"):
                    raise ValueError(str(internal.get("error")))
                if not invalidation_requires_recovery:
                    evidence = OutcomeEvidence(
                        execution_id=feedback.execution_id,
                        source="action-feedback",
                        summary=feedback.error or feedback.outcome.status,
                    )
                    active_outcome_id = candidate_graph.active_outcome_id
                    if active_outcome_id is None:
                        raise ValueError("failed action has no active outcome")
                    candidate_graph.fail(active_outcome_id, evidence)
                    recovery_ids = (active_outcome_id,)
                else:
                    recovery_ids = invalidated
        except Exception as exc:
            self._restore_state(checkpoint)
            if leased is not None:
                self._pending_action_lease = None
                self.controller.recovery_tracker = candidate_tracker
                active_outcome_id = candidate_graph.active_outcome_id
                if active_outcome_id is not None:
                    candidate_graph.fail(
                        active_outcome_id,
                        OutcomeEvidence(
                            execution_id=feedback.execution_id,
                            source="invalid-action-feedback",
                            summary=f"feedback could not be safely absorbed: {exc}",
                        ),
                    )
                    self.controller.outcome_graph = candidate_graph
                    self._recovery_outcome_ids = (active_outcome_id,)
                recovery = self.controller.state.request_recovery(
                    reason="dispatched action feedback was invalid",
                    evidence=[str(exc)],
                )
                if recovery.get("ok"):
                    self._move(LifecycleEvent.ACTION_FAILED)
            if "physical action budget exhausted" in str(exc):
                self._move(LifecycleEvent.BUDGET_EXHAUSTED)
            return self._error(str(exc))
        self.controller.outcome_graph = candidate_graph
        self.controller.recovery_tracker = candidate_tracker
        self._pending_action_lease = None
        if failed:
            self._recovery_outcome_ids = recovery_ids
        self._move(
            LifecycleEvent.ACTION_FAILED if failed else LifecycleEvent.ACTION_PROGRESSED
        )
        return {"ok": True, **self.execution_status()}

    def physical_action_allowed(
        self, tool_name: str, execution_id: str | None = None
    ) -> dict[str, Any]:
        """Preflight one physical dispatch against lifecycle and run budget."""
        rejected = self._require_operation("configured_physical_tool")
        if rejected is not None:
            return {"allowed": False, **rejected}
        active = self.controller.state.active_step
        if active is None or tool_name not in active.allowed_tools:
            return {
                "allowed": False,
                **self._error("tool is not allowed by active step"),
            }
        finishing_violation = self.controller.state.finishing_tool_violation(tool_name)
        if finishing_violation is not None:
            return {"allowed": False, **self._error(finishing_violation)}
        if self._pending_action_lease is not None:
            return {
                "allowed": False,
                **self._error("another physical action lease is already pending"),
            }
        snapshot = self._tracker().snapshot()
        limit = self._tracker().policy.max_total_physical_actions_per_run
        if snapshot["physical_actions_consumed"] >= limit:
            self._move(LifecycleEvent.BUDGET_EXHAUSTED)
            return {
                "allowed": False,
                **self._error("physical action budget exhausted"),
            }
        if not isinstance(execution_id, str) or not execution_id.strip():
            return {
                "allowed": False,
                **self._error("physical action preflight requires execution_id"),
            }
        candidate = deepcopy(self._tracker())
        try:
            candidate.record_physical_action()
        except ValueError as exc:
            self._move(LifecycleEvent.BUDGET_EXHAUSTED)
            return {"allowed": False, **self._error(str(exc))}
        self.controller.recovery_tracker = candidate
        self._pending_action_lease = (tool_name, execution_id.strip())
        return {
            "allowed": True,
            "execution_id": execution_id.strip(),
            "plan_step_id": active.id,
            **self.execution_status(),
        }

    def submit_observation(
        self,
        observation_refs: list[str],
        observed_at: list[str] | None = None,
    ) -> dict[str, Any]:
        rejected = self._require_operation("submit_observation")
        if rejected is not None:
            return rejected
        if self._lifecycle_state is HarnessLifecycleState.OBSERVATION_REQUIRED:
            return self._advance_stage(observation_refs)
        if observed_at is None:
            return self._error("predicate verification requires observation timestamps")
        if self._pending_final_outcome_ids:
            return self._error("verify_outcomes is required for final verification")
        return self._verify_active(observation_refs, observed_at)

    def verify_step(
        self,
        verdict: str,
        evidence: list[str],
        summary: str,
        execution_id: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Accept conservative manual verification without authorizing success."""
        rejected = self._require_operation("verify_step")
        if rejected is not None:
            return rejected
        if verdict in {"fail", "retry"}:
            outcome_id = self._active_outcome_id()
            result = self.controller.state.verify_step(
                "fail",
                evidence,
                summary,
                execution_id=execution_id,
                require_execution_id=self.controller.require_execution_id,
            )
            if not result.get("ok"):
                return self._error(str(result.get("error")))
            self._graph().fail(
                outcome_id,
                OutcomeEvidence(execution_id, "manual-verification", summary),
            )
            self._recovery_outcome_ids = (outcome_id,)
            self._move(LifecycleEvent.VERIFICATION_FAILED)
            return {"ok": True, **self.execution_status()}
        self._move(LifecycleEvent.VERIFICATION_UNCERTAIN)
        return {
            "ok": True,
            "verification": {
                "verdict": "uncertain",
                "reason": "manual evidence cannot authorize outcome satisfaction",
            },
            **self.execution_status(),
        }

    def verify_observation(
        self,
        verdict: str,
        evidence: list[str],
        summary: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        del confidence
        rejected = self._require_operation("verify_observation")
        if rejected is not None:
            return rejected
        if verdict == "fail":
            outcome_id = self._active_outcome_id()
            result = self.controller.state.verify_observation("fail", evidence, summary)
            if not result.get("ok"):
                return self._error(str(result.get("error")))
            self._graph().fail(
                outcome_id,
                OutcomeEvidence(None, "manual-verification", summary),
            )
            self._recovery_outcome_ids = (outcome_id,)
            self._move(LifecycleEvent.VERIFICATION_FAILED)
            return {"ok": True, **self.execution_status()}
        self._move(LifecycleEvent.VERIFICATION_UNCERTAIN)
        return {
            "ok": True,
            "verification": {"verdict": "uncertain"},
            **self.execution_status(),
        }

    def verify_outcomes(
        self,
        outcome_ids: list[str],
        observation_refs: list[str],
        observed_at: list[str],
    ) -> dict[str, Any]:
        rejected = self._require_operation("verify_outcomes")
        if rejected is not None:
            return rejected
        requested = tuple(outcome_ids)
        if requested != self._pending_final_outcome_ids:
            return self._error(
                "final verification must cover the pending outcomes exactly"
            )
        try:
            results = tuple(
                (
                    outcome_id,
                    self._coordinate(outcome_id, observation_refs, observed_at),
                )
                for outcome_id in requested
            )
            undecided = next(
                (
                    (outcome_id, result)
                    for outcome_id, result in results
                    if not result.authorized
                    and not (result.authoritative and result.verdict == "not_satisfied")
                ),
                None,
            )
            if undecided is not None:
                outcome_id, result = undecided
                event = (
                    LifecycleEvent.BUDGET_EXHAUSTED
                    if self._recovery_exhausted(outcome_id)
                    else LifecycleEvent.VERIFICATION_UNCERTAIN
                )
                self._move(event)
                return self._verification_response(result)
            candidate = deepcopy(self._graph())
            negative = next(
                (
                    (outcome_id, result)
                    for outcome_id, result in results
                    if result.authoritative and result.verdict == "not_satisfied"
                ),
                None,
            )
            if negative is not None:
                outcome_id, result = negative
                candidate.reject_final_verification(
                    outcome_id,
                    _outcome_evidence(
                        result,
                        execution_id=None,
                        observation_refs=observation_refs,
                        observed_at=observed_at,
                    ),
                )
                state_result = self.controller.state.request_recovery(
                    reason=result.reason,
                    evidence=list(result.evidence_refs),
                )
                if not state_result.get("ok"):
                    return self._error(str(state_result.get("error")))
                self.controller.outcome_graph = candidate
                self._pending_final_outcome_ids = ()
                self._recovery_outcome_ids = (outcome_id,)
                self._move(LifecycleEvent.VERIFICATION_FAILED)
                return self._verification_response(result)
            for outcome_id, result in results:
                candidate.mark_final_verified(
                    outcome_id,
                    _outcome_evidence(
                        result,
                        execution_id=None,
                        observation_refs=observation_refs,
                        observed_at=observed_at,
                    ),
                )
        except Exception as exc:
            return self._error(f"final verification failed safely: {exc}")
        self.controller.outcome_graph = candidate
        self._pending_final_outcome_ids = ()
        self._complete_step_or_stage()
        return {"ok": True, **self.execution_status()}

    def request_exception(
        self, reason: str, evidence_refs: list[str]
    ) -> dict[str, Any]:
        if self.controller.state.awaiting_verification:
            return self._error(
                "verification is pending; submit accepted fresh evidence "
                "instead of requesting physical recovery"
            )
        if self._pending_final_outcome_ids:
            return self._error(
                "request_exception cannot bypass pending final verification"
            )
        rejected = self._require_operation("request_exception")
        if rejected is not None:
            return rejected
        if self._pending_action_lease is not None:
            return self._error(
                "request_exception cannot bypass a pending physical action lease; "
                "runtime must record feedback or abort the leased action"
            )
        try:
            recovery_outcome_id = self._recovery_outcome_id()
        except ValueError as exc:
            return self._error(str(exc))
        result = self.controller.state.request_recovery(
            reason=reason,
            evidence=evidence_refs,
        )
        if not result.get("ok"):
            return self._error(str(result.get("error")))
        self._recovery_outcome_ids = (recovery_outcome_id,)
        self._move(LifecycleEvent.EXCEPTION_REQUESTED)
        return {"ok": True, **self.execution_status()}

    def abort_physical_action(
        self,
        execution_id: str,
        reason: str,
        evidence: list[str],
    ) -> dict[str, Any]:
        """Fail closed when a dispatched leased action cannot return feedback."""
        rejected = self._require_operation("configured_physical_tool")
        if rejected is not None:
            return rejected
        lease = self._pending_action_lease
        if lease is None or lease[1] != execution_id:
            return self._error("abort does not match the pending physical action lease")
        if not isinstance(reason, str) or not reason.strip():
            return self._error("abort reason must be non-empty")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(item, str) and item.strip() for item in evidence)
        ):
            return self._error("abort evidence must be non-empty strings")
        graph = deepcopy(self._graph())
        outcome_id = graph.active_outcome_id
        if outcome_id is None:
            return self._error("leased action has no active outcome")
        graph.fail(
            outcome_id,
            OutcomeEvidence(
                execution_id=execution_id,
                source="physical-action-abort",
                summary=reason.strip(),
            ),
        )
        checkpoint = self._state_checkpoint()
        requested = self.controller.state.request_recovery(
            reason=reason,
            evidence=evidence,
        )
        if not requested.get("ok"):
            self._restore_state(checkpoint)
            return self._error(str(requested.get("error")))
        self.controller.outcome_graph = graph
        self._recovery_outcome_ids = (outcome_id,)
        self._pending_action_lease = None
        self._move(LifecycleEvent.ACTION_FAILED)
        return {"ok": True, **self.execution_status()}

    def apply_recovery(
        self,
        reason: str,
        observations: list[str],
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        rejected = self._require_operation("apply_recovery")
        if rejected is not None:
            return rejected
        if not isinstance(reason, str) or not reason.strip():
            return self._error("recovery reason must be non-empty")
        if (
            not isinstance(observations, list)
            or not observations
            or not all(isinstance(item, str) and item.strip() for item in observations)
        ):
            return self._error("recovery observations must be non-empty strings")
        if constraints is not None and (
            not isinstance(constraints, list)
            or not all(isinstance(item, str) and item.strip() for item in constraints)
        ):
            return self._error("recovery constraints must be strings")
        active_outcome_id = self._recovery_outcome_id()
        stage = self._stage_for_outcome(active_outcome_id)
        recovering_earlier_stage = stage.id != self._stage_runtime().active.id
        active = self.controller.state.active_step
        attempts = self.controller.state.snapshot()["active_step_attempts"] or 0
        active_matches_target = bool(
            active is not None and active.covers_outcomes == (active_outcome_id,)
        )
        remaining = max(
            0,
            (active.max_attempts if active_matches_target and active else 0) - attempts,
        )
        last_feedback = self.controller.state.last_feedback
        safe_step_retry = bool(
            last_feedback is not None
            and active_matches_target
            and last_feedback.outcome.status == "retryable_failure"
            and last_feedback.outcome.retryable is True
        )
        candidate_tracker = deepcopy(self._tracker())
        decision = candidate_tracker.decide(
            active_outcome_id,
            stage.id,
            safe_step_retry=safe_step_retry,
            step_attempts_remaining=remaining,
        )
        if decision.scope == "terminal_failure":
            self._move(LifecycleEvent.BUDGET_EXHAUSTED)
            return {"ok": False, "scope": decision.scope, **self.execution_status()}
        candidate_graph = deepcopy(self._graph())
        candidate_stages = deepcopy(self._stage_runtime())
        try:
            if decision.scope == "retry_step":
                recovery_ids = (active_outcome_id,)
                generated = None
                replacement_stages = ()
            else:
                generated_or_error = self._prepare_recovery_plan(
                    decision.scope, reason, observations, constraints or ()
                )
                if isinstance(generated_or_error, str):
                    return self._error(generated_or_error)
                generated, replacement_stages, _replacement_steps, recovery_ids = (
                    generated_or_error
                )
                if (
                    decision.scope == "repartition_remaining_stages"
                    and not recovering_earlier_stage
                ):
                    candidate_stages.replace_remaining(replacement_stages, recovery_ids)
            candidate_graph.prepare_recovery(recovery_ids, reason=reason)
            if (
                candidate_graph.active_outcome_id is not None
                and candidate_graph.active_outcome_id != active_outcome_id
            ):
                candidate_graph.suspend_active_for_recovery(reason=reason)
            if candidate_graph.active_outcome_id is None:
                candidate_graph.select(active_outcome_id)
            candidate_tracker.consume(
                decision,
                outcome_id=active_outcome_id,
                stage_id=stage.id,
            )
        except Exception as exc:
            return self._error(f"recovery validation failed safely: {exc}")

        checkpoint = self._state_checkpoint()
        try:
            if decision.scope == "retry_step":
                state_result = self.controller.state.resume_recovery_retry(
                    reason=reason
                )
            elif recovering_earlier_stage:
                assert generated is not None
                state_result = self.controller.state.insert_recovery_plan(
                    reason,
                    generated["steps"],
                    self.controller._strict_goal,
                    stage_id=stage.id,
                    outcome_ids=recovery_ids,
                )
            elif decision.scope == "replan_outcome":
                assert generated is not None
                state_result = self.controller.state.replan_active_step(
                    reason,
                    generated["steps"],
                    self.controller._strict_goal,
                    stage_id=stage.id,
                    outcome_id=active_outcome_id,
                )
            else:
                assert generated is not None
                state_result = self.controller.state.replan(
                    reason,
                    generated["steps"],
                    self.controller._strict_goal,
                    stage_id=replacement_stages[0].id,
                    stage_outcome_ids=replacement_stages[0].outcome_ids,
                    enforce_global_budget=False,
                )
            if not state_result.get("ok"):
                self._restore_state(checkpoint)
                return self._error(str(state_result.get("error")))
        except Exception as exc:
            self._restore_state(checkpoint)
            return self._error(f"recovery registration failed safely: {exc}")
        self.controller.outcome_graph = candidate_graph
        self.controller.stage_runtime = candidate_stages
        self.controller.recovery_tracker = candidate_tracker
        self._move(LifecycleEvent.RECOVERY_APPLIED)
        return {
            "ok": True,
            "scope": decision.scope,
            "outcome_id": active_outcome_id,
            **self.execution_status(),
        }

    def finish(self, status: str, summary: str) -> dict[str, Any]:
        rejected = self._require_operation("finish")
        if rejected is not None:
            return rejected
        normalized = str(status).strip().lower()
        if normalized == "failed":
            normalized = "failure"
        if self._lifecycle_state is HarnessLifecycleState.COMPLETE:
            audit = self._graph().final_audit()
            if normalized != "success" or not audit["contract_success"]:
                return self._error("finish(success) requires contract_success=true")
        elif normalized != "failure":
            return self._error("terminal failure requires finish(failure)")
        if not str(summary).strip():
            return self._error("finish summary must be non-empty")
        self._finished = {"status": normalized, "summary": summary}
        return {"ok": True, **self._finished, **self.execution_status()}

    def _verify_active(
        self, observation_refs: list[str], observed_at: list[str]
    ) -> dict[str, Any]:
        outcome_id = self._active_outcome_id()
        try:
            result = self._coordinate(outcome_id, observation_refs, observed_at)
        except Exception as exc:
            return self._error(f"verification request rejected safely: {exc}")
        if not result.authorized:
            if result.verdict == "not_satisfied" and result.authoritative:
                return self._verification_failed(result, observation_refs, observed_at)
            self._move(LifecycleEvent.VERIFICATION_UNCERTAIN)
            return self._verification_response(result)
        graph = self._graph()
        definition = next(item for item in graph.definitions if item.id == outcome_id)
        evidence = _outcome_evidence(
            result,
            execution_id=self.controller.state.pending_execution_id,
            observation_refs=observation_refs,
            observed_at=observed_at,
        )
        candidate = deepcopy(graph)
        candidate.satisfy(outcome_id, evidence)
        checkpoint = self._state_checkpoint()
        try:
            active = self.controller.state.active_step
            if active is not None and active.step_type == "observation":
                applied = self.controller.state.verify_observation(
                    "pass", list(result.evidence_refs), result.reason
                )
            else:
                applied = self.controller.state.verify_step(
                    "pass",
                    list(result.evidence_refs),
                    result.reason,
                    execution_id=self.controller.state.pending_execution_id,
                    require_execution_id=self.controller.require_execution_id,
                    source="verifier",
                )
        except Exception as exc:
            self._restore_state(checkpoint)
            return self._error(f"verification state update failed safely: {exc}")
        if not applied.get("ok"):
            self._restore_state(checkpoint)
            return self._error(str(applied.get("error")))
        self.controller.outcome_graph = candidate
        if definition.lifecycle != "event":
            self._pending_final_outcome_ids = (outcome_id,)
            self._move(LifecycleEvent.OUTCOME_SATISFIED_AWAITING_FINAL_VERIFICATION)
        else:
            self._complete_step_or_stage()
        return self._verification_response(result, ok=True)

    def _verification_failed(
        self,
        result: PredicateVerificationResult,
        observation_refs: list[str],
        observed_at: list[str],
    ) -> dict[str, Any]:
        outcome_id = self._active_outcome_id()
        evidence = _outcome_evidence(
            result,
            execution_id=self.controller.state.pending_execution_id,
            observation_refs=observation_refs,
            observed_at=observed_at,
        )
        candidate = deepcopy(self._graph())
        candidate.fail(outcome_id, evidence)
        checkpoint = self._state_checkpoint()
        active = self.controller.state.active_step
        if active is not None and active.step_type == "observation":
            applied = self.controller.state.verify_observation(
                "fail", [result.reason], result.reason
            )
        else:
            applied = self.controller.state.verify_step(
                "fail",
                [result.reason],
                result.reason,
                execution_id=self.controller.state.pending_execution_id,
                require_execution_id=self.controller.require_execution_id,
                source="verifier",
            )
        if not applied.get("ok"):
            self._restore_state(checkpoint)
            return self._error(str(applied.get("error")))
        self.controller.outcome_graph = candidate
        self._recovery_outcome_ids = (outcome_id,)
        self._move(LifecycleEvent.VERIFICATION_FAILED)
        return self._verification_response(result)

    def _complete_step_or_stage(self) -> None:
        graph = self._graph()
        remaining_recovery = self._remaining_recovery_ids()
        if remaining_recovery:
            requested = self.controller.state.request_recovery(
                reason="another invalidated outcome still requires recovery",
                evidence=["immutable recovery target remains unresolved"],
            )
            if not requested.get("ok"):
                raise RuntimeError(str(requested.get("error")))
            self._move(LifecycleEvent.VERIFICATION_FAILED)
            return
        self._recovery_outcome_ids = ()
        if not self.controller.state.completed:
            self._assign_active_outcome()
            self._move(LifecycleEvent.OUTCOME_SATISFIED)
            return
        stage_ids = self._stage_runtime().active.outcome_ids
        snapshot = {item["id"]: item for item in graph.snapshot()["outcomes"]}
        stage_complete = all(
            snapshot[outcome_id]["status"] == "satisfied"
            and snapshot[outcome_id]["final_verified"]
            for outcome_id in stage_ids
        )
        if not stage_complete:
            self._move(LifecycleEvent.VERIFICATION_UNCERTAIN)
            return
        if self._stage_runtime().has_next:
            self._move(LifecycleEvent.STAGE_COMPLETED)
        else:
            audit = graph.final_audit()
            if audit["contract_success"]:
                self._move(LifecycleEvent.CONTRACT_COMPLETED)
            else:
                self._move(LifecycleEvent.VERIFICATION_UNCERTAIN)

    def _advance_stage(self, observation_refs: list[str]) -> dict[str, Any]:
        runtime = self._stage_runtime()
        next_stage = runtime.stages[runtime.active_index + 1]
        remaining_stages = runtime.stages[runtime.active_index + 1 :]
        remaining_ids = tuple(
            outcome_id for stage in remaining_stages for outcome_id in stage.outcome_ids
        )
        if next_stage.outcome_ids[0] not in self._graph().ready_ids:
            return self._error(
                "next stage cannot advance because its first outcome is not ready"
            )
        generator = self.controller.plan_generator
        assert generator is not None
        try:
            generated = generator.generate_replan(
                goal=self.controller._strict_goal,
                reason="advance to the next outcome stage",
                observations=observation_refs,
                constraints=self.controller.planning_constraints([]),
                current_plan=self._planning_context(
                    remaining_ids,
                    active_outcome_id=next_stage.outcome_ids[0],
                    active_stage_id=next_stage.id,
                ),
                physical_tools=self.controller.physical_tools,
            )
            proposed_stages = _compiled_stages(generated.get("stages"))
            if proposed_stages != remaining_stages:
                raise ValueError(
                    "next-stage proposal must preserve every remaining contract stage"
                )
            steps = self.controller.state.parse_steps(generated.get("steps"))
            _validate_stage_steps(steps, next_stage.id, next_stage.outcome_ids)
        except Exception as exc:
            return self._error(f"next-stage generation failed: {exc}")
        registered = self.controller.state.register_next_stage(
            self.controller._strict_goal or "instruction",
            generated["steps"],
            stage_id=next_stage.id,
            stage_outcome_ids=next_stage.outcome_ids,
        )
        if not registered.get("ok"):
            return self._error(str(registered.get("error")))
        runtime.advance(stage_complete=True)
        self._assign_active_outcome()
        self._move(LifecycleEvent.STAGE_ADVANCED)
        return {"ok": True, **self.execution_status()}

    def _prepare_recovery_plan(self, scope, reason, observations, constraints):
        graph = self._graph()
        recovery_outcome_id = self._recovery_outcome_id()
        stage = self._stage_for_outcome(recovery_outcome_id)
        if scope == "replan_outcome":
            recovery_ids = (recovery_outcome_id,)
        elif (
            scope == "repartition_remaining_stages"
            and stage.id == self._stage_runtime().active.id
        ):
            remaining_stage_ids = tuple(
                outcome_id
                for candidate in self._stage_runtime().stages[
                    self._stage_runtime().active_index :
                ]
                for outcome_id in candidate.outcome_ids
            )
            recovery_ids = graph.recovery_ids_for(remaining_stage_ids)
        else:
            recovery_ids = graph.recovery_ids_for(stage.outcome_ids)
        if not recovery_ids:
            recovery_ids = (self._recovery_outcome_id(),)
        generator = self.controller.plan_generator
        assert generator is not None
        try:
            generated = generator.generate_replan(
                goal=self.controller._strict_goal,
                reason=reason,
                observations=list(observations),
                constraints=self.controller.planning_constraints(list(constraints)),
                current_plan=self._planning_context(
                    recovery_ids,
                    active_outcome_id=recovery_outcome_id,
                    active_stage_id=stage.id,
                ),
                physical_tools=self.controller.physical_tools,
            )
            raw_stages = generated.get("stages")
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError("recovery plan requires stages")
            from rpent.planning.outcome_program import CompiledStage

            stages = tuple(
                CompiledStage(
                    id=str(item.get("id", "")).strip(),
                    intent=str(item.get("intent", "")).strip(),
                    outcome_ids=tuple(item.get("outcome_ids", ())),
                )
                for item in raw_stages
            )
            flattened = tuple(value for item in stages for value in item.outcome_ids)
            if (
                any(
                    not item.id or not item.intent or not item.outcome_ids
                    for item in stages
                )
                or len({item.id for item in stages}) != len(stages)
                or any(
                    len(item.outcome_ids) != len(set(item.outcome_ids))
                    for item in stages
                )
            ):
                raise ValueError("recovery stages must be non-empty and unique")
            if flattened != recovery_ids:
                raise ValueError(
                    "replacement stages must exactly partition recovery outcomes"
                )
            if scope != "repartition_remaining_stages" and (
                len(stages) != 1 or stages[0].id != stage.id
            ):
                raise ValueError("local recovery must preserve the active stage ID")
            if (
                scope == "repartition_remaining_stages"
                and stage.id == self._stage_runtime().active.id
            ):
                candidate = StageRuntime(
                    self._stage_runtime().stages,
                    active_index=self._stage_runtime().active_index,
                )
                candidate.replace_remaining(stages, recovery_ids)
            steps = self.controller.state.parse_steps(generated.get("steps"))
            _validate_stage_steps(steps, stages[0].id, stages[0].outcome_ids)
        except Exception as exc:
            return f"recovery generation failed: {exc}"
        return generated, stages, steps, recovery_ids

    def _planning_context(
        self,
        required_ids: tuple[str, ...],
        *,
        active_outcome_id: str | None = None,
        active_stage_id: str | None = None,
    ) -> dict[str, Any]:
        outcomes = self._graph().snapshot()["outcomes"]
        completed_ids = [
            item["id"] for item in outcomes if item["status"] == "satisfied"
        ]
        required = set(required_ids)
        return {
            "goal": self.controller._strict_goal,
            "strategy": self.controller._strict_strategy,
            "outcome_program": self.controller.outcome_program.to_payload(),
            "required_outcome_ids": list(required_ids),
            "completed_outcome_ids": completed_ids,
            "deferred_outcome_ids": [
                item["id"]
                for item in outcomes
                if item["status"] != "satisfied" and item["id"] not in required
            ],
            "active_outcome_id": active_outcome_id or self._recovery_outcome_id(),
            "active_stage_id": active_stage_id or self._stage_runtime().active.id,
            **self.controller.state.control_planning_context(include_capabilities=False),
        }

    def _coordinate(self, outcome_id, refs, observed_at):
        coordinator = self.controller.verification_coordinator
        if coordinator is None:
            return PredicateVerificationResult(
                verdict="not_applicable",
                evidence_refs=(),
                verifier="verification-coordinator",
                verifier_version="1",
                confidence=None,
                authoritative=False,
                reason="no predicate verifier is configured",
            )
        definition = next(
            item for item in self._graph().definitions if item.id == outcome_id
        )
        feedback = self.controller.state.last_feedback
        execution_id = self.controller.state.pending_execution_id
        if execution_id is None and feedback is not None:
            outcome = next(
                item
                for item in self._graph().snapshot()["outcomes"]
                if item["id"] == outcome_id
            )
            if any(
                item["execution_id"] == feedback.execution_id
                for item in outcome["evidence"]
            ):
                execution_id = feedback.execution_id
        request = PredicateVerificationRequest(
            run_id=self.controller.trace_path.stem or "harness-run",
            contract_id=self.controller.outcome_program.contract_id,
            outcome_id=outcome_id,
            success_predicate=definition.success_predicate,
            execution_id=execution_id,
            observation_refs=tuple(refs),
            observed_at=tuple(observed_at),
            feedback=feedback,
        )
        return coordinator.verify(request)

    def _assign_active_outcome(self) -> None:
        self._assign_active_outcome_on(self._graph())

    def _assign_active_outcome_on(self, graph: OutcomeGraph) -> None:
        if graph.active_outcome_id is not None:
            return
        active = self.controller.state.active_step
        if active is None:
            return
        ready = set(graph.ready_ids)
        outcome_id = next(
            (value for value in active.covers_outcomes if value in ready), None
        )
        if outcome_id is None:
            raise ValueError("active step does not cover a ready outcome")
        graph.select(outcome_id)

    def _recovery_outcome_id(self) -> str:
        graph = self._graph()
        remaining = self._remaining_recovery_ids()
        if remaining:
            return remaining[0]
        if graph.active_outcome_id is not None:
            return graph.active_outcome_id
        stage_ids = graph.recovery_ids_for(self._stage_runtime().active.outcome_ids)
        if stage_ids:
            return stage_ids[0]
        active = self.controller.state.active_step
        if active and active.covers_outcomes:
            return active.covers_outcomes[0]
        runtime = self._stage_runtime()
        if runtime.has_next:
            return runtime.stages[runtime.active_index + 1].outcome_ids[0]
        raise ValueError("recovery has no unresolved outcome")

    def _remaining_recovery_ids(self) -> tuple[str, ...]:
        if not self._recovery_outcome_ids or self.controller.outcome_graph is None:
            return ()
        snapshot = {
            item["id"]: item
            for item in self.controller.outcome_graph.snapshot()["outcomes"]
        }
        return tuple(
            outcome_id
            for outcome_id in self._recovery_outcome_ids
            if snapshot[outcome_id]["status"] != "satisfied"
            or not snapshot[outcome_id]["final_verified"]
        )

    def _stage_for_outcome(self, outcome_id: str) -> Any:
        for stage in self._stage_runtime().stages:
            if outcome_id in stage.outcome_ids:
                return stage
        raise ValueError(f"recovery outcome has no stage: {outcome_id}")

    def _active_outcome_id(self) -> str:
        outcome_id = self._graph().active_outcome_id
        if outcome_id is None:
            raise ValueError("no outcome is active")
        return outcome_id

    def _graph(self) -> OutcomeGraph:
        graph = self.controller.outcome_graph
        if graph is None:
            raise ValueError("outcome graph is not initialized")
        return graph

    def _stage_runtime(self) -> StageRuntime[Any]:
        runtime = self.controller.stage_runtime
        if runtime is None:
            raise ValueError("stage runtime is not initialized")
        return runtime

    def _tracker(self) -> RecoveryTracker:
        tracker = self.controller.recovery_tracker
        if tracker is None:
            raise ValueError("recovery tracker is not initialized")
        return tracker

    def _recovery_exhausted(self, outcome_id: str) -> bool:
        stage = self._stage_for_outcome(outcome_id)
        decision = deepcopy(self._tracker()).decide(
            outcome_id,
            stage.id,
            safe_step_retry=False,
            step_attempts_remaining=0,
        )
        return decision.scope == "terminal_failure"

    def _state_checkpoint(self) -> dict[str, Any]:
        return deepcopy(
            {
                key: value
                for key, value in self.controller.state.__dict__.items()
                if key != "_record"
            }
        )

    def _restore_state(self, checkpoint: dict[str, Any]) -> None:
        recorder = self.controller.state._record
        self.controller.state.__dict__.clear()
        self.controller.state.__dict__.update(deepcopy(checkpoint))
        self.controller.state._record = recorder

    def _move(self, event: LifecycleEvent) -> None:
        if self._lifecycle_state is None:
            raise ValueError("session is not started")
        self._lifecycle_state = transition(self._lifecycle_state, event)

    def _require_operation(self, operation: str) -> dict[str, Any] | None:
        state = self._lifecycle_state
        pending = bool(
            self.controller.recovery_tracker is not None
            and self.controller.recovery_tracker.pending_recovery_action is not None
        )
        available = (
            ()
            if state is None
            else allowed_operations(state, recovery_action_pending=pending)
        )
        if operation not in available:
            if pending and operation == "request_exception":
                return self._error(
                    "request_exception cannot consume another recovery scope before "
                    "the recovered plan attempts a physical action"
                )
            label = state.value if state is not None else "unstarted"
            return self._error(f"{operation} is not allowed in state {label}")
        return None

    def _verification_response(
        self, result: PredicateVerificationResult, *, ok: bool = False
    ) -> dict[str, Any]:
        return {
            **({"ok": True} if ok else {}),
            "verification": asdict(result),
            **self.execution_status(),
        }

    def _error(self, message: str) -> dict[str, Any]:
        return {"error": message, **self.execution_status()}


def _compile_generated(goal: str, generated: dict[str, Any]) -> OutcomeProgram:
    existing = generated.get("outcome_program")
    serialized = isinstance(existing, (OutcomeProgram, dict))
    if isinstance(existing, OutcomeProgram):
        program = _program_from_payload(existing.to_payload())
    elif isinstance(existing, dict):
        program = _program_from_payload(existing)
    else:
        program = compile_outcome_program(
            goal,
            generated.get("outcomes"),
            generated.get("stages"),
        )
    if program.source_instruction.strip() != str(goal).strip():
        raise ValueError("outcome program source_instruction does not match goal")
    if "goal" in generated and str(generated["goal"]).strip() != str(goal).strip():
        raise ValueError("generated plan goal does not match source instruction")
    identities = {
        "contract_id": program.contract_id,
        "outcome_program_schema_version": program.schema_version,
    }
    if serialized:
        for field, expected in identities.items():
            if field in generated and generated[field] != expected:
                raise ValueError(f"top-level {field} does not match contract identity")
        for field, expected in (
            ("outcomes", program.to_payload()["outcomes"]),
            ("stages", program.to_payload()["stages"]),
        ):
            if field in generated and _canonical(generated[field]) != _canonical(
                expected
            ):
                raise ValueError(f"top-level {field} does not match contract identity")
        if "required_outcome_ids" in generated and tuple(
            generated["required_outcome_ids"]
        ) != tuple(item.id for item in program.outcomes):
            raise ValueError(
                "top-level required_outcome_ids does not match contract identity"
            )
    return program


def _program_from_payload(payload: dict[str, Any]) -> OutcomeProgram:
    try:
        diagnostics_raw = payload["compile_diagnostics"]
        outcomes_raw = payload["outcomes"]
        stages_raw = payload["stages"]
        explicit = None
        if diagnostics_raw.get("confidence") == 1.0:
            explicit = [
                {
                    "id": item["id"],
                    "number": item["instruction_index"],
                    "text": item["source_text"],
                }
                for item in outcomes_raw
            ]
        regenerated = compile_outcome_program(
            payload["source_instruction"],
            [
                {
                    "id": item["id"],
                    "instruction_index": item["instruction_index"],
                    "description": item["description"],
                    "success_predicate": item["success_predicate"],
                    "depends_on": item["depends_on"],
                    "lifecycle": item["lifecycle"],
                }
                for item in outcomes_raw
            ],
            list(stages_raw),
            explicit_contract=explicit,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid serialized outcome program: {exc}") from exc
    if _canonical(regenerated.to_payload()) != _canonical(payload):
        raise ValueError("serialized outcome program contract identity is invalid")
    return regenerated


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _graph_from_program(program: OutcomeProgram) -> OutcomeGraph:
    return OutcomeGraph(
        [
            AtomicOutcome(
                id=item.id,
                instruction_index=item.instruction_index,
                description=item.description,
                success_predicate=item.success_predicate,
                depends_on=item.depends_on,
                lifecycle=item.lifecycle,  # type: ignore[arg-type]
            )
            for item in program.outcomes
        ]
    )


def _validate_stage_steps(
    steps: list[PlanStep], stage_id: str, outcome_ids: tuple[str, ...]
) -> None:
    if any(len(step.covers_outcomes) != 1 for step in steps):
        raise ValueError("each strict staged step must cover exactly one outcome")
    flattened = tuple(value for step in steps for value in step.covers_outcomes)
    if flattened != outcome_ids:
        raise ValueError(
            "stage steps must cover every stage outcome exactly once in order"
        )
    if any(step.stage_id != stage_id for step in steps):
        raise ValueError("stage steps must use the active stage ID")


def _compiled_stages(raw: Any) -> tuple[CompiledStage, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("remaining stages must be a non-empty array")
    return tuple(
        CompiledStage(
            id=str(item.get("id", "")).strip(),
            intent=str(item.get("intent", "")).strip(),
            outcome_ids=tuple(item.get("outcome_ids", ())),
        )
        for item in raw
    )


def _outcome_evidence(
    result: PredicateVerificationResult,
    *,
    execution_id: str | None,
    observation_refs: list[str],
    observed_at: list[str],
) -> OutcomeEvidence:
    evidence_ref = result.evidence_refs[0]
    index = observation_refs.index(evidence_ref)
    return OutcomeEvidence(
        execution_id=execution_id,
        source="predicate-verifier",
        summary=result.reason,
        confidence=result.confidence,
        observation_id=evidence_ref,
        captured_at=observed_at[index],
        verifier=result.verifier,
        verifier_version=result.verifier_version,
        predicate_result=result.verdict,
        authoritative=result.authoritative,
    )


__all__ = ["HarnessSession"]
