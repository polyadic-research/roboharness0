"""Provider-independent plan state machine for Harness execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Callable
from uuid import uuid4

from robots.robocasa.capabilities import (
    CONTROL_MODES,
    PRIVILEGED_PHYSICAL_TOOLS,
    READ_ONLY_PERCEPTION_TOOLS,
    TIERS,
    VLA_TOOLS,
    capability_entry,
    capability_summary,
    is_composite_task,
    load_capabilities,
    unmeasured_composite,
)
from robots.robocasa.control_modes import (
    DESTRUCTIVE_FAILURE_TYPES,
    ESCALATION_TYPICAL_FACTOR,
    FALLBACK_FAILURES,
    FALLBACK_RESERVE_STEPS,
    LLM_DRIVEN_MODES,
    MAX_LLM_FALLBACKS,
    MIN_SEGMENT_STEPS,
    NO_PROGRESS_WINDOW_STEPS,
    POLICY_DRIVEN_MODES,
    PROGRESS_DISTANCE_M,
    SEGMENT_CALL_QUOTA,
    SEGMENT_NO_PROGRESS_STEPS,
    SEGMENT_STEPS_PER_CALL,
    SUBTASK_CARRY,
    SUBTASK_WHOLE,
    TAKEOVER_NONE_STEPS,
    TAKEOVER_TYPICAL_FACTOR,
    TRANSITION_BASE_STEPS,
    TRANSITION_DEFAULT_TRAVEL_M,
    TRANSITION_MAX_TRAVEL_M,
    TRANSITION_STEPS_PER_M,
)
from rpent.harness.finishing import finishing_motion
from rpent.harness.lifecycle import (
    FINISHING_FALLBACK_FLOOR_STEPS,
    FINISHING_LEASE_CALLS,
    FINISHING_LEASE_STEPS,
    FINISHING_MAX_BORROW_STEPS,
    FINISHING_PROGRESS_CALLS,
    FINISHING_TOOLS,
    POLICY_PHYSICAL_FAILURE_TYPES,
    TASK_CONFLICT_TYPES,
    consumes_physical_attempt,
)
from rpent.harness.policy_progress import review_policy_progress
from rpent.planning.schemas import ActionFeedback, PlanStep, StepVerdict

RecordEvent = Callable[..., None]
# (operation identity, subtask name); see ``operation_key`` and ``subtask_name``.
SubtaskKey = tuple[tuple[str, ...], str]
MAX_STEP_ATTEMPTS = 64
_MAX_IDENTITY_CHARS = 120


class PlanStateMachine:
    """Own plan progress, verification state, and bounded repair state."""

    def __init__(
        self,
        *,
        max_replans: int | None,
        environment_horizon_only: bool = False,
        physical_tools: set[str] | None = None,
        record: RecordEvent | None = None,
        policy_family: str | None = None,
        task_name: str | None = None,
        transition_takeover_enabled: bool = True,
        op_takeover_enabled: bool = True,
    ) -> None:
        if max_replans is not None and max_replans < 0:
            raise ValueError("max_replans must be >= 0")
        self.max_replans = max_replans
        self.environment_horizon_only = environment_horizon_only
        self.physical_tools = set(physical_tools or ())
        self._record = record or (lambda event, **payload: None)
        self._steps: list[PlanStep] = []
        self._active_index = 0
        self._awaiting_verification = False
        self._needs_replan = False
        self._replans = 0
        self._execution_mode = "observe"
        self._pending_execution_id: str | None = None
        self._last_feedback: ActionFeedback | None = None
        self._environment_success_seen = False
        self._last_step_id: int | None = None
        self._finishing_borrowed: dict[tuple, int] = {}
        self._finishing_limits: dict[tuple, int] = {}
        self._finishing_withdrawn: set[tuple] = set()
        self._last_verdict: StepVerdict | None = None
        self._step_attempts: dict[str, int] = {}
        # Control-mode accounting per subtask (operation identity + subtask
        # name), kept across replanning: steps, real failures, progress
        # windows, LLM segments with quotas, fallbacks and escalations.
        self._subtasks: dict[SubtaskKey, dict[str, Any]] = {}
        # Measured facts of the physical execution awaiting its verdict.
        self._pending_accounting: dict[str, Any] | None = None
        self._episode_horizon: int | None = None
        self._original_policy_instruction: str | None = None
        self._whole_task_recovery_authorized = False
        self._semantic_policy_replanning_authorized = False
        self._semantic_policy_restricted = False
        self._semantic_scope_identity: SubtaskKey | None = None
        self.configure_control_modes(
            policy_family=policy_family, task_name=task_name,
            transition_takeover_enabled=transition_takeover_enabled,
            op_takeover_enabled=op_takeover_enabled,
        )

    def configure_control_modes(
        self, *, policy_family: str | None, task_name: str | None,
        transition_takeover_enabled: bool = True, op_takeover_enabled: bool = True,
    ) -> None:
        """Freeze selected VLA/task contracts before plan registration."""
        if self.has_plan:
            raise ValueError("control configuration is frozen by the registered plan")
        if type(transition_takeover_enabled) is not bool or type(op_takeover_enabled) is not bool:
            raise ValueError("control-mode ablation switches must be booleans")
        self.policy_family = policy_family
        self.task_name = task_name
        self.transition_takeover_enabled = transition_takeover_enabled
        self.op_takeover_enabled = op_takeover_enabled
        self.capabilities = load_capabilities(policy_family) if policy_family else None
        self._unmeasured_composite_task = unmeasured_composite(self.capabilities, task_name)
        self._composite_task = is_composite_task(task_name)
        if policy_family and self.capabilities is None:
            self._record("capability_table_missing", policy=policy_family, tier="none",
                         warning="selected VLA has no measured capability catalog")

    def set_episode_horizon(self, horizon: int | None) -> None:
        """Bind the episode horizon that caps LLM segment quotas (None: uncapped)."""
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
            raise ValueError("episode horizon must be a positive integer")
        self._episode_horizon = horizon

    def set_policy_instruction(self, instruction: str) -> None:
        """Bind the live environment sentence, never the planner's subtask text."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("original policy instruction must be non-empty")
        if self._original_policy_instruction not in {None, instruction}:
            raise ValueError("original policy instruction is frozen for this episode")
        self._original_policy_instruction = instruction

    @property
    def whole_task_policy_first_active(self) -> bool:
        return (
            self._unmeasured_composite_task
            and not self._whole_task_recovery_authorized
        )

    @property
    def semantic_policy_replanning_allowed(self) -> bool:
        return self._semantic_policy_replanning_authorized

    def parse_steps(self, steps: list[dict[str, Any]]) -> list[PlanStep]:
        """Validate proposals and registration through the same control contract."""
        return parse_steps(
            steps, self.physical_tools, policy_family=self.policy_family,
            task_name=self.task_name,
            transition_takeover_enabled=self.transition_takeover_enabled,
            op_takeover_enabled=self.op_takeover_enabled,
            control_constraints=self.control_constraints(),
        )

    def escalated_operations(self) -> set[SubtaskKey]:
        """Subtasks whose takeover is legal regardless of tier; fallbacks excluded."""
        return {
            key for key, ledger in self._subtasks.items()
            if ledger["escalation"] is not None and ledger["fallback"] is None
        }

    def control_constraints(self) -> dict[str, Any]:
        """Per-subtask plan constraints shared by registration and candidate parsing."""
        return {
            "whole_task_policy_first": (
                self.whole_task_policy_first_active and not self.semantic_policy_replanning_allowed
            ),
            "semantic_policy_only": self._semantic_policy_restricted,
            "semantic_scope_identity": self._semantic_scope_identity,
            "original_policy_instruction": self._original_policy_instruction,
            "whole_task_operation": next(
                (operation_key(step) for step in self._steps if step.step_type == "action"), None,
            ),
            "fallbacks": {
                key: ledger["fallback"]["from_mode"]
                for key, ledger in self._subtasks.items() if ledger["fallback"] is not None
            },
            "escalated": self.escalated_operations(),
            "llm_exhausted": {
                key for key, ledger in self._subtasks.items()
                if ledger["fallbacks"] >= MAX_LLM_FALLBACKS and ledger["segment"] is None
            },
        }

    def control_planning_context(self, *, include_capabilities: bool = True) -> dict[str, Any]:
        def ident(key: SubtaskKey) -> dict[str, Any]:
            return {"operation_key": list(key[0]), "subtask": key[1]}

        ledgers = sorted(self._subtasks.items())
        context = {
            "policy": self.policy_family,
            "transition_takeover_enabled": self.transition_takeover_enabled,
            "op_takeover_enabled": self.op_takeover_enabled,
            # Only physically executed failures count (1.0.11 A4).
            "control_mode_failures": [
                {**ident(key), "failed_verifications": dict(ledger["failures"])}
                for key, ledger in ledgers if ledger["failures"]
            ],
            "control_mode_fallbacks": [dict(ledger["fallback"]) for _, ledger in ledgers if ledger["fallback"]],
            "control_mode_escalations": [
                {**ledger["escalation"], **ident(key)}
                for key, ledger in ledgers
                if key in self.escalated_operations()
            ],
            "control_segments": [
                {**ident(key), **_segment_view(ledger)} for key, ledger in ledgers if ledger["segment"]
            ],
        }
        if include_capabilities:
            # Planning requests only; tool-result snapshots stay compact.
            context["control_subtasks"] = [
                {
                    **ident(key), "policy_steps": ledger["policy_steps"], "llm_steps": ledger["llm_steps"],
                    "steps_since_progress": ledger["since_progress"], "passed": ledger["passed"],
                    "llm_segments_used": ledger["segments"], "llm_fallbacks": ledger["fallbacks"],
                    "max_llm_fallbacks": MAX_LLM_FALLBACKS,
                }
                for key, ledger in ledgers
            ]
            context["capabilities"] = capability_summary(self.capabilities, task_name=self.task_name)
        if self._unmeasured_composite_task:
            context["task_execution_scope"] = {
                "task_name": self.task_name, "whole_task_first": True,
                "skill_row": "none", "tier": "none", "control_mode": "monitor",
                "instruction": self._original_policy_instruction,
                "decomposition_allowed": self.semantic_policy_replanning_allowed or self._whole_task_recovery_authorized,
                "motor_takeover_allowed": self._whole_task_recovery_authorized,
            }
        if self._composite_task:
            context["policy_review_scope"] = {
                "semantic_replanning_allowed": self.semantic_policy_replanning_allowed,
                "policy_only": self._semantic_policy_restricted,
                "current_goal_progress_evidence": True,
                "instruction_change_required": False,
            }
        return context

    @property
    def has_plan(self) -> bool:
        return bool(self._steps)

    @property
    def awaiting_verification(self) -> bool:
        return self._awaiting_verification

    @property
    def needs_replan(self) -> bool:
        return self._needs_replan

    @property
    def can_replan(self) -> bool:
        """Permit a replacement after a resolved retry, never while evidence is pending."""
        return not self._awaiting_verification and (
            self._needs_replan or (
                self._execution_mode in {"recover", "plan"}
                and self._last_verdict is not None
                and self._last_verdict.verdict == "retry"
            )
        )

    @property
    def completed(self) -> bool:
        return bool(self._steps) and self._active_index >= len(self._steps)

    @property
    def completion_review_required(self) -> bool:
        """Require reconsideration of local passes at a live completion mismatch."""
        feedback = self.last_feedback
        return (
            self.environment_horizon_only
            and self.completed
            and self.needs_replan
            and not self.environment_success
            and feedback is not None
            and feedback.execution_status == "completed"
            and not feedback.environment_terminated
            and not feedback.environment_truncated
        )

    def reopen_completed_subtask(self, key: SubtaskKey) -> None:
        """Withdraw local pass credit without replenishing any execution budget."""
        ledger = self._subtasks.get(key)
        if ledger is not None and ledger["passed"]:
            ledger["passed"] = False
            self._record(
                "completed_subtask_reopened", operation_key=list(key[0]), subtask=key[1],
                reason="completed-step review requires repair or remains uncertain",
            )

    @property
    def active_step(self) -> PlanStep | None:
        if 0 <= self._active_index < len(self._steps):
            return self._steps[self._active_index]
        return None

    @property
    def execution_mode(self) -> str:
        """Return the current observe/plan/act/verify/recover lifecycle mode."""
        return self._execution_mode

    @property
    def pending_execution_id(self) -> str | None:
        """Return the physical execution currently awaiting verification."""
        return self._pending_execution_id

    @property
    def pending_verification_execution_id(self) -> str | None:
        """Return the execution still eligible for predicate verification."""
        return self._pending_execution_id

    @property
    def environment_success(self) -> bool:
        """Return whether completed feedback has reached env success.

        Sticky: the episode ends at success, so a later call the toolkit then
        refuses must not erase it (1.0.11 review).
        """
        return self._environment_success_seen

    @property
    def replans_used(self) -> int:
        return self._replans

    @property
    def replans_remaining(self) -> int | None:
        if self.max_replans is None:
            return None
        return self.max_replans - self._replans

    @property
    def last_verdict(self) -> StepVerdict | None:
        return self._last_verdict

    @property
    def last_feedback(self) -> ActionFeedback | None:
        """Return the latest normalized physical feedback."""
        return self._last_feedback

    def enter_plan_mode(self, *, reason: str) -> None:
        """Switch to read-only planning without mutating the active plan."""
        self._set_execution_mode("plan", reason=reason)

    def _set_execution_mode(self, mode: str, *, reason: str) -> None:
        if mode == self._execution_mode:
            return
        previous = self._execution_mode
        self._execution_mode = mode
        self._record(
            "execution_mode_changed",
            previous_mode=previous,
            mode=mode,
            reason=reason,
        )

    def set_physical_tools(self, tools: set[str]) -> None:
        self.physical_tools = set(tools)
        self._record("physical_tools_configured", tools=sorted(self.physical_tools))

    def snapshot(self) -> dict[str, Any]:
        active = self.active_step
        attempts = self._step_attempts.get(active.id, 0) if active else 0
        return {
            "has_plan": self.has_plan,
            "can_replan": self.can_replan,
            "status": (
                "needs_replan"
                if self._needs_replan
                else "completed"
                if self.completed
                else "awaiting_verification"
                if self._awaiting_verification
                else "ready"
                if active is not None
                else "unplanned"
            ),
            "active_step_index": self._active_index if active is not None else None,
            "active_step": asdict(active) if active is not None else None,
            "steps_total": len(self._steps),
            "replans_used": self.replans_used,
            "replans_remaining": self.replans_remaining,
            "execution_mode": self._execution_mode,
            "pending_execution_id": self._pending_execution_id,
            "pending_verification_execution_id": self._pending_execution_id,
            "active_step_attempts": attempts,
            "active_step_max_attempts": active.max_attempts if active else None,
            "environment_success": self.environment_success,
            "last_verdict": (
                asdict(self._last_verdict) if self._last_verdict is not None else None
            ),
            **self.control_planning_context(include_capabilities=False),
        }

    def register_plan(
        self,
        goal: str,
        steps: list[dict[str, Any]],
        *,
        stage_id: str | None = None,
        stage_outcome_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Validate and activate an already-generated initial plan."""
        if self._steps:
            return {
                "error": "plan already exists; use replan after verification failure",
                "harness": self.snapshot(),
            }
        try:
            parsed = self.parse_steps(steps)
            _validate_stage_steps(parsed, stage_id, stage_outcome_ids)
        except ValueError as exc:
            return {"error": str(exc), "harness": self.snapshot()}
        self._steps = parsed
        self._active_index = 0
        self._awaiting_verification = False
        self._needs_replan = False
        self._pending_execution_id = None
        self._set_execution_mode("act", reason="plan_registered")
        self._step_attempts = {}
        self._record(
            "plan_registered",
            goal=goal,
            steps=[asdict(step) for step in self._steps],
        )
        return {"ok": True, "goal": goal, "harness": self.snapshot()}

    def create_plan(self, goal: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Compatibility alias for callers that used the old state API."""
        return self.register_plan(goal, steps)

    def view_plan(self) -> dict[str, Any]:
        return {
            "harness": self.snapshot(),
            "steps": [asdict(step) for step in self._steps],
        }

    def verify_step(
        self,
        verdict: str,
        evidence: list[str],
        summary: str,
        *,
        execution_id: str | None = None,
        failure_type: str | None = None,
        recommended_recovery: str | None = None,
        metrics: dict[str, float] | None = None,
        progress_evidence: dict[str, Any] | None = None,
        confidence: float | None = None,
        source: str = "agent",
        require_execution_id: bool = False,
    ) -> dict[str, Any]:
        """Compatibility wrapper that builds and applies a structured verdict."""
        try:
            structured = StepVerdict(
                verdict=verdict,
                evidence=tuple(evidence),
                summary=summary,
                failure_type=failure_type,  # type: ignore[arg-type]
                recommended_recovery=recommended_recovery,
                metrics=dict(metrics or {}),
                progress_evidence=progress_evidence,
                execution_id=execution_id,
                step_id=self.active_step.id if self.active_step else None,
                confidence=confidence,
                source=source,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            return {"error": str(exc), "harness": self.snapshot()}
        return self.apply_verdict(
            structured,
            require_execution_id=require_execution_id,
        )

    def verify_observation(
        self,
        verdict: str,
        evidence: list[str],
        summary: str,
        *,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Apply evidence to an observation-only step without a physical call."""
        active = self.active_step
        if active is None or active.step_type != "observation":
            return {
                "error": "active step is not observation-only",
                "harness": self.snapshot(),
            }
        if self._needs_replan or self._pending_execution_id is not None:
            return {
                "error": "observation cannot be verified in the current state",
                "harness": self.snapshot(),
            }
        if verdict not in {"pass", "fail", "uncertain"}:
            return {
                "error": "observation verdict must be pass, fail, or uncertain",
                "harness": self.snapshot(),
            }
        self._awaiting_verification = True
        return self.verify_step(
            verdict,
            evidence,
            summary,
            confidence=confidence,
            source="agent",
        )

    def apply_verdict(
        self,
        verdict: StepVerdict,
        *,
        require_execution_id: bool = False,
    ) -> dict[str, Any]:
        """Apply one verdict bound to the latest physical execution."""
        if not self._awaiting_verification:
            return {
                "error": "no physical action is awaiting verification",
                "harness": self.snapshot(),
            }
        if require_execution_id and not verdict.execution_id:
            return {
                "error": "verification requires execution_id",
                "harness": self.snapshot(),
            }
        if (
            verdict.execution_id is not None
            and verdict.execution_id != self._pending_execution_id
        ):
            return {
                "error": (
                    "verification execution_id does not match the pending "
                    f"action: {verdict.execution_id!r}"
                ),
                "harness": self.snapshot(),
            }
        verified_step = self.active_step
        conflict = self._ledger(verified_step).get("task_conflict") if (
            verified_step and verified_step.step_type == "action"
        ) else None
        if verdict.verdict == "pass" and (conflict or verdict.failure_type in TASK_CONFLICT_TYPES):
            return {"error": "resolve the confirmed task conflict before accepting this step", "harness": self.snapshot()}
        if verdict.verdict == "progress" and verdict.failure_type in TASK_CONFLICT_TYPES:
            return {"error": "a confirmed task conflict cannot be a progress verdict", "harness": self.snapshot()}
        step_id = verified_step.id if verified_step is not None else None
        verified_execution = self._pending_execution_id
        self._last_verdict = verdict
        escalated = False
        if verdict.verdict == "uncertain":
            # Uncertain evidence is not authoritative. Keep the exact same
            # execution pending so it can be checked again; do not authorize
            # another physical action or enter recovery.
            self._awaiting_verification = True
            self._set_execution_mode("verify", reason="verification_uncertain")
        elif verdict.verdict == "pass":
            self._awaiting_verification = False
            self._pending_execution_id = None
            # Environment termination is evidence about this action, not
            # authority to skip later observation/audit steps or outcomes.
            self._active_index += 1
            if (
                self.environment_horizon_only
                and self.completed
                and not self.environment_success
                and not (
                    self.last_feedback is not None
                    and (
                        self.last_feedback.environment_terminated
                        or self.last_feedback.environment_truncated
                    )
                )
            ):
                self._needs_replan = True
            self._set_execution_mode(
                "plan" if self._needs_replan else "completed" if self.completed else "act",
                reason="step_passed",
            )
        elif verdict.verdict == "progress":
            self._awaiting_verification = False
            self._pending_execution_id = None
            active = self.active_step
            attempts = self._step_attempts.get(step_id or "", 0)
            if active is not None and attempts < active.max_attempts:
                self._set_execution_mode("act", reason="step_progress")
            else:
                self._needs_replan = True
                self._set_execution_mode(
                    "plan",
                    reason="progress_budget_exhausted",
                )
                escalated = True
        elif verdict.verdict == "retry":
            self._awaiting_verification = False
            self._pending_execution_id = None
            active = self.active_step
            attempts = self._step_attempts.get(step_id or "", 0)
            if active is not None and attempts < active.max_attempts:
                self._set_execution_mode("recover", reason="bounded_retry")
            else:
                self._needs_replan = True
                self._set_execution_mode("plan", reason="retry_budget_exhausted")
                escalated = True
        else:
            self._awaiting_verification = False
            self._pending_execution_id = None
            self._needs_replan = True
            self._set_execution_mode(
                "plan",
                reason=f"{verdict.verdict}_verification",
            )
        accounting: dict[str, Any] = {}
        if (
            verified_step is not None and verified_step.step_type == "action"
            and verdict.verdict != "uncertain"
        ):
            accounting = self._account_verdict(verified_step, verdict, verified_execution)
        self._record(
            "step_verified",
            step=step_id,
            control_mode=verified_step.control_mode if verified_step else None,
            operation_key=list(operation_key(verified_step)) if verified_step else None,
            subtask=subtask_name(verified_step) if verified_step else None,
            **accounting,
            verdict=verdict.verdict,
            evidence=list(verdict.evidence),
            summary=verdict.summary,
            execution_id=verdict.execution_id,
            failure_type=verdict.failure_type,
            recommended_recovery=verdict.recommended_recovery,
            metrics=verdict.metrics,
            progress_evidence=verdict.progress_evidence,
            confidence=verdict.confidence,
            source=verdict.source,
            retry_escalated=escalated,
        )
        return {
            "ok": True,
            "verdict": asdict(verdict),
            "retry_escalated": escalated,
            "harness": self.snapshot(),
        }

    # ------------------------------------------------------------------
    # Control-mode accounting (1.0.11 A4, A5, B1-B5)
    # ------------------------------------------------------------------

    def _ledger(self, step: PlanStep) -> dict[str, Any]:
        key = subtask_key(step)
        ledger = self._subtasks.get(key)
        if ledger is None:
            ledger = self._subtasks[key] = {
                "policy_steps": 0, "llm_steps": 0, "since_progress": 0,
                "best_distance": None, "failures": {}, "passed": False,
                "segments": 0, "fallbacks": 0, "segment": None, "fallback": None, "escalation": None,
                "transition_seen": False,
            }
        return ledger

    def _fallback_blocks(self, step: PlanStep) -> str | None:
        return _constraint_error(step, self.control_constraints())

    def _account_execution(self, feedback: ActionFeedback) -> None:
        """Count one physical call against the active subtask before its verdict."""
        step = self.active_step
        if step is None or step.step_type != "action":
            self._pending_accounting = None
            return
        executed = feedback.raw_result.get("executed_actions")
        executed = executed if isinstance(executed, int) and not isinstance(executed, bool) and executed > 0 else 0
        grasp = feedback.raw_result.get("grasp_status")
        holding = bool(
            isinstance(grasp, dict) and grasp.get("retention_monitor_active") is True
            and not feedback.raw_result.get("grasp_events")
        )
        ledger = self._ledger(step)
        if step.control_mode in POLICY_DRIVEN_MODES:
            ledger["policy_steps"] += executed
            if ledger["segment"] is not None:
                # A hand-back is not a fresh start: the next LLM segment of the
                # subtask continues this quota (review of 1.0.11).
                ledger["carry_steps"] = ledger["segment"]["steps_used"]
                ledger["carry_calls"] = ledger["segment"]["calls_used"]
                self._close_segment(step, ledger, reason="policy_mode")
        elif step.control_mode in LLM_DRIVEN_MODES:
            segment = ledger["segment"] or self._open_segment(step, ledger, feedback, executed)
            segment["steps_used"] += executed
            segment["calls_used"] += 1
            ledger["llm_steps"] += executed
            if segment["exhausted"] is None:
                if segment["steps_used"] >= segment["step_quota"]:
                    segment["exhausted"] = "step_quota"
                elif segment["calls_used"] >= segment["call_quota"]:
                    segment["exhausted"] = "call_quota"
        environment = feedback.raw_result.get("environment")
        if isinstance(environment, dict) and isinstance(environment.get("step_id"), int):
            self._last_step_id = environment["step_id"]
        self._pending_accounting = {
            "execution_id": feedback.execution_id, "executed": executed, "holding": holding,
            "finishing": finishing_motion(feedback),
            "target_review_pause": feedback.raw_result.get("stop_reason") in {
                "task_conflict_review_required", "prerelease_review_required",
            },
            # Rejections, validation errors and guard pauses move nothing (A4).
            "physical": executed > 0 and feedback.execution_status == "completed" and not feedback.error,
        }

    def _open_segment(self, step: PlanStep, ledger: dict[str, Any], feedback: ActionFeedback,
                      executed: int) -> dict[str, Any]:
        step_quota, basis = self._segment_step_quota(step, feedback, executed)
        ledger["segments"] += 1
        ledger["transition_seen"] |= step.control_mode == "transition"
        carried = ledger.get("carry_steps", 0), ledger.get("carry_calls", 0)
        ledger["carry_steps"] = ledger["carry_calls"] = 0
        segment = {
            "mode": step.control_mode, "step": step.id,
            "steps_used": carried[0], "calls_used": carried[1],
            "step_quota": step_quota,
            "call_quota": max(SEGMENT_CALL_QUOTA, math.ceil(step_quota / SEGMENT_STEPS_PER_CALL)),
            "since_progress": 0,
            "no_progress_limit": SEGMENT_NO_PROGRESS_STEPS, "exhausted": None, "quota_basis": basis,
        }
        ledger["segment"] = segment
        self._record("control_segment_opened", operation_key=list(operation_key(step)),
                     subtask=subtask_name(step), segment_index=ledger["segments"],
                     llm_fallbacks=ledger["fallbacks"], max_llm_fallbacks=MAX_LLM_FALLBACKS,
                     **_segment_view(ledger))
        return segment

    def _segment_step_quota(self, step: PlanStep, feedback: ActionFeedback,
                            executed: int) -> tuple[int, dict[str, Any]]:
        typical = self._operation_typical_steps(step)
        if step.control_mode == "transition":
            travel = TRANSITION_DEFAULT_TRAVEL_M if step.travel_m is None else min(step.travel_m, TRANSITION_MAX_TRAVEL_M)
            nominal = math.ceil(TRANSITION_BASE_STEPS + TRANSITION_STEPS_PER_M * travel)
            basis: dict[str, Any] = {"kind": "transition", "travel_m": travel,
                                     "travel_declared": step.travel_m is not None}
        else:
            own = capability_entry(self.capabilities, step.skill_row)
            own_typical = own["typical_steps"] if own else None
            nominal = math.ceil(TAKEOVER_TYPICAL_FACTOR * own_typical) if own_typical else TAKEOVER_NONE_STEPS
            basis = {"kind": "takeover", "typical_steps": own_typical}
        reserve = math.ceil(typical) if typical else FALLBACK_RESERVE_STEPS
        quota = nominal
        environment = feedback.raw_result.get("environment")
        step_id = environment.get("step_id") if isinstance(environment, dict) else None
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            # A refused first call carries no step id; use the last known one.
            step_id, executed = self._last_step_id, 0
        if self._episode_horizon is not None and isinstance(step_id, int) and not isinstance(step_id, bool):
            remaining = self._episode_horizon - (step_id - executed)
            quota = min(quota, remaining - reserve)
            basis.update(horizon_remaining=remaining, fallback_reserve=reserve)
        basis["nominal"] = nominal
        return max(MIN_SEGMENT_STEPS, quota), basis

    def _operation_typical_steps(self, step: PlanStep) -> float | None:
        """Typical steps of the row the operation's policy steps use (fallback reserve)."""
        rows = [step.skill_row] + [
            other.skill_row for other in self._steps
            if operation_key(other) == operation_key(step) and other.control_mode in POLICY_DRIVEN_MODES
        ]
        for row in rows:
            entry = capability_entry(self.capabilities, row) if row != "none" else None
            if entry and entry["typical_steps"]:
                return entry["typical_steps"]
        return None

    def _close_segment(self, step: PlanStep, ledger: dict[str, Any], *, reason: str) -> None:
        view = _segment_view(ledger)
        ledger["segment"] = None
        self._record("control_segment_closed", operation_key=list(operation_key(step)),
                     subtask=subtask_name(step), reason=reason, **view)

    def _account_verdict(self, step: PlanStep, verdict: StepVerdict,
                         execution_id: str | None) -> dict[str, Any]:
        """Apply one final verdict to its subtask: progress, failures, quotas, escalation."""
        pending = self._pending_accounting
        if pending is None or pending["execution_id"] != execution_id:
            pending = {"executed": 0, "holding": False, "physical": False}
        self._pending_accounting = None
        ledger = self._ledger(step)
        if verdict.failure_type in TASK_CONFLICT_TYPES:
            ledger["task_conflict"] = {
                "phase": "grasp" if verdict.failure_type == "wrong_target" else "release",
                "conflict_type": verdict.failure_type, "expected_target": None,
                "observed_target": None, "evidence": verdict.summary,
            }
        progress = self._progress_signal(ledger, verdict, pending, step=step)
        finishing_phase = self._review_finishing_progress(step, ledger, verdict, pending)
        executed = pending["executed"]
        segment = ledger["segment"]
        if progress:
            ledger["since_progress"] = 0
        else:
            ledger["since_progress"] += executed
        if segment is not None:
            segment["since_progress"] = 0 if progress else segment["since_progress"] + executed
        # Untyped/non-observable effects are semantic review evidence, not motor failure.
        physical_failure = (
            verdict.verdict in {"fail", "retry"} and pending["physical"]
            and (verdict.failure_type in POLICY_PHYSICAL_FAILURE_TYPES
                 if step.control_mode in POLICY_DRIVEN_MODES else verdict.failure_type != "timeout")
        )
        if physical_failure:
            ledger["failures"][step.control_mode] = ledger["failures"].get(step.control_mode, 0) + 1
        if verdict.verdict == "pass":
            ledger["passed"] = True
            ledger["escalation"] = None
            if segment is not None:
                self._close_segment(step, ledger, reason="passed")
        elif step.control_mode in LLM_DRIVEN_MODES and segment is not None:
            reason = segment["exhausted"]
            if verdict.verdict in {"fail", "retry"} and verdict.failure_type in DESTRUCTIVE_FAILURE_TYPES:
                reason = "destructive_failure"
            elif segment.get("finishing") and executed > 0 and finishing_phase is None:
                reason = "finishing_progress_lost"
            elif reason is None and segment["since_progress"] >= segment["no_progress_limit"]:
                reason = "no_progress"
            if reason in {"step_quota", "call_quota"} and self._try_finishing_lease(step, ledger, finishing_phase):
                reason = None
            if reason is not None:
                self._fall_back(step, ledger, reason=reason)
        elif step.control_mode in POLICY_DRIVEN_MODES:
            self._review_policy_subtask(step, ledger, physical_failure=physical_failure,
                                        failure_type=(ledger.get("task_conflict") or {}).get("conflict_type")
                                        or verdict.failure_type)
            if ledger.get("task_conflict"):
                self._needs_replan = True
                self._set_execution_mode("plan", reason="task_conflict")
        return {"physical_failure": physical_failure, "progress_signal": progress}

    def _progress_signal(self, ledger: dict[str, Any], verdict: StepVerdict,
                         pending: dict[str, Any], *, step: PlanStep | None = None) -> bool:
        """Public progress: pass/progress verdict, closer measured target, maintained hold."""
        if verdict.failure_type in TASK_CONFLICT_TYPES or ledger.get("task_conflict"):
            return False
        if pending.get("target_review_pause") and pending.get("executed", 0) == 0:
            return False
        step = step or self.active_step
        if self._current_goal_policy_scope(step) and step.control_mode in POLICY_DRIVEN_MODES:
            signal, boundary = review_policy_progress(
                ledger, verdict.progress_evidence, observation_step=self._last_step_id,
                physical=pending.get("physical", False) and verdict.verdict in {"pass", "progress"}
                and verdict.failure_type is None, distance_threshold=PROGRESS_DISTANCE_M,
            )
            ledger["phase_boundary"] = boundary
            return signal
        closer = False
        distance = verdict.metrics.get("target_distance_m")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool) and math.isfinite(distance):
            best = ledger["best_distance"]
            closer = best is not None and best - distance >= PROGRESS_DISTANCE_M
            ledger["best_distance"] = distance if best is None else min(best, distance)
        if verdict.verdict in {"pass", "progress"}:
            return True
        if verdict.failure_type in DESTRUCTIVE_FAILURE_TYPES:
            return False
        return closer or pending["holding"]

    def _review_finishing_progress(self, step, ledger, verdict, pending):
        if step.control_mode != "takeover" or not pending.get("executed"):
            return None
        facts = pending.get("finishing", {})
        qualified = (pending.get("physical") and verdict.verdict == "progress"
                     and verdict.failure_type is None and not ledger.get("task_conflict")
                     and facts and not facts.get("unsafe"))
        phase = None
        if qualified:
            if ledger.get("finish_released"):
                if facts.get("moving") and facts.get("open") and not pending.get("holding"):
                    phase = "withdraw"
            elif pending.get("holding") and facts.get("stable_hold") and facts.get("moving"):
                ledger["finish_streak"] = ledger.get("finish_streak", 0) + 1
                if ledger["finish_streak"] >= FINISHING_PROGRESS_CALLS:
                    phase = "held"
            elif facts.get("released") and ledger.get("finish_streak", 0) >= FINISHING_PROGRESS_CALLS:
                ledger["finish_released"] = True
                phase = "release"
        if phase is None and not (qualified and pending.get("holding") and facts.get("stable_hold") and facts.get("moving")):
            ledger["finish_streak"] = 0
        return phase

    def _try_finishing_lease(self, step, ledger, phase):
        segment = ledger["segment"]
        if phase is None or self._episode_horizon is None or type(self._last_step_id) is not int or not 0 <= self._last_step_id <= self._episode_horizon:
            return False
        reserve = segment["quota_basis"].get("fallback_reserve", FALLBACK_RESERVE_STEPS)
        floor = min(reserve, FINISHING_FALLBACK_FLOOR_STEPS)
        key = operation_key(step)
        cap = self._finishing_limits.setdefault(key, min(FINISHING_MAX_BORROW_STEPS, max(0,reserve-floor)))
        borrowed = self._finishing_borrowed.get(key, 0)
        if phase in {"release","withdraw"} and key in self._finishing_withdrawn:
            return False
        remaining = self._episode_horizon-self._last_step_id
        grant = min(FINISHING_LEASE_STEPS, cap-borrowed, remaining-floor)
        if grant < 2:
            return False
        self._finishing_borrowed[key] = borrowed+grant
        if phase in {"release","withdraw"}:
            self._finishing_withdrawn.add(key)
        segment.update(step_quota=segment["steps_used"]+grant,
                       call_quota=segment["calls_used"]+FINISHING_LEASE_CALLS,
                       exhausted=None, finishing=True)
        segment["quota_basis"]["finishing"] = {
            "phase":phase,"borrowed_total":borrowed+grant,"borrow_cap":cap,"fallback_floor":floor}
        self._record("control_finishing_lease_granted",operation_key=list(key),subtask=subtask_name(step),
                     phase="withdraw" if phase=="release" else phase,granted_steps=grant,
                     granted_calls=FINISHING_LEASE_CALLS,borrowed_total=borrowed+grant,
                     borrow_cap=cap,horizon_remaining=remaining,fallback_floor=floor)
        return True

    def finishing_tool_violation(self, name):
        if self.finishing_action_budget() is not None and name not in FINISHING_TOOLS:
            return "finishing lease permits only bounded arm placement, release and withdrawal; replan monitor for policy fallback"
        return None

    def finishing_action_budget(self):
        """Remaining actions in an active lease; the final stop shares this cap."""
        step = self.active_step
        if step is None or step.control_mode != "takeover":
            return None
        segment = self._ledger(step)["segment"]
        if not segment or not segment.get("finishing"):
            return None
        return max(0,segment["step_quota"]-segment["steps_used"])

    def _fall_back(self, step: PlanStep, ledger: dict[str, Any], *, reason: str) -> None:
        view = _segment_view(ledger)
        ledger["finish_streak"] = 0
        ledger["finish_released"] = False
        ledger["segment"] = None
        ledger["escalation"] = None
        # The policy fallback gets its own no-progress window and failure
        # count; earlier policy failures already led to this LLM segment.
        ledger["since_progress"] = 0
        ledger["failures"]["delegate"] = ledger["failures"]["monitor"] = 0
        ledger["carry_steps"] = ledger["carry_calls"] = 0
        ledger["fallbacks"] += 1
        payload = {
            "operation_key": operation_key(step), "subtask": subtask_name(step),
            "from_mode": step.control_mode, "to_mode": "monitor", "reason": reason,
            "failed_verifications": ledger["failures"].get(step.control_mode, 0),
            "llm_segments_used": ledger["segments"], "llm_fallbacks": ledger["fallbacks"],
            "max_llm_fallbacks": MAX_LLM_FALLBACKS,
            "step": step.id, **view,
        }
        ledger["fallback"] = payload
        self._record("control_mode_fallback", **payload)
        # Even a bounded retry or progress must register a policy plan first.
        self._needs_replan = True
        self._set_execution_mode("plan", reason="control_mode_fallback")

    def _review_policy_subtask(self, step: PlanStep, ledger: dict[str, Any], *,
                               physical_failure: bool, failure_type: str | None = None) -> None:
        """Offer LLM control again for a stalled or failing policy subtask (B2, B3, B5, B7)."""
        if ledger["passed"]:
            return
        failures = ledger["failures"].get("delegate", 0) + ledger["failures"].get("monitor", 0)
        typical = self._exact_row_typical_steps(step)
        stalled = ledger["since_progress"] >= NO_PROGRESS_WINDOW_STEPS
        reason = None
        if failure_type in TASK_CONFLICT_TYPES:
            reason = "task_conflict"
        elif physical_failure and failure_type in DESTRUCTIVE_FAILURE_TYPES:
            reason = "destructive_failure"
        elif failures >= FALLBACK_FAILURES:
            reason = "failures"
        elif self._current_goal_policy_scope(step) and (
            physical_failure or ledger.get("phase_boundary") or stalled
        ):
            self._authorize_semantic_policy_review(
                step, ledger, reason="phase_boundary" if ledger.get("phase_boundary") else
                "physical_failure_review" if physical_failure else "current_goal_stall",
            )
            return
        elif physical_failure and any(
            other["fallback"] is not None for key, other in self._subtasks.items() if key[0] == operation_key(step)
        ):
            reason = "policy_failed_after_fallback"
        elif typical is not None:
            if stalled and ledger["policy_steps"] >= ESCALATION_TYPICAL_FACTOR * typical:
                reason = "steps_no_progress"
        elif stalled:
            reason = "no_progress"
        if reason is None:
            return
        self._whole_task_recovery_authorized = True
        self._semantic_policy_restricted = False
        self._clear_operation_fallbacks(step, reason=reason)
        if not self.op_takeover_enabled:
            return
        if ledger["fallbacks"] >= MAX_LLM_FALLBACKS:
            return
        if ledger["escalation"] is not None:
            return
        entry = capability_entry(self.capabilities, step.skill_row)
        payload = {
            "operation_key": operation_key(step), "subtask": subtask_name(step),
            "from_mode": step.control_mode, "reason": reason,
            "steps": ledger["policy_steps"], "steps_since_progress": ledger["since_progress"],
            "no_progress_window": NO_PROGRESS_WINDOW_STEPS,
            "typical_steps": entry["typical_steps"] if entry else None,
            "step_escalation": typical is not None, "factor": ESCALATION_TYPICAL_FACTOR,
            "failed_verifications": failures, "step": step.id,
        }
        ledger["escalation"] = payload
        self._record("control_mode_escalation_available", **payload)

    def _current_goal_policy_scope(self, step):
        # Preserve measured atomic-operation behavior inside an already decomposed plan.
        return step is not None and (self._unmeasured_composite_task or (
            self._composite_task and step.skill_row == self.task_name
        ))

    def _authorize_semantic_policy_review(self, step, ledger, *, reason):
        """Review policy goals without granting privileged motion or resetting counts."""
        self._semantic_policy_replanning_authorized = True
        self._semantic_policy_restricted = not self._whole_task_recovery_authorized
        self._semantic_scope_identity = self._semantic_scope_identity or subtask_key(step)
        self._needs_replan = True
        self._set_execution_mode("plan", reason="policy_semantic_review")
        self._record("policy_semantic_replanning_available", reason=reason,
                     operation_key=list(operation_key(step)), subtask=subtask_name(step),
                     steps=ledger["policy_steps"], steps_since_progress=ledger["since_progress"],
                     motor_takeover_allowed=not self._semantic_policy_restricted)

    def report_task_conflict(self, conflict: dict[str, Any] | None) -> None:
        """Record a fresh tool-validated semantic veto without inventing motor failures.

        Permission remains scoped to the active operation/subtask, all quotas
        remain binding, and pending verification never authorizes physical action.
        """
        step = self.active_step
        if step is None or step.step_type != "action":
            raise ValueError("task review requires an active physical plan step")
        ledger = self._ledger(step)
        ledger["task_conflict"] = conflict
        if conflict is None:
            if (ledger.get("escalation") or {}).get("reason") == "task_conflict":
                ledger["escalation"] = None
            return
        self._record("task_conflict_reported", operation_key=list(operation_key(step)),
                     subtask=subtask_name(step), **conflict)
        if step.control_mode in POLICY_DRIVEN_MODES:
            self._review_policy_subtask(step, ledger, physical_failure=False,
                                        failure_type=conflict["conflict_type"])
            if not self.awaiting_verification:
                self._needs_replan = True
                self._set_execution_mode("plan", reason="task_conflict")

    def _clear_operation_fallbacks(self, step: PlanStep, *, reason: str) -> None:
        """A failing or stalled policy re-opens LLM control of its operation (B5, B7)."""
        if not (self.op_takeover_enabled or self.transition_takeover_enabled):
            return
        for key, ledger in sorted(self._subtasks.items()):
            if key[0] != operation_key(step) or ledger["fallback"] is None:
                continue
            if ledger["fallbacks"] >= MAX_LLM_FALLBACKS:
                continue
            cleared = ledger["fallback"]
            ledger["fallback"] = None
            self._record("control_mode_fallback_cleared", operation_key=list(key[0]), subtask=key[1],
                         reason=reason, from_mode=cleared["from_mode"], trigger_step=step.id)

    def _exact_row_typical_steps(self, step: PlanStep) -> float | None:
        """Typical steps only for a whole operation on an exact measured row (B2 vs B3)."""
        if subtask_name(step) != SUBTASK_WHOLE or self.capabilities is None:
            return None
        if any(key[0] == operation_key(step) and ledger["transition_seen"] for key, ledger in self._subtasks.items()):
            return None  # the operation was split around a carry (or folded it back)
        row = self.capabilities["rows"].get(step.skill_row)
        if row is None or row.get("family") != "atomic":
            return None
        return row.get("typical_steps")

    def abandon_pending_action(
        self,
        *,
        reason: str,
        evidence: list[str],
        failure_type: str | None = None,
    ) -> dict[str, Any]:
        """Abandon one unverifiable execution so recovery may be selected.

        This is the internal state mutation used after a request-exception
        lifecycle event. It deliberately does not select a recovery scope or
        consume a recovery/replan budget.
        """
        if not self._awaiting_verification or self._pending_execution_id is None:
            return {
                "error": "no physical action is awaiting verification",
                "harness": self.snapshot(),
            }
        return self._request_recovery(
            reason=reason,
            evidence=evidence,
            failure_type=failure_type,
            event="pending_action_abandoned",
        )

    def request_recovery(
        self,
        *,
        reason: str,
        evidence: list[str],
        failure_type: str | None = None,
    ) -> dict[str, Any]:
        """Mark internal recovery need without choosing scope or consuming budget."""
        return self._request_recovery(
            reason=reason,
            evidence=evidence,
            failure_type=failure_type,
            event="recovery_requested",
        )

    def _request_recovery(
        self,
        *,
        reason: str,
        evidence: list[str],
        failure_type: str | None,
        event: str,
    ) -> dict[str, Any]:
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            return {"error": "reason must be non-empty", "harness": self.snapshot()}
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(item, str) and item.strip() for item in evidence)
        ):
            return {
                "error": "evidence must be a non-empty array of strings",
                "harness": self.snapshot(),
            }
        if not self.has_plan:
            return {"error": "no registered plan", "harness": self.snapshot()}
        if self._needs_replan:
            return {
                "error": "recovery is already required",
                "harness": self.snapshot(),
            }
        if self._execution_mode == "recover":
            return {
                "error": "recovery is already in progress",
                "harness": self.snapshot(),
            }

        active = self.active_step
        execution_id = self._pending_execution_id
        self._awaiting_verification = False
        self._pending_execution_id = None
        self._needs_replan = True
        self._set_execution_mode("plan", reason="recovery_requested")
        self._record(
            event,
            step=active.id if active is not None else None,
            execution_id=execution_id,
            reason=normalized_reason,
            evidence=[item.strip() for item in evidence],
            failure_type=failure_type,
        )
        return {"ok": True, "harness": self.snapshot()}

    def resume_recovery_retry(self, *, reason: str) -> dict[str, Any]:
        """Resume the active step without replacing it or spending a budget."""
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            return {"error": "reason must be non-empty", "harness": self.snapshot()}
        if not self._needs_replan:
            return {
                "error": "step retry is only allowed while recovery is required",
                "harness": self.snapshot(),
            }
        if self._awaiting_verification or self._pending_execution_id is not None:
            return {
                "error": "cannot retry while awaiting verification",
                "harness": self.snapshot(),
            }
        active = self.active_step
        if active is None:
            return {
                "error": "no active step is available to retry",
                "harness": self.snapshot(),
            }
        blocked = self._fallback_blocks(active)
        if blocked:
            return {"error": f"{blocked}; register a replacement plan", "harness": self.snapshot()}
        attempts = self._step_attempts.get(active.id, 0)
        if attempts >= active.max_attempts:
            return {
                "error": "active step attempt budget is exhausted",
                "harness": self.snapshot(),
            }
        self._needs_replan = False
        self._set_execution_mode("act", reason="recovery_retry_resumed")
        self._record(
            "recovery_retry_resumed",
            step=active.id,
            reason=normalized_reason,
            attempts=attempts,
            max_attempts=active.max_attempts,
        )
        return {"ok": True, "harness": self.snapshot()}

    def replan(
        self,
        reason: str,
        steps: list[dict[str, Any]],
        goal: str | None = None,
        *,
        stage_id: str | None = None,
        stage_outcome_ids: tuple[str, ...] = (),
        enforce_global_budget: bool = True,
    ) -> dict[str, Any]:
        if not self._steps:
            return {
                "error": "cannot replan before register_plan",
                "harness": self.snapshot(),
            }
        if not self.can_replan:
            return {
                "error": (
                    "replan is only allowed after authoritative failure, "
                    "a verified retry, an abandoned action, or an exhausted step-attempt budget"
                ),
                "harness": self.snapshot(),
            }
        if (
            enforce_global_budget
            and self.max_replans is not None
            and self._replans >= self.max_replans
        ):
            return {"error": "replan budget exhausted", "harness": self.snapshot()}
        try:
            replacement = self.parse_steps(steps)
            _validate_stage_steps(replacement, stage_id, stage_outcome_ids)
        except ValueError as exc:
            return {"error": str(exc), "harness": self.snapshot()}
        old_remaining = [asdict(step) for step in self._steps[self._active_index :]]
        self._steps = self._steps[: self._active_index] + replacement
        self._needs_replan = False
        self._replans += 1
        self._set_execution_mode("act", reason="repair_registered")
        for step in replacement:
            self._step_attempts.pop(step.id, None)
        self._record(
            "plan_replanned",
            reason=reason,
            goal=goal,
            old_remaining=old_remaining,
            replacement=[asdict(step) for step in replacement],
        )
        return {"ok": True, "harness": self.snapshot()}

    def replan_active_step(
        self,
        reason: str,
        steps: list[dict[str, Any]],
        goal: str | None = None,
        *,
        stage_id: str,
        outcome_id: str,
    ) -> dict[str, Any]:
        """Replace only the active outcome step and preserve its stage tail."""
        if not self._steps or not self._needs_replan:
            return {
                "error": "active-step replan requires registered recovery state",
                "harness": self.snapshot(),
            }
        try:
            replacement = self.parse_steps(steps)
            _validate_stage_steps(replacement, stage_id, (outcome_id,))
        except ValueError as exc:
            return {"error": str(exc), "harness": self.snapshot()}
        active = self.active_step
        replace_active = bool(
            active is not None and active.covers_outcomes == (outcome_id,)
        )
        old_step = asdict(active) if replace_active and active is not None else None
        tail_index = self._active_index + 1 if replace_active else self._active_index
        self._steps = (
            self._steps[: self._active_index] + replacement + self._steps[tail_index:]
        )
        self._needs_replan = False
        self._replans += 1
        self._set_execution_mode("act", reason="active_outcome_repaired")
        for step in replacement:
            self._step_attempts.pop(step.id, None)
        self._record(
            "active_step_replanned",
            reason=reason,
            goal=goal,
            old_step=old_step,
            inserted_before_active=not replace_active,
            replacement=[asdict(step) for step in replacement],
        )
        return {"ok": True, "harness": self.snapshot()}

    def insert_recovery_plan(
        self,
        reason: str,
        steps: list[dict[str, Any]],
        goal: str | None = None,
        *,
        stage_id: str,
        outcome_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Insert recovery work before the current stage tail without rewinding it."""
        if not self._steps or not self._needs_replan:
            return {
                "error": "recovery insertion requires registered recovery state",
                "harness": self.snapshot(),
            }
        try:
            replacement = self.parse_steps(steps)
            _validate_stage_steps(replacement, stage_id, outcome_ids)
        except ValueError as exc:
            return {"error": str(exc), "harness": self.snapshot()}
        self._steps = (
            self._steps[: self._active_index]
            + replacement
            + self._steps[self._active_index :]
        )
        self._needs_replan = False
        self._replans += 1
        self._set_execution_mode("act", reason="earlier_outcome_recovery_inserted")
        for step in replacement:
            self._step_attempts.pop(step.id, None)
        self._record(
            "recovery_plan_inserted",
            reason=reason,
            goal=goal,
            stage_id=stage_id,
            outcome_ids=list(outcome_ids),
            replacement=[asdict(step) for step in replacement],
        )
        return {"ok": True, "harness": self.snapshot()}

    def register_next_stage(
        self,
        goal: str,
        steps: list[dict[str, Any]],
        *,
        stage_id: str,
        stage_outcome_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Atomically replace a completed stage with validated next-stage steps."""
        if not self.completed:
            return {
                "error": "current stage plan is not complete",
                "harness": self.snapshot(),
            }
        try:
            parsed = self.parse_steps(steps)
            _validate_stage_steps(parsed, stage_id, stage_outcome_ids)
        except ValueError as exc:
            return {"error": str(exc), "harness": self.snapshot()}

        self._steps = parsed
        self._active_index = 0
        self._awaiting_verification = False
        self._needs_replan = False
        self._pending_execution_id = None
        self._last_verdict = None
        self._step_attempts = {}
        self._set_execution_mode("act", reason="next_stage_registered")
        self._record(
            "next_stage_registered",
            goal=goal,
            stage_id=stage_id,
            steps=[asdict(step) for step in parsed],
        )
        return {"ok": True, "harness": self.snapshot()}

    def record_tool_result(
        self,
        name: str,
        input_dict: dict[str, Any],
        result: dict[str, Any],
        *,
        execution_id: str | None = None,
    ) -> None:
        """Compatibility path for callers without normalized ActionFeedback."""
        from rpent.planning.feedback import action_feedback_from_result

        feedback = action_feedback_from_result(
            execution_id=execution_id or uuid4().hex,
            tool_name=name,
            physical=name in self.physical_tools,
            result=result,
        )
        self.record_action_feedback(feedback, input_dict=input_dict)

    def record_action_feedback(
        self,
        feedback: ActionFeedback,
        *,
        input_dict: dict[str, Any],
    ) -> None:
        """Track one normalized tool outcome and require physical verification."""
        name = feedback.tool_name
        is_physical = name in self.physical_tools
        self._record(
            "tool_result",
            tool=name,
            physical=is_physical,
            args=input_dict,
            execution_id=feedback.execution_id,
            execution_status=feedback.execution_status,
            error=feedback.error,
            failure_type=feedback.failure_type,
        )
        if is_physical:
            self._account_execution(feedback)
            self._last_feedback = feedback
            if feedback.execution_status == "completed" and feedback.environment_success:
                self._environment_success_seen = True
            self._awaiting_verification = True
            self._pending_execution_id = feedback.execution_id
            self._set_execution_mode("verify", reason="physical_action_completed")
            step_id = self.active_step.id if self.active_step is not None else None
            attempt_consumed = consumes_physical_attempt(
                tool_name=name, execution_status=feedback.execution_status,
                result=feedback.raw_result,
            )
            if step_id is not None and attempt_consumed:
                self._step_attempts[step_id] = self._step_attempts.get(step_id, 0) + 1
            self._record(
                "verification_required",
                tool=name,
                step=step_id,
                execution_id=feedback.execution_id,
                attempt=self._step_attempts.get(step_id or "", 0),
                attempt_consumed=attempt_consumed,
            )


def plan_input_schema(
    *,
    require_goal: bool,
    replan: bool = False,
) -> dict[str, Any]:
    """Return the JSON schema used by legacy Harness control tools."""
    properties: dict[str, Any] = {
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "intent": {"type": "string"},
                    **control_step_schema(),
                    "allowed_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "preconditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "on_failure": {"type": "string"},
                    "max_attempts": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_STEP_ATTEMPTS,
                    },
                },
                "required": ["id", "intent", "allowed_tools"],
            },
        },
    }
    if require_goal:
        properties = {"goal": {"type": "string"}, **properties}
    if replan:
        properties = {
            "reason": {"type": "string"},
            "goal": {"type": "string"},
            **properties,
        }
    required = ["steps"]
    if require_goal:
        required.insert(0, "goal")
    if replan:
        required.insert(0, "reason")
    return {"type": "object", "properties": properties, "required": required}


def parse_steps(
    raw_steps: list[dict[str, Any]],
    physical_tools: set[str],
    *,
    policy_family: str | None = None,
    task_name: str | None = None,
    transition_takeover_enabled: bool = True,
    op_takeover_enabled: bool = True,
    control_constraints: Mapping[str, Any] | None = None,
    required_monitor_operations: set[SubtaskKey] | None = None,
    escalated_operations: set[SubtaskKey] | None = None,
) -> list[PlanStep]:
    """Validate untrusted LLM plan steps against physical tool contracts.

    ``control_constraints`` comes from ``PlanStateMachine.control_constraints``;
    the two older keyword sets are accepted as its ``fallbacks`` (from
    takeover) and ``escalated`` members.
    """
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty array")
    if len(raw_steps) > 32:
        raise ValueError("a plan may contain at most 32 steps")
    parsed: list[PlanStep] = []
    ids: set[str] = set()
    catalog = load_capabilities(policy_family) if policy_family else None
    if type(transition_takeover_enabled) is not bool or type(op_takeover_enabled) is not bool:
        raise ValueError("control-mode ablation switches must be booleans")
    constraints = {
        "fallbacks": {
            **dict.fromkeys(required_monitor_operations or (), "takeover"),
            **dict((control_constraints or {}).get("fallbacks", {})),
        },
        "escalated": set(escalated_operations or ()) | set((control_constraints or {}).get("escalated", ())),
        "llm_exhausted": set((control_constraints or {}).get("llm_exhausted", ())),
    }
    protect_whole_task = (control_constraints or {}).get(
        "whole_task_policy_first", unmeasured_composite(catalog, task_name),
    )
    semantic_only = (control_constraints or {}).get("semantic_policy_only", False)
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"step {index} must be an object")
        step_id = str(raw.get("id", "")).strip()
        intent = str(raw.get("intent", "")).strip()
        step_type = str(raw.get("step_type", "action")).strip()
        allowed_raw = raw.get("allowed_tools", [])
        if not step_id or not intent:
            raise ValueError(f"step {index} needs non-empty id and intent")
        if step_id in ids:
            raise ValueError(f"duplicate plan step id: {step_id!r}")
        if step_type not in {"action", "observation"}:
            raise ValueError(f"step {step_id!r} has invalid step_type {step_type!r}")
        if not isinstance(allowed_raw, list):
            raise ValueError(f"step {step_id!r} allowed_tools must be a list")
        if step_type == "action" and not allowed_raw:
            raise ValueError(f"step {step_id!r} needs a non-empty allowed_tools list")
        if step_type == "observation" and allowed_raw:
            raise ValueError(f"observation step {step_id!r} cannot allow tools")
        allowed = tuple(str(name) for name in allowed_raw)
        invalid = sorted(set(allowed) - physical_tools - READ_ONLY_PERCEPTION_TOOLS)
        if invalid:
            raise ValueError(
                f"step {step_id!r} contains non-physical/unknown tools: {invalid}"
            )
        ids.add(step_id)
        control = _parse_control_fields(
            raw, catalog, task_name=task_name,
            transition_takeover_enabled=transition_takeover_enabled,
            op_takeover_enabled=op_takeover_enabled,
        )
        escalation_required = control.pop("escalation_required")
        requested_attempts = _positive_attempts(
            raw.get("max_attempts", 2),
            step_id,
        )
        # max_attempts is the total physical-call budget while this step is
        # active, not merely the number of retries. A step that explicitly
        # permits a multi-tool chain (for example pi0_pick -> set_gripper)
        # must not make its own expected sequence impossible to execute.
        minimum_tool_calls = len(set(allowed) & physical_tools)
        if "pi0_pick" in allowed:
            # A grasp step needs room for one safe miss/re-observation cycle.
            # For pi0_pick -> set_gripper this yields three calls: retry the
            # grasp once if needed, then lock the successful hold.
            minimum_tool_calls += 1
        parsed.append(
            PlanStep(
                id=step_id,
                intent=intent,
                allowed_tools=allowed,
                step_type=step_type,  # type: ignore[arg-type]
                covers_outcomes=_string_tuple(
                    raw.get("covers_outcomes", []),
                    "covers_outcomes",
                    step_id,
                ),
                stage_id=(
                    str(raw["stage_id"]).strip() if raw.get("stage_id") else None
                ),
                preconditions=_string_tuple(
                    raw.get("preconditions", []),
                    "preconditions",
                    step_id,
                ),
                expected_evidence=_string_tuple(
                    raw.get("expected_evidence", []),
                    "expected_evidence",
                    step_id,
                ),
                on_failure=str(raw.get("on_failure", "replan")),
                max_attempts=max(requested_attempts, minimum_tool_calls),
                **control,
                **_parse_identity_fields(raw, step_id, control["control_mode"]),
            )
        )
        if parsed[-1].step_type != "action":
            continue
        blocked = _constraint_error(parsed[-1], constraints)
        if blocked:
            raise ValueError(f"step {step_id!r}: {blocked}")
        if escalation_required and subtask_key(parsed[-1]) not in constraints["escalated"]:
            listed = [
                {"operation_key": list(key[0]), "subtask": key[1]} for key in sorted(constraints["escalated"])
            ]
            raise ValueError(
                f"step {step_id!r}: {escalation_required} unless the harness lists "
                "the operation and subtask as escalation-available; keep the escalated "
                f"operation and subtask names (escalation-available: {listed})"
            )
        # 1.0.11 H4: takeover went 0/5 in the 1.0.10 full run, always in contact.
        # Until it is validated, an escalated takeover positions with the
        # privileged tools and hands the contact to the policy (staged).
        if escalation_required and not set(parsed[-1].allowed_tools) & VLA_TOOLS:
            raise ValueError(
                f"step {step_id!r}: an escalated takeover is staged: list the "
                "operation's policy tool in allowed_tools and hand the contact to "
                "the policy with its frozen sentence"
            )
    if semantic_only:
        identity = (control_constraints or {}).get("semantic_scope_identity")
        for action in (step for step in parsed if step.step_type == "action"):
            if (action.control_mode != "monitor"
                    or not set(action.allowed_tools) <= {"continue_vla", "execute_vla", *READ_ONLY_PERCEPTION_TOOLS}
                    or not set(action.allowed_tools) & {"continue_vla", "execute_vla"}):
                raise ValueError("semantic policy review permits monitor VLA steps only, not motor takeover")
            if identity is not None and subtask_key(action) != identity:
                raise ValueError("semantic policy review preserves operation/subtask identity and counters")
    if protect_whole_task:
        actions = [step for step in parsed if step.step_type == "action"]
        if len(actions) != 1 or any(
            step.control_mode != "monitor" or step.skill_row != "none" or step.tier != "none"
            or subtask_name(step) != SUBTASK_WHOLE
            or not set(step.allowed_tools) <= {"continue_vla", "execute_vla", *READ_ONLY_PERCEPTION_TOOLS}
            or not set(step.allowed_tools) & {"continue_vla", "execute_vla"}
            for step in actions
        ):
            raise ValueError(
                "unmeasured composite requires one whole-task monitor operation "
                "using continue_vla/execute_vla until verified physical failures authorize recovery"
            )
        original = (control_constraints or {}).get("original_policy_instruction")
        if original is not None and _identity_text(original) not in _identity_text(actions[0].intent):
            raise ValueError("whole-task monitor must quote the complete original instruction")
        original_operation = (control_constraints or {}).get("whole_task_operation")
        if original_operation is not None and operation_key(actions[0]) != original_operation:
            raise ValueError("whole-task operation identity must remain unchanged across replanning")
    return parsed


def _constraint_error(step: PlanStep, constraints: Mapping[str, Any]) -> str | None:
    """Return why per-subtask history forbids this step's mode, or None."""
    key = subtask_key(step)
    from_mode = constraints.get("fallbacks", {}).get(key)
    if step.control_mode in {"takeover", "transition"} and key in constraints.get("llm_exhausted", ()):
        return (f"{MAX_LLM_FALLBACKS} LLM-driven segments of this subtask already ended in "
                "fallback; keep it with the policy (monitor)")
    if from_mode == "transition" and step.control_mode == "transition":
        return ("control_mode_fallback: this carry's transition quota ended; fold the carry "
                "into the next operation under monitor")
    if from_mode is not None and from_mode != "transition" and step.control_mode not in {"monitor", "transition"}:
        return "control_mode_fallback requires monitor for this operation subtask"
    return None


def _parse_identity_fields(raw: dict[str, Any], step_id: str, control_mode: str) -> dict[str, Any]:
    """Stable operation/subtask names and a transition's travel distance (1.0.11 A5)."""
    fields: dict[str, Any] = {}
    for name in ("operation", "subtask"):
        value = raw.get(name)
        if value is None:
            fields[name] = None
            continue
        if not isinstance(value, str):
            raise ValueError(f"step {step_id!r} field {name!r} must be a string")
        value = " ".join(value.split())
        if len(value) > _MAX_IDENTITY_CHARS:
            raise ValueError(f"step {step_id!r} field {name!r} must be at most {_MAX_IDENTITY_CHARS} characters")
        fields[name] = value or None
    travel = raw.get("travel_m")
    if travel is not None:
        if isinstance(travel, bool) or not isinstance(travel, (int, float)) or not math.isfinite(travel) or travel < 0:
            raise ValueError(f"step {step_id!r} travel_m must be a finite nonnegative number")
        if travel > TRANSITION_MAX_TRAVEL_M:
            raise ValueError(f"step {step_id!r} travel_m must be at most {TRANSITION_MAX_TRAVEL_M} metres")
    # Only a transition's carry distance sizes a quota; elsewhere it is ignored.
    fields["travel_m"] = float(travel) if travel is not None and control_mode == "transition" else None
    return fields


def control_step_schema() -> dict[str, Any]:
    """Shared optional schema fields; omission retains historical-plan compatibility."""
    return {
        "skill_row": {"type": "string", "minLength": 1},
        "tier": {"type": "string", "enum": sorted(TIERS)},
        "control_mode": {"type": "string", "enum": sorted(CONTROL_MODES)},
        "operation": {
            "type": "string", "minLength": 1, "maxLength": _MAX_IDENTITY_CHARS,
            "description": (
                "Stable name of the complete operation, unchanged across replanning; "
                "the pick, carry and place steps of one pick-and-place share it."
            ),
        },
        "subtask": {
            "type": "string", "minLength": 1, "maxLength": _MAX_IDENTITY_CHARS,
            "description": (
                "Part of the operation this step covers (for example pick, carry, place, "
                "grasp, pour); omit for the whole operation. Keep it unchanged across replanning."
            ),
        },
        "travel_m": {
            "type": "number", "minimum": 0, "maximum": TRANSITION_MAX_TRAVEL_M,
            "description": "Transition only: planar base travel to the handoff pose, from back_project.",
        },
    }


def _parse_control_fields(raw: dict, catalog: dict | None, *, task_name: str | None,
                          transition_takeover_enabled: bool, op_takeover_enabled: bool) -> dict:
    fields = {"skill_row": raw.get("skill_row", "none"), "tier": raw.get("tier", "none"),
              "control_mode": raw.get("control_mode", "monitor")}
    if not isinstance(fields["skill_row"], str) or not fields["skill_row"].strip():
        raise ValueError("skill_row must be a non-empty string")
    fields["skill_row"] = fields["skill_row"].strip()
    for key, values in (("tier", TIERS), ("control_mode", CONTROL_MODES)):
        if not isinstance(fields[key], str) or fields[key] not in values:
            raise ValueError(f"invalid {key}: {fields[key]!r}")
    entry = capability_entry(catalog, fields["skill_row"])
    if catalog is None:
        fields["tier"] = "none"
    else:
        expected = entry["tier"] if entry else "none"
        if fields["tier"] != expected:
            raise ValueError(f"skill_row {fields['skill_row']!r} requires tier={expected}")
    mode = fields["control_mode"]
    # Takeover beyond the tier rule or a declared mode limit is legal only for
    # an operation the harness has escalated; parse_steps checks its key.
    escalation_required = None
    allowed_modes = entry["allowed_control_modes"] if entry else None
    # A transition executes no row: a declared entry's mode limit governs how
    # the operation itself runs, never its carry (1.0.11 replay D07/3, E1b).
    if allowed_modes is not None and mode != "transition" and mode not in allowed_modes:
        message = f"declared skill_row {fields['skill_row']!r} allows control_mode in {allowed_modes}"
        if mode != "takeover":
            raise ValueError(message)
        escalation_required = message
    if mode == "takeover":
        if not op_takeover_enabled:
            raise ValueError("op_takeover_enabled=false rejects takeover")
        if fields["tier"] not in {"weak", "none"}:
            escalation_required = escalation_required or "takeover requires weak or none tier"
    fields["escalation_required"] = escalation_required
    if mode == "transition":
        if not transition_takeover_enabled:
            raise ValueError("transition_takeover_enabled=false rejects transition")
        if not is_composite_task(task_name):
            raise ValueError("transition requires registry family=composite")
        if set(raw.get("allowed_tools", ())) & VLA_TOOLS:
            raise ValueError("transition cannot allow VLA tools")
    if set(raw.get("allowed_tools", ())) & PRIVILEGED_PHYSICAL_TOOLS and mode not in {"takeover", "transition"}:
        raise ValueError("privileged physical tools require takeover or transition")
    missing = any(key not in raw for key in ("skill_row", "tier", "control_mode"))
    # Preserve the parser's provenance through candidate serialization/re-parsing.
    fields["defaulted"] = raw.get("defaulted") is True or (raw.get("step_type", "action") == "action" and missing)
    return fields


_CONTROL_AUDIT_TEXT = re.compile(
    r"\b(?:row|tier|control_mode|operation|subtask)\s*=\s*[^;,\s]+\s*[;,]?"
)
# The frozen VLA sentence quoted in intent: double or typographic quotes, or
# single quotes that are not apostrophes inside a word.
_QUOTED_INSTRUCTION = (
    re.compile(r'"([^"]{8,})"'),
    re.compile(r"“([^”]{8,})”"),
    re.compile(r"(?<!\w)'([^']{8,}?)'(?!\w)"),
    re.compile(r"‘([^’]{8,})’"),
)


def _identity_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _frozen_instruction(intent: str) -> str | None:
    for pattern in _QUOTED_INSTRUCTION:
        match = pattern.search(intent)
        if match:
            return _identity_text(match.group(1))
    return None


def operation_key(step: PlanStep) -> tuple[str, ...]:
    """Stable identity of one complete operation across replanning (1.0.11 A5).

    The planner-named ``operation`` wins; then immutable outcome coverage;
    otherwise the matched row plus the frozen quoted instruction. Rewritten
    preconditions, evidence, step IDs and audit text never change it.
    """
    if step.operation:
        return ("operation", _identity_text(step.operation))
    if step.covers_outcomes:
        return tuple(sorted(step.covers_outcomes))
    instruction = _frozen_instruction(step.intent)
    if instruction is not None:
        return ("legacy-operation", step.stage_id or "", step.skill_row, instruction)
    # Intent repeats 'row=..; tier=..; control_mode=..' for audit; a mode change
    # must not change the identity of the same operation.
    intent = _identity_text(_CONTROL_AUDIT_TEXT.sub("", step.intent))
    return ("legacy-operation", step.stage_id or "", step.skill_row if step.skill_row != "none" else intent)


def subtask_name(step: PlanStep) -> str:
    """Planner-named part of the operation; a transition defaults to the carry."""
    if step.subtask:
        return _identity_text(step.subtask)
    return SUBTASK_CARRY if step.control_mode == "transition" else SUBTASK_WHOLE


def subtask_key(step: PlanStep) -> SubtaskKey:
    """Unit of step counting, quotas, fallback and escalation (1.0.11 B1)."""
    return (operation_key(step), subtask_name(step))


def _segment_view(ledger: dict[str, Any]) -> dict[str, Any]:
    segment = ledger.get("segment")
    if not segment:
        return {}
    return {
        "mode": segment["mode"], "steps_used": segment["steps_used"],
        "step_quota": segment["step_quota"], "calls_used": segment["calls_used"],
        "call_quota": segment["call_quota"], "steps_since_progress": segment["since_progress"],
        "no_progress_limit": segment["no_progress_limit"], "quota_exhausted": segment["exhausted"],
        "quota_basis": dict(segment["quota_basis"]),
    }


def _validate_stage_steps(
    steps: list[PlanStep],
    stage_id: str | None,
    stage_outcome_ids: tuple[str, ...],
) -> None:
    """Keep legacy plans compatible while strictly binding staged plans."""
    if stage_id is None:
        return
    expected = tuple(stage_outcome_ids)
    if (
        not expected
        or any(
            not isinstance(outcome_id, str) or not outcome_id.strip()
            for outcome_id in expected
        )
        or len(expected) != len(set(expected))
    ):
        raise ValueError("stage_outcome_ids must be non-empty and unique")
    for step in steps:
        if step.stage_id != stage_id:
            raise ValueError(f"step {step.id!r} must use active stage_id {stage_id!r}")
        if not step.covers_outcomes:
            raise ValueError(f"staged step {step.id!r} must cover an outcome")
    flattened = tuple(
        outcome_id for step in steps for outcome_id in step.covers_outcomes
    )
    if flattened != expected:
        raise ValueError(
            "staged steps must cover stage outcomes exactly once and in order: "
            f"expected {list(expected)}, got {list(flattened)}"
        )


def _string_tuple(value: Any, field: str, step_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"step {step_id!r} field {field!r} must be an array of strings"
        )
    return tuple(value)


def _positive_attempts(value: Any, step_id: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_STEP_ATTEMPTS
    ):
        raise ValueError(
            f"step {step_id!r} max_attempts must be an integer between 1 and "
            f"{MAX_STEP_ATTEMPTS}"
        )
    return value
