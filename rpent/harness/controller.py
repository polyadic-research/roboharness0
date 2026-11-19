"""Compatibility facade joining plan state, safety gate, tools, and trace."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from robots.robocasa.control_modes import IDENTITY_FIELDS
from rpent.harness.gate import PlanSafetyGate
from rpent.harness.hooks import (
    FinishGuardHook,
    HarnessHookPipeline,
    PhysicalStateHook,
    PlanGateHook,
    StructuredFeedbackVerifierHook,
    ToolHookContext,
)
from rpent.harness.lifecycle import (
    COMPLETED_STEP_REVIEW_GUIDANCE,
    COMPLETED_STEP_REVIEW_VERDICTS,
    PLANNING_INFORMATION_MAX_ROUNDS,
    POLICY_PROGRESS_GUIDANCE,
)
from rpent.harness.policy_progress import progress_evidence_schema
from rpent.harness.state import PlanStateMachine, operation_key, subtask_name
from rpent.planning.information import (
    OBJECT_STATES_SCHEMA,
    validate_information_request,
    validate_object_states,
)
from rpent.planning.schemas import ActionFeedback
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.harness.outcomes import OutcomeGraph
    from rpent.harness.recovery import RecoveryTracker
    from rpent.harness.stages import StageRuntime
    from rpent.harness.verification import VerificationCoordinator
    from rpent.planning.events import EventRecorder
    from rpent.planning.generator import PlanGenerator
    from rpent.planning.outcome_program import OutcomeProgram

CONTROL_TOOLS = {
    "create_plan",
    "register_plan",
    "view_plan",
    "verify_step",
    "replan",
    "provide_planner_information",
}


class HarnessController:
    """Expose legacy LLM tools over a reusable state machine and pure gate."""

    VALID_MODES = {"trace", "plan-gated"}

    def __init__(
        self,
        *,
        mode: str,
        trace_path: str | Path,
        max_replans: int | None = 2,
        environment_horizon_only: bool = False,
        physical_tools: set[str] | None = None,
        event_recorder: EventRecorder | None = None,
        plan_generator: PlanGenerator | None = None,
        use_chinese: bool = False,
        require_execution_id: bool = False,
        auto_verify_tool_signals: bool = False,
        require_environment_success: bool = False,
        require_object_states: bool = False,
        policy_family: str | None = None,
        task_name: str | None = None,
        transition_takeover_enabled: bool = True,
        op_takeover_enabled: bool = True,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported harness mode: {mode!r}")
        self.require_object_states = require_object_states
        self._pending_information: dict[str, Any] | None = None
        self.mode = mode
        self.trace_path = Path(trace_path)
        self.event_recorder = event_recorder
        self.plan_generator = plan_generator
        self.use_chinese = bool(use_chinese)
        self.require_execution_id = bool(require_execution_id)
        self.auto_verify_tool_signals = bool(auto_verify_tool_signals)
        self.require_environment_success = bool(require_environment_success)
        self._proposals: dict[str, dict[str, Any]] = {}
        self._active_plan: dict[str, Any] | None = None
        self._verified_facts: list[dict[str, Any]] = []
        self._completed_step_reviews: dict[str, dict[str, Any]] = {}
        self._review_observation_id: str | None = None
        self._completion_review_revision = 0
        # Strict outcome-aware domains are installed atomically by
        # ``HarnessSession.start_instruction``.  They intentionally live on the
        # controller so the session owns only the external lifecycle protocol.
        self.outcome_program: OutcomeProgram | None = None
        self.outcome_graph: OutcomeGraph | None = None
        self.stage_runtime: StageRuntime[Any] | None = None
        self.verification_coordinator: VerificationCoordinator | None = None
        self.recovery_tracker: RecoveryTracker | None = None
        self._strict_goal: str | None = None
        self._strict_strategy: str | None = None
        self._strict_session_active = False
        self.state = PlanStateMachine(
            max_replans=max_replans,
            environment_horizon_only=environment_horizon_only,
            physical_tools=physical_tools,
            record=self._record,
            policy_family=policy_family,
            task_name=task_name,
            transition_takeover_enabled=transition_takeover_enabled,
            op_takeover_enabled=op_takeover_enabled,
        )
        self._configure_generator_controls()
        self.gate = PlanSafetyGate(mode=mode)
        post_hooks: list[Any] = [PhysicalStateHook()]
        if self.auto_verify_tool_signals:
            post_hooks.append(
                StructuredFeedbackVerifierHook(use_chinese=self.use_chinese)
            )
        self.hooks = HarnessHookPipeline(
            pre_hooks=(
                PlanGateHook(gate=self.gate, control_tools=CONTROL_TOOLS),
                FinishGuardHook(
                    mode=mode,
                    require_environment_success=self.require_environment_success,
                ),
            ),
            post_hooks=tuple(post_hooks),
        )
        self._record(
            "harness_started",
            mode=mode,
            max_replans=max_replans,
            planning_loop="v3" if self.require_execution_id else "v2",
            auto_verify_tool_signals=self.auto_verify_tool_signals,
            require_execution_id=self.require_execution_id,
            require_environment_success=self.require_environment_success,
        )

    @property
    def max_replans(self) -> int | None:
        return self.state.max_replans

    @property
    def physical_tools(self) -> set[str]:
        return set(self.state.physical_tools)

    def set_physical_tools(self, tools: set[str]) -> None:
        if self._strict_session_active:
            raise ValueError("physical tools are frozen by the active strict session")
        self.state.set_physical_tools(tools)

    def configure_control_modes(self, *, policy_family: str | None, task_name: str | None) -> None:
        """Bind trusted episode metadata before any initial planning request."""
        self.state.configure_control_modes(
            policy_family=policy_family, task_name=task_name,
            transition_takeover_enabled=self.state.transition_takeover_enabled,
            op_takeover_enabled=self.state.op_takeover_enabled,
        )
        self._configure_generator_controls()

    def _configure_generator_controls(self) -> None:
        configure = getattr(self.plan_generator, "configure_control_modes", None)
        if callable(configure):
            kwargs: dict[str, Any] = {
                "policy_family": self.state.policy_family,
                "task_name": self.state.task_name,
                "transition_takeover_enabled": self.state.transition_takeover_enabled,
                "op_takeover_enabled": self.state.op_takeover_enabled,
            }
            # Candidate parsing must see the same fallbacks, escalations and
            # exhausted LLM segments as registration; older generators only
            # accept the static fields or the two 1.0.10 sets.
            parameters = list(inspect.signature(configure).parameters.values())
            names = {p.name for p in parameters}
            var_keyword = any(p.kind is p.VAR_KEYWORD for p in parameters)
            constraints = self.state.control_constraints()
            if "control_constraints" in names or var_keyword:
                kwargs["control_constraints"] = constraints
            elif "escalated_operations" in names:
                kwargs["required_monitor_operations"] = {
                    key for key, mode in constraints["fallbacks"].items() if mode != "transition"
                }
                kwargs["escalated_operations"] = constraints["escalated"]
            configure(**kwargs)

    def planning_constraints(self, constraints: list[str]) -> list[str]:
        """Supply capability data through the existing grounded-request contract."""
        # Every generation call passes through here, so refresh the dynamic sets.
        self._configure_generator_controls()
        context = self.state.control_planning_context()
        if (
            self.state.policy_family is None and not context["control_mode_failures"]
            and not context["control_mode_escalations"] and not context["control_mode_fallbacks"]
            and not context["control_segments"]
            and self.state.transition_takeover_enabled and self.state.op_takeover_enabled
        ):
            return list(constraints)
        return [*constraints, "CONTROL MODE CONTRACT: " + json.dumps(context, ensure_ascii=False, sort_keys=True)]

    def tool_definitions(self) -> list[tuple[str, dict[str, Any], Callable[..., dict]]]:
        """Return control tools used by legacy episode-owning runtimes."""
        if self._strict_session_active:
            return []
        definitions = [
            (
                "create_plan",
                {
                    "name": "create_plan",
                    "description": (
                        "Ask the planning LLM to generate a grounded candidate "
                        "plan. This does NOT activate the plan. Pass concrete "
                        "observations from image/proprioception/geometry tools, "
                        "then inspect the returned strategy and steps and call "
                        "register_plan with its proposal_id."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "observations": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                                "description": (
                                    "Concrete grounded scene facts available to "
                                    "the nested planning request."
                                ),
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional safety or task constraints.",
                            },
                        },
                        "required": ["goal", "observations"],
                    },
                },
                self.create_plan,
            ),
            (
                "register_plan",
                {
                    "name": "register_plan",
                    "description": (
                        "Validate and activate one unchanged plan proposal "
                        "returned by create_plan or replan. Physical actions "
                        "remain blocked until registration succeeds."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"proposal_id": {"type": "string"}},
                        "required": ["proposal_id"],
                    },
                },
                self.register_plan,
            ),
            (
                "view_plan",
                {
                    "name": "view_plan",
                    "description": "Read the active harness plan and execution status.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                self.view_plan,
            ),
            (
                "verify_step",
                {
                    "name": "verify_step",
                    "description": (
                        "Verify the most recent physical action using observed "
                        "image/proprioception/geometry evidence. pass advances "
                        "the plan; uncertain keeps verification pending. " + POLICY_PROGRESS_GUIDANCE
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": [
                                    "progress",
                                    "pass",
                                    "retry",
                                    "fail",
                                    "uncertain",
                                ],
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Concrete observations, not hidden-state claims."
                                ),
                            },
                            "summary": {"type": "string"},
                            "execution_id": {
                                "type": "string",
                                "description": (
                                    "Pending physical execution_id shown in the "
                                    "harness state. Required by v3 profiles."
                                ),
                            },
                            "failure_type": {
                                "type": ["string", "null"],
                                "enum": [
                                    "policy_rejected",
                                    "target_not_grounded",
                                    "wrong_target",
                                    "wrong_destination",
                                    "wrong_allocation",
                                    "grasp_missed",
                                    "recovery_exhausted",
                                    "object_dropped",
                                    "destination_not_grounded",
                                    "motion_blocked",
                                    "unsafe_trajectory",
                                    "effect_not_observable",
                                    "predicate_not_satisfied",
                                    "timeout",
                                    "controller_error",
                                    None,
                                ],
                            },
                            "recommended_recovery": {
                                "type": ["string", "null"],
                            },
                            "confidence": {
                                "type": ["number", "null"],
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "metrics": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "progress_evidence": progress_evidence_schema(),
                        },
                        "required": [
                            "verdict",
                            "evidence",
                            "summary",
                            *(["execution_id"] if self.require_execution_id else []),
                        ],
                    },
                },
                self.verify_step,
            ),
            (
                "replan",
                {
                    "name": "replan",
                    "description": (
                        "Ask the planning LLM for a grounded replacement of the "
                        "unfinished plan after resolved fail/retry verification or "
                        "an exhausted step budget. Pending verification must be resolved. "
                        "This returns an inactive proposal; inspect it and call "
                        "register_plan with its proposal_id. Replanning is limited "
                        "by the harness budget."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "observations": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                                "description": (
                                    "Concrete evidence about the failed or "
                                    "uncertain action and current scene."
                                ),
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional safety or recovery constraints."
                                ),
                            },
                        },
                        "required": ["reason", "observations"],
                    },
                },
                self.replan,
            ),
        ]
        if self.state.environment_horizon_only:
            replan_spec = next(spec for name, spec, _ in definitions if name == "replan")
            replan_spec["description"] += " " + COMPLETED_STEP_REVIEW_GUIDANCE
            replan_spec["input_schema"]["properties"].update({
                "review_observation_id": {
                    "type": "string",
                    "description": "Fresh completion_review.observation_id; required only for a completion review.",
                },
                "completed_step_reviews": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Required at a completion mismatch: review every listed fact_id exactly once.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "verdict": {"type": "string", "enum": list(COMPLETED_STEP_REVIEW_VERDICTS)},
                            "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                        },
                        "required": ["fact_id", "verdict", "evidence"],
                        "additionalProperties": False,
                    },
                },
            })
        if self.require_object_states:
            for name, spec, _handler in definitions:
                if name in {"create_plan", "replan"}:
                    schema = spec["input_schema"]
                    schema["properties"]["object_states"] = deepcopy(
                        OBJECT_STATES_SCHEMA
                    )
                    schema["required"].append("object_states")
                    spec["description"] += (
                        " Include current states of every task-related object and fixture; "
                        "the planner may return information_required instead of a proposal."
                    )
            definitions.append(
                (
                    "provide_planner_information",
                    {
                        "name": "provide_planner_information",
                        "description": (
                            "Answer pending planner questions from public evidence and resume the "
                            "same initial plan or replan. Read-only: does not register a plan, "
                            "advance physics, spend recovery attempts or authorize motion. "
                            "Supply the full updated snapshot; unknown is an explicit answer."
                        ),
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "request_id": {"type": "string"},
                                "observations": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                                "object_states": deepcopy(OBJECT_STATES_SCHEMA),
                            },
                            "required": ["request_id", "observations", "object_states"],
                        },
                    },
                    self.provide_planner_information,
                )
            )
        return definitions

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            **self.state.snapshot(),
            "contract_success": (
                self.state.environment_success
                if self.require_environment_success
                else self.state.completed
            ),
            **self._information_snapshot(),
            **self._completion_review_snapshot(),
        }

    def _completion_review_snapshot(self) -> dict[str, Any]:
        if not self.state.completion_review_required:
            return {}
        return {"completion_review": {
            "required": True,
            "environment_success": False,
            "observation_id": self._review_observation_id,
            "facts": deepcopy(self._verified_facts),
        }}

    def _information_snapshot(self) -> dict[str, Any]:
        pending = self._pending_information
        if pending is None:
            return {}
        return {
            "planner_information": {
                "status": "information_budget_exhausted"
                if pending["exhausted"]
                else "information_required",
                "request_id": pending["request_id"],
                "questions": deepcopy(pending["questions"]),
                "remaining_rounds": max(
                    0,
                    PLANNING_INFORMATION_MAX_ROUNDS
                    - len(pending["context"]["information_history"]),
                ),
            }
        }

    def protocol_continuation(self) -> str | None:
        """Explain the next legal operation after a reply without a tool call.

        Only the environment-horizon loop uses this: its episode may end only
        through an accepted finish, so a backend that stops with prose would
        otherwise leave the run INCOMPLETE while physical budget remains.
        """
        state = self.state
        if self.mode != "plan-gated" or not state.environment_horizon_only:
            return None
        snapshot = state.snapshot()
        feedback = state.last_feedback
        environment_ended = feedback is not None and (
            feedback.environment_terminated
            or feedback.environment_truncated
            or feedback.execution_status != "completed"
        )
        if self._pending_information is not None:
            if self._pending_information["exhausted"]:
                # End INCOMPLETE rather than repeatedly requesting an answer
                # the protocol no longer accepts. This is not native failure
                # or success and never grants physical action authority.
                return None
            next_operation = (
                "answer the pending planner questions from public evidence with "
                "provide_planner_information; do not register an incomplete plan or move"
            )
        elif environment_ended and not state.environment_success:
            next_operation = (
                "the environment has ended without native success; call "
                "finish(status='failure', summary=...)"
            )
        elif state.awaiting_verification:
            next_operation = (
                "call verify_step with execution_id="
                f"{snapshot['pending_execution_id']!r} and fresh evidence"
            )
        elif state.environment_success:
            next_operation = (
                "native environment success was observed; call "
                "finish(status='success', summary=...)"
            )
        elif not state.has_plan:
            next_operation = (
                "observe the scene, call create_plan with grounded observations, "
                "then register_plan"
            )
        elif state.completion_review_required:
            next_operation = COMPLETED_STEP_REVIEW_GUIDANCE
        elif state.needs_replan:
            next_operation = (
                "call replan with grounded observations for the unfinished task, "
                "then register_plan with the returned proposal_id"
            )
        else:
            active = snapshot["active_step"] or {}
            next_operation = (
                "call one physical tool from the active step's allowed_tools "
                f"{list(active.get('allowed_tools') or [])} to continue the "
                "original task"
            )
        return (
            "HARNESS PROTOCOL: your last reply contained no tool call, but no "
            "finish has been accepted, so the run is still open. A reply "
            "without a tool call ends the run as an INCOMPLETE protocol error. "
            f"Harness execution_mode={snapshot['execution_mode']!r}, "
            f"environment_success={state.environment_success}, "
            f"environment_ended={environment_ended}. Next legal operation: "
            f"{next_operation}. Read episode_budget in the latest tool result "
            "for steps_remaining; local tool or step limits do not end the "
            "episode."
        )

    def strict_snapshot(self) -> dict[str, Any]:
        """Return one read-only view of all strict runtime domain owners."""
        return {
            **self.snapshot(),
            "outcome_program": (
                self.outcome_program.to_payload()
                if self.outcome_program is not None
                else None
            ),
            "outcome_graph": (
                self.outcome_graph.snapshot()
                if self.outcome_graph is not None
                else None
            ),
            "stage": (
                {
                    "active_index": self.stage_runtime.active_index,
                    "active": self.stage_runtime.active.id,
                    "stages": [stage.id for stage in self.stage_runtime.stages],
                }
                if self.stage_runtime is not None
                else None
            ),
            "recovery": (
                self.recovery_tracker.snapshot()
                if self.recovery_tracker is not None
                else None
            ),
        }

    def before_tool(
        self,
        name: str,
        input_dict: dict[str, Any],
        *,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        if self._pending_information is not None and name in self.physical_tools:
            reason = "pending planner information does not authorize physical action"
            self._record("plan_violation", tool=name, reason=reason, blocked=True)
            return {"allowed": False, "reason": reason}
        context = ToolHookContext(
            execution_id=execution_id or "unbound",
            name=name,
            input_dict=input_dict,
            physical=name in self.physical_tools,
        )
        decision = self.hooks.before_tool(context, self.state)
        if decision.reason is None:
            return {
                "allowed": True,
                **({"warning": decision.warning} if decision.warning else {}),
            }
        blocked = not decision.allowed
        self._record(
            "plan_violation",
            tool=name,
            reason=decision.reason,
            blocked=blocked,
        )
        if blocked:
            return {"allowed": False, "reason": decision.reason}
        return {"allowed": True, "warning": decision.warning}

    def after_tool(
        self,
        name: str,
        input_dict: dict[str, Any],
        result: dict[str, Any],
        *,
        execution_id: str | None = None,
        feedback: ActionFeedback | None = None,
    ) -> dict[str, Any]:
        if self._strict_session_active:
            return self._strict_session_error()
        if feedback is None:
            from rpent.planning.feedback import action_feedback_from_result

            feedback = action_feedback_from_result(
                execution_id=execution_id or uuid4().hex,
                tool_name=name,
                physical=name in self.physical_tools,
                result=result,
            )
        if feedback.physical:
            self._review_observation_id = None
        elif (
            name == "observe" and self.state.completion_review_required
            and feedback.execution_status == "completed" and not feedback.error
            and not result.get("error") and type(result.get("step")) is int
        ):
            self._review_observation_id = feedback.execution_id
        context = ToolHookContext(
            execution_id=feedback.execution_id,
            name=name,
            input_dict=input_dict,
            physical=feedback.physical,
        )
        verdicts = self.hooks.after_tool(context, feedback, self.state)
        for verdict in verdicts:
            active_step = self.state.active_step
            applied = self.state.apply_verdict(
                verdict,
                require_execution_id=True,
            )
            if applied.get("ok") and verdict.verdict == "pass":
                self._record_verified_fact(
                    active_step,
                    evidence=list(verdict.evidence),
                    summary=verdict.summary,
                    execution_id=verdict.execution_id,
                )
            self._record(
                "automatic_verification",
                execution_id=feedback.execution_id,
                tool=name,
                verdict=verdict.verdict,
                source=verdict.source,
                applied=bool(applied.get("ok")),
                error=applied.get("error"),
            )
        return self.snapshot()

    def create_plan(
        self,
        goal: str,
        observations: list[str],
        constraints: list[str] | None = None,
        object_states: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate and cache a candidate plan without activating it."""
        if self._pending_information is not None:
            return self._with_mode(
                {
                    "error": "answer the pending request with provide_planner_information",
                    "harness": self.state.snapshot(),
                }
            )
        try:
            object_states = self._validate_scene(object_states)
        except ValueError as exc:
            return self._with_mode(
                {"error": str(exc), "harness": self.state.snapshot()}
            )
        if self._strict_session_active:
            return self._strict_session_error()
        if self.state.has_plan:
            return self._with_mode(
                {
                    "error": (
                        "an active plan already exists; use replan after "
                        "verification failure"
                    ),
                    "harness": self.state.snapshot(),
                }
            )
        if self.plan_generator is None:
            return self._with_mode(
                {
                    "error": "plan generator is not configured",
                    "harness": self.state.snapshot(),
                }
            )
        goal = str(goal).strip()
        if not goal:
            return self._with_mode(
                {"error": "goal must be non-empty", "harness": self.state.snapshot()}
            )
        if (
            not isinstance(observations, list)
            or not observations
            or not all(isinstance(item, str) and item.strip() for item in observations)
        ):
            return self._with_mode(
                {
                    "error": "observations must be a non-empty array of strings",
                    "harness": self.state.snapshot(),
                }
            )
        constraints = [] if constraints is None else constraints
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) and item.strip() for item in constraints
        ):
            return self._with_mode(
                {
                    "error": "constraints must be an array of strings",
                    "harness": self.state.snapshot(),
                }
            )
        language_error = self._chinese_text_error(
            [*observations, *constraints],
            "observations/constraints",
        )
        if language_error is not None:
            return self._with_mode(
                {"error": language_error, "harness": self.state.snapshot()}
            )

        return self._run_planning_request(
            {
                "kind": "initial",
                "goal": goal,
                "observations": observations,
                "constraints": constraints,
                "object_states": object_states,
                "information_history": [],
            }
        )

    def register_plan(self, proposal_id: str) -> dict[str, Any]:
        """Activate an unchanged initial or repair proposal."""
        if self._strict_session_active:
            return self._strict_session_error()
        if self._pending_information is not None:
            return self._with_mode(
                {
                    "error": "planner information is pending; no proposal may be registered",
                    "harness": self.state.snapshot(),
                }
            )
        proposal = self._proposals.get(str(proposal_id))
        if proposal is None:
            return self._with_mode(
                {
                    "error": f"unknown plan proposal: {proposal_id!r}",
                    "harness": self.state.snapshot(),
                }
            )
        if self.state.completion_review_required and (
            self._review_observation_id is None
            or proposal.get("completion_review_revision") != self._completion_review_revision
            or proposal.get("review_observation_id") != self._review_observation_id
        ):
            return self._with_mode({
                "error": "completion review required: observe and replan with completed_step_reviews before registration",
                "harness": self.snapshot(),
            })
        proposal_type = proposal.get("proposal_type", "initial")
        if proposal_type == "initial":
            result = self.state.register_plan(proposal["goal"], proposal["steps"])
        elif proposal_type == "replan":
            result = self.state.replan(
                proposal["reason"],
                proposal["steps"],
                proposal["goal"],
            )
        else:
            result = {
                "error": f"unknown plan proposal type: {proposal_type!r}",
                "harness": self.state.snapshot(),
            }
        if result.get("ok"):
            self._active_plan = deepcopy(proposal)
            self._proposals.clear()
            self._record(
                "proposal_registered",
                proposal_id=proposal_id,
                proposal_type=proposal_type,
                goal=proposal["goal"],
                strategy=proposal["strategy"],
                reason=proposal.get("reason"),
                steps=proposal["steps"],
            )
            result = {
                **result,
                "proposal_id": proposal_id,
                "proposal_type": proposal_type,
                "strategy": proposal["strategy"],
            }
        return self._with_mode(result)

    def view_plan(self) -> dict[str, Any]:
        result = self.state.view_plan()
        if self._active_plan is not None:
            result = {
                **result,
                "proposal_id": self._active_plan["proposal_id"],
                "goal": self._active_plan["goal"],
                "strategy": self._active_plan["strategy"],
            }
        return self._with_mode(result)

    def verify_step(
        self,
        verdict: str,
        evidence: list[str],
        summary: str,
        execution_id: str | None = None,
        failure_type: str | None = None,
        recommended_recovery: str | None = None,
        confidence: float | None = None,
        metrics: dict[str, float] | None = None,
        progress_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._strict_session_active:
            return self._strict_session_error()
        language_values = (
            [*evidence, summary] if isinstance(evidence, list) else [summary]
        )
        if recommended_recovery:
            language_values.append(recommended_recovery)
        language_error = self._chinese_text_error(
            language_values,
            "verify_step evidence/summary/recovery",
        )
        if language_error is not None:
            return self._with_mode(
                {"error": language_error, "harness": self.state.snapshot()}
            )
        active_step = self.state.active_step
        result = self.state.verify_step(
            verdict,
            evidence,
            summary,
            execution_id=execution_id,
            failure_type=failure_type,
            recommended_recovery=recommended_recovery,
            confidence=confidence,
            metrics=metrics,
            progress_evidence=progress_evidence,
            require_execution_id=self.require_execution_id,
        )
        if result.get("ok") and verdict == "pass" and active_step is not None:
            self._record_verified_fact(
                active_step,
                evidence=list(evidence),
                summary=summary,
                execution_id=execution_id,
            )
        return self._with_mode(result)

    def replan(
        self,
        reason: str,
        observations: list[str],
        constraints: list[str] | None = None,
        object_states: list[dict[str, Any]] | None = None,
        completed_step_reviews: list[dict[str, Any]] | None = None,
        review_observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate and cache a repair proposal without activating it."""
        if self._pending_information is not None:
            return self._with_mode(
                {
                    "error": "answer the pending request with provide_planner_information",
                    "harness": self.state.snapshot(),
                }
            )
        try:
            object_states = self._validate_scene(object_states)
        except ValueError as exc:
            return self._with_mode(
                {"error": str(exc), "harness": self.state.snapshot()}
            )
        if self._strict_session_active:
            return self._strict_session_error()
        if not self.state.has_plan:
            return self._with_mode(
                {
                    "error": "cannot replan before register_plan",
                    "harness": self.state.snapshot(),
                }
            )
        if not self.state.can_replan:
            return self._with_mode(
                {
                    "error": (
                        "replan requires resolved fail/retry verification or an exhausted step budget; "
                        "resolve pending verification first"
                    ),
                    "harness": self.state.snapshot(),
                }
            )
        remaining = self.state.replans_remaining
        if remaining is not None and remaining <= 0:
            return self._with_mode(
                {
                    "error": "replan budget exhausted",
                    "harness": self.state.snapshot(),
                }
            )
        if self.plan_generator is None:
            return self._with_mode(
                {
                    "error": "plan generator is not configured",
                    "harness": self.state.snapshot(),
                }
            )
        if self._active_plan is None:
            return self._with_mode(
                {
                    "error": "active plan metadata is unavailable",
                    "harness": self.state.snapshot(),
                }
            )

        reason = str(reason).strip()
        if not reason:
            return self._with_mode(
                {"error": "reason must be non-empty", "harness": self.state.snapshot()}
            )
        validation_error = _validate_grounding_inputs(observations, constraints)
        if validation_error is not None:
            return self._with_mode(
                {"error": validation_error, "harness": self.state.snapshot()}
            )
        normalized_constraints = [] if constraints is None else constraints
        language_error = self._chinese_text_error(
            [reason, *observations, *normalized_constraints],
            "replan reason/observations/constraints",
        )
        if language_error is not None:
            return self._with_mode(
                {"error": language_error, "harness": self.state.snapshot()}
            )
        if self.state.completion_review_required:
            try:
                self._review_completed_steps(completed_step_reviews, review_observation_id)
            except ValueError as exc:
                return self._with_mode({"error": str(exc), "harness": self.snapshot()})
        elif completed_step_reviews is not None or review_observation_id is not None:
            return self._with_mode({
                "error": "completed-step review is only allowed at a live completion mismatch",
                "harness": self.snapshot(),
            })
        current_plan = self._current_plan_context()
        goal = self._active_plan["goal"]

        return self._run_planning_request(
            {
                "kind": "replan",
                "goal": goal,
                "reason": reason,
                "observations": observations,
                "constraints": normalized_constraints,
                "current_plan": current_plan,
                "object_states": object_states,
                "information_history": [],
            }
        )

    def _validate_scene(self, objects: Any) -> list[dict[str, Any]] | None:
        if objects is None:
            if self.require_object_states:
                raise ValueError(
                    "object_states is required for task targets and fixtures"
                )
            return None
        return validate_object_states(objects)

    def provide_planner_information(
        self,
        request_id: str,
        observations: list[str],
        object_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Resume one pending request with updated public evidence, without motion."""
        if self._strict_session_active:
            return self._strict_session_error()
        pending = self._pending_information
        if pending is None or request_id != pending["request_id"]:
            return self._with_mode(
                {
                    "error": "unknown or stale planner information request_id",
                    "harness": self.state.snapshot(),
                }
            )
        if pending.get("exhausted"):
            return self._with_mode(
                {
                    "error": "planner information budget exhausted; unresolved facts remain unknown",
                    "harness": self.state.snapshot(),
                }
            )
        try:
            error = _validate_grounding_inputs(observations, [])
            if error:
                raise ValueError(error)
            objects = validate_object_states(object_states)
            previous = {
                o["id"]: o for o in pending["context"].get("object_states") or []
            }
            current = {o["id"]: o for o in objects}
            if set(previous) - set(current):
                raise ValueError(
                    "object_states must retain every previously reported task object"
                )
            for identity, obj in previous.items():
                if current[identity]["observation_step"] < obj["observation_step"]:
                    raise ValueError(
                        "stale object state evidence cannot answer a newer request"
                    )
            for question in pending["questions"]:
                obj = current.get(question["object_id"])
                if obj is None:
                    raise ValueError(f"answer missing object {question['object_id']}")
                fields = set(obj) | set(obj["states"])
                for field in question["fields"]:
                    if field not in fields:
                        raise ValueError(
                            f"answer missing requested field {field}; use unknown if unconfirmed"
                        )
            language_error = self._chinese_text_error(
                observations, "planner information observations"
            )
            if language_error:
                raise ValueError(language_error)
        except ValueError as exc:
            return self._with_mode(
                {"error": str(exc), "harness": self.state.snapshot()}
            )
        context = deepcopy(pending["context"])
        context["information_history"].append(
            {
                "request_id": request_id,
                "questions": deepcopy(pending["questions"]),
                "request_observations": deepcopy(context["observations"]),
                "request_object_states": deepcopy(context["object_states"]),
                "observations": list(observations),
                "object_states": deepcopy(objects),
            }
        )
        context["observations"] = list(observations)
        context["object_states"] = objects
        # A valid supplied answer consumes an exchange even if the provider
        # subsequently fails; failed generations must not enable unbounded retries.
        pending["context"] = deepcopy(context)
        self._record(
            "planner_information_supplied",
            request_id=request_id,
            observations=observations,
            object_states=objects,
        )
        return self._run_planning_request(context)

    def _run_planning_request(self, context: dict[str, Any]) -> dict[str, Any]:
        kind = context["kind"]
        self.state.enter_plan_mode(reason=context.get("reason", "create_plan"))
        kwargs = {key: context[key] for key in ("goal", "observations", "constraints")}
        kwargs["constraints"] = self.planning_constraints(kwargs["constraints"])
        kwargs["physical_tools"] = self.physical_tools
        if context.get("object_states") is not None:
            kwargs["object_states"] = context["object_states"]
        if context["information_history"]:
            kwargs["information_history"] = context["information_history"]
        if kind == "replan":
            kwargs.update(
                reason=context["reason"], current_plan=context["current_plan"]
            )
        try:
            generator = self.plan_generator
            generated = (
                generator.generate(**kwargs)
                if kind == "initial"
                else generator.generate_replan(**kwargs)
            )
            if "information_request" in generated:
                if not self.require_object_states:
                    raise ValueError(
                        "planner information requests are not enabled for this controller"
                    )
                questions = validate_information_request(
                    generated["information_request"]
                )
                language_error = self._chinese_text_error(
                    [
                        text
                        for question in questions
                        for text in (question["question"], question["reason"])
                    ],
                    "planner questions/reasons",
                )
                if language_error:
                    raise ValueError(language_error)
                if any(
                    generated.get(key) is not None
                    for key in ("strategy", "steps", "outcomes", "stages")
                ):
                    raise ValueError("planner questions cannot also register a plan")
                rounds = len(context["information_history"])
                exhausted = rounds >= PLANNING_INFORMATION_MAX_ROUNDS
                request_id = f"information-{uuid4().hex[:12]}"
                self._pending_information = {
                    "request_id": request_id,
                    "questions": questions,
                    "context": deepcopy(context),
                    "exhausted": exhausted,
                }
                self._proposals.clear()
                result = {
                    "ok": not exhausted,
                    "status": "information_budget_exhausted"
                    if exhausted
                    else "information_required",
                    "request_id": request_id,
                    "questions": questions,
                    "remaining_rounds": max(
                        0, PLANNING_INFORMATION_MAX_ROUNDS - rounds
                    ),
                    "registered": False,
                    "harness": self.state.snapshot(),
                }
                if exhausted:
                    result["error"] = (
                        "planner information budget exhausted; report missing evidence and leave "
                        "the run INCOMPLETE. Do not guess missing states, move, or claim success"
                    )
                self._record(
                    "planner_information_requested",
                    **{k: v for k, v in result.items() if k != "harness"},
                )
                return self._with_mode(result)
            strategy = str(generated.get("strategy", "")).strip()
            if not strategy:
                raise ValueError("plan generator returned an empty strategy")
            parsed = self.state.parse_steps(generated.get("steps"))
            self._validate_generated_language(strategy, parsed)
        except Exception as exc:
            if self._pending_information is not None:
                self._pending_information["exhausted"] = (
                    len(context["information_history"])
                    >= PLANNING_INFORMATION_MAX_ROUNDS
                )
            self._record(
                "plan_generation_failed"
                if kind == "initial"
                else "replan_generation_failed",
                goal=context["goal"],
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._with_mode(
                {
                    "error": f"{'plan' if kind == 'initial' else 'replan'} generation failed: {exc}",
                    "harness": self.state.snapshot(),
                }
            )
        self._pending_information = None
        proposal_id = f"{'plan' if kind == 'initial' else 'replan'}-{uuid4().hex[:12]}"
        proposal = {
            "proposal_id": proposal_id,
            "proposal_type": kind,
            "goal": context["goal"],
            "strategy": strategy,
            "steps": [_step_payload(step) for step in parsed],
        }
        if kind == "replan":
            proposal["reason"] = context["reason"]
            if context["current_plan"].get("completion_review", {}).get("required"):
                proposal["completion_review_revision"] = self._completion_review_revision
                proposal["review_observation_id"] = self._review_observation_id
        if context.get("object_states") is not None:
            proposal["object_states"] = deepcopy(context["object_states"])
        self._proposals[proposal_id] = deepcopy(proposal)
        self._record(
            "plan_proposed" if kind == "initial" else "replan_proposed",
            **proposal,
            observations=context["observations"],
            constraints=context["constraints"],
            information_history=context["information_history"],
            **({"current_plan": context["current_plan"]} if kind == "replan" else {}),
        )
        return self._with_mode(
            {
                "ok": True,
                **deepcopy(proposal),
                "registered": False,
                "harness": self.state.snapshot(),
            }
        )

    def _current_plan_context(self) -> dict[str, Any]:
        plan = self.state.view_plan()
        active_index = self.state.snapshot()["active_step_index"]
        steps = plan["steps"]
        split = len(steps) if active_index is None else active_index
        facts = self._verified_facts
        if self.state.environment_horizon_only:
            # A later acceptance of the same operation supersedes its previous
            # interpretation, while the complete audit history remains intact.
            latest = {(tuple(fact["operation_key"]), fact["subtask"]): fact for fact in facts}
            facts = list(latest.values())
        return {
            "proposal_id": self._active_plan["proposal_id"],
            "strategy": self._active_plan["strategy"],
            "verified_facts": deepcopy([
                fact for fact in facts
                if self._completed_step_reviews.get(fact.get("fact_id"), {}).get("verdict", "confirmed") == "confirmed"
            ]),
            **({
                "completed_step_history": deepcopy(self._verified_facts),
                "completed_step_reviews": deepcopy(list(self._completed_step_reviews.values())),
                **self._completion_review_snapshot(),
            } if self._completed_step_reviews else {}),
            "completed_steps": deepcopy(steps[:split]),
            "unfinished_steps": deepcopy(steps[split:]),
            **self.state.control_planning_context(include_capabilities=False),
        }

    def _review_completed_steps(
        self, reviews: list[dict[str, Any]] | None, observation_id: str | None,
    ) -> None:
        """Validate the whole read-only review before changing any local credit."""
        if not self._review_observation_id or observation_id != self._review_observation_id:
            raise ValueError("completion review requires a fresh observe and its observation_id")
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("completed_step_reviews must review every completed fact")
        expected = {fact["fact_id"] for fact in self._verified_facts}
        validated: dict[str, dict[str, Any]] = {}
        for review in reviews:
            if not isinstance(review, dict):
                raise ValueError("each completed-step review must be an object")
            fact_id = review.get("fact_id")
            if not isinstance(fact_id, str) or fact_id not in expected or fact_id in validated:
                raise ValueError("completed-step review fact_id is unknown or duplicated")
            verdict = review.get("verdict")
            if verdict not in COMPLETED_STEP_REVIEW_VERDICTS:
                raise ValueError("completed-step review has an invalid verdict")
            evidence = review.get("evidence")
            if not isinstance(evidence, list) or not evidence or not all(
                isinstance(item, str) and item.strip() for item in evidence
            ):
                raise ValueError("completed-step review requires concrete evidence")
            language_error = self._chinese_text_error(evidence, "completed-step review evidence")
            if language_error:
                raise ValueError(language_error)
            validated[fact_id] = {
                "fact_id": fact_id, "verdict": verdict,
                "evidence": list(evidence), "observation_id": observation_id,
            }
        if set(validated) != expected:
            raise ValueError("completed_step_reviews must cover every completed fact exactly once")
        self._completed_step_reviews.update(validated)
        self._completion_review_revision += 1
        latest = {
            (tuple(fact["operation_key"]), fact["subtask"]): fact
            for fact in self._verified_facts
        }
        for key, fact in latest.items():
            if validated[fact["fact_id"]]["verdict"] != "confirmed":
                self.state.reopen_completed_subtask(key)
        self._record("completed_steps_reviewed", reviews=list(validated.values()))

    def _record_verified_fact(
        self,
        active_step: Any,
        *,
        evidence: list[str],
        summary: str,
        execution_id: str | None,
    ) -> None:
        if active_step is None:
            return
        self._verified_facts.append(
            {
                **({
                    "fact_id": f"fact-{len(self._verified_facts) + 1}",
                    "expected_evidence": list(active_step.expected_evidence),
                    "operation_key": list(operation_key(active_step)),
                    "subtask": subtask_name(active_step),
                } if self.state.environment_horizon_only else {}),
                "step_id": active_step.id,
                "intent": active_step.intent,
                "summary": summary,
                "evidence": evidence,
                "execution_id": execution_id,
            }
        )

    def _chinese_text_error(
        self,
        values: list[Any],
        field: str,
    ) -> str | None:
        if not self.use_chinese:
            return None
        invalid = [
            str(value)
            for value in values
            if str(value).strip() and not _contains_cjk(str(value))
        ]
        if invalid:
            return (
                f"{field} must use Simplified Chinese when --use_chines is set; "
                f"non-Chinese text: {invalid[0]!r}"
            )
        return None

    def _validate_generated_language(self, strategy: str, steps: list[Any]) -> None:
        values: list[Any] = [strategy]
        for step in steps:
            values.extend(
                [
                    step.intent,
                    *step.preconditions,
                    *step.expected_evidence,
                ]
            )
            if step.on_failure != "replan":
                values.append(step.on_failure)
        error = self._chinese_text_error(values, "generated planning fields")
        if error is not None:
            raise ValueError(error)

    def _with_mode(self, result: dict[str, Any]) -> dict[str, Any]:
        harness = result.get("harness")
        if isinstance(harness, dict):
            result = dict(result)
            result["harness"] = {
                "mode": self.mode,
                **harness,
                **self._information_snapshot(),
                **self._completion_review_snapshot(),
            }
        return result

    def _strict_session_error(self) -> dict[str, Any]:
        return {
            "error": "legacy controller mutation is disabled by the active strict session",
            "harness": self.strict_snapshot(),
        }

    def _record(self, event: str, **payload: Any) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        if self.event_recorder is not None:
            try:
                mirror_payload = dict(payload)
                if "source" in mirror_payload:
                    # JsonlEventRecorder reserves ``source`` for the event
                    # producer. Preserve a verifier/hook source separately.
                    mirror_payload["verdict_source"] = mirror_payload.pop("source")
                self.event_recorder.record(
                    event,
                    source="harness",
                    **mirror_payload,
                )
            except Exception as exc:
                get_logger("harness").warning(
                    "failed to mirror harness event %s: %s",
                    event,
                    exc,
                )


def _step_payload(step: Any) -> dict[str, Any]:
    """Return a JSON-shaped step suitable for later registration."""
    return {
        "id": step.id,
        "intent": step.intent,
        "allowed_tools": list(step.allowed_tools),
        "preconditions": list(step.preconditions),
        "expected_evidence": list(step.expected_evidence),
        "on_failure": step.on_failure,
        "max_attempts": step.max_attempts,
        "skill_row": step.skill_row,
        "tier": step.tier,
        "control_mode": step.control_mode,
        "defaulted": step.defaulted,
        **{name: getattr(step, name) for name in IDENTITY_FIELDS if getattr(step, name, None) is not None},
        "covers_outcomes": list(step.covers_outcomes),
        "stage_id": step.stage_id,
        "step_type": step.step_type,
    }


def _validate_grounding_inputs(
    observations: Any,
    constraints: Any,
) -> str | None:
    if (
        not isinstance(observations, list)
        or not observations
        or not all(isinstance(item, str) and item.strip() for item in observations)
    ):
        return "observations must be a non-empty array of strings"
    if constraints is not None and (
        not isinstance(constraints, list)
        or not all(isinstance(item, str) and item.strip() for item in constraints)
    ):
        return "constraints must be an array of strings"
    return None


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in text
    )
