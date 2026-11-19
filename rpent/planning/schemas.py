"""Shared, serializable contracts for planning, execution, and feedback.

These types are deliberately provider- and environment-independent.  Raw
robot/tool payloads should be normalized into these contracts before planning
logic or evaluation code consumes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from rpent.planning.outcome_program import OutcomeProgram

FailureType = Literal[
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
]
OutcomeStatus = Literal[
    "success",
    "progress",
    "retryable_failure",
    "failure",
    "uncertain",
    "not_applicable",
]
OutcomeSource = Literal["adapter", "tool-signal", "unknown"]


@dataclass(frozen=True)
class SkillOutcome:
    """Evidence-backed local effect of one completed skill call."""

    status: OutcomeStatus
    failure_type: FailureType | None = None
    retryable: bool | None = None
    progress: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    source: OutcomeSource = "unknown"

    def __post_init__(self) -> None:
        if self.status not in {
            "success",
            "progress",
            "retryable_failure",
            "failure",
            "uncertain",
            "not_applicable",
        }:
            raise ValueError(f"invalid outcome status: {self.status!r}")
        if self.source not in {"adapter", "tool-signal", "unknown"}:
            raise ValueError(f"invalid outcome source: {self.source!r}")
        if self.progress is not None and not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        if self.status == "retryable_failure" and self.retryable is not True:
            raise ValueError("retryable_failure requires retryable=True")
        if self.status == "success" and self.failure_type is not None:
            raise ValueError("success cannot carry failure_type")
        if self.status not in {"uncertain", "not_applicable"} and not self.evidence:
            raise ValueError(f"{self.status} outcome requires evidence")
        for key, value in self.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"metric {key!r} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {key!r} must be finite")


def _unknown_skill_outcome() -> SkillOutcome:
    return SkillOutcome(status="uncertain", source="unknown")


def _non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True)
class TaskSpec:
    """Normalized user goal, separate from environment success metadata."""

    raw_instruction: str
    verb: str
    target_object: str
    destination: str | None = None
    spatial_constraints: tuple[str, ...] = ()
    success_description: str = ""
    ambiguities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.raw_instruction, "raw_instruction")
        _non_empty(self.verb, "verb")
        _non_empty(self.target_object, "target_object")


@dataclass(frozen=True)
class ObservationSummary:
    """Compact planner-facing state at one logical robot step."""

    step: int
    image_paths: tuple[str, ...]
    task_instruction: str | None = None
    eef_position: tuple[float, float, float] | None = None
    proprioception: tuple[float, ...] | None = None
    gripper_opening: float | None = None
    holding_object: str | None = None
    visible_objects: tuple[dict[str, Any], ...] = ()
    last_action: dict[str, Any] | None = None
    last_action_result: dict[str, Any] | None = None
    environment_success: bool = False
    environment_terminated: bool = False
    environment_truncated: bool = False

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be >= 0")


@dataclass(frozen=True)
class Stage:
    """One immutable high-level partition of task outcomes."""

    id: str
    intent: str
    outcome_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty(self.id, "stage.id")
        _non_empty(self.intent, "stage.intent")
        if not self.outcome_ids or any(not value.strip() for value in self.outcome_ids):
            raise ValueError("stage outcome_ids must be non-empty")
        if len(self.outcome_ids) != len(set(self.outcome_ids)):
            raise ValueError("stage outcome_ids must be unique")


@dataclass(frozen=True)
class Plan:
    """A versioned high-level plan whose next step may be executed."""

    goal: str
    steps: tuple[PlanStep, ...]
    assumptions: tuple[str, ...] = ()
    plan_id: str | None = None
    stages: tuple[Stage, ...] = ()
    contract_id: str | None = None
    outcome_program_schema_version: str | None = None
    required_outcome_ids: tuple[str, ...] = ()
    outcome_program: OutcomeProgram | None = None

    def __post_init__(self) -> None:
        _non_empty(self.goal, "goal")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        stage_ids = [stage.id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("plan stage ids must be unique")
        if (self.contract_id is None) != (self.outcome_program_schema_version is None):
            raise ValueError(
                "contract_id and outcome_program_schema_version must be supplied together"
            )
        if self.contract_id is not None:
            _non_empty(self.contract_id, "contract_id")
            _non_empty(
                self.outcome_program_schema_version or "",
                "outcome_program_schema_version",
            )
        if len(self.required_outcome_ids) != len(set(self.required_outcome_ids)):
            raise ValueError("required_outcome_ids must be unique")
        if self.outcome_program is not None:
            if self.contract_id != self.outcome_program.contract_id:
                raise ValueError("plan contract_id must match outcome_program")
            if (
                self.outcome_program_schema_version
                != self.outcome_program.schema_version
            ):
                raise ValueError("plan schema version must match outcome_program")


@dataclass(frozen=True)
class PlanStep:
    """One auditable action or observation step in a structured task plan."""

    id: str
    intent: str
    allowed_tools: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    expected_evidence: tuple[str, ...] = ()
    on_failure: str = "replan"
    max_attempts: int = 2
    covers_outcomes: tuple[str, ...] = ()
    stage_id: str | None = None
    step_type: Literal["action", "observation"] = "action"
    skill_row: str = "none"
    tier: Literal["reliable", "unstable", "weak", "none"] = "none"
    control_mode: Literal["delegate", "monitor", "takeover", "transition"] = "monitor"
    defaulted: bool = False
    # Stable operation identity across replanning (1.0.11). None keeps the
    # legacy identity; subtask None means the whole operation (carry for a
    # transition); travel_m sizes a transition's step quota.
    operation: str | None = None
    subtask: str | None = None
    travel_m: float | None = None

    def __post_init__(self) -> None:
        _non_empty(self.id, "id")
        _non_empty(self.intent, "intent")
        if self.step_type not in {"action", "observation"}:
            raise ValueError(f"invalid plan step type: {self.step_type!r}")
        _non_empty(self.skill_row, "skill_row")
        for name in ("operation", "subtask"):
            value = getattr(self, name)
            if value is not None:
                _non_empty(value, name)
        if self.travel_m is not None and (
            isinstance(self.travel_m, bool) or not math.isfinite(self.travel_m) or self.travel_m < 0
        ):
            raise ValueError("travel_m must be a finite nonnegative number")
        if self.tier not in {"reliable", "unstable", "weak", "none"}:
            raise ValueError(f"invalid tier: {self.tier!r}")
        if self.control_mode not in {"delegate", "monitor", "takeover", "transition"}:
            raise ValueError(f"invalid control_mode: {self.control_mode!r}")
        if self.step_type == "action" and not self.allowed_tools:
            raise ValueError("allowed_tools must be non-empty")
        if self.step_type == "observation" and self.allowed_tools:
            raise ValueError("observation steps cannot allow tools")
        if len(self.covers_outcomes) != len(set(self.covers_outcomes)):
            raise ValueError("covers_outcomes must be unique")
        if any(not value.strip() for value in self.covers_outcomes):
            raise ValueError("covers_outcomes must contain non-empty IDs")
        if self.stage_id is not None:
            _non_empty(self.stage_id, "stage_id")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


@dataclass(frozen=True)
class PlanningDecision:
    """One auditable output from a planning policy.

    ``summary`` is an operational explanation, not hidden chain-of-thought.
    """

    kind: Literal["act", "replan", "finish"]
    summary: str
    plan_step_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    expected_evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"act", "replan", "finish"}:
            raise ValueError(f"invalid decision kind: {self.kind!r}")
        _non_empty(self.summary, "summary")
        if self.kind == "act" and not (self.plan_step_id and self.tool_name):
            raise ValueError("act decisions require plan_step_id and tool_name")
        if self.kind != "act" and (self.tool_name or self.tool_args):
            raise ValueError("only act decisions may contain a tool call")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class StepVerdict:
    """Evidence-grounded assessment of a completed physical action."""

    verdict: Literal["progress", "pass", "retry", "fail", "uncertain"]
    evidence: tuple[str, ...]
    summary: str
    failure_type: FailureType | None = None
    recommended_recovery: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    progress_evidence: dict[str, Any] | None = None
    execution_id: str | None = None
    step_id: str | None = None
    confidence: float | None = None
    source: Literal["agent", "tool-signal", "verifier", "hook"] = "agent"

    def __post_init__(self) -> None:
        if self.verdict not in {
            "progress",
            "pass",
            "retry",
            "fail",
            "uncertain",
        }:
            raise ValueError(f"invalid verdict: {self.verdict!r}")
        if not self.evidence:
            raise ValueError("evidence must be non-empty")
        _non_empty(self.summary, "summary")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        from rpent.harness.policy_progress import validate_progress_evidence
        validate_progress_evidence(self.progress_evidence)


ExecutionStatus = Literal[
    "completed",
    "rejected",
    "timeout",
    "controller_error",
]


@dataclass(frozen=True)
class ActionFeedback:
    """Normalized result of one attempted tool execution."""

    execution_id: str
    tool_name: str
    physical: bool
    execution_status: ExecutionStatus
    observation: ObservationSummary | None = None
    error: str | None = None
    failure_type: FailureType | None = None
    environment_success: bool = False
    environment_terminated: bool = False
    environment_truncated: bool = False
    measurements: dict[str, float] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict, repr=False)
    outcome_adapter_error: str | None = None
    outcome: SkillOutcome = field(default_factory=_unknown_skill_outcome)

    def __post_init__(self) -> None:
        _non_empty(self.execution_id, "execution_id")
        _non_empty(self.tool_name, "tool_name")
        if self.execution_status not in {
            "completed",
            "rejected",
            "timeout",
            "controller_error",
        }:
            raise ValueError(f"invalid execution status: {self.execution_status!r}")
        if self.execution_status == "completed" and self.error:
            raise ValueError("completed feedback cannot contain an error")
