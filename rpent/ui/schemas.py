"""Pydantic response models for retained Evaluation and Case facts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from rpent.recording.contracts import JSON_SAFE_INTEGER_MAX

JSONSafeInteger = Annotated[int, Field(ge=0, le=JSON_SAFE_INTEGER_MAX)]

ExecutionState = Literal[
    "queued",
    "starting",
    "running",
    "finalizing",
    "terminal",
    "cancelled",
    "worker_lost",
]


class _Projection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ArtifactHealthView(_Projection):
    status: Literal["complete", "degraded", "partial", "unavailable"]
    issues: tuple[str, ...] = ()


class UsageView(_Projection):
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)


class AttemptSummaryView(_Projection):
    status: str | None = None
    success: bool | None = None
    step_count: int | None = Field(default=None, ge=0)
    action_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    planner_elapsed_seconds: float | None = Field(default=None, ge=0)
    vla_chunks: int | None = Field(default=None, ge=0)
    retained_frame_count: int | None = Field(default=None, ge=0)
    started_at: str | None = None
    finished_at: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    usage: UsageView = Field(default_factory=UsageView)
    artifact_health: ArtifactHealthView | None = None


class AttemptListItemView(_Projection):
    attempt_id: str
    summary: AttemptSummaryView


class FailureCountView(_Projection):
    reason: str
    count: int = Field(ge=0)


class EvaluationMetricsView(_Projection):
    scheduled: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    successful: int = Field(default=0, ge=0)
    running: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    transitions: int | None = Field(default=None, ge=0)
    mean_duration_seconds: float | None = Field(default=None, ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    artifact_complete: int = Field(default=0, ge=0)
    artifact_degraded: int = Field(default=0, ge=0)
    artifact_unavailable: int = Field(default=0, ge=0)
    failures: tuple[FailureCountView, ...] = ()


class OfficialMetricsView(_Projection):
    available: bool = False
    task_success_rates: dict[str, float] = Field(default_factory=dict)
    split_success_rates: dict[str, float] = Field(default_factory=dict)
    overall_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    completed_rollouts: int | None = Field(default=None, ge=0)
    scheduled_rollouts: int | None = Field(default=None, ge=0)
    completed_tasks: int | None = Field(default=None, ge=0)
    scheduled_tasks: int | None = Field(default=None, ge=0)
    is_complete: bool = False

    @model_validator(mode="after")
    def validate_counts(self) -> OfficialMetricsView:
        counts = (
            self.completed_rollouts,
            self.scheduled_rollouts,
            self.completed_tasks,
            self.scheduled_tasks,
        )
        if self.available and any(value is None for value in counts):
            raise ValueError("available official metrics require complete counts")
        if (
            self.completed_rollouts is not None
            and self.scheduled_rollouts is not None
            and self.completed_rollouts > self.scheduled_rollouts
        ):
            raise ValueError("completed rollouts cannot exceed scheduled rollouts")
        if (
            self.completed_tasks is not None
            and self.scheduled_tasks is not None
            and self.completed_tasks > self.scheduled_tasks
        ):
            raise ValueError("completed tasks cannot exceed scheduled tasks")
        expected_complete = (
            self.available
            and self.completed_rollouts == self.scheduled_rollouts
            and self.completed_tasks == self.scheduled_tasks
        )
        if self.is_complete != expected_complete:
            raise ValueError("official completeness does not match counts")
        return self


class RPentDiagnosticsView(_Projection):
    available: bool = False
    selected_rollouts: int | None = Field(default=None, ge=0)
    scheduled_rollouts: int | None = Field(default=None, ge=0)
    contract_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    system_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    selected_cohort: dict[str, JsonValue] = Field(default_factory=dict)
    all_retained_audit: dict[str, JsonValue] = Field(default_factory=dict)
    structured_evidence: dict[str, JsonValue] = Field(default_factory=dict)


class QualificationCheckView(_Projection):
    id: str = Field(min_length=1)
    accepted: bool
    evidence: JsonValue
    remediation: str = Field(min_length=1)


class QualificationView(_Projection):
    available: bool = False
    profile_id: str | None = None
    status: Literal["partial", "rejected", "qualified"] | None = None
    accepted: bool = False
    checks: tuple[QualificationCheckView, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> QualificationView:
        if self.accepted != (self.status == "qualified"):
            raise ValueError("qualification status does not match acceptance")
        if self.available and (self.profile_id is None or self.status is None):
            raise ValueError("available qualification requires identity and status")
        if len({check.id for check in self.checks}) != len(self.checks):
            raise ValueError("qualification checks must be unique")
        if self.accepted and not all(check.accepted for check in self.checks):
            raise ValueError("qualified projection contains a failed check")
        return self


class SubmissionView(_Projection):
    available: bool = False
    download_url: str | None = None


class ArtifactDescriptor(_Projection):
    artifact_id: str
    kind: str
    relative_path: str
    available: bool
    byte_size: int | None = Field(default=None, ge=0)


class RecordingProfileView(_Projection):
    mode: Literal["none", "primary", "selected", "all"]
    camera_ids: tuple[str, ...] = ()


class CameraDescriptorView(_Projection):
    resource_id: str
    camera_id: str
    label: str
    role: str | None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    channels: int = Field(gt=0)
    fps: float = Field(gt=0)
    media_type: str


class DataCameraView(_Projection):
    id: str
    label: str
    role: str | None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)


class DataDimensionView(_Projection):
    index: int = Field(ge=0)
    name: str
    label: str
    group: str
    unit: str | None = None


class DataVectorView(_Projection):
    shape: tuple[int, ...]
    dimensions: tuple[DataDimensionView, ...]


class DataSignalView(_Projection):
    id: str
    label: str
    kind: str
    authoritative: bool


class CaseDataSchemaView(_Projection):
    schema_version: int = Field(ge=1)
    cameras: tuple[DataCameraView, ...]
    action: DataVectorView
    state: DataVectorView
    signals: tuple[DataSignalView, ...]


class CommandView(_Projection):
    evaluation_id: str
    operation: Literal[
        "start", "resume", "rerun", "cancel", "retry", "action_replay", "force_stop"
    ]
    state: Literal["queued", "running", "completed", "cancelled", "failed"]


class StorageEstimateBasisView(_Projection):
    total_horizon_steps: int = Field(ge=0)
    retained_camera_count_min: int = Field(ge=0)
    retained_camera_count_max: int = Field(ge=0)
    bytes_per_camera_step_min: int = Field(gt=0)
    bytes_per_camera_step_max: int = Field(gt=0)
    contract_camera_count: int = Field(ge=0)
    task_resolution: str = Field(min_length=1)
    note: str = Field(min_length=1)


class StorageEstimateView(_Projection):
    minimum_bytes: int = Field(ge=0)
    maximum_bytes: int = Field(ge=0)
    basis: StorageEstimateBasisView


class LaunchWorkloadView(_Projection):
    profile_id: str | None = None
    task_count: int = Field(gt=0)
    rollouts_per_task: int = Field(gt=0)
    case_count: int = Field(gt=0)
    batch_count: int = Field(gt=0)
    scene_split: str | None = None
    seed: int | None = None
    horizon_source: str | None = None
    estimated_runtime_seconds: float | None = Field(default=None, gt=0)
    locked_effective_facts: dict[str, JsonValue] = Field(default_factory=dict)
    locked_field_names: tuple[str, ...] = ()


class LaunchPreviewView(_Projection):
    case_count: int = Field(gt=0)
    task_count: int = Field(gt=0)
    episodes_per_task: int = Field(gt=0)
    total_horizon_steps: int = Field(gt=0)
    recording_recommendation: Literal["none", "all", "primary"]
    storage_estimate: StorageEstimateView
    task_resolution: str = Field(min_length=1)
    effective_config: dict[str, JsonValue] = Field(default_factory=dict)
    prerequisites: tuple["LaunchPrerequisiteView", ...] = ()
    workload: LaunchWorkloadView
    requires_workload_confirmation: bool
    preview_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class LaunchPreflightView(_Projection):
    prerequisites: tuple["LaunchPrerequisiteView", ...] = ()


class SimulatorSummaryView(_Projection):
    simulator_id: str
    capabilities: dict[str, JsonValue]


class LaunchOwnershipView(_Projection):
    field: str = Field(min_length=1)
    values: list[JsonValue] = Field(min_length=1)


class LaunchSuggestionsWhenView(LaunchOwnershipView):
    suggestions: list[JsonValue] = Field(default_factory=list)


class LaunchSchemaPropertyView(_Projection):
    title: str
    type: Literal["string", "integer", "number", "boolean", "array"]
    description: str | None = None
    enum: list[JsonValue] | None = None
    suggestions: list[JsonValue] | None = None
    choice_labels: list[str] | None = None
    selection_mode: Literal["single", "multiple"] | None = None
    exclusive_choices: list[JsonValue] | None = None
    allow_custom_value: bool | None = None
    items: dict[str, JsonValue] | None = None
    step: float | None = None
    display_scale: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    default: JsonValue = None
    group: str = "task"
    order: int = 0
    visible_in: tuple[str, ...] = ()
    owned_when: LaunchOwnershipView | None = None
    visible_when: LaunchOwnershipView | None = None
    suggestions_when: list[LaunchSuggestionsWhenView] = Field(default_factory=list)


class LaunchGroupView(_Projection):
    id: str
    label: str
    order: int
    collapsed: bool = False


class LaunchPrerequisiteView(_Projection):
    id: str
    label: str
    status: Literal["ready", "blocked", "warning", "not-applicable"]
    message: str


class LaunchSchemaView(_Projection):
    version: int = Field(default=1, gt=0)
    default_mode: str = "hybrid"
    modes: tuple[str, ...] = ("hybrid",)
    groups: tuple[LaunchGroupView, ...] = ()
    type: Literal["object"]
    additionalProperties: Literal[False] = False
    properties: dict[str, LaunchSchemaPropertyView]
    required: list[str] = Field(default_factory=list)


class LaunchAcceptedView(_Projection):
    ok: Literal[True]


class ErrorView(_Projection):
    detail: str


class ConsolePageView(_Projection):
    offset: JSONSafeInteger
    next_offset: JSONSafeInteger
    text: str
    truncated: bool
    complete: bool

    @model_validator(mode="after")
    def validate_offsets(self) -> ConsolePageView:
        if self.next_offset < self.offset:
            raise ValueError("next_offset must not precede offset")
        return self


class CaseSummary(_Projection):
    case_id: str
    ordinal: int
    task_name: str
    task_index: int | None
    episode_index: int
    seed: int
    execution_state: ExecutionState
    attempt_id: str | None
    episode_result: dict[str, JsonValue] | None
    artifact_health: ArtifactHealthView | None
    current_step: int | None
    started_at: str | None = None
    finished_at: str | None = None
    run_summary: AttemptSummaryView | None = None


class CaseDetail(CaseSummary):
    evaluation_id: str
    manifest_id: str
    last_sequence: JSONSafeInteger
    case: dict[str, JsonValue]
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    recording_profile: RecordingProfileView | None = None
    cameras: tuple[CameraDescriptorView, ...] = ()
    retained_camera_ids: tuple[str, ...] = ()
    data_schema: CaseDataSchemaView | None = None
    attempts: tuple[AttemptListItemView, ...] = ()
    attempts_total: int = Field(default=0, ge=0)
    attempts_truncated: bool = False
    available_commands: tuple[Literal["rerun", "action_replay"], ...] = ()
    playback_artifact_id: str | None = None

    @field_validator("available_commands")
    @classmethod
    def validate_available_commands(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(dict.fromkeys(value)):
            raise ValueError("available_commands must be stable and unique")
        expected = tuple(
            command for command in ("rerun", "action_replay") if command in value
        )
        if value != expected:
            raise ValueError("available_commands must use canonical order")
        return value


class EvaluationListItem(_Projection):
    evaluation_id: str
    manifest_id: str
    created_at: str
    environment: str
    task_selector: str
    execution_state: ExecutionState
    scheduled: int
    completed: int
    metrics: EvaluationMetricsView = Field(default_factory=EvaluationMetricsView)


class TaskSummaryView(_Projection):
    task_name: str
    scheduled: int = Field(ge=0)
    completed: int = Field(ge=0)
    successful: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_totals(self) -> TaskSummaryView:
        if not self.successful <= self.completed <= self.scheduled:
            raise ValueError("task successful <= completed <= scheduled is required")
        expected = self.successful / self.scheduled if self.scheduled else 0.0
        if self.success_rate != expected:
            raise ValueError("task success_rate does not match totals")
        return self


class StatusCountView(_Projection):
    status: Literal[
        "success",
        "terminal_failure",
        "truncated",
        "timeout",
        "error",
        "cancelled",
    ]
    count: int = Field(ge=0)


class EvaluationSummaryView(_Projection):
    manifest_id: str
    evaluation_id: str | None = None
    scheduled: int = Field(ge=0)
    completed: int = Field(ge=0)
    successful: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    errors: int = Field(ge=0)
    timeouts: int = Field(ge=0)
    status_counts: tuple[StatusCountView, ...]
    per_task: tuple[TaskSummaryView, ...]

    @model_validator(mode="after")
    def validate_totals(self) -> EvaluationSummaryView:
        if not self.successful <= self.completed <= self.scheduled:
            raise ValueError("successful <= completed <= scheduled is required")
        counts = {item.status: item.count for item in self.status_counts}
        if len(counts) != len(self.status_counts):
            raise ValueError("status counts must have unique names")
        if sum(counts.values()) > self.scheduled:
            raise ValueError("status counts cannot exceed scheduled")
        terminal_count = sum(
            count for status, count in counts.items() if status != "cancelled"
        )
        if self.completed != terminal_count:
            raise ValueError("completed does not match terminal status counts")
        if self.successful != counts.get("success", 0):
            raise ValueError("successful does not match success status count")
        if self.errors != counts.get("error", 0):
            raise ValueError("errors do not match error status count")
        if self.timeouts != counts.get("timeout", 0):
            raise ValueError("timeouts do not match timeout status count")
        expected = self.successful / self.scheduled if self.scheduled else 0.0
        if self.success_rate != expected:
            raise ValueError("success_rate does not match totals")
        if self.per_task and (
            sum(item.scheduled for item in self.per_task) != self.scheduled
            or sum(item.completed for item in self.per_task) != self.completed
            or sum(item.successful for item in self.per_task) != self.successful
        ):
            raise ValueError("per-task totals do not match evaluation totals")
        return self


class PreparationErrorView(_Projection):
    error_type: str
    message: str


class PreparationPhaseView(_Projection):
    phase: Literal[
        "evaluation_created",
        "sam3_loading",
        "vla_loading",
        "simulator_starting",
        "planner_starting",
        "running",
        "finalizing",
    ]
    status: Literal[
        "pending",
        "loading",
        "ready",
        "reused",
        "skipped",
        "failed",
        "cancelled",
    ]
    started_at: str | None
    finished_at: str | None
    detail: str | None


class PreparationView(_Projection):
    status: Literal["preparing", "ready", "failed", "cancelled"]
    current_phase: str
    phases: tuple[PreparationPhaseView, ...]
    error: PreparationErrorView | None


class EvaluationDetail(EvaluationListItem):
    manifest: dict[str, JsonValue]
    summary: EvaluationSummaryView | None
    cases: tuple[CaseSummary, ...]
    preparation: PreparationView | None = None
    metrics: EvaluationMetricsView = Field(default_factory=EvaluationMetricsView)
    official_metrics: OfficialMetricsView = Field(default_factory=OfficialMetricsView)
    rpent_diagnostics: RPentDiagnosticsView = Field(
        default_factory=RPentDiagnosticsView
    )
    qualification: QualificationView = Field(default_factory=QualificationView)
    submission: SubmissionView = Field(default_factory=SubmissionView)
    available_commands: tuple[
        Literal["cancel", "resume", "retry", "force_stop"], ...
    ] = ()

    @field_validator("available_commands")
    @classmethod
    def validate_available_commands(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(dict.fromkeys(value)):
            raise ValueError("available_commands must be stable and unique")
        expected = tuple(
            command
            for command in ("cancel", "resume", "retry", "force_stop")
            if command in value
        )
        if value != expected:
            raise ValueError("available_commands must use canonical order")
        return value


class EventPage(_Projection):
    evaluation_id: str
    case_id: str
    attempt_id: str | None
    since: JSONSafeInteger
    last_sequence: JSONSafeInteger
    incomplete_tail: bool
    has_more: bool
    events: tuple[dict[str, JsonValue], ...]


class ActionCameraClipView(_Projection):
    camera_id: str
    artifact_id: str


class OperationSummaryView(_Projection):
    operation_id: str
    execution_id: str | None
    tool: str
    plan_step_id: str | None = None
    status: Literal["running", "succeeded", "failed"]
    started_sequence: JSONSafeInteger | None
    finished_sequence: JSONSafeInteger | None
    first_step_id: JSONSafeInteger | None
    last_step_id: JSONSafeInteger | None
    executed_actions: JSONSafeInteger | None
    result: dict[str, JsonValue] | None
    elapsed_seconds: float | None = Field(ge=0)
    error: str | None
    clip_artifact_id: str | None
    has_physical_transitions: bool = False
    camera_clips: tuple[ActionCameraClipView, ...] = ()


class OperationPage(_Projection):
    evaluation_id: str
    case_id: str
    attempt_id: str
    last_sequence: JSONSafeInteger
    incomplete_tail: bool
    first_step_id: JSONSafeInteger | None
    latest_step_id: JSONSafeInteger | None
    operations: tuple[OperationSummaryView, ...]


class TransitionAnchorView(_Projection):
    step_id: JSONSafeInteger
    sequence: JSONSafeInteger


class SequenceWindowView(_Projection):
    from_sequence: JSONSafeInteger
    to_sequence: JSONSafeInteger

    @model_validator(mode="after")
    def validate_order(self) -> SequenceWindowView:
        if self.to_sequence < self.from_sequence:
            raise ValueError("sequence window must be ordered")
        return self


class StepGapView(_Projection):
    from_step: JSONSafeInteger
    to_step: JSONSafeInteger

    @model_validator(mode="after")
    def validate_order(self) -> StepGapView:
        if self.to_step < self.from_step:
            raise ValueError("step gap must be ordered")
        return self


class TimelineWindow(_Projection):
    evaluation_id: str
    case_id: str
    attempt_id: str
    requested_step: JSONSafeInteger | None
    selected_transition: TransitionAnchorView | None
    first_transition: TransitionAnchorView | None
    latest_transition: TransitionAnchorView | None
    transition_count: JSONSafeInteger
    event_window: SequenceWindowView | None
    gap: StepGapView | None
    incomplete_tail: bool


__all__ = [
    "ActionCameraClipView",
    "ArtifactDescriptor",
    "ArtifactHealthView",
    "CameraDescriptorView",
    "CaseDataSchemaView",
    "CaseDetail",
    "CaseSummary",
    "CommandView",
    "EvaluationDetail",
    "EvaluationListItem",
    "EvaluationSummaryView",
    "ErrorView",
    "EventPage",
    "ExecutionState",
    "LaunchAcceptedView",
    "LaunchOwnershipView",
    "LaunchSchemaPropertyView",
    "LaunchSchemaView",
    "LaunchSuggestionsWhenView",
    "LaunchPreviewView",
    "OperationPage",
    "OperationSummaryView",
    "RecordingProfileView",
    "StorageEstimateBasisView",
    "StorageEstimateView",
    "StepGapView",
    "SequenceWindowView",
    "SimulatorSummaryView",
    "StatusCountView",
    "TaskSummaryView",
    "TimelineWindow",
    "TransitionAnchorView",
]
