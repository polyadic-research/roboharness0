"""Versioned planning policy configuration and shared planning schemas."""

from typing import TYPE_CHECKING, Any

from rpent.planning.events import (
    EventRecorder,
    JsonlEventRecorder,
    NullEventRecorder,
)
from rpent.planning.feedback import action_feedback_from_result
from rpent.planning.generator import PlanGenerator
from rpent.planning.manifest import build_experiment_manifest, content_hash
from rpent.planning.observation import observation_from_tool_result
from rpent.planning.outcomes import (
    UNKNOWN_OUTCOME_ADAPTER,
    SkillOutcomeAdapter,
    UnknownPhysicalOutcomeAdapter,
)
from rpent.planning.replay import RecordedDecision, load_events, recorded_decisions
from rpent.planning.schemas import (
    ActionFeedback,
    FailureType,
    ObservationSummary,
    OutcomeSource,
    OutcomeStatus,
    Plan,
    PlanningDecision,
    PlanStep,
    SkillOutcome,
    StepVerdict,
    TaskSpec,
)
from rpent.planning.stack import (
    AUTHORITATIVE_PLANNING_PROFILE,
    DEFAULT_PLANNING_PROFILE,
    LEGACY_AUTHORITATIVE_PLANNING_PROFILE,
    ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE,
    ROBOCASA_HOOKED_GATED_PLANNING_PROFILE,
    ROBOCASA_HORIZON_PLANNING_PROFILE,
    ROBOCASA_PRECISE_PLANNING_PROFILE,
    ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE,
    EffectivePlanningConfig,
    PlanningProfile,
    PlanningStack,
    get_planning_profile,
    list_planning_profile_ids,
    list_planning_profiles,
    resolve_planning_config,
    resolve_planning_stack,
    resolve_required_planning_profile,
)

if TYPE_CHECKING:
    from rpent.planning.runtime import PlanningOrchestrator


def __getattr__(name: str) -> Any:
    """Lazily expose the orchestrator, which depends on Toolkit."""
    if name == "PlanningOrchestrator":
        from rpent.planning.runtime import PlanningOrchestrator

        return PlanningOrchestrator
    raise AttributeError(name)


__all__ = [
    "ActionFeedback",
    "AUTHORITATIVE_PLANNING_PROFILE",
    "DEFAULT_PLANNING_PROFILE",
    "EffectivePlanningConfig",
    "EventRecorder",
    "FailureType",
    "JsonlEventRecorder",
    "LEGACY_AUTHORITATIVE_PLANNING_PROFILE",
    "NullEventRecorder",
    "ObservationSummary",
    "OutcomeSource",
    "OutcomeStatus",
    "Plan",
    "PlanGenerator",
    "PlanStep",
    "PlanningDecision",
    "PlanningOrchestrator",
    "PlanningProfile",
    "PlanningStack",
    "ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE",
    "ROBOCASA_HOOKED_GATED_PLANNING_PROFILE",
    "ROBOCASA_HORIZON_PLANNING_PROFILE",
    "ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE",
    "ROBOCASA_PRECISE_PLANNING_PROFILE",
    "RecordedDecision",
    "StepVerdict",
    "SkillOutcome",
    "TaskSpec",
    "UNKNOWN_OUTCOME_ADAPTER",
    "SkillOutcomeAdapter",
    "UnknownPhysicalOutcomeAdapter",
    "action_feedback_from_result",
    "build_experiment_manifest",
    "content_hash",
    "get_planning_profile",
    "list_planning_profiles",
    "list_planning_profile_ids",
    "load_events",
    "observation_from_tool_result",
    "recorded_decisions",
    "resolve_planning_config",
    "resolve_required_planning_profile",
    "resolve_planning_stack",
]
