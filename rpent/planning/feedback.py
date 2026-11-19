"""Normalize raw toolkit outcomes into stable action feedback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rpent.planning.observation import (
    environment_status_from_tool_result,
    observation_from_tool_result,
)
from rpent.planning.outcomes import (
    UNKNOWN_OUTCOME_ADAPTER,
    SkillOutcomeAdapter,
)
from rpent.planning.schemas import (
    ActionFeedback,
    ExecutionStatus,
    FailureType,
    SkillOutcome,
)


def action_feedback_from_result(
    *,
    execution_id: str,
    tool_name: str,
    physical: bool,
    result: Mapping[str, Any],
    rejected: bool = False,
    outcome_adapter: SkillOutcomeAdapter = UNKNOWN_OUTCOME_ADAPTER,
) -> ActionFeedback:
    """Create environment-independent feedback from a raw tool result."""
    error = result.get("error")
    status: ExecutionStatus
    failure_type: FailureType | None
    if rejected:
        status = "rejected"
        failure_type = "policy_rejected"
    elif error and "timeout" in str(error).lower():
        status = "timeout"
        failure_type = "timeout"
    elif error:
        status = "controller_error"
        failure_type = "controller_error"
    else:
        status = "completed"
        failure_type = None

    observation = observation_from_tool_result(result)
    environment_success, environment_terminated, environment_truncated = (
        environment_status_from_tool_result(result)
    )

    adapter_error = None
    if status != "completed":
        outcome = SkillOutcome(
            status="failure",
            failure_type=failure_type,
            retryable=False,
            evidence=(f"execution_status={status}", f"error={error or 'unknown'}"),
            source="tool-signal",
        )
    elif not physical:
        outcome = SkillOutcome(status="not_applicable", source="unknown")
    else:
        try:
            outcome = outcome_adapter.classify(
                tool_name=tool_name,
                result=result,
                observation=observation,
            )
        except Exception as exc:
            adapter_error = f"{type(exc).__name__}: {exc}"
            outcome = SkillOutcome(status="uncertain", source="unknown")

    return ActionFeedback(
        execution_id=execution_id,
        tool_name=tool_name,
        physical=physical,
        execution_status=status,
        outcome=outcome,
        observation=observation,
        error=str(error) if error else None,
        failure_type=failure_type,
        environment_success=environment_success,
        environment_terminated=environment_terminated,
        environment_truncated=environment_truncated,
        measurements=_numeric_measurements(result),
        raw_result=dict(result),
        outcome_adapter_error=adapter_error,
    )


def _numeric_measurements(result: Mapping[str, Any]) -> dict[str, float]:
    """Keep scalar diagnostic values without guessing their semantics."""
    measurements: dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            measurements[str(key)] = float(value)
    return measurements
