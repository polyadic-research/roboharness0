"""Stable, presentation-ready summaries built from retained domain facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from rpent.ui.schemas import (
    ArtifactHealthView,
    AttemptSummaryView,
    CaseSummary,
    EvaluationMetricsView,
    FailureCountView,
    UsageView,
)

_TERMINAL_STATUSES = {
    "success",
    "terminal_failure",
    "truncated",
    "timeout",
    "error",
}
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "tool_calls",
)


def build_attempt_summary(
    result: Mapping[str, Any] | None,
    *,
    artifact_health: ArtifactHealthView | None,
    usage: Mapping[str, Any] | None,
    planner_elapsed_seconds: float | None = None,
    vla_chunks: int | None = None,
    retained_frame_count: int | None = None,
) -> AttemptSummaryView | None:
    """Combine one Attempt's retained result, usage, and artifact health."""
    if (
        result is None
        and usage is None
        and artifact_health is None
        and planner_elapsed_seconds is None
        and vla_chunks is None
        and retained_frame_count is None
    ):
        return None
    status = _string(result, "status")
    started_at = _string(result, "started_at")
    finished_at = _string(result, "finished_at")
    return AttemptSummaryView(
        status=status,
        success=None if status is None else status == "success",
        step_count=_count(result, "step_count"),
        action_count=_count(result, "action_count"),
        duration_seconds=_duration(started_at, finished_at),
        planner_elapsed_seconds=planner_elapsed_seconds,
        vla_chunks=vla_chunks,
        retained_frame_count=retained_frame_count,
        started_at=started_at,
        finished_at=finished_at,
        error_type=_string(result, "error_type"),
        error_message=_string(result, "error_message"),
        usage=UsageView(**{name: _count(usage, name) for name in _USAGE_FIELDS}),
        artifact_health=artifact_health,
    )


def aggregate_evaluation(cases: Iterable[CaseSummary]) -> EvaluationMetricsView:
    """Aggregate Case projections without interpreting log text."""
    items = tuple(cases)
    completed = tuple(
        item
        for item in items
        if item.run_summary is not None
        and item.run_summary.status in _TERMINAL_STATUSES
    )
    successful = sum(item.run_summary.success is True for item in completed)
    durations = tuple(
        item.run_summary.duration_seconds
        for item in completed
        if item.run_summary.duration_seconds is not None
    )
    health = Counter(
        item.run_summary.artifact_health.status
        for item in items
        if item.run_summary is not None and item.run_summary.artifact_health is not None
    )
    failures = Counter(
        item.run_summary.error_type or item.run_summary.status or "unknown"
        for item in completed
        if item.run_summary.success is False
    )
    return EvaluationMetricsView(
        scheduled=len(items),
        completed=len(completed),
        successful=successful,
        running=sum(item.execution_state == "running" for item in items),
        success_rate=successful / len(completed) if completed else None,
        transitions=_sum_summary(items, "step_count"),
        mean_duration_seconds=sum(durations) / len(durations) if durations else None,
        input_tokens=_sum_usage(items, "input_tokens"),
        output_tokens=_sum_usage(items, "output_tokens"),
        tool_calls=_sum_usage(items, "tool_calls"),
        artifact_complete=health["complete"],
        artifact_degraded=health["degraded"] + health["partial"],
        artifact_unavailable=health["unavailable"],
        failures=tuple(
            FailureCountView(reason=reason, count=count)
            for reason, count in sorted(
                failures.items(), key=lambda item: (-item[1], item[0])
            )
        ),
    )


def _count(values: Mapping[str, Any] | None, name: str) -> int | None:
    if values is None:
        return None
    value = values.get(name)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _string(values: Mapping[str, Any] | None, name: str) -> str | None:
    if values is None:
        return None
    value = values.get(name)
    return value if isinstance(value, str) and value else None


def _duration(started_at: str | None, finished_at: str | None) -> float | None:
    if started_at is None or finished_at is None:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
        if started.utcoffset() is None or finished.utcoffset() is None:
            return None
        seconds = (finished - started).total_seconds()
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _sum_usage(cases: tuple[CaseSummary, ...], name: str) -> int | None:
    values = tuple(
        getattr(item.run_summary.usage, name)
        for item in cases
        if item.run_summary is not None
        and getattr(item.run_summary.usage, name) is not None
    )
    return sum(values) if values else None


def _sum_summary(cases: tuple[CaseSummary, ...], name: str) -> int | None:
    values = tuple(
        getattr(item.run_summary, name)
        for item in cases
        if item.run_summary is not None and getattr(item.run_summary, name) is not None
    )
    return sum(values) if values else None
