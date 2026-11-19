"""Backend-independent terminal classification for planner episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

TerminalClassification = Literal["SUCCESS", "FAILED", "INCOMPLETE"]


@dataclass(frozen=True)
class TerminationResult:
    """Stable process-level interpretation of an agent runtime result."""

    classification: TerminalClassification
    error_code: str | None
    summary: str
    exit_code: int
    error: str | None = None


def classify_termination(
    *,
    finish_result: Mapping[str, Any] | None,
    error: str | None,
    harness_snapshot: Mapping[str, Any],
    stop_reason: str | None = None,
    require_environment_success: bool = False,
) -> TerminationResult:
    """Classify completion without inferring success from a clean process exit."""
    finish_status = str((finish_result or {}).get("status", "")).strip().lower()
    harness_state = str(
        harness_snapshot.get("state", harness_snapshot.get("status", ""))
    ).strip().lower()

    if harness_state == "terminal_failure":
        message = error or "Harness reached terminal_failure"
        return TerminationResult(
            "FAILED", "terminal_failure", message, 1, message
        )

    if stop_reason == "episode_budget_exhausted":
        message = error or "episode liveness budget exhausted"
        return TerminationResult(
            "FAILED", "episode_budget_exhausted", message, 1, message
        )

    if finish_status == "failure":
        message = error or "planner reported failure"
        return TerminationResult(
            "FAILED", "explicit_failure", message, 1, error
        )

    if error is not None:
        return TerminationResult(
            "FAILED",
            "runtime_error",
            error or "planner runtime failed",
            1,
            error,
        )

    if (
        finish_status == "success"
        and harness_snapshot.get("contract_success") is True
    ):
        if (
            require_environment_success
            and harness_snapshot.get("environment_success") is not True
        ):
            message = (
                "incomplete_termination: successful finish lacks authoritative "
                "environment success"
            )
            return TerminationResult(
                "INCOMPLETE",
                "environment_success_required",
                message,
                1,
                message,
            )
        return TerminationResult(
            "SUCCESS", None, "planner finished successfully", 0
        )

    if finish_status == "success":
        message = "incomplete_termination: successful finish lacks contract success"
    else:
        message = "incomplete_termination: planner ended without finish"
    summary = f"{message} (stop_reason={stop_reason})" if stop_reason else message
    return TerminationResult(
        "INCOMPLETE",
        "incomplete_termination",
        summary,
        1,
        message,
    )


__all__ = [
    "TerminalClassification",
    "TerminationResult",
    "classify_termination",
]
