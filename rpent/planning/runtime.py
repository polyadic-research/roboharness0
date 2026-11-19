"""Stable orchestration boundary around current and future planning backends."""

from __future__ import annotations

import queue

from rpent.planning.events import EventRecorder, NullEventRecorder
from rpent.runtime import AgentRuntime, PlannerResult
from rpent.tools.toolkit import Toolkit


class PlanningOrchestrator:
    """Record lifecycle events while delegating provider-specific agent I/O."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        recorder: EventRecorder | None = None,
    ) -> None:
        self.runtime = runtime
        self.recorder = recorder or NullEventRecorder()

    def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        """Run one episode through a backend and emit stable lifecycle events."""
        self.recorder.record(
            "planning_started",
            max_turns=max_turns,
            interactive=input_queue is not None,
            instruction=user_message,
        )
        try:
            result = self.runtime.solve(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        except Exception as exc:
            self.recorder.record(
                "planning_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.recorder.record(
            "planning_finished",
            finish=result.finish_result,
            stats=result.stats,
            error=result.error,
            **_staged_contract_metadata(result.finish_result),
        )
        return result


def _staged_contract_metadata(finish_result: object) -> dict[str, object]:
    """Extract stable contract identity for traces without changing results."""
    if not isinstance(finish_result, dict):
        return {}
    program = finish_result.get("outcome_program")
    if not isinstance(program, dict):
        return {}
    stages = finish_result.get("stages")
    stage_ids = (
        [str(item.get("id", "")) for item in stages if isinstance(item, dict)]
        if isinstance(stages, list)
        else []
    )
    required = finish_result.get("required_outcome_ids")
    return {
        "contract_id": program.get("contract_id"),
        "outcome_program_schema_version": program.get("schema_version"),
        "required_outcome_ids": list(required) if isinstance(required, list) else [],
        "stage_ids": stage_ids,
    }
