"""Read recorded planning events as offline decision examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordedDecision:
    """One proposed tool call joined with its eventual normalized feedback."""

    execution_id: str
    sequence: int
    tool_name: str
    physical: bool
    tool_args: dict[str, Any]
    feedback: dict[str, Any] | None


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load valid JSON object events, tolerating a partial final JSONL line."""
    events: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def recorded_decisions(events: list[dict[str, Any]]) -> list[RecordedDecision]:
    """Join ``tool_call_proposed`` events with feedback by execution id."""
    feedback_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "action_feedback":
            feedback = event.get("feedback")
            if isinstance(feedback, dict) and feedback.get("execution_id"):
                feedback_by_id[str(feedback["execution_id"])] = feedback
        elif event.get("event") == "tool_execution_finished":
            feedback = event.get("feedback")
            if isinstance(feedback, dict) and feedback.get("execution_id"):
                feedback_by_id[str(feedback["execution_id"])] = feedback

    decisions: list[RecordedDecision] = []
    for event in events:
        if event.get("event") != "tool_call_proposed":
            continue
        execution_id = str(event.get("execution_id", ""))
        if not execution_id:
            continue
        args = event.get("args")
        decisions.append(
            RecordedDecision(
                execution_id=execution_id,
                sequence=int(event.get("sequence", 0)),
                tool_name=str(event.get("tool", "")),
                physical=bool(event.get("physical", False)),
                tool_args=dict(args) if isinstance(args, dict) else {},
                feedback=feedback_by_id.get(execution_id),
            )
        )
    return decisions
