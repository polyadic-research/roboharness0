"""Conservative physical-outcome classification for RoboCasa primitives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rpent.planning.schemas import ObservationSummary, SkillOutcome


class RoboCasaSkillOutcomeAdapter:
    """Expose local progress without replacing official task success."""

    id = "robocasa-skill-outcome-v2"

    def classify(
        self,
        *,
        tool_name: str,
        result: Mapping[str, Any],
        observation: ObservationSummary | None,
    ) -> SkillOutcome:
        del observation
        environment = result.get("environment")
        if isinstance(environment, Mapping) and environment.get("success") is True:
            return SkillOutcome(
                status="success",
                evidence=("RoboCasa environment.success=true",),
                source="tool-signal",
            )
        events = result.get("grasp_events", ())
        if result.get("stop_reason") in {"task_conflict_review_required", "prerelease_review_required"}:
            return SkillOutcome(status="uncertain", source="adapter", evidence=(
                "policy action paused before execution for public target review; not a failed grasp",
            ))
        if any(isinstance(event, Mapping) and event.get("type") == "grasp_lost" for event in events):
            return SkillOutcome(
                status="uncertain",
                evidence=(
                    "grasp_lost: previously confirmed grasp width collapsed or changed abruptly",
                    "execution interrupted; review current images and release_commanded before continuing",
                    "proprioception does not establish object identity or a confirmed object drop",
                ),
                source="adapter",
            )
        if tool_name != "smolvla_pick":
            return SkillOutcome(status="uncertain", source="unknown")
        if result.get("stop_reason") == "preclose_review_required":
            return SkillOutcome(
                status="progress",
                evidence=("grasp paused before closure for a fresh wrist-camera alignment review",),
                source="adapter",
            )
        if result.get("success") is True:
            return SkillOutcome(
                status="progress",
                progress=0.5,
                evidence=(
                    "ordered end-effector descent, gripper contact width, and lift passed",
                    "target identity and stable retention still require observation",
                ),
                source="adapter",
            )
        reason = result.get("failure_reason")
        evidence = (
            "gripper closed empty after descent"
            if reason == "closed_empty"
            else f"grasp evidence incomplete: {reason or 'unknown'}"
        )
        return SkillOutcome(
            status="retryable_failure",
            failure_type="grasp_missed",
            retryable=True,
            evidence=(evidence,),
            source="adapter",
        )


__all__ = ["RoboCasaSkillOutcomeAdapter"]
