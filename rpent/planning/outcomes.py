"""Provider-independent physical outcome adapter seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from rpent.planning.schemas import ObservationSummary, SkillOutcome


class SkillOutcomeAdapter(Protocol):
    """Classify the physical outcome of a successfully executed skill."""

    id: str

    def classify(
        self,
        *,
        tool_name: str,
        result: Mapping[str, Any],
        observation: ObservationSummary | None,
    ) -> SkillOutcome: ...


class UnknownPhysicalOutcomeAdapter:
    """Conservatively avoid inferring physical success from tool execution."""

    id = "unknown-physical-outcome-v1"

    def classify(
        self,
        *,
        tool_name: str,
        result: Mapping[str, Any],
        observation: ObservationSummary | None,
    ) -> SkillOutcome:
        del tool_name, result, observation
        return SkillOutcome(status="uncertain", source="unknown")


UNKNOWN_OUTCOME_ADAPTER = UnknownPhysicalOutcomeAdapter()
