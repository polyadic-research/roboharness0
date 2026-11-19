"""Provider-independent interfaces for structured planning policies."""

from __future__ import annotations

from typing import Protocol

from rpent.planning.schemas import (
    ActionFeedback,
    ObservationSummary,
    Plan,
    PlanningDecision,
    TaskSpec,
)


class PlanningPolicy(Protocol):
    """Create, advance, and repair a robot plan using normalized contracts."""

    def create_plan(self, task: TaskSpec, observation: ObservationSummary) -> Plan:
        """Create a high-level plan from the initial grounded observation."""
        ...

    def decide(
        self,
        *,
        task: TaskSpec,
        observation: ObservationSummary,
        plan: Plan,
        feedback: ActionFeedback | None,
    ) -> PlanningDecision:
        """Choose one next action, replan request, or terminal decision."""
        ...

    def repair(
        self,
        *,
        task: TaskSpec,
        observation: ObservationSummary,
        plan: Plan,
        feedback: ActionFeedback,
    ) -> Plan:
        """Replace the unfinished portion of a plan after observed failure."""
        ...
