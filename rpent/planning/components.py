"""Compatibility layer for the former two-stage component resolver."""

from rpent.planning.stack import PlanningStack

ResolvedPlanningComponents = PlanningStack


def resolve_planning_components(config: PlanningStack) -> PlanningStack:
    """Return the already-resolved stack unchanged."""
    return config


__all__ = ["ResolvedPlanningComponents", "resolve_planning_components"]
