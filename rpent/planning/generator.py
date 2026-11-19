"""Provider-independent contract for generating candidate execution plans."""

from __future__ import annotations

from typing import Any, Protocol


class PlanGenerator(Protocol):
    """Generate a grounded candidate plan without activating it."""

    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
        object_states: list[dict[str, Any]] | None = None,
        information_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a strategy and steps, or enabled public-information questions."""
        ...

    def generate_replan(
        self,
        *,
        goal: str,
        reason: str,
        observations: list[str],
        constraints: list[str],
        current_plan: dict[str, Any],
        physical_tools: set[str],
        object_states: list[dict[str, Any]] | None = None,
        information_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return an unfinished-plan replacement or enabled information questions."""
        ...


__all__ = ["PlanGenerator"]
