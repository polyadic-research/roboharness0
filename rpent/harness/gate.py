"""Pure plan/safety gate for physical tool decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rpent.harness.state import PlanStateMachine


@dataclass(frozen=True)
class GateDecision:
    """Result of checking one proposed tool call."""

    allowed: bool
    reason: str | None = None
    warning: str | None = None


class PlanSafetyGate:
    """Check plan protocol without mutating plan state or writing trace files."""

    def __init__(self, *, mode: str) -> None:
        if mode not in {"trace", "plan-gated"}:
            raise ValueError(f"unsupported harness mode: {mode!r}")
        self.mode = mode

    def evaluate(
        self,
        *,
        name: str,
        input_dict: dict[str, Any],
        state: PlanStateMachine,
        control_tools: set[str],
    ) -> GateDecision:
        """Allow, warn, or reject one proposed tool call."""
        if name in control_tools or name not in state.physical_tools:
            return GateDecision(allowed=True)

        reason = self._physical_violation(name, state)
        if reason is None:
            return GateDecision(allowed=True)
        if self.mode == "trace":
            return GateDecision(allowed=True, reason=reason, warning=reason)
        return GateDecision(allowed=False, reason=reason)

    @staticmethod
    def _physical_violation(
        name: str,
        state: PlanStateMachine,
    ) -> str | None:
        if not state.has_plan:
            return (
                "physical action requires create_plan followed by register_plan first"
            )
        if state.awaiting_verification:
            return "verify the previous physical action before another physical action"
        if state.needs_replan:
            return "failed/uncertain verification requires replan before acting"
        active = state.active_step
        if active is None:
            return "plan is complete; no further physical actions are allowed"
        if name not in active.allowed_tools:
            return f"tool {name!r} is not allowed by active plan step {active.id!r}"
        return state.finishing_tool_violation(name)
