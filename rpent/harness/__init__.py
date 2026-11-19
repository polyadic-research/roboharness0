"""Runtime planning and verification harness for physical-agent tool calls."""

from rpent.harness.controller import HarnessController
from rpent.harness.gate import GateDecision, PlanSafetyGate
from rpent.harness.hooks import (
    FinishGuardHook,
    HarnessHookPipeline,
    StructuredFeedbackVerifierHook,
    ToolHookContext,
)
from rpent.harness.session import HarnessSession
from rpent.harness.state import PlanStateMachine

# Short canonical names. The longer names remain as compatibility aliases.
Harness = HarnessController
PlanGate = PlanSafetyGate
PlanState = PlanStateMachine

__all__ = [
    "GateDecision",
    "FinishGuardHook",
    "Harness",
    "HarnessController",
    "HarnessSession",
    "HarnessHookPipeline",
    "PlanGate",
    "PlanSafetyGate",
    "PlanState",
    "PlanStateMachine",
    "StructuredFeedbackVerifierHook",
    "ToolHookContext",
]
