"""Public entrypoint for LLM agent runtimes."""

from rpent.runtime.base import (
    AgentRuntime,
    PlannerResult,
    RuntimeResult,
    add_mcp_prefix,
    strip_mcp_prefix,
)
from rpent.runtime.factory import (
    authoritative_runtime_options,
    build_agent_runtime,
    build_runtime,
)
from rpent.runtime.plan_generation import RuntimePlanGenerator
from rpent.runtime.predicate_verification import RuntimePredicateVerifier
from rpent.runtime.termination import TerminationResult, classify_termination

__all__ = [
    "AgentRuntime",
    "PlannerResult",
    "RuntimeResult",
    "RuntimePlanGenerator",
    "RuntimePredicateVerifier",
    "TerminationResult",
    "add_mcp_prefix",
    "authoritative_runtime_options",
    "build_agent_runtime",
    "build_runtime",
    "classify_termination",
    "strip_mcp_prefix",
]
