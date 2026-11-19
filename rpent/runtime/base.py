"""Small public contract for episode-owning LLM runtimes."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rpent.tools.toolkit import Toolkit

MCP_TOOL_PREFIX = "mcp__rpent__"


def add_mcp_prefix(name: str) -> str:
    """Return the namespaced MCP name for a bare RPent tool."""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """Return a bare RPent tool name."""
    return name.removeprefix(MCP_TOOL_PREFIX)


class RuntimeResult:
    """Serializable result from an episode-owning runtime."""

    __slots__ = ("finish_result", "messages", "stats", "error", "stop_reason")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
        stop_reason: str | None = None,
    ) -> None:
        self.finish_result = finish_result
        self.messages = messages or []
        self.stats = stats or {}
        self.error = error
        if stop_reason is None:
            stop_reason = self.stats.get("stop_reason")
        elif stop_reason:
            self.stats["stop_reason"] = stop_reason
        self.stop_reason = stop_reason


class AgentRuntime(Protocol):
    """Own a complete model/tool episode."""

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int | None,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> RuntimeResult:
        """Run until finish, error, or the turn budget is exhausted."""
        ...


# Compatibility name used by existing integrations and transcript tests.
PlannerResult = RuntimeResult


__all__ = [
    "AgentRuntime",
    "MCP_TOOL_PREFIX",
    "PlannerResult",
    "RuntimeResult",
    "add_mcp_prefix",
    "strip_mcp_prefix",
]
