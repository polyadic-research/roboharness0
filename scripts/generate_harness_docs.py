#!/usr/bin/env python3
"""Generate and verify documentation from the Harness lifecycle contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from rpent.harness.lifecycle import (
    ALLOWED_OPERATIONS,
    COMPLETED_STEP_REVIEW_GUIDANCE,
    EPISODE_LIVENESS_GUIDANCE,
    FINISHING_RESERVE_GUIDANCE,
    PHYSICAL_ATTEMPT_BUDGET_GUIDANCE,
    PLANNING_STEP_GRANULARITY,
    POLICY_FAILURE_TAKEOVER_GUIDANCE,
    TASK_ACTION_REVIEW_GUIDANCE,
    TERMINAL_STATES,
    TRANSITIONS,
    HarnessLifecycleState,
    allowed_operations,
)

DOC_ROOT = Path("docs/harness")
GENERATED_NOTICE = "<!-- GENERATED FILE: edit rpent/harness/lifecycle.py instead. -->"


def render_mermaid() -> str:
    """Render the declared lifecycle as deterministic Mermaid source."""
    lines = [
        "stateDiagram-v2",
        "    [*] --> action_required: successful start_instruction",
    ]
    for source in HarnessLifecycleState:
        for event, target in TRANSITIONS[source].items():
            lines.append(f"    {source.value} --> {target.value}: {event.value}")
    for state in sorted(TERMINAL_STATES, key=lambda item: item.value):
        lines.append(f"    {state.value} --> [*]")
    return "\n".join(lines) + "\n"


def render_lifecycle_markdown() -> str:
    """Render the human-readable lifecycle reference."""
    lines = [
        GENERATED_NOTICE,
        "",
        "# Harness Lifecycle Reference",
        "",
        "This file is generated from `rpent/harness/lifecycle.py`, the single",
        "source of truth for the ChatGPT-visible lifecycle.",
        "",
        "## Planning granularity and step attempt budget",
        "",
        PLANNING_STEP_GRANULARITY,
        "",
        PHYSICAL_ATTEMPT_BUDGET_GUIDANCE,
        "",
        "Only completed policy-tool calls paused for pre-close, pre-release or semantic target review",
        "with consistent zero-execution and cancelled-action counters receive",
        "the attempt exemption. Verification remains pending; no outcome is",
        "completed and no physical continuation is authorized by the pause.",
        "",
        "## Bounded takeover finishing reserve",
        "",
        FINISHING_RESERVE_GUIDANCE,
        "",
        "## Episode liveness and verified retries",
        "",
        EPISODE_LIVENESS_GUIDANCE,
        "",
        "## First VLA failure and LLM takeover",
        "",
        POLICY_FAILURE_TAKEOVER_GUIDANCE,
        "",
        "## Explicit task-target conflict review",
        "",
        TASK_ACTION_REVIEW_GUIDANCE,
        "",
        "## Legacy environment-horizon completion review",
        "",
        COMPLETED_STEP_REVIEW_GUIDANCE,
        "",
        "This read-only gate uses the existing plan/replan mode. It does not add",
        "a strict HarnessSession state, predicate authority, or recovery scope.",
        "",
        "## Bootstrap boundary",
        "",
        "`start_instruction` is a one-shot pre-lifecycle bootstrap. The session is",
        "unstarted (`None`) until successful plan registration assigns the initial",
        "`action_required` state. It is not an allowed lifecycle operation and is",
        "rejected after the session has started.",
        "",
        "## States and allowed operations",
        "",
        "| State | Terminal | Allowed operations |",
        "|---|---:|---|",
    ]
    for state in HarnessLifecycleState:
        operations = ", ".join(f"`{item}`" for item in ALLOWED_OPERATIONS[state])
        terminal = "yes" if state in TERMINAL_STATES else "no"
        lines.append(f"| `{state.value}` | {terminal} | {operations} |")
    lines.extend(
        [
            "",
            "## Conditional recovery guard",
            "",
            "After a non-terminal `apply_recovery` consumes a recovery scope,",
            "`action_required` must attempt a configured physical operation before",
            "another `request_exception` can consume a higher recovery scope. While",
            "that action is pending, the allowed operations are: "
            + ", ".join(
                f"`{item}`"
                for item in allowed_operations(
                    HarnessLifecycleState.ACTION_REQUIRED,
                    recovery_action_pending=True,
                )
            )
            + ".",
            "",
            "## Legal transitions",
            "",
            "| Source | Event | Target |",
            "|---|---|---|",
        ]
    )
    for source in HarnessLifecycleState:
        for event, target in TRANSITIONS[source].items():
            lines.append(f"| `{source.value}` | `{event.value}` | `{target.value}` |")
    lines.extend(
        ["", "## Diagram", "", "```mermaid", render_mermaid().rstrip(), "```", ""]
    )
    return "\n".join(lines)


GENERATED_DOCUMENTS = {
    Path("lifecycle.md"): render_lifecycle_markdown(),
    Path("lifecycle.mmd"): GENERATED_NOTICE + "\n" + render_mermaid(),
}


def documents_are_current(root: Path, documents: Mapping[Path, str]) -> bool:
    """Return whether every generated artifact exactly matches its source."""
    for relative_path, expected in documents.items():
        try:
            actual = (root / relative_path).read_text(encoding="utf-8")
        except OSError:
            return False
        if actual != expected:
            return False
    return True


def write_documents(root: Path = DOC_ROOT) -> None:
    """Write all generated artifacts."""
    root.mkdir(parents=True, exist_ok=True)
    for relative_path, content in GENERATED_DOCUMENTS.items():
        (root / relative_path).write_text(content, encoding="utf-8")


def main() -> int:
    """Write generated documents or verify that they are current."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_documents()
        return 0
    if documents_are_current(DOC_ROOT, GENERATED_DOCUMENTS):
        return 0
    print("Harness lifecycle documentation is stale; run with --write")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
