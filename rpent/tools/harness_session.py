"""Agent-facing tool definitions for the authoritative Harness session."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rpent.harness.evidence import EvidenceResolver
from rpent.harness.session import HarnessSession

LEGACY_CONTROL_TOOLS = frozenset(
    {
        "create_plan",
        "register_plan",
        "view_plan",
        "replan",
        "request_replan",
        "select_outcome",
        "advance_stage",
    }
)


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        input_schema["required"] = list(required)
    return {"name": name, "description": description, "input_schema": input_schema}


_STRINGS = {"type": "array", "items": {"type": "string"}}

SESSION_TOOL_SPECS = (
    _schema(
        "start_instruction",
        "Compile and start the immutable outcome contract for this instruction.",
        {"goal": {"type": "string"}, "observations": _STRINGS, "constraints": _STRINGS},
        ("goal", "observations"),
    ),
    _schema("execution_status", "Read the authoritative Harness lifecycle status."),
    _schema(
        "submit_observation",
        "Submit run-owned evidence for predicate verification or stage advancement.",
        {"observation_refs": _STRINGS},
        ("observation_refs",),
    ),
    _schema(
        "verify_step",
        "Submit a conservative manual step verdict; it cannot authorize success.",
        {
            "verdict": {
                "type": "string",
                "enum": ["pass", "fail", "retry", "uncertain"],
            },
            "evidence": _STRINGS,
            "summary": {"type": "string"},
            "execution_id": {"type": "string"},
        },
        ("verdict", "evidence", "summary"),
    ),
    _schema(
        "verify_observation",
        "Submit a conservative manual observation verdict.",
        {
            "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            "evidence": _STRINGS,
            "summary": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        ("verdict", "evidence", "summary"),
    ),
    _schema(
        "verify_outcomes",
        "Authoritatively verify all pending durable or terminal outcomes using "
        "evidence from a new read-only observation collected after "
        "submit_observation. Content-addressed refs may repeat only when the new "
        "observation gives them a later capture time.",
        {"outcome_ids": _STRINGS, "observation_refs": _STRINGS},
        ("outcome_ids", "observation_refs"),
    ),
    _schema(
        "request_exception",
        "Request Harness-selected recovery for the active outcome.",
        {"reason": {"type": "string"}, "evidence_refs": _STRINGS},
        ("reason", "evidence_refs"),
    ),
    _schema(
        "apply_recovery",
        "Apply the recovery scope selected by the Harness.",
        {
            "reason": {"type": "string"},
            "observations": _STRINGS,
            "constraints": _STRINGS,
        },
        ("reason", "observations"),
    ),
    _schema(
        "finish",
        "End the run only after the Harness reaches a terminal lifecycle state.",
        {
            "status": {"type": "string", "enum": ["success", "failure"]},
            "summary": {"type": "string"},
        },
        ("status", "summary"),
    ),
)

SESSION_TOOL_NAMES = frozenset(spec["name"] for spec in SESSION_TOOL_SPECS)


def tool_definitions(
    session: HarnessSession,
    resolver: EvidenceResolver,
    *,
    trusted_evidence_paths: set[Path] | None = None,
    trusted_evidence_captures: dict[Path, str] | None = None,
) -> tuple[tuple[str, dict[str, Any], Callable[..., dict[str, Any]]], ...]:
    """Bind minimal static schemas to one session and evidence root."""

    def resolve_trusted(observation_refs: list[str]):
        resolved = resolver.resolve_many(observation_refs)
        trusted = trusted_evidence_paths or set()
        if any(item.path not in trusted for item in resolved):
            raise ValueError(
                "only artifacts registered from environment tool output can "
                "verify outcomes"
            )
        return resolved

    def observed_at(resolved) -> list[str]:
        captures = trusted_evidence_captures or {}
        return [captures.get(item.path, item.captured_at) for item in resolved]

    def submit_observation(observation_refs: list[str], **_: Any) -> dict[str, Any]:
        status = session.execution_status()
        if status.get("state") == "observation_required":
            return session.submit_observation(observation_refs=list(observation_refs))
        resolved = resolve_trusted(observation_refs)
        return session.submit_observation(
            observation_refs=list(observation_refs),
            observed_at=observed_at(resolved),
        )

    def verify_outcomes(
        outcome_ids: list[str], observation_refs: list[str], **_: Any
    ) -> dict[str, Any]:
        resolved = resolve_trusted(observation_refs)
        return session.verify_outcomes(
            outcome_ids=list(outcome_ids),
            observation_refs=list(observation_refs),
            observed_at=observed_at(resolved),
        )

    def finish(status: str, summary: str) -> dict[str, Any]:
        result = session.finish(status, summary)
        if result.get("ok") is True:
            return {**result, "_finish": True}
        return result

    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "start_instruction": session.start_instruction,
        "execution_status": session.execution_status,
        "submit_observation": submit_observation,
        "verify_step": session.verify_step,
        "verify_observation": session.verify_observation,
        "verify_outcomes": verify_outcomes,
        "request_exception": session.request_exception,
        "apply_recovery": session.apply_recovery,
        "finish": finish,
    }
    return tuple(
        (spec["name"], spec, handlers[spec["name"]]) for spec in SESSION_TOOL_SPECS
    )


__all__ = [
    "LEGACY_CONTROL_TOOLS",
    "SESSION_TOOL_NAMES",
    "SESSION_TOOL_SPECS",
    "tool_definitions",
]
