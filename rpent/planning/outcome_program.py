"""Versioned immutable representation compiled from a user instruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from rpent.planning.coverage import validate_outcome_coverage

_OUTCOME_LIFECYCLES = {"event", "persistent", "terminal"}


@dataclass(frozen=True)
class CompileDiagnostics:
    ambiguous_clauses: tuple[str, ...] = ()
    unsupported_clauses: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class CompiledOutcome:
    id: str
    instruction_index: int
    source_text: str
    description: str
    success_predicate: str
    depends_on: tuple[str, ...]
    lifecycle: str


@dataclass(frozen=True)
class CompiledStage:
    id: str
    intent: str
    outcome_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutcomeProgram:
    schema_version: str
    contract_id: str
    source_instruction: str
    outcomes: tuple[CompiledOutcome, ...]
    stages: tuple[CompiledStage, ...]
    compile_diagnostics: CompileDiagnostics

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def compile_outcome_program(
    source_instruction: str,
    outcomes: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    *,
    explicit_contract: list[dict[str, Any]] | None = None,
) -> OutcomeProgram:
    """Validate and freeze planner output as an immutable outcome contract."""
    source = str(source_instruction).strip()
    if not source or not outcomes:
        raise ValueError("instruction and outcomes must be non-empty")

    expected_ids = tuple(f"outcome-{index}" for index in range(1, len(outcomes) + 1))
    actual_ids = tuple(str(item.get("id", "")).strip() for item in outcomes)
    actual_indices = tuple(int(item.get("instruction_index", -1)) for item in outcomes)
    if actual_ids != expected_ids or actual_indices != tuple(
        range(1, len(outcomes) + 1)
    ):
        raise ValueError(
            "compiled outcomes require sequential outcome-N IDs and indices"
        )

    explicit = tuple(explicit_contract or ())
    if explicit and tuple(str(item.get("id", "")) for item in explicit) != actual_ids:
        raise ValueError("compiled outcomes do not preserve explicit outcome IDs")
    if explicit:
        for declared, generated in zip(explicit, outcomes, strict=True):
            if (
                int(declared.get("number", -1)) != int(generated["instruction_index"])
                or str(declared.get("text", "")).strip()
                != str(generated.get("description", "")).strip()
            ):
                raise ValueError(
                    "compiled outcomes must preserve explicit outcome text"
                )

    explicit_text = {str(item["id"]): str(item["text"]).strip() for item in explicit}
    compiled_outcomes = tuple(
        _compile_outcome(item, explicit_text=explicit_text) for item in outcomes
    )
    _validate_dependencies(compiled_outcomes)

    compiled_stages = tuple(_compile_stage(item) for item in stages)
    stage_ids = tuple(stage.id for stage in compiled_stages)
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("compiled stage ids must be unique")
    try:
        validate_outcome_coverage(
            tuple(stage.outcome_ids for stage in compiled_stages), actual_ids
        )
    except ValueError as exc:
        if "unknown IDs" in str(exc):
            raise ValueError(
                f"compiled stages reference an unknown outcome: {exc}"
            ) from exc
        raise ValueError(
            "compiled stages must cover each original outcome exactly once and in order: "
            f"{exc}"
        ) from exc

    diagnostics = CompileDiagnostics(confidence=1.0 if explicit else None)
    canonical = {
        "schema_version": "outcome-program/v1",
        "source_instruction": source,
        "outcomes": [asdict(item) for item in compiled_outcomes],
        "stages": [asdict(item) for item in compiled_stages],
        "compile_diagnostics": asdict(diagnostics),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return OutcomeProgram(
        schema_version="outcome-program/v1",
        contract_id=f"contract-{digest}",
        source_instruction=source,
        outcomes=compiled_outcomes,
        stages=compiled_stages,
        compile_diagnostics=diagnostics,
    )


def _compile_outcome(
    item: dict[str, Any], *, explicit_text: dict[str, str]
) -> CompiledOutcome:
    outcome_id = str(item["id"]).strip()
    description = str(item.get("description", "")).strip()
    success_predicate = str(item.get("success_predicate", "")).strip()
    lifecycle = str(item.get("lifecycle", "")).strip()
    if not description:
        raise ValueError(f"outcome description must be non-empty: {outcome_id}")
    if not success_predicate:
        raise ValueError(f"outcome success_predicate must be non-empty: {outcome_id}")
    if lifecycle not in _OUTCOME_LIFECYCLES:
        raise ValueError(f"invalid outcome lifecycle: {lifecycle}")
    dependencies = tuple(str(value).strip() for value in item.get("depends_on", ()))
    if any(not value for value in dependencies) or len(dependencies) != len(
        set(dependencies)
    ):
        raise ValueError(
            f"outcome dependencies must be unique non-empty IDs: {outcome_id}"
        )
    return CompiledOutcome(
        id=outcome_id,
        instruction_index=int(item["instruction_index"]),
        source_text=explicit_text.get(outcome_id, description),
        description=description,
        success_predicate=success_predicate,
        depends_on=dependencies,
        lifecycle=lifecycle,
    )


def _compile_stage(item: dict[str, Any]) -> CompiledStage:
    stage_id = str(item.get("id", "")).strip()
    intent = str(item.get("intent", "")).strip()
    if not stage_id:
        raise ValueError("compiled stage id must be non-empty")
    if not intent:
        raise ValueError(f"compiled stage intent must be non-empty: {stage_id}")
    return CompiledStage(
        id=stage_id,
        intent=intent,
        outcome_ids=tuple(str(value).strip() for value in item.get("outcome_ids", ())),
    )


def _validate_dependencies(outcomes: tuple[CompiledOutcome, ...]) -> None:
    known = {item.id for item in outcomes}
    for item in outcomes:
        unknown = sorted(set(item.depends_on) - known)
        if unknown:
            raise ValueError(f"unknown dependency for {item.id}: {unknown}")

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {item.id: item for item in outcomes}

    def visit(outcome_id: str) -> None:
        if outcome_id in visiting:
            raise ValueError("outcome dependency graph contains a cycle")
        if outcome_id in visited:
            return
        visiting.add(outcome_id)
        for dependency_id in by_id[outcome_id].depends_on:
            visit(dependency_id)
        visiting.remove(outcome_id)
        visited.add(outcome_id)

    for outcome in outcomes:
        visit(outcome.id)


__all__ = [
    "CompileDiagnostics",
    "CompiledOutcome",
    "CompiledStage",
    "OutcomeProgram",
    "compile_outcome_program",
]
