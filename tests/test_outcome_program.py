"""Tests for the immutable instruction-to-outcome compilation format."""

import dataclasses

import pytest

from rpent.planning.outcome_program import compile_outcome_program


def _outcomes() -> list[dict[str, object]]:
    return [
        {
            "id": "outcome-1",
            "instruction_index": 1,
            "description": "cup remains in drawer",
            "success_predicate": "cup is visibly inside the drawer",
            "depends_on": [],
            "lifecycle": "terminal",
        }
    ]


def _stages() -> list[dict[str, object]]:
    return [{"id": "stage-1", "intent": "place cup", "outcome_ids": ["outcome-1"]}]


def test_compiles_versioned_stable_immutable_program() -> None:
    first = compile_outcome_program(
        "place the cup in the drawer", _outcomes(), _stages()
    )
    second = compile_outcome_program(
        "place the cup in the drawer", _outcomes(), _stages()
    )

    assert first.schema_version == "outcome-program/v1"
    assert first.contract_id == second.contract_id
    assert first.source_instruction == "place the cup in the drawer"
    assert first.outcomes[0].source_text == "cup remains in drawer"
    assert first.compile_diagnostics.confidence is None
    assert isinstance(first.outcomes, tuple)
    assert isinstance(first.stages, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.contract_id = "rewritten"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.outcomes[0].description = "rewritten"  # type: ignore[misc]


def test_preserves_explicit_numbered_source_text_and_certain_diagnostics() -> None:
    outcomes = _outcomes()
    outcomes[0]["description"] = "open drawer"
    outcomes[0]["success_predicate"] = "drawer is visibly open"

    program = compile_outcome_program(
        "1. open drawer",
        outcomes,
        _stages(),
        explicit_contract=[{"id": "outcome-1", "number": 1, "text": "open drawer"}],
    )

    assert program.outcomes[0].source_text == "open drawer"
    assert program.compile_diagnostics.confidence == 1.0
    assert program.compile_diagnostics.ambiguous_clauses == ()


def test_rejects_rewritten_explicit_clause() -> None:
    with pytest.raises(ValueError, match="preserve explicit outcome text"):
        compile_outcome_program(
            "1. open drawer",
            _outcomes(),
            _stages(),
            explicit_contract=[{"id": "outcome-1", "number": 1, "text": "open drawer"}],
        )


def test_rejects_non_sequential_natural_contract() -> None:
    outcomes = _outcomes()
    outcomes[0]["id"] = "result-a"

    with pytest.raises(ValueError, match="sequential outcome-N"):
        compile_outcome_program("place cup", outcomes, _stages())


@pytest.mark.parametrize(
    "stages, message",
    [
        ([], "exactly once"),
        ([{"id": "stage-1", "intent": "place", "outcome_ids": []}], "exactly once"),
        (
            [
                {"id": "stage-1", "intent": "place", "outcome_ids": ["outcome-1"]},
                {"id": "stage-2", "intent": "repeat", "outcome_ids": ["outcome-1"]},
            ],
            "exactly once",
        ),
        (
            [{"id": "stage-1", "intent": "invent", "outcome_ids": ["outcome-9"]}],
            "unknown outcome",
        ),
    ],
)
def test_stage_partition_covers_each_original_outcome_exactly_once(
    stages: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_outcome_program("place cup", _outcomes(), stages)


def test_compilation_does_not_alias_mutable_input_payloads() -> None:
    outcomes = _outcomes()
    stages = _stages()
    program = compile_outcome_program("place cup", outcomes, stages)

    outcomes[0]["description"] = "silently replaced"
    stages[0]["outcome_ids"] = []

    assert program.outcomes[0].description == "cup remains in drawer"
    assert program.stages[0].outcome_ids == ("outcome-1",)
