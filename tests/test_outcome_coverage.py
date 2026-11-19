"""Tests for planner-only outcome coverage validation."""

from __future__ import annotations

import pytest

from rpent.planning.coverage import (
    parse_step_outcome_coverage,
    validate_outcome_coverage,
)


def test_parses_step_outcome_coverage_as_grouped_string_arrays() -> None:
    assert parse_step_outcome_coverage(
        [
            {"covers_outcomes": ["outcome-1", "outcome-2"]},
            {"covers_outcomes": ["outcome-3"]},
        ]
    ) == (("outcome-1", "outcome-2"), ("outcome-3",))


def test_accepts_grouped_exact_coverage_in_required_order() -> None:
    coverage = (
        ("outcome-1", "outcome-2"),
        ("outcome-3",),
    )

    validate_outcome_coverage(
        coverage,
        ("outcome-1", "outcome-2", "outcome-3"),
    )


@pytest.mark.parametrize(
    ("coverage", "message"),
    [
        ((("outcome-1", "outcome-9"),), "unknown"),
        ((("outcome-1",), ("outcome-1", "outcome-2")), "duplicate"),
        ((("outcome-1",),), "missing"),
        ((("outcome-2", "outcome-1"),), "order"),
    ],
)
def test_rejects_invalid_flattened_coverage(
    coverage: tuple[tuple[str, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_outcome_coverage(coverage, ("outcome-1", "outcome-2"))


@pytest.mark.parametrize(
    "value",
    ["outcome-1", [1], [""], ["outcome-1", " outcome-2"]],
)
def test_rejects_noncanonical_string_arrays(value: object) -> None:
    with pytest.raises(ValueError, match="string array"):
        parse_step_outcome_coverage(
            [{"covers_outcomes": value}],
        )


def test_requires_coverage_field_when_contract_is_explicit() -> None:
    with pytest.raises(ValueError, match="missing"):
        coverage = parse_step_outcome_coverage([{}, {"covers_outcomes": ["outcome-2"]}])
        validate_outcome_coverage(
            coverage,
            ("outcome-1", "outcome-2"),
        )
