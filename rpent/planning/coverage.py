"""Planner-only validation for explicit outcome coverage metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def parse_outcome_ids(value: object, *, field: str) -> tuple[str, ...]:
    """Parse a canonical JSON string array without silently normalizing it."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a canonical string array")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate outcome IDs")
    return tuple(value)


def validate_outcome_coverage(
    step_coverage: Sequence[Sequence[str]],
    required_outcome_ids: Sequence[str],
) -> None:
    """Validate that grouped coverage is exact, unique, and ordered."""
    required = tuple(required_outcome_ids)
    flattened = tuple(item for group in step_coverage for item in group)
    unknown = [item for item in flattened if item not in required]
    if unknown:
        raise ValueError(f"outcome coverage contains unknown IDs: {unknown}")

    duplicates = [
        item for index, item in enumerate(flattened) if item in flattened[:index]
    ]
    if duplicates:
        raise ValueError(f"outcome coverage contains duplicate IDs: {duplicates}")

    missing = [item for item in required if item not in flattened]
    if missing:
        raise ValueError(f"outcome coverage is missing required IDs: {missing}")
    if flattened != required:
        raise ValueError(
            "outcome coverage order does not match required_outcome_ids: "
            f"expected {list(required)}, got {list(flattened)}"
        )


def parse_step_outcome_coverage(
    steps: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    """Parse optional per-step ``covers_outcomes`` string arrays."""
    grouped: list[tuple[str, ...]] = []
    for index, step in enumerate(steps, start=1):
        value = step.get("covers_outcomes", [])
        grouped.append(parse_outcome_ids(value, field=f"step {index} covers_outcomes"))
    return tuple(grouped)


__all__ = [
    "parse_outcome_ids",
    "parse_step_outcome_coverage",
    "validate_outcome_coverage",
]
