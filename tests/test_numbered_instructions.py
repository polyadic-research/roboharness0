"""Tests for pure numbered-instruction parsing."""

from __future__ import annotations

import pytest

from rpent.planning.instructions import (
    enumerated_atomic_count,
    parse_numbered_outcomes,
)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Steps: 1. open; 2. insert; 3. close", ["open", "insert", "close"]),
        ("1) open, 2) insert, 3) close", ["open", "insert", "close"]),
        ("(1) open; (2) insert; (3) close", ["open", "insert", "close"]),
        ("（1）打开；（2）放入；（3）关闭", ["打开", "放入", "关闭"]),
        ("第一步打开；第二步放入；第三步关闭", ["打开", "放入", "关闭"]),
    ],
)
def test_parses_supported_numbering_with_stable_identity(
    instruction: str,
    expected: list[str],
) -> None:
    outcomes = parse_numbered_outcomes(instruction)

    assert [outcome.id for outcome in outcomes] == [
        f"outcome-{index}" for index in range(1, len(expected) + 1)
    ]
    assert [outcome.number for outcome in outcomes] == list(range(1, len(expected) + 1))
    assert [outcome.text for outcome in outcomes] == expected
    assert enumerated_atomic_count(instruction) == len(expected)


def test_supports_multi_digit_arabic_numbers() -> None:
    instruction = "; ".join(f"{number}. step {number}" for number in range(1, 12))

    outcomes = parse_numbered_outcomes(instruction)

    assert outcomes[-1].id == "outcome-11"
    assert outcomes[-1].number == 11
    assert outcomes[-1].text == "step 11"


def test_parses_numbered_clauses_separated_by_sentence_periods() -> None:
    instruction = "1. open the drawer. 2. pick the cup. 3. close the drawer"

    outcomes = parse_numbered_outcomes(instruction)

    assert [outcome.text for outcome in outcomes] == [
        "open the drawer",
        "pick the cup",
        "close the drawer",
    ]


def test_supports_multi_digit_chinese_ordinals() -> None:
    numbers = (
        "一",
        "二",
        "三",
        "四",
        "五",
        "六",
        "七",
        "八",
        "九",
        "十",
        "十一",
    )
    instruction = "；".join(
        f"第{number}步 step {index}" for index, number in enumerate(numbers, start=1)
    )

    outcomes = parse_numbered_outcomes(instruction)

    assert outcomes[-1].id == "outcome-11"
    assert outcomes[-1].number == 11
    assert outcomes[-1].text == "step 11"


@pytest.mark.parametrize(
    "instruction",
    [
        "Put the bowl on the plate.",
        "1. Put the bowl on the plate.",
        "Use version 1.5, then continue.",
        "1.5, 2.5",
    ],
)
def test_does_not_parse_non_enumerated_prose(instruction: str) -> None:
    assert parse_numbered_outcomes(instruction) == ()
    assert enumerated_atomic_count(instruction) is None


def test_rejects_an_empty_numbered_clause() -> None:
    with pytest.raises(ValueError, match="outcome 2 is empty"):
        parse_numbered_outcomes("1. open; 2. ; 3. close")


@pytest.mark.parametrize(
    "instruction",
    [
        "1. one; 1. duplicate; 2. two",
        "1. one; 3. three",
        "1. one; 3. three; 2. two",
    ],
)
def test_rejects_non_sequential_numbering(instruction: str) -> None:
    with pytest.raises(ValueError, match="expected outcome 2"):
        parse_numbered_outcomes(instruction)
