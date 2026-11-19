"""Pure helpers for explicitly numbered instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_NUMBERED_CLAUSE = re.compile(
    r"(?:^|[\n.,，；;:：])\s*"
    r"(?:"
    r"[（(](?P<wrapped>[1-9]\d*)[）)]"
    r"|(?P<arabic>[1-9]\d*)(?:[、):：]|\.(?!\d))"
    r"|第(?P<chinese>[一二三四五六七八九十]+)[步项]"
    r")\s*"
)


@dataclass(frozen=True)
class InstructionOutcome:
    """One explicitly numbered outcome from an instruction."""

    id: str
    number: int
    text: str


def parse_numbered_outcomes(instruction: str) -> tuple[InstructionOutcome, ...]:
    """Return continuous numbered clauses, or no outcomes for ordinary prose."""
    source = str(instruction)
    matches = list(_NUMBERED_CLAUSE.finditer(source))
    if len(matches) < 2:
        return ()

    outcomes: list[InstructionOutcome] = []
    for expected, match in enumerate(matches, start=1):
        raw_number = match.group("wrapped") or match.group("arabic")
        number = (
            int(raw_number)
            if raw_number is not None
            else _chinese_number_to_int(match.group("chinese"))
        )
        if number != expected:
            raise ValueError(
                "numbered instruction is not continuous: "
                f"expected outcome {expected}, got {number}"
            )

        end = matches[expected].start() if expected < len(matches) else len(source)
        text = source[match.end() : end].strip(" \t\r\n,，;；")
        if not text:
            raise ValueError(f"numbered instruction outcome {number} is empty")
        outcomes.append(
            InstructionOutcome(
                id=f"outcome-{number}",
                number=number,
                text=text,
            )
        )
    return tuple(outcomes)


def enumerated_atomic_count(instruction: str) -> int | None:
    """Return the clause count when an explicit multi-item enumeration exists."""
    count = len(parse_numbered_outcomes(instruction))
    return count or None


def _chinese_number_to_int(value: str | None) -> int:
    if not value:
        raise ValueError("Chinese outcome number must be non-empty")
    if value == "十":
        return 10
    if "十" not in value:
        return _CHINESE_DIGITS[value]
    tens, ones = value.split("十", 1)
    tens_value = _CHINESE_DIGITS[tens] if tens else 1
    ones_value = _CHINESE_DIGITS[ones] if ones else 0
    return tens_value * 10 + ones_value


__all__ = [
    "InstructionOutcome",
    "enumerated_atomic_count",
    "parse_numbered_outcomes",
]
