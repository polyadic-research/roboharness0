"""Optional policy capability text; no policy execution or lifecycle decisions."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

SKILL_DIRECTORY = Path(__file__).resolve().parent


def load_policy_skill(policy_family: str) -> str:
    """Read <policy_family>.md, or return empty for an unavailable family/file."""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", policy_family) is None:
        return ""
    try:
        return (
            (SKILL_DIRECTORY / f"{policy_family}.md")
            .read_text(encoding="utf-8")
            .strip()
        )
    except FileNotFoundError:
        return ""


def strip_sections(text: str, headings: Iterable[str]) -> str:
    """Remove whole ``## <heading>`` sections from a skill for ablation runs.

    Matching is exact on the heading title; unknown headings are ignored. The
    text before the first ``##`` heading and every other section are kept.
    """
    drop = {heading.strip() for heading in headings}
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("## "):
            skipping = line[3:].strip() in drop
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()
