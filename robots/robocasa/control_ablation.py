"""One configuration for control-mode gates and the prompts they authorize."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def control_options(args: Any) -> dict[str, bool]:
    """Convert RoboCasa CLI opt-outs to the planning stack's enabled flags."""
    return {
        "transition_takeover_enabled": not getattr(args, "no_transition_takeover", False),
        "op_takeover_enabled": not getattr(args, "no_op_takeover", False),
    }


def prompt_ablation(options: Mapping[str, Any]) -> set[str]:
    """Select matching executor/planner sections from the same enabled flags."""
    return {
        section
        for key, section in (
            ("transition_takeover_enabled", "TRANSITIONS"),
            ("op_takeover_enabled", "TAKEOVER"),
        )
        if options.get(key, True) is False
    }


def control_metadata(
    family: str, skill: str, options: Mapping[str, bool],
) -> dict[str, Any]:
    """Hash the catalog bytes and the actual, possibly ablated skill text."""
    from robots.robocasa.policy_skills import SKILL_DIRECTORY

    catalog = SKILL_DIRECTORY / f"{family}.capabilities.json"
    return {
        **options,
        "policy_family": family,
        "capabilities_sha256": (
            hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.is_file() else None
        ),
        "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
    }
