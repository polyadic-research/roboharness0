"""Measured policy capability catalogs and plan-time control contracts."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from robots.robocasa import control_modes

CAPABILITY_DIRECTORY = Path(__file__).parent / "policy_skills"
TIERS = frozenset(control_modes.TIERS)
CONTROL_MODES = frozenset(control_modes.CONTROL_MODES)
VLA_TOOLS = frozenset(control_modes.POLICY_TOOLS)
PRIVILEGED_PHYSICAL_TOOLS = frozenset(control_modes.PRIVILEGED_TOOLS)
READ_ONLY_PERCEPTION_TOOLS = frozenset(control_modes.ALLOWLISTABLE_READ_ONLY_TOOLS)
DECLARED_KINDS = frozenset({"proxy", "demonstrated"})
DECLARABLE_MODES = frozenset({"delegate", "monitor", "takeover"})


def point_tier(successes: int, episodes: int, thresholds: dict) -> str:
    rate = successes / episodes
    return (
        "reliable"
        if rate >= thresholds["reliable"]
        else "unstable"
        if rate >= thresholds["unstable"]
        else "weak"
    )


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (float, int))
        and math.isfinite(value)
    )


def validate_capabilities(data: Any, *, policy: str | None = None) -> dict:
    """Fail closed for malformed catalogs; absent catalogs are handled separately."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("capability schema_version must be 1")
    name = data.get("policy")
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
        raise ValueError("capability policy must be a safe family name")
    if policy is not None and name != policy:
        raise ValueError("capability policy does not match selected VLA")
    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, dict) or any(
        not isinstance(checkpoint.get(key), str) or not checkpoint[key].strip()
        for key in ("repo_id", "path", "revision")
    ):
        raise ValueError("capability checkpoint requires repo_id, path, revision")
    thresholds = data.get("thresholds")
    if (
        not isinstance(thresholds, dict)
        or any(not _number(thresholds.get(key)) for key in ("reliable", "unstable"))
        or not 0 < thresholds["unstable"] < thresholds["reliable"] <= 1
    ):
        raise ValueError("capability thresholds must be ordered probabilities")
    rows = data.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("capability rows must be an object")
    for task, row in rows.items():
        if not isinstance(task, str) or not task.strip() or task == "none":
            raise ValueError("invalid capability skill_row")
        if not isinstance(row, dict) or row.get("family") not in {
            "atomic",
            "composite",
        }:
            raise ValueError(f"{task}: invalid family")
        n, k = row.get("episodes"), row.get("successes")
        if type(n) is not int or type(k) is not int or not 0 <= k <= n or n < 1:
            raise ValueError(f"{task}: invalid successes/episodes")
        if row.get("tier") != point_tier(k, n, thresholds):
            raise ValueError(f"{task}: tier does not match point estimate")
        interval = row.get("wilson95")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(_number(v) for v in interval)
            or not 0 <= interval[0] <= interval[1] <= 1
        ):
            raise ValueError(f"{task}: invalid wilson95")
        crosses = any(interval[0] < t < interval[1] for t in thresholds.values())
        if (
            type(row.get("crosses_threshold")) is not bool
            or row["crosses_threshold"] != crosses
        ):
            raise ValueError(f"{task}: inconsistent crosses_threshold")
        steps = row.get("typical_steps")
        if steps is not None and (not _number(steps) or steps <= 0):
            raise ValueError(f"{task}: invalid typical_steps")
        if not isinstance(row.get("source"), str) or not row["source"].strip():
            raise ValueError(f"{task}: missing source")
    _validate_declared(data.get("declared", {}), rows)
    handback = data.get("handback")
    if "handback" not in data:
        return data
    if not isinstance(handback, dict) or type(handback.get("provisional")) is not bool:
        raise ValueError("handback requires provisional flag")
    fixtures = handback.get("by_fixture")
    if not isinstance(fixtures, dict):
        raise ValueError("handback by_fixture must be an object")
    by_task = handback.get("by_task", {})
    if not isinstance(by_task, dict):
        raise ValueError("handback by_task must be an object")
    for task, limits in by_task.items():
        if not isinstance(task, str) or not task.strip() or not isinstance(limits, dict):
            raise ValueError("handback by_task needs task names and limit objects")
        if not isinstance(limits.get("fixture"), str) or not limits["fixture"].strip():
            raise ValueError(f"handback by_task {task!r} needs its fixture class")
    for limits in (handback.get("default"), *fixtures.values(), *by_task.values()):
        _validate_handback_limits(limits)
    return data


def _validate_handback_limits(limits: Any) -> None:
    """Training-start-pose limits (1.0.11 D1): distance to the fixture front,
    heading error and, when measured, lateral offset and arm start position."""
    if not isinstance(limits, dict):
        raise ValueError("handback limits must be an object")
    distance, heading = limits.get("distance_m"), limits.get("heading_deg_max")
    if (
        not isinstance(distance, list)
        or len(distance) != 2
        or not all(_number(v) for v in distance)
        or not 0 <= distance[0] <= distance[1]
    ):
        raise ValueError("handback distance_m must be an ordered range")
    if not _number(heading) or not 0 <= heading <= 180:
        raise ValueError("handback heading_deg_max must be in [0, 180]")
    lateral = limits.get("lateral_m")
    if lateral is not None and (
        not isinstance(lateral, list) or len(lateral) != 2
        or not all(_number(v) for v in lateral) or lateral[0] > lateral[1]
    ):
        raise ValueError("handback lateral_m must be an ordered range")
    median = limits.get("front_distance_m_median")
    if median is not None and (not _number(median) or median < 0):
        raise ValueError("handback front_distance_m_median must be nonnegative")
    eef = limits.get("eef_start_base_m")
    if eef is not None and (not isinstance(eef, list) or len(eef) != 3 or not all(_number(v) for v in eef)):
        raise ValueError("handback eef_start_base_m must be three numbers")
    episodes = limits.get("episodes")
    if episodes is not None and (type(episodes) is not int or episodes < 1):
        raise ValueError("handback episodes must be a positive integer")


def _validate_declared(declared: Any, rows: dict) -> None:
    """Skill-declared operations: a proxy of one measured row, or a demonstrated op.

    Each entry names the control modes it allows, so a proxy can inherit a
    measured tier while being capped (e.g. reliable proxy -> monitor only).
    """
    if not isinstance(declared, dict):
        raise ValueError("declared must be an object")
    for name, entry in declared.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name == "none"
            or name in rows
        ):
            raise ValueError(f"invalid declared skill_row {name!r}")
        if not isinstance(entry, dict) or entry.get("kind") not in DECLARED_KINDS:
            raise ValueError(f"{name}: declared kind must be proxy or demonstrated")
        for key in ("operation", "evidence"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"{name}: declared entry requires {key}")
        modes = entry.get("allowed_control_modes")
        if (
            not isinstance(modes, list)
            or not modes
            or len(set(modes)) != len(modes)
            or not set(modes) <= DECLARABLE_MODES
        ):
            raise ValueError(
                f"{name}: allowed_control_modes must list delegate/monitor/takeover"
            )
        if entry["kind"] == "proxy":
            if entry.get("proxy_row") not in rows or "tier" in entry:
                raise ValueError(
                    f"{name}: proxy needs a measured proxy_row and inherits its tier"
                )
        elif (
            entry.get("tier") not in {"reliable", "unstable", "weak"}
            or "proxy_row" in entry
        ):
            raise ValueError(
                f"{name}: demonstrated entry needs its own reliable/unstable/weak tier"
            )


def capability_entry(catalog: dict | None, skill_row: str) -> dict | None:
    """Resolve a measured row or declared entry to tier, timing and mode limits."""
    if catalog is None:
        return None
    row = catalog["rows"].get(skill_row)
    if row is not None:
        return {
            "tier": row["tier"],
            "typical_steps": row["typical_steps"],
            "allowed_control_modes": None,
        }
    entry = catalog.get("declared", {}).get(skill_row)
    if entry is None:
        return None
    proxy = catalog["rows"].get(entry.get("proxy_row"))
    return {
        "tier": proxy["tier"] if proxy else entry["tier"],
        "typical_steps": proxy["typical_steps"] if proxy else None,
        "allowed_control_modes": list(entry["allowed_control_modes"]),
    }


def load_capabilities(
    policy: str, *, directory: Path = CAPABILITY_DIRECTORY
) -> dict | None:
    """Load <policy>.capabilities.json; return None only when it is absent."""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", policy) is None:
        raise ValueError("invalid policy family")
    try:
        data = json.loads((directory / f"{policy}.capabilities.json").read_text())
    except FileNotFoundError:
        return None
    return validate_capabilities(data, policy=policy)


def capability_summary(catalog: dict | None, *, task_name: str | None = None) -> list[dict]:
    """Expose measured row names, tiers and timing, without fixture coordinates.

    The planning contract carries only atomic rows, declared entries and the
    current task's own row (1.0.11 D3): a composite row never matches a single
    operation, and all composite rows cost about 9-10k tokens per request.
    """
    instructions = _atomic_instructions()
    rows = [
        {
            "skill_row": task,
            "tier": row["tier"],
            "typical_steps": row["typical_steps"],
            # Match against the task definition, not a shorter skill-table label.
            **(
                {"instruction": instructions[task]}
                if row["family"] == "atomic" and task in instructions
                else {}
            ),
            **({"family": "composite"} if row["family"] == "composite" else {}),
        }
        for task, row in sorted((catalog or {}).get("rows", {}).items())
        if row["family"] == "atomic" or task == task_name
    ]
    for name, entry in sorted((catalog or {}).get("declared", {}).items()):
        resolved = capability_entry(catalog, name)
        rows.append(
            {
                "skill_row": name,
                "tier": resolved["tier"],
                "typical_steps": resolved["typical_steps"],
                "declared": entry["kind"],
                **(
                    {"proxy_row": entry["proxy_row"]}
                    if entry["kind"] == "proxy"
                    else {}
                ),
                "allowed_control_modes": resolved["allowed_control_modes"],
                "operation": entry["operation"],
            }
        )
    return rows


@lru_cache(maxsize=1)
def _atomic_instructions() -> dict[str, str]:
    """Official RoboCasa instructions of atomic tasks from the checked-in registry."""
    data = json.loads((Path(__file__).parent / "task_groups.json").read_text())
    return {
        name: task["instruction"]
        for name, task in data["tasks"].items()
        if task.get("family") == "atomic" and isinstance(task.get("instruction"), str)
    }


def is_composite_task(task_name: str | None) -> bool:
    """Use the checked-in registry, never the planner's task-scope claim."""
    data = json.loads((Path(__file__).parent / "task_groups.json").read_text())
    return data["tasks"].get(task_name, {}).get("family") == "composite"


def unmeasured_composite(catalog: dict | None, task_name: str | None) -> bool:
    """An absent whole-task measurement is unknown, not measured weak ability."""
    return bool(
        catalog is not None
        and catalog.get("policy") in control_modes.LANGUAGE_POLICY_FAMILIES
        and is_composite_task(task_name)
        and task_name not in catalog.get("rows", {})
    )
