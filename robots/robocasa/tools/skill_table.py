"""Generate the policy skill's tables from its capability JSON (1.0.11 D2).

The skill Markdown is what the planner reads; ``<policy>.capabilities.json``
is the authority the harness checks plans against. This tool rewrites the two
generated parts of the Markdown from the JSON so that a re-measurement update
is one command plus a review of the diff:

- the operation table: one row per measured atomic row, with its official
  instruction (task registry), tier, threshold marker, counts and typical
  steps. The Attention column stays hand-written; a new row gets a default
  text by tier.
- the handoff table between ``<!-- handback:begin -->`` and
  ``<!-- handback:end -->``: the training start pose per fixture class from the
  ``handback`` block (see ``build_handback``).

Usage::

    python -m robots.robocasa.tools.skill_table --policy pi05 [--write | --check]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from robots.robocasa.capabilities import (
    CAPABILITY_DIRECTORY,
    _atomic_instructions,
    load_capabilities,
)

TABLE_HEADER = "| Task (official instruction) | Tier | 典型步数 / Typical steps | Attention |"
HANDBACK_BEGIN = "<!-- handback:begin -->"
HANDBACK_END = "<!-- handback:end -->"
DEFAULT_ATTENTION = {
    "reliable": "Keep full operation unless clear failure. [D]",
    "unstable": "Verify the result; failure modes unaudited. [D]",
    "weak": "Takeover; verify the result. [D]",
}
_ROW = re.compile(r"^\| ([A-Za-z0-9]+):")


def _steps(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:g}" if float(value).is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")


def table_row(task: str, row: dict, instruction: str, attention: str) -> str:
    tier = row["tier"].capitalize() + ("*" if row["crosses_threshold"] else "")
    if row["episodes"] != 50:
        tier += f" ({row['successes']}/{row['episodes']})"
    return f'| {task}: "{instruction}" | {tier} | {_steps(row["typical_steps"])} | {attention} |'


def _table_bounds(lines: list[str]) -> tuple[int, int]:
    start = lines.index(TABLE_HEADER)
    end = start + 2  # header and separator
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return start + 2, end


def render_operation_table(catalog: dict, skill_text: str) -> str:
    """Rewrite the operation table rows; keep existing order and Attention text."""
    lines = skill_text.split("\n")
    first, end = _table_bounds(lines)
    attention: dict[str, str] = {}
    order: list[str] = []
    for line in lines[first:end]:
        match = _ROW.match(line)
        if match:
            order.append(match.group(1))
            attention[match.group(1)] = line.rstrip().rstrip("|").rsplit("|", 1)[1].strip()
    instructions = _atomic_instructions()
    atomic = {task: row for task, row in catalog["rows"].items() if row["family"] == "atomic"}
    missing = sorted(set(atomic) - set(instructions))
    if missing:
        raise ValueError(f"atomic rows without a registry instruction: {missing}")
    tasks = [task for task in order if task in atomic] + sorted(set(atomic) - set(order))
    rows = [
        table_row(task, atomic[task], instructions[task],
                  attention.get(task) or DEFAULT_ATTENTION[atomic[task]["tier"]])
        for task in tasks
    ]
    return "\n".join([*lines[:first], *rows, *lines[end:]])


def handback_table(catalog: dict) -> list[str]:
    handback = catalog["handback"]
    lines = [
        "| Fixture | Base to front (m): median [p10–p90] | Fixture centre lateral (m) | Heading max | Arm start, base frame (m) | Episodes |",
        "|---|---|---|---:|---|---:|",
    ]
    for name, limits in sorted(handback["by_fixture"].items()):
        lateral = limits.get("lateral_m")
        eef = limits.get("eef_start_base_m")
        lines.append(
            f"| {name} | {limits.get('front_distance_m_median', float('nan')):.2f} "
            f"[{limits['distance_m'][0]:.2f}–{limits['distance_m'][1]:.2f}] | "
            + (f"{lateral[0]:+.2f} to {lateral[1]:+.2f}" if lateral else "—")
            + f" | {limits['heading_deg_max']:g}° | "
            + (f"[{eef[0]:.2f}, {eef[1]:.2f}, {eef[2]:.2f}]" if eef else "—")
            + f" | {limits.get('episodes', '—')} |"
        )
    return lines


def render_handback(catalog: dict, skill_text: str) -> str:
    if HANDBACK_BEGIN not in skill_text:
        return skill_text
    head, rest = skill_text.split(HANDBACK_BEGIN, 1)
    _, tail = rest.split(HANDBACK_END, 1)
    return "\n".join([head.rstrip("\n") + "\n", HANDBACK_BEGIN, *handback_table(catalog), HANDBACK_END]) + tail


def render(catalog: dict, skill_text: str) -> str:
    return render_handback(catalog, render_operation_table(catalog, skill_text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default="pi05")
    parser.add_argument("--skill", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the skill file in place")
    mode.add_argument("--check", action="store_true", help="fail when the skill differs from the JSON")
    args = parser.parse_args(argv)
    skill = args.skill or CAPABILITY_DIRECTORY / f"{args.policy}.md"
    catalog = load_capabilities(args.policy)
    if catalog is None:
        raise SystemExit(f"no capability catalog for {args.policy}")
    current = skill.read_text()
    rendered = render(catalog, current)
    if args.write:
        skill.write_text(rendered)
    elif rendered != current:
        sys.stdout.writelines(difflib.unified_diff(
            current.splitlines(keepends=True), rendered.splitlines(keepends=True),
            fromfile=str(skill), tofile=f"{skill} (generated)",
        ))
        if args.check:
            return 1
    print(json.dumps({"skill": str(skill), "changed": rendered != current, "written": args.write}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
