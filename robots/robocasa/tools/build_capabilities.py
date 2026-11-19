"""Build measured capability rows reproducibly and merge supplemental snapshots."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from robots.robocasa.capabilities import point_tier, validate_capabilities

SOURCE_LABEL = "reviews/pi05-capability-skill-20261006/per_task.json"
THRESHOLDS = {"reliable": 0.70, "unstable": 0.30}
CHECKPOINT = {
    "repo_id": "robocasa/robocasa365_checkpoints",
    "path": "pi05_pretrain_human300/multitask_learning/75000",
    "revision": "c484448aba1a9b60a04c9b0ca117241518ea69f3",
}


def build_catalog(records: list[dict], *, source: str = SOURCE_LABEL, policy: str = "pi05") -> dict:
    rows = {}
    for item in records:
        task = item["task"]
        if task in rows:
            raise ValueError(f"duplicate source task: {task}")
        n, k = item["n"], item["successful"]
        low, high = item["wilson95_low"], item["wilson95_high"]
        rows[task] = {
            "family": "atomic"
            if (item.get("group") or item["family"]).startswith("atomic")
            else "composite",
            "tier": point_tier(k, n, THRESHOLDS),
            "successes": k,
            "episodes": n,
            "wilson95": [low, high],
            "crosses_threshold": any(low < t < high for t in THRESHOLDS.values()),
            "typical_steps": item["first_success_step_median_success_only"],
            "source": source,
        }
    return validate_capabilities(
        {
            "schema_version": 1,
            "policy": policy,
            "checkpoint": (
                xiaomi_checkpoint() if policy == "xiaomi-mibot" else CHECKPOINT
            ),
            "thresholds": THRESHOLDS,
            "rows": rows,
            # No measured opening-pose statistics yet: explicit non-operational placeholders.
            **({"handback": {
                "provisional": True,
                "default": {"distance_m": [0.0, 0.0], "heading_deg_max": 0.0},
                "by_fixture": {},
            }} if policy == "pi05" else {}),
        }
    )


def xiaomi_checkpoint() -> dict:
    from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT

    return {"repo_id": XIAOMI_MIBOT_CHECKPOINT.repo_id, "path": ".",
            "revision": XIAOMI_MIBOT_CHECKPOINT.revision}


def report_records(report: dict, tasks: list[str]) -> list[dict]:
    """Select completed rows from a re-measurement capability report.

    The report keeps every task in a ``tasks`` list with Chinese tier labels;
    only counts, Wilson bounds, family and success-only timing are used here,
    and tiers are recomputed from the point estimate.
    """
    if not tasks:
        raise ValueError("a capability report source needs --tasks")
    if report.get("policy") != "pi05" or report.get("thresholds") != THRESHOLDS:
        raise ValueError("report policy/thresholds differ from the catalog")
    rows = {row["task"]: row for row in report["tasks"]}
    records = []
    for task in tasks:
        row = rows.get(task)
        if row is None:
            raise ValueError(f"{task}: not in the report")
        if not row.get("complete") or row["n"] != row["scheduled"]:
            raise ValueError(
                f"{task}: re-measurement incomplete ({row['n']}/{row['scheduled']})"
            )
        records.append(row)
    return records


def merge_catalog(existing: dict, incoming: dict) -> dict:
    """Merge authoritative row snapshots without dropping unrelated rows/handback."""
    validate_capabilities(existing)
    validate_capabilities(incoming)
    for key in ("schema_version", "policy", "checkpoint", "thresholds"):
        if existing[key] != incoming[key]:
            raise ValueError(f"cannot merge different {key}")
    result = deepcopy(existing)
    for task, row in incoming["rows"].items():
        old = result["rows"].get(task)
        if old and row["episodes"] < old["episodes"]:
            raise ValueError(
                f"{task}: incoming snapshot would reduce measured episodes"
            )
        result["rows"][task] = {**(old or {}), **deepcopy(row)}
    result.setdefault("declared", {}).update(deepcopy(incoming.get("declared", {})))
    if not result["declared"]:
        result.pop("declared")
    if "handback" not in incoming:
        return validate_capabilities(result)
    result.setdefault("handback", deepcopy(incoming["handback"]))
    # A provisional source must never erase measured fixture statistics.
    result["handback"]["by_fixture"].update(
        deepcopy(incoming["handback"]["by_fixture"])
    )
    by_task = {**result["handback"].get("by_task", {}),
               **deepcopy(incoming["handback"].get("by_task", {}))}
    if by_task:
        result["handback"]["by_task"] = by_task
    if not incoming["handback"]["provisional"]:
        result["handback"].update(
            provisional=False, default=deepcopy(incoming["handback"]["default"])
        )
    return validate_capabilities(result)


SKILL_ROW = re.compile(
    r"\| ([A-Za-z0-9]+):.*?\| (Reliable|Unstable|Weak)(\*)?(?: \((\d+)/(\d+)\))? \| ([0-9.]+|—) \|"
)


def check_skill_consistency(
    catalog: dict, skill_text: str, *, require_all_atomic: bool = False
) -> int:
    """Check every Markdown operation-table row against the measured catalog.

    Rows may carry their counts as ``Tier* (k/n)``; those must match the JSON.
    With ``require_all_atomic`` every measured atomic row must appear in the
    table, and every declared entry must be named in the skill text.
    """
    checked = set()
    for line in skill_text.splitlines():
        cells = line.split("|")
        match = SKILL_ROW.match(line)
        # A tier row the pattern cannot read would otherwise be skipped silently.
        if (
            len(cells) > 3
            and cells[2].strip().split(" ")[0].rstrip("*")
            in {"Reliable", "Unstable", "Weak"}
            and match is None
        ):
            raise ValueError(f"unparseable skill table row: {line}")
        if match is None:
            continue
        task, tier, star, k, n, steps = match.groups()
        row = catalog["rows"].get(task)
        typical = None if steps == "—" else float(steps)
        if (
            row is None
            or row["tier"] != tier.lower()
            or row["typical_steps"] != typical
        ):
            raise ValueError(f"{task}: Markdown tier/typical steps differ from JSON")
        if row["crosses_threshold"] != bool(star):
            raise ValueError(f"{task}: Markdown threshold marker differs from JSON")
        if k is not None and (int(k), int(n)) != (row["successes"], row["episodes"]):
            raise ValueError(f"{task}: Markdown counts differ from JSON")
        if k is None and row["episodes"] != 50:
            raise ValueError(
                f"{task}: rows outside the 50-episode benchmark must show k/n"
            )
        checked.add(task)
    atomic = {
        task for task, row in catalog["rows"].items() if row["family"] == "atomic"
    }
    if len(checked) < 18 or not checked <= atomic:
        raise ValueError("expected all 18 original atomic operation rows in Markdown")
    if require_all_atomic:
        if checked != atomic:
            raise ValueError(
                f"atomic rows missing from Markdown: {sorted(atomic - checked)}"
            )
        missing = [
            name for name in catalog.get("declared", {}) if name not in skill_text
        ]
        if missing:
            raise ValueError(f"declared entries missing from Markdown: {missing}")
    return len(checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-label", default=SOURCE_LABEL)
    parser.add_argument("--policy", choices=("pi05", "xiaomi-mibot"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge", type=Path, action="append", default=[])
    parser.add_argument(
        "--tasks",
        default="",
        help="comma-separated rows to take from a capability report source",
    )
    parser.add_argument(
        "--declared", type=Path, help="JSON object replacing the declared section"
    )
    parser.add_argument(
        "--handback", type=Path,
        help="build_handback output (its handback block replaces the catalog's)",
    )
    parser.add_argument("--require-all-atomic", action="store_true")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    if isinstance(source, dict) and isinstance(source.get("tasks"), list):
        source = report_records(source, [t for t in args.tasks.split(",") if t])
    catalog = (
        build_catalog(source, source=args.source_label,
                      policy=args.policy or ("xiaomi-mibot" if args.source_label.startswith("xiaomi-official-summary-") else "pi05"))
        if isinstance(source, list)
        else validate_capabilities(source)
    )
    # Existing output is the merge base. Default rebuilds preserve supplementary rows.
    if args.output.exists():
        catalog = merge_catalog(json.loads(args.output.read_text()), catalog)
    for path in args.merge:
        catalog = merge_catalog(catalog, json.loads(path.read_text()))
    if args.declared:
        catalog = validate_capabilities(
            {**catalog, "declared": json.loads(args.declared.read_text())}
        )
    if args.handback:
        handback = json.loads(args.handback.read_text())
        catalog = validate_capabilities(
            {**catalog, "handback": handback.get("handback", handback)}
        )
    count = (
        check_skill_consistency(
            catalog, args.skill.read_text(), require_all_atomic=args.require_all_atomic
        )
        if args.skill
        else None
    )
    rendered = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit("capability output is not reproducible")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(
        json.dumps(
            {
                "rows": len(catalog["rows"]),
                "atomic": sum(
                    r["family"] == "atomic" for r in catalog["rows"].values()
                ),
                "skill_rows_checked": count,
                "reproducible": args.check,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
