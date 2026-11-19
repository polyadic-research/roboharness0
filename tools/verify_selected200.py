"""Verify and recompute the unchanged Harness 2.0 historical selected200 records.

This offline check reads the five preserved records. It does not re-open raw
archives, install environments, call a model, or run a simulator.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
SOURCE_DIGEST = "751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc"
RECORDS = {
    "README.zh-CN.md": "8d6223a85a707224bfa933c4c1f589ad4038e5dffdfc8d5819a189c0ba9d4e03",
    "selected200-index-20261107-r002.json": "1a801d939d2837116b4fe4a7ab025f9fcb24466a794e1099a7e50256683ad018",
    "selected200-summary-20261107-r002.json": "2b07a05b6934cf6cc99fc4aec4669a103512d6c4309ce20d8b0a643db5eac5de",
    "verification.json": "4e64e7f6393a47da5be1c12539edae882a7c95ebe612f8ff3595eb0333045e69",
    "version-identity.json": "0162be3194adc7a132ef89652234339c3849735615668f63fd22f4f7c9de6cb8",
}
BATCHES = {"fresh50": "01", "followup01": "02", "followup02": "03", "followup04": "04"}
SPLITS = {"A": ("atomic_seen", 18), "C": ("composite_seen", 16), "U": ("composite_unseen", 16)}


def verify(root: Path) -> dict:
    """Recompute the selected records, rejecting any replacement or relabelling."""
    folder = root / "results/2.0-selected200"
    records = {}
    for name, expected in RECORDS.items():
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Historical record SHA-256 mismatch: {name}")
        if name.endswith(".json"):
            records[name] = json.loads(raw)
    identity = records["version-identity.json"]
    source = identity["scientific_source"]
    if source["commit"] != SOURCE_COMMIT or source["source_digest"] != SOURCE_DIGEST:
        raise ValueError("The selected source is not the current historical 2.0 baseline")

    index = records["selected200-index-20261107-r002.json"]
    summary = records["selected200-summary-20261107-r002.json"]
    rows = index["selected_results"]
    expected_scenes = {
        f"{prefix}{number:02d}-{suffix}"
        for prefix, (_, size) in SPLITS.items()
        for number in range(1, size + 1)
        for suffix in BATCHES.values()
    }
    if len(rows) != 200 or {row["scene_id"] for row in rows} != expected_scenes:
        raise ValueError("The original 200 unique scenes must remain in the denominator")
    if len({row["selected_attempt_id"] for row in rows}) != 200:
        raise ValueError("The selection must contain 200 distinct original attempts")

    splits = defaultdict(Counter)
    batches = defaultdict(Counter)
    overrides = set()
    for row in rows:
        status = row["native_status"]
        if row["native_score_accepted"] is not True or status not in {"success", "failure"}:
            raise ValueError("Every selected scene must retain its accepted native result")
        batch, scene = row["batch"], row["scene_id"]
        if batch not in BATCHES or not scene.endswith("-" + BATCHES[batch]):
            raise ValueError("Original scene-to-batch selection changed")
        splits[SPLITS[scene[0]][0]][status] += 1
        batches[batch][status] += 1
        override = row.get("planning_timeout_override_s")
        if override is not None:
            if override != 600:
                raise ValueError("An original planning timeout override changed")
            overrides.add(scene)

    def counts(counter: Counter) -> dict:
        successes, failures = counter["success"], counter["failure"]
        cases = successes + failures
        return {
            "cases": cases, "successes": successes, "failures": failures,
            "success_rate": successes / cases,
            "success_rate_percent": successes * 100 / cases,
        }

    split_counts = {name: counts(splits[name]) for name, _ in SPLITS.values()}
    batch_counts = {name: counts(batches[name]) for name in BATCHES}
    successes = sum(row["successes"] for row in split_counts.values())
    failures = sum(row["failures"] for row in split_counts.values())
    if (successes, failures) != (141, 59):
        raise ValueError("The selected native result total changed")
    if (summary["native_completed"], summary["successes"], summary["failures"]) != (200, successes, failures):
        raise ValueError("Preserved summary and selected index disagree")
    if summary["success_rate"] != successes / 200:
        raise ValueError("Preserved overall success rate disagrees with the fixed denominator")
    if summary["included_batches"] != list(BATCHES):
        raise ValueError("The selected four original batches changed")
    for batch in summary["batches"]:
        recomputed = batch_counts[batch["batch"]]
        if (recomputed["cases"], recomputed["successes"], recomputed["failures"]) != (
            batch["native_completed"], batch["successes"], batch["failures"]
        ):
            raise ValueError("Original per-batch counts disagree with selected attempts")
    disclosure = summary["parameter_override_disclosure"]
    if overrides != set(disclosure["user_requested_600second_nested_planning_cases"]) or len(overrides) != 10:
        raise ValueError("The ten disclosed planning timeout overrides changed")
    excluded = ["astra50-followup-03-20261104-r001"]
    if index["excluded_batch_ids"] != excluded:
        raise ValueError("The original followup03 exclusion changed")
    return {
        "passed": True,
        "source_commit": SOURCE_COMMIT,
        "source_digest": SOURCE_DIGEST,
        "record_sha256": dict(RECORDS),
        "cases": 200, "successes": successes, "failures": failures,
        "success_rate": successes / 200, "success_rate_percent": successes / 2,
        "splits": split_counts,
        "split_counts": split_counts,
        "split_success_rates": {name: row["success_rate"] for name, row in split_counts.items()},
        "batch_counts": batch_counts,
        "planning_timeout_override_cases": len(overrides),
        "planning_timeout_overrides": sorted(overrides),
        "excluded_batch_ids": excluded,
        "mixed_protocol_supplemental_selection": True,
        "all200_full_workflow_canary_and_archive_complete": summary["all200_full_workflow_canary_and_archive_complete"],
        "current_scope_full_Windows_archive_verified_cases": summary["current_scope_full_Windows_archive_verified_cases"],
        "is_official_2500": False,
        "official_full_evaluation_claimed": False,
        "raw_archives_reverified": False,
        "new_model_calls": 0,
        "new_policy_predictions": 0,
        "new_simulator_episodes": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
