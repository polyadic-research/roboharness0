"""Convert official Xiaomi episode evidence into per-task capability statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def convert_summary(summary: dict, *, registry_path: Path | None = None) -> list[dict]:
    registry = json.loads((registry_path or Path(__file__).parents[1] / "task_groups.json").read_text())
    groups = {task: group for group in ("atomic_seen", "composite_seen", "composite_unseen")
              for task in registry["groups"][group]["tasks"]}
    records, seen = [], set()
    for task, stats in sorted(summary["tasks"].items()):
        if task not in groups:
            raise ValueError(f"unknown task: {task}")
        episodes = stats["episodes"]
        n = len(episodes)
        if n < 1 or stats["num_episodes"] != n:
            raise ValueError(f"{task}: episode count mismatch")
        horizon = stats["horizon"]
        indices = set()
        for episode in episodes:
            index = episode["global_episode_index"]
            if index in seen or episode["episode"] in indices:
                raise ValueError(f"{task}: duplicate episode")
            if (type(episode["success"]) is not bool or type(episode["steps"]) is not int
                    or not 0 < episode["steps"] <= horizon or type(episode["seed"]) is not int):
                raise ValueError(f"{task}: invalid episode evidence")
            seen.add(index)
            indices.add(episode["episode"])
        k = sum(episode["success"] for episode in episodes)
        if stats["successes"] != k or not math.isclose(stats["success_rate"], k / n, abs_tol=1e-12):
            raise ValueError(f"{task}: success summary differs from episode evidence")
        z = 1.959963984540054
        denominator = 1 + z * z / n
        centre = (k / n + z * z / (2 * n)) / denominator
        radius = z * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / denominator
        records.append({
            "task": task, "group": groups[task], "n": n, "successful": k,
            "success_rate": k / n, "wilson95_low": max(0.0, centre - radius),
            "wilson95_high": min(1.0, centre + radius),
            "first_success_step_median_success_only": median(
                episode["steps"] for episode in episodes if episode["success"]
            ) if k else None,
            "horizons": [horizon], "status_counts": {"success": k, "failure": n - k},
            "episode_seeds": [episode["seed"] for episode in episodes],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = convert_summary(json.loads(args.source.read_text()))
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
