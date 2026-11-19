"""Handoff (training start) poses from a policy re-measurement export (1.0.11 D1).

Each successful atomic episode of a capability re-measurement starts the robot
where RoboCasa places it for that task. That start pose is the pose the policy
is used to: transitions should hand control back there (backlog B7). This tool
reads offline episode artifacts only (episode records, the initial-pose
summary, the initial scene XML and the step-0 public state) and writes:

- per task: the referenced fixtures with the base's perpendicular distance to
  the fixture front, the lateral offset of the fixture centre, the heading error
  against facing the front, and the arm's initial end-effector pose in the base
  frame;
- per fixture class: the same statistics over the anchor fixture (the
  referenced fixture whose front is nearest) of every successful episode;
- the ``handback`` block of ``<policy>.capabilities.json``.

Geometry from the scene XML (fixture extents) is offline statistics, like the
existing ``handoff-pose.json``. At run time the planner only receives these
summary numbers and measures the live fixture front with ``back_project``.

Usage::

    python -m robots.robocasa.tools.build_handback --out handback.json \
        /path/to/pi05-capability-20261008-parallel /path/to/...-recovery
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
NAVIGATION_EXCLUSION_REASON = "navigation start pose is not a fixture-facing handoff pose"


def _navigation_tasks() -> set[str]:
    metadata = json.loads((Path(__file__).parents[1] / "task_groups.json").read_text())
    return {"NavigateKitchen"} | {
        name for name, task in metadata["tasks"].items()
        if task.get("category") in {"navigate", "navigation"}
    }


def _quat_wxyz_matrix(value: str | None) -> np.ndarray:
    if not value:
        return np.eye(3)
    w, x, y, z = (float(v) for v in value.split())
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vector(value: str | None) -> np.ndarray:
    return np.zeros(3) if not value else np.array([float(v) for v in value.split()])


def fixture_front_offset(root: ET.Element, body_name: str) -> float | None:
    """Local -y extent of a fixture's box geoms (RoboCasa fixtures face local -y).

    Nested bodies (doors, drawers) are included at their closed XML pose.
    Returns the (negative) local y of the front face, or None without boxes.
    """
    body = next((b for b in root.iter("body") if b.get("name") == body_name), None)
    if body is None:
        return None
    front: float | None = None

    def visit(element: ET.Element, rotation: np.ndarray, offset: np.ndarray) -> None:
        nonlocal front
        for child in element:
            if child.tag == "geom" and child.get("type") == "box" and child.get("size"):
                half = _vector(child.get("size"))
                local = rotation @ _quat_wxyz_matrix(child.get("quat"))
                center = offset + rotation @ _vector(child.get("pos"))
                low = center[1] - float(np.abs(local[1]) @ half)
                front = low if front is None else min(front, low)
            elif child.tag == "body":
                visit(child, rotation @ _quat_wxyz_matrix(child.get("quat")),
                      offset + rotation @ _vector(child.get("pos")))

    visit(body, np.eye(3), np.zeros(3))
    return front


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _relative_eef(state: np.ndarray, base_world: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Base-relative EEF position and xyzw quaternion from a 16D state row.

    The runtime order (``robots.robocasa.observation.STATE_SLICES``) starts with
    the base; the capability benchmark's trace stores the EEF first. Pick the
    layout whose base slice matches the recorded world base position.
    """
    if state.shape != (16,):
        return None
    if np.allclose(state[0:3], base_world, atol=1e-3):
        return state[7:10], state[10:14]
    if np.allclose(state[7:10], base_world, atol=1e-3):
        return state[0:3], state[3:7]
    return None


def episode_pose(attempt: Path) -> list[dict[str, Any]] | None:
    """Start pose of one attempt against each task-referenced fixture."""
    pose_path, trace_path, xml_path = (attempt / name for name in ("handoff-pose.json", "trace.npz", "initial.xml"))
    if not (pose_path.is_file() and trace_path.is_file() and xml_path.is_file()):
        return None
    pose = json.loads(pose_path.read_text())
    with np.load(trace_path, allow_pickle=False) as trace:
        state = np.asarray(trace["states"][0], dtype=float)
    root = ET.parse(xml_path).getroot()
    base = np.asarray(pose["robot_base_position_world_m"][:2], dtype=float)
    eef = _relative_eef(state, np.asarray(pose["robot_base_position_world_m"], dtype=float))
    if eef is None:
        return None
    yaw = float(pose["robot_base_yaw_world_rad"])
    rows = []
    for fixture in pose["fixtures"]:
        if not fixture.get("task_reference"):
            continue
        front_y = fixture_front_offset(root, fixture["body"])
        if front_y is None:
            continue
        fyaw = float(fixture["yaw_world_rad"])
        rotation = np.array([[math.cos(fyaw), -math.sin(fyaw)], [math.sin(fyaw), math.cos(fyaw)]])
        center = np.asarray(fixture["position_world_m"][:2], dtype=float)
        front_point = center + rotation @ np.array([0.0, front_y])
        outward = rotation @ np.array([0.0, -1.0])
        distance = float((base - front_point) @ outward)
        bearing = float(fixture["bearing_relative_robot_rad"])
        rows.append({
            "fixture_class": fixture["class"],
            "reference_keys": list(fixture.get("reference_keys", [])),
            "front_distance_m": distance,
            "lateral_offset_m": float(fixture["distance_center_xy_m"]) * math.sin(bearing),
            "facing_error_deg": math.degrees(_wrap(float(fixture["fixture_yaw_relative_robot_rad"]) + math.pi / 2)),
            "eef_position_base_m": eef[0].tolist(),
            "eef_quaternion_base_xyzw": eef[1].tolist(),
            "base_yaw_world_rad": yaw,
        })
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"median": float(np.median(array)), "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90))}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eef = np.asarray([row["eef_position_base_m"] for row in rows], dtype=float)
    return {
        "episodes": len(rows),
        "front_distance_m": _stats([row["front_distance_m"] for row in rows]),
        "lateral_offset_m": _stats([row["lateral_offset_m"] for row in rows]),
        "facing_error_deg": _stats([row["facing_error_deg"] for row in rows]),
        "abs_facing_error_deg_p90": float(np.percentile([abs(row["facing_error_deg"]) for row in rows], 90)),
        "eef_position_base_m": {
            "median": np.median(eef, axis=0).tolist(),
            "p10": np.percentile(eef, 10, axis=0).tolist(),
            "p90": np.percentile(eef, 90, axis=0).tolist(),
        },
    }


def collect(roots: list[Path]) -> dict[str, Any]:
    """Successful atomic episodes of every export root, deduplicated by episode."""
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchor_tasks: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    navigation = _navigation_tasks()
    excluded: Counter[str] = Counter()
    sources = []
    for root in roots:
        used = 0
        for record_path in sorted(root.glob("rounds/*/completed/*.json")):
            record = json.loads(record_path.read_text())
            if record.get("group") != "atomic" or record.get("success") is not True:
                continue
            artifact = Path(str(record.get("artifact_dir", "")))
            # The record names the remote path; the export keeps its tail.
            parts = artifact.parts
            if "rounds" not in parts:
                continue
            local = root.joinpath(*parts[parts.index("rounds"):])
            key = "/".join(parts[parts.index("rounds"):])
            if key in seen:
                continue
            task = record["task"]
            if task in navigation:
                # Count all excluded successes, even when their pose is unavailable.
                seen.add(key)
                excluded[task] += 1
                continue
            rows = episode_pose(local)
            if not rows:
                continue
            seen.add(key)
            used += 1
            for row in rows:
                label = "+".join(row["reference_keys"]) or row["fixture_class"]
                by_task[task][f"{row['fixture_class']}:{label}"].append(row)
            anchor = min(rows, key=lambda row: row["front_distance_m"])
            anchors[anchor["fixture_class"]].append(anchor)
            anchor_tasks[anchor["fixture_class"]].add(task)
        sources.append({"root": root.name, "successful_atomic_episodes": used})
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": ("start pose of successful atomic re-measurement episodes; base distance to the "
                  "fixture front along its outward normal, lateral offset of the fixture centre "
                  "in the base frame (left positive), heading error against facing the front "
                  "(positive: base turned left), initial EEF pose in the base frame"),
        "sources": sources,
        "excluded_tasks": [
            {"task": task, "reason": NAVIGATION_EXCLUSION_REASON, "episodes": count}
            for task, count in sorted(excluded.items())
        ],
        "by_task": {
            task: {fixture: _summary(rows) for fixture, rows in sorted(fixtures.items())}
            for task, fixtures in sorted(by_task.items())
        },
        "by_fixture": {
            name: {**_summary(rows), "tasks": sorted(anchor_tasks[name])}
            for name, rows in sorted(anchors.items())
        },
    }


def handback_block(summary: dict[str, Any], *, min_episodes: int = 3) -> dict[str, Any]:
    """The capabilities-file ``handback`` block; classes with few episodes are left out."""
    def limits(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "distance_m": [round(max(0.0, item["front_distance_m"]["p10"]), 3),
                           round(max(0.0, item["front_distance_m"]["p90"]), 3)],
            "heading_deg_max": round(min(180.0, item["abs_facing_error_deg_p90"]), 1),
            "front_distance_m_median": round(item["front_distance_m"]["median"], 3),
            "lateral_m": [round(item["lateral_offset_m"]["p10"], 3), round(item["lateral_offset_m"]["p90"], 3)],
            "eef_start_base_m": [round(v, 3) for v in item["eef_position_base_m"]["median"]],
            "episodes": item["episodes"],
        }

    fixtures = {name: limits(item) for name, item in summary["by_fixture"].items() if item["episodes"] >= min_episodes}
    pooled = list(summary["by_fixture"].values())
    episodes = sum(item["episodes"] for item in pooled)
    default = {
        "distance_m": [round(min(v["distance_m"][0] for v in fixtures.values()), 3),
                       round(max(v["distance_m"][1] for v in fixtures.values()), 3)] if fixtures else [0.0, 0.0],
        "heading_deg_max": round(max((v["heading_deg_max"] for v in fixtures.values()), default=0.0), 1),
    }
    by_task = {}
    for task, rows in summary["by_task"].items():
        anchor_name, anchor = min(rows.items(), key=lambda pair: pair[1]["front_distance_m"]["median"])
        if anchor["episodes"] >= 1:
            by_task[task] = {"fixture": anchor_name.split(":", 1)[0], **limits(anchor)}
    return {"provisional": True, "default": default, "by_fixture": fixtures, "by_task": by_task,
            "source": {"episodes": episodes, "sources": summary["sources"],
                       "excluded_tasks": summary.get("excluded_tasks", [])}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="re-measurement export roots")
    parser.add_argument("--out", type=Path, required=True, help="summary JSON (by_task, by_fixture, handback)")
    parser.add_argument("--min-episodes", type=int, default=3)
    args = parser.parse_args(argv)
    summary = collect(args.roots)
    summary["handback"] = handback_block(summary, min_episodes=args.min_episodes)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
