"""Navigation exclusions and exact regression on the formal Xiaomi pose export."""

import json
from pathlib import Path

import pytest

from robots.robocasa.tools import build_handback

REASON = "navigation start pose is not a fixture-facing handoff pose"


def _record(root, index, task):
    completed = root / "rounds/r1/completed"
    completed.mkdir(parents=True, exist_ok=True)
    artifact = root / "rounds/r1/episodes" / str(index)
    (completed / f"{index:04d}.json").write_text(json.dumps({
        "task": task, "success": True, "group": "atomic", "artifact_dir": str(artifact),
    }))
    return artifact


def _pose(distance, fixture="Fridge"):
    return [{"fixture_class": fixture, "reference_keys": ["target"],
             "front_distance_m": distance, "lateral_offset_m": 0.0,
             "facing_error_deg": 0.0, "eef_position_base_m": [0.25, 0.0, 0.6]}]


@pytest.mark.parametrize("task", ["NavigateKitchen", "WalkToCounter"])
def test_navigation_is_excluded_from_all_handoff_statistics_and_recorded(tmp_path, monkeypatch, task):
    read_text = Path.read_text

    def metadata(path, *args, **kwargs):
        if path.name == "task_groups.json":
            return json.dumps({"tasks": {"WalkToCounter": {"category": "navigate"}}})
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", metadata)
    poses = {_record(tmp_path, i, "CloseFridge"): _pose(0.5) for i in range(3)}
    poses[_record(tmp_path, 3, task)] = _pose(-1.141, "StandMixer")
    poses[_record(tmp_path, 4, task)] = None
    monkeypatch.setattr(build_handback, "episode_pose", poses.get)
    summary = build_handback.collect([tmp_path, tmp_path])
    block = build_handback.handback_block(summary)
    assert set(summary["by_task"]) == {"CloseFridge"}
    assert set(summary["by_fixture"]) == {"Fridge"}
    assert set(block["by_task"]) == {"CloseFridge"}
    assert set(block["by_fixture"]) == {"Fridge"}
    assert block["default"]["distance_m"] == [0.5, 0.5]
    assert block["source"]["episodes"] == 3
    assert block["source"]["excluded_tasks"] == [{"task": task, "reason": REASON, "episodes": 2}]


def test_non_navigation_statistics_match_formal_export_bytes(tmp_path, monkeypatch):
    fixture = json.loads((Path(__file__).parent / "fixtures/xiaomi_handback_20261012.json").read_text())
    poses = {_record(tmp_path, i, r["task"]): r["rows"] for i, r in enumerate(fixture["records"])}
    monkeypatch.setattr(build_handback, "episode_pose", poses.get)
    summary = build_handback.collect([tmp_path])
    block = build_handback.handback_block(summary, min_episodes=3)
    actual = {"by_task": summary["by_task"], "by_fixture": summary["by_fixture"],
              "handback": {k: block[k] for k in ("by_task", "by_fixture", "default")}}
    assert json.dumps(actual, indent=2, sort_keys=True).encode() == json.dumps(fixture["expected"], indent=2, sort_keys=True).encode()
    assert block["source"]["excluded_tasks"] == [{"task": "NavigateKitchen", "reason": REASON, "episodes": 13}]
