"""1.0.11 capability data: training start poses (D1) and generated skill tables (D2)."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest

import robots.robocasa.policy_skills as skills
from robots.robocasa.capabilities import load_capabilities, validate_capabilities
from robots.robocasa.tools import build_handback, skill_table

SCENE = """<mujoco><worldbody>
  <body name="fridge_main" pos="1.0 0 0.9">
    <geom name="g0" type="mesh" mesh="m"/>
    <geom name="g1" type="box" size="0.3 0.25 0.4" pos="0 0 0"/>
    <body name="fridge_door" pos="0 -0.25 0">
      <geom name="g2" type="box" size="0.3 0.02 0.4" pos="0 -0.02 0"/>
    </body>
  </body>
</worldbody></mujoco>"""


def write_episode(root, index, *, success=True, group="atomic", base=(0.0, 0.0), eef=(0.25, 0.0, 0.6)):
    attempt = root / "rounds" / "r1" / "episodes" / f"{index:04d}-Task-{index:02d}" / "attempts" / "t"
    attempt.mkdir(parents=True)
    (attempt / "initial.xml").write_text(SCENE)
    (attempt / "handoff-pose.json").write_text(json.dumps({
        "robot_base_position_world_m": [*base, 0.7], "robot_base_yaw_world_rad": 0.0,
        "fixtures": [{"body": "fridge_main", "class": "FridgeFrenchDoor", "task_reference": True,
                      "reference_keys": ["fridge"], "position_world_m": [1.0, 0.1, 0.9],
                      "yaw_world_rad": -math.pi / 2, "distance_center_xy_m": math.hypot(1.0 - base[0], 0.1 - base[1]),
                      "bearing_relative_robot_rad": math.atan2(0.1 - base[1], 1.0 - base[0]),
                      "fixture_yaw_relative_robot_rad": -math.pi / 2},
                     {"body": "wall", "class": "Wall", "task_reference": False}],
    }))
    # Capability-benchmark trace layout: EEF first, base at [7:10].
    state = np.array([*eef, 0, 0, 0, 1, *base, 0.7, 0, 0, 0, 1, 0.04, -0.04], dtype=np.float32)
    np.savez(attempt / "trace.npz", states=state[None], actions=np.zeros((1, 12)))
    completed = root / "rounds" / "r1" / "completed"
    completed.mkdir(parents=True, exist_ok=True)
    (completed / f"{index:04d}.json").write_text(json.dumps({
        "task": "PickPlaceFridgeDrawerToShelf", "group": group, "success": success,
        "artifact_dir": f"/remote/x/rounds/r1/episodes/{index:04d}-Task-{index:02d}/attempts/t",
    }))


def test_front_face_includes_closed_doors_of_nested_bodies():
    import xml.etree.ElementTree as ET

    assert build_handback.fixture_front_offset(ET.fromstring(SCENE), "fridge_main") == pytest.approx(-0.29)


def test_handback_uses_successful_atomic_start_poses_only(tmp_path):
    write_episode(tmp_path, 1)
    write_episode(tmp_path, 2, base=(0.1, 0.0))
    write_episode(tmp_path, 3, success=False, base=(0.5, 0.0))
    write_episode(tmp_path, 4, group="composite", base=(0.5, 0.0))
    summary = build_handback.collect([tmp_path])
    fridge = summary["by_fixture"]["FridgeFrenchDoor"]
    assert fridge["episodes"] == 2 and fridge["tasks"] == ["PickPlaceFridgeDrawerToShelf"]
    # Front face at x = 1.0 - 0.29: the bases at x = 0 and 0.1 stand 0.71 and 0.61 m away.
    assert fridge["front_distance_m"]["median"] == pytest.approx(0.66)
    assert fridge["facing_error_deg"]["median"] == pytest.approx(0.0)
    assert fridge["eef_position_base_m"]["median"] == pytest.approx([0.25, 0.0, 0.6])
    block = build_handback.handback_block(summary, min_episodes=2)
    entry = block["by_fixture"]["FridgeFrenchDoor"]
    assert entry["eef_start_base_m"] == [0.25, 0.0, 0.6] and entry["episodes"] == 2
    assert block["by_task"]["PickPlaceFridgeDrawerToShelf"]["fixture"] == "FridgeFrenchDoor"
    assert block["source"]["sources"] == [{"root": tmp_path.name, "successful_atomic_episodes": 2}]
    catalog = deepcopy(load_capabilities("xiaomi-mibot"))
    validate_capabilities({**catalog, "handback": block})


@pytest.mark.parametrize("field,value", [
    ("lateral_m", [0.2, -0.2]), ("eef_start_base_m", [0.2, 0.0]), ("episodes", 0),
    ("front_distance_m_median", -0.1),
])
def test_handback_limits_are_validated(field, value):
    catalog = deepcopy(load_capabilities("xiaomi-mibot"))
    catalog["handback"]["by_fixture"]["Microwave"][field] = value
    with pytest.raises(ValueError, match="handback"):
        validate_capabilities(catalog)


def test_shipped_capabilities_carry_the_measured_handback():
    handback = load_capabilities("xiaomi-mibot")["handback"]
    assert handback["provisional"] is True
    assert {"Microwave", "FridgeFrenchDoor", "Sink", "Counter"} <= set(handback["by_fixture"])
    for limits in handback["by_fixture"].values():
        assert limits["episodes"] >= 3 and len(limits["eef_start_base_m"]) == 3
    assert handback["by_task"]["PickPlaceFridgeDrawerToShelf"]["fixture"].startswith("Fridge")


def test_shipped_skill_tables_match_the_capability_json():
    catalog = load_capabilities("xiaomi-mibot")
    text = skills.load_policy_skill("xiaomi-mibot")
    assert skill_table.render(catalog, text) == text
    assert skill_table.HANDBACK_BEGIN in text and "| Microwave |" in text


def test_generator_updates_counts_keeps_attention_and_adds_new_rows():
    catalog = deepcopy(load_capabilities("xiaomi-mibot"))
    text = skills.load_policy_skill("xiaomi-mibot")
    row = catalog["rows"]["CloseMicrowave"]
    row.update(successes=11, episodes=15, tier="reliable", wilson95=[0.48, 0.89], crosses_threshold=True)
    catalog["rows"]["OpenDrawerNew"] = {**catalog["rows"]["OpenDrawer"]}
    with pytest.raises(ValueError, match="registry instruction"):
        skill_table.render_operation_table(catalog, text)
    del catalog["rows"]["OpenDrawerNew"]
    rendered = skill_table.render_operation_table(catalog, text)
    line = next(line for line in rendered.splitlines() if line.startswith("| CloseMicrowave:"))
    original_line = next(item for item in text.splitlines() if item.startswith("| CloseMicrowave:"))
    assert "| Reliable* (11/15) |" in line
    assert line.rsplit("|", 2)[1] == original_line.rsplit("|", 2)[1]
    # A row missing from the Markdown is appended with the default attention text.
    removed_text = "\n".join(line for line in text.splitlines() if not line.startswith("| OpenFridge:"))
    regenerated = skill_table.render_operation_table(catalog, removed_text)
    original_fridge = next(item for item in text.splitlines() if item.startswith("| OpenFridge:"))
    assert original_fridge in regenerated
