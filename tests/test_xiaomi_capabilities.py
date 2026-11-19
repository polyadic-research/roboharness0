from copy import deepcopy

import pytest


def test_official_summary_conversion_preserves_counts_and_success_only_timing():
    from robots.robocasa.tools.convert_xiaomi_summary import convert_summary

    summary = {
        "tasks": {
            "OpenStandMixerHead": {
                "num_episodes": 3,
                "successes": 2,
                "success_rate": 2 / 3,
                "horizon": 500,
                "episodes": [
                    {
                        "episode": 0,
                        "global_episode_index": 0,
                        "seed": 7,
                        "success": True,
                        "steps": 100,
                    },
                    {
                        "episode": 1,
                        "global_episode_index": 1,
                        "seed": 8,
                        "success": False,
                        "steps": 500,
                    },
                    {
                        "episode": 2,
                        "global_episode_index": 2,
                        "seed": 9,
                        "success": True,
                        "steps": 200,
                    },
                ],
            }
        }
    }
    record = convert_summary(summary)[0]
    assert record["task"] == "OpenStandMixerHead"
    assert record["group"] == "atomic_seen"
    assert (record["n"], record["successful"]) == (3, 2)
    assert record["first_success_step_median_success_only"] == 150
    assert 0 < record["wilson95_low"] < 2 / 3 < record["wilson95_high"] < 1
    summary["tasks"]["OpenStandMixerHead"]["successes"] = 3
    with pytest.raises(ValueError, match="success"):
        convert_summary(summary)


def test_xiaomi_catalog_uses_complete_measured_rows_and_handback():
    from robots.robocasa.capabilities import load_capabilities
    from robots.robocasa.policy_skills import load_policy_skill
    from robots.robocasa.tools.build_capabilities import check_skill_consistency
    from robots.robocasa.tools.skill_table import render
    from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT

    catalog = load_capabilities("xiaomi-mibot")
    assert catalog is not None
    assert catalog["checkpoint"]["revision"] == XIAOMI_MIBOT_CHECKPOINT.revision
    atomic = {
        task: row for task, row in catalog["rows"].items() if row["family"] == "atomic"
    }
    composite = {
        task: row
        for task, row in catalog["rows"].items()
        if row["family"] == "composite"
    }
    assert len(atomic) == 65 and len(composite) == 5
    assert all(row["episodes"] == 15 for row in atomic.values())
    assert all(row["episodes"] == 5 for row in composite.values())
    assert sum(row["successes"] for row in atomic.values()) == 746
    assert sum(row["successes"] for row in composite.values()) == 11
    assert atomic["NavigateKitchen"]["successes"] == 13
    assert atomic["NavigateKitchen"]["typical_steps"] == 159
    assert all(
        row["source"] in {"xiaomi-capability-20261012", "xiaomi-capability-round2-20261012"}
        for row in catalog["rows"].values()
    )
    assert not catalog.get("declared")
    handback = catalog["handback"]
    assert handback["provisional"]
    assert handback["by_fixture"] and handback["by_task"]
    assert "NavigateKitchen" not in handback["by_task"]
    assert handback["source"]["episodes"] == 576
    assert handback["source"]["excluded_tasks"] == [{
        "task": "NavigateKitchen",
        "reason": "navigation start pose is not a fixture-facing handoff pose",
        "episodes": 13,
    }]
    assert all(row["episodes"] >= 3 for row in handback["by_fixture"].values())
    assert set(handback["by_task"]) <= set(atomic)
    assert all(
        0 < row["episodes"] <= atomic[task]["successes"]
        for task, row in handback["by_task"].items()
    )
    text = load_policy_skill("xiaomi-mibot")
    assert {"## Instruction format", "## Control modes", "## Decisions"} <= set(text.splitlines())
    assert "xiaomi-capability-20261012" in text
    assert "交接位姿待测量" not in text and "正式表见纯评测任务" not in text
    assert "pour_into" in text
    assert check_skill_consistency(catalog, text, require_all_atomic=True) == 65
    assert render(catalog, text) == text


def test_xiaomi_rebuild_retains_measured_handback_without_new_pose_data():
    from robots.robocasa.capabilities import load_capabilities, validate_capabilities
    from robots.robocasa.tools.build_capabilities import merge_catalog

    catalog = load_capabilities("xiaomi-mibot")
    assert validate_capabilities(catalog, policy="xiaomi-mibot") == catalog
    original = deepcopy(catalog)
    incoming = deepcopy(catalog)
    incoming.pop("handback")
    rebuilt = merge_catalog(catalog, incoming)
    assert rebuilt["handback"] == original["handback"]
    assert rebuilt["rows"] == original["rows"]
    assert catalog == original
