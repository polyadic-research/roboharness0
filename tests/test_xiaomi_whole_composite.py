"""Xiaomi skill: a long task whose own composite row reaches 30% is delivered whole (10/06 rule, made explicit 10/12)."""

import re

import pytest

from robots.robocasa.capabilities import capability_summary, load_capabilities
from robots.robocasa.policy_skills import load_policy_skill
from rpent.harness.state import PlanStateMachine
from tests.test_harness_1011_control import TOOLS, act, named

MODE_BY_TIER = {"reliable": "delegate", "unstable": "monitor"}


def _rule_text() -> str:
    text = load_policy_skill("xiaomi-mibot")
    start = text.index("- **Long, delivered whole**")
    ends = [i for i in (text.find("\n- ", start + 1), text.find("\n\n", start + 1)) if i > 0]
    return text[start:min(ends)]


def _eligible_composites() -> list[tuple[str, str]]:
    rows = load_capabilities("xiaomi-mibot")["rows"]
    eligible = [(name, row["tier"]) for name, row in sorted(rows.items())
                if row["family"] == "composite" and row["tier"] in MODE_BY_TIER]
    assert eligible, "the Xiaomi catalog has no composite row at or above 30%"
    return eligible


def _ineligible_composites() -> list[str]:
    rows = load_capabilities("xiaomi-mibot")["rows"]
    return [name for name, row in sorted(rows.items())
            if row["family"] == "composite" and row["tier"] not in MODE_BY_TIER]


def test_rule_is_generic_and_states_the_threshold_and_the_way_back():
    rule = _rule_text()
    for phrase in ("own composite row", "reliable or unstable", "**30%**", "classify the task",
                   "as short", "task_language verbatim", "every", "control_mode_escalations",
                   "weak own row"):
        assert phrase in rule, phrase
    names = [name for name, row in load_capabilities("xiaomi-mibot")["rows"].items() if row["family"] == "composite"]
    assert not [name for name in names if name in rule]
    assert re.search(r"\bD\d\d\b", rule) is None and "seed" not in rule
    assert "Future policies may permit" not in load_policy_skill("xiaomi-mibot")


@pytest.mark.parametrize("task,tier", _eligible_composites())
def test_planner_sees_the_own_row_and_a_one_step_whole_task_plan_registers(task, tier):
    summary = capability_summary(load_capabilities("xiaomi-mibot"), task_name=task)
    [own] = [row for row in summary if row["skill_row"] == task]
    assert own["family"] == "composite" and own["tier"] == tier
    state = PlanStateMachine(max_replans=40, physical_tools=TOOLS, policy_family="xiaomi-mibot", task_name=task)
    whole = {"id": "whole", "intent": f"Original sentence. row={task}", "allowed_tools": ["continue_vla"],
             "max_attempts": 30, "operation": "whole_task", "skill_row": task, "tier": tier,
             "control_mode": MODE_BY_TIER[tier]}
    assert state.register_plan("goal", [whole])["ok"]


def test_a_tier_mismatch_on_the_own_row_is_still_rejected():
    task, tier = _eligible_composites()[0]
    state = PlanStateMachine(max_replans=40, physical_tools=TOOLS, policy_family="xiaomi-mibot", task_name=task)
    wrong = "unstable" if tier == "reliable" else "reliable"
    result = state.register_plan("goal", [{"id": "whole", "intent": "x", "allowed_tools": ["continue_vla"],
                                           "max_attempts": 30, "operation": "whole_task", "skill_row": task,
                                           "tier": wrong, "control_mode": MODE_BY_TIER[wrong]}])
    assert not result.get("ok") and "requires tier" in result.get("error", "")


def test_whole_task_step_escalates_on_a_progress_window_not_a_step_threshold():
    task, tier = _eligible_composites()[0]
    events = []
    state = PlanStateMachine(max_replans=40, physical_tools=TOOLS, policy_family="xiaomi-mibot", task_name=task,
                             record=lambda event, **payload: events.append({"event": event, **payload}))
    whole = {"id": "whole", "intent": "x", "allowed_tools": ["continue_vla"], "max_attempts": 30,
             "operation": "whole_task", "skill_row": task, "tier": tier, "control_mode": MODE_BY_TIER[tier]}
    assert state.register_plan("goal", [whole])["ok"]
    act(state, 60, tool="continue_vla")  # progress resets the window, however long the task is
    assert not named(events, "control_mode_escalation_available")
    act(state, 130, tool="continue_vla", verdict="retry", failure_type="timeout")
    assert not named(events, "control_mode_escalation_available")
    [offer] = named(events, "policy_semantic_replanning_available")
    assert offer["reason"] == "current_goal_stall"
    assert state.semantic_policy_replanning_allowed and state.needs_replan


def test_weak_composites_are_not_eligible():
    rule = _rule_text()
    assert "weak own row" in rule and "long from the start" in rule
    for task in _ineligible_composites():
        assert load_capabilities("xiaomi-mibot")["rows"][task]["tier"] not in MODE_BY_TIER
