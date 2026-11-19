"""Bounded placement finishing leases, on the 1.2.3 motion interface."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.harness.state import operation_key
from tests.test_harness_1011_control import NONE_TAKEOVER, act, machine, named, step
from tests.test_robocasa_motion_handoff import MotionController


def facts(*, held=True, move=True, release=False, unstable=False):
    before = {
        "eef_position": [0.2, 0, 0.8],
        "base_position": [0, 0, 0],
        "base_quaternion": [0, 0, 0, 1],
        "gripper_opening": 0.02,
    }
    after = deepcopy(before)
    if move:
        after["eef_position"][2] -= 0.02
    after["gripper_opening"] = 0.08 if release else 0.03 if unstable else 0.02
    return {
        "measured_before": before,
        "measured_after": after,
        "waypoint_results": [{"target_position": after["eef_position"]}],
        "grasp_status": {"retention_monitor_active": held},
        "gripper_command_executed": release,
        **(
            {
                "gripper_settle": {
                    "command": "open",
                    "stable": True,
                    "release_confirmed": True,
                }
            }
            if release
            else {}
        ),
    }


def started(*, bad=None, horizon=3900):
    events = []
    s = machine(events, horizon=horizon)
    assert s.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(s, 200, tool="move_eef_path", step_id=3500, raw=facts())
    act(s, 200, tool="move_eef_path", step_id=3700, raw=facts(**(bad or {})))
    return s, events


def test_two_verified_moving_stable_holds_grant_a_short_lease():
    s, e = started()
    assert not s.needs_replan
    [grant] = named(e, "control_finishing_lease_granted")
    assert grant["granted_steps"] == 50 and grant["borrowed_total"] == 50
    assert s.finishing_action_budget() == 50
    segment = s.control_planning_context(include_capabilities=False)[
        "control_segments"
    ][0]
    assert (
        segment["step_quota"] == 450 and segment["call_quota"] == 4
    )  # actual calls 2 plus 2


@pytest.mark.parametrize("bad", [{"move": False}, {"held": False}, {"unstable": True}])
def test_holding_only_or_unstable_or_empty_cannot_borrow(bad):
    s, e = started(bad=bad)
    assert s.needs_replan and not named(e, "control_finishing_lease_granted")
    assert named(e, "control_mode_fallback")


def test_existing_progress_verdict_without_public_motion_does_not_extend_nominal_quota():
    e = []
    s = machine(e, horizon=3900)
    s.register_plan("goal", [step(**NONE_TAKEOVER)])
    act(s, 490, step_id=3500)
    act(s, 20, step_id=3520)
    assert named(e, "control_mode_fallback") and not named(
        e, "control_finishing_lease_granted"
    )


def test_release_then_one_withdrawal_lease_without_forcing_retention_after_opening():
    s, e = started()
    act(
        s,
        20,
        tool="move_eef_path",
        step_id=3720,
        raw=facts(held=False, move=False, release=True),
    )
    assert not s.needs_replan
    act(s, 30, tool="move_eef_path", step_id=3750, raw=facts(held=False, release=True))
    assert not s.needs_replan
    assert named(e, "control_finishing_lease_granted")[-1]["phase"] == "withdraw"
    act(s, 50, tool="move_eef_path", step_id=3800, raw=facts(held=False, release=True))
    assert s.needs_replan  # no indefinite empty-hand withdrawal leases


def test_unproductive_physical_call_ends_an_active_lease_immediately():
    s, e = started()
    act(s, 5, tool="move_eef_path", step_id=3705, raw=facts(move=False))
    assert named(e, "control_mode_fallback")[-1]["reason"] == "finishing_progress_lost"


def test_grasp_loss_overrides_a_claimed_progress_verdict():
    s, e = started()
    raw = facts()
    raw["grasp_events"] = [{"type": "grasp_lost", "release_commanded": False}]
    act(s, 5, tool="move_eef_path", step_id=3705, raw=raw)
    assert named(e, "control_mode_fallback")[-1]["reason"] == "finishing_progress_lost"


def test_borrowing_is_capped_and_preserves_the_original_horizon_and_policy_floor():
    s, e = started()
    for global_step in [3750, 3800, 3850]:
        act(s, 50, tool="move_eef_path", step_id=global_step, raw=facts())
    assert s.needs_replan
    grants = named(e, "control_finishing_lease_granted")
    assert sum(g["granted_steps"] for g in grants) == 150
    assert s._last_step_id == 3850 and s._episode_horizon == 3900


def test_tool_motion_budget_honors_the_active_lease_including_its_stop():
    s, _ = started()
    c = MotionController()
    tk = RoboCasaToolkit(
        c, initial_observation=c.observation, instruction="goal", episode_horizon=3900
    )
    tk._harness = SimpleNamespace(state=s)
    tk._operation._metrics["step_count"] = 3700
    assert tk._motion_action_budget(150) == 50


def test_same_operation_replanning_does_not_refund_borrowed_reserve():
    s, _ = started()
    key = operation_key(s.active_step)
    act(s, 0, verdict="retry", failure_type="timeout", step_id=3700)
    assert s.replan(
        "same operation, changed audit id", [step(id="new", **NONE_TAKEOVER)]
    )["ok"]
    assert s._finishing_borrowed[key] == 50
    assert s.finishing_action_budget() == 50


def test_missing_horizon_cannot_borrow_and_a_pass_needs_no_lease():
    e = []
    s = machine(e)
    s.register_plan("goal", [step(**NONE_TAKEOVER)])
    act(s, 250, tool="move_eef_path", raw=facts())
    act(s, 250, tool="move_eef_path", raw=facts())
    assert s.needs_replan and not named(e, "control_finishing_lease_granted")
    e = []
    s = machine(e, horizon=3900)
    s.register_plan("goal", [step(**NONE_TAKEOVER)])
    act(s, 500, tool="move_eef_path", step_id=3700, raw=facts(), verdict="pass")
    assert not named(e, "control_finishing_lease_granted") and not named(
        e, "control_mode_fallback"
    )


def test_active_finishing_lease_blocks_policy_and_base_tools_but_keeps_primitive_motion():
    from rpent.harness.gate import PlanSafetyGate

    s, _ = started()
    assert s.finishing_tool_violation("continue_vla")
    assert s.finishing_tool_violation("navigate_to")
    assert s.finishing_tool_violation("move_eef_path") is None
    assert s.finishing_tool_violation("move_pose")
    # Existing allowlists and pending verification keep precedence in the gate.
    assert PlanSafetyGate(mode="plan-gated")._physical_violation("navigate_to", s)


def test_open_to_close_is_refused_during_finishing_even_with_an_old_approval():
    s, _ = started()
    c = MotionController()
    tk = RoboCasaToolkit(
        c, initial_observation=c.observation, instruction="goal", episode_horizon=3900
    )
    tk._harness = SimpleNamespace(state=s)
    tk._last_executed_gripper = "open"
    with pytest.raises(ValueError, match="new grasp"):
        tk._require_reviewed_direct_close("close")
    assert c.actions == []


@pytest.mark.parametrize(
    "problem", ["away", "base", "nan", "conflict", "release_without_confirmation"]
)
def test_invalid_or_conflicting_feedback_cannot_grant_a_lease(problem):
    events = []
    s = machine(events, horizon=3900)
    s.register_plan("goal", [step(**NONE_TAKEOVER)])
    act(s, 200, tool="move_eef_path", step_id=3500, raw=facts())
    raw = facts()
    failure = None
    if problem == "away":
        raw["waypoint_results"][0]["target_position"] = [0.2, 0, 1]
    elif problem == "base":
        raw["measured_after"]["base_position"] = [0.1, 0, 0]
    elif problem == "nan":
        raw["measured_after"]["eef_position"][0] = float("nan")
    elif problem == "conflict":
        failure = "wrong_destination"
    else:
        raw["measured_after"]["gripper_opening"] = 0.08
    act(
        s,
        200,
        tool="move_eef_path",
        step_id=3700,
        raw=raw,
        failure_type=failure,
        verdict="fail" if failure else "progress",
    )
    assert s.needs_replan and not named(events, "control_finishing_lease_granted")
