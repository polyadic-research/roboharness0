"""D06 regressions: wide grasps, releases and operation-local review gates."""

from types import SimpleNamespace

import pytest

from rpent.harness import Harness
from tests.test_robocasa_grasp_progress import command, state
from tests.test_robocasa_preclose_review import action, call, review, setup


def wide_progress():
    from robots.robocasa.grasp_progress import GraspProgress

    return GraspProgress(
        state(0.08, z=0.5), descent_threshold_m=0.05,
        ascent_threshold_m=0.05, closed_threshold=0.03,
        empty_threshold=0.002, open_threshold=0.06,
    )


@pytest.mark.parametrize("width", [0.06, 0.0602, 0.063])
def test_sustained_stable_wide_close_then_lift_confirms_motion(width):
    progress = wide_progress()
    for offset in (0, 0.0002, -0.0002, 0.0001, 0):
        progress.observe(state(width + offset, z=0.41), command(1))
    assert progress.closure_confirmed
    assert not progress.confirmed
    progress.observe(state(width, z=0.48), command(1))
    assert progress.confirmed and progress.retention_monitor_active
    assert progress.diagnostics()["closure_evidence"] == "stable_close_command"
    cycle = progress.cycle
    assert progress.observe(state(0.07, z=0.48), command(-1)) is None
    assert not progress.confirmed and not progress.retention_monitor_active
    assert progress.cycle > cycle


@pytest.mark.parametrize("widths", [
    [0.0602] * 4, [0.08] * 6,
    [0.07, 0.066, 0.062, 0.058, 0.054, 0.05],
    [0.06, 0.063, 0.06, 0.063, 0.06, 0.063],
])
def test_wide_close_needs_duration_stability_and_measured_narrowing(widths):
    progress = wide_progress()
    for width in widths:
        progress.observe(state(width, z=0.41), command(1))
    progress.observe(state(widths[-1], z=0.48), command(1))
    assert not progress.confirmed


def test_lift_before_stable_wide_close_is_not_reused():
    progress = wide_progress()
    progress.observe(state(0.08, z=0.41), command(-1))
    progress.observe(state(0.08, z=0.48), command(-1))
    for _ in range(6):
        progress.observe(state(0.0602, z=0.48), command(1))
    assert progress.closure_confirmed and not progress.confirmed


def test_wide_closure_must_stay_stable_until_lift_and_then_monitor_loss():
    progress = wide_progress()
    for _ in range(5):
        progress.observe(state(0.0602, z=0.41), command(1))
    progress.observe(state(0.064, z=0.48), command(1))
    assert not progress.confirmed
    for _ in range(5):
        progress.observe(state(0.064, z=0.41), command(1))
    progress.observe(state(0.064, z=0.48), command(1))
    assert progress.confirmed
    event = progress.observe(state(0.0, z=0.48), command(1))
    assert event["reason"] == "empty_closure"


@pytest.mark.parametrize("gripper", [-1, 0, 0.49])
def test_open_command_breaks_a_wide_close_run(gripper):
    progress = wide_progress()
    for _ in range(4):
        progress.observe(state(0.0602, z=0.41), command(1))
    progress.observe(state(0.0602, z=0.41), command(gripper))
    progress.observe(state(0.0602, z=0.48), command(1))
    assert not progress.confirmed


@pytest.mark.parametrize("tool", ["smolvla_pick", "continue_vla", "execute_vla"])
@pytest.mark.parametrize("old_review", ["rejected", "approved", "pending"])
def test_changed_effective_instruction_resets_gate_before_first_close(tool, old_review):
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    if old_review == "approved":
        review(toolkit)
    elif old_review == "rejected":
        review(toolkit, between=False)
        review(toolkit, between=False)
    old = toolkit._pick_progress
    old.observe(state(0.08, z=0.16), command(-1))
    cycle = old.cycle
    kwargs = {"instruction": "Place the bowl in the microwave.", "intervene": True}
    if tool != "execute_vla":
        kwargs["max_chunks"] = 1
    result = call(toolkit, tool, **kwargs)
    assert result["stop_reason"] == "preclose_review_required"
    assert simulator.actions == []
    assert toolkit._grasp_review_status()["consecutive_rejections"] == 0
    assert toolkit._grasp_review_status()["approved_step"] is None
    assert old.cycle > cycle and not old.descent_confirmed


def test_same_instruction_and_audit_notes_preserve_pending_cycle():
    toolkit, _, _ = setup([action(close=True)], [action(close=True)])
    call(toolkit, "smolvla_pick")
    review(toolkit, between=False)
    cycle = toolkit._pick_progress.cycle
    call(toolkit, "smolvla_pick", instruction="A different audit note.", intervene=False)
    assert toolkit._pick_progress.cycle == cycle
    assert toolkit._grasp_review_status()["consecutive_rejections"] == 1
    toolkit._reviewed_policy_instruction(toolkit._active_policy_instruction, True)
    assert toolkit._pick_progress.cycle == cycle


@pytest.mark.parametrize("strict", [False, True])
def test_plan_operation_change_with_same_policy_language_resets_review(tmp_path, strict):
    toolkit, simulator, _ = setup([action(close=True)], [action(close=True)])
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    # Both dispatch paths expose the same authoritative active plan state.
    if strict:
        toolkit._harness_session = SimpleNamespace(controller=harness)
        harness.set_physical_tools({"smolvla_pick"})
    else:
        toolkit.attach_harness(harness)
    assert harness.state.register_plan("Two operations.", [
        {"id": "broccoli", "intent": "Place broccoli in bowl.",
         "allowed_tools": ["smolvla_pick"], "max_attempts": 8},
        {"id": "bowl", "intent": "Place bowl in microwave.",
         "allowed_tools": ["smolvla_pick"], "max_attempts": 8},
    ])["ok"]
    toolkit.smolvla_pick(max_chunks=1)
    toolkit.observe()
    for _ in range(2):
        toolkit.review_grasp(expected_step=0, object_between_fingers=False,
                             narrow_axis_aligned=False, evidence="Empty fingers.")
    cycle = toolkit._pick_progress.cycle
    # Model the authoritative plan advancing after completed verification.
    harness.state._active_index = 1
    result = toolkit.smolvla_pick(max_chunks=1)
    assert result["stop_reason"] == "preclose_review_required"
    assert simulator.actions == []
    assert toolkit._pick_progress.cycle > cycle
    assert toolkit._grasp_review_status()["consecutive_rejections"] == 0


def test_sustained_wide_close_release_does_not_request_preclose_review():
    toolkit, simulator, _ = setup(
        [action(dz=-1.2), action(close=True)], [action(close=True)] * 5,
        [action()], [action(close=True)],
    )
    original_step = simulator.step

    def wide_step(value):
        transition = original_step(value)
        if value[11] >= 0.5:
            transition.observation["state"][14:16] = (0.0301, -0.0301)
        return transition

    simulator.step = wide_step
    call(toolkit, "smolvla_pick")
    review(toolkit)
    call(toolkit, "continue_vla", max_chunks=2)
    assert toolkit._pick_progress.closure_confirmed
    released = call(toolkit, "execute_vla")
    assert released["executed_actions"] == 1
    status = toolkit.observe()["grasp_review"]
    assert not status["required"] and status["consecutive_rejections"] == 0
    closed = call(toolkit, "smolvla_pick", intervene=True,
                  instruction="Place the bowl in the microwave.")
    assert closed["stop_reason"] == "preclose_review_required"
    assert closed["executed_actions"] == 0
