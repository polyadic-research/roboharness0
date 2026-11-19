"""Command-aware retention monitoring across release and reacquisition."""

import numpy as np
import pytest

from robots.robocasa.grasp_progress import GraspProgress


def state(width, z=0.48):
    result = np.zeros(16, dtype=np.float64)
    result[9] = z
    result[14:16] = [width / 2, -width / 2]
    return result


def command(gripper):
    result = np.zeros(12)
    result[11] = gripper
    return result


def confirmed_progress():
    progress = GraspProgress(
        state(0.08, z=0.5),
        descent_threshold_m=0.05,
        ascent_threshold_m=0.05,
        closed_threshold=0.03,
        empty_threshold=0.002,
        open_threshold=0.06,
    )
    assert progress.observe(state(0.02, z=0.41), command(1.0)) is None
    assert progress.observe(state(0.02), command(1.0)) is None
    assert progress.confirmed and progress.retention_monitor_active
    return progress


def test_fast_narrowing_only_alerts_if_width_collapses_to_empty():
    progress = confirmed_progress()
    assert progress.observe(state(0.008), command(1.0)) is None
    assert progress.retention_monitor_active
    event = progress.observe(state(0.002), command(1.0))
    assert event["reason"] == "empty_closure"
    assert event["release_commanded"] is False
    assert not progress.retention_monitor_active


@pytest.mark.parametrize("action", [None, np.zeros(11), command(-1.0), command(0.0), command(0.49)])
def test_width_increase_requires_a_close_command_to_alert(action):
    progress = confirmed_progress()
    assert progress.observe(state(0.035), action) is None
    assert progress.retention_monitor_active


@pytest.mark.parametrize("gripper", [0.5, 1.0])
def test_width_increase_at_threshold_alerts_under_close_command(gripper):
    progress = confirmed_progress()
    event = progress.observe(state(0.03), command(gripper))
    assert event["reason"] == "width_jump"
    assert event["release_commanded"] is False


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gradual_release_disarms_at_open_threshold_even_after_closure_reset(dtype):
    progress = confirmed_progress()
    for width in (0.028, 0.036, 0.044, 0.052, 0.059):
        assert progress.observe(state(width), command(0.0)) is None
        assert progress.retention_monitor_active
    assert progress.closure_state is None
    assert progress.observe(state(0.06).astype(dtype), command(0.0)) is None
    assert not progress.retention_monitor_active
    assert not progress.confirmed


@pytest.mark.parametrize("width", [0.06, 0.08])
def test_fast_commanded_release_resets_instead_of_alerting(width):
    progress = confirmed_progress()
    assert progress.observe(state(width), command(0.0)) is None
    assert not progress.retention_monitor_active
    assert not progress.confirmed


def test_gradual_widening_under_close_command_keeps_empty_loss_detection():
    progress = confirmed_progress()
    for width in (0.028, 0.036, 0.044, 0.052, 0.06):
        assert progress.observe(state(width), command(1.0)) is None
        assert progress.retention_monitor_active
    event = progress.observe(state(0.0), command(1.0))
    assert event["reason"] == "empty_closure"
