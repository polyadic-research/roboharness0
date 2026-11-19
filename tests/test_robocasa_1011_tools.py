"""1.0.11 tool fixes: takeover grasp review, held-object closure, multi-step
closure, holonomic navigation, near-repeat warning and zoom provenance
(backlog A1-A3, B4, C1, C4)."""

import hashlib
import json

import numpy as np
import pytest

from robots.robocasa.privileged_tools import GRIPPER_SETTLE_MAX_STEPS
from robots.robocasa.toolkit import _rotate_base_quaternion
from rpent.episode.contracts import StepResult
from tests.test_robocasa_motion_handoff import call
from tests.test_robocasa_privileged_tools import (
    PublicController,
    args,
    seed_closed,
    setup,
)


def close_action():
    action = np.zeros(12, np.float32)
    action[11] = 1
    return action


# --- A1: a takeover can open its own review; every open-to-close is reviewed -------------


def test_takeover_review_without_policy_pick_then_close_at_the_reviewed_pose():
    toolkit, controller = setup()
    toolkit.observe()
    assert not toolkit._preclose_active
    error = toolkit.execute_tool("move_eef_path", args("move_eef_path", waypoints=[{"position": [0.2, 0.1, 0.5]}],
                                                       gripper="close")).result["error"]
    assert "review_grasp" in error and controller.actions == []
    review = call(toolkit, "review_grasp", expected_step=0, object_between_fingers=True,
                  narrow_axis_aligned=True, evidence="Pan handle centred between the fingers.")
    assert review["approved"] and toolkit._preclose_active
    result = call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=[{"position": [0.2, 0.1, 0.5]}],
                                                    gripper="close"))
    assert result["gripper_command_executed"] and controller.actions[-1][11] >= 0.5


def test_rejected_review_never_authorizes_a_direct_close():
    toolkit, controller = setup()
    toolkit.observe()
    for _ in range(2):
        review = call(toolkit, "review_grasp", expected_step=0, object_between_fingers=False,
                      narrow_axis_aligned=False, evidence="Fingers beside the handle.")
    assert review["policy_close_allowed"] and not review["approved"]
    assert "direct close stays unauthorized" in review["next_action"]
    assert "review_grasp" in toolkit.execute_tool("navigate_to", args("navigate_to", gripper="close")).result["error"]


# --- A2: keeping a closure is not a new closure ---------------------------------------------


@pytest.mark.parametrize("tool", ["navigate_to", "move_eef_path", "adjust_torso_to"])
def test_held_object_keeps_closing_without_review(tool):
    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    toolkit._preclose_active = True  # a stale review cycle from an earlier pick
    gripper = {} if tool == "adjust_torso_to" else {"gripper": "close"}
    result = call(toolkit, tool, **args(tool, expected_step=controller.step, **gripper))
    assert all(action[11] >= 0.5 for action in controller.actions[1:])
    assert result["holding_speed_limited"]


def test_review_is_refused_while_the_gripper_is_already_closed():
    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    with pytest.raises(ValueError, match="already closed"):
        toolkit.review_grasp(expected_step=controller.step, object_between_fingers=True,
                             narrow_axis_aligned=True, evidence="Bowl held.")


def test_policy_guard_lets_a_held_closure_continue_after_a_step_change():
    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    toolkit._preclose_active = True
    toolkit._preclose_closing = False  # what a plan-step change used to leave behind
    observation = {"state": controller.observation["state"].copy()}
    assert toolkit._guard_grasp_close(observation, close_action()) is None
    toolkit._operation.execute_actions((np.zeros(12, np.float32),))  # opened again
    assert toolkit._guard_grasp_close(observation, close_action()) == "preclose_review_required"


def test_step_change_keeps_retention_watch_of_a_held_object():
    toolkit, controller = setup()
    seed_closed(toolkit, controller)
    toolkit._reset_grasp_operation()
    assert toolkit._pick_progress.retention_monitor_active
    toolkit._operation.execute_actions((np.zeros(12, np.float32),))  # released
    toolkit._reset_grasp_operation()
    assert not toolkit._pick_progress.retention_monitor_active


# --- A3: a direct close is held until the opening settles -----------------------------------


class ClosingController(PublicController):
    """Fingers close 15 mm per close command until they meet an object width."""

    def __init__(self, stop_at):
        super().__init__()
        self.stop_at = stop_at
        self.observation["state"][14:16] = [0.04, -0.04]

    def execute_action(self, action, *, _action_index=0):
        opening = float(np.abs(self.observation["state"][14:16]).sum())
        super().execute_action(action, _action_index=_action_index)
        if action[11] >= 0.5:
            opening = max(self.stop_at, opening - 0.015)
        else:
            opening = min(0.08, opening + 0.015)
        self.observation["state"][14:16] = [opening / 2, -opening / 2]
        return StepResult(episode_id="test-motion", step_id=self.step, observation=self.observation,
                          reward=0, success=False, terminated=False, truncated=False)


@pytest.mark.parametrize("stop_at,confirmed,empty", [(0.03, True, False), (0.0, False, True)])
def test_close_settles_over_several_steps_and_reports_the_opening(stop_at, confirmed, empty):
    controller = ClosingController(stop_at)
    from robots.robocasa.toolkit import RoboCasaToolkit

    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Pick up the pan.", episode_horizon=500)
    toolkit.observe()
    call(toolkit, "review_grasp", expected_step=0, object_between_fingers=True, narrow_axis_aligned=True,
         evidence="Handle between the fingers.")
    position = controller.observation["state"][7:10].tolist()
    result = call(toolkit, "move_eef_path", expected_step=0, waypoints=[{"position": position}], gripper="close")
    settle = result["gripper_settle"]
    assert 5 < settle["steps"] <= GRIPPER_SETTLE_MAX_STEPS and settle["stable"]
    assert settle["opening_before_m"] == pytest.approx(0.08)
    assert settle["opening_after_m"] == pytest.approx(stop_at)
    assert settle["closure_confirmed"] is confirmed and settle["closed_empty"] is empty
    assert sum(action[11] >= 0.5 for action in controller.actions) == settle["steps"] + 1


def test_release_settles_until_open():
    controller = ClosingController(0.03)
    from robots.robocasa.toolkit import RoboCasaToolkit

    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Place the pan.", episode_horizon=500)
    toolkit._operation.execute_actions((close_action(),) * 4)
    toolkit.observe()
    position = controller.observation["state"][7:10].tolist()
    result = call(toolkit, "move_eef_path", expected_step=controller.step, waypoints=[{"position": position}],
                  gripper="open")
    assert result["gripper_settle"]["release_confirmed"]
    assert result["gripper_settle"]["opening_after_m"] == pytest.approx(0.08)


# --- C1: holonomic navigation and stall direction -------------------------------------------


def test_navigation_translates_sideways_without_turning_first():
    toolkit, controller = setup()
    toolkit.observe()
    result = call(toolkit, "navigate_to", **args("navigate_to", target={"relative": [0, 0.3, 0]}))
    assert result["target_reached"] and result["travel_mode"] == "holonomic"
    motion = np.array(controller.actions[1:-1])
    assert np.all(motion[:, 2] == 0) and np.any(motion[:, 1] > 0)


def test_turn_first_keeps_the_previous_behaviour():
    toolkit, controller = setup()
    toolkit.observe()
    result = call(toolkit, "navigate_to", **args("navigate_to", target={"relative": [0, 0.3, 0]}, turn_first=True))
    assert result["travel_mode"] == "turn_first"
    assert np.any(np.array(controller.actions)[:, 2] != 0)


def test_stall_names_the_blocked_direction_and_repeat_is_flagged():
    toolkit, controller = setup(stuck=True)
    toolkit.observe()
    target = {"relative": [-0.4, 0, 0]}
    first = call(toolkit, "navigate_to", **args("navigate_to", target=target))
    assert first["interrupt_reason"] == "base_stalled"
    assert first["base_stall"]["blocked"] == "translation"
    assert first["base_stall"]["blocked_direction"] == "backward"
    assert "repeat_warning" not in first
    toolkit.observe()
    second = call(toolkit, "navigate_to", **args("navigate_to", expected_step=controller.step, target=target))
    assert "repeat_warning" in second
    toolkit.observe()
    third = call(toolkit, "navigate_to", **args("navigate_to", expected_step=controller.step,
                                                target={"relative": [0, 0.4, 0]}))
    assert "repeat_warning" not in third and third["base_stall"]["blocked_direction"] == "left"


def test_path_repeat_warning_only_for_an_unreached_target():
    toolkit, controller = setup(stuck=True)
    toolkit.observe()
    points = [{"position": [0.3, 0.1, 0.5]}]
    assert "repeat_warning" not in call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=points))
    toolkit.observe()
    again = call(toolkit, "move_eef_path", **args("move_eef_path", expected_step=controller.step, waypoints=points))
    assert "repeat_warning" in again


# --- C4: zoom provenance -------------------------------------------------------------------


def test_zoom_result_carries_step_and_hidden_derivation_into_the_image_archive(tmp_path):
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from rpent.context.projectors import _extract_images
    from rpent.recording.llm_images import LLMImageArchive
    from rpent.runtime.api import _content_blocks_to_pydantic

    toolkit, controller = setup()
    toolkit.observe()
    tool_result = toolkit.execute_tool("zoom", {"camera_id": "robot0_agentview_left", "box": [0, 0, 2, 2]})
    result = tool_result.result
    assert result["step"] == result["source_step"] == 0
    derivation = result["_image_derivation"]["_image_bytes"]
    assert derivation == {"kind": "zoom", "source_camera": "robot0_agentview_left", "source_step": 0,
                          "box": [0, 0, 2, 2], "scale": 4.0, "source_size": derivation["source_size"],
                          "output_size": [8, 8]}
    text, images = _extract_images(result)
    assert "_image_derivation" not in text and len(images) == 1
    assert "_image_derivation" not in tool_result.projection.text
    archive = LLMImageArchive(tmp_path)
    archive.note_tool_result(tool_result)
    text, images = _content_blocks_to_pydantic(tool_result.content_blocks)
    archive.capture_model_request([ModelRequest(parts=[UserPromptPart(content=[text, *images])])], model="m")
    [image] = [part.data for part in images]
    row = json.loads((tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()[0])
    [record] = row["images"]
    assert record["sha256"] == hashlib.sha256(image).hexdigest()
    assert record["observation_step"] == 0 and record["camera_id"] == "robot0_agentview_left"
    assert record["derivation"]["kind"] == "zoom" and record["derivation"]["output_size"] == [8, 8]


# --- Restored states (fork): archived command and public holding evidence ---------------


def test_restored_state_with_an_archived_close_keeps_the_hold_without_review():
    toolkit, controller = setup()
    controller.observation["state"][14:16] = [0.012, -0.012]  # stopped on a bowl rim
    toolkit._operation.observation["state"][14:16] = [0.012, -0.012]
    original = toolkit._resolve_gripper_command

    def archived(gripper):  # what the fork infrastructure installs
        if gripper == "keep" and not toolkit._operation._action_history:
            return "close", "fork_source_last_confirmed_command"
        return original(gripper)

    toolkit._resolve_gripper_command = archived
    toolkit.observe()
    with pytest.raises(ValueError, match="already closed"):
        toolkit.review_grasp(expected_step=0, object_between_fingers=True,
                             narrow_axis_aligned=True, evidence="Bowl held.")
    result = call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=[{"position": [0.2, 0.1, 0.55]}],
                                                    gripper="keep"))
    assert result["holding_speed_limited"] and result["holding_basis"] == "closed_on_object_width"
    assert all(action[11] >= 0.5 for action in controller.actions)


def test_width_holding_needs_a_close_command_and_a_nonempty_width():
    toolkit, controller = setup(width=0.0)
    toolkit._operation.execute_actions((close_action(),))
    toolkit.observe()
    empty = call(toolkit, "navigate_to", **args("navigate_to", expected_step=1, gripper="keep"))
    assert not empty["holding_speed_limited"] and empty["holding_basis"] is None


# --- Action history stays inside the gateway prompt limit ---------------------------------


def test_privileged_single_step_records_are_merged_for_models():
    toolkit, controller = setup()
    toolkit.observe()
    result = call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=[{"position": [0.2, 0.1, 0.6]}]))
    raw = toolkit._operation.action_history_since()
    assert len(raw) == result["executed_actions"] > 5  # one native chunk per step
    [record] = result["action_history"]["records"]
    assert record["control_source"] == "move_eef_path" and record["chunk_count"] == len(raw)
    assert record["actions"] == [a for r in raw for a in r["actions"]]
    assert record["step_ids"] == list(range(1, len(raw) + 1))
    snapshot = toolkit.action_history_snapshot()
    assert snapshot["total_chunks"] == len(raw) and snapshot["records"] == [record]


def test_policy_chunks_and_faulted_records_are_not_merged():
    from robots.robocasa.toolkit import _compact_history

    def record(index, source="move_eef_path", **extra):
        return {"chunk_index": index, "control_source": source, "instruction": None, "episode_id": "e",
                "step_ids": [index + 1], "actions": [[0.0] * 12], "actions_scope": "validated_chunk",
                "requested_actions": 1, "executed_actions": 1, "cancelled_actions": 0,
                "status": "completed", "attempted_action_index": None, "stop_reason": None, **extra}

    records = [record(0), record(1), record(2, status="faulted"), record(3), record(4, source="vla"),
               record(5, source="vla"), record(6, source="navigate_to"), record(8)]
    merged = _compact_history(records)
    assert [r.get("chunk_count", 1) for r in merged] == [2, 1, 1, 1, 1, 1, 1]
    assert merged[0]["step_ids"] == [1, 2] and merged[2]["step_ids"] == [4]


def _long_takeover_history(rows=835, seed=0):
    """One merged move_eef_path record like D10 attempt 1 of the 1.2.0 full eval (835 rows, ~136k chars)."""
    from robots.robocasa.toolkit import _compact_history

    rng = np.random.default_rng(seed)
    raw = [{"chunk_index": 10 + i, "control_source": "move_eef_path", "instruction": None, "episode_id": "e",
            "step_ids": [546 + i], "actions": [rng.uniform(-1, 1, 12).tolist()], "actions_scope": "validated_chunk",
            "requested_actions": 1, "executed_actions": 1, "cancelled_actions": 0,
            "status": "completed", "attempted_action_index": None, "stop_reason": None} for i in range(rows)]
    vla = [{"chunk_index": i, "control_source": "vla", "instruction": "Pick up the pan.", "episode_id": "e",
            "step_ids": list(range(1 + 16 * i, 17 + 16 * i)), "actions": rng.uniform(-1, 1, (16, 12)).tolist(),
            "actions_scope": "validated_chunk", "requested_actions": 16, "executed_actions": 16,
            "cancelled_actions": 0, "status": "completed", "attempted_action_index": None, "stop_reason": None}
           for i in range(10)]
    records = vla + _compact_history(raw)
    return {"schema": "robocasa-action-history/v1", "total_chunks": 10 + rows,
            "total_executed_actions": 160 + rows, "records": records}


def test_one_oversized_takeover_record_keeps_its_newest_rows_within_the_window():
    from rpent.runtime.plan_generation import (
        _HISTORY_WINDOW_CHARS,
        _recent_action_history,
    )

    history = _long_takeover_history()
    merged = history["records"][-1]
    assert len(merged["actions"]) == 835 and len(json.dumps(merged, separators=(",", ":"))) > 100_000
    recent = _recent_action_history(history)["records"]
    assert len(recent) == 1
    tail = recent[0]
    assert len(json.dumps(recent, separators=(",", ":"))) <= _HISTORY_WINDOW_CHARS
    kept = len(tail["actions"])
    assert 100 <= kept <= 250 and tail["rows_omitted_before"] == 835 - kept
    assert tail["step_ids"] == merged["step_ids"][-kept:] and tail["actions"] == merged["actions"][-kept:]
    assert tail["chunk_index"] == merged["chunk_index"] + 835 - kept
    assert tail["chunk_count"] == tail["executed_actions"] == kept
    assert f"start_chunk={merged['chunk_index']}" in tail["rows_note"]


def test_paging_walks_through_an_oversized_record_without_losing_a_row():
    from rpent.runtime.plan_generation import _HISTORY_PAGE_CHARS, _action_history_page

    history = _long_takeover_history()
    seen_steps, cursor, pages = [], 0, 0
    while cursor is not None:
        page = _action_history_page(history, cursor)["action_history"]
        assert len(json.dumps(page["records"], separators=(",", ":"))) <= _HISTORY_PAGE_CHARS
        for record in page["records"]:
            seen_steps.extend(record["step_ids"])
        cursor = page["next_chunk"]
        pages += 1
        assert pages < 100
    expected = [s for record in history["records"] for s in record["step_ids"]]
    assert seen_steps == expected  # every confirmed step exactly once, in order


def test_planner_history_window_and_pages_are_bounded_by_characters():
    import json

    from rpent.runtime.plan_generation import (
        _HISTORY_PAGE_CHARS,
        _HISTORY_WINDOW_CHARS,
        _action_history_page,
        _recent_action_history,
    )

    toolkit, controller = setup(horizon=2000)
    for index in range(6):  # a takeover's worth of separate privileged calls
        toolkit.observe()
        call(toolkit, "move_eef_path", **args("move_eef_path", expected_step=controller.step,
                                               waypoints=[{"position": [0.2, 0.1 + 0.05 * (index % 2), 0.6]}]))
    history = toolkit.action_history_snapshot()
    recent = _recent_action_history(history)["records"]
    assert len(json.dumps(recent, separators=(",", ":"))) <= _HISTORY_WINDOW_CHARS
    page = _action_history_page(history, 0)["action_history"]["records"]
    assert len(json.dumps(page, separators=(",", ":"))) <= _HISTORY_PAGE_CHARS or len(page) == 1


# --- move_eef_path under load (1.0.11 E2/P1: deviation stops while still moving) ----------


class LoadedController(PublicController):
    """A held load sags: each step loses `deadband_m` of the commanded EEF delta."""

    def __init__(self, deadband_m=0.0045, rotation_rate=1.0):
        super().__init__()
        self.deadband_m, self.rotation_rate = deadband_m, rotation_rate

    def execute_action(self, action, *, _action_index=0):
        before = self.observation["state"].copy()
        commanded = np.asarray(action[5:8], dtype=float) * 0.05
        transition = super().execute_action(action, _action_index=_action_index)
        state = self.observation["state"]
        size = float(np.linalg.norm(commanded))
        moved = commanded * (max(0.0, size - self.deadband_m) / size) if size > 1e-12 else commanded
        state[7:10] = before[7:10] + moved
        state[10:14] = _rotate_base_quaternion(before[10:14], np.asarray(action[8:11]) * 0.5 * self.rotation_rate)
        return transition


def loaded_setup(**kwargs):
    from robots.robocasa.toolkit import RoboCasaToolkit

    controller = LoadedController(**kwargs)
    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Lift the pan.", episode_horizon=500)
    seed_closed(toolkit, controller)
    return toolkit, controller


def test_loaded_lift_raises_the_command_instead_of_stopping():
    toolkit, controller = loaded_setup()
    start = controller.observation["state"][7:10].copy()
    target = (start + [0, 0, 0.06]).tolist()
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": target}], gripper="keep", max_steps=150)
    assert result["target_reached"] and result["interrupt_reason"] is None
    assert result["load_compensation"]["level"] >= 1 and result["holding_speed_limited"]
    # The base cap alone gives 0.5 mm/step; the 1.0.10 rule stopped after 8 steps.
    assert max(np.linalg.norm(np.array(a[5:8]) * 0.05) for a in controller.actions[1:]) > 0.005 + 1e-9


def test_blocked_load_still_stops_after_compensation_is_exhausted():
    toolkit, controller = loaded_setup(deadband_m=1.0)  # nothing moves
    start = controller.observation["state"][7:10].copy()
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": (start + [0, 0, 0.06]).tolist()}], gripper="keep", max_steps=150)
    assert result["interrupt_reason"] == "deviation"
    assert result["load_compensation"]["level"] == 2
    assert result["executed_actions"] <= 3 * 16 + 3


def test_rotation_progress_counts_even_when_position_tracking_is_slow():
    toolkit, controller = loaded_setup()
    start = controller.observation["state"][7:10].copy()
    # A pour: rotate while holding nearly the same point. Position closes in at
    # 0.5 mm/step under load; the 1.0.10 rule stopped on that alone (E2).
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": (start + [0.02, 0, 0]).tolist(), "rotation_degrees": [0, 0, 60]}],
                  gripper="keep", max_steps=150)
    assert result["target_reached"], result["waypoint_results"]


def test_free_motion_stall_rule_is_unchanged():
    toolkit, controller = setup(stuck=True)
    toolkit.observe()
    result = call(toolkit, "move_eef_path", **args("move_eef_path", waypoints=[{"position": [0.3, 0.1, 0.5]}]))
    assert result["interrupt_reason"] == "deviation" and result["executed_actions"] <= 8 + 3


def test_close_on_a_compliant_handle_settles_on_the_contact_rate():
    widths = [0.0799, 0.078, 0.0736, 0.0676, 0.0608, 0.0534, 0.0458, 0.038, 0.0303, 0.0245,
              0.0243, 0.0239, 0.0232, 0.0225, 0.0218, 0.0212, 0.0200]

    class Creeping(PublicController):
        def __init__(self):
            super().__init__()
            self.closes = 0
            self.observation["state"][14:16] = [0.04, -0.04]

        def execute_action(self, action, *, _action_index=0):
            transition = super().execute_action(action, _action_index=_action_index)
            if action[11] >= 0.5:
                self.closes += 1
            width = widths[min(self.closes, len(widths) - 1)]
            self.observation["state"][14:16] = [width / 2, -width / 2]
            return transition

    from robots.robocasa.toolkit import RoboCasaToolkit

    controller = Creeping()
    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Pick up the pan.", episode_horizon=500)
    toolkit.observe()
    call(toolkit, "review_grasp", expected_step=0, object_between_fingers=True, narrow_axis_aligned=True,
         evidence="Handle between the fingers.")
    position = controller.observation["state"][7:10].tolist()
    settle = call(toolkit, "move_eef_path", expected_step=0, waypoints=[{"position": position}],
                  gripper="close")["gripper_settle"]
    assert settle["stable"] and settle["settle_rule"] == "contact_rate" and settle["closure_confirmed"]
    assert settle["steps"] <= 13


# --- review of 1.0.11 before the full evaluation --------------------------------------------


class SlippingController(LoadedController):
    """The held object slips out after `slip_after` steps; the fingers close on nothing."""

    def __init__(self, slip_after, **kwargs):
        super().__init__(**kwargs)
        self.slip_after, self.moves = slip_after, 0

    def execute_action(self, action, *, _action_index=0):
        transition = super().execute_action(action, _action_index=_action_index)
        self.moves += 1
        if self.moves > self.slip_after:
            self.observation["state"][14:16] = [0.0005, -0.0005]
        return transition


def test_a_slip_during_a_lift_stops_with_grasp_lost():
    from robots.robocasa.toolkit import RoboCasaToolkit

    controller = SlippingController(slip_after=4, deadband_m=0.0)
    toolkit = RoboCasaToolkit(controller, initial_observation=controller.observation,
                              instruction="Lift the pan.", episode_horizon=500)
    seed_closed(toolkit, controller)
    # Holding known only from a close on an object width: the grasp watch is
    # not armed (a sideways approach, or a short descent before the close).
    toolkit._pick_progress = None
    start = controller.observation["state"][7:10].copy()
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": (start + [0, 0, 0.12]).tolist()}], gripper="keep", max_steps=60)
    assert result["holding_basis"] == "closed_on_object_width"
    assert result["interrupt_reason"] == "grasp_lost" and result["closed_empty"]
    assert not result["target_reached"] and result["executed_actions"] <= 8


def test_a_sagging_load_settles_short_of_the_waypoint_instead_of_aborting():
    toolkit, controller = loaded_setup(deadband_m=0.006)  # 6 mm sag > 5 mm tolerance
    start = controller.observation["state"][7:10].copy()
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": (start + [0, 0, 0.04]).tolist()},
                             {"position": (start + [0.03, 0, 0.04]).tolist()}],
                  gripper="keep", max_steps=150)
    rows = result["waypoint_results"]
    assert result["target_reached"] and len(rows) == 2
    assert rows[0]["settled_under_load"] and 0.005 < rows[0]["position_error_m"] <= 0.01


def test_a_descent_starts_again_from_the_holding_step():
    toolkit, controller = loaded_setup()
    start = controller.observation["state"][7:10].copy()
    before = len(controller.actions)
    step0 = controller.step
    result = call(toolkit, "move_eef_path", expected_step=controller.step,
                  waypoints=[{"position": (start + [0, 0, 0.06]).tolist()},
                             {"position": (start + [0, 0, 0.0]).tolist()}],
                  gripper="keep", max_steps=150)
    rows = result["waypoint_results"]
    assert rows[0]["load_level"] >= 1
    first_descent = controller.actions[before + rows[1]["start_step"] - step0]
    assert np.linalg.norm(np.asarray(first_descent[5:8]) * 0.05) <= 0.005 + 1e-9


def test_diagonal_travel_at_the_friction_floor_keeps_the_floor_on_one_axis():
    from robots.robocasa.privileged_tools import (
        BASE_COMMAND_MIN,
        _holonomic_translation,
    )

    # The acceleration ramp starts every travel, held or free, at the floor.
    command = _holonomic_translation(np.array([0.5, 0.5]), 0.0, BASE_COMMAND_MIN)
    assert max(abs(command)) >= BASE_COMMAND_MIN - 1e-9
    assert np.linalg.norm(command) <= BASE_COMMAND_MIN + 1e-9
    free = _holonomic_translation(np.array([0.5, 0.5]), 0.0, 0.6)
    assert np.allclose(free, [0.6 / np.sqrt(2)] * 2)


def test_history_pages_by_native_chunk_index():
    from rpent.runtime.plan_generation import _action_history_page

    records = [{"chunk_index": 0, "chunk_count": 3, "actions": [[0]] * 3},
               {"chunk_index": 3, "actions": [[0]]}, {"chunk_index": 4, "actions": [[0]]}]
    history = {"total_chunks": 5, "records": records}
    page = _action_history_page(history, 3)["action_history"]
    assert page["records"][0]["chunk_index"] == 3 and page["next_chunk"] is None
    assert _action_history_page(history, 1)["action_history"]["records"][0]["chunk_index"] == 0
    with pytest.raises(ValueError, match="exceeds"):
        _action_history_page(history, 6)


def test_a_confirmed_grasp_clears_the_closing_review_flag():
    from types import SimpleNamespace

    toolkit, controller = setup()
    toolkit._preclose_active = True
    toolkit._preclose_closing = True
    toolkit._pick_progress = SimpleNamespace(confirmed=True, retention_monitor_active=False)
    toolkit._guard_grasp_close(controller.observation, close_action())
    assert toolkit._preclose_closing is False
