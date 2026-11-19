"""Grasp-loss interrupts must report the actual stopped state to the planner."""

import io

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.binding import make_robocasa_binding
from robots.robocasa.outcomes import RoboCasaSkillOutcomeAdapter
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.episode.contracts import StepResult
from rpent.episode.controller import EpisodeController
from rpent.planning.feedback import action_feedback_from_result

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def observation(z=0.5, width=0.08, color=0):
    state = np.zeros(16, dtype=np.float32)
    state[9] = z
    state[10:14] = [0, 0, 0, 1]
    state[14:16] = [width / 2, -width / 2]
    return {
        "state": state,
        **{camera: np.full((256, 256, 3), color, dtype=np.uint8) for camera in CAMERAS},
        "robot0_agentview_center": np.full((256, 256, 3), color, dtype=np.uint8),
    }


class Simulator:
    def __init__(self, observations, *, episode_id="episode-1", success_at=None):
        self.observations = list(observations)
        self.episode_id = episode_id
        self.step_id = 0
        self.success_at = success_at

    def step(self, action):
        self.step_id += 1
        return StepResult(
            self.episode_id, self.step_id, self.observations.pop(0), 0.0,
            self.step_id == self.success_at, False, False,
        )


class Policy:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.requests = 0

    def predict(self, observation, instruction, *, max_actions=None):
        self.requests += 1
        return self.chunks.pop(0)


def actions(count, gripper=1.0):
    result = np.zeros((count, 12), dtype=np.float32)
    result[:, 11] = gripper
    return result


def setup(widths, *, gripper=1.0, success_at=None):
    simulator = Simulator([
        observation(0.41, 0.02, 1), observation(0.48, 0.02, 2),
        *[observation(0.48, width, index + 3) for index, width in enumerate(widths)],
    ], success_at=success_at)
    policy = Policy([actions(2), actions(len(widths), gripper)])
    controller = EpisodeController(simulator, policy, make_robocasa_binding())
    toolkit = RoboCasaToolkit(controller, initial_observation=observation(), instruction="Move the cup.")
    toolkit.execute_vla()
    assert toolkit._pick_progress.confirmed is True
    return toolkit, simulator, policy


@pytest.mark.parametrize("width,reason", [(0.0, "empty_closure"), (0.035, "width_jump")])
@pytest.mark.parametrize("tool", ["execute_vla", "continue_vla"])
def test_policy_chunk_stops_at_first_grasp_loss(tool, width, reason):
    toolkit, simulator, policy = setup([width, 0.02, 0.02, 0.02])
    arguments = {"instruction": "Transport the cup.", "max_chunks": 3} if tool == "continue_vla" else {}
    returned = toolkit.execute_tool(tool, arguments)
    result = returned.result
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1
    assert result["cancelled_actions"] == 3
    assert simulator.step_id == 3
    assert policy.requests == 2
    event = result["grasp_events"][0]
    assert event["type"] == "grasp_lost"
    assert event["reason"] == reason
    assert event["step"] == event["image_step"] == 3
    assert event["episode_id"] == "episode-1"
    assert event["release_commanded"] is False
    assert event["evidence_scope"] == "proprioceptive_grasp_loss"
    assert "grasp_lost" in returned.projection.text
    assert len(returned.projection.images) >= 3
    # 1.2.2: tell the LLM to find the object before acting on an assumed location.
    assert "find it in" in result["grasp_lost_next"] and "do not carry it on" in result["grasp_lost_next"]
    assert "grasp_lost_next" in returned.projection.text
    for key in ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes"):
        pixels = np.asarray(Image.open(io.BytesIO(result[key])))
        assert pixels.shape == (256, 256, 3)
        assert np.all(pixels == 3)
    history = result["action_history"]["records"][-1]
    assert history["step_ids"] == [3]
    assert history["executed_actions"] == 1
    assert len(history["actions"]) == 4


def test_small_width_noise_does_not_interrupt():
    toolkit, simulator, policy = setup([0.021, 0.019, 0.022, 0.02])
    result = toolkit.execute_vla()
    assert result["executed_actions"] == 4
    assert not result.get("grasp_events")
    assert simulator.step_id == 6


def test_commanded_release_does_not_interrupt_and_disarms_monitor():
    toolkit, simulator, policy = setup([0.04, 0.08], gripper=0.0)
    result = toolkit.execute_vla()
    assert not result.get("grasp_events")
    assert result["executed_actions"] == 2
    assert not toolkit._pick_progress.confirmed
    assert not toolkit._pick_progress.retention_monitor_active


@pytest.mark.parametrize("closing_step_m,closing_steps", [(0.0101, 6), (0.0118, 5)])
def test_release_then_fast_second_grasp_completes_and_rearms_loss_monitor(closing_step_m, closing_steps):
    # Open slowly enough that the old monitor survives the entire release.
    release_widths = [0.028, 0.036, 0.044, 0.052, 0.06, 0.068, 0.076, 0.08]
    toolkit, simulator, policy = setup(release_widths, gripper=0.0)
    released = toolkit.execute_vla()
    assert released["executed_actions"] == len(release_widths)
    assert not released.get("grasp_events")

    widths = [0.08 - closing_step_m * step for step in range(1, closing_steps + 1)]
    simulator.observations = [
        observation(0.40, 0.08),
        *[observation(0.40, width) for width in widths],
        observation(0.47, widths[-1]),
    ]
    second_grasp = actions(closing_steps + 2)
    second_grasp[0, 11] = 0.0
    policy.chunks = [second_grasp]
    acquired = toolkit.execute_vla()
    assert acquired["executed_actions"] == closing_steps + 2
    assert not acquired.get("grasp_events")
    assert toolkit._pick_progress.confirmed
    assert toolkit._pick_progress.retention_monitor_active
    second_grasp_step = simulator.step_id

    simulator.observations = [observation(0.47, 0.0)]
    policy.chunks = [actions(1)]
    lost = toolkit.execute_vla()
    assert lost["stop_reason"] == "grasp_lost"
    assert lost["grasp_events"][0]["reason"] == "empty_closure"
    assert lost["grasp_events"][0]["confirmed_grasp_step"] == second_grasp_step


def test_loss_inside_acquisition_does_not_report_stale_success():
    simulator = Simulator([observation(0.41, 0.08), observation(0.41, 0.02),
                           observation(0.48, 0.02), observation(0.48, 0.0), observation()])
    approach = actions(2)
    approach[0, 11] = -1
    policy = Policy([approach, actions(4)])
    toolkit = RoboCasaToolkit(EpisodeController(simulator, policy, make_robocasa_binding()),
                             initial_observation=observation(), instruction="Move the cup.")
    paused = toolkit.smolvla_pick("Acquire the cup.", max_chunks=1)
    assert paused["stop_reason"] == "preclose_review_required"
    toolkit.observe()
    toolkit.review_grasp(expected_step=1, object_between_fingers=True, narrow_axis_aligned=True,
                         evidence="Wrist image shows the intended narrow grasp axis.")
    result = toolkit.smolvla_pick("Acquire the cup.", max_chunks=2)
    assert result["success"] is False
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 3
    assert result["cancelled_actions"] == 2
    assert result["resumed_actions"] == 1


def test_direct_servo_stops_on_loss_and_cannot_reuse_old_confirmation():
    toolkit, simulator, policy = setup([0.0, 0.0, 0.0])
    result = toolkit.servo_eef(target_position=[0.05, 0, 0.48], target_quaternion=[0, 0, 0, 1],
                               gripper="close", max_steps=3)
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1
    assert result["target_reached"] is False
    assert toolkit._pick_progress.confirmed is False


def test_no_drop_alert_before_a_confirmed_grasp():
    simulator = Simulator([observation(0.41, 0.0), observation(0.48, 0.0)])
    policy = Policy([actions(2)])
    toolkit = RoboCasaToolkit(EpisodeController(simulator, policy, make_robocasa_binding()),
                             initial_observation=observation(), instruction="Move the cup.")
    toolkit.execute_vla()
    result = toolkit.smolvla_pick("Acquire the cup.", max_chunks=1)
    assert result["failure_reason"] == "closed_empty"
    assert not result.get("grasp_events")


def test_grasp_alert_is_advisory_and_official_success_takes_precedence():
    toolkit, simulator, policy = setup([0.0, 0.02], success_at=3)
    result = toolkit.execute_vla()
    feedback = action_feedback_from_result(execution_id="drop", tool_name="execute_vla", physical=True,
                                          result=result, outcome_adapter=RoboCasaSkillOutcomeAdapter())
    assert feedback.outcome.status == "success"
    assert feedback.environment_success is True
    assert simulator.step_id == 3


def test_grasp_alert_requires_verification_without_fabricating_object_drop(tmp_path):
    from rpent.harness import Harness

    toolkit, simulator, policy = setup([0.0, 0.02])
    harness = Harness(mode="plan-gated", trace_path=tmp_path / "trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan("Move the cup.", [{"id": "transport", "intent": "Transport the cup.",
        "max_attempts": 8, "allowed_tools": ["execute_vla", "continue_vla"]}])["ok"]
    returned = toolkit.execute_tool("execute_vla", {})
    feedback = harness.state.last_feedback
    assert feedback.outcome.status == "uncertain"
    assert any("grasp_lost" in text for text in feedback.outcome.evidence)
    assert feedback.outcome.failure_type is None
    assert "grasp_lost" in returned.projection.text
    blocked = toolkit.execute_tool("execute_vla", {}).result
    assert "error" in blocked
    assert simulator.step_id == 3


def test_prompt_explains_grasp_alert_and_observation_requirement():
    from robots.robocasa.prompts.system import system_prompt

    prompt = " ".join(system_prompt().values())
    assert "grasp_lost" in prompt
    assert "release_commanded" in prompt
    assert "current images" in prompt


@pytest.mark.parametrize("width,reason", [(0.002, "empty_closure"), (0.03, "width_jump")])
def test_threshold_boundaries_interrupt(width, reason):
    toolkit, simulator, policy = setup([width, 0.02])
    result = toolkit.execute_vla()
    assert result["grasp_events"][0]["reason"] == reason
    assert simulator.step_id == 3


def test_adjust_eef_and_resume_vla_share_the_same_loss_monitor():
    toolkit, simulator, policy = setup([0.0, 0.0])
    toolkit.observe()
    result = toolkit.adjust_eef(expected_step=2, delta_position=[0.02, 0, 0], gripper="keep", max_steps=2)
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1
    assert result["interruption_scope"] == "control_step"

    toolkit, simulator, policy = setup([0.02, 0.0, 0.02])
    toolkit.observe()
    toolkit.adjust_eef(expected_step=2, delta_position=[0.02, 0, 0], gripper="keep", max_steps=1)
    toolkit.observe()
    # A fresh policy prediction has two actions left to send; the first loses grasp.
    policy.chunks[0] = actions(2)
    result = toolkit.resume_vla(max_actions=2)
    assert result["stop_reason"] == "grasp_lost"
    assert result["executed_actions"] == 1
    assert result["cancelled_actions"] == 1
    assert result["grasp_events"][0]["step"] == 4


def test_alert_is_emitted_once_and_reacquisition_can_start_a_new_cycle():
    toolkit, simulator, policy = setup([0.0, 0.0])
    first = toolkit.execute_vla()
    assert len(first["grasp_events"]) == 1
    assert not toolkit.observe().get("grasp_events")
    old_cycle = first["grasp_events"][0]["grasp_cycle"]
    simulator.observations = [observation(0.4, 0.02), observation(0.47, 0.02), observation(0.47, 0.0)]
    policy.chunks = [actions(2), actions(1)]
    toolkit.execute_vla()
    assert toolkit._pick_progress.confirmed
    second = toolkit.execute_vla()
    assert len(second["grasp_events"]) == 1
    assert second["grasp_events"][0]["grasp_cycle"] > old_cycle
    assert second["grasp_events"][0]["confirmed_grasp_step"] == 5


def test_new_episode_cannot_inherit_a_grasp_alert():
    toolkit, simulator, policy = setup([0.0, 0.02])
    simulator.episode_id = "episode-2"
    simulator.step_id = 0
    result = toolkit.execute_vla()
    assert not result.get("grasp_events")
    assert result["environment"]["episode_id"] == "episode-2"
    assert toolkit._pick_progress.confirmed is False


def test_monitor_remains_active_through_gradual_width_change():
    toolkit, simulator, policy = setup([0.025, 0.033, 0.025, 0.018, 0.011, 0.005, 0.001, 0.02])
    result = toolkit.execute_vla()
    assert result["grasp_events"][0]["reason"] == "empty_closure"
    assert result["executed_actions"] == 7
    assert result["cancelled_actions"] == 1


@pytest.mark.parametrize("compact", [False, True])
def test_alert_and_all_three_images_survive_context_projection(compact):
    from rpent.context import resolve_context_policy

    toolkit, simulator, policy = setup([0.0, 0.02])
    toolkit.set_context_policy(resolve_context_policy("episode-compact-v1" if compact else "raw-tool-result-v1"))
    result = toolkit.execute_tool("execute_vla", {})
    assert "grasp_lost" in result.projection.text
    assert "release_commanded" in result.projection.text
    assert len(result.projection.images) == 3


def test_transition_guard_fault_keeps_confirmed_history_and_faults_controller():
    from rpent.episode.controller import EpisodeExecutionError, EpisodeFaultedError
    from rpent.tools.vla import VLAOperation

    simulator = Simulator([observation(), observation()])
    policy = Policy([actions(2)])
    controller = EpisodeController(simulator, policy, make_robocasa_binding())
    def failing_guard(action, transition):
        raise RuntimeError("invalid sensor feedback")
    operation = VLAOperation(controller, initial_observation=observation(), instruction="Move the cup.",
                             capture_action_history=True, transition_guard=failing_guard)
    with pytest.raises(EpisodeExecutionError, match="guard failed"):
        operation.execute()
    assert operation.metrics["step_count"] == 1
    record = operation.action_history_since()[0]
    assert record["step_ids"] == [1]
    assert record["executed_actions"] == record["cancelled_actions"] == 1
    assert record["status"] == "faulted"
    with pytest.raises(EpisodeFaultedError):
        operation.execute()


def test_alert_images_are_archived_with_exact_bytes_cameras_and_stopped_step(tmp_path):
    import json

    from pydantic_ai import BinaryContent
    from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

    from rpent.recording.llm_images import LLMImageArchive

    toolkit, simulator, policy = setup([0.0, 0.02])
    returned = toolkit.execute_tool("execute_vla", {})
    archive = LLMImageArchive(tmp_path)
    archive.note_tool_result(returned)
    raw = [returned.result[key] for key in ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes")]
    messages = [ModelRequest(parts=[
        ToolReturnPart(tool_name="execute_vla", tool_call_id=returned.call_id, content=returned.projection.text),
        UserPromptPart(content=[BinaryContent(data=value, media_type="image/png") for value in raw]),
    ])]
    archive.capture_model_request(messages, model="test-model")
    record = json.loads((tmp_path / "llm_inputs/requests.jsonl").read_text().splitlines()[0])
    assert [image["camera_id"] for image in record["images"]] == list(CAMERAS)
    assert all(image["observation_step"] == 3 for image in record["images"])
    for image, value in zip(record["images"], raw, strict=True):
        assert (tmp_path / image["relative_path"]).read_bytes() == value
