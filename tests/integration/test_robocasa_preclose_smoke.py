"""Opt-in real RoboCasa control-boundary smoke with a deterministic probe policy."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_PRECLOSE_SMOKE") != "1",
    reason="requires pinned RoboCasa, assets and EGL",
)


def test_real_robocasa_pauses_before_close_and_keeps_correction_open(tmp_path):
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import RoboCasaEnvFacade, make_upstream_environment
    from robots.robocasa.session import RoboCasaSession
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.episode.controller import EpisodeController
    from rpent.harness import Harness

    task = "ArrangeDrinkware"
    registry = json.loads((Path(__file__).resolve().parents[2] / "robots/robocasa/task_groups.json").read_text())
    horizon = get_task_horizon(task)
    facade = RoboCasaEnvFacade(
        environment_factory=make_upstream_environment,
        dependency_identity={"robocasa_version": robocasa.__version__},
    )
    facade.load({
        "task": task, "split": "pretrain", "seed": 7, "robot": "PandaOmron",
        "camera_size": 256,
        "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        "object_registries": ["lightwheel", "objaverse"], "rendering_mode": "headless",
        "upstream_horizon": horizon, "safety_ceiling": horizon,
        "task_registry_commit": ROBOCASA_COMMIT, "task_registry_digest": registry["registry_digest"],
    })
    simulator = RoboCasaSession(facade, task=task, seed=7)
    initial = simulator.reset(task, 7)

    class ProbePolicy:
        def __init__(self):
            self.instructions = []

        def predict(self, observation, instruction, *, max_actions=None):
            del observation
            self.instructions.append(instruction)
            values = np.zeros((3, 12), dtype=np.float32)
            values[:, 11] = [-1, 1, 1] if len(self.instructions) == 1 else [1, 1, 1]
            return values[:max_actions]

    policy = ProbePolicy()
    toolkit = RoboCasaToolkit(
        EpisodeController(simulator, policy, make_robocasa_binding()),
        initial_observation=initial, instruction=simulator.instruction,
        forward_step_guidance=True, actions_per_policy_request=5, episode_horizon=horizon,
    )
    artifacts = Path(os.environ.get("RPENT_PRECLOSE_SMOKE_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    harness = Harness(mode="plan-gated", trace_path=artifacts / "harness_trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan(simulator.instruction, [{
        "id": "acquire", "intent": "Acquire the observed target.",
        "allowed_tools": ["smolvla_pick", "adjust_eef", "resume_vla"],
        "max_attempts": 3,
    }])["ok"]

    def call(name, **kwargs):
        value = toolkit.execute_tool(name, kwargs).result
        assert "error" not in value, value
        return value

    try:
        paused = call("smolvla_pick")
        assert paused["stop_reason"] == "preclose_review_required"
        assert paused["executed_actions"] == 1 and paused["paused_actions"] == 2
        assert paused["cancelled_actions"] == 0
        assert toolkit.evaluation_metrics["step_count"] == 1
        assert harness.state.snapshot()["active_step_attempts"] == 1
        observed = call("observe")
        images = []
        for camera, key in zip(
            ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
            ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes"),
        ):
            content = observed[key]
            (artifacts / f"{camera}.png").write_bytes(content)
            images.append({"camera": camera, "step": observed["step"],
                           "sha256": hashlib.sha256(content).hexdigest()})
        rejected = call(
            "review_grasp", expected_step=observed["step"], object_between_fingers=False,
            narrow_axis_aligned=False,
            evidence="This control-boundary smoke has no LLM visual assessment; alignment remains unconfirmed.",
        )
        assert rejected["approved"] is False
        call("verify_step", verdict="progress",
             evidence=["The close command was paused; wrist alignment remains unconfirmed."],
             summary="Correct alignment while keeping the gripper open.")
        corrected = call(
            "adjust_eef", expected_step=observed["step"], delta_position=[0, 0, 0.002],
            rotation_degrees=[0, 0, 1], gripper="open", max_steps=2,
        )
        assert corrected["policy_instruction"] == simulator.instruction
        assert all(row[11] < 0.5 for record in corrected["action_history"]["records"] for row in record["actions"])
        call("observe")
        call("verify_step", verdict="progress",
             evidence=["The bounded open-gripper correction executed; alignment is still unconfirmed."],
             summary="Review the fresh policy handoff.")
        assert harness.state.snapshot()["active_step_attempts"] == 2
        resumed = call("resume_vla")
        assert resumed["stop_reason"] == "preclose_review_required"
        assert resumed["executed_actions"] == 0
        assert len(policy.instructions) == 2 and set(policy.instructions) == {simulator.instruction}
        assert toolkit._consecutive_closed_empty == 0
        assert harness.state.snapshot()["active_step_attempts"] == 2
        assert harness.state.awaiting_verification
        call("observe")
        fallback = call(
            "review_grasp", expected_step=toolkit.evaluation_metrics["step_count"],
            object_between_fingers=False, narrow_axis_aligned=False,
            evidence="Second unconfirmed wrist review in the same closure cycle; probe policy may attempt closure.",
        )
        assert fallback["approved"] is False and fallback["policy_close_allowed"] is True
        assert fallback["grasp_review"]["consecutive_rejections"] == 2
        call("verify_step", verdict="progress",
             evidence=["The handoff executed zero actions and paused for wrist review."],
             summary="Acquire after fresh alignment review.")
        assert not harness.state.needs_replan
        assert not harness.state.completed and not harness.state.environment_success
        attempted = call("smolvla_pick", max_chunks=1)
        assert attempted["executed_actions"] == 3
        assert harness.state.snapshot()["active_step_attempts"] == 3
        assert toolkit._grasp_review_status()["consecutive_rejections"] == 0
        result = {
            "machine": "sds-server", "task": task, "seed": 7,
            "original_instruction": simulator.instruction, "policy_requests": len(policy.instructions),
            "paused_step": observed["step"], "paused_before_close": paused["paused_actions"],
            "correction_actions": corrected["executed_actions"], "resumed_actions_without_approval": 0,
            "rejected_reviews_before_fallback": 2,
            "fallback_policy_actions": attempted["executed_actions"],
            "fallback_visual_approval": fallback["approved"],
            "charged_attempts": harness.state.snapshot()["active_step_attempts"],
            "max_attempts": harness.state.active_step.max_attempts,
            "remaining_attempts": 0, "replan_required": harness.state.needs_replan,
            "images": images,
            "purpose": "real simulator and formal controller smoke with a deterministic probe policy; no learned-policy or LLM task-success claim",
        }
        (artifacts / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        simulator.close()


def test_real_robocasa_approved_review_resumes_without_prediction(tmp_path):
    """Synthetic approval tests transport only, without asserting visual alignment."""
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import RoboCasaEnvFacade, make_upstream_environment
    from robots.robocasa.session import RoboCasaSession
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.episode.controller import EpisodeController
    from rpent.harness import Harness

    task = "ArrangeDrinkware"
    registry = json.loads((Path(__file__).resolve().parents[2] / "robots/robocasa/task_groups.json").read_text())
    horizon = get_task_horizon(task)
    facade = RoboCasaEnvFacade(environment_factory=make_upstream_environment,
                              dependency_identity={"robocasa_version": robocasa.__version__})
    facade.load({
        "task": task, "split": "pretrain", "seed": 7, "robot": "PandaOmron",
        "camera_size": 256,
        "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        "object_registries": ["lightwheel", "objaverse"], "rendering_mode": "headless",
        "upstream_horizon": horizon, "safety_ceiling": horizon,
        "task_registry_commit": ROBOCASA_COMMIT, "task_registry_digest": registry["registry_digest"],
    })
    simulator = RoboCasaSession(facade, task=task, seed=7)
    initial = simulator.reset(task, 7)

    class ProbePolicy:
        def __init__(self):
            self.calls = 0
            self.original = np.zeros((3, 12), dtype=np.float32)
            self.original[:, 11] = [-1, 1, 1]

        def predict(self, observation, instruction, *, max_actions=None):
            del observation, instruction
            self.calls += 1
            assert self.calls == 1, "approval must resume instead of predicting"
            return self.original[:max_actions].copy()

    policy = ProbePolicy()
    toolkit = RoboCasaToolkit(
        EpisodeController(simulator, policy, make_robocasa_binding()),
        initial_observation=initial, instruction=simulator.instruction,
        forward_step_guidance=True, actions_per_policy_request=5, episode_horizon=horizon,
    )
    artifacts = Path(os.environ.get("RPENT_PRECLOSE_SMOKE_ARTIFACTS", tmp_path)) / "approved-continuation"
    artifacts.mkdir(parents=True, exist_ok=True)
    harness = Harness(mode="plan-gated", trace_path=artifacts / "harness_trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan(simulator.instruction, [{
        "id": "acquire", "intent": "Synthetic action-continuation transport probe.",
        "allowed_tools": ["smolvla_pick", "continue_vla"], "max_attempts": 3,
    }])["ok"]

    def call(name, **kwargs):
        result = toolkit.execute_tool(name, kwargs).result
        assert "error" not in result, result
        return result

    try:
        paused = call("smolvla_pick", max_chunks=1)
        assert paused["executed_actions"] == 1 and paused["paused_actions"] == 2
        assert paused["cancelled_actions"] == 0
        observed = call("observe")
        approval = call("review_grasp", expected_step=observed["step"],
                        object_between_fingers=True, narrow_axis_aligned=True,
                        evidence="Synthetic approval injected solely for a control-boundary test; no actual visual alignment or task-success claim.")
        assert approval["approved"] and toolkit.evaluation_metrics["step_count"] == 1
        blocked = toolkit.execute_tool("continue_vla", {"max_chunks": 1}).result
        assert "error" in blocked and toolkit.evaluation_metrics["step_count"] == 1
        call("verify_step", verdict="progress",
             evidence=["Synthetic approval recorded; the physical suffix remains unexecuted."],
             summary="Resume the retained commands under the original verification gate.")
        resumed = call("continue_vla", max_chunks=1)
        assert resumed["executed_actions"] == resumed["resumed_actions"] == 2
        assert policy.calls == 1 and toolkit.evaluation_metrics["step_count"] == 3
        records = toolkit._operation.action_history_since()
        executed = [action for record in records for action in record["actions"][:record["executed_actions"]]]
        np.testing.assert_array_equal(executed, policy.original)
        (artifacts / "result.json").write_text(json.dumps({
            "machine": "sds-server", "task": task, "seed": 7,
            "paused_actions": 2, "resumed_actions": 2, "policy_predictions": policy.calls,
            "all_original_actions_preserved": True, "verification_gate_preserved": True,
            "LLM_calls": 0, "learned_policy_calls": 0,
            "synthetic_approval": True, "task_success_claimed": False,
        }, indent=2) + "\n")
    finally:
        simulator.close()
