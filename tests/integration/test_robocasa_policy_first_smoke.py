"""Opt-in native whole-task transport check; no LLM or learned policy."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_POLICY_FIRST_SMOKE") != "1",
    reason="requires pinned RoboCasa, assets and EGL",
)


def test_native_unknown_composite_keeps_the_policy_instruction_and_all_actions(tmp_path):
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import RoboCasaEnvFacade, make_upstream_environment
    from robots.robocasa.session import RoboCasaSession
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.episode.controller import EpisodeController
    from rpent.harness import Harness

    task = "CuttingToolSelection"
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
    output = Path(os.environ.get("RPENT_POLICY_FIRST_SMOKE_ARTIFACTS", tmp_path))
    output.mkdir(parents=True, exist_ok=True)

    class Probe:
        def __init__(self):
            self.instructions = []
            self.chunks = []

        def predict(self, observation, instruction, *, max_actions=None):
            assert observation is not None
            self.instructions.append(instruction)
            values = np.zeros((16, 12), dtype=np.float32)
            values[:, 11] = np.tile([-1, -1, 1, 1], 4)
            self.chunks.append(values[:max_actions].copy())
            return values[:max_actions]

    try:
        initial = simulator.reset(task, 7)
        policy = Probe()
        toolkit = RoboCasaToolkit(
            EpisodeController(simulator, policy, make_robocasa_binding()),
            initial_observation=initial, instruction=simulator.instruction,
            forward_step_guidance=True, actions_per_policy_request=16, episode_horizon=horizon,
        )
        harness = Harness(mode="plan-gated", trace_path=output / "trace.jsonl",
                          policy_family="xiaomi-mibot", task_name=task, max_replans=None)
        toolkit.attach_harness(harness)
        plan = {
            "id": "whole", "operation": "whole_task", "skill_row": "none", "tier": "none",
            "control_mode": "monitor", "intent": f'Execute "{simulator.instruction}"',
            "allowed_tools": ["continue_vla"], "max_attempts": 64,
        }
        assert harness.state.register_plan(simulator.instruction, [plan])["ok"]
        toolkit._preclose_active = True  # Exercise the scoped no-interruption rule.
        actions = []
        for index in range(2):
            assert harness.state.whole_task_policy_first_active
            observed = toolkit.execute_tool("observe", {}).result
            assert observed["grasp_review"]["enabled"] is False
            review = toolkit.execute_tool("review_task_action", {
                "expected_step": observed["step"], "phase": "release", "verdict": "watch",
                "expected_target": "injected destination", "observed_target": "unconfirmed destination",
                "evidence": "Injected review must not arm a watch in the protected native stream.",
            }).result
            assert "whole-task" in review.get("error", "")
            assert toolkit._task_release_watch is None
            result = toolkit.execute_tool("continue_vla", {"max_chunks": 1}).result
            assert "error" not in result, result
            assert result["executed_actions"] == 16
            assert result["environment"]["success"] is False
            assert result["cancelled_actions"] == result["paused_actions"] == 0
            for record in result["action_history"]["records"]:
                actions.extend(record["actions"][:record["executed_actions"]])
            assert harness.state.verify_step(
                "progress" if index == 0 else "retry",
                ["Native probe actions executed without command replacement; official goal remains false."],
                "Progress first, then one explicit verified physical failure to exercise takeover.",
                failure_type=None if index == 0 else "predicate_not_satisfied",
            )["ok"]
        np.testing.assert_array_equal(np.asarray(actions), np.concatenate(policy.chunks))
        assert len(policy.instructions) == 2 and set(policy.instructions) == {simulator.instruction}
        assert toolkit.evaluation_metrics["step_count"] == 32
        assert not harness.state.whole_task_policy_first_active
        assert harness.state.needs_replan
        [escalation] = harness.state.control_planning_context(include_capabilities=False)["control_mode_escalations"]
        assert escalation["reason"] == "failures" and escalation["failed_verifications"] == 1
        assert harness.state.replan("First verified VLA failure; hand over without a second policy attempt", [
            {**plan, "control_mode": "takeover", "allowed_tools": ["move_eef_path"]},
        ])["ok"]
        assert harness.state.active_step.control_mode == "takeover"
        (output / "result.json").write_text(json.dumps({
            "passed": True, "machine": "sds-server", "task": task, "seed": 7,
            "native_actions": 32, "probe_predictions": 2, "original_instruction": simulator.instruction,
            "all_actions_match_probe": True, "LLM_calls": 0, "learned_policy_calls": 0,
            "failures_before_takeover": escalation["failed_verifications"],
            "takeover_plan_registered": True, "second_failed_vla_attempt_required": False,
            "claims_task_success": False,
        }, indent=2) + "\n")
    finally:
        simulator.close()
