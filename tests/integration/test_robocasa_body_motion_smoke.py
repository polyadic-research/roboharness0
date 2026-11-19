"""Opt-in real-simulator base/torso direction and bounded handoff checks."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_BODY_SMOKE") != "1",
    reason="requires pinned RoboCasa, assets and EGL",
)


def test_real_base_and_torso_commands_and_policy_handoff(tmp_path):
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
            values = np.zeros((max_actions, 12), dtype=np.float32)
            values[:, 4] = 1
            values[:, 11] = -1
            return values

    policy = ProbePolicy()
    toolkit = RoboCasaToolkit(
        EpisodeController(simulator, policy, make_robocasa_binding()),
        initial_observation=initial, instruction=simulator.instruction,
        forward_step_guidance=True, actions_per_policy_request=5, episode_horizon=horizon,
    )
    artifacts = Path(os.environ.get("RPENT_BODY_SMOKE_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    harness = Harness(mode="plan-gated", trace_path=artifacts / "harness_trace.jsonl")
    toolkit.attach_harness(harness)
    assert harness.state.register_plan(simulator.instruction, [{
        "id": "body", "intent": "Probe bounded body control.", "max_attempts": 20,
        "allowed_tools": ["move_base", "adjust_torso", "resume_vla"],
    }])["ok"]

    def call(name, **kwargs):
        result = toolkit.execute_tool(name, kwargs).result
        assert "error" not in result, result
        return result

    def verify():
        call("verify_step", verdict="progress", evidence=["Control probe completed; task success unclaimed."],
             summary="Inspect next body command.")

    # Simulator-internal joints are inspected only by this test oracle. Runtime
    # tools see public camera/proprioception and never receive these values.
    robot = facade._environment.env.robots[0]
    base = robot.composite_controller.part_controllers["base"]
    torso = robot.composite_controller.part_controllers["torso"]

    def joints():
        return (robot.sim.data.qpos[base.qpos_index].copy(),
                robot.sim.data.qpos[torso.qpos_index].copy())

    probes = []
    try:
        for axis in range(3):
            for sign in (1, -1):
                observed = call("observe")
                before_base, before_torso = joints()
                command = [0.0] * 3
                command[axis] = sign * 0.15
                result = call("move_base", expected_step=observed["step"], velocity=command,
                              gripper="open", max_steps=5)
                after_base, after_torso = joints()
                assert result["executed_actions"] == 5
                assert result["body_motion"]["stop_command_executed"]
                delta = after_base - before_base
                assert sign * delta[axis] > 1e-4, (command, delta)
                assert abs(float((after_torso - before_torso)[0])) < 0.01
                assert np.max(np.abs(base.goal_qvel)) < 1e-6
                probes.append({"tool": "move_base", "command": command,
                               "oracle_joint_delta": delta.tolist(), "feedback": result["body_motion"]})
                call("observe")
                assert "error" in toolkit.execute_tool("resume_vla", {}).result
                verify()
                resumed = call("resume_vla", max_actions=2)
                assert resumed["executed_actions"] == 2
                call("observe")
                verify()
        for sign in (1, -1):
            observed = call("observe")
            before_base, before_torso = joints()
            result = call("adjust_torso", expected_step=observed["step"], command=sign * 0.2,
                          gripper="open", max_steps=5)
            after_base, after_torso = joints()
            delta = float((after_torso - before_torso)[0])
            assert sign * delta > 1e-4, delta
            assert np.linalg.norm(after_base - before_base) < 0.01
            assert result["executed_actions"] == 5
            probes.append({"tool": "adjust_torso", "command": sign * 0.2,
                           "oracle_joint_delta": delta, "feedback": result["body_motion"]})
            observed = call("observe")
            for camera, key in observed["_image_camera_ids"].items():
                (artifacts / f"{camera}.png").write_bytes(observed[key])
            verify()
            call("resume_vla", max_actions=2)
            call("observe")
            verify()
        assert len(policy.instructions) == 8
        assert set(policy.instructions) == {simulator.instruction}
        report = {"task": task, "seed": 7, "probes": probes, "LLM_calls": 0,
                  "learned_policy_calls": 0, "probe_policy_calls": 8,
                  "confirmed_steps": toolkit.evaluation_metrics["step_count"],
                  "purpose": "control direction and Harness smoke; no task success-rate claim"}
        (artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        simulator.close()
