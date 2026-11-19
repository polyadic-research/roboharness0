"""Opt-in real RoboCasa controls, public torso observability and physical contact."""

import io
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_PRIVILEGED_SMOKE") != "1",
    reason="requires pinned RoboCasa, assets, isolated resource lock and EGL",
)


def test_real_privileged_controls_and_contact_interrupt(tmp_path):
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import (
        PreparedEnvironment,
        RoboCasaEnvFacade,
        make_upstream_environment,
    )
    from robots.robocasa.session import RoboCasaSession
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.episode.controller import EpisodeController
    from rpent.harness import Harness

    # Test setup/oracles may inspect simulator internals. These values are
    # never passed to a tool: runtime targets below use only returned 16D state.
    artifacts = Path(os.environ.get("RPENT_PRIVILEGED_SMOKE_ARTIFACTS", tmp_path))
    artifacts.mkdir(parents=True, exist_ok=True)
    task, seed = "ArrangeDrinkware", 7
    horizon = get_task_horizon(task)
    registry = json.loads((Path(__file__).resolve().parents[2] / "robots/robocasa/task_groups.json").read_text())
    probes = []

    class NoPolicy:
        def predict(self, *args, **kwargs):
            raise AssertionError("physical tools and zoom must not request policy inference")

    class AuditedController(EpisodeController):
        def execute_action(self, action, *, _action_index=0):
            command = np.array(action, copy=True).tolist()
            transition = super().execute_action(action, _action_index=_action_index)
            row = {"episode_id": transition.episode_id, "step": transition.step_id,
                   "action": command, "state": np.asarray(transition.observation["state"]).tolist(),
                   "monotonic": time.monotonic()}
            with (artifacts / "control-steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            return transition

    def make_session(*, obstacle=False, label="controls"):
        def factory(config):
            adapter = make_upstream_environment(config)
            original_reset = adapter.reset

            def reset():
                prepared = original_reset()
                env = adapter.env
                robot = env.robots[0]
                base = robot.composite_controller.part_controllers["base"]
                # Relocate the test robot to open floor, using its own joints,
                # not fixture/object coordinates. This is a smoke fixture,
                # not a fixed-scene evaluation or task-success experiment.
                env.sim.data.qpos[base.qpos_index[:2]] += [5, 5]
                env.sim.data.qvel[:] = 0
                env.sim.forward()
                if obstacle:
                    state = env.sim.get_state().flatten()
                    world_eef = env.sim.data.site_xpos[robot.eef_site_id["right"]].copy()
                    xml = ET.fromstring(env.sim.model.get_xml())
                    ET.SubElement(xml.find("worldbody"), "geom", {
                        "name": "privileged_smoke_contact_block", "type": "box",
                        "pos": " ".join(map(str, world_eef - [0, 0, 0.12])),
                        "size": "0.25 0.25 0.025", "rgba": "0.8 0.2 0.2 1",
                        "contype": "1", "conaffinity": "1", "friction": "1 0.005 0.0001",
                    })
                    env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    robot = env.robots[0]
                for controller in robot.composite_controller.part_controllers.values():
                    controller.update(force=True)
                robot.composite_controller.update_state()
                robot.composite_controller.reset()
                # Rebuild the same standard public observation after setup.
                raw = env._get_observations(force_update=True)
                mapped = adapter._gym_environment.get_observation(raw)
                adapter._latest_observation = None
                return PreparedEnvironment(adapter._with_overview(mapped), prepared.info, prepared.instruction)

            adapter.reset = reset
            return adapter

        facade = RoboCasaEnvFacade(environment_factory=factory,
                                   dependency_identity={"robocasa_version": robocasa.__version__})
        facade.load({"task": task, "split": "pretrain", "seed": seed, "robot": "PandaOmron",
                     "camera_size": 256,
                     "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                     "object_registries": ["lightwheel", "objaverse"], "rendering_mode": "headless",
                     "upstream_horizon": horizon, "safety_ceiling": horizon,
                     "task_registry_commit": ROBOCASA_COMMIT, "task_registry_digest": registry["registry_digest"]})
        simulator = RoboCasaSession(facade, task=task, seed=seed)
        initial = simulator.reset(task, seed)
        toolkit = RoboCasaToolkit(AuditedController(simulator, NoPolicy(), make_robocasa_binding()),
                                  initial_observation=initial, instruction=simulator.instruction,
                                  forward_step_guidance=True, episode_horizon=horizon)
        harness = Harness(mode="plan-gated", trace_path=artifacts / ("contact-trace.jsonl" if obstacle else f"{label}-trace.jsonl"))
        toolkit.attach_harness(harness)
        assert harness.state.register_plan(simulator.instruction, [{
            "id": "probe", "intent": "Measure bounded controls, no task success claim.",
            "max_attempts": 10, "allowed_tools": ["navigate_to", "move_eef_path", "adjust_torso_to"],
            "skill_row": "none", "tier": "none", "control_mode": "takeover",
        }])["ok"]
        return simulator, facade, toolkit

    def call(toolkit, name, **kwargs):
        start = time.monotonic()
        result = toolkit.execute_tool(name, kwargs).result
        assert "error" not in result, result
        measured = {k: v for k, v in result.items() if k not in {"action_history", "traceback"} and not k.startswith("_image")}
        row = {"tool": name, "elapsed_seconds": time.monotonic() - start, "result": measured}
        if name in {"navigate_to", "move_eef_path", "adjust_torso_to", "zoom"}:
            if "action_history" in result:
                (artifacts / f"{len(probes):02d}-{name}-action-history.json").write_text(json.dumps(result["action_history"], indent=2) + "\n")
            probes.append(row)
            (artifacts / "probes.json").write_text(json.dumps(probes, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        return result

    def verify(toolkit):
        call(toolkit, "observe")
        call(toolkit, "verify_step", verdict="progress", evidence=["Measured control probe; task success unclaimed."],
             summary="Inspect the next independent control probe.")

    simulator, facade, toolkit = make_session(label="navigation")
    try:
        observed = call(toolkit, "observe")
        result = call(toolkit, "navigate_to", expected_step=observed["step"],
                      target={"relative": [1, 0, 90]}, gripper="open", max_steps=200)
        assert result["target_reached"], result
        assert 0.96 <= np.linalg.norm(np.subtract(result["measured_after"]["base_position"][:2],
                                                result["measured_before"]["base_position"][:2])) <= 1.04
        # The zero arm-mode sync step keeps the arm goal base-relative, so the
        # hand rides with the base instead of chasing a stale world target.
        arm_drift = np.linalg.norm(np.subtract(result["measured_after"]["eef_position"],
                                               result["measured_before"]["eef_position"]))
        probes[-1]["arm_drift_base_frame_m"] = float(arm_drift)
        assert result["arm_goal_sync_step"] and arm_drift < 0.05, arm_drift
        verify(toolkit)
        # Transition order: an EEF path straight after navigation must track.
        state = np.array(call(toolkit, "observe")["state"]["proprioception"])
        result = call(toolkit, "move_eef_path", expected_step=toolkit.evaluation_metrics["step_count"],
                      waypoints=[{"position": (state[7:10] + [0, 0, 0.05]).tolist()}], gripper="open", max_steps=60)
        assert result["target_reached"], result
        verify(toolkit)
    finally:
        simulator.close()

    # Probe EEF tracking and torso lift from a fresh reachable arm configuration.
    simulator, facade, toolkit = make_session(label="eef-torso-zoom")
    try:
        state = np.array(toolkit.observe()["state"]["proprioception"])
        points = [{"position": (state[7:10] + delta).tolist()} for delta in ([0, 0, 0.03], [0, 0, 0.06], [0.02, 0, 0.06])]
        result = call(toolkit, "move_eef_path", expected_step=toolkit.evaluation_metrics["step_count"],
                      waypoints=points, gripper="open", max_steps=150)
        assert result["target_reached"] and len(result["waypoint_results"]) == 3, result
        verify(toolkit)
        robot = facade._environment.env.robots[0]
        torso = robot.composite_controller.part_controllers["torso"]
        before = float(robot.sim.data.qpos[torso.qpos_index][0])
        result = call(toolkit, "adjust_torso_to", expected_step=toolkit.evaluation_metrics["step_count"],
                      delta_m=0.1, max_steps=50)
        after = float(robot.sim.data.qpos[torso.qpos_index][0])
        probes[-1]["oracle_torso_delta_m"] = after - before
        assert result["target_reached"] and result["torso_measurement"] == "public_base_center_z", result
        assert abs(result["measured_delta_m"] - 0.1) <= 0.003, result
        assert abs((after - before) - 0.1) <= 0.005, (before, after)
        verify(toolkit)
        observed = call(toolkit, "observe")
        image = Image.open(io.BytesIO(observed["_image_bytes"]))
        result = call(toolkit, "zoom", camera_id="robot0_agentview_left", box=[32, 32, 96, 96])
        (artifacts / "zoom.png").write_bytes(result["_image_bytes"])
        assert Image.open(io.BytesIO(result["_image_bytes"])).size == (256, 256)
        assert image.size == (256, 256)
        for camera, key in observed["_image_camera_ids"].items():
            (artifacts / f"{camera}.png").write_bytes(observed[key])
    finally:
        simulator.close()

    simulator, facade, toolkit = make_session(obstacle=True)
    try:
        observed = call(toolkit, "observe")
        state = np.array(observed["state"]["proprioception"])
        points = [{"position": (state[7:10] - [0, 0, 0.25]).tolist()},
                  {"position": (state[7:10] - [0, 0, 0.26]).tolist()}]
        result = call(toolkit, "move_eef_path", expected_step=observed["step"],
                      waypoints=points, gripper="open", max_steps=150)
        assert result["interrupt_reason"] == "deviation", result
        assert len(result["waypoint_results"]) == 1
        assert result["stop_command_executed"]
        report = {"machine": "sds-server", "task": task, "seed": seed, "probes": probes,
                  "LLM_calls": 0, "learned_policy_calls": 0,
                  "fixture": "real RoboCasa open-floor relocation; injected physical contact block for deviation",
                  "purpose": "control and Harness smoke, not fixed-scene success-rate evaluation"}
        (artifacts / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        simulator.close()
