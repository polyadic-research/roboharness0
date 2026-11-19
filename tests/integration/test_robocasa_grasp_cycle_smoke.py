"""Opt-in real simulator smoke for operation-local grasp review budgets."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_PRECLOSE_SMOKE") != "1",
    reason="requires pinned RoboCasa, assets and EGL",
)


def test_changed_operation_cannot_use_previous_rejection_fallback(tmp_path):
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.binding import make_robocasa_binding
    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import RoboCasaEnvFacade, make_upstream_environment
    from robots.robocasa.session import RoboCasaSession
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.episode.controller import EpisodeController

    task = "ArrangeDrinkware"
    registry = json.loads((Path(__file__).resolve().parents[2]
                           / "robots/robocasa/task_groups.json").read_text())
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
        "task_registry_commit": ROBOCASA_COMMIT,
        "task_registry_digest": registry["registry_digest"],
    })
    simulator = RoboCasaSession(facade, task=task, seed=7)
    initial = simulator.reset(task, 7)

    class ProbePolicy:
        def predict(self, observation, instruction, *, max_actions=None):
            values = np.zeros((1, 12), dtype=np.float32)
            values[:, 11] = 1
            return values[:max_actions]

    toolkit = RoboCasaToolkit(
        EpisodeController(simulator, ProbePolicy(), make_robocasa_binding()),
        initial_observation=initial, instruction=simulator.instruction,
        forward_step_guidance=True, actions_per_policy_request=5,
        episode_horizon=horizon,
    )

    def call(name, **kwargs):
        result = toolkit.execute_tool(name, kwargs).result
        assert "error" not in result, result
        return result

    try:
        assert call("smolvla_pick", max_chunks=1)["executed_actions"] == 0
        observed = call("observe")
        for _ in range(2):
            rejected = call("review_grasp", expected_step=observed["step"],
                            object_between_fingers=False, narrow_axis_aligned=False,
                            evidence="Probe smoke has no visual approval; target remains unconfirmed.")
        assert rejected["policy_close_allowed"] and not rejected["approved"]
        cycle = toolkit._pick_progress.cycle
        paused = call("smolvla_pick", instruction="Pick up the next observed cup.",
                      intervene=True, max_chunks=1)
        assert paused["stop_reason"] == "preclose_review_required"
        assert paused["executed_actions"] == 0
        assert toolkit.evaluation_metrics["step_count"] == 0
        assert toolkit._pick_progress.cycle > cycle
        assert paused["grasp_review"]["consecutive_rejections"] == 0
        result = {
            "machine": "sds-server", "task": task, "seed": 7,
            "previous_rejections": 2, "new_operation_rejections": 0,
            "unreviewed_close_actions": 0,
            "purpose": "real simulator control-boundary probe; no learned-policy or LLM success claim",
        }
        output = Path(os.environ.get("RPENT_PRECLOSE_SMOKE_ARTIFACTS", tmp_path))
        output.mkdir(parents=True, exist_ok=True)
        (output / "grasp-cycle-result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        simulator.close()
