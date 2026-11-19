"""Opt-in real-simulator budget smoke; uses zero controls, not learned-policy scores."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBOCASA_HORIZON_SMOKE") != "1",
    reason="set RPENT_RUN_ROBOCASA_HORIZON_SMOKE=1 in a RoboCasa/EGL environment",
)


def test_real_environment_recovery_reaches_upstream_horizon(tmp_path):
    import robocasa  # Registers the upstream Gym environments.
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from robots.robocasa.dependencies import ROBOCASA_COMMIT
    from robots.robocasa.env_server import RoboCasaEnvFacade, make_upstream_environment
    from rpent.harness.hooks import ToolHookContext
    from rpent.planning import resolve_planning_stack
    from rpent.planning.schemas import ActionFeedback

    task = "CloseDrawer"
    registry = json.loads(
        (
            Path(__file__).resolve().parents[2] / "robots/robocasa/task_groups.json"
        ).read_text()
    )
    horizon = get_task_horizon(task)
    facade = RoboCasaEnvFacade(
        environment_factory=make_upstream_environment,
        dependency_identity={"robocasa_version": robocasa.__version__},
    )
    facade.load(
        {
            "task": task,
            "split": "pretrain",
            "seed": 7,
            "robot": "PandaOmron",
            "camera_size": 256,
            "camera_names": [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            "object_registries": ["lightwheel", "objaverse"],
            "rendering_mode": "headless",
            "upstream_horizon": horizon,
            "safety_ceiling": horizon,
            "task_registry_commit": ROBOCASA_COMMIT,
            "task_registry_digest": registry["registry_digest"],
        }
    )
    harness = resolve_planning_stack("robocasa-horizon-gated-v1").build_harness(
        trace_path=tmp_path / "harness.jsonl"
    )
    harness.set_physical_tools({"move"})
    step = [{"id": "move", "intent": "close drawer", "allowed_tools": ["move"]}]
    assert harness.state.register_plan("close drawer", step)["ok"]
    try:
        initial = facade.reset(task, 7)
        episode_id = initial["episode_id"]
        for index in range(1, horizon + 1):
            context = ToolHookContext(f"control-{index}", "move", {}, True)
            assert harness.hooks.before_tool(context, harness.state).allowed
            result = facade.step(np.zeros(12, dtype=np.float32))
            assert result["episode_id"] == episode_id
            assert result["step_id"] == index
            feedback = ActionFeedback(
                execution_id=context.execution_id,
                tool_name="move",
                physical=True,
                execution_status="completed",
                environment_success=result["success"],
                environment_terminated=result["terminated"],
                environment_truncated=result["truncated"],
            )
            for verdict in harness.hooks.after_tool(context, feedback, harness.state):
                assert harness.state.apply_verdict(verdict, require_execution_id=True)[
                    "ok"
                ]
            if result["success"] or result["terminated"] or result["truncated"]:
                break
            assert harness.state.verify_step(
                "fail",
                ["native task predicate remains false after zero control"],
                "exercise recovery without resetting the environment",
                execution_id=context.execution_id,
            )["ok"]
            assert harness.state.replan("continue within the same horizon", step)["ok"]
        assert index > 100
        assert result["success"] or (index == horizon and result["truncated"])
        with pytest.raises(RuntimeError, match="terminal transition"):
            facade.step(np.zeros(12, dtype=np.float32))
        status = "success" if result["success"] else "failure"
        assert harness.hooks.before_tool(
            ToolHookContext("finish", "finish", {"status": status}, False),
            harness.state,
        ).allowed
        print(
            json.dumps(
                {
                    "task": task,
                    "horizon": horizon,
                    "steps": index,
                    "replans": harness.state.replans_used,
                    "native_success": result["success"],
                    "truncated": result["truncated"],
                    "reset_count": 1,
                    "robocasa_version": robocasa.__version__,
                    "purpose": "real environment budget smoke, not a VLA benchmark",
                }
            )
        )
    finally:
        facade.close()
