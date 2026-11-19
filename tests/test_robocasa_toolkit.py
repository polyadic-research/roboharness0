from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robots.robocasa.binding import CHECKPOINT_CHUNK_SIZE
from robots.robocasa.toolkit import RoboCasaToolkit
from rpent.episode.contracts import ChunkExecutionResult, StepResult


class _Controller:
    def __init__(self) -> None:
        self.instructions = []
        self.max_actions = []

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        del observation
        self.instructions.append(instruction)
        self.max_actions.append(max_actions)
        return ChunkExecutionResult((), 0)


class _ScriptedController:
    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.instructions = []
        self.max_actions = []

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        del observation
        self.instructions.append(instruction)
        self.max_actions.append(max_actions)
        return self.chunks.pop(0)


class _BudgetController:
    def __init__(self) -> None:
        self.instructions = []
        self.max_actions = []
        self.next_step = 1

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        del observation
        self.instructions.append(instruction)
        self.max_actions.append(max_actions)
        actions = int(max_actions)
        result = _chunk(
            self.next_step,
            z=0.5,
            gripper_qpos=(0.02, -0.02),
            actions=actions,
        )
        self.next_step += actions
        return result


class _ServoController:
    def __init__(self, observations) -> None:
        self.observations = list(observations)
        self.actions = []

    def execute_action(self, action, *, _action_index=0):
        self.actions.append((np.array(action, copy=True), _action_index))
        observation = self.observations.pop(0)
        return StepResult(
            episode_id="episode-0001",
            step_id=len(self.actions),
            observation=observation,
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
        )


class _PerceptionSource:
    def __init__(self, *, step_id: int = 0) -> None:
        self.step_id = step_id
        self.calls: list[str] = []

    def rgbd(self, camera_name: str) -> dict:
        self.calls.append(camera_name)
        return {
            "episode_id": "episode-0001",
            "step_id": self.step_id,
            "camera_name": camera_name,
            "rgb": np.zeros((6, 8, 3), dtype=np.uint8),
            "depth_m": np.full((6, 8), 2.0, dtype=np.float32),
            "intrinsic_K": np.array(
                [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "extrinsic_cam2world": np.eye(4, dtype=np.float64),
        }


class _ScriptedServoController(_ScriptedController):
    def __init__(self, chunks, servo_observations) -> None:
        super().__init__(chunks)
        self.servo_observations = list(servo_observations)
        self.actions = []

    def execute_action(self, action, *, _action_index=0):
        self.actions.append((np.array(action, copy=True), _action_index))
        observation = self.servo_observations.pop(0)
        return StepResult(
            episode_id="episode-0001",
            step_id=100 + len(self.actions),
            observation=observation,
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
        )


def _state(*, z: float, gripper_qpos: tuple[float, float]) -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[9] = z
    state[14:16] = gripper_qpos
    return state


def _pose_state(
    *,
    position: tuple[float, float, float],
    gripper_qpos: tuple[float, float],
) -> np.ndarray:
    state = _state(z=position[2], gripper_qpos=gripper_qpos)
    state[7:10] = position
    state[10:14] = (0.0, 0.0, 0.0, 1.0)
    return state


def _pose_observation(
    *,
    position: tuple[float, float, float],
    gripper_qpos: tuple[float, float],
) -> dict[str, np.ndarray]:
    return {
        **_observation(),
        "state": _pose_state(
            position=position,
            gripper_qpos=gripper_qpos,
        ),
    }


def _pose_chunk(
    start_step: int,
    *,
    position: tuple[float, float, float],
    gripper_qpos: tuple[float, float],
    actions: int,
    success: bool = False,
) -> ChunkExecutionResult:
    observation = _pose_observation(
        position=position,
        gripper_qpos=gripper_qpos,
    )
    transitions = tuple(
        StepResult(
            episode_id="episode-0001",
            step_id=start_step + offset,
            observation=observation,
            reward=0.0,
            success=success and offset == actions - 1,
            terminated=False,
            truncated=False,
        )
        for offset in range(actions)
    )
    return ChunkExecutionResult(transitions, actions)


def _chunk(
    start_step: int,
    *,
    z: float,
    gripper_qpos: tuple[float, float],
    success: bool = False,
    terminated: bool = False,
    actions: int = 25,
) -> ChunkExecutionResult:
    transitions = tuple(
        StepResult(
            episode_id="episode-0001",
            step_id=start_step + offset,
            observation={
                **_observation(),
                "state": _state(z=z, gripper_qpos=gripper_qpos),
            },
            reward=0.0,
            success=success and offset == actions - 1,
            terminated=terminated and offset == actions - 1,
            truncated=False,
        )
        for offset in range(actions)
    )
    return ChunkExecutionResult(transitions, actions)


def _trajectory_chunk(
    start_step: int,
    states: list[tuple[float, tuple[float, float]]],
    *,
    success: bool = False,
    terminated: bool = False,
    gripper_actions: tuple[float, ...] | None = None,
) -> ChunkExecutionResult:
    transitions = tuple(
        StepResult(
            episode_id="episode-0001",
            step_id=start_step + offset,
            observation={
                **_observation(),
                "state": _state(z=z, gripper_qpos=gripper_qpos),
            },
            reward=0.0,
            success=success and offset == len(states) - 1,
            terminated=terminated and offset == len(states) - 1,
            truncated=False,
        )
        for offset, (z, gripper_qpos) in enumerate(states)
    )
    actions = ()
    if gripper_actions is not None:
        actions = tuple(
            np.array([0.0] * 11 + [gripper], dtype=np.float32)
            for gripper in gripper_actions
        )
    return ChunkExecutionResult(transitions, len(states), actions=actions)


def _observation() -> dict[str, np.ndarray]:
    return {
        "state": np.zeros(16, dtype=np.float32),
        "robot0_agentview_center": np.zeros((6, 8, 3), dtype=np.uint8),
        "robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def _initial_pick_observation() -> dict[str, np.ndarray]:
    return {
        **_observation(),
        "state": _state(z=0.5, gripper_qpos=(0.02, -0.02)),
    }


@pytest.mark.parametrize("tool_name", ("continue_vla", "smolvla_pick"))
def test_planner_step_guidance_reaches_policy_after_replan(tmp_path, tool_name):
    from rpent.harness import Harness

    task = "Pick the peeler from the drawer and place it on the counter."
    initial = "Approach the peeler in the open drawer with the gripper open."
    corrected = "Back away from the drawer front and align above the same peeler."

    class PlanGenerator:
        def generate(self, **kwargs):
            return self.plan(kwargs["goal"], initial)

        def generate_replan(self, **kwargs):
            return self.plan(kwargs["goal"], corrected)

        @staticmethod
        def plan(goal, intent):
            return {
                "goal": goal,
                "strategy": intent,
                "steps": [{
                    "id": "acquire-peeler",
                    "intent": intent,
                    "allowed_tools": [tool_name],
                    "expected_evidence": ["The same peeler is retained."],
                }],
            }

    controller = _BudgetController()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction=task,
        forward_step_guidance=True,
        actions_per_policy_request=5,
        episode_horizon=750,
    )
    toolkit.attach_harness(Harness(
        mode="plan-gated", trace_path=tmp_path / "trace.jsonl",
        plan_generator=PlanGenerator(), require_environment_success=True,
    ))
    proposed = toolkit.execute_tool(
        "create_plan", {"goal": task, "observations": ["The peeler is in the drawer."]},
    )
    assert toolkit.execute_tool(
        "register_plan", {"proposal_id": proposed.result["proposal_id"]},
    ).result["ok"]
    first = toolkit.execute_tool(tool_name, {"intervene": True, "instruction": initial})
    assert first.result["failure_reason"] == "budget_exhausted"
    assert first.result["executed_actions"] == 50
    assert toolkit.execute_tool("verify_step", {
        "verdict": "fail", "evidence": ["The gripper has not reached the peeler."],
        "summary": "Approach failed; change the current action.",
    }).result["ok"]
    repair = toolkit.execute_tool("replan", {
        "reason": "The gripper is stalled above the drawer front.",
        "observations": ["The same peeler is still supported in the drawer."],
    })
    assert toolkit.execute_tool(
        "register_plan", {"proposal_id": repair.result["proposal_id"]},
    ).result["ok"]
    kept = toolkit.execute_tool(tool_name, {"intervene": False})
    assert kept.result["instruction"] == initial
    assert toolkit.execute_tool("verify_step", {
        "verdict": "fail", "evidence": ["The target is still in the drawer."],
        "summary": "Continued execution did not acquire the peeler.",
    }).result["ok"]
    repair = toolkit.execute_tool("replan", {
        "reason": "A grounded instruction correction is now needed.",
        "observations": ["The peeler is still supported in the drawer."],
    })
    assert toolkit.execute_tool(
        "register_plan", {"proposal_id": repair.result["proposal_id"]},
    ).result["ok"]
    second = toolkit.execute_tool(tool_name, {"intervene": True, "instruction": corrected})

    first_prompt = initial
    second_prompt = corrected
    assert controller.instructions == [first_prompt] * 20 + [second_prompt] * 10
    assert first.result["instruction"] == first_prompt
    assert second.result["instruction"] == second_prompt
    assert second.result["step_guidance"] == corrected
    assert second.result["episode_budget"]["steps_used"] == 150
    assert second.result["harness"]["contract_success"] is False
    assert toolkit.observe()["task_language"] == task


def test_forwarded_execute_vla_updates_guidance_only_with_explicit_intervention():
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller, initial_observation=_observation(),
        instruction="Close the right drawer.", forward_step_guidance=True,
        actions_per_policy_request=5,
    )
    rejected = toolkit.execute_tool("execute_vla", {"intervene": True})
    assert "instruction" in rejected.result["error"]
    assert controller.instructions == []
    for guidance in ("Approach the right drawer front.", "Withdraw before pushing again."):
        result = toolkit.execute_tool("execute_vla", {
            "intervene": True, "instruction": guidance,
        })
        assert "error" not in result.result
    assert controller.instructions == [
        "Approach the right drawer front.",
        "Withdraw before pushing again.",
    ]
    assert controller.max_actions == [5, 5]


@pytest.mark.parametrize("guidance", ("   ", "x" * 1001))
def test_forwarded_execute_vla_rejects_invalid_guidance_before_policy(guidance):
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller, initial_observation=_observation(),
        instruction="Close the right drawer.", forward_step_guidance=True,
    )
    with pytest.raises(ValueError, match="instruction"):
        toolkit.execute_vla(instruction=guidance, intervene=True)
    assert controller.instructions == []


def test_forwarded_vla_tool_schemas_explain_current_step_guidance():
    toolkit = RoboCasaToolkit(
        _Controller(), initial_observation=_observation(),
        instruction="Close the right drawer.", forward_step_guidance=True,
    )
    specs = {spec["name"]: spec for spec in toolkit.get_tools_spec()}
    for name in ("execute_vla", "continue_vla", "smolvla_pick"):
        schema = specs[name]["input_schema"]
        assert "instruction" not in schema.get("required", [])
        assert "forwarded" in schema["properties"]["instruction"]["description"]
        assert "intervene=true" in schema["properties"]["instruction"]["description"]
        assert schema["properties"]["intervene"]["default"] is False


def test_prompt_explains_pi05_step_guidance_after_replanning():
    from robots.robocasa.prompts.system import system_prompt

    action = system_prompt()["ACTION"]
    assert "For a language-instructed VLA" in action
    assert "intervene=false" in action
    assert "after replanning" in action
    assert "recovery, and closed_empty" in action
    assert "exactly unchanged" in action
    assert "without appending the original task" in action
    assert "training task ID" in action
    assert "all run the same checkpoint policy on its canonical task instruction" not in action


def test_legacy_planner_write_is_confined_to_agent_artifacts(tmp_path) -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        agent_artifact_root=tmp_path / "agent_artifacts",
    )

    written = toolkit.execute_tool(
        "write_text_file",
        {"path": "audit.json", "content": '{"ok": true}'},
    )
    escaped = toolkit.execute_tool(
        "write_text_file",
        {"path": "../escaped.json", "content": "unsafe"},
    )

    audit_path = tmp_path / "agent_artifacts" / "audit.json"
    assert written.result["path"] == str(audit_path)
    assert audit_path.read_text() == '{"ok": true}'
    assert "outside run output" in escaped.result["error"]
    assert not (tmp_path / "escaped.json").exists()
    write_spec = next(
        spec for spec in toolkit.get_tools_spec() if spec["name"] == "write_text_file"
    )
    assert "agent_artifacts" in write_spec["description"]


def test_legacy_planner_write_rejects_agent_artifact_root_symlink(tmp_path) -> None:
    attempt_root = tmp_path / "attempt"
    outside_root = tmp_path / "outside"
    attempt_root.mkdir()
    outside_root.mkdir()
    (attempt_root / "agent_artifacts").symlink_to(
        outside_root,
        target_is_directory=True,
    )
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        agent_artifact_root=attempt_root / "agent_artifacts",
    )

    escaped = toolkit.execute_tool(
        "write_text_file",
        {"path": "audit.json", "content": "unsafe"},
    )

    assert "outside run output" in escaped.result["error"]
    assert not (outside_root / "audit.json").exists()


def test_smolvla_pick_does_not_succeed_after_one_incomplete_25_action_chunk() -> None:
    controller = _ScriptedController([_chunk(1, z=0.43, gripper_qpos=(0.02, -0.02))])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=1)

    assert result["success"] is False
    assert result["failure_reason"] == "budget_exhausted"
    assert result["chunks_used"] == 1
    assert result["executed_actions"] == 25


def test_smolvla_pick_succeeds_after_descent_closure_and_lift_across_chunks() -> None:
    controller = _ScriptedController(
        [
            _chunk(1, z=0.42, gripper_qpos=(0.02, -0.02)),
            _chunk(26, z=0.41, gripper_qpos=(0.006, -0.006)),
            _chunk(51, z=0.48, gripper_qpos=(0.006, -0.006)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=4)

    assert result["success"] is True
    assert result["chunks_used"] == 3
    assert result["diagnostics"]["descent_confirmed"] is True
    assert result["diagnostics"]["closure_confirmed"] is True
    assert result["diagnostics"]["post_descent_ascent_m"] >= 0.05


def test_smolvla_pick_rejects_fully_closed_empty_gripper() -> None:
    controller = _ScriptedController(
        [
            _chunk(1, z=0.41, gripper_qpos=(0.02, -0.02)),
            _chunk(26, z=0.41, gripper_qpos=(0.0, 0.0)),
            _chunk(51, z=0.48, gripper_qpos=(0.0, 0.0)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=3)

    assert result["success"] is False
    assert result["failure_reason"] == "closed_empty"


def test_smolvla_pick_stops_on_first_reliable_closed_empty_cycle() -> None:
    controller = _ScriptedController(
        [
            _trajectory_chunk(
                1,
                [
                    (0.44, (0.02, -0.02)),
                    (0.41, (0.0, 0.0)),
                    (0.47, (0.0, 0.0)),
                ],
            ),
            _chunk(4, z=0.5, gripper_qpos=(0.02, -0.02)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=2)

    assert result["success"] is False
    assert result["failure_reason"] == "closed_empty"
    assert result["stop_reason"] == "closed_empty"
    assert result["chunks_used"] == 1
    assert result["transitions_used"] == result["executed_actions"] == 3
    assert len(controller.instructions) == 1
    assert result["observation_summary"]["gripper_opening"] == 0.0
    assert result["observation_summary"]["eef_z"] == pytest.approx(0.47)


def test_smolvla_pick_stops_without_requesting_more_chunks_on_terminal_success() -> (
    None
):
    controller = _ScriptedController(
        [
            _chunk(
                1,
                z=0.43,
                gripper_qpos=(0.02, -0.02),
                success=True,
                terminated=True,
            ),
            _chunk(26, z=0.4, gripper_qpos=(0.01, -0.01)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=3)

    assert result["environment"]["success"] is True
    assert result["chunks_used"] == 1
    assert len(controller.instructions) == 1


def test_smolvla_pick_terminal_disables_retry_and_later_execution() -> None:
    controller = _ScriptedController(
        [
            _trajectory_chunk(
                1,
                [(0.49, (0.02, -0.02))],
                terminated=True,
            )
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        actions_per_policy_request=25,
    )

    terminal = toolkit.smolvla_pick("grasp the kettle", max_chunks=1)
    blocked = toolkit.smolvla_pick("grasp the kettle again", max_chunks=1)

    assert terminal["failure_reason"] == "episode_terminal"
    assert terminal["retry_allowed"] is False
    assert blocked["stop_reason"] == "episode_terminal"
    assert blocked["retry_allowed"] is False
    assert blocked["transitions_used"] == 0
    assert len(controller.instructions) == 1


def test_smolvla_pick_allows_only_one_consecutive_closed_empty_recovery() -> None:
    empty_cycle = [
        (0.44, (0.02, -0.02)),
        (0.41, (0.0, 0.0)),
        (0.47, (0.0, 0.0)),
    ]
    controller = _ScriptedController(
        [
            _trajectory_chunk(1, empty_cycle),
            _trajectory_chunk(4, empty_cycle),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        actions_per_policy_request=25,
    )

    first = toolkit.smolvla_pick("first grasp", max_chunks=1)
    second = toolkit.smolvla_pick("replanned grasp", max_chunks=1)
    blocked = toolkit.smolvla_pick("third grasp", max_chunks=1)

    assert first["failure_reason"] == "closed_empty"
    assert first["retry_allowed"] is True
    assert second["failure_reason"] == "closed_empty"
    assert second["retry_allowed"] is False
    assert second["stop_reason"] == "closed_empty_retry_exhausted"
    assert blocked["stop_reason"] == "closed_empty_retry_exhausted"
    assert blocked["transitions_used"] == 0
    assert len(controller.instructions) == 2


def test_smolvla_pick_keeps_checkpoint_instruction_stable_across_chunks() -> None:
    controller = _ScriptedController(
        [
            _chunk(1, z=0.45, gripper_qpos=(0.02, -0.02)),
            _chunk(26, z=0.43, gripper_qpos=(0.02, -0.02)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    toolkit.smolvla_pick("grasp the kettle handle", max_chunks=2)

    assert controller.instructions == [
        "KettleBoiling",
        "KettleBoiling",
    ]
    assert controller.max_actions == [25, 25]


def test_smolvla_pick_guides_the_single_closed_empty_retry() -> None:
    empty_cycle = [
        (0.44, (0.02, -0.02)),
        (0.41, (0.0, 0.0)),
        (0.47, (0.0, 0.0)),
    ]
    controller = _ScriptedController(
        [
            _trajectory_chunk(1, empty_cycle),
            _chunk(4, z=0.44, gripper_qpos=(0.02, -0.02)),
            _chunk(29, z=0.41, gripper_qpos=(0.02, -0.02)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    first = toolkit.smolvla_pick("grasp the initial target", max_chunks=1)
    second = toolkit.smolvla_pick(
        "grasp the corrected right-counter kettle",
        max_chunks=2,
    )
    blocked = toolkit.smolvla_pick("third grasp", max_chunks=1)

    assert first["failure_reason"] == "closed_empty"
    assert first["retry_allowed"] is True
    assert second["failure_reason"] == "budget_exhausted"
    assert second["retry_allowed"] is False
    assert blocked["stop_reason"] == "closed_empty_retry_exhausted"
    assert blocked["transitions_used"] == 0
    assert controller.instructions == [
        "KettleBoiling",
        (
            "KettleBoiling\nCurrent action-step guidance: "
            "grasp the corrected right-counter kettle"
        ),
        (
            "KettleBoiling\nCurrent action-step guidance: "
            "grasp the corrected right-counter kettle"
        ),
    ]


def test_smolvla_pick_persists_policy_close_pulse_during_single_recovery() -> None:
    empty_cycle = [
        (0.44, (0.02, -0.02)),
        (0.41, (0.0, 0.0)),
        (0.47, (0.0, 0.0)),
    ]
    openings = (
        0.079,
        0.073,
        0.067,
        0.061,
        0.057,
        0.053,
        0.060,
        0.068,
        0.074,
        0.078,
        0.079,
    )
    candidate_states = [
        (0.40, (opening / 2.0, -opening / 2.0)) for opening in openings
    ]
    recovery_chunk = _trajectory_chunk(
        4,
        candidate_states,
        gripper_actions=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0) + (-1.0,) * 5,
    )
    for transition, opening in zip(
        recovery_chunk.transitions,
        openings,
        strict=True,
    ):
        transition.observation["state"][7:10] = (0.45, -0.55, 0.40)
        transition.observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
        transition.observation["state"][14:16] = (
            opening / 2.0,
            -opening / 2.0,
        )

    servo_observations = []
    for z, opening in (
        (0.40, 0.040),
        (0.40, 0.025),
        (0.40, 0.020),
        (0.45, 0.040),
        (0.46, 0.040),
        (0.46, 0.040),
        (0.46, 0.025),
        (0.46, 0.020),
    ):
        observation = _observation()
        observation["state"][7:10] = (0.45, -0.55, z)
        observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
        observation["state"][14:16] = (opening / 2.0, -opening / 2.0)
        servo_observations.append(observation)

    controller = _ScriptedServoController(
        [_trajectory_chunk(1, empty_cycle), recovery_chunk],
        servo_observations,
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    first = toolkit.smolvla_pick("grasp the initial target", max_chunks=1)
    recovered = toolkit.smolvla_pick(
        "grasp the corrected right-counter kettle handle",
        max_chunks=1,
    )

    assert first["failure_reason"] == "closed_empty"
    assert recovered["success"] is True
    assert recovered["stop_reason"] == "recovery_close_persisted"
    assert recovered["diagnostics"]["closure_confirmed"] is True
    assert recovered["diagnostics"]["post_descent_ascent_m"] >= 0.05
    assert recovered["diagnostics"]["recovery_pose"]["source"] == (
        "policy_partial_closure_pose"
    )
    assert recovered["executed_actions"] == 19
    assert len(controller.actions) == 8
    assert all(action[0][11] == 1.0 for action in controller.actions)
    assert not controller.servo_observations


def test_smolvla_pick_does_not_latch_open_policy_actions() -> None:
    empty_cycle = [
        (0.44, (0.02, -0.02)),
        (0.41, (0.0, 0.0)),
        (0.47, (0.0, 0.0)),
    ]
    openings = (0.079, 0.073, 0.067, 0.061, 0.057, 0.053)
    recovery_chunk = _trajectory_chunk(
        4,
        [(0.40, (opening / 2.0, -opening / 2.0)) for opening in openings],
        gripper_actions=(-1.0,) * len(openings),
    )
    controller = _ScriptedServoController(
        [_trajectory_chunk(1, empty_cycle), recovery_chunk],
        [],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    toolkit.smolvla_pick("grasp the initial target", max_chunks=1)
    recovered = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=1)

    assert recovered["success"] is False
    assert recovered["failure_reason"] == "budget_exhausted"
    assert recovered["retry_allowed"] is False
    assert controller.actions == []


def test_smolvla_pick_does_not_return_to_a_stale_policy_close_pose() -> None:
    empty_cycle = [
        (0.44, (0.02, -0.02)),
        (0.41, (0.0, 0.0)),
        (0.47, (0.0, 0.0)),
    ]
    openings = (0.079, 0.073, 0.067, 0.061, 0.057, 0.053, 0.079)
    recovery_chunk = _trajectory_chunk(
        4,
        [(0.40, (opening / 2.0, -opening / 2.0)) for opening in openings],
        gripper_actions=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
    )
    for transition in recovery_chunk.transitions:
        transition.observation["state"][7:10] = (0.45, -0.55, 0.40)
        transition.observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    recovery_chunk.transitions[-1].observation["state"][7] = 0.50
    controller = _ScriptedServoController(
        [_trajectory_chunk(1, empty_cycle), recovery_chunk],
        [],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    toolkit.smolvla_pick("grasp the initial target", max_chunks=1)
    recovered = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=1)

    assert recovered["failure_reason"] == "budget_exhausted"
    assert recovered["retry_allowed"] is False
    assert controller.actions == []


def test_smolvla_pick_reports_first_closure_pose_for_bounded_recovery() -> None:
    low = _chunk(1, z=0.39, gripper_qpos=(0.02, -0.02))
    low.transitions[-1].observation["state"][7:10] = (0.21, -0.08, 0.39)
    low.transitions[-1].observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    closure = _chunk(26, z=0.41, gripper_qpos=(0.01, -0.01))
    closure.transitions[0].observation["state"][7:10] = (0.24, -0.09, 0.41)
    closure.transitions[0].observation["state"][10:14] = (0.0, 0.0, 1.0, 0.0)
    controller = _ScriptedController([low, closure])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=2)

    recovery_pose = result["diagnostics"]["recovery_pose"]
    assert recovery_pose["source"] == "first_confirmed_closure_pose"
    assert np.allclose(recovery_pose["position"], [0.24, -0.09, 0.41])
    assert np.allclose(recovery_pose["quaternion"], [0.0, 0.0, 1.0, 0.0])


def test_smolvla_pick_hides_recovery_pose_after_unconfirmed_closure() -> None:
    low = _chunk(1, z=0.39, gripper_qpos=(0.02, -0.02))
    closure = _chunk(26, z=0.41, gripper_qpos=(0.01, -0.01))
    closure.transitions[0].observation["state"][7:10] = (0.24, -0.09, 0.41)
    closure.transitions[0].observation["state"][10:14] = (0.0, 0.0, 1.0, 0.0)
    empty_without_lift = _chunk(51, z=0.412, gripper_qpos=(0.0, 0.0))
    controller = _ScriptedController([low, closure, empty_without_lift])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=3)

    assert result["failure_reason"] == "budget_exhausted"
    assert result["diagnostics"]["descent_confirmed"] is True
    assert result["diagnostics"]["closure_confirmed"] is False
    assert result["diagnostics"]["recovery_pose"] is None


def test_smolvla_pick_stops_after_first_empty_grasp_cycle() -> None:
    first_low = _chunk(1, z=0.39, gripper_qpos=(0.02, -0.02))
    first_empty = _chunk(26, z=0.41, gripper_qpos=(0.0, 0.0))
    first_empty.transitions[0].observation["state"][14:16] = (0.01, -0.01)
    first_raised = _chunk(51, z=0.48, gripper_qpos=(0.006, -0.006))
    second_low = _chunk(76, z=0.37, gripper_qpos=(0.02, -0.02))
    second_empty = _chunk(101, z=0.39, gripper_qpos=(0.0, 0.0))
    second_empty.transitions[0].observation["state"][7:10] = (0.31, -0.12, 0.39)
    second_empty.transitions[0].observation["state"][14:16] = (0.01, -0.01)
    second_raised = _chunk(126, z=0.46, gripper_qpos=(0.006, -0.006))
    should_not_run = _chunk(151, z=0.49, gripper_qpos=(0.02, -0.02))
    controller = _ScriptedController(
        [
            first_low,
            first_empty,
            first_raised,
            second_low,
            second_empty,
            second_raised,
            should_not_run,
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=7)

    assert result["success"] is False
    assert result["failure_reason"] == "closed_empty"
    assert result["chunks_used"] == 3
    assert len(controller.instructions) == 3
    assert result["diagnostics"]["recovery_pose"] is None


def test_smolvla_pick_does_not_offer_recovery_pose_before_confirmed_descent() -> None:
    shallow = _chunk(1, z=0.48, gripper_qpos=(0.04, -0.04))
    controller = _ScriptedController([shallow])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.smolvla_pick("grasp the kettle handle", max_chunks=1)

    assert result["diagnostics"]["descent_confirmed"] is False
    assert result["diagnostics"]["recovery_pose"] is None


@pytest.mark.parametrize("budget_scope", ["chunks", "transitions"])
def test_pick_progress_survives_tool_budgets_and_changed_audit_guidance(budget_scope) -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [(0.41, (0.04, -0.04))]),
        _trajectory_chunk(2, [(0.41, (0.01, -0.01))]),
        _trajectory_chunk(3, [(0.48, (0.01, -0.01))]),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
        actions_per_policy_request=1,
        max_transitions_per_tool_call=1 if budget_scope == "transitions" else 250,
    )
    max_chunks = 10 if budget_scope == "transitions" else 1
    descent = toolkit.smolvla_pick("approach", max_chunks=max_chunks)
    closure = toolkit.smolvla_pick("corrected approach description", max_chunks=max_chunks)
    assert closure["diagnostics"]["closure_confirmed"] is True
    lift = toolkit.smolvla_pick("lift", max_chunks=max_chunks)
    assert lift["success"] is True
    already_held = toolkit.smolvla_pick("check acquisition again", max_chunks=1)

    assert descent["success"] is False
    assert descent["stop_reason"] == (
        "transition_budget_exhausted" if budget_scope == "transitions" else "max_chunks_exhausted"
    )
    assert closure["diagnostics"]["closure_confirmed"] is True
    assert lift["success"] is True
    assert already_held["success"] is True
    assert already_held["executed_actions"] == 0
    assert controller.instructions == ["Place the toasted item on a plate."] * 3


@pytest.mark.parametrize("intermediate_tool", ["continue_vla", "execute_vla"])
def test_pick_progress_tracks_intervening_policy_tools(intermediate_tool) -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [(0.41, (0.04, -0.04))]),
        _trajectory_chunk(2, [(0.41, (0.01, -0.01)), (0.48, (0.01, -0.01))]),
        _trajectory_chunk(4, [(0.48, (0.01, -0.01))]),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )
    toolkit.smolvla_pick("approach", max_chunks=1)
    if intermediate_tool == "continue_vla":
        toolkit.continue_vla("finish acquisition", max_chunks=1)
    else:
        toolkit.execute_vla()

    result = toolkit.smolvla_pick("check acquisition", max_chunks=1)

    assert result["success"] is True
    assert result["executed_actions"] == 0
    assert len(controller.instructions) == 2


def test_pick_does_not_reuse_lift_that_preceded_closure() -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [
            (0.41, (0.04, -0.04)),
            (0.48, (0.04, -0.04)),
            (0.48, (0.01, -0.01)),
        ]),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )

    result = toolkit.smolvla_pick("acquire", max_chunks=1)

    assert result["success"] is False
    assert result["diagnostics"]["closure_confirmed"] is True


@pytest.mark.parametrize("final_opening", [0.0, 0.08])
def test_pick_checks_final_chunk_state_after_lift(final_opening) -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [
            (0.41, (0.01, -0.01)),
            (0.48, (0.01, -0.01)),
            (0.48, (final_opening / 2, -final_opening / 2)),
        ]),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )

    result = toolkit.smolvla_pick("acquire", max_chunks=1)

    assert result["success"] is False
    assert result["diagnostics"]["closure_confirmed"] is False
    assert result["diagnostics"]["recovery_pose"] is None


@pytest.mark.parametrize("release_opening", [0.04, 0.08])
def test_released_pick_requires_new_descent_closure_and_lift(release_opening) -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [(0.41, (0.01, -0.01)), (0.48, (0.01, -0.01))]),
        _trajectory_chunk(3, [
            (0.48, (release_opening / 2, -release_opening / 2)),
            (0.48, (0.01, -0.01)),
        ]),
        _trajectory_chunk(5, [(0.48, (0.01, -0.01))]),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )
    assert toolkit.smolvla_pick("acquire", max_chunks=1)["success"] is True
    toolkit.continue_vla("release", max_chunks=1)

    result = toolkit.smolvla_pick("acquire next object", max_chunks=1)

    assert result["success"] is False
    assert result["diagnostics"]["closure_confirmed"] is False
    assert result["diagnostics"]["recovery_pose"] is None


def test_servo_release_invalidates_previous_pick_progress() -> None:
    acquired = _trajectory_chunk(
        1, [(0.41, (0.01, -0.01)), (0.48, (0.01, -0.01))]
    )
    for transition in acquired.transitions:
        transition.observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    controller = _ScriptedServoController(
        [
            acquired,
            _trajectory_chunk(4, [(0.48, (0.01, -0.01))]),
        ],
        [_pose_observation(position=(0.0, 0.0, 0.48), gripper_qpos=(0.04, -0.04))],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )
    assert toolkit.smolvla_pick("acquire", max_chunks=1)["success"] is True
    toolkit.servo_eef(
        target_position=[0.0, 0.0, 0.48],
        target_quaternion=[0.0, 0.0, 0.0, 1.0],
        gripper="open",
        max_steps=1,
    )

    result = toolkit.smolvla_pick("acquire again", max_chunks=1)

    assert result["success"] is False
    assert result["diagnostics"]["closure_confirmed"] is False


def test_terminal_continue_blocks_reuse_of_confirmed_pick() -> None:
    controller = _ScriptedController([
        _trajectory_chunk(1, [(0.41, (0.01, -0.01)), (0.48, (0.01, -0.01))]),
        _trajectory_chunk(3, [(0.49, (0.01, -0.01))], terminated=True),
    ])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Place the toasted item on a plate.",
    )
    toolkit.smolvla_pick("acquire", max_chunks=1)
    toolkit.continue_vla("place", max_chunks=1)

    result = toolkit.smolvla_pick("check acquisition", max_chunks=1)

    assert result["success"] is False
    assert result["stop_reason"] == "episode_terminal"
    assert result["executed_actions"] == 0


class _RecordedPickController:
    """Feed immutable recorded transitions; no simulator or inference calls."""

    def __init__(self, fixture) -> None:
        self.transitions = fixture["transitions"]
        self.position = 0
        self.instructions = []

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        del observation
        self.instructions.append(instruction)
        rows = self.transitions[self.position:self.position + int(max_actions)]
        assert rows
        self.position += len(rows)
        transitions = tuple(
            StepResult(
                episode_id="recorded-toaster-seed8",
                step_id=row["step_id"],
                observation={"state": np.asarray(row["state"], dtype=np.float32)},
                reward=0.0,
                success=row["success"],
                terminated=row["terminated"],
                truncated=row["truncated"],
            )
            for row in rows
        )
        return ChunkExecutionResult(
            transitions,
            len(rows),
            actions=tuple(np.asarray(row["action"], dtype=np.float32) for row in rows),
        )


def test_recorded_toaster_seed8_recognizes_grasp_across_fifty_step_calls() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/robocasa/toaster_pick_s8.json").read_text()
    )
    controller = _RecordedPickController(fixture)
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation={"state": np.asarray(fixture["initial_state"], dtype=np.float32)},
        instruction=fixture["instruction"],
        actions_per_policy_request=5,
    )
    result = None
    for _ in range(7):
        result = toolkit.smolvla_pick("acquire the toasted item")
        if result["success"]:
            break

    assert result["success"] is True
    assert result["diagnostics"]["closure_confirmed"] is True
    assert result["observation_summary"]["step"] <= 350
    assert controller.instructions == [fixture["instruction"]] * len(controller.instructions)
    position = controller.position
    cached = toolkit.smolvla_pick("check the same acquisition")
    assert cached["success"] is True
    assert cached["observation_summary"]["step"] == position
    assert cached["executed_actions"] == 0
    assert controller.position == position
    while controller.position < len(controller.transitions):
        toolkit.continue_vla("transport, place, and release")
    assert toolkit.evaluation_metrics["simulator_success"] is False
    assert toolkit.evaluation_metrics["step_count"] == 600


def test_toolkit_exposes_policy_neutral_tools_without_reset_or_skill_catalog() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )
    names = {spec["name"] for spec in toolkit.get_tools_spec()}

    assert {
        "observe",
        "execute_vla",
        "servo_eef",
        "smolvla_pick",
        "continue_vla",
        "back_project",
        "finish",
    }.issubset(names)
    assert "reset" not in names
    assert toolkit._harness_physical_tools == {
        "execute_vla",
        "servo_eef",
        "smolvla_pick",
        "continue_vla",
        "adjust_eef",
        "move_base",
        "adjust_torso",
        "resume_vla",
        "navigate_to",
        "move_eef_path",
        "adjust_torso_to",
        "pour_into",
        "place_container",
    }
    assert toolkit.skill_outcome_adapter_id == "robocasa-skill-outcome-v2"
    assert "back_project" in toolkit._harness_evidence_tools
    assert "back_project" not in toolkit._harness_physical_tools
    assert not {name for name in names if name.startswith(("libero", "pi0"))}


@pytest.mark.parametrize(
    "camera",
    ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
)
def test_back_project_localizes_current_pixels_without_executing_actions(camera) -> None:
    observation = _observation()
    observation["state"][6] = 1.0
    observation["state"][7:10] = (0.0, 0.0, 1.8)
    observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    source = _PerceptionSource()
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=observation,
        instruction="Boil water in the kettle.",
        perception_source=source,
    )

    result = toolkit.execute_tool(
        "back_project",
        {
            "camera": camera,
            "expected_step": 0,
            "points": [[3, 4], [3, 5], [4, 4], [4, 5], [2, 4]],
        },
    ).result

    assert result["camera"] == camera
    assert result["episode_id"] == "episode-0001"
    assert result["step"] == 0
    assert result["valid_samples"] == 5
    assert len(result["relative_base_xyz"]) == 3
    assert result["recommended_standoff_xyz"] == pytest.approx(
        [0.0, 0.0, 1.9]
    )
    assert result["recommended_standoff_quaternion"] == [0.0, 0.0, 0.0, 1.0]
    assert result["recommended_standoff_distance_m"] == pytest.approx(0.1)
    assert source.calls == [camera]
    assert toolkit.evaluation_metrics["action_count"] == 0


def test_burner_control_requires_grounded_standoff_and_fresh_observe() -> None:
    start = _observation()
    start["state"][6] = 1.0
    start["state"][7:10] = (0.0, 0.0, 1.8)
    start["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    start["state"][14:16] = (0.04, -0.04)
    standoff = _pose_observation(
        position=(0.0, 0.0, 1.9),
        gripper_qpos=(0.04, -0.04),
    )
    policy_chunk = _pose_chunk(
        2,
        position=(0.0, 0.0, 1.9),
        gripper_qpos=(0.04, -0.04),
        actions=50,
    )
    controller = _ScriptedServoController([policy_chunk], [standoff])
    source = _PerceptionSource()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=start,
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=50,
        perception_source=source,
        max_transitions_per_tool_call=50,
    )

    with pytest.raises(ValueError, match="RGB-D grounding"):
        toolkit.continue_vla(
            "operate the grounded control",
            max_chunks=1,
            step_intent="burner_control",
        )

    grounding = toolkit.back_project(
        "robot0_agentview_left",
        0,
        [[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]],
    )
    with pytest.raises(ValueError, match="servo_eef standoff"):
        toolkit.continue_vla(
            "operate the grounded control",
            max_chunks=1,
            step_intent="burner_control",
        )

    servo = toolkit.servo_eef(
        grounding["recommended_standoff_xyz"],
        grounding["recommended_standoff_quaternion"],
        "open",
        max_steps=1,
    )
    assert servo["target_reached"] is True
    with pytest.raises(ValueError, match="fresh observe"):
        toolkit.continue_vla(
            "operate the grounded control",
            max_chunks=1,
            step_intent="burner_control",
        )

    assert toolkit.observe()["step"] == 101
    result = toolkit.continue_vla(
        "operate the grounded control",
        max_chunks=1,
        step_intent="burner_control",
    )

    assert result["executed_actions"] == 50
    assert controller.instructions == ["KettleBoiling"]


def test_mismatched_servo_does_not_authorize_grounded_burner_control() -> None:
    start = _observation()
    start["state"][6] = 1.0
    start["state"][7:10] = (0.0, 0.0, 1.8)
    start["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    start["state"][14:16] = (0.04, -0.04)
    unrelated_pose = _pose_observation(
        position=(0.05, 0.0, 1.9),
        gripper_qpos=(0.04, -0.04),
    )
    controller = _ServoController([unrelated_pose])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=start,
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        perception_source=_PerceptionSource(),
    )
    toolkit.back_project(
        "robot0_agentview_left",
        0,
        [[3, 4], [3, 4], [3, 4], [3, 4], [3, 4]],
    )

    toolkit.servo_eef(
        [0.05, 0.0, 1.9],
        [0.0, 0.0, 0.0, 1.0],
        "open",
        max_steps=1,
    )
    toolkit.observe()

    with pytest.raises(ValueError, match="RGB-D grounding"):
        toolkit.continue_vla(
            "operate the grounded control",
            max_chunks=1,
            step_intent="burner_control",
        )


def test_back_project_rejects_display_only_camera_before_reading_rgbd() -> None:
    observation = _observation()
    observation["state"][6] = 1.0
    observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    source = _PerceptionSource()
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=observation,
        instruction="Turn off the stove.",
        perception_source=source,
    )

    arguments = {
        "camera": "robot0_agentview_center",
        "expected_step": 0,
        "points": [[3, 4]],
    }
    result = toolkit.execute_tool("back_project", arguments).result

    assert "not a planner observation camera" in result["error"]
    with pytest.raises(ValueError, match="not a planner observation camera"):
        toolkit.back_project(**arguments)
    assert source.calls == []
    assert toolkit._precontact_grounding is None
    assert toolkit.evaluation_metrics["action_count"] == 0


def test_back_project_rejects_stale_step_before_reading_rgbd() -> None:
    observation = _observation()
    observation["state"][6] = 1.0
    source = _PerceptionSource(step_id=0)
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=observation,
        instruction="Boil water in the kettle.",
        perception_source=source,
    )

    result = toolkit.execute_tool(
        "back_project",
        {
            "camera": "robot0_agentview_left",
            "expected_step": 1,
            "points": [[3, 4]],
        },
    ).result

    assert "current observation step" in result["error"]
    assert source.calls == []
    assert toolkit.evaluation_metrics["action_count"] == 0


def test_back_project_fails_closed_without_a_perception_source() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )

    result = toolkit.execute_tool(
        "back_project",
        {
            "camera": "robot0_agentview_left",
            "expected_step": 0,
            "points": [[1, 1]],
        },
    ).result

    assert "perception source is not configured" in result["error"]
    assert toolkit.evaluation_metrics["action_count"] == 0


def test_back_project_schema_bounds_cameras_and_pixel_batch() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )

    spec = next(
        item for item in toolkit.get_tools_spec() if item["name"] == "back_project"
    )
    properties = spec["input_schema"]["properties"]

    assert properties["camera"]["enum"] == [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    assert properties["points"]["minItems"] == 1
    assert properties["points"]["maxItems"] == 32
    assert properties["points"]["items"]["minItems"] == 2
    assert properties["points"]["items"]["maxItems"] == 2


def test_servo_eef_uses_bounded_pose_feedback_and_open_gripper_command() -> None:
    start = _observation()
    start["state"][7:10] = (0.0, 0.0, 0.0)
    start["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    halfway = _observation()
    halfway["state"][7:10] = (0.025, -0.025, 0.01)
    halfway["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    halfway["state"][14:16] = (0.04, -0.04)
    target = _observation()
    target["state"][7:10] = (0.05, -0.05, 0.02)
    target["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    target["state"][14:16] = (0.04, -0.04)
    controller = _ServoController([halfway, target])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=start,
        instruction="Boil water in the kettle.",
    )

    result = toolkit.servo_eef(
        target_position=[0.05, -0.05, 0.02],
        target_quaternion=[0.0, 0.0, 0.0, 1.0],
        gripper="open",
        max_steps=5,
        settle_steps=1,
    )

    assert result["target_reached"] is True
    assert result["executed_actions"] == 2
    first = controller.actions[0][0]
    assert first.dtype == np.float32
    assert first.shape == (12,)
    assert first[4] == -1.0
    assert np.allclose(first[5:8], [1.0, -1.0, 0.4])
    assert np.allclose(first[8:11], 0.0)
    assert first[11] == -1.0


def test_servo_eef_settles_closed_gripper_at_an_already_reached_pose() -> None:
    observation = _observation()
    observation["state"][7:10] = (0.2, -0.1, 0.3)
    observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    controller = _ServoController([_observation(), _observation(), _observation()])
    for queued in controller.observations:
        queued["state"][7:10] = observation["state"][7:10]
        queued["state"][10:14] = observation["state"][10:14]
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=observation,
        instruction="Boil water in the kettle.",
    )
    servo = {
        "target_position": [0.2, -0.1, 0.3],
        "target_quaternion": [0.0, 0.0, 0.0, 1.0],
        "gripper": "close",
        "max_steps": 3,
        "settle_steps": 3,
    }
    # 1.0.11 A1: a direct close from an open gripper needs a pose review.
    with pytest.raises(ValueError, match="review_grasp"):
        toolkit.servo_eef(**servo)
    toolkit.observe()
    toolkit.review_grasp(
        expected_step=0, object_between_fingers=True, narrow_axis_aligned=True,
        evidence="Kettle handle between both fingers.",
    )

    result = toolkit.servo_eef(**servo)

    assert result["target_reached"] is True
    assert result["executed_actions"] == 3
    assert all(action[0][11] == 1.0 for action in controller.actions)


def test_servo_eef_does_not_finish_open_until_gripper_is_open() -> None:
    observation = _observation()
    observation["state"][7:10] = (0.2, -0.1, 0.3)
    observation["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    queued = []
    for opening in (0.006, 0.02, 0.04, 0.07, 0.079):
        item = _observation()
        item["state"][7:10] = observation["state"][7:10]
        item["state"][10:14] = observation["state"][10:14]
        item["state"][14:16] = (opening / 2.0, -opening / 2.0)
        queued.append(item)
    controller = _ServoController(queued)
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=observation,
        instruction="Boil water in the kettle.",
    )

    result = toolkit.servo_eef(
        target_position=[0.2, -0.1, 0.3],
        target_quaternion=[0.0, 0.0, 0.0, 1.0],
        gripper="open",
        max_steps=5,
        settle_steps=2,
    )

    assert result["target_reached"] is True
    assert result["executed_actions"] == 5
    assert result["final_gripper_opening"] > 0.07
    assert all(action[0][11] == -1.0 for action in controller.actions)


def test_servo_eef_bounds_quaternion_error_as_axis_angle_action() -> None:
    start = _observation()
    start["state"][10:14] = (0.0, 0.0, 0.0, 1.0)
    target = _observation()
    half_angle = np.pi / 4.0
    target["state"][10:14] = (0.0, 0.0, np.sin(half_angle), np.cos(half_angle))
    target["state"][14:16] = (0.04, -0.04)
    controller = _ServoController([target])
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=start,
        instruction="Boil water in the kettle.",
    )

    result = toolkit.servo_eef(
        target_position=[0.0, 0.0, 0.0],
        target_quaternion=[0.0, 0.0, np.sin(half_angle), np.cos(half_angle)],
        gripper="open",
        max_steps=1,
    )

    assert result["target_reached"] is True
    assert np.allclose(controller.actions[0][0][8:11], [0.0, 0.0, 1.0])


def test_continue_vla_runs_stable_guidance_until_official_success() -> None:
    controller = _ScriptedController(
        [
            _chunk(1, z=0.5, gripper_qpos=(0.02, -0.02)),
            _chunk(
                26,
                z=0.5,
                gripper_qpos=(0.02, -0.02),
                success=True,
                terminated=True,
            ),
            _chunk(51, z=0.5, gripper_qpos=(0.02, -0.02)),
        ]
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.continue_vla("place the kettle on the burner", max_chunks=3)

    assert result["success"] is True
    assert result["chunks_used"] == 2
    assert controller.instructions == ["KettleBoiling", "KettleBoiling"]


def test_continue_vla_retries_burner_control_from_current_observation() -> None:
    anchor_position = (0.4, 0.2, 0.4)
    open_gripper = (0.04, -0.04)
    controller = _ScriptedServoController(
        [
            _pose_chunk(
                1 + chunk_index * 25,
                position=(
                    0.1,
                    0.3,
                    anchor_position[2] - min(chunk_index + 1, 6) * 0.01,
                ),
                gripper_qpos=open_gripper,
                actions=25,
            )
            for chunk_index in range(20)
        ],
        servo_observations=[
            _pose_observation(
                position=anchor_position,
                gripper_qpos=open_gripper,
            )
        ],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=anchor_position,
            gripper_qpos=open_gripper,
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
        max_transitions_per_tool_call=250,
    )

    first = toolkit.continue_vla(
        "turn on the burner matching the observed kettle placement",
        max_chunks=10,
        step_intent="burner_control",
    )
    retry = toolkit.continue_vla(
        "turn on the same burner control",
        max_chunks=10,
        step_intent="burner_control",
    )

    assert first["instruction"] == "KettleBoiling"
    assert retry["instruction"] == "KettleBoiling"
    assert controller.instructions == ["KettleBoiling"] * 20
    assert controller.max_actions == [25] * 20
    assert first["anchor_restore"] == {"attempted": False}
    assert retry["anchor_restore"] == {"attempted": False}
    assert retry["executed_actions"] == retry["transitions_used"] == 250
    assert retry["transition_budget"] == 250
    assert retry["policy_transition_budget"] == 5 * CHECKPOINT_CHUNK_SIZE
    assert controller.actions == []


def test_burner_control_restores_only_after_approach_does_not_converge() -> None:
    anchor_position = (0.4, 0.2, 0.4)
    open_gripper = (0.04, -0.04)
    controller = _ScriptedServoController(
        [
            _pose_chunk(
                1 + chunk_index * 25,
                position=(0.1, 0.3, 0.55),
                gripper_qpos=open_gripper,
                actions=25,
            )
            for chunk_index in range(20)
        ],
        servo_observations=[
            _pose_observation(
                position=anchor_position,
                gripper_qpos=open_gripper,
            ),
            _pose_observation(
                position=anchor_position,
                gripper_qpos=open_gripper,
            ),
        ],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=anchor_position,
            gripper_qpos=open_gripper,
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
        max_transitions_per_tool_call=250,
    )

    first = toolkit.continue_vla(
        "turn on the burner matching the observed kettle placement",
        max_chunks=10,
        step_intent="burner_control",
    )
    retry = toolkit.continue_vla(
        "turn on the same burner control",
        max_chunks=10,
        step_intent="burner_control",
    )

    assert first["failure_reason"] == "burner_approach_not_converging"
    assert first["executed_actions"] == first["transitions_used"] == 151
    assert first["anchor_restore"]["attempted"] is True
    assert first["anchor_restore"]["target_reached"] is True
    assert first["anchor_restore"]["executed_actions"] == 1
    assert first["approach_guard"]["policy_actions"] == 150
    assert first["approach_guard"]["max_close_run"] == 0
    assert first["approach_guard"]["eef_descent_m"] < 0.0
    assert retry["failure_reason"] == "burner_approach_not_converging"
    assert retry["anchor_restore"]["attempted"] is True
    assert retry["anchor_restore"]["target_reached"] is True
    assert retry["anchor_restore"]["executed_actions"] == 1
    assert retry["executed_actions"] == retry["transitions_used"] == 151
    assert retry["policy_transition_budget"] == 5 * CHECKPOINT_CHUNK_SIZE
    assert controller.instructions == ["KettleBoiling"] * 12
    assert controller.max_actions == [25] * 12
    assert len(controller.actions) == 2
    assert all(action[0][11] == pytest.approx(-1.0) for action in controller.actions)


def test_burner_control_keeps_approach_that_descends_after_an_initial_lift() -> None:
    anchor_position = (0.4, 0.2, 0.4)
    open_gripper = (0.04, -0.04)
    chunk_end_z = (0.4, 0.41, 0.42, 0.43, 0.44) + (0.405,) * 5
    controller = _ScriptedServoController(
        [
            _pose_chunk(
                1 + chunk_index * 25,
                position=(0.4, 0.2, z),
                gripper_qpos=open_gripper,
                actions=25,
            )
            for chunk_index, z in enumerate(chunk_end_z)
        ],
        servo_observations=[
            _pose_observation(
                position=anchor_position,
                gripper_qpos=open_gripper,
            )
        ],
    )
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=anchor_position,
            gripper_qpos=open_gripper,
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
        max_transitions_per_tool_call=250,
    )

    result = toolkit.continue_vla(
        "turn on the grounded burner knob",
        max_chunks=10,
        step_intent="burner_control",
    )

    assert result["failure_reason"] == "budget_exhausted"
    assert result["executed_actions"] == 250
    assert result["anchor_restore"] == {"attempted": False}
    assert result["approach_guard"]["eef_descent_m"] < 0.0
    assert result["approach_guard"]["peak_descent_m"] >= 0.03
    assert controller.actions == []


def test_burner_control_keeps_approach_with_sustained_close_intent() -> None:
    open_gripper = (0.04, -0.04)
    chunks = []
    for chunk_index in range(10):
        close = chunk_index == 5
        chunks.append(
            _trajectory_chunk(
                1 + chunk_index * 25,
                [(0.55, open_gripper)] * 25,
                gripper_actions=(1.0 if close else -1.0,) * 25,
            )
        )
    controller = _ScriptedController(chunks)
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=(0.4, 0.2, 0.4),
            gripper_qpos=open_gripper,
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
        max_transitions_per_tool_call=250,
    )

    result = toolkit.continue_vla(
        "turn on the grounded burner knob",
        max_chunks=10,
        step_intent="burner_control",
    )

    assert result["failure_reason"] == "budget_exhausted"
    assert result["executed_actions"] == 250
    assert result["anchor_restore"] == {"attempted": False}
    assert result["approach_guard"]["max_close_run"] == 25


def test_burner_control_rejects_budget_that_cannot_drain_checkpoint_chunk() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=(0.4, 0.2, 0.4),
            gripper_qpos=(0.04, -0.04),
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    with pytest.raises(ValueError, match="drain one 50-action checkpoint chunk"):
        toolkit.continue_vla(
            "turn on the grounded burner knob",
            max_chunks=1,
            step_intent="burner_control",
        )

    assert controller.instructions == []


def test_continue_vla_rejects_burner_control_for_other_checkpoint_tasks() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Close the fridge door.",
        policy_instruction="CloseFridge",
    )

    with pytest.raises(
        ValueError,
        match="does not support step_intent='burner_control'",
    ):
        toolkit.continue_vla(
            "turn on the grounded burner knob",
            step_intent="burner_control",
        )

    assert controller.instructions == []


def test_continue_vla_requires_open_gripper_before_burner_control() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=(0.4, 0.2, 0.4),
            gripper_qpos=(0.02, -0.02),
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
    )

    with pytest.raises(ValueError, match="requires an open gripper"):
        toolkit.continue_vla(
            "turn on the grounded burner knob",
            step_intent="burner_control",
        )

    assert controller.instructions == []


def test_continue_vla_allows_long_open_actuator_approach_before_success() -> None:
    chunks = [
        _trajectory_chunk(
            1 + chunk_index * 25,
            [(0.5, (0.04, -0.04))] * 25,
            success=chunk_index == 5,
            gripper_actions=(
                (-1.0,) * 24 + (1.0,)
                if chunk_index == 5
                else (-1.0,) * 25
            ),
        )
        for chunk_index in range(6)
    ]
    controller = _ScriptedController(chunks)
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_pose_observation(
            position=(0.4, 0.2, 0.4),
            gripper_qpos=(0.04, -0.04),
        ),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
    )

    result = toolkit.continue_vla(
        "turn on the grounded burner knob",
        max_chunks=6,
        step_intent="burner_control",
    )

    assert result["success"] is True
    assert result["failure_reason"] is None
    assert result["stop_reason"] == "success"
    assert result["chunks_used"] == 6
    assert result["transitions_used"] == 150
    assert controller.instructions == ["KettleBoiling"] * 6


def test_continue_vla_enforces_single_call_transition_budget() -> None:
    controller = _BudgetController()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
        actions_per_policy_request=25,
        max_transitions_per_tool_call=60,
    )

    result = toolkit.continue_vla("place the kettle on the burner", max_chunks=10)

    assert controller.max_actions == [25, 25, 10]
    assert result["chunks_used"] == 3
    assert result["transitions_used"] == result["executed_actions"] == 60
    assert result["transition_budget"] == 60
    assert result["stop_reason"] == "transition_budget_exhausted"


def test_execute_vla_schema_explains_configured_action_horizon() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )
    spec = next(
        item for item in toolkit.get_tools_spec() if item["name"] == "execute_vla"
    )

    assert "one policy action chunk" in spec["description"]
    max_actions = spec["input_schema"]["properties"]["max_actions"]
    assert "low-level actions" in max_actions["description"]
    assert "configured execution horizon" in max_actions["description"]
    assert "instruction" not in spec["input_schema"]["properties"]


def test_repeated_vla_tools_use_bounded_configurable_max_chunks_schema() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )
    specs = {item["name"]: item for item in toolkit.get_tools_spec()}

    pick_max_chunks = specs["smolvla_pick"]["input_schema"]["properties"][
        "max_chunks"
    ]
    assert pick_max_chunks["default"] == 10
    assert pick_max_chunks["minimum"] == 10
    assert pick_max_chunks["maximum"] == 12
    assert "omit" in pick_max_chunks["description"].lower()
    assert "grasp attempt" in pick_max_chunks["description"]

    continue_max_chunks = specs["continue_vla"]["input_schema"]["properties"][
        "max_chunks"
    ]
    assert continue_max_chunks["default"] == 10
    assert continue_max_chunks["minimum"] == 1
    assert continue_max_chunks["maximum"] == 12
    assert (
        "policy_task_id"
        not in specs["continue_vla"]["input_schema"]["properties"]
    )
    assert "policy_phase" not in specs["continue_vla"]["input_schema"][
        "properties"
    ]
    assert "guide_policy" not in specs["continue_vla"]["input_schema"][
        "properties"
    ]
    step_intent = specs["continue_vla"]["input_schema"]["properties"][
        "step_intent"
    ]
    assert step_intent["enum"] == ["task_progress"]
    assert step_intent["default"] == "task_progress"
    assert "policy instruction" in step_intent["description"]
    step_instruction = specs["continue_vla"]["input_schema"]["properties"][
        "instruction"
    ]
    assert "audit-only" in step_instruction["description"]
    assert "never forwarded" in step_instruction["description"]
    assert "checkpoint task" in specs["continue_vla"]["description"]

    with pytest.raises(ValueError, match="between 1 and 12"):
        toolkit.continue_vla("continue", max_chunks=13)


def test_kettle_checkpoint_schema_declares_burner_control_capability() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Boil water in the kettle.",
        policy_instruction="KettleBoiling",
    )
    spec = next(
        item for item in toolkit.get_tools_spec() if item["name"] == "continue_vla"
    )

    assert spec["input_schema"]["properties"]["step_intent"]["enum"] == [
        "task_progress",
        "burner_control",
    ]


def test_execute_vla_caps_each_request_at_configured_action_horizon() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_observation(),
        instruction="Close the fridge door.",
        actions_per_policy_request=25,
    )

    toolkit.execute_tool("execute_vla", {})
    toolkit.execute_tool("execute_vla", {"max_actions": 10})
    toolkit.execute_tool("execute_vla", {"max_actions": 40})

    assert controller.max_actions == [25, 10, 25]


def test_observe_keeps_three_planner_cameras_and_tags_display_overview() -> None:
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Close the fridge door.",
    )

    result = toolkit.execute_tool("observe", {}).result

    assert result["state_shape"] == [16]
    assert result["image_keys"] == [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    assert result["task_language"] == "Close the fridge door."
    for key in ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes"):
        assert Image.open(io.BytesIO(result[key])).size == (4, 4)
    assert Image.open(io.BytesIO(result["_image_center_bytes"])).size == (8, 6)
    assert result["_image_camera_ids"] == {
        "robot0_agentview_center": "_image_center_bytes",
        "robot0_agentview_left": "_image_bytes",
        "robot0_agentview_right": "_image_right_bytes",
        "robot0_eye_in_hand": "_image_wrist_bytes",
    }
    assert toolkit.evaluation_metrics["action_count"] == 0


@pytest.mark.parametrize("policy_id", ("raw-tool-result-v1", "episode-compact-v1"))
def test_observe_image_labels_match_localization_schema_without_overview(policy_id):
    import base64

    from rpent.context import resolve_context_policy

    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation=_observation(),
        instruction="Turn off the stove.",
    )
    toolkit.set_context_policy(resolve_context_policy(policy_id))
    observed = toolkit.execute_tool("observe", {})
    spec = next(
        item for item in toolkit.get_tools_spec() if item["name"] == "back_project"
    )
    cameras = observed.result["image_keys"]
    assert spec["input_schema"]["properties"]["camera"]["enum"] == cameras
    assert len(observed.projection.images) == len(cameras) == 3
    assert len(observed.content_blocks) == 7
    assert "robot0_agentview_center" not in observed.projection.text
    for index, camera in enumerate(cameras):
        label, image = observed.content_blocks[1 + 2 * index : 3 + 2 * index]
        assert label == {
            "type": "text", "text": f"Image {index + 1} camera: {camera}",
        }
        assert image["type"] == "image"
        field = observed.result["_image_camera_ids"][camera]
        assert base64.b64decode(image["source"]["data"]) == observed.result[field]


def test_episode_budget_is_distinct_from_per_call_budget_and_survives_projection():
    from rpent.context import EpisodeCompactContextV1

    toolkit = RoboCasaToolkit(
        _BudgetController(), initial_observation=_observation(),
        instruction="Close the drawer.", actions_per_policy_request=5,
        episode_horizon=450,
    )
    toolkit.set_context_policy(EpisodeCompactContextV1())
    first = json.loads(toolkit.execute_tool("observe", {}).projection.text)
    assert first["episode_budget"] == {
        "scope": "episode", "horizon": 450, "steps_used": 0,
        "steps_remaining": 450, "exhausted": False,
    }
    for _ in range(5):
        action = toolkit.execute_tool("continue_vla", {"instruction": "close"})
    assert action.result["transition_budget"] == 250
    assert action.result["transition_budget_scope"] == "tool_call"
    assert action.result["episode_budget"]["steps_used"] == 250
    assert action.result["episode_budget"]["steps_remaining"] == 200
    assert action.result["episode_budget"]["exhausted"] is False
    observed = json.loads(toolkit.execute_tool("observe", {}).projection.text)
    assert observed["episode_budget"] == action.result["episode_budget"]
    for _ in range(4):
        action = toolkit.execute_tool("continue_vla", {"instruction": "close"})
    assert action.result["episode_budget"]["steps_remaining"] == 0
    assert action.result["episode_budget"]["exhausted"] is True


def test_observe_retains_authoritative_environment_terminal_signals():
    terminal = StepResult(
        episode_id="episode-0001", step_id=450, observation=_observation(),
        reward=0.0, success=False, terminated=False, truncated=True,
    )
    toolkit = RoboCasaToolkit(
        _ScriptedController([ChunkExecutionResult((terminal,), 1)]),
        initial_observation=_observation(), instruction="Close the drawer.",
        episode_horizon=450,
    )
    toolkit.execute_tool("execute_vla", {})
    observed = toolkit.execute_tool("observe", {}).result
    assert observed["environment"]["truncated"] is True
    assert observed["environment"]["success"] is False
    assert observed["episode_budget"]["exhausted"] is True


def test_policy_uses_training_task_id_without_hiding_live_instruction() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_observation(),
        instruction="Close the fridge door.",
        policy_instruction="CloseFridge",
    )

    assert toolkit.execute_tool("observe", {}).result["task_language"] == (
        "Close the fridge door."
    )
    toolkit.execute_tool("execute_vla", {})

    assert controller.instructions == ["CloseFridge"]


def test_execute_vla_rejects_free_text_policy_instruction_override() -> None:
    controller = _Controller()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_observation(),
        instruction="Complete the live manipulation task.",
        policy_instruction="CanonicalTaskId",
        actions_per_policy_request=25,
    )

    rejected = toolkit.execute_tool(
        "execute_vla",
        {
            "instruction": "Grasp the visible object by its handle.",
            "max_actions": 1,
        },
    )
    toolkit.execute_tool("execute_vla", {})

    assert "bad arguments for execute_vla" in rejected.result["error"]
    assert controller.instructions == ["CanonicalTaskId"]
    assert controller.max_actions == [25]


def test_prompt_is_simulator_and_contract_authoritative_without_libero_terms() -> None:
    from robots.robocasa.prompts.system import system_prompt

    text = " ".join(system_prompt().values())
    assert "environment.success" in text
    assert "contract_success=true" in text
    assert "reward" in text
    assert "reset" in text
    assert "libero" not in text.lower()
    assert "profile" not in text.lower()


def test_prompt_says_each_execute_vla_call_runs_a_policy_chunk() -> None:
    from robots.robocasa.prompts.system import system_prompt

    action = system_prompt()["ACTION"]
    assert "one tool call executes one policy action chunk" in action
    assert "configured execution horizon" in action
    assert "cap low-level actions" in action
    assert "new registered long-task operation" in action
    assert "coordinates" in action
    assert "fixed low-level action sequence" in action
    assert "environment.success=false" in action
    assert "fresh observe" in action
    assert "actuator state change" in action
    assert "placement invalidation" in action
    assert "one at a time" not in action


def test_prompt_requires_closed_loop_tools_for_grasp_and_continuous_motion() -> None:
    from robots.robocasa.prompts.system import system_prompt

    action = system_prompt()["ACTION"]

    assert "smolvla_pick" in action
    assert "movable-object acquisition grasp" in action
    assert "continue_vla" in action
    assert "servo_eef" in action
    assert "non-privileged trajectory evidence" in action
    assert "Do not call verify_step(pass)" in action
    assert "retryable failure" in action


def test_prompt_keeps_grounded_burner_control_on_canonical_checkpoint_task() -> None:
    from robots.robocasa.prompts.system import task_rules

    action = task_rules("KettleBoiling", policy_family="smolvla")

    assert "specific burner" in action
    assert "policy_phase" not in action
    assert "physically supporting the small gray kettle" in action
    assert "avoid directional burner labels" in action.lower()
    assert "TurnOnStove" not in action
    assert 'step_intent="burner_control"' in action
    assert "canonical KettleBoiling policy instruction unchanged" in action
    assert "instruction is audit-only" in action
    assert "guide_policy" not in action
    assert "policy_task_id" not in action
    assert "knob matching the observed burner" in action
    assert "Do not assume a fixed burner side" in action
    assert "same grounded knob across retries" in action
    assert "transport, placement, and release" in action
    assert "one combined physical plan step" in action
    assert "do not split them into separate physical plan steps" in action
    assert "already shows the kettle stably supported and released" in action
    assert "does not require lateral gripper withdrawal" in action
    assert "an open gripper remaining adjacent to or occluding the kettle" in action
    assert "is not by itself evidence of continued contact or load" in action
    assert "positive evidence of gripper contact, load, or unstable support" in action
    assert "non-physical observe and verify_step calls" in action
    assert "must not become a separate physical plan step" in action
    assert "Never call continue_vla only to reconfirm placement or ground a knob" in (
        action
    )
    assert "blind burner-control rollout" in action
    assert 'leave step_intent="task_progress"' in action
    assert "actuator_no_progress" not in action
    assert "do not repeat the same burner-control call unchanged" not in action
    assert "hold the gripper closed" not in action


def test_prompt_requires_step_aligned_multi_pixel_knob_localization() -> None:
    from robots.robocasa.prompts.system import task_rules

    action = task_rules("KettleBoiling", policy_family="smolvla")

    assert "back_project" in action
    assert "at least five interior pixels" in action
    assert "same observe step" in action
    assert "relative_base_xyz" in action
    assert "measured surface point" in action
    assert "not a direct collision target" in action
    assert "current observed quaternion" in action
    assert "fixed task or seed coordinate" in action
    assert "must include both servo_eef and continue_vla" in action
    assert "must call servo_eef" in action
    assert "recommended_standoff_xyz" in action
    assert "recommended_standoff_quaternion" in action
    assert "Before the first burner_control call" in action
    assert "fresh wrist view" in action


def test_prompt_requires_target_identity_continuity_across_physical_steps() -> None:
    from robots.robocasa.prompts.system import system_prompt

    prompt = system_prompt()

    assert "initial location" in prompt["OBSERVATION"]
    assert "Track the same target across" in prompt["OBSERVATION"]
    assert "observations and returned textual feedback" in prompt["OBSERVATION"]
    assert "last reliable visual target identity and location" in prompt["EVIDENCE"]
    assert "pre-existing object at the destination" in prompt["OBSERVATION"]
    assert "source location" in prompt["OBSERVATION"]
    assert "cannot be the task target" in prompt["OBSERVATION"]
    assert "already at the destination" in prompt["OBSERVATION"]
    assert "Do not merge adjacent source and destination regions" in (
        prompt["OBSERVATION"]
    )
    assert "support surface" in prompt["OBSERVATION"]
    assert "overlaps the destination surface" in prompt["OBSERVATION"]
    assert "When visually similar objects are present" in prompt["ACTION"]
    assert "target identity" in prompt["ACTION"]
    assert "distinguish it from those distractors" in prompt["ACTION"]


def test_prompt_treats_verified_vertical_lift_as_source_departure() -> None:
    from robots.robocasa.prompts.system import system_prompt, task_rules

    observation = system_prompt()["OBSERVATION"]
    action = task_rules("KettleBoiling", policy_family="pi05")

    assert "does not require lateral displacement" in observation
    assert "absence from the source image region" in observation
    assert (
        "In that same matching KettleBoiling scene, when smolvla_pick returns "
        "stop_reason=recovery_close_persisted"
    ) in action
    assert "descent_confirmed=true" in action
    assert "closure_confirmed=true" in action
    assert "non-empty gripper opening" in action
    assert "close and lift target_reached" in action
    assert "post-descent vertical ascent" in action
    assert "same target retained under the gripper" in action
    assert "distractor remains separate" in action
    assert "empty gripper" in action
    assert "held distractor" in action
    assert "target separated from the gripper" in action
    assert "independently supported by the source surface" in action


def test_prompt_uses_observed_grasp_recovery_pose_before_horizon_is_exhausted() -> None:
    from robots.robocasa.prompts.system import system_prompt

    prompt = system_prompt()

    assert "recovery_pose" in prompt["ACTION"]
    assert "reserve horizon" in prompt["ACTION"]
    assert "servo_eef" in prompt["ACTION"]
    assert "descent_confirmed=true" in prompt["ACTION"]
    assert "closure_confirmed=true" in prompt["ACTION"]
    assert "retain its recovery_pose before calling observe" in prompt["ACTION"]
    assert "does not erase" in prompt["ACTION"]
    assert "failure_type='recovery_exhausted'" in prompt["ACTION"]


def test_prompt_pins_the_evidence_backed_kettle_recovery_guidance() -> None:
    from robots.robocasa.prompts.system import task_rules

    action = task_rules("KettleBoiling", policy_family="pi05")
    pinned_guidance = (
        "Recovery grasp: acquire and lift the small gray kettle still on the "
        "right countertop directly in front of the toaster-like appliance; "
        "exclude the larger handled pot on the stove."
    )
    matching_condition = (
        "Only when the live KettleBoiling scene has a small gray kettle on the "
        "right countertop in front of a toaster-like appliance and a larger "
        "handled pot already on the stove"
    )

    assert pinned_guidance in action
    assert "use that sentence verbatim" in action
    assert matching_condition in action
    assert action.index(matching_condition) < action.index(pinned_guidance)
    assert "Otherwise, do not reuse this sentence" in action


def test_prompt_honors_runtime_retry_guard_without_fixed_chunk_count() -> None:
    from robots.robocasa.prompts.system import system_prompt

    action = system_prompt()["ACTION"]

    assert "bounded configurable chunk budget" in action
    assert "retry_allowed=false" in action
    assert "at most one recovery" in action
    assert "Omit max_chunks from every smolvla_pick call" in action
    assert "runtime default can use the full bounded acquisition budget" in action
    assert "max_chunks=60" not in action
    assert "max_chunks=" not in action


def test_rendered_prompt_injects_hooked_gated_harness_protocol() -> None:
    from robots.robocasa import get_env_spec
    from robots.robocasa.prompts.system import task_rules
    from rpent.planning import resolve_planning_stack

    stack = resolve_planning_stack("libero-hooked-gated-v3")
    text = get_env_spec().prompts.render(
        "system",
        variables={
            "harness_protocol": stack.prompt_context(),
            "task_rules": task_rules("CloseDrawer", policy_family="smolvla"),
            "policy_skill": "",
        },
    )

    assert "PLANNING PROFILE: libero-hooked-gated-v3" in text
    assert "planning-loop-v3" in text
    assert "execute_vla" in text
    assert "{harness_protocol}" not in text
