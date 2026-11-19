from __future__ import annotations

import numpy as np

from rpent.episode.contracts import ChunkExecutionResult, StepResult
from rpent.tools.vla import VLAOperation


class _Controller:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        self.calls.append((observation, instruction, max_actions))
        return self.result

    def execute_action(self, action, *, _action_index=0):
        self.calls.append((action, _action_index))
        return self.result.transitions[_action_index]


def _step(step_id, *, success=False, terminated=False):
    return StepResult(
        episode_id="episode-0001",
        step_id=step_id,
        observation={"state": np.full(8, step_id, dtype=np.float32)},
        reward=0.0,
        success=success,
        terminated=terminated,
        truncated=False,
    )


def test_execute_vla_serializes_policy_neutral_chunk_result() -> None:
    controller = _Controller(
        ChunkExecutionResult(tuple(_step(index) for index in range(1, 11)), 10)
    )
    operation = VLAOperation(
        controller,
        initial_observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="close the refrigerator",
    )

    result = operation.execute(max_actions=10)

    assert result == {
        "environment": {
            "episode_id": "episode-0001",
            "step_id": 10,
            "success": False,
            "terminated": False,
            "truncated": False,
        },
        "requested_actions": 10,
        "executed_actions": 10,
        "cancelled_actions": 0,
        "paused_actions": 0,
        "resumed_actions": 0,
    }
    assert controller.calls[0][1:] == ("close the refrigerator", 10)
    assert "libero_terminated" not in result


def test_execute_vla_preserves_terminal_suffix_cancellation() -> None:
    controller = _Controller(
        ChunkExecutionResult(
            (_step(1), _step(2), _step(3, success=True, terminated=True)),
            8,
        )
    )
    operation = VLAOperation(controller, initial_observation={}, instruction="task")

    result = operation.execute()

    assert result["environment"]["success"] is True
    assert result["executed_actions"] == 3
    assert result["cancelled_actions"] == 5
    assert operation.observation["state"][0] == 3
    assert operation.metrics == {
        "step_count": 3,
        "action_count": 3,
        "requested_actions": 8,
        "cancelled_actions": 5,
        "paused_actions": 0,
        "resumed_actions": 0,
        "simulator_success": True,
        "terminated": True,
        "truncated": False,
    }


def test_execute_vla_exposes_last_chunk_observations_in_transition_order() -> None:
    actions = tuple(np.full(12, index, dtype=np.float32) for index in (1, 2))
    controller = _Controller(
        ChunkExecutionResult((_step(1), _step(2)), 2, actions=actions)
    )
    operation = VLAOperation(controller, initial_observation={}, instruction="task")

    operation.execute()

    assert [item["state"][0] for item in operation.last_observations] == [1, 2]
    assert [item[11] for item in operation.last_actions] == [1, 2]
    assert operation.last_observations[0] is not controller.result.transitions[0].observation
    assert operation.last_actions[0] is not actions[0]


def test_execute_vla_instruction_override_is_scoped_to_one_call() -> None:
    controller = _Controller(ChunkExecutionResult((_step(1),), 1))
    operation = VLAOperation(
        controller,
        initial_observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="CanonicalTaskId",
    )

    operation.execute(
        instruction="CanonicalTaskId\nCurrent action-step guidance: grasp"
    )
    operation.execute()

    assert [call[1] for call in controller.calls] == [
        "CanonicalTaskId\nCurrent action-step guidance: grasp",
        "CanonicalTaskId",
    ]


def test_execute_actions_updates_observation_metrics_and_terminal_cancellation() -> (
    None
):
    controller = _Controller(
        ChunkExecutionResult(
            (_step(1), _step(2, success=True, terminated=True), _step(3)),
            3,
        )
    )
    operation = VLAOperation(controller, initial_observation={}, instruction="task")
    actions = tuple(np.full(12, index, dtype=np.float32) for index in range(3))

    result = operation.execute_actions(actions)

    assert result["environment"]["success"] is True
    assert result["requested_actions"] == 3
    assert result["executed_actions"] == 2
    assert result["cancelled_actions"] == 1
    assert operation.observation["state"][0] == 2
    assert [call[1] for call in controller.calls] == [0, 1]
    assert [action[0] for action in operation.last_actions] == [0, 1]
