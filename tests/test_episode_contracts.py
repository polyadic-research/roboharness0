from __future__ import annotations

import inspect
from typing import get_type_hints

import numpy as np
import pytest

from rpent.episode.contracts import ChunkExecutionResult, SimulatorSession, StepResult


def result(**overrides: object) -> StepResult:
    values = {
        "episode_id": "episode-7",
        "step_id": 3,
        "observation": {"state": np.zeros(8, dtype=np.float32)},
        "reward": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
    }
    values.update(overrides)
    return StepResult(**values)


def test_simulator_session_exposes_read_only_episode_id() -> None:
    episode_id = inspect.getattr_static(SimulatorSession, "episode_id")

    assert isinstance(episode_id, property)
    assert episode_id.fset is None
    assert get_type_hints(episode_id.fget)["return"] is str


@pytest.mark.parametrize("field", ["success", "terminated", "truncated"])
def test_step_result_is_terminal_for_each_terminal_cause(field: str) -> None:
    assert result(**{field: True}).terminal is True


def test_step_result_is_not_terminal_while_episode_can_continue() -> None:
    assert result().terminal is False


def test_step_result_rejects_negative_step_id() -> None:
    with pytest.raises(ValueError, match="step_id"):
        result(step_id=-1)


def test_chunk_execution_result_reports_executed_and_cancelled_actions() -> None:
    first = result(step_id=0)
    final = result(step_id=1, success=True)

    chunk = ChunkExecutionResult(
        transitions=(first, final),
        requested_actions=4,
    )

    assert chunk.executed_actions == 2
    assert chunk.cancelled_actions == 2
    assert chunk.final is final


def test_paused_suffix_is_distinct_from_terminal_cancellation() -> None:
    chunk = ChunkExecutionResult((result(),), 4, paused_actions=3, resumed_actions=1)
    assert chunk.cancelled_actions == 0 and chunk.executed_actions == 1
    with pytest.raises(ValueError, match="paused_actions"):
        ChunkExecutionResult((result(),), 4, paused_actions=4)
    with pytest.raises(ValueError, match="resumed_actions"):
        ChunkExecutionResult((result(),), 4, resumed_actions=2)


@pytest.mark.parametrize(
    ("transitions", "requested_actions"),
    [
        ((), -1),
        ((result(),), 0),
    ],
)
def test_chunk_execution_result_rejects_invalid_requested_actions(
    transitions: tuple[StepResult, ...], requested_actions: int
) -> None:
    with pytest.raises(ValueError, match="requested_actions"):
        ChunkExecutionResult(
            transitions=transitions,
            requested_actions=requested_actions,
        )


def test_chunk_execution_result_rejects_actions_without_matching_transitions() -> None:
    with pytest.raises(ValueError, match="actions must match"):
        ChunkExecutionResult(
            transitions=(result(),),
            requested_actions=1,
            actions=(np.zeros(7), np.zeros(7)),
        )
