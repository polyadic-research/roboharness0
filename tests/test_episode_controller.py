from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from rpent.episode import EpisodeExecutionError, EpisodeFaultedError
from rpent.episode.binding import FeatureBinding, FeatureValidationError, PolicyBinding
from rpent.episode.contracts import POLICY_REVIEW_PAUSE_REASON, StepResult
from rpent.episode.controller import EpisodeController, EpisodeTerminalError


class FakePolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.predict_calls: list[tuple[Mapping[str, Any], str, int | None]] = []

    def predict(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> np.ndarray:
        self.predict_calls.append((observation, instruction, max_actions))
        return self.actions


class FakeSimulator:
    def __init__(
        self,
        *,
        success_on_step: int | None = None,
        terminated_on_step: int | None = None,
        truncated_on_step: int | None = None,
        failure_on_step: int | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.success_on_step = success_on_step
        self.terminated_on_step = terminated_on_step
        self.truncated_on_step = truncated_on_step
        self.failure_on_step = failure_on_step
        self.failure = failure
        self.attempted_actions: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    @property
    def step_count(self) -> int:
        return len(self.actions)

    def step(self, action: np.ndarray) -> StepResult:
        self.attempted_actions.append(action)
        attempted_step = len(self.attempted_actions)
        if attempted_step == self.failure_on_step:
            raise self.failure or RuntimeError("simulator step failed")
        self.actions.append(action)
        step_id = self.step_count
        return StepResult(
            episode_id="episode-1",
            step_id=step_id,
            observation={"state": np.full(8, step_id, dtype=np.float32)},
            reward=0.0,
            success=step_id == self.success_on_step,
            terminated=step_id == self.terminated_on_step,
            truncated=step_id == self.truncated_on_step,
        )


def seven_dimensional_binding() -> PolicyBinding:
    return PolicyBinding(
        name="seven-dimensional-test-policy",
        inputs=(
            FeatureBinding(
                native_key="state",
                policy_key="observation.state",
                shape=(8,),
                dtype=np.dtype(np.float32),
            ),
        ),
        action_shape=(7,),
        action_dtype=np.dtype(np.float32),
        action_low=-1.0,
        action_high=1.0,
    )


def paused_controller(*, terminal_at=None):
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[1:, 6] = 1
    actions[:, 0] = [0.1, 0.2, 0.3, 0.4]
    policy = FakePolicy(actions)
    simulator = FakeSimulator(success_on_step=terminal_at)
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())
    approved = [False]
    controller.set_policy_action_guard(
        lambda _obs, action: POLICY_REVIEW_PAUSE_REASON
        if action[6] > 0.5 and not approved[0] else None
    )
    first = controller.execute_policy_chunk({"state": np.zeros(8, dtype=np.float32)}, "move")
    assert first.executed_actions == 1
    return controller, simulator, policy, approved, first, actions


def test_paused_suffix_respects_smaller_call_limits_without_new_predictions():
    controller, simulator, policy, approved, first, actions = paused_controller()
    approved[0] = True
    assert first.paused_actions == 3 and first.cancelled_actions == 0
    observed = first.final.observation
    for _ in range(3):
        resumed = controller.execute_policy_chunk(observed, "move", max_actions=1)
        assert resumed.executed_actions == resumed.resumed_actions == 1
        assert resumed.requested_actions == 1 and resumed.cancelled_actions == 0
        observed = resumed.final.observation
    assert len(policy.predict_calls) == 1
    np.testing.assert_array_equal(simulator.actions, actions)


@pytest.mark.parametrize("change", ["instruction", "state", "direct"])
def test_changed_context_discards_paused_suffix(change):
    controller, _simulator, policy, approved, first, _actions = paused_controller()
    approved[0] = True
    observed = first.final.observation
    instruction = "move"
    if change == "state":
        observed["state"][0] += 0.01
    elif change == "instruction":
        instruction = "new operation"
    else:
        observed = controller.execute_action(np.zeros(7, dtype=np.float32)).observation
    policy.actions = np.zeros((1, 7), dtype=np.float32)
    result = controller.execute_policy_chunk(observed, instruction, max_actions=1)
    assert result.resumed_actions == 0
    assert len(policy.predict_calls) == 2


def test_terminal_during_resume_cancels_the_remaining_suffix():
    controller, simulator, policy, approved, first, _actions = paused_controller(terminal_at=2)
    approved[0] = True
    result = controller.execute_policy_chunk(first.final.observation, "move")
    assert result.executed_actions == result.resumed_actions == 1
    assert result.cancelled_actions == 2 and result.paused_actions == 0
    assert simulator.step_count == 2 and len(policy.predict_calls) == 1
    with pytest.raises(EpisodeTerminalError):
        controller.execute_policy_chunk(result.final.observation, "move")


def test_passes_bound_observation_instruction_and_limit_to_policy() -> None:
    state = np.zeros(8, dtype=np.float32)
    simulator = FakeSimulator()
    policy = FakePolicy(np.zeros((1, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    controller.execute_policy_chunk(
        observation={"state": state},
        instruction="close the refrigerator",
        max_actions=4,
    )

    assert len(policy.predict_calls) == 1
    observation, instruction, max_actions = policy.predict_calls[0]
    assert observation == {"observation.state": state}
    assert instruction == "close the refrigerator"
    assert max_actions == 4


def test_non_terminal_chunk_executes_every_action() -> None:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7) / 21
    simulator = FakeSimulator()
    policy = FakePolicy(actions)
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    result = controller.execute_policy_chunk(
        observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="move",
    )

    assert result.executed_actions == 3
    assert result.cancelled_actions == 0
    assert result.final is not None and not result.final.terminal
    assert simulator.step_count == 3
    np.testing.assert_array_equal(np.stack(simulator.actions), actions)
    np.testing.assert_array_equal(np.stack(result.actions), actions)
    assert result.actions[0] is not actions[0]


def test_success_cancels_unexecuted_chunk_suffix() -> None:
    simulator = FakeSimulator(success_on_step=2)
    policy = FakePolicy(np.zeros((3, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    result = controller.execute_policy_chunk(
        observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="close the refrigerator",
    )

    assert result.executed_actions == 2
    assert result.cancelled_actions == 1
    assert len(result.actions) == 2
    assert result.final is not None and result.final.success
    assert simulator.step_count == 2


@pytest.mark.parametrize("terminal_field", ["terminated", "truncated"])
def test_terminal_transition_cancels_unexecuted_chunk_suffix(
    terminal_field: str,
) -> None:
    terminal_step = {f"{terminal_field}_on_step": 2}
    simulator = FakeSimulator(**terminal_step)
    policy = FakePolicy(np.zeros((3, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    result = controller.execute_policy_chunk(
        observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="move",
    )

    assert result.executed_actions == 2
    assert result.cancelled_actions == 1
    assert result.final is not None
    assert getattr(result.final, terminal_field)
    assert simulator.step_count == 2


def test_acting_after_terminal_fails_before_policy_prediction() -> None:
    simulator = FakeSimulator(success_on_step=1)
    policy = FakePolicy(np.zeros((2, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())
    controller.execute_policy_chunk(
        observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="move",
    )
    calls_before_terminal_attempt = len(policy.predict_calls)

    with pytest.raises(
        EpisodeTerminalError,
        match="cannot act after a terminal transition",
    ):
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="move again",
        )

    assert len(policy.predict_calls) == calls_before_terminal_attempt
    assert simulator.step_count == 1


def test_policy_action_chunk_is_validated_before_simulator_step() -> None:
    simulator = FakeSimulator()
    policy = FakePolicy(np.zeros((2, 6), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    with pytest.raises(FeatureValidationError, match=r"expected \(chunk_length, 7\)"):
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="move",
        )

    assert simulator.step_count == 0


def test_simulator_failure_preserves_progress_and_faults_controller() -> None:
    original = RuntimeError("simulator transport disconnected")
    simulator = FakeSimulator(failure_on_step=2, failure=original)
    policy = FakePolicy(np.zeros((3, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    with pytest.raises(
        EpisodeExecutionError,
        match="simulator failed while executing action index 1",
    ) as raised:
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="move",
        )

    assert raised.value.__cause__ is original
    assert raised.value.confirmed_actions == 1
    assert len(raised.value.confirmed_transitions) == 1
    assert raised.value.confirmed_transitions[0].step_id == 1
    assert raised.value.requested_actions == 3
    assert raised.value.attempted_action_index == 1
    assert simulator.step_count == 1
    assert len(simulator.attempted_actions) == 2

    predict_calls = len(policy.predict_calls)
    attempted_actions = len(simulator.attempted_actions)
    with pytest.raises(
        EpisodeFaultedError,
        match="cannot act after an execution fault",
    ):
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="try again",
        )

    assert len(policy.predict_calls) == predict_calls
    assert len(simulator.attempted_actions) == attempted_actions


@pytest.mark.parametrize("max_actions", [True, False, 0, -1])
def test_invalid_max_actions_is_rejected_before_policy_prediction(
    max_actions: object,
) -> None:
    simulator = FakeSimulator()
    policy = FakePolicy(np.zeros((1, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    with pytest.raises(ValueError, match="max_actions must be a positive integer"):
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="move",
            max_actions=max_actions,  # type: ignore[arg-type]
        )

    assert policy.predict_calls == []
    assert simulator.attempted_actions == []


def test_policy_chunk_over_limit_fails_closed_before_simulator_step() -> None:
    simulator = FakeSimulator()
    policy = FakePolicy(np.zeros((3, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    with pytest.raises(
        EpisodeExecutionError,
        match="policy returned 3 actions, exceeding max_actions=2",
    ) as raised:
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="move",
            max_actions=2,
        )

    assert raised.value.confirmed_transitions == ()
    assert raised.value.confirmed_actions == 0
    assert raised.value.requested_actions == 3
    assert raised.value.attempted_action_index is None
    assert simulator.attempted_actions == []

    predict_calls = len(policy.predict_calls)
    with pytest.raises(EpisodeFaultedError):
        controller.execute_policy_chunk(
            observation={"state": np.zeros(8, dtype=np.float32)},
            instruction="try again",
        )

    assert len(policy.predict_calls) == predict_calls
    assert simulator.attempted_actions == []


@pytest.mark.parametrize(("chunk_length", "max_actions"), [(2, 2), (1, 2)])
def test_policy_chunk_at_or_below_limit_executes(
    chunk_length: int,
    max_actions: int,
) -> None:
    simulator = FakeSimulator()
    policy = FakePolicy(np.zeros((chunk_length, 7), dtype=np.float32))
    controller = EpisodeController(simulator, policy, seven_dimensional_binding())

    result = controller.execute_policy_chunk(
        observation={"state": np.zeros(8, dtype=np.float32)},
        instruction="move",
        max_actions=max_actions,
    )

    assert result.executed_actions == chunk_length
    assert len(simulator.attempted_actions) == chunk_length
