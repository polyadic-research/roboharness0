from __future__ import annotations

from collections.abc import Mapping
from contextvars import copy_context
from typing import Any

import numpy as np
import pytest

from rpent.episode.binding import FeatureBinding, PolicyBinding
from rpent.episode.contracts import StepResult
from rpent.episode.controller import (
    EpisodeController,
    EpisodeExecutionError,
    TransitionObserverError,
)
from rpent.episode.recording import RecorderTransitionObserver
from rpent.tools.toolkit import Toolkit


class FakePolicy:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions

    def predict(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> np.ndarray:
        del observation, instruction, max_actions
        return self.actions


class FakeSimulator:
    def __init__(
        self,
        *,
        terminal_step: int | None = None,
        failure_step: int | None = None,
    ) -> None:
        self.terminal_step = terminal_step
        self.failure_step = failure_step
        self.actions: list[np.ndarray] = []
        self.attempts = 0

    def step(self, action: np.ndarray) -> StepResult:
        self.attempts += 1
        if self.attempts == self.failure_step:
            raise RuntimeError("step failed")
        self.actions.append(action)
        return StepResult(
            episode_id="episode-1",
            step_id=self.attempts,
            observation={"state": np.full(8, self.attempts, dtype=np.float32)},
            reward=0.0,
            success=self.attempts == self.terminal_step,
            terminated=False,
            truncated=False,
        )


class SpyObserver:
    def __init__(self, *, fail_confirmed: bool = False, fail_failed: bool = False):
        self.confirmed_pairs: list[tuple[np.ndarray, StepResult]] = []
        self.failed_attempts: list[tuple[np.ndarray, int, str]] = []
        self.fail_confirmed = fail_confirmed
        self.fail_failed = fail_failed

    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        self.confirmed_pairs.append((action.copy(), result))
        action.fill(99)
        if self.fail_confirmed:
            raise OSError("recorder unavailable")

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        self.failed_attempts.append((action.copy(), action_index, type(error).__name__))
        if self.fail_failed:
            raise OSError("failure journal unavailable")


def binding() -> PolicyBinding:
    return PolicyBinding(
        name="test",
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
        action_low=-100.0,
        action_high=100.0,
    )


def execute(
    simulator: FakeSimulator,
    observer: SpyObserver,
    actions: np.ndarray,
):
    return EpisodeController(
        simulator,
        FakePolicy(actions),
        binding(),
        transition_observer=observer,
    ).execute_policy_chunk(
        {"state": np.zeros(8, dtype=np.float32)},
        "move",
    )


def test_observes_each_confirmed_action_and_skips_terminal_suffix() -> None:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    simulator = FakeSimulator(terminal_step=2)
    observer = SpyObserver()

    result = execute(simulator, observer, actions)

    assert result.executed_actions == 2
    assert [item.step_id for _, item in observer.confirmed_pairs] == [1, 2]
    assert simulator.attempts == 2
    np.testing.assert_array_equal(observer.confirmed_pairs[0][0], actions[0])
    np.testing.assert_array_equal(simulator.actions[0], actions[0])


def test_reports_failed_attempt_without_manufacturing_a_transition() -> None:
    simulator = FakeSimulator(failure_step=2)
    observer = SpyObserver()

    with pytest.raises(EpisodeExecutionError) as raised:
        execute(simulator, observer, np.zeros((3, 7), dtype=np.float32))

    assert len(raised.value.confirmed_transitions) == 1
    assert observer.failed_attempts == [
        (pytest.approx(np.zeros(7, dtype=np.float32)), 1, "RuntimeError")
    ]
    assert [item.step_id for _, item in observer.confirmed_pairs] == [1]


def test_confirmed_observer_failure_preserves_the_physical_transition() -> None:
    observer = SpyObserver(fail_confirmed=True)

    with pytest.raises(TransitionObserverError) as raised:
        execute(FakeSimulator(), observer, np.zeros((2, 7), dtype=np.float32))

    assert [item.step_id for item in raised.value.confirmed_transitions] == [1]
    assert raised.value.attempted_action_index == 0
    assert isinstance(raised.value.__cause__, OSError)


def test_failed_observer_failure_does_not_mask_the_simulator_failure() -> None:
    observer = SpyObserver(fail_failed=True)

    with pytest.raises(EpisodeExecutionError) as raised:
        execute(
            FakeSimulator(failure_step=1),
            observer,
            np.zeros((1, 7), dtype=np.float32),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert isinstance(raised.value.observer_error, OSError)


class FakeCaseRecorder:
    def __init__(self) -> None:
        self.transitions: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.operation_clip_submissions: list[tuple[str, tuple[int, ...]]] = []

    def record_transition(self, **payload: Any) -> None:
        self.transitions.append(payload)

    def record_event(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))

    def submit_operation_clips(
        self, operation_id: str, step_ids: tuple[int, ...]
    ) -> None:
        self.operation_clip_submissions.append((operation_id, step_ids))


def test_recorder_observer_uses_reset_relative_time_and_operation_context() -> None:
    times = iter((9.0, 10.0, 10.25))
    recorder = FakeCaseRecorder()
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: next(times))
    observer.reset()
    transition = StepResult(
        "episode-1", 4, {"state": np.zeros(8)}, 0.0, False, False, False
    )

    with observer.operation("operation-1", "execution-1") as operation:
        observer.confirmed(np.ones(7), transition)

    assert recorder.transitions[0]["timestamp_s"] == pytest.approx(0.25)
    assert recorder.transitions[0]["operation_id"] == "operation-1"
    assert recorder.transitions[0]["execution_id"] == "execution-1"
    assert operation.step_ids == [4]


class CapturingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        self.events.append((event, payload))
        return {"event": event, **payload}


def test_toolkit_records_physical_operation_with_chunk_and_step_references() -> None:
    simulator = FakeSimulator(terminal_step=2)
    recorder = FakeCaseRecorder()
    times = iter((10.0, 10.1, 10.2, 10.3))
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: next(times))
    controller = EpisodeController(
        simulator,
        FakePolicy(np.zeros((3, 7), dtype=np.float32)),
        binding(),
        transition_observer=observer,
    )
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move"})
    toolkit.attach_transition_observer(controller.transition_observer)
    events = CapturingEvents()
    toolkit.attach_event_recorder(events)

    def move() -> dict[str, Any]:
        chunk = controller.execute_policy_chunk(
            {"state": np.zeros(8, dtype=np.float32)}, "move"
        )
        return {
            "requested_actions": chunk.requested_actions,
            "executed_actions": chunk.executed_actions,
            "cancelled_actions": chunk.cancelled_actions,
            "log": {
                "result": {
                    "final_dist_m": 0.31,
                    "steps_used": chunk.executed_actions,
                    "max_steps": chunk.requested_actions,
                }
            },
        }

    toolkit.add_tool("move", {"name": "move", "input_schema": {}}, move)
    result = toolkit.execute_tool("move", {})

    names = [name for name, _ in events.events]
    assert "tool_execution_started" in names
    assert "tool_execution_finished" in names
    assert names.count("operation_started") == 1
    assert names.count("operation_completed") == 1
    started = next(
        payload for name, payload in events.events if name == "operation_started"
    )
    completed = next(
        payload for name, payload in events.events if name == "operation_completed"
    )
    assert started["execution_id"] == result.call_id == completed["execution_id"]
    assert completed["requested_actions"] == 3
    assert completed["executed_actions"] == 2
    assert completed["cancelled_actions"] == 1
    assert completed["first_step_id"] == 1
    assert completed["last_step_id"] == 2
    assert completed["result"] == {
        "final_dist_m": 0.31,
        "steps_used": 2,
        "max_steps": 3,
    }
    assert recorder.operation_clip_submissions == [
        (completed["operation_id"], (1, 2))
    ]


def test_failed_chunk_operation_retains_confirmed_progress() -> None:
    simulator = FakeSimulator(failure_step=2)
    recorder = FakeCaseRecorder()
    times = iter((10.0, 10.1, 10.2))
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: next(times))
    controller = EpisodeController(
        simulator,
        FakePolicy(np.zeros((3, 7), dtype=np.float32)),
        binding(),
        transition_observer=observer,
    )
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move"})
    toolkit.attach_transition_observer(observer)
    events = CapturingEvents()
    toolkit.attach_event_recorder(events)

    def move() -> dict[str, Any]:
        controller.execute_policy_chunk(
            {"state": np.zeros(8, dtype=np.float32)}, "move"
        )
        raise AssertionError("unreachable")

    toolkit.add_tool("move", {"name": "move", "input_schema": {}}, move)
    result = toolkit.execute_tool("move", {})

    failed = next(
        payload for name, payload in events.events if name == "operation_failed"
    )
    assert (
        result.result["error"] == "simulator failed while executing action index 1 of 3"
    )
    assert failed["requested_actions"] == 3
    assert failed["executed_actions"] == 1
    assert failed["cancelled_actions"] == 2
    assert failed["first_step_id"] == failed["last_step_id"] == 1
    assert recorder.operation_clip_submissions == [(failed["operation_id"], (1,))]


def test_harness_block_does_not_start_a_physical_operation() -> None:
    class BlockingHarness:
        def set_physical_tools(self, tools):
            del tools

        def before_tool(self, name, args, *, execution_id):
            del name, args, execution_id
            return {"allowed": False, "reason": "blocked"}

        def snapshot(self):
            return {}

        def tool_definitions(self):
            return ()

    events = CapturingEvents()
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move"})
    toolkit.add_tool("move", {"name": "move", "input_schema": {}}, lambda: {})
    toolkit.attach_harness(BlockingHarness())
    toolkit.attach_event_recorder(events)

    toolkit.execute_tool("move", {})

    assert "operation_started" not in [name for name, _ in events.events]


def test_observer_reinstall_after_action_is_rejected_before_reset() -> None:
    recorder = FakeCaseRecorder()
    times = iter((0.0, 10.0, 11.0, 12.0))
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: next(times))
    controller = EpisodeController(
        FakeSimulator(),
        FakePolicy(np.zeros((1, 7), dtype=np.float32)),
        binding(),
        transition_observer=observer,
    )
    controller.execute_action(np.zeros(7, dtype=np.float32))

    with pytest.raises(RuntimeError, match="after action attempt"):
        controller.set_transition_observer(observer)

    controller.execute_action(np.zeros(7, dtype=np.float32))
    assert [item["timestamp_s"] for item in recorder.transitions] == [1.0, 2.0]


def test_observer_reset_failure_leaves_previous_observer_installed() -> None:
    controller = EpisodeController(
        FakeSimulator(),
        FakePolicy(np.zeros((1, 7), dtype=np.float32)),
        binding(),
    )
    old = controller.transition_observer

    class BrokenObserver(SpyObserver):
        def reset(self) -> None:
            raise OSError("clock unavailable")

    with pytest.raises(OSError, match="clock unavailable"):
        controller.set_transition_observer(BrokenObserver())

    assert controller.transition_observer is old


def test_recorder_operation_contexts_are_isolated_when_interleaved() -> None:
    recorder = FakeCaseRecorder()
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: 1.0)
    context_a = copy_context()
    context_b = copy_context()
    scope_a = observer.operation("operation-a", "execution-a")
    scope_b = observer.operation("operation-b", "execution-b")
    context_a.run(scope_a.__enter__)
    context_b.run(scope_b.__enter__)

    context_a.run(
        observer.confirmed,
        np.zeros(7),
        StepResult("episode-1", 1, {"state": []}, 0.0, False, False, False),
    )
    context_b.run(
        observer.confirmed,
        np.zeros(7),
        StepResult("episode-1", 2, {"state": []}, 0.0, False, False, False),
    )
    context_a.run(scope_a.__exit__, None, None, None)
    context_b.run(scope_b.__exit__, None, None, None)

    assert [item["operation_id"] for item in recorder.transitions] == [
        "operation-a",
        "operation-b",
    ]


def test_nested_operation_context_restores_after_exception() -> None:
    recorder = FakeCaseRecorder()
    observer = RecorderTransitionObserver(recorder, monotonic=lambda: 1.0)

    with observer.operation("outer", "execution-outer"):
        with pytest.raises(RuntimeError, match="stop inner"):
            with observer.operation("inner", "execution-inner"):
                observer.confirmed(
                    np.zeros(7),
                    StepResult("episode-1", 1, {"state": []}, 0.0, False, False, False),
                )
                raise RuntimeError("stop inner")
        observer.confirmed(
            np.zeros(7),
            StepResult("episode-1", 2, {"state": []}, 0.0, False, False, False),
        )

    assert [item["operation_id"] for item in recorder.transitions] == [
        "inner",
        "outer",
    ]


def test_strict_post_handler_failure_keeps_completed_physical_operation(
    tmp_path, monkeypatch
) -> None:
    class Session:
        def __init__(self) -> None:
            class Controller:
                def set_physical_tools(self, tools):
                    self.physical_tools = set(tools)

            self.controller = Controller()
            self.abort_calls: list[tuple[str, str]] = []

        def physical_action_allowed(self, name, execution_id):
            del name
            return {"allowed": True, "execution_id": execution_id}

        def abort_physical_action(self, execution_id, reason, evidence):
            del evidence
            self.abort_calls.append((execution_id, reason))
            return {"ok": True, "state": "recovery_required"}

        def __getattr__(self, name):
            return lambda **kwargs: {"ok": True, "operation": name, **kwargs}

    def fail_normalization(**kwargs):
        del kwargs
        raise RuntimeError("feedback normalization failed")

    monkeypatch.setattr(
        "rpent.planning.feedback.action_feedback_from_result",
        fail_normalization,
    )
    events = CapturingEvents()
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move"})
    toolkit.configure_harness_evidence_tools({"move"})
    toolkit.add_tool(
        "move",
        {"name": "move", "input_schema": {}},
        lambda: {
            "requested_actions": 4,
            "executed_actions": 2,
            "cancelled_actions": 2,
            "environment": {"step_id": 8},
        },
    )
    session = Session()
    toolkit.attach_harness_session(session, tmp_path)
    toolkit.attach_event_recorder(events)

    result = toolkit.execute_tool("move", {})

    operation_events = [
        (name, payload)
        for name, payload in events.events
        if name.startswith("operation_")
    ]
    assert [name for name, _ in operation_events] == [
        "operation_started",
        "operation_completed",
    ]
    completed = operation_events[-1][1]
    assert completed["execution_id"] == result.call_id
    assert completed["requested_actions"] == 4
    assert completed["executed_actions"] == 2
    assert completed["cancelled_actions"] == 2
    assert completed["first_step_id"] == 7
    assert completed["last_step_id"] == 8
    assert "feedback normalization failed" in result.result["error"]
