"""Ownership and cleanup tests for the canonical episode runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import rpent.episode as episode_api
from rpent.episode.contracts import StepResult
from rpent.episode.presets import PresetRegistry, RuntimePreset
from rpent.episode.runtime import EpisodeBootstrap, open_episode


def test_runtime_boundary_is_exported_without_private_preset_types() -> None:
    assert episode_api.EpisodeBootstrap is EpisodeBootstrap
    assert episode_api.open_episode is open_episode
    assert not hasattr(episode_api, "RuntimePreset")


class _Dependency:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"start failed: {self.name}")

    def stop(self) -> None:
        self.events.append(f"stop:{self.name}")
        if self.fail_stop:
            raise RuntimeError(f"stop failed: {self.name}")


class _Simulator:
    def __init__(
        self,
        events: list[str],
        *,
        fail_reset: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.events = events
        self.fail_reset = fail_reset
        self.fail_close = fail_close
        self.reset_calls = 0
        self.step_calls = 0
        self._episode_id = "episode-17"
        self.observation = {"state": np.zeros(1, dtype=np.float32)}

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def reset(self, task: str, seed: int) -> dict[str, Any]:
        self.events.append(f"reset:simulator:{task}:{seed}")
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("simulator reset failed")
        return self.observation

    def observe(self) -> dict[str, Any]:
        return self.observation

    def step(self, action: np.ndarray) -> StepResult:
        del action
        self.step_calls += 1
        return StepResult(
            episode_id=self.episode_id,
            step_id=self.step_calls,
            observation=self.observation,
            reward=1.0,
            success=True,
            terminated=True,
            truncated=False,
        )

    def close(self) -> None:
        self.events.append("close:simulator")
        if self.fail_close:
            raise RuntimeError("simulator close failed")


class _Policy:
    def __init__(
        self,
        events: list[str],
        *,
        fail_reset: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.events = events
        self.fail_reset = fail_reset
        self.fail_close = fail_close

    def reset(self, episode_id: str) -> None:
        self.events.append(f"reset:policy:{episode_id}")
        if self.fail_reset:
            raise RuntimeError("policy reset failed")

    def predict(
        self,
        observation: dict[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> np.ndarray:
        del observation, instruction, max_actions
        return np.zeros((1, 1), dtype=np.float32)

    def close(self) -> None:
        self.events.append("close:policy")
        if self.fail_close:
            raise RuntimeError("policy close failed")


class _Binding:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def bind_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.events.append("validate:binding")
        if self.fail:
            raise ValueError("binding failed")
        return observation

    def validate_action_chunk(self, actions: Any) -> np.ndarray:
        return np.asarray(actions)


class _Toolkit:
    def __init__(
        self,
        events: list[str],
        *,
        fail_close: bool = False,
    ) -> None:
        self.events = events
        self.fail_close = fail_close

    def close(self) -> None:
        self.events.append("close:toolkit")
        if self.fail_close:
            raise RuntimeError("toolkit close failed")


@dataclass
class _RuntimeFakes:
    bootstrap: EpisodeBootstrap
    simulator: _Simulator


def _fakes(
    events: list[str],
    *,
    fail: str | None = None,
    cleanup_failures: frozenset[str] = frozenset(),
    toolkit_has_close: bool = True,
) -> _RuntimeFakes:
    simulator = _Simulator(
        events,
        fail_reset=fail == "simulator-reset",
        fail_close="simulator" in cleanup_failures,
    )
    policy = _Policy(
        events,
        fail_reset=fail == "policy-reset",
        fail_close="policy" in cleanup_failures,
    )
    dependencies = (
        _Dependency(
            "d1",
            events,
            fail_start=fail == "dependency-1",
            fail_stop="d1" in cleanup_failures,
        ),
        _Dependency(
            "d2",
            events,
            fail_start=fail == "dependency-2",
            fail_stop="d2" in cleanup_failures,
        ),
    )

    def simulator_factory() -> _Simulator:
        events.append("construct:simulator")
        if fail == "simulator-factory":
            raise RuntimeError("simulator factory failed")
        return simulator

    def policy_factory() -> _Policy:
        events.append("construct:policy")
        if fail == "policy-factory":
            raise RuntimeError("policy factory failed")
        return policy

    def toolkit_factory(controller: Any, observation: dict[str, Any]) -> Any:
        assert controller is not None
        assert observation is simulator.observation
        events.append("construct:toolkit")
        if fail == "toolkit-factory":
            raise RuntimeError("toolkit factory failed")
        if not toolkit_has_close:
            return object()
        return _Toolkit(events, fail_close="toolkit" in cleanup_failures)

    return _RuntimeFakes(
        bootstrap=EpisodeBootstrap(
            preset_name="test",
            task="suite:3",
            seed=17,
            instruction="do the task",
            simulator_factory=simulator_factory,
            policy_factory=policy_factory,
            binding=_Binding(events, fail=fail == "binding"),  # type: ignore[arg-type]
            toolkit_factory=toolkit_factory,
            dependencies=dependencies,  # type: ignore[arg-type]
        ),
        simulator=simulator,
    )


def test_open_episode_owns_exact_startup_and_reverse_cleanup_order() -> None:
    events: list[str] = []
    fakes = _fakes(events)

    with open_episode(fakes.bootstrap) as runtime:
        events.append("yield")
        assert runtime.episode_id == "episode-17"
        assert runtime.task == "suite:3"
        assert runtime.seed == 17
        assert runtime.observation is fakes.simulator.observation
        assert runtime.simulator is fakes.simulator

    assert events == [
        "start:d1",
        "start:d2",
        "construct:simulator",
        "construct:policy",
        "reset:simulator:suite:3:17",
        "validate:binding",
        "reset:policy:episode-17",
        "construct:toolkit",
        "yield",
        "close:toolkit",
        "close:policy",
        "close:simulator",
        "stop:d2",
        "stop:d1",
    ]
    assert fakes.simulator.reset_calls == 1


def test_open_episode_records_scene_and_policy_reset_durations(
    monkeypatch,
) -> None:
    import rpent.episode.runtime as runtime_module

    events: list[str] = []
    recorded = []
    times = iter((10.0, 12.5, 20.0, 20.25))
    monkeypatch.setattr(
        runtime_module,
        "record_worker_startup_event",
        lambda phase, event, **fields: recorded.append((phase, event, fields))
        or True,
    )

    with open_episode(
        _fakes(events).bootstrap,
        duration_clock=lambda: next(times),
    ):
        pass

    assert recorded == [
        ("scene_initialization", "completed", {"duration_ms": 2500}),
        ("policy_episode_reset", "completed", {"duration_ms": 250}),
    ]


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("dependency-1", ["start:d1", "stop:d1"]),
        (
            "dependency-2",
            ["start:d1", "start:d2", "stop:d2", "stop:d1"],
        ),
        (
            "simulator-factory",
            ["start:d1", "start:d2", "construct:simulator", "stop:d2", "stop:d1"],
        ),
        (
            "policy-factory",
            [
                "start:d1",
                "start:d2",
                "construct:simulator",
                "construct:policy",
                "close:simulator",
                "stop:d2",
                "stop:d1",
            ],
        ),
        (
            "simulator-reset",
            [
                "start:d1",
                "start:d2",
                "construct:simulator",
                "construct:policy",
                "reset:simulator:suite:3:17",
                "close:policy",
                "close:simulator",
                "stop:d2",
                "stop:d1",
            ],
        ),
        (
            "binding",
            [
                "start:d1",
                "start:d2",
                "construct:simulator",
                "construct:policy",
                "reset:simulator:suite:3:17",
                "validate:binding",
                "close:policy",
                "close:simulator",
                "stop:d2",
                "stop:d1",
            ],
        ),
        (
            "policy-reset",
            [
                "start:d1",
                "start:d2",
                "construct:simulator",
                "construct:policy",
                "reset:simulator:suite:3:17",
                "validate:binding",
                "reset:policy:episode-17",
                "close:policy",
                "close:simulator",
                "stop:d2",
                "stop:d1",
            ],
        ),
        (
            "toolkit-factory",
            [
                "start:d1",
                "start:d2",
                "construct:simulator",
                "construct:policy",
                "reset:simulator:suite:3:17",
                "validate:binding",
                "reset:policy:episode-17",
                "construct:toolkit",
                "close:policy",
                "close:simulator",
                "stop:d2",
                "stop:d1",
            ],
        ),
    ],
)
def test_startup_failure_closes_only_acquired_resources(
    stage: str,
    expected: list[str],
) -> None:
    events: list[str] = []

    with pytest.raises((RuntimeError, ValueError)):
        with open_episode(_fakes(events, fail=stage).bootstrap):
            pytest.fail("runtime must not yield")

    assert events == expected


def test_body_error_is_not_masked_and_all_cleanup_continues(caplog) -> None:
    events: list[str] = []
    bootstrap = _fakes(
        events,
        cleanup_failures=frozenset({"toolkit", "policy", "simulator", "d1", "d2"}),
    ).bootstrap

    with pytest.raises(LookupError, match="body failed"):
        with open_episode(bootstrap):
            events.append("yield")
            raise LookupError("body failed")

    assert events[-5:] == [
        "close:toolkit",
        "close:policy",
        "close:simulator",
        "stop:d2",
        "stop:d1",
    ]
    assert "cleanup" in caplog.text.lower()


def test_cleanup_errors_are_suppressed_without_a_body_error(caplog) -> None:
    events: list[str] = []
    bootstrap = _fakes(
        events,
        cleanup_failures=frozenset({"policy", "simulator"}),
    ).bootstrap

    with open_episode(bootstrap):
        events.append("yield")

    assert events[-4:] == ["close:policy", "close:simulator", "stop:d2", "stop:d1"]
    assert "cleanup" in caplog.text.lower()


def test_toolkit_without_close_is_supported() -> None:
    events: list[str] = []

    with open_episode(_fakes(events, toolkit_has_close=False).bootstrap):
        events.append("yield")

    assert "close:toolkit" not in events
    assert events[-4:] == ["close:policy", "close:simulator", "stop:d2", "stop:d1"]


def test_terminal_transition_never_triggers_an_automatic_reset() -> None:
    events: list[str] = []
    fakes = _fakes(events)

    with open_episode(fakes.bootstrap) as runtime:
        result = runtime.controller.execute_policy_chunk(
            runtime.observation,
            fakes.bootstrap.instruction,
        )

    assert result.final is not None and result.final.terminal
    assert fakes.simulator.reset_calls == 1


def test_private_preset_registry_rejects_duplicates_and_unknown_names() -> None:
    registry = PresetRegistry()
    preset = RuntimePreset("test", lambda: object())
    registry.register(preset)

    assert registry.get("test") is preset
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(preset)
    with pytest.raises(KeyError, match="unknown"):
        registry.get("missing")
