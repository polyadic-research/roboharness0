"""Single ownership boundary for one simulator-policy episode."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rpent.episode.binding import PolicyBinding
from rpent.episode.contracts import PolicySession, SimulatorSession
from rpent.episode.controller import EpisodeController
from rpent.episode.recording import CompositeTransitionObserver
from rpent.evaluation.startup_log import record_worker_startup_event
from rpent.recording.display import DisplayAdapter, NullDisplayAdapter
from rpent.utils.logging import get_logger

logger = get_logger("episode_runtime")


@dataclass(frozen=True, slots=True)
class EpisodeBootstrap:
    """Late-bound factories and immutable identity for one episode."""

    preset_name: str
    task: str
    seed: int
    instruction: str
    simulator_factory: Callable[[], SimulatorSession]
    policy_factory: Callable[[], PolicySession]
    binding: PolicyBinding
    toolkit_factory: Callable[[EpisodeController, Mapping[str, Any]], Any]
    display_adapter: DisplayAdapter = NullDisplayAdapter()
    dependencies: tuple[Any, ...] = ()
    runtime_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for field_name in ("preset_name", "task"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        for field_name in (
            "simulator_factory",
            "policy_factory",
            "toolkit_factory",
        ):
            if not callable(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be callable")
        try:
            dependencies = tuple(self.dependencies)
        except TypeError as exc:
            raise ValueError("dependencies must be iterable") from exc
        object.__setattr__(self, "dependencies", dependencies)


@dataclass(slots=True)
class EpisodeRuntime:
    """Resources made available while :func:`open_episode` is active."""

    episode_id: str
    task: str
    seed: int
    reset_monotonic: float
    observation: dict[str, Any]
    simulator: SimulatorSession
    policy: PolicySession
    controller: EpisodeController
    toolkit: Any

    def set_transition_observer(self, observer: Any) -> None:
        """Attach a reset-relative observer before Harness execution starts."""
        policy_observer = getattr(self.policy, "transition_observer", None)
        if all(
            callable(getattr(policy_observer, method, None))
            for method in ("confirmed", "failed")
        ):
            observer = CompositeTransitionObserver(observer, policy_observer)
        self.controller.set_transition_observer(observer)
        attach = getattr(self.toolkit, "attach_transition_observer", None)
        if callable(attach):
            attach(observer)


@contextmanager
def open_episode(
    bootstrap: EpisodeBootstrap,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    duration_clock: Callable[[], float] = time.monotonic,
) -> Iterator[EpisodeRuntime]:
    """Open one episode and timestamp the boundary immediately before reset."""

    cleanups: list[tuple[str, Callable[[], None]]] = []
    try:
        for dependency in bootstrap.dependencies:
            cleanups.append(
                (
                    f"dependency {getattr(dependency, 'name', type(dependency).__name__)}",
                    dependency.stop,
                )
            )
            dependency.start()
            for phase, duration_ms in getattr(
                dependency, "startup_timings", {}
            ).items():
                record_worker_startup_event(
                    phase, "completed", duration_ms=int(duration_ms)
                )

        simulator = bootstrap.simulator_factory()
        cleanups.append(("simulator", simulator.close))

        policy = bootstrap.policy_factory()
        cleanups.append(("policy", policy.close))

        reset_monotonic = monotonic()
        scene_started = duration_clock()
        observation = simulator.reset(bootstrap.task, bootstrap.seed)
        record_worker_startup_event(
            "scene_initialization",
            "completed",
            duration_ms=_milliseconds(duration_clock() - scene_started),
        )
        policy_observation = bootstrap.binding.bind_observation(observation)
        policy_reset_started = duration_clock()
        policy.reset(simulator.episode_id)
        initialize_observation = getattr(policy, "initialize_observation", None)
        if callable(initialize_observation):
            initialize_observation(policy_observation)
        record_worker_startup_event(
            "policy_episode_reset",
            "completed",
            duration_ms=_milliseconds(duration_clock() - policy_reset_started),
        )

        controller = EpisodeController(simulator, policy, bootstrap.binding)
        toolkit = bootstrap.toolkit_factory(controller, observation)
        attach_observer = getattr(toolkit, "attach_transition_observer", None)
        if callable(attach_observer):
            attach_observer(controller.transition_observer)
        toolkit_close = getattr(toolkit, "close", None)
        if callable(toolkit_close):
            cleanups.append(("toolkit", toolkit_close))

        yield EpisodeRuntime(
            episode_id=simulator.episode_id,
            task=bootstrap.task,
            seed=bootstrap.seed,
            reset_monotonic=reset_monotonic,
            observation=observation,
            simulator=simulator,
            policy=policy,
            controller=controller,
            toolkit=toolkit,
        )
    finally:
        for label, cleanup in reversed(cleanups):
            try:
                cleanup()
            except Exception as exc:
                logger.warning("episode cleanup failed for %s: %s", label, exc)


def _milliseconds(duration_s: float) -> int:
    return max(0, round(duration_s * 1000))


__all__ = ["EpisodeBootstrap", "EpisodeRuntime", "open_episode"]
