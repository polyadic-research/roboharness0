"""Immutable episode identities and deterministic schedule expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Iterable, Literal

from rpent.planning.manifest import content_hash


@dataclass(frozen=True, slots=True)
class EpisodeProtocol:
    """Execution identity shared by every task in one evaluation request."""

    environment: str
    preset_name: str
    horizon_ceiling: int | None
    checkpoint_repo_id: str
    checkpoint_revision: str
    binding_name: str

    def __post_init__(self) -> None:
        for name in (
            "environment",
            "preset_name",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "binding_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.horizon_ceiling is not None and not _positive_integer(
            self.horizon_ceiling
        ):
            raise ValueError("horizon_ceiling must be None or a positive integer")


@dataclass(frozen=True, slots=True)
class EpisodeCase:
    """One fully resolved simulator-policy episode."""

    case_id: str = field(init=False)
    ordinal: int
    environment: str
    preset_name: str
    task_name: str
    task_index: int | None
    instruction: str
    seed: int
    episode_index: int
    horizon: int
    checkpoint_repo_id: str
    checkpoint_revision: str
    binding_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, Integral) or isinstance(self.ordinal, bool):
            raise ValueError("ordinal must be a non-negative integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if self.task_index is not None and (
            not isinstance(self.task_index, Integral)
            or isinstance(self.task_index, bool)
            or self.task_index < 0
        ):
            raise ValueError("task_index must be a non-negative integer or None")
        for name in ("seed", "episode_index"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not _positive_integer(self.horizon):
            raise ValueError("horizon must be a positive integer")
        for name in (
            "environment",
            "preset_name",
            "task_name",
            "instruction",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "binding_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "task_index", _optional_int(self.task_index))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "case_id", _case_digest(self))

    def identity_payload(self) -> dict[str, Any]:
        """Return only fields that can change episode execution."""
        return {
            "environment": self.environment,
            "preset_name": self.preset_name,
            "task_name": self.task_name,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "horizon": self.horizon,
            "checkpoint_repo_id": self.checkpoint_repo_id,
            "checkpoint_revision": self.checkpoint_revision,
            "binding_name": self.binding_name,
        }

    def to_payload(self) -> dict[str, Any]:
        """Return the verified JSON representation including schedule order."""
        return {
            "case_id": self.case_id,
            "ordinal": self.ordinal,
            **self.identity_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> EpisodeCase:
        """Rebuild one case and reject a stale or spoofed digest."""
        if not isinstance(payload, dict):
            raise ValueError("episode case payload must be an object")
        try:
            expected_id = payload["case_id"]
            case = cls(
                ordinal=payload["ordinal"],
                environment=payload["environment"],
                preset_name=payload["preset_name"],
                task_name=payload["task_name"],
                task_index=payload["task_index"],
                instruction=payload["instruction"],
                seed=payload["seed"],
                episode_index=payload["episode_index"],
                horizon=payload["horizon"],
                checkpoint_repo_id=payload["checkpoint_repo_id"],
                checkpoint_revision=payload["checkpoint_revision"],
                binding_name=payload["binding_name"],
            )
        except KeyError as exc:
            raise ValueError(f"episode case is missing {exc.args[0]!r}") from exc
        if expected_id != case.case_id:
            raise ValueError(
                f"case_id mismatch: expected {case.case_id}, actual {expected_id}"
            )
        return case


def expand_episode_cases(
    resolved_tasks: Iterable[Any],
    protocol: EpisodeProtocol,
    *,
    episodes: int = 1,
    base_seed: int = 0,
    seed_policy: Literal[
        "base_seed_plus_episode_index",
        "shared_environment_sequence",
    ] = "base_seed_plus_episode_index",
) -> tuple[EpisodeCase, ...]:
    """Expand tasks in registry order, then comparable per-task seeds."""
    if not isinstance(protocol, EpisodeProtocol):
        raise ValueError("protocol must be an EpisodeProtocol")
    if not _positive_integer(episodes):
        raise ValueError("episodes must be a positive integer")
    if (
        not isinstance(base_seed, Integral)
        or isinstance(base_seed, bool)
        or base_seed < 0
    ):
        raise ValueError("base_seed must be a non-negative integer")
    if seed_policy not in {
        "base_seed_plus_episode_index",
        "shared_environment_sequence",
    }:
        raise ValueError(f"unknown seed_policy: {seed_policy!r}")

    tasks = tuple(resolved_tasks)
    seen: set[tuple[str, int | None]] = set()
    cases: list[EpisodeCase] = []
    for task in tasks:
        name = str(getattr(task, "name", "")).strip()
        index = getattr(task, "index", None)
        identity = (name, index)
        if identity in seen:
            raise ValueError(f"duplicate resolved task identity: {identity!r}")
        seen.add(identity)
        instruction = str(
            getattr(task, "language", None) or getattr(task, "instruction", "")
        ).strip()
        upstream_horizon = getattr(task, "upstream_horizon", None)
        if not _positive_integer(upstream_horizon):
            raise ValueError(f"task {name!r} horizon must be positive")
        horizon = int(upstream_horizon)
        if protocol.horizon_ceiling is not None:
            horizon = min(horizon, int(protocol.horizon_ceiling))
        for episode_index in range(int(episodes)):
            seed = int(base_seed)
            if seed_policy == "base_seed_plus_episode_index":
                seed += episode_index
            cases.append(
                EpisodeCase(
                    ordinal=len(cases),
                    environment=protocol.environment,
                    preset_name=protocol.preset_name,
                    task_name=name,
                    task_index=index,
                    instruction=instruction,
                    seed=seed,
                    episode_index=episode_index,
                    horizon=horizon,
                    checkpoint_repo_id=protocol.checkpoint_repo_id,
                    checkpoint_revision=protocol.checkpoint_revision,
                    binding_name=protocol.binding_name,
                )
            )
    return tuple(cases)


def _case_digest(case: EpisodeCase) -> str:
    return content_hash(case.identity_payload())


def _positive_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _optional_int(value: Integral | None) -> int | None:
    return None if value is None else int(value)


__all__ = ["EpisodeCase", "EpisodeProtocol", "expand_episode_cases"]
