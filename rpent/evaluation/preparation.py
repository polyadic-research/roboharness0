"""Pure, immutable inputs produced by an environment evaluation Adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from rpent.evaluation.cases import EpisodeProtocol


class PolicyDaemon(Protocol):
    """Minimal shared-policy lifecycle consumed by the coordinator."""

    endpoint: str | None

    def start(self) -> Any:
        """Start or attach to the shared policy runtime."""
        raise NotImplementedError

    def stop(self) -> None:
        """Release the shared policy runtime."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PreparedEvaluation:
    """Environment-owned, side-effect-free evaluation manifest inputs."""

    tasks: tuple[Any, ...]
    protocol: EpisodeProtocol
    environment: Mapping[str, Any]
    simulator: Mapping[str, Any]
    policy: Mapping[str, Any]
    contracts: Mapping[str, Any]
    runtime: Mapping[str, Any]
    schedule: Mapping[str, Any] = field(default_factory=dict)
    qualification: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if not tasks:
            raise ValueError("prepared evaluation requires at least one task")
        if not isinstance(self.protocol, EpisodeProtocol):
            raise ValueError("protocol must be an EpisodeProtocol")
        object.__setattr__(self, "tasks", tasks)
        for name in (
            "environment",
            "simulator",
            "policy",
            "contracts",
            "runtime",
            "schedule",
            "qualification",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze_mapping(value))


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = ["PolicyDaemon", "PreparedEvaluation"]
