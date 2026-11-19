"""Atomic runtime ownership of completed and remaining outcome stages."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar


class StageLike(Protocol):
    id: str
    outcome_ids: tuple[str, ...]


StageT = TypeVar("StageT", bound=StageLike)


class StageRuntime(Generic[StageT]):
    def __init__(self, stages: tuple[StageT, ...], *, active_index: int = 0) -> None:
        if not stages or not 0 <= active_index < len(stages):
            raise ValueError("stage runtime requires a valid active stage")
        self._require_unique_stage_ids(stages)
        self._stages = stages
        self._active_index = active_index

    @property
    def stages(self) -> tuple[StageT, ...]:
        return self._stages

    @property
    def active(self) -> StageT:
        return self._stages[self._active_index]

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def completed(self) -> tuple[StageT, ...]:
        return self._stages[: self._active_index]

    @property
    def has_next(self) -> bool:
        return self._active_index + 1 < len(self._stages)

    def advance(self, *, stage_complete: bool) -> StageT:
        if not stage_complete:
            raise ValueError("current stage must be complete before advancing")
        if not self.has_next:
            raise ValueError("cannot advance beyond the final stage")
        self._active_index += 1
        return self.active

    def replace_remaining(
        self,
        replacement: tuple[StageT, ...],
        unresolved_outcome_ids: tuple[str, ...],
    ) -> tuple[StageT, ...]:
        if not replacement:
            raise ValueError("remaining stage replacement must be non-empty")
        if not unresolved_outcome_ids or len(unresolved_outcome_ids) != len(
            set(unresolved_outcome_ids)
        ):
            raise ValueError("unresolved outcomes must be unique and non-empty")
        candidate = (*self._stages[: self._active_index], *replacement)
        self._require_unique_stage_ids(candidate)
        declared = tuple(
            outcome_id for stage in replacement for outcome_id in stage.outcome_ids
        )
        if declared != unresolved_outcome_ids:
            raise ValueError(
                "replacement stages must exactly partition unresolved outcomes in order"
            )
        self._stages = tuple(candidate)
        return self._stages

    @staticmethod
    def _require_unique_stage_ids(stages: tuple[StageT, ...]) -> None:
        ids = tuple(stage.id for stage in stages)
        if any(not stage_id.strip() for stage_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("stage ids must be non-empty and unique")


__all__ = ["StageLike", "StageRuntime"]
