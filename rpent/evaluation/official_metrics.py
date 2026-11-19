"""Fail-closed official success metrics over accepted task-batch evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rpent.evaluation.batches import (
    TaskBatchEvidence,
    select_complete_task_batch,
    validate_task_batch_result,
)
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult, aggregate_results


@dataclass(frozen=True, slots=True)
class OfficialMetrics:
    manifest_id: str
    task_success_rates: Mapping[str, float]
    split_success_rates: Mapping[str, float]
    overall_success_rate: float
    completed_rollouts: int
    scheduled_rollouts: int
    completed_tasks: int
    scheduled_tasks: int
    is_complete: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "available": True,
            "manifest_id": self.manifest_id,
            "task_success_rates": dict(self.task_success_rates),
            "split_success_rates": dict(self.split_success_rates),
            "overall_success_rate": self.overall_success_rate,
            "completed_rollouts": self.completed_rollouts,
            "scheduled_rollouts": self.scheduled_rollouts,
            "completed_tasks": self.completed_tasks,
            "scheduled_tasks": self.scheduled_tasks,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_payload(
        cls, manifest: EvaluationManifest, payload: Any
    ) -> OfficialMetrics:
        """Validate a retained projection against its immutable workload."""
        expected = {
            "schema_version",
            "available",
            "manifest_id",
            "task_success_rates",
            "split_success_rates",
            "overall_success_rate",
            "completed_rollouts",
            "scheduled_rollouts",
            "completed_tasks",
            "scheduled_tasks",
            "is_complete",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("official metrics fields do not match")
        if (
            payload["schema_version"] != 1
            or payload["available"] is not True
            or payload["manifest_id"] != manifest.manifest_id
        ):
            raise ValueError("official metrics identity mismatch")
        task_names = tuple(dict.fromkeys(case.task_name for case in manifest.cases))
        task_rates = _rates(payload["task_success_rates"], set(task_names))
        membership = _split_membership(manifest, task_names)
        split_rates = _rates(payload["split_success_rates"], set(membership))
        expected_split_rates = {
            split: sum(task_rates[task] for task in members) / len(members)
            for split, members in membership.items()
        }
        if split_rates != expected_split_rates:
            raise ValueError("official split rates do not match task rates")
        overall = _rate(payload["overall_success_rate"])
        if overall != sum(task_rates.values()) / len(task_names):
            raise ValueError("official overall rate does not match task rates")
        counts = tuple(
            _count(payload[name])
            for name in (
                "completed_rollouts",
                "scheduled_rollouts",
                "completed_tasks",
                "scheduled_tasks",
            )
        )
        completed_rollouts, scheduled_rollouts, completed_tasks, scheduled_tasks = (
            counts
        )
        if scheduled_rollouts != len(manifest.cases) or scheduled_tasks != len(
            task_names
        ):
            raise ValueError("official workload counts do not match manifest")
        task_sizes = tuple(
            sum(case.task_name == task for case in manifest.cases)
            for task in task_names
        )
        possible = {(0, 0)}
        for size in task_sizes:
            possible |= {
                (task_count + 1, rollout_count + size)
                for task_count, rollout_count in tuple(possible)
            }
        if (
            completed_rollouts > scheduled_rollouts
            or completed_tasks > scheduled_tasks
            or (completed_tasks, completed_rollouts) not in possible
            or sum(rate > 0 for rate in task_rates.values()) > completed_tasks
        ):
            raise ValueError("official completion counts are inconsistent")
        is_complete = payload["is_complete"]
        if not isinstance(is_complete, bool) or is_complete != (
            completed_rollouts == scheduled_rollouts
            and completed_tasks == scheduled_tasks
        ):
            raise ValueError("official completeness does not match counts")
        return cls(
            manifest.manifest_id,
            MappingProxyType(task_rates),
            MappingProxyType(split_rates),
            overall,
            completed_rollouts,
            scheduled_rollouts,
            completed_tasks,
            scheduled_tasks,
            is_complete,
        )


def aggregate_official_metrics(
    manifest: EvaluationManifest,
    results: Iterable[EpisodeResult],
    batches: Iterable[TaskBatchEvidence],
) -> OfficialMetrics:
    """Aggregate native simulator success from one accepted batch per task."""
    retained = tuple(results)
    aggregate_results(manifest, retained)
    central_by_attempt = {result.attempt_id: result for result in retained}
    task_names = tuple(dict.fromkeys(case.task_name for case in manifest.cases))
    task_cases = {
        task: tuple(case for case in manifest.cases if case.task_name == task)
        for task in task_names
    }
    grouped: dict[str, list[TaskBatchEvidence]] = {task: [] for task in task_names}
    for evidence in batches:
        if evidence.task_name not in grouped:
            raise ValueError("official batch evidence references unknown task")
        grouped[evidence.task_name].append(evidence)

    selected: dict[str, TaskBatchEvidence] = {}
    for task_name in task_names:
        evidence = select_complete_task_batch(grouped[task_name])
        if evidence is None:
            continue
        expected_cases = task_cases[task_name]
        if tuple(result.case_id for result in evidence.results) != tuple(
            case.case_id for case in expected_cases
        ):
            raise ValueError("accepted official batch must contain exact ordered Cases")
        for case, result in zip(expected_cases, evidence.results, strict=True):
            validate_task_batch_result(
                manifest, case, evidence.batch_attempt_id, result
            )
            if central_by_attempt.get(result.attempt_id) != result:
                raise ValueError(
                    "accepted official batch result is absent or drifting centrally"
                )
        selected[task_name] = evidence

    rates: dict[str, float] = {}
    for task_name in task_names:
        evidence = selected.get(task_name)
        successes = (
            sum(result.simulator_success for result in evidence.results)
            if evidence is not None
            else 0
        )
        rates[task_name] = successes / len(task_cases[task_name])

    split_membership = _split_membership(manifest, task_names)
    split_rates = {
        split: sum(rates[task] for task in members) / len(members)
        for split, members in split_membership.items()
    }
    completed_rollouts = sum(len(evidence.results) for evidence in selected.values())
    return OfficialMetrics(
        manifest_id=manifest.manifest_id,
        task_success_rates=MappingProxyType(rates),
        split_success_rates=MappingProxyType(split_rates),
        overall_success_rate=sum(rates.values()) / len(task_names),
        completed_rollouts=completed_rollouts,
        scheduled_rollouts=len(manifest.cases),
        completed_tasks=len(selected),
        scheduled_tasks=len(task_names),
        is_complete=len(selected) == len(task_names),
    )


def _split_membership(
    manifest: EvaluationManifest,
    task_names: tuple[str, ...],
) -> Mapping[str, tuple[str, ...]]:
    declared = manifest.schedule.get("task_set_membership")
    if declared is None:
        return MappingProxyType({"all": task_names})
    if not isinstance(declared, Mapping):
        raise ValueError("task_set_membership must be a mapping")
    split_names = manifest.schedule.get("task_sets")
    if not isinstance(split_names, (tuple, list)) or not split_names:
        raise ValueError("task_sets must declare stable split order")
    membership: dict[str, tuple[str, ...]] = {}
    flattened = []
    for split in split_names:
        if not isinstance(split, str) or not split:
            raise ValueError("task_sets contains an invalid split")
        members = declared.get(split)
        if not isinstance(members, (tuple, list)) or not members:
            raise ValueError("task_set_membership contains an invalid split")
        membership[split] = tuple(members)
        flattened.extend(members)
    if tuple(flattened) != task_names or len(set(flattened)) != len(flattened):
        raise ValueError("task_set_membership does not exactly partition tasks")
    return MappingProxyType(membership)


def _rate(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("official success rate is invalid")
    return float(value)


def _rates(value: Any, expected: set[str]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("official success-rate membership mismatch")
    return {str(name): _rate(rate) for name, rate in value.items()}


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("official count is invalid")
    return value


__all__ = ["OfficialMetrics", "aggregate_official_metrics"]
