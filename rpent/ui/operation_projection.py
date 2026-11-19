"""Compact operation summaries projected from retained case events."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from rpent.ui.schemas import ActionCameraClipView, OperationSummaryView


@dataclass(slots=True)
class _Operation:
    operation_id: str
    first_sequence: int
    execution_id: str | None = None
    tool: str | None = None
    plan_step_id: str | None = None
    status: str = "running"
    started_sequence: int | None = None
    finished_sequence: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    first_step_id: int | None = None
    last_step_id: int | None = None
    executed_actions: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    clip_artifact_id: str | None = None
    has_physical_transitions: bool = False
    camera_clips: tuple[ActionCameraClipView, ...] = ()


class OperationProjection:
    """Consume an event journal and expose one row per physical operation."""

    def __init__(self) -> None:
        self._operations: dict[str, _Operation] = {}
        self._clips: list[tuple[str | None, int | None, str]] = []
        self._camera_clips: set[tuple[str, str, str]] = set()
        self._ready_camera_clips: dict[
            str, tuple[tuple[str, str], ...] | None
        ] = {}

    def copy(self) -> OperationProjection:
        return copy.deepcopy(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperationProjection):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def consume(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("event_type")
        if event_type == "action_camera_clip_available":
            self._consume_camera_clip(event)
            return
        if event_type == "action_camera_clips_ready":
            self._consume_camera_clips_ready(event)
            return
        if event_type == "action_clip_available":
            self._consume_clip(event)
            return
        if event_type not in {
            "operation_started",
            "operation_completed",
            "operation_failed",
            "transition_recorded",
        }:
            return

        payload = _mapping(event.get("payload"))
        operation_id = _text(event.get("operation_id"))
        sequence = _index(event.get("sequence"))
        if sequence is None or operation_id is None:
            return
        if event_type == "transition_recorded" and operation_id not in self._operations:
            return
        operation = self._operations.setdefault(
            operation_id,
            _Operation(operation_id=operation_id, first_sequence=sequence),
        )
        operation.first_sequence = min(operation.first_sequence, sequence)
        operation.execution_id = (
            operation.execution_id
            or _text(event.get("execution_id"))
        )

        if event_type == "operation_started":
            operation.started_sequence = sequence
            operation.started_at = _text(event.get("recorded_at"))
            operation.tool = _text(payload.get("tool")) or operation.tool
            operation.plan_step_id = (
                _text(payload.get("plan_step_id")) or operation.plan_step_id
            )
            operation.status = "running"
            return
        if event_type in {"operation_completed", "operation_failed"}:
            operation.finished_sequence = sequence
            operation.finished_at = _text(event.get("recorded_at"))
            operation.tool = _text(payload.get("tool")) or operation.tool
            operation.plan_step_id = (
                _text(payload.get("plan_step_id")) or operation.plan_step_id
            )
            first_step_id = _index(payload.get("first_step_id"))
            last_step_id = _index(payload.get("last_step_id"))
            if first_step_id is not None:
                operation.first_step_id = first_step_id
            if last_step_id is not None:
                operation.last_step_id = last_step_id
            operation.executed_actions = _index(payload.get("executed_actions"))
            result = payload.get("result")
            operation.result = dict(result) if isinstance(result, Mapping) else None
            operation.error = _text(payload.get("error"))
            operation.status = (
                "failed" if event_type == "operation_failed" else "succeeded"
            )
            return

        step_id = _index(event.get("step_id"))
        operation.has_physical_transitions = True
        if step_id is not None:
            operation.first_step_id = (
                step_id
                if operation.first_step_id is None
                else min(operation.first_step_id, step_id)
            )
            operation.last_step_id = (
                step_id
                if operation.last_step_id is None
                else max(operation.last_step_id, step_id)
            )

    def summaries(
        self, valid_camera_clip_artifacts: frozenset[str] = frozenset()
    ) -> tuple[OperationSummaryView, ...]:
        operations = sorted(
            (item for item in self._operations.values() if item.tool is not None),
            key=lambda item: item.first_sequence,
        )
        by_id = {item.operation_id: item for item in operations}
        for operation_id, ordinal, artifact_id in self._clips:
            operation = by_id.get(operation_id) if operation_id is not None else None
            if operation is None and ordinal is not None and 0 < ordinal <= len(operations):
                operation = operations[ordinal - 1]
            if operation is not None:
                operation.clip_artifact_id = artifact_id
        by_pair: dict[tuple[str, str], set[str]] = {}
        by_artifact: dict[str, set[tuple[str, str]]] = {}
        for operation_id, camera_id, artifact_id in self._camera_clips:
            pair = (operation_id, camera_id)
            by_pair.setdefault(pair, set()).add(artifact_id)
            by_artifact.setdefault(artifact_id, set()).add(pair)
        for operation in operations:
            ready = self._ready_camera_clips.get(operation.operation_id)
            if ready is None:
                operation.camera_clips = ()
                continue
            ready_pairs = set(ready)
            operation.camera_clips = tuple(
                ActionCameraClipView(camera_id=camera_id, artifact_id=artifact_id)
                for (operation_id, camera_id), artifact_ids in sorted(by_pair.items())
                if operation_id == operation.operation_id
                and (camera_id, next(iter(artifact_ids), "")) in ready_pairs
                and operation.has_physical_transitions
                and len(artifact_ids) == 1
                for artifact_id in artifact_ids
                if artifact_id in valid_camera_clip_artifacts
                and len(by_artifact.get(artifact_id, ())) == 1
            )
        return tuple(self._view(operation) for operation in operations)

    def _consume_camera_clip(self, event: Mapping[str, Any]) -> None:
        payload = _mapping(event.get("payload"))
        operation_id = _text(event.get("operation_id"))
        camera_id = _text(payload.get("camera_id"))
        artifact_id = _text(payload.get("artifact_id"))
        if operation_id is None or camera_id is None or artifact_id is None:
            return
        self._camera_clips.add((operation_id, camera_id, artifact_id))

    def _consume_camera_clips_ready(self, event: Mapping[str, Any]) -> None:
        payload = _mapping(event.get("payload"))
        operation_id = _text(event.get("operation_id"))
        clips = payload.get("clips")
        if operation_id is None or not isinstance(clips, list):
            return
        pairs: list[tuple[str, str]] = []
        for item in clips:
            if not isinstance(item, Mapping):
                self._ready_camera_clips[operation_id] = None
                return
            camera_id = _text(item.get("camera_id"))
            artifact_id = _text(item.get("artifact_id"))
            if camera_id is None or artifact_id is None:
                self._ready_camera_clips[operation_id] = None
                return
            pairs.append((camera_id, artifact_id))
        if len(set(pairs)) != len(pairs) or len({item[0] for item in pairs}) != len(
            pairs
        ):
            self._ready_camera_clips[operation_id] = None
            return
        normalized = tuple(pairs)
        previous = self._ready_camera_clips.get(operation_id, normalized)
        self._ready_camera_clips[operation_id] = (
            normalized if previous == normalized else None
        )

    def _consume_clip(self, event: Mapping[str, Any]) -> None:
        payload = _mapping(event.get("payload"))
        artifact_id = _text(payload.get("artifact_id"))
        if artifact_id is None:
            return
        self._clips.append(
            (
                _text(event.get("operation_id")),
                _index(payload.get("step")),
                artifact_id,
            )
        )

    @staticmethod
    def _view(operation: _Operation) -> OperationSummaryView:
        assert operation.tool is not None
        return OperationSummaryView(
            operation_id=operation.operation_id,
            execution_id=operation.execution_id,
            tool=operation.tool,
            plan_step_id=operation.plan_step_id,
            status=operation.status,
            started_sequence=operation.started_sequence,
            finished_sequence=operation.finished_sequence,
            first_step_id=operation.first_step_id,
            last_step_id=operation.last_step_id,
            executed_actions=operation.executed_actions,
            result=operation.result,
            elapsed_seconds=_elapsed(operation.started_at, operation.finished_at),
            error=operation.error,
            clip_artifact_id=operation.clip_artifact_id,
            has_physical_transitions=operation.has_physical_transitions,
            camera_clips=operation.camera_clips,
        )


def _elapsed(started_at: str | None, finished_at: str | None) -> float | None:
    if started_at is None or finished_at is None:
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        seconds = (finished - started).total_seconds()
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _index(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )
