"""Append-only task-batch attempt identities and state transitions."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rpent.evaluation.manifest import EvaluationManifest
from rpent.planning.manifest import canonical_json

TaskBatchStatus = Literal["running", "complete", "failed", "cancelled"]
_STATUSES = {"running", "complete", "failed", "cancelled"}


@dataclass(frozen=True, slots=True)
class TaskBatchAttempt:
    """One immutable state record for a task-scoped execution attempt."""

    manifest_id: str
    task_name: str
    batch_attempt_id: str
    status: TaskBatchStatus
    expected_case_ids: tuple[str, ...]
    completed_case_ids: tuple[str, ...]
    started_at: str
    finished_at: str | None

    def __post_init__(self) -> None:
        for name in ("manifest_id", "task_name", "batch_attempt_id", "started_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        validate_task_batch_attempt_id(self.batch_attempt_id)
        if self.status not in _STATUSES:
            raise ValueError(f"invalid task batch status: {self.status!r}")
        expected = tuple(self.expected_case_ids)
        completed = tuple(self.completed_case_ids)
        if not expected or not all(_non_empty_string(value) for value in expected):
            raise ValueError("expected_case_ids must contain non-empty strings")
        if len(set(expected)) != len(expected):
            raise ValueError("expected_case_ids contains duplicate Cases")
        if not all(_non_empty_string(value) for value in completed):
            raise ValueError("completed_case_ids must contain non-empty strings")
        if len(set(completed)) != len(completed):
            raise ValueError("completed_case_ids contains duplicate Cases")
        if completed != expected[: len(completed)]:
            raise ValueError("completed_case_ids must be an exact ordered prefix")
        if self.status == "running":
            if completed:
                raise ValueError("running batch record cannot contain completed Cases")
            if self.finished_at is not None:
                raise ValueError("running batch record cannot have finished_at")
        else:
            if not _non_empty_string(self.finished_at):
                raise ValueError("terminal batch record requires finished_at")
        if self.status == "complete" and completed != expected:
            raise ValueError("complete batch record must contain every expected Case")
        started = _timestamp(self.started_at, "started_at")
        if self.finished_at is not None:
            finished = _timestamp(self.finished_at, "finished_at")
            if finished < started:
                raise ValueError("finished_at cannot precede started_at")
        object.__setattr__(self, "expected_case_ids", expected)
        object.__setattr__(self, "completed_case_ids", completed)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> TaskBatchAttempt:
        if not isinstance(payload, Mapping):
            raise ValueError("task batch attempt must be a JSON object")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ValueError(f"invalid task batch attempt fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TaskBatchEvidence:
    """Validated local evidence for exactly one task-batch attempt."""

    task_name: str
    batch_attempt_id: str
    status: TaskBatchStatus
    results: tuple[Any, ...]
    worker_exit_code: int | None = None

    @property
    def complete(self) -> bool:
        return self.status == "complete" and self.worker_exit_code == 0


@dataclass(frozen=True, slots=True)
class TaskBatchProcessOutcome:
    """Coordinator-observed process outcome bound to one batch attempt."""

    manifest_id: str
    task_name: str
    batch_attempt_id: str
    worker_exit_code: int

    def __post_init__(self) -> None:
        for name in ("manifest_id", "task_name", "batch_attempt_id"):
            if not _non_empty_string(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty string")
        validate_task_batch_attempt_id(self.batch_attempt_id)
        if isinstance(self.worker_exit_code, bool) or not isinstance(
            self.worker_exit_code, int
        ):
            raise ValueError("worker_exit_code must be an integer")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> TaskBatchProcessOutcome:
        if not isinstance(payload, Mapping):
            raise ValueError("task batch process outcome must be a JSON object")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ValueError(
                f"invalid task batch process outcome fields: {exc}"
            ) from exc


def load_task_batch_process_outcomes(
    path: str | Path,
    manifest: EvaluationManifest,
) -> tuple[TaskBatchProcessOutcome, ...]:
    """Load the coordinator-owned append-only process outcome ledger."""
    records, _valid_end = _read_records(Path(path))
    outcomes = tuple(TaskBatchProcessOutcome.from_payload(record) for record in records)
    _validate_process_outcomes(manifest, outcomes)
    return outcomes


def append_task_batch_process_outcome(
    path: str | Path,
    manifest: EvaluationManifest,
    outcome: TaskBatchProcessOutcome,
) -> None:
    """Append and fsync one unique coordinator-observed process outcome."""
    if outcome.manifest_id != manifest.manifest_id:
        raise ValueError("task batch process outcome belongs to a foreign manifest")
    target = Path(path)
    records, valid_end = _read_records(target)
    existing = tuple(TaskBatchProcessOutcome.from_payload(record) for record in records)
    _validate_process_outcomes(manifest, (*existing, outcome))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size != valid_end:
        with target.open("r+b") as stream:
            stream.truncate(valid_end)
    encoded = (canonical_json(outcome.to_payload()) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short append while writing task batch process outcome")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_task_batch_attempts(
    path: str | Path,
    manifest: EvaluationManifest,
) -> tuple[TaskBatchAttempt, ...]:
    """Load and validate the complete append-only batch ledger."""
    records, _valid_end = _read_records(Path(path))
    attempts = tuple(TaskBatchAttempt.from_payload(record) for record in records)
    _validate_attempts(manifest, attempts)
    return attempts


def append_task_batch_attempt(
    path: str | Path,
    manifest: EvaluationManifest,
    attempt: TaskBatchAttempt,
) -> None:
    """Recover one partial tail, validate the transition, then append one line."""
    if not isinstance(attempt, TaskBatchAttempt):
        raise ValueError("attempt must be a TaskBatchAttempt")
    target = Path(path)
    records, valid_end = _read_records(target)
    existing = tuple(TaskBatchAttempt.from_payload(record) for record in records)
    _validate_attempts(manifest, (*existing, attempt))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size != valid_end:
        with target.open("r+b") as stream:
            stream.truncate(valid_end)
    encoded = (canonical_json(attempt.to_payload()) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short append while writing task batch attempt")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evaluate_task_batch_evidence(
    manifest: EvaluationManifest,
    task_name: str,
    batch_attempt_id: str,
    attempts: Iterable[TaskBatchAttempt],
    results: Iterable[Any],
    process_outcome: TaskBatchProcessOutcome | None = None,
) -> TaskBatchEvidence:
    """Validate one attempt's ledger and exact ordered local result prefix."""
    records = tuple(attempts)
    retained = tuple(results)
    _validate_attempts(manifest, records)
    if not records:
        raise ValueError("task batch evidence requires a running record")
    if any(
        record.task_name != task_name or record.batch_attempt_id != batch_attempt_id
        for record in records
    ):
        raise ValueError("task batch evidence contains mixed attempt identity")
    expected = tuple(
        case.case_id for case in manifest.cases if case.task_name == task_name
    )
    result_ids = tuple(getattr(result, "case_id", None) for result in retained)
    if result_ids != expected[: len(result_ids)]:
        raise ValueError("task batch results must be an exact ordered Case prefix")
    if any(
        getattr(result, "manifest_id", None) != manifest.manifest_id
        or getattr(result, "task_name", None) != task_name
        or getattr(result, "task_batch_attempt_id", None) != batch_attempt_id
        for result in retained
    ):
        raise ValueError("task batch result identity drift")
    terminal = records[-1]
    if terminal.status == "running" and len(records) != 1:
        raise ValueError("running task batch evidence has extra transitions")
    if terminal.status != "running" and len(records) != 2:
        raise ValueError("terminal task batch evidence requires one transition")
    if terminal.status != "running" and terminal.completed_case_ids != result_ids:
        raise ValueError("task batch ledger/result prefix mismatch")
    if terminal.status == "complete" and result_ids != expected:
        raise ValueError("complete task batch evidence requires every Case result")
    if process_outcome is not None and (
        process_outcome.manifest_id != manifest.manifest_id
        or process_outcome.task_name != task_name
        or process_outcome.batch_attempt_id != batch_attempt_id
    ):
        raise ValueError("task batch process outcome identity drift")
    return TaskBatchEvidence(
        task_name,
        batch_attempt_id,
        terminal.status,
        retained,
        None if process_outcome is None else process_outcome.worker_exit_code,
    )


def select_complete_task_batch(
    evidences: Iterable[TaskBatchEvidence],
) -> TaskBatchEvidence | None:
    """Select the sole complete attempt, failing closed on mixed completions."""
    complete = tuple(evidence for evidence in evidences if evidence.complete)
    if len(complete) > 1:
        raise ValueError("task has multiple complete batch attempts")
    return complete[0] if complete else None


def validate_task_batch_result(
    manifest: EvaluationManifest,
    case: Any,
    batch_attempt_id: str,
    result: Any,
) -> None:
    """Validate canonical per-Case identity and artifact ownership for import."""
    validate_task_batch_attempt_id(batch_attempt_id)
    expected_attempt_id = (
        f"{batch_attempt_id}-{case.episode_index:03d}-{case.case_id[:12]}"
    )
    validate_case_attempt_id(expected_attempt_id)
    if (
        getattr(result, "attempt_id", None) != expected_attempt_id
        or getattr(result, "task_batch_attempt_id", None) != batch_attempt_id
    ):
        raise ValueError("task batch result attempt identity is not canonical")
    root = Path(str(manifest.runtime.get("evaluation_root") or "")).resolve()
    expected_artifact = root / "cases" / case.case_id / "attempts" / expected_attempt_id
    artifact = Path(str(getattr(result, "artifact_dir", ""))).resolve()
    if artifact != expected_artifact:
        raise ValueError("task batch result artifact directory is not canonical")


def _validate_attempts(
    manifest: EvaluationManifest,
    attempts: Iterable[TaskBatchAttempt],
) -> None:
    if not isinstance(manifest, EvaluationManifest):
        raise ValueError("manifest must be an EvaluationManifest")
    states: dict[tuple[str, str], TaskBatchAttempt] = {}
    for attempt in attempts:
        if attempt.manifest_id != manifest.manifest_id:
            raise ValueError("task batch attempt belongs to a foreign manifest")
        expected = tuple(
            case.case_id
            for case in manifest.cases
            if case.task_name == attempt.task_name
        )
        if not expected:
            raise ValueError(
                f"task batch attempt references unknown task {attempt.task_name}"
            )
        if attempt.expected_case_ids != expected:
            raise ValueError(
                "task batch attempt does not use exact ordered Case membership"
            )
        key = (attempt.task_name, attempt.batch_attempt_id)
        previous = states.get(key)
        if previous is None:
            if attempt.status != "running":
                raise ValueError("task batch attempt must start with running")
        elif previous.status == "running":
            if attempt.status == "running":
                raise ValueError("duplicate running task batch record")
            if attempt.started_at != previous.started_at:
                raise ValueError("task batch attempt started_at identity changed")
        else:
            raise ValueError("task batch attempt already has a terminal record")
        states[key] = attempt


def _validate_process_outcomes(
    manifest: EvaluationManifest,
    outcomes: Iterable[TaskBatchProcessOutcome],
) -> None:
    task_names = {case.task_name for case in manifest.cases}
    seen: set[str] = set()
    for outcome in outcomes:
        if outcome.manifest_id != manifest.manifest_id:
            raise ValueError("task batch process outcome belongs to a foreign manifest")
        if outcome.task_name not in task_names:
            raise ValueError("task batch process outcome references unknown task")
        if outcome.batch_attempt_id in seen:
            raise ValueError("duplicate task batch process outcome")
        seen.add(outcome.batch_attempt_id)


def _read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    offset = 0
    for index, line in enumerate(lines):
        start = offset
        offset += len(line)
        if index == len(lines) - 1 and not data.endswith((b"\n", b"\r")):
            return records, start
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt task batch JSONL line {index + 1}: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ValueError(
                f"corrupt task batch JSONL line {index + 1}: not an object"
            )
        records.append(decoded)
    return records, len(data)


def _timestamp(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parsed


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_task_batch_attempt_id(value: Any) -> str:
    """Return one safe portable batch ID or reject it before side effects."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None
    ):
        raise ValueError(
            "task_batch_attempt_id must be a safe portable non-empty string"
        )
    return value


def validate_case_attempt_id(value: Any) -> str:
    """Return one safe portable Case attempt ID with its own length budget."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", value) is None
    ):
        raise ValueError("attempt_id must be a safe portable non-empty string")
    return value


__all__ = [
    "TaskBatchAttempt",
    "TaskBatchEvidence",
    "TaskBatchProcessOutcome",
    "TaskBatchStatus",
    "append_task_batch_attempt",
    "append_task_batch_process_outcome",
    "evaluate_task_batch_evidence",
    "load_task_batch_attempts",
    "load_task_batch_process_outcomes",
    "select_complete_task_batch",
    "validate_case_attempt_id",
    "validate_task_batch_attempt_id",
    "validate_task_batch_result",
]
