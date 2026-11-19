"""Append-only episode results and denominator-safe aggregation."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from rpent.evaluation.batches import validate_task_batch_attempt_id
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.planning.manifest import canonical_json, content_hash

EpisodeStatus = Literal[
    "success",
    "terminal_failure",
    "truncated",
    "timeout",
    "error",
    "cancelled",
]
TERMINAL_STATUSES: tuple[EpisodeStatus, ...] = (
    "success",
    "terminal_failure",
    "truncated",
    "timeout",
    "error",
)
RESULT_STATUSES: tuple[EpisodeStatus, ...] = (
    *TERMINAL_STATUSES,
    "cancelled",
)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One retained terminal outcome for exactly one manifest case."""

    manifest_id: str
    case_id: str
    status: EpisodeStatus
    environment: str
    preset_name: str
    task_name: str
    task_index: int | None
    seed: int
    episode_index: int
    worker_pid: int | None
    worker_exit_code: int | None
    started_at: str
    finished_at: str
    artifact_dir: str
    simulator_success: bool
    contract_success: bool
    terminated: bool
    truncated: bool
    step_count: int
    action_count: int
    checkpoint_repo_id: str
    checkpoint_revision: str
    binding_name: str
    error_type: str | None = None
    error_message: str | None = None
    evaluation_id: str = ""
    attempt_id: str = ""
    task_batch_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"invalid episode result status: {self.status!r}")
        if not self.evaluation_id:
            object.__setattr__(self, "evaluation_id", self.manifest_id)
        if not self.attempt_id:
            object.__setattr__(
                self,
                "attempt_id",
                "legacy-"
                + content_hash(
                    {
                        "manifest_id": self.manifest_id,
                        "case_id": self.case_id,
                        "status": self.status,
                        "finished_at": self.finished_at,
                        "artifact_dir": self.artifact_dir,
                    }
                ),
            )
        for name in (
            "evaluation_id",
            "manifest_id",
            "case_id",
            "attempt_id",
            "environment",
            "preset_name",
            "task_name",
            "started_at",
            "finished_at",
            "artifact_dir",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "binding_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.task_batch_attempt_id is not None:
            validate_task_batch_attempt_id(self.task_batch_attempt_id)
        for name in ("seed", "episode_index", "step_count", "action_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("worker_pid", "worker_exit_code", "task_index"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{name} must be an integer or None")
        for name in (
            "simulator_success",
            "contract_success",
            "terminated",
            "truncated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.status == "success" and not (
            self.simulator_success and self.contract_success
        ):
            raise ValueError(
                "success requires simulator_success=true and contract_success=true"
            )
        if self.status == "success" and self.worker_exit_code != 0:
            raise ValueError("success requires worker_exit_code=0")
        if self.status == "truncated" and not self.truncated:
            raise ValueError("truncated status requires truncated=true")

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON object stored in the result ledger."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> EpisodeResult:
        """Validate one decoded result object."""
        if not isinstance(payload, dict):
            raise ValueError("episode result must be a JSON object")
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid episode result fields: {exc}") from exc


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Materialized aggregate derived only from retained results."""

    manifest_id: str
    scheduled: int
    completed: int
    successful: int
    success_rate: float
    errors: int
    timeouts: int
    status_counts: Mapping[str, int]
    per_task: Mapping[str, Mapping[str, int | float]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "scheduled": self.scheduled,
            "completed": self.completed,
            "successful": self.successful,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "status_counts": dict(self.status_counts),
            "per_task": {task: dict(values) for task, values in self.per_task.items()},
        }


def aggregate_results(
    manifest: EvaluationManifest,
    results: Iterable[EpisodeResult],
) -> EvaluationSummary:
    """Aggregate retained records using all scheduled cases as denominator."""
    if not isinstance(manifest, EvaluationManifest):
        raise ValueError("manifest must be an EvaluationManifest")
    case_by_id = {case.case_id: case for case in manifest.cases}
    retained = tuple(results)
    seen_attempts: set[str] = set()
    for result in retained:
        if not isinstance(result, EpisodeResult):
            raise ValueError("results must contain EpisodeResult values")
        if result.attempt_id in seen_attempts:
            raise ValueError(
                f"duplicate result attempt {result.attempt_id} "
                f"for case {result.case_id}"
            )
        seen_attempts.add(result.attempt_id)
        _validate_result(manifest, case_by_id, result)

    # The ledger is append-only, so its last retained attempt is authoritative.
    latest_by_case: dict[str, EpisodeResult] = {}
    for result in retained:
        latest_by_case[result.case_id] = result
    latest = tuple(latest_by_case.values())
    completed_results = tuple(
        result for result in latest if result.status != "cancelled"
    )

    counts = Counter(result.status for result in latest)
    status_counts = MappingProxyType(
        {status: counts.get(status, 0) for status in RESULT_STATUSES}
    )
    scheduled = len(manifest.cases)
    successful = counts.get("success", 0)
    per_task: dict[str, Mapping[str, int | float]] = {}
    for task_name in dict.fromkeys(case.task_name for case in manifest.cases):
        task_cases = [case for case in manifest.cases if case.task_name == task_name]
        task_ids = {case.case_id for case in task_cases}
        completed = sum(result.case_id in task_ids for result in completed_results)
        successes = sum(
            result.case_id in task_ids and result.status == "success"
            for result in completed_results
        )
        per_task[task_name] = MappingProxyType(
            {
                "scheduled": len(task_cases),
                "completed": completed,
                "successful": successes,
                "success_rate": successes / len(task_cases),
            }
        )
    return EvaluationSummary(
        manifest_id=manifest.manifest_id,
        scheduled=scheduled,
        completed=len(completed_results),
        successful=successful,
        success_rate=successful / scheduled,
        errors=counts.get("error", 0),
        timeouts=counts.get("timeout", 0),
        status_counts=status_counts,
        per_task=MappingProxyType(per_task),
    )


class ResultStore:
    """Sequential append/read boundary for one manifest result ledger."""

    def __init__(self, path: str | Path, manifest: EvaluationManifest) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self._lock = threading.Lock()

    def load(self) -> tuple[EpisodeResult, ...]:
        records, _valid_end = _read_records(self.path)
        results = tuple(EpisodeResult.from_payload(record) for record in records)
        aggregate_results(self.manifest, results)
        return results

    def append(self, result: EpisodeResult) -> None:
        """Validate, recover one crash tail, and append one canonical line."""
        disposition = self.append_or_confirm(result)
        if disposition == "identical":
            raise ValueError(
                f"duplicate result attempt {result.attempt_id} for case {result.case_id}"
            )

    def append_or_confirm(
        self, result: EpisodeResult, *, require_existing: bool = False
    ) -> Literal["appended", "identical"]:
        """Append once, confirm exact retries, and reject Attempt payload drift."""
        with self._lock:
            return self._append_or_confirm(
                result, require_existing=require_existing
            )

    def _append_or_confirm(
        self, result: EpisodeResult, *, require_existing: bool = False
    ) -> Literal["appended", "identical"]:
        case_by_id = {case.case_id: case for case in self.manifest.cases}
        _validate_result(self.manifest, case_by_id, result)
        existing_records, valid_end = _read_records(self.path)
        existing = tuple(
            EpisodeResult.from_payload(record) for record in existing_records
        )
        matching = next(
            (item for item in existing if item.attempt_id == result.attempt_id), None
        )
        if matching is not None:
            if matching != result:
                raise ValueError("existing attempt payload conflicts with accepted result")
            return "identical"
        if require_existing:
            raise ValueError(
                "already_accepted_repair requires an accepted result"
            )
        aggregate_results(self.manifest, (*existing, result))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size != valid_end:
            with self.path.open("r+b") as stream:
                stream.truncate(valid_end)
        encoded = (canonical_json(result.to_payload()) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short append while writing episode result")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return "appended"


def write_summary(path: str | Path, summary: EvaluationSummary) -> None:
    """Atomically replace the materialized summary for retained results."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(summary.to_payload()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_result(
    manifest: EvaluationManifest,
    case_by_id: Mapping[str, EpisodeCase],
    result: EpisodeResult,
) -> None:
    if result.manifest_id != manifest.manifest_id:
        raise ValueError("result belongs to a foreign manifest")
    allowed_evaluation_ids = {manifest.evaluation_id}
    if manifest.legacy_identity:
        allowed_evaluation_ids.add(manifest.manifest_id)
    if result.evaluation_id not in allowed_evaluation_ids:
        raise ValueError("result belongs to a foreign evaluation")
    case = case_by_id.get(result.case_id)
    if case is None:
        raise ValueError(f"result references unknown case {result.case_id}")
    expected = {
        "environment": case.environment,
        "preset_name": case.preset_name,
        "task_name": case.task_name,
        "task_index": case.task_index,
        "seed": case.seed,
        "episode_index": case.episode_index,
        "checkpoint_repo_id": case.checkpoint_repo_id,
        "checkpoint_revision": case.checkpoint_revision,
        "binding_name": case.binding_name,
    }
    drift = {
        name: (value, getattr(result, name))
        for name, value in expected.items()
        if getattr(result, name) != value
    }
    if drift:
        raise ValueError(f"result case identity does not match manifest: {drift}")


def _read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    offset = 0
    for index, line in enumerate(lines):
        line_start = offset
        offset += len(line)
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = index == len(lines) - 1 and not data.endswith(
                (b"\n", b"\r")
            )
            if is_incomplete_tail:
                return records, line_start
            raise ValueError(f"corrupt result JSONL line {index + 1}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"corrupt result JSONL line {index + 1}: not an object")
        records.append(decoded)
    return records, len(data)


__all__ = [
    "EpisodeResult",
    "EpisodeStatus",
    "EvaluationSummary",
    "ResultStore",
    "RESULT_STATUSES",
    "TERMINAL_STATUSES",
    "aggregate_results",
    "write_summary",
]
