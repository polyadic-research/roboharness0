"""Durable, strictly validated Evaluation preparation progress."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal

from rpent.recording.media import atomic_write_json, ensure_directory

PREPARATION_PHASES = (
    "evaluation_created",
    "sam3_loading",
    "vla_loading",
    "simulator_starting",
    "planner_starting",
    "running",
    "finalizing",
)
PreparationPhaseStatus = Literal[
    "pending",
    "loading",
    "ready",
    "reused",
    "skipped",
    "failed",
    "cancelled",
]
PreparationStatus = Literal["preparing", "ready", "failed", "cancelled"]

_PHASE_STATUSES = {
    "pending",
    "loading",
    "ready",
    "reused",
    "skipped",
    "failed",
    "cancelled",
}
_COMPLETED_PHASE_STATUSES = {"ready", "reused", "skipped"}
_TERMINAL_PROGRESS_STATUSES = {"failed", "cancelled"}


@dataclass(frozen=True, slots=True)
class PreparationError:
    error_type: str
    message: str

    def __post_init__(self) -> None:
        _safe_text(self.error_type, "error_type")
        _safe_text(self.message, "message")


@dataclass(frozen=True, slots=True)
class PreparationPhase:
    phase: str
    status: PreparationPhaseStatus
    started_at: str | None
    finished_at: str | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class PreparationSnapshot:
    schema_version: int
    evaluation_id: str
    manifest_id: str
    status: PreparationStatus
    current_phase: str
    phases: tuple[PreparationPhase, ...]
    error: PreparationError | None


class PreparationProgress:
    """Atomically advance one run-owned preparation journal."""

    def __init__(self, path: Path, *, clock: Callable[[], str] | None = None) -> None:
        self.path = Path(path)
        self._clock = clock or _utc_now

    @classmethod
    def create(
        cls,
        path: str | Path,
        evaluation_id: str,
        manifest_id: str,
        *,
        clock: Callable[[], str] | None = None,
    ) -> PreparationProgress:
        _safe_text(evaluation_id, "evaluation_id")
        _safe_text(manifest_id, "manifest_id")
        progress = cls(Path(path), clock=clock)
        created_at = _timestamp(progress._clock(), "created_at")
        phases = [
            {
                "phase": phase,
                "status": "ready" if index == 0 else "pending",
                "started_at": created_at if index == 0 else None,
                "finished_at": created_at if index == 0 else None,
                "detail": None,
            }
            for index, phase in enumerate(PREPARATION_PHASES)
        ]
        payload = {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "manifest_id": manifest_id,
            "status": "preparing",
            "current_phase": "evaluation_created",
            "phases": phases,
            "error": None,
        }
        with _locked(progress.path):
            if progress.path.exists():
                raise FileExistsError(progress.path)
            atomic_write_json(progress.path, payload)
        return progress

    def advance(
        self,
        phase: str,
        status: PreparationPhaseStatus,
        *,
        detail: str | None = None,
        error: PreparationError | None = None,
        allow_replay: bool = False,
    ) -> None:
        if phase not in PREPARATION_PHASES:
            raise ValueError(f"unknown preparation phase: {phase}")
        if status not in _PHASE_STATUSES:
            raise ValueError(f"unknown preparation status: {status}")
        if detail is not None:
            _safe_text(detail, "detail")
        if status == "failed":
            if not isinstance(error, PreparationError):
                raise ValueError("failed preparation requires a PreparationError")
        elif error is not None:
            raise ValueError("preparation error is valid only for failed status")

        with _locked(self.path):
            snapshot = _load(self.path)
            if allow_replay and _transition_was_already_applied(
                snapshot, phase, status
            ):
                return
            if snapshot.status in _TERMINAL_PROGRESS_STATUSES:
                raise ValueError("terminal preparation cannot advance")
            phase_index = PREPARATION_PHASES.index(phase)
            current = snapshot.phases[phase_index]
            _validate_prior_phases(snapshot.phases, phase_index)
            timestamp = _timestamp(self._clock(), "phase timestamp")
            updated = _transition(current, status, timestamp, detail)
            phases = list(snapshot.phases)
            phases[phase_index] = updated
            overall: PreparationStatus = snapshot.status
            selected_error = snapshot.error
            if status == "failed":
                overall = "failed"
                selected_error = error
            elif status == "cancelled":
                overall = "cancelled"
            elif phase == "running" and status in _COMPLETED_PHASE_STATUSES:
                overall = "ready"
            payload = _payload(
                PreparationSnapshot(
                    schema_version=1,
                    evaluation_id=snapshot.evaluation_id,
                    manifest_id=snapshot.manifest_id,
                    status=overall,
                    current_phase=phase,
                    phases=tuple(phases),
                    error=selected_error,
                )
            )
            atomic_write_json(self.path, payload)


def load_preparation(
    path: str | Path,
    *,
    evaluation_id: str,
    manifest_id: str,
) -> PreparationSnapshot | None:
    selected = Path(path)
    if not selected.exists():
        return None
    snapshot = _load(selected)
    return validate_preparation_identity(
        snapshot,
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
    )


def parse_preparation(
    payload: object,
    *,
    evaluation_id: str,
    manifest_id: str,
) -> PreparationSnapshot:
    """Parse an already-contained preparation payload for read-only projection."""
    return validate_preparation_identity(
        _snapshot(payload),
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
    )


def advance_preparation_from_environment(
    phase: str,
    status: PreparationPhaseStatus,
    *,
    detail: str | None = None,
    error: PreparationError | None = None,
) -> bool:
    """Advance the journal identified by private worker environment metadata."""
    names = (
        "RPENT_PREPARATION_PATH",
        "RPENT_EVALUATION_ID",
        "RPENT_MANIFEST_ID",
    )
    values = tuple(os.environ.get(name) for name in names)
    if values == (None, None, None):
        return False
    if any(value is None or not value for value in values):
        raise ValueError("incomplete preparation worker identity")
    path_value, evaluation_id, manifest_id = values
    path = Path(path_value)
    if not path.is_file():
        return False
    load_preparation(
        path,
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
    )
    PreparationProgress(path).advance(
        phase,
        status,
        detail=detail,
        error=error,
        allow_replay=True,
    )
    return True


def validate_preparation_identity(
    snapshot: PreparationSnapshot,
    *,
    evaluation_id: str,
    manifest_id: str,
) -> PreparationSnapshot:
    """Reject a journal that does not belong to the selected Evaluation."""
    if snapshot.evaluation_id != evaluation_id:
        raise ValueError("preparation evaluation_id mismatch")
    if snapshot.manifest_id != manifest_id:
        raise ValueError("preparation manifest_id mismatch")
    return snapshot


def _load(path: Path) -> PreparationSnapshot:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_raise_constant(value)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid preparation JSON: {exc}") from exc
    return _snapshot(payload)


def _snapshot(payload: object) -> PreparationSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("preparation payload must be an object")
    expected = {
        "schema_version",
        "evaluation_id",
        "manifest_id",
        "status",
        "current_phase",
        "phases",
        "error",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("invalid preparation schema")
    evaluation_id = _safe_text(payload["evaluation_id"], "evaluation_id")
    manifest_id = _safe_text(payload["manifest_id"], "manifest_id")
    status = payload["status"]
    if status not in {"preparing", "ready", "failed", "cancelled"}:
        raise ValueError("invalid overall preparation status")
    current_phase = payload["current_phase"]
    if current_phase not in PREPARATION_PHASES:
        raise ValueError("invalid current preparation phase")
    raw_phases = payload["phases"]
    if not isinstance(raw_phases, list) or [
        item.get("phase") if isinstance(item, dict) else None for item in raw_phases
    ] != list(PREPARATION_PHASES):
        raise ValueError("preparation phases must use canonical order")
    phases = tuple(_phase(item) for item in raw_phases)
    error = _error(payload["error"])
    if status == "failed" and error is None:
        raise ValueError("failed preparation requires an error")
    if status != "failed" and error is not None:
        raise ValueError("only failed preparation may contain an error")
    return PreparationSnapshot(
        schema_version=1,
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
        status=status,
        current_phase=current_phase,
        phases=phases,
        error=error,
    )


def _phase(payload: object) -> PreparationPhase:
    if not isinstance(payload, dict) or set(payload) != {
        "phase",
        "status",
        "started_at",
        "finished_at",
        "detail",
    }:
        raise ValueError("invalid preparation phase record")
    status = payload["status"]
    if status not in _PHASE_STATUSES:
        raise ValueError("invalid preparation phase status")
    started = _optional_timestamp(payload["started_at"], "started_at")
    finished = _optional_timestamp(payload["finished_at"], "finished_at")
    if status == "pending" and (started is not None or finished is not None):
        raise ValueError("pending preparation phase cannot have timestamps")
    if status == "loading" and (started is None or finished is not None):
        raise ValueError("loading preparation phase has invalid timestamps")
    if status not in {"pending", "loading"} and (started is None or finished is None):
        raise ValueError("completed preparation phase requires timestamps")
    detail = payload["detail"]
    if detail is not None:
        detail = _safe_text(detail, "detail")
    return PreparationPhase(payload["phase"], status, started, finished, detail)


def _error(payload: object) -> PreparationError | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"error_type", "message"}:
        raise ValueError("invalid preparation error")
    return PreparationError(payload["error_type"], payload["message"])


def _transition(
    current: PreparationPhase,
    status: str,
    timestamp: str,
    detail: str | None,
) -> PreparationPhase:
    if current.status == "pending" and status == "loading":
        return PreparationPhase(current.phase, status, timestamp, None, detail)
    if current.status == "pending" and status in {"reused", "skipped"}:
        return PreparationPhase(current.phase, status, timestamp, timestamp, detail)
    if current.status == "loading" and status in {"ready", "failed", "cancelled"}:
        return PreparationPhase(
            current.phase,
            status,
            current.started_at,
            timestamp,
            detail if detail is not None else current.detail,
        )
    raise ValueError(
        f"illegal preparation transition: {current.status} -> {status}"
    )


def _validate_prior_phases(phases: tuple[PreparationPhase, ...], index: int) -> None:
    incomplete = [
        item.phase
        for item in phases[:index]
        if item.status not in _COMPLETED_PHASE_STATUSES
    ]
    if incomplete:
        raise ValueError(f"prior preparation phase is incomplete: {incomplete[0]}")


def _transition_was_already_applied(
    snapshot: PreparationSnapshot,
    phase: str,
    status: PreparationPhaseStatus,
) -> bool:
    if snapshot.status in _TERMINAL_PROGRESS_STATUSES:
        return True
    selected = snapshot.phases[PREPARATION_PHASES.index(phase)]
    if selected.status == status:
        return True
    return selected.status in _COMPLETED_PHASE_STATUSES and status in {
        "loading",
        "ready",
        "reused",
        "skipped",
        "failed",
        "cancelled",
    }


def _payload(snapshot: PreparationSnapshot) -> dict[str, object]:
    return {
        "schema_version": snapshot.schema_version,
        "evaluation_id": snapshot.evaluation_id,
        "manifest_id": snapshot.manifest_id,
        "status": snapshot.status,
        "current_phase": snapshot.current_phase,
        "phases": [
            {
                "phase": item.phase,
                "status": item.status,
                "started_at": item.started_at,
                "finished_at": item.finished_at,
                "detail": item.detail,
            }
            for item in snapshot.phases
        ],
        "error": (
            None
            if snapshot.error is None
            else {
                "error_type": snapshot.error.error_type,
                "message": snapshot.error.message,
            }
        ),
    }


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    ensure_directory(path.parent)
    with path.with_name(f".{path.name}.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _optional_timestamp(value: object, name: str) -> str | None:
    return None if value is None else _timestamp(value, name)


def _timestamp(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be a UTC timestamp")
    return value


def _safe_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 4096
        or "\r" in value
        or "\n" in value
    ):
        raise ValueError(f"{name} must be safe non-empty text")
    return value


def _raise_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PREPARATION_PHASES",
    "PreparationError",
    "PreparationPhase",
    "PreparationProgress",
    "PreparationSnapshot",
    "load_preparation",
]
