"""Single post-reap transaction for accepted Episode results and projections."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rpent.evaluation.attempt_registration import AttemptRegistration
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult, ResultStore

if TYPE_CHECKING:
    from rpent.ui.schemas import CaseDetail, OperationPage

NoWorkerReason = Literal[
    "launch_failed",
    "cancelled_before_launch",
    "startup_recovery",
    "already_accepted_repair",
]
_NO_WORKER_REASONS = frozenset(
    {
        "launch_failed",
        "cancelled_before_launch",
        "startup_recovery",
        "already_accepted_repair",
    }
)


@dataclass(frozen=True, slots=True)
class ReapedWorker:
    """Proof that the coordinator completed process lifecycle ownership."""

    scope_id: str
    exit_status: int

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise ValueError("ReapedWorker scope_id must be non-empty")
        if isinstance(self.exit_status, bool) or not isinstance(self.exit_status, int):
            raise ValueError("ReapedWorker exit_status must be an integer")


@dataclass(frozen=True, slots=True)
class NoWorker:
    """Closed proof for result paths that never owned a live process."""

    reason: NoWorkerReason

    def __post_init__(self) -> None:
        if self.reason not in _NO_WORKER_REASONS:
            raise ValueError("invalid NoWorker reason")


@dataclass(frozen=True, slots=True)
class FinalizedResultCandidate:
    """One immutable result paired with process-completion evidence."""

    result: EpisodeResult
    completion: ReapedWorker | NoWorker


class ResultAcceptance:
    """Commit one result, assemble its snapshot, and publish terminal display state."""

    def __init__(
        self,
        *,
        manifest: EvaluationManifest,
        store: ResultStore,
        runtime_store: Any,
        assembler: Callable[[EpisodeResult], CaseDetail],
        operation_reconstructor: Callable[[EpisodeResult], OperationPage] | None = None,
        snapshot_writer: Callable[[Path, object], None] | None = None,
        accepted_observer: Callable[[EpisodeResult, CaseDetail], object] | None = None,
        update_publisher: Callable[[object], object] | None = None,
    ) -> None:
        if not isinstance(manifest, EvaluationManifest):
            raise TypeError("manifest must be an EvaluationManifest")
        if not isinstance(store, ResultStore):
            raise TypeError("store must be a ResultStore")
        if not callable(assembler):
            raise TypeError("assembler must be callable")
        if operation_reconstructor is not None and not callable(
            operation_reconstructor
        ):
            raise TypeError("operation_reconstructor must be callable")
        if snapshot_writer is not None and not callable(snapshot_writer):
            raise TypeError("snapshot_writer must be callable")
        if accepted_observer is not None and not callable(accepted_observer):
            raise TypeError("accepted_observer must be callable")
        if update_publisher is not None and not callable(update_publisher):
            raise TypeError("update_publisher must be callable")
        self.manifest = manifest
        self.store = store
        self.runtime_store = runtime_store
        self.assembler = assembler
        self.operation_reconstructor = operation_reconstructor
        self.snapshot_writer = snapshot_writer
        self.accepted_observer = accepted_observer
        self.update_publisher = update_publisher
        self._locks_guard = threading.Lock()
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._diagnostic_lock = threading.Lock()
        self._accepted = 0
        self._identical = 0
        self._assembly_failures = 0
        self._snapshot_write_failures = 0
        self._proof_rejections = 0

    def accept(self, candidate: FinalizedResultCandidate) -> EpisodeResult:
        """Accept only a closed result path and never weaken ledger authority."""
        try:
            result, completion = self._validate_candidate(candidate)
        except (TypeError, ValueError):
            self._increment("_proof_rejections")
            raise
        case_lock = self._case_lock((result.evaluation_id, result.case_id))
        with case_lock:
            repairing_accepted = (
                isinstance(completion, NoWorker)
                and completion.reason == "already_accepted_repair"
            )
            try:
                disposition = self.store.append_or_confirm(
                    result, require_existing=repairing_accepted
                )
            except ValueError:
                if repairing_accepted:
                    self._increment("_proof_rejections")
                raise
            self._ensure_runtime_entry(result)
            pending = self.runtime_store.mark_terminal_pending(result)
            self._publish_update(pending)
            entry = self.runtime_store.get_entry(
                result.evaluation_id, result.case_id, result.attempt_id
            )
            if entry is None:
                raise RuntimeError("terminal-pending runtime entry disappeared")
            try:
                detail = self.assembler(result)
                self._validate_detail(detail, result)
                from rpent.ui.case_snapshot import encode_case_snapshot
                from rpent.ui.operation_snapshot import encode_operation_snapshot

                payload = encode_case_snapshot(detail)
                if repairing_accepted and entry.phase == "sealed":
                    operation_page = self.runtime_store.get_operations(
                        result.evaluation_id,
                        result.case_id,
                        result.attempt_id,
                    )
                elif self.operation_reconstructor is not None and (
                    repairing_accepted
                    or entry.last_sequence != detail.last_sequence
                ):
                    operation_page = self.operation_reconstructor(result)
                else:
                    operation_page = self.runtime_store.finalize_operations(detail)
                if (
                    operation_page.last_sequence != detail.last_sequence
                    or operation_page.incomplete_tail
                ):
                    raise ValueError("operation snapshot sequence mismatch")
                operation_payload = encode_operation_snapshot(
                    operation_page,
                    self.manifest.manifest_id,
                )
            except Exception:
                self._increment("_assembly_failures")
                unavailable = self.runtime_store.mark_unavailable(
                    (result.evaluation_id, result.case_id, result.attempt_id),
                    entry.attempt_epoch,
                    "terminal_snapshot_assembly_failed",
                )
                self._publish_update(unavailable)
                raise
            try:
                if self.snapshot_writer is None:
                    from rpent.recording.media import atomic_write_json

                    writer = atomic_write_json
                else:
                    writer = self.snapshot_writer
                writer(Path(result.artifact_dir) / "case_detail.json", payload)
                writer(
                    Path(result.artifact_dir) / "operations.json",
                    operation_payload,
                )
            except OSError:
                self._increment("_snapshot_write_failures")
                note_failure = getattr(
                    self.runtime_store, "note_snapshot_write_failure", None
                )
                if callable(note_failure):
                    note_failure()
            terminal = self.runtime_store.publish_terminal(
                detail, result, operation_page
            )
            self._publish_update(terminal)
            if self.accepted_observer is not None:
                self.accepted_observer(result, detail)
            self._increment("_accepted")
            if disposition == "identical":
                self._increment("_identical")
            return result

    def snapshot(self) -> dict[str, int]:
        with self._diagnostic_lock:
            return {
                "accepted": self._accepted,
                "identical": self._identical,
                "assembly_failures": self._assembly_failures,
                "snapshot_write_failures": self._snapshot_write_failures,
                "proof_rejections": self._proof_rejections,
            }

    def _validate_candidate(
        self, candidate: FinalizedResultCandidate
    ) -> tuple[EpisodeResult, ReapedWorker | NoWorker]:
        if not isinstance(candidate, FinalizedResultCandidate):
            raise TypeError("candidate must be a FinalizedResultCandidate")
        result = candidate.result
        completion = candidate.completion
        if not isinstance(result, EpisodeResult):
            raise TypeError("candidate result must be an EpisodeResult")
        if not isinstance(completion, (ReapedWorker, NoWorker)):
            raise TypeError("candidate completion proof is invalid")
        expected_scope = result.task_batch_attempt_id or result.attempt_id
        if isinstance(completion, ReapedWorker):
            if completion.scope_id != expected_scope:
                raise ValueError("completion proof scope mismatch")
        elif completion.reason == "launch_failed":
            if (
                result.status != "error"
                or result.worker_pid is not None
                or result.worker_exit_code is not None
            ):
                raise ValueError("launch_failed proof does not match result")
        elif completion.reason == "cancelled_before_launch" and (
            result.status != "cancelled" or result.worker_pid is not None
        ):
            raise ValueError("cancelled_before_launch proof does not match result")
        self._validate_artifact_root(result)
        return result, completion

    def _validate_artifact_root(self, result: EpisodeResult) -> None:
        root_value = self.manifest.runtime.get("evaluation_root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("manifest evaluation_root is unavailable")
        expected = (
            Path(root_value).resolve()
            / "cases"
            / result.case_id
            / "attempts"
            / result.attempt_id
        )
        if Path(result.artifact_dir).resolve() != expected:
            raise ValueError("accepted result artifact root mismatch")

    def _ensure_runtime_entry(self, result: EpisodeResult) -> None:
        if (
            self.runtime_store.get_entry(
                result.evaluation_id, result.case_id, result.attempt_id
            )
            is not None
        ):
            return
        current = self.runtime_store.get_entry(result.evaluation_id, result.case_id)
        if current is not None and current.phase not in {"sealed", "unavailable"}:
            raise ValueError("accepted result Attempt is not current")
        case = next(
            (item for item in self.manifest.cases if item.case_id == result.case_id),
            None,
        )
        if case is None:
            raise ValueError("accepted result references an unknown Case")
        self.runtime_store.start_attempt(
            AttemptRegistration(
                1,
                result.evaluation_id,
                result.manifest_id,
                result.case_id,
                result.attempt_id,
                result.started_at,
            ),
            self.manifest,
            case,
        )

    @staticmethod
    def _validate_detail(detail: CaseDetail, result: EpisodeResult) -> None:
        from rpent.ui.schemas import CaseDetail

        if not isinstance(detail, CaseDetail):
            raise TypeError("assembler must return CaseDetail")
        if (
            detail.evaluation_id != result.evaluation_id
            or detail.manifest_id != result.manifest_id
            or detail.case_id != result.case_id
            or detail.attempt_id != result.attempt_id
            or detail.episode_result != result.to_payload()
            or detail.execution_state
            != ("cancelled" if result.status == "cancelled" else "terminal")
            or isinstance(detail.last_sequence, bool)
            or not isinstance(detail.last_sequence, int)
            or detail.last_sequence < 0
        ):
            raise ValueError("assembled Case detail does not match accepted result")

    def _case_lock(self, key: tuple[str, str]) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _increment(self, name: str) -> None:
        with self._diagnostic_lock:
            setattr(self, name, getattr(self, name) + 1)

    def _publish_update(self, update: object | None) -> None:
        if update is not None and self.update_publisher is not None:
            self.update_publisher(update)


__all__ = [
    "FinalizedResultCandidate",
    "NoWorker",
    "NoWorkerReason",
    "ReapedWorker",
    "ResultAcceptance",
]
