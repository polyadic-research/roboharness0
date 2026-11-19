"""Safe filesystem projections over retained Evaluation artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, TypeVar
from urllib.parse import quote

from rpent.evaluation.attempt_registration import (
    AttemptRegistration,
    load_case_registration,
    load_current_attempt,
)
from rpent.evaluation.batches import (
    load_task_batch_attempts,
    validate_task_batch_attempt_id,
    validate_task_batch_result,
)
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import (
    EVALUATION_MANIFEST_SCHEMA_VERSION,
    EvaluationManifest,
)
from rpent.evaluation.official_metrics import OfficialMetrics
from rpent.evaluation.preparation_progress import parse_preparation
from rpent.evaluation.results import EpisodeResult, aggregate_results
from rpent.evaluation.startup_log import read_startup_events, render_startup_console
from rpent.recording import media
from rpent.recording.contracts import (
    JSON_SAFE_INTEGER_MAX,
    RECORDING_SCHEMA_VERSION,
    ArtifactHealth,
    CameraDescriptor,
    EventEnvelope,
    FrameReference,
    RecordingProfile,
    TransitionRecord,
)
from rpent.recording.data_schema import build_data_schema
from rpent.ui.console import (
    open_console_download as open_console_file,
)
from rpent.ui.console import (
    read_console_page,
    read_console_text,
)
from rpent.ui.json_safety import require_bounded_json_nesting
from rpent.ui.operation_projection import OperationProjection
from rpent.ui.projection_summary import aggregate_evaluation, build_attempt_summary
from rpent.ui.schemas import (
    ArtifactDescriptor,
    ArtifactHealthView,
    AttemptListItemView,
    AttemptSummaryView,
    CameraDescriptorView,
    CaseDataSchemaView,
    CaseDetail,
    CaseSummary,
    ConsolePageView,
    EvaluationDetail,
    EvaluationListItem,
    EvaluationMetricsView,
    EvaluationSummaryView,
    EventPage,
    ExecutionState,
    OfficialMetricsView,
    OperationPage,
    PreparationView,
    QualificationCheckView,
    QualificationView,
    RecordingProfileView,
    RPentDiagnosticsView,
    SubmissionView,
    TimelineWindow,
)
from rpent.ui.timeline_index import RetainedTimelineIndex

_T = TypeVar("_T")
_CacheKey = tuple[Path, int, int, int, int]
_EventGeneration = tuple[int, int, int, int, int]
_EventOffsetKey = tuple[Path, int, int, int, int, int, int]
_MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_RECORDING_DECLARATION_BYTES = 256 * 1024
_MAX_MEDIA_BYTES = 8 * 1024 * 1024
_INDEX_FILENAME = ".rpent-evaluations.json"
_INDEX_LOCK_FILENAME = ".rpent-evaluations.lock"
_INDEX_SCHEMA_VERSION = 3
_RECOVERED_LIST_STATES = frozenset({"queued", "starting", "running", "finalizing"})
_MAX_LEGACY_ATTEMPT_SCAN = 4_096


@contextmanager
def _locked_index(boundary: Path, *, exclusive: bool) -> Iterator[None]:
    media.ensure_directory(boundary)
    fd = os.open(boundary / _INDEX_LOCK_FILENAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass(frozen=True, slots=True)
class EvaluationMetadata:
    evaluation_id: str
    manifest_id: str
    case_ids: frozenset[str]
    allow_selective_case_rerun: bool


@dataclass(frozen=True, slots=True)
class CurrentAttemptIdentity:
    evaluation_id: str
    manifest_id: str
    case_id: str
    attempt_id: str | None


class CurrentAttemptUnavailable(RuntimeError):
    """The authoritative current-Attempt locator cannot be validated."""


@dataclass(frozen=True, slots=True)
class MediaBytes:
    data: bytes
    media_type: str
    etag: str
    size: int


def _initial_evaluation_list_item(
    manifest: EvaluationManifest,
    execution_state: ExecutionState,
) -> EvaluationListItem:
    scheduled = len(manifest.cases)
    return EvaluationListItem(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        created_at=manifest.created_at,
        environment=str(manifest.environment.get("name", "unknown")),
        task_selector=manifest.task_selector,
        execution_state=execution_state,
        scheduled=scheduled,
        completed=0,
        metrics=EvaluationMetricsView(scheduled=scheduled),
    )


def _validate_persistent_list_item(
    item: EvaluationListItem,
    manifest: EvaluationManifest,
) -> None:
    expected = {
        "created_at": manifest.created_at,
        "environment": str(manifest.environment.get("name", "unknown")),
        "task_selector": manifest.task_selector,
        "scheduled": len(manifest.cases),
    }
    for name, value in expected.items():
        if getattr(item, name) != value:
            raise ValueError(f"list item {name} mismatch")
    metrics = item.metrics
    if (
        item.completed < 0
        or item.completed > item.scheduled
        or metrics.scheduled != item.scheduled
        or metrics.completed != item.completed
        or metrics.successful > item.completed
        or metrics.running + item.completed > item.scheduled
    ):
        raise ValueError("invalid list item counts")


def _contract_shape(contracts: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = contracts.get(key)
    shape = value.get("shape") if isinstance(value, Mapping) else None
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) > 1
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in shape
        )
    ):
        return ()
    return tuple(shape)


def _empty_console_page(*, offset: int, complete: bool) -> ConsolePageView:
    return ConsolePageView(
        offset=0,
        next_offset=0,
        text="",
        truncated=offset > 0,
        complete=complete,
    )


def _startup_console_text(root: Path, case_id: str) -> str:
    events = tuple(
        item
        for item in read_startup_events(root / "startup.jsonl")
        if item.case_id is None or item.case_id == case_id
    )
    rendered = render_startup_console(events)
    return f"{rendered}\n" if rendered else ""


class EvaluationRepository:
    """Project retained files below explicitly configured roots."""

    def __init__(
        self,
        roots: tuple[str | Path, ...],
        *,
        max_jsonl_bytes: int = 64 * 1024 * 1024,
        max_jsonl_records: int = 100_000,
        max_event_page_size: int = 1_000,
        max_event_record_bytes: int = 1024 * 1024,
        max_event_tail_bytes: int = 4 * 1024 * 1024,
        event_checkpoint_capacity: int = 4_096,
        timeline_index_capacity: int = 4_096,
        timeline_cache_capacity: int = 32,
        max_timeline_seek_records: int = 8_192,
        max_timeline_seek_bytes: int = 8 * 1024 * 1024,
        max_timeline_response_bytes: int = 16 * 1024,
    ) -> None:
        for name, value in (
            ("max_jsonl_bytes", max_jsonl_bytes),
            ("max_jsonl_records", max_jsonl_records),
            ("max_event_page_size", max_event_page_size),
            ("max_event_record_bytes", max_event_record_bytes),
            ("max_event_tail_bytes", max_event_tail_bytes),
            ("event_checkpoint_capacity", event_checkpoint_capacity),
            ("timeline_index_capacity", timeline_index_capacity),
            ("timeline_cache_capacity", timeline_cache_capacity),
            ("max_timeline_seek_records", max_timeline_seek_records),
            ("max_timeline_seek_bytes", max_timeline_seek_bytes),
            ("max_timeline_response_bytes", max_timeline_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.max_jsonl_bytes = max_jsonl_bytes
        self.max_jsonl_records = max_jsonl_records
        self.max_event_page_size = max_event_page_size
        self.max_event_record_bytes = max_event_record_bytes
        self.max_event_tail_bytes = max_event_tail_bytes
        self.event_checkpoint_capacity = event_checkpoint_capacity
        self.max_timeline_response_bytes = max_timeline_response_bytes
        self._index_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._index: Mapping[str, tuple[Path, EvaluationManifest]] = MappingProxyType(
            {}
        )
        self._index_owners: Mapping[str, Path] = MappingProxyType({})
        self._list_items: Mapping[str, EvaluationListItem] = MappingProxyType({})
        self._cache_lock = threading.Lock()
        self._cache: dict[_CacheKey, Any] = {}
        self._runtime_metrics_cache_lock = threading.Lock()
        self._runtime_metrics_cache: dict[
            _CacheKey, tuple[float | None, int | None, bool, int]
        ] = {}
        self._event_offsets_lock = threading.Lock()
        self._event_offsets: OrderedDict[_EventOffsetKey, int] = OrderedDict()
        self._timeline_index = RetainedTimelineIndex(
            max_records=max_jsonl_records,
            max_bytes=max_jsonl_bytes,
            max_record_bytes=max_event_record_bytes,
            index_capacity=timeline_index_capacity,
            cache_capacity=timeline_cache_capacity,
            max_seek_records=max_timeline_seek_records,
            max_seek_bytes=max_timeline_seek_bytes,
        )
        self._legacy_active_identities: dict[tuple[str, str], str] = {}
        self._legacy_identity_unavailable: set[tuple[str, str]] = set()
        self._result_index_lock = threading.Lock()
        self._result_index: Mapping[tuple[str, str], tuple[EpisodeResult, ...]] = (
            MappingProxyType({})
        )
        self._attempt_summary_index: Mapping[
            tuple[str, str], tuple[AttemptListItemView, ...]
        ] = MappingProxyType({})
        self._attempt_summary_totals: Mapping[tuple[str, str], int] = MappingProxyType(
            {}
        )
        self._accepted_final_sequences: Mapping[tuple[str, str, str], int] = (
            MappingProxyType({})
        )
        self._result_index_unavailable: set[str] = set()
        self._runtime_store: Any | None = None
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_metrics = {
            "missing_locators": 0,
            "invalid_locators": 0,
            "stale_locators": 0,
            "legacy_scan_limit_failures": 0,
            "degraded_summaries": 0,
        }
        self.refresh_index(force=True, recover=False)
        self._build_startup_result_index()
        self.prime_startup_identities()
        self.refresh_index(force=True)

    def list_evaluations(self) -> tuple[EvaluationListItem, ...]:
        with self._index_lock:
            items = self._list_items
        return tuple(item for _, item in sorted(items.items()))

    def index_fingerprint(self) -> Sequence[tuple[int, int, int, int] | None]:
        values = []
        for root in self.roots:
            try:
                info = (root / _INDEX_FILENAME).stat()
            except FileNotFoundError:
                values.append(None)
            else:
                values.append(
                    (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                )
        return tuple(values)

    def bind_runtime_store(self, runtime_store: Any) -> None:
        case_summary = getattr(runtime_store, "case_summary", None)
        if not callable(case_summary):
            raise TypeError("runtime_store must expose case_summary")
        self._runtime_store = runtime_store
        configure_operations = getattr(
            runtime_store, "configure_operation_snapshots", None
        )
        if callable(configure_operations):
            configure_operations(
                loader=self.load_terminal_operation_snapshot,
                reconstructor=self.reconstruct_operations,
                sequence_reader=self.terminal_operation_sequence,
                persister=self.persist_terminal_operation_snapshot,
            )

    def snapshot(self) -> dict[str, int]:
        with self._diagnostic_lock:
            metrics = dict(self._diagnostic_metrics)
        with self._result_index_lock:
            summary_entries = sum(
                len(items) for items in self._attempt_summary_index.values()
            )
            unavailable_ledgers = len(self._result_index_unavailable)
        return {
            **metrics,
            "attempt_summary_entries": summary_entries,
            "unavailable_result_ledgers": unavailable_ledgers,
        }

    def refresh_index(self, *, force: bool = False, recover: bool = True) -> None:
        if not self._refresh_lock.acquire(blocking=force):
            return
        try:
            discovered, owners, list_items = (
                self._discover() if recover else self._discover(recover=False)
            )
            self._publish_discovery(discovered, owners, list_items)
        finally:
            self._refresh_lock.release()

    def register_evaluation(
        self,
        root: str | Path,
        manifest: EvaluationManifest,
        *,
        execution_state: ExecutionState | None = None,
    ) -> None:
        if not isinstance(manifest, EvaluationManifest):
            raise TypeError("manifest must be an EvaluationManifest")
        resolved = Path(root).resolve()
        boundary = max(
            (item for item in self.roots if resolved.is_relative_to(item)),
            key=lambda item: len(item.parts),
            default=self.roots[0],
        )
        with self._refresh_lock:
            with self._index_lock:
                boundary = self._index_owners.get(manifest.evaluation_id, boundary)
            accepted_results: tuple[EpisodeResult, ...] = ()
            accepted_summaries: dict[str, tuple[AttemptListItemView, ...]] = {}
            accepted_final_sequences: dict[tuple[str, str, str], int] = {}
            with _locked_index(boundary, exclusive=True):
                boundary_index, boundary_owners, boundary_items, dirty = (
                    self._discover_boundary(boundary, recover=True)
                )
                indexed, owners, list_items = self._replace_boundary_discovery(
                    boundary,
                    boundary_index,
                    boundary_owners,
                    boundary_items,
                )
                is_new = manifest.evaluation_id not in indexed
                updated_index, updated_owners, updated_items = (
                    self._register_boundary_entry(
                        boundary=boundary,
                        root=resolved,
                        manifest=manifest,
                        execution_state=execution_state,
                        indexed=indexed,
                        owners=owners,
                        list_items=list_items,
                    )
                )
                needs_write = (
                    dirty
                    or updated_index != indexed
                    or updated_owners != owners
                    or updated_items != list_items
                )
                if is_new:
                    accepted_results = self._results(resolved, manifest)
                    for case in manifest.cases:
                        case_summaries, final_sequences = (
                            self._build_attempt_indexes(
                                resolved,
                                manifest,
                                case.case_id,
                                tuple(
                                    result
                                    for result in accepted_results
                                    if result.case_id == case.case_id
                                ),
                            )
                        )
                        accepted_summaries[case.case_id] = case_summaries
                        accepted_final_sequences.update(final_sequences)
                if needs_write:
                    self._write_boundary_index(
                        boundary,
                        updated_index,
                        updated_owners,
                        updated_items,
                    )
            self._publish_discovery(
                updated_index,
                updated_owners,
                updated_items,
            )
            with self._result_index_lock:
                result_index = dict(self._result_index)
                summary_index = dict(self._attempt_summary_index)
                summary_totals = dict(self._attempt_summary_totals)
                final_sequences = dict(self._accepted_final_sequences)
                for case in manifest.cases:
                    case_results = tuple(
                        result
                        for result in accepted_results
                        if result.case_id == case.case_id
                    )
                    result_index.setdefault(
                        (manifest.evaluation_id, case.case_id),
                        case_results,
                    )
                    summary_index.setdefault(
                        (manifest.evaluation_id, case.case_id),
                        accepted_summaries.get(case.case_id, ()),
                    )
                    summary_totals.setdefault(
                        (manifest.evaluation_id, case.case_id), len(case_results)
                    )
                final_sequences.update(accepted_final_sequences)
                self._result_index = MappingProxyType(result_index)
                self._attempt_summary_index = MappingProxyType(summary_index)
                self._attempt_summary_totals = MappingProxyType(summary_totals)
                self._accepted_final_sequences = MappingProxyType(final_sequences)

    def update_evaluation_list_item(
        self,
        evaluation_id: str,
        *,
        execution_state: ExecutionState | None = None,
    ) -> EvaluationListItem:
        with self._refresh_lock:
            with self._index_lock:
                boundary = self._index_owners.get(evaluation_id)
            if boundary is None:
                raise KeyError(evaluation_id)
            with _locked_index(boundary, exclusive=True):
                boundary_index, boundary_owners, boundary_items, _ = (
                    self._discover_boundary(boundary, recover=True)
                )
                indexed, owners, list_items = self._replace_boundary_discovery(
                    boundary,
                    boundary_index,
                    boundary_owners,
                    boundary_items,
                )
                found = indexed.get(evaluation_id)
                if found is None:
                    raise KeyError(evaluation_id)
                root, manifest = found
                item = self._evaluation_item(root, manifest)
                if execution_state is not None:
                    item = EvaluationListItem.model_validate(
                        item.model_dump() | {"execution_state": execution_state}
                    )
                list_items[evaluation_id] = item
                self._write_boundary_index(
                    boundary,
                    indexed,
                    owners,
                    list_items,
                )
            self._publish_discovery(indexed, owners, list_items)
            return item

    def _register_boundary_entry(
        self,
        *,
        boundary: Path,
        root: Path,
        manifest: EvaluationManifest,
        execution_state: ExecutionState | None,
        indexed: dict[str, tuple[Path, EvaluationManifest]],
        owners: dict[str, Path],
        list_items: dict[str, EvaluationListItem],
    ) -> tuple[
        dict[str, tuple[Path, EvaluationManifest]],
        dict[str, Path],
        dict[str, EvaluationListItem],
    ]:
        existing = indexed.get(manifest.evaluation_id)
        if existing is not None:
            if existing[0] != root or existing[1].manifest_id != manifest.manifest_id:
                raise ValueError(f"duplicate evaluation_id {manifest.evaluation_id!r}")
            if execution_state is None:
                return indexed, owners, list_items
        else:
            if any(indexed_root == root for indexed_root, _ in indexed.values()):
                raise ValueError("evaluation root is already registered")
            indexed = indexed | {manifest.evaluation_id: (root, manifest)}
            owners = owners | {manifest.evaluation_id: boundary}
        item = (
            self._evaluation_item(root, manifest)
            if execution_state is None
            else _initial_evaluation_list_item(manifest, execution_state)
        )
        if list_items.get(manifest.evaluation_id) == item:
            return indexed, owners, list_items
        return indexed, owners, list_items | {manifest.evaluation_id: item}

    def _write_boundary_index(
        self,
        boundary: Path,
        indexed: Mapping[str, tuple[Path, EvaluationManifest]],
        owners: Mapping[str, Path],
        list_items: Mapping[str, EvaluationListItem],
    ) -> None:
        payload = _persistent_index_payload(boundary, indexed, owners, list_items)
        try:
            existing = self._cached(
                boundary / _INDEX_FILENAME,
                boundary,
                json.loads,
                max_bytes=_MAX_JSON_BYTES,
            )
        except FileNotFoundError:
            existing = None
        if existing == payload:
            return
        media.atomic_write_json(
            boundary / _INDEX_FILENAME,
            payload,
        )

    def _publish_discovery(
        self,
        indexed: dict[str, tuple[Path, EvaluationManifest]],
        owners: dict[str, Path],
        list_items: dict[str, EvaluationListItem],
    ) -> None:
        with self._index_lock:
            self._index = MappingProxyType(indexed)
            self._index_owners = MappingProxyType(owners)
            self._list_items = MappingProxyType(list_items)

    def evaluation_metadata(self, evaluation_id: str) -> EvaluationMetadata:
        _, manifest = self._lookup(evaluation_id)
        return EvaluationMetadata(
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            case_ids=frozenset(case.case_id for case in manifest.cases),
            allow_selective_case_rerun=(
                manifest.qualification.get("allow_selective_case_rerun") is not False
            ),
        )

    def load_runtime_registration(
        self,
        evaluation_id: str,
        manifest_id: str,
        case_id: str,
        attempt_id: str,
        *,
        require_current: bool = True,
    ) -> tuple[AttemptRegistration, EvaluationManifest, EpisodeCase]:
        """Load exactly one durable Attempt registration without discovery."""
        root, manifest = self._lookup(evaluation_id)
        if manifest.manifest_id != manifest_id:
            raise ValueError("lifecycle manifest identity mismatch")
        case = next((item for item in manifest.cases if item.case_id == case_id), None)
        if case is None:
            raise KeyError((evaluation_id, case_id))
        try:
            registration = load_case_registration(
                self._attempt_root(root, case_id, attempt_id) / "case.json",
                manifest=manifest,
                case=case,
                attempt_id=attempt_id,
            )
            if require_current:
                current = load_current_attempt(
                    root / "cases" / case_id / "current_attempt.json",
                    evaluation_id=evaluation_id,
                    manifest_id=manifest_id,
                    case_id=case_id,
                )
                if current != registration:
                    raise ValueError("lifecycle Attempt identity mismatch")
        except FileNotFoundError as exc:
            raise ValueError("lifecycle Attempt registration is unavailable") from exc
        except OSError as exc:
            raise ValueError("lifecycle Attempt registration is unavailable") from exc
        return registration, manifest, case

    def current_attempt_identity(
        self, evaluation_id: str, case_id: str
    ) -> CurrentAttemptIdentity:
        root, manifest = self._lookup(evaluation_id)
        case = next((item for item in manifest.cases if item.case_id == case_id), None)
        if case is None:
            raise KeyError((evaluation_id, case_id))
        case_results = self._indexed_case_results(root, manifest, case_id)
        latest_result = case_results[-1] if case_results else None
        active_attempt = self._selected_current_attempt(
            root, manifest, case, latest_result
        )
        return CurrentAttemptIdentity(
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            case_id=case.case_id,
            attempt_id=(
                active_attempt
                if active_attempt is not None
                else latest_result.attempt_id
                if latest_result is not None
                else None
            ),
        )

    def authorize_attempt(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
    ) -> CurrentAttemptIdentity:
        """Authorize an Attempt from retained results or current identity only."""
        root, manifest = self._lookup(evaluation_id)
        case = next((item for item in manifest.cases if item.case_id == case_id), None)
        if case is None:
            raise KeyError((evaluation_id, case_id))
        case_results = self._indexed_case_results(root, manifest, case_id)
        if attempt_id is not None and any(
            result.attempt_id == attempt_id for result in case_results
        ):
            return CurrentAttemptIdentity(
                evaluation_id=manifest.evaluation_id,
                manifest_id=manifest.manifest_id,
                case_id=case.case_id,
                attempt_id=attempt_id,
            )
        selected = self.current_attempt_identity(evaluation_id, case_id)
        requested = selected.attempt_id if attempt_id is None else attempt_id
        if requested is None:
            return selected
        if requested == selected.attempt_id or any(
            result.attempt_id == requested for result in case_results
        ):
            return CurrentAttemptIdentity(
                evaluation_id=manifest.evaluation_id,
                manifest_id=manifest.manifest_id,
                case_id=case.case_id,
                attempt_id=requested,
            )
        raise KeyError((evaluation_id, case_id, requested))

    def prime_startup_identities(self) -> None:
        """Perform the one bounded migration scan allowed during startup."""
        with self._index_lock:
            indexed = tuple(
                (*value, self._list_items[evaluation_id].execution_state)
                for evaluation_id, value in self._index.items()
            )
        for root, manifest, execution_state in indexed:
            if execution_state not in _RECOVERED_LIST_STATES:
                continue
            if manifest.evaluation_id in self._result_index_unavailable:
                for case in manifest.cases:
                    self._legacy_identity_unavailable.add(
                        (manifest.evaluation_id, case.case_id)
                    )
                continue
            results = tuple(
                item
                for case in manifest.cases
                for item in self._indexed_case_results(root, manifest, case.case_id)
            )
            by_case: dict[str, list[EpisodeResult]] = {}
            for result in results:
                by_case.setdefault(result.case_id, []).append(result)
            for case in manifest.cases:
                key = (manifest.evaluation_id, case.case_id)
                locator = root / "cases" / case.case_id / "current_attempt.json"
                if locator.exists():
                    continue
                self._increment_diagnostic("missing_locators")
                case_results = tuple(by_case.get(case.case_id, ()))
                try:
                    selected = self._legacy_active_attempt_scan(
                        root,
                        manifest,
                        case,
                        {result.attempt_id for result in case_results},
                        case_results[-1] if case_results else None,
                    )
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    CurrentAttemptUnavailable,
                ) as exc:
                    if "4096" in str(exc) or "4097" in str(exc):
                        self._increment_diagnostic("legacy_scan_limit_failures")
                    else:
                        self._increment_diagnostic("invalid_locators")
                    self._legacy_active_identities.pop(key, None)
                    self._legacy_identity_unavailable.add(key)
                    continue
                if selected is not None:
                    self._legacy_active_identities[key] = selected

    def startup_runtime_registrations(
        self,
    ) -> tuple[tuple[AttemptRegistration, EvaluationManifest, EpisodeCase], ...]:
        """Return known active registrations without discovering Attempt paths."""
        registrations, _unresolved = self.startup_runtime_registration_snapshot()
        return registrations

    def startup_runtime_registration_snapshot(
        self,
    ) -> tuple[
        tuple[tuple[AttemptRegistration, EvaluationManifest, EpisodeCase], ...],
        bool,
    ]:
        """Return registrations and whether active known locator paths need retry."""
        with self._index_lock:
            indexed = tuple(
                (*value, self._list_items[evaluation_id].execution_state)
                for evaluation_id, value in self._index.items()
            )
        registrations: list[
            tuple[AttemptRegistration, EvaluationManifest, EpisodeCase]
        ] = []
        unresolved = False
        for root, manifest, execution_state in indexed:
            if execution_state not in _RECOVERED_LIST_STATES:
                continue
            if manifest.evaluation_id in self._result_index_unavailable:
                unresolved = True
                continue
            accepted = {
                (case.case_id, result.attempt_id)
                for case in manifest.cases
                for result in self._indexed_case_results(root, manifest, case.case_id)
            }
            for case in manifest.cases:
                try:
                    identity = self.current_attempt_identity(
                        manifest.evaluation_id, case.case_id
                    )
                except (KeyError, CurrentAttemptUnavailable):
                    self._increment_diagnostic("invalid_locators")
                    unresolved = True
                    continue
                attempt_id = identity.attempt_id
                if attempt_id is None:
                    if not any(
                        accepted_case_id == case.case_id
                        for accepted_case_id, _attempt_id in accepted
                    ):
                        unresolved = True
                    continue
                if (case.case_id, attempt_id) in accepted:
                    unresolved = True
                    continue
                locator = root / "cases" / case.case_id / "current_attempt.json"
                try:
                    registration = load_current_attempt(
                        locator,
                        evaluation_id=manifest.evaluation_id,
                        manifest_id=manifest.manifest_id,
                        case_id=case.case_id,
                    )
                except FileNotFoundError:
                    projection = self._optional_json(
                        self._attempt_root(root, case.case_id, attempt_id)
                        / "case.json",
                        root,
                    )
                    if projection is None or not isinstance(
                        projection.get("started_at"), str
                    ):
                        unresolved = True
                        continue
                    registration = AttemptRegistration(
                        1,
                        manifest.evaluation_id,
                        manifest.manifest_id,
                        case.case_id,
                        attempt_id,
                        projection["started_at"],
                    )
                except (OSError, TypeError, ValueError):
                    unresolved = True
                    continue
                registrations.append((registration, manifest, case))
        return tuple(registrations), unresolved

    def _build_startup_result_index(self) -> None:
        with self._index_lock:
            indexed = tuple(self._index.values())
        grouped: dict[tuple[str, str], tuple[EpisodeResult, ...]] = {}
        summaries: dict[tuple[str, str], tuple[AttemptListItemView, ...]] = {}
        totals: dict[tuple[str, str], int] = {}
        final_sequences: dict[tuple[str, str, str], int] = {}
        unavailable: set[str] = set()
        for root, manifest in indexed:
            try:
                results = self._results(root, manifest)
            except (OSError, ValueError):
                results = ()
                unavailable.add(manifest.evaluation_id)
            for case in manifest.cases:
                key = (manifest.evaluation_id, case.case_id)
                case_results = tuple(
                    item for item in results if item.case_id == case.case_id
                )
                grouped[key] = case_results
                summaries[key], case_final_sequences = self._build_attempt_indexes(
                    root, manifest, case.case_id, case_results
                )
                final_sequences.update(case_final_sequences)
                totals[key] = len(case_results)
        with self._result_index_lock:
            self._result_index = MappingProxyType(grouped)
            self._attempt_summary_index = MappingProxyType(summaries)
            self._attempt_summary_totals = MappingProxyType(totals)
            self._accepted_final_sequences = MappingProxyType(final_sequences)
            self._result_index_unavailable = unavailable

    def note_accepted_result(
        self, result: EpisodeResult, detail: CaseDetail | None = None
    ) -> None:
        """Advance the bounded startup result authority after acceptance."""
        if not isinstance(result, EpisodeResult):
            raise TypeError("result must be an EpisodeResult")
        key = (result.evaluation_id, result.case_id)
        with self._result_index_lock:
            indexed = dict(self._result_index)
            summaries = dict(self._attempt_summary_index)
            totals = dict(self._attempt_summary_totals)
            retained = indexed.get(key, ())
            matching = next(
                (item for item in retained if item.attempt_id == result.attempt_id),
                None,
            )
            if matching is not None and matching != result:
                raise ValueError("accepted result identity drift")
            if matching is None:
                indexed[key] = (*retained, result)
                totals[key] = len(retained) + 1
            summary = (
                detail.run_summary
                if detail is not None and detail.run_summary is not None
                else build_attempt_summary(
                    result.to_payload(),
                    artifact_health=ArtifactHealthView(
                        status="unavailable",
                        issues=("artifact metadata unavailable",),
                    ),
                    usage=None,
                )
            )
            if summary is not None:
                items = summaries.get(key, ())
                replacement = AttemptListItemView(
                    attempt_id=result.attempt_id, summary=summary
                )
                position = next(
                    (
                        index
                        for index, item in enumerate(items)
                        if item.attempt_id == result.attempt_id
                    ),
                    None,
                )
                if matching is None:
                    summaries[key] = (*items, replacement)[-256:]
                elif position is not None:
                    summaries[key] = (
                        *items[:position],
                        replacement,
                        *items[position + 1 :],
                    )
            self._result_index = MappingProxyType(indexed)
            self._attempt_summary_index = MappingProxyType(summaries)
            self._attempt_summary_totals = MappingProxyType(totals)
            final_sequences = dict(self._accepted_final_sequences)
            if detail is not None:
                final_sequences[(*key, result.attempt_id)] = detail.last_sequence
            self._accepted_final_sequences = MappingProxyType(final_sequences)
            self._result_index_unavailable.discard(result.evaluation_id)

    def _build_attempt_indexes(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case_id: str,
        results: tuple[EpisodeResult, ...],
    ) -> tuple[
        tuple[AttemptListItemView, ...], dict[tuple[str, str, str], int]
    ]:
        from rpent.ui.case_snapshot import SnapshotIdentity, decode_case_snapshot

        unavailable = ArtifactHealthView(
            status="unavailable", issues=("artifact metadata unavailable",)
        )
        items: list[AttemptListItemView] = []
        final_sequences: dict[tuple[str, str, str], int] = {}
        summary_start = max(0, len(results) - 256)
        for index, result in enumerate(results):
            summary = None
            path = (
                self._attempt_root(root, case_id, result.attempt_id)
                / "case_detail.json"
            )
            try:
                detail = decode_case_snapshot(
                    self._json(path, root),
                    SnapshotIdentity(
                        manifest.evaluation_id,
                        manifest.manifest_id,
                        case_id,
                        result.attempt_id,
                    ),
                    result,
                )
                summary = detail.run_summary
                final_sequences[
                    (manifest.evaluation_id, case_id, result.attempt_id)
                ] = detail.last_sequence
            except (FileNotFoundError, OSError, TypeError, ValueError):
                if index >= summary_start:
                    self._increment_diagnostic("degraded_summaries")
            if index < summary_start:
                continue
            if summary is None:
                summary = build_attempt_summary(
                    result.to_payload(), artifact_health=unavailable, usage=None
                )
            if summary is not None:
                items.append(
                    AttemptListItemView(attempt_id=result.attempt_id, summary=summary)
                )
        return tuple(items), final_sequences

    def _increment_diagnostic(self, name: str) -> None:
        with self._diagnostic_lock:
            self._diagnostic_metrics[name] += 1

    def _indexed_case_results(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case_id: str,
    ) -> tuple[EpisodeResult, ...]:
        key = (manifest.evaluation_id, case_id)
        with self._result_index_lock:
            retained = self._result_index.get(key)
        if retained is not None:
            return retained
        return tuple(
            item for item in self._results(root, manifest) if item.case_id == case_id
        )

    def _legacy_active_attempt_scan(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        result_attempts: set[str],
        latest_result: EpisodeResult | None,
    ) -> str | None:
        attempts_root = root / "cases" / case.case_id / "attempts"
        candidates: list[tuple[datetime, str]] = []
        for index, attempt_name in enumerate(
            _iter_directory_beneath(root, attempts_root), start=1
        ):
            if index > _MAX_LEGACY_ATTEMPT_SCAN:
                raise CurrentAttemptUnavailable(
                    "legacy current Attempt scan limit exceeded"
                )
            if attempt_name in result_attempts:
                continue
            attempt_root = attempts_root / attempt_name
            projection = self._optional_json(attempt_root / "case.json", root)
            if projection is None:
                continue
            if "registered_at" in projection:
                # New-style registrations are authoritative only after their locator
                # commit. A failed locator write must remain an unreachable orphan.
                continue
            if (
                projection.get("schema_version") != 1
                or projection.get("evaluation_id") != manifest.evaluation_id
                or projection.get("manifest_id") != manifest.manifest_id
                or projection.get("case_id") != case.case_id
                or projection.get("attempt_id") != attempt_name
                or projection.get("case") != case.to_payload()
            ):
                continue
            started_at = projection.get("started_at")
            started = (
                _file_mtime(attempt_root / "case.json", root)
                if started_at is None
                else _parse_timestamp(started_at, require_utc=True)
            )
            candidates.append((started, attempt_name))
        if not candidates:
            return None
        latest_started = max(started for started, _attempt_id in candidates)
        newest = tuple(
            attempt_id
            for started, attempt_id in candidates
            if started == latest_started
        )
        if len(newest) != 1:
            raise CurrentAttemptUnavailable(
                "legacy current Attempt ordering is ambiguous"
            )
        if latest_result is not None and latest_started <= _parse_timestamp(
            latest_result.finished_at, require_utc=False
        ):
            return None
        return newest[0]

    def _selected_current_attempt(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        latest_result: EpisodeResult | None,
    ) -> str | None:
        locator_path = root / "cases" / case.case_id / "current_attempt.json"
        try:
            registration = load_current_attempt(
                locator_path,
                evaluation_id=manifest.evaluation_id,
                manifest_id=manifest.manifest_id,
                case_id=case.case_id,
            )
        except FileNotFoundError:
            registration = None
        except (OSError, TypeError, ValueError) as exc:
            raise CurrentAttemptUnavailable(
                "current Attempt locator is unavailable"
            ) from exc
        if registration is not None:
            try:
                retained = load_case_registration(
                    self._attempt_root(root, case.case_id, registration.attempt_id)
                    / "case.json",
                    manifest=manifest,
                    case=case,
                    attempt_id=registration.attempt_id,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise CurrentAttemptUnavailable(
                    "current Attempt registration is unavailable"
                ) from exc
            if retained != registration:
                raise CurrentAttemptUnavailable(
                    "current Attempt registration identity mismatch"
                )
            if latest_result is None or latest_result.attempt_id == retained.attempt_id:
                return retained.attempt_id
            if _parse_timestamp(
                retained.registered_at, require_utc=True
            ) > _parse_timestamp(latest_result.finished_at, require_utc=False):
                return retained.attempt_id
            return None
        legacy = self._legacy_active_identities.get(
            (manifest.evaluation_id, case.case_id)
        )
        if legacy is not None:
            return legacy
        if (manifest.evaluation_id, case.case_id) in self._legacy_identity_unavailable:
            raise CurrentAttemptUnavailable("current Attempt identity is unavailable")
        return None

    def get_evaluation(self, evaluation_id: str) -> EvaluationDetail:
        root, manifest = self._lookup(evaluation_id)
        with self._result_index_lock:
            result_index_unavailable = evaluation_id in self._result_index_unavailable
        if result_index_unavailable:
            raise ValueError("accepted result ledger JSON is unavailable")
        preparation = self._preparation(root, manifest)
        results = tuple(
            result
            for case in manifest.cases
            for result in self._indexed_case_results(root, manifest, case.case_id)
        )
        latest = _latest_results(results)
        cases = tuple(
            self._fast_case_summary(
                manifest,
                case,
                latest.get(case.case_id),
            )
            for case in manifest.cases
        )
        item = self._evaluation_item(
            root, manifest, cases=cases, preparation=preparation
        )
        summary = self._summary(root, manifest)
        official, diagnostics, qualification, submission = self._official_views(
            root, manifest
        )
        return EvaluationDetail(
            **item.model_dump(),
            manifest=manifest.to_payload(),
            summary=summary,
            cases=cases,
            preparation=preparation,
            official_metrics=official,
            rpent_diagnostics=diagnostics,
            qualification=qualification,
            submission=submission,
        )

    def _fast_case_summary(
        self,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        result: EpisodeResult | None,
    ) -> CaseSummary:
        if self._runtime_store is not None:
            retained = self._runtime_store.case_summary(
                manifest.evaluation_id, case.case_id
            )
            if retained is not None:
                return retained
        run_summary = None
        if result is not None:
            with self._result_index_lock:
                items = self._attempt_summary_index.get(
                    (manifest.evaluation_id, case.case_id), ()
                )
            selected = next(
                (item for item in items if item.attempt_id == result.attempt_id),
                None,
            )
            run_summary = None if selected is None else selected.summary
        return CaseSummary(
            case_id=case.case_id,
            ordinal=case.ordinal,
            task_name=case.task_name,
            task_index=case.task_index,
            episode_index=case.episode_index,
            seed=case.seed,
            execution_state=(
                "queued"
                if result is None
                else "cancelled"
                if result.status == "cancelled"
                else "terminal"
            ),
            attempt_id=None if result is None else result.attempt_id,
            episode_result=None if result is None else result.to_payload(),
            artifact_health=(
                None if run_summary is None else run_summary.artifact_health
            ),
            current_step=None if result is None else result.step_count,
            started_at=None if result is None else result.started_at,
            finished_at=None if result is None else result.finished_at,
            run_summary=run_summary,
        )

    def get_case(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
        _authorize_attempt: bool = True,
    ) -> CaseDetail:
        root, manifest = self._lookup(evaluation_id)
        case = next((item for item in manifest.cases if item.case_id == case_id), None)
        if case is None:
            raise KeyError((evaluation_id, case_id))
        results = self._results(root, manifest)
        case_results = tuple(item for item in results if item.case_id == case_id)
        if attempt_id is None:
            result = case_results[-1] if case_results else None
        else:
            result = next(
                (item for item in case_results if item.attempt_id == attempt_id),
                None,
            )
            if _authorize_attempt:
                self.authorize_attempt(evaluation_id, case_id, attempt_id)
            elif result is None:
                raise KeyError((evaluation_id, case_id, attempt_id))
        summary, last_sequence = self._case_summary_with_sequence(
            root,
            manifest,
            case.to_payload(),
            result,
            _result_attempts(results).get(case_id, set()),
            forced_attempt_id=attempt_id,
        )
        artifacts = self._artifacts(
            root, manifest.evaluation_id, case_id, summary.attempt_id
        )
        if (
            summary.episode_result is not None
            and summary.artifact_health is not None
            and summary.artifact_health.status == "unavailable"
        ):
            recording_profile, cameras, retained_camera_ids = None, (), ()
        else:
            try:
                recording_profile, cameras, retained_camera_ids = (
                    self._recording_projection(
                        root,
                        manifest.evaluation_id,
                        case_id,
                        summary.attempt_id,
                        finalized=summary.episode_result is not None,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError):
                recording_profile, cameras, retained_camera_ids = None, (), ()
                if summary.episode_result is not None:
                    health = summary.artifact_health
                    issue = "recording discovery unavailable"
                    issues = () if health is None else health.issues
                    if issue not in issues:
                        issues = (*issues, issue)
                    final_health = ArtifactHealthView(status="degraded", issues=issues)
                    run_summary = summary.run_summary
                    summary = summary.model_copy(
                        update={
                            "artifact_health": final_health,
                            "run_summary": (
                                None
                                if run_summary is None
                                else run_summary.model_copy(
                                    update={"artifact_health": final_health}
                                )
                            ),
                        }
                    )
        attempt_items = self._attempt_items(
            root, manifest, case.to_payload(), case_results
        )
        attempt_window = attempt_items[-256:]
        detail = CaseDetail(
            **summary.model_dump(),
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            last_sequence=last_sequence,
            case=case.to_payload(),
            artifacts=artifacts,
            recording_profile=recording_profile,
            cameras=cameras,
            retained_camera_ids=retained_camera_ids,
            data_schema=self._data_schema_projection(manifest, case, cameras),
            attempts=attempt_window,
            attempts_total=len(attempt_items),
            attempts_truncated=len(attempt_items) > len(attempt_window),
            playback_artifact_id=select_playback_artifact(
                artifacts=artifacts,
                artifact_health=summary.artifact_health,
                cameras=cameras,
                retained_camera_ids=retained_camera_ids,
            ),
        )
        if result is not None:
            self.note_accepted_result(result, detail)
        return detail

    def load_recording_declaration(self, identity: CurrentAttemptIdentity):
        """Load one authorized active recording declaration with bounded I/O."""
        from rpent.ui.case_runtime import RecordingDeclaration

        if not isinstance(identity, CurrentAttemptIdentity):
            raise TypeError("identity must be a CurrentAttemptIdentity")
        selected = self.authorize_attempt(
            identity.evaluation_id, identity.case_id, identity.attempt_id
        )
        if selected != identity or identity.attempt_id is None:
            raise ValueError("recording declaration identity mismatch")
        root, manifest = self._lookup(identity.evaluation_id)
        case = next(item for item in manifest.cases if item.case_id == identity.case_id)
        profile, cameras, retained = self._recording_projection(
            root,
            identity.evaluation_id,
            identity.case_id,
            identity.attempt_id,
            finalized=False,
            max_bytes=MAX_RECORDING_DECLARATION_BYTES,
        )
        if profile is None:
            raise FileNotFoundError("recording declaration is not available")
        return RecordingDeclaration(
            profile=profile,
            cameras=cameras,
            retained_camera_ids=retained,
            data_schema=self._data_schema_projection(manifest, case, cameras),
            artifacts=self._artifacts(
                root,
                identity.evaluation_id,
                identity.case_id,
                identity.attempt_id,
            ),
        )

    def read_event_journal(
        self,
        identity: CurrentAttemptIdentity,
        *,
        through_sequence: int | None = None,
    ):
        """Read one complete, bounded, contiguous authorized event journal."""
        from rpent.recording.notifications import validate_notification
        from rpent.ui.case_runtime import JournalRead

        if not isinstance(identity, CurrentAttemptIdentity):
            raise TypeError("identity must be a CurrentAttemptIdentity")
        if through_sequence is not None:
            _json_safe_index(through_sequence, "through_sequence")
        selected = self.authorize_attempt(
            identity.evaluation_id, identity.case_id, identity.attempt_id
        )
        if selected != identity or identity.attempt_id is None:
            raise ValueError("event journal identity mismatch")
        root, _ = self._lookup(identity.evaluation_id)
        path = (
            self._attempt_root(root, identity.case_id, identity.attempt_id)
            / "events.jsonl"
        )
        records, _incomplete = self._jsonl(path, root)
        events: list[EventEnvelope] = []
        expected_sequence = 1
        for payload in records:
            event = EventEnvelope.from_payload(payload)
            validate_notification(event)
            if (
                event.evaluation_id != identity.evaluation_id
                or event.case_id != identity.case_id
                or event.attempt_id != identity.attempt_id
            ):
                raise ValueError("event journal identity mismatch")
            if event.sequence != expected_sequence:
                raise ValueError("event journal sequence is not contiguous")
            expected_sequence += 1
            if through_sequence is None or event.sequence <= through_sequence:
                events.append(event)
        high_water = events[-1].sequence if events else 0
        return JournalRead(tuple(events), high_water)

    def read_event_high_water(self, identity: CurrentAttemptIdentity) -> int:
        """Read only the bounded last complete journal record."""
        if not isinstance(identity, CurrentAttemptIdentity):
            raise TypeError("identity must be a CurrentAttemptIdentity")
        selected = self.authorize_attempt(
            identity.evaluation_id, identity.case_id, identity.attempt_id
        )
        if selected != identity or identity.attempt_id is None:
            raise ValueError("event journal identity mismatch")
        root, _ = self._lookup(identity.evaluation_id)
        path = (
            self._attempt_root(root, identity.case_id, identity.attempt_id)
            / "events.jsonl"
        )
        try:
            with _open_beneath(root, path) as stream:
                size = os.fstat(stream.fileno()).st_size
                window = min(size, self.max_event_tail_bytes)
                stream.seek(size - window)
                raw = stream.read(window)
        except FileNotFoundError:
            return 0
        if size > window:
            boundary = raw.find(b"\n")
            if boundary < 0:
                raise ValueError("event journal tail exceeds byte limit")
            raw = raw[boundary + 1 :]
        if raw and not raw.endswith(b"\n"):
            boundary = raw.rfind(b"\n")
            raw = b"" if boundary < 0 else raw[: boundary + 1]
        line = next((item for item in reversed(raw.splitlines()) if item.strip()), None)
        if line is None:
            return 0
        if len(line) > self.max_event_record_bytes:
            raise ValueError("event record byte limit exceeded")
        event = EventEnvelope.from_payload(_decode_event(line, path))
        if (
            event.evaluation_id != identity.evaluation_id
            or event.case_id != identity.case_id
            or event.attempt_id != identity.attempt_id
        ):
            raise ValueError("event journal identity mismatch")
        return event.sequence

    def load_terminal_case_snapshot(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None = None,
    ) -> CaseDetail:
        """Load and verify one selected terminal snapshot."""
        from rpent.ui.case_snapshot import SnapshotIdentity, decode_case_snapshot

        root, manifest = self._lookup(evaluation_id)
        case_results = self._indexed_case_results(root, manifest, case_id)
        selected_attempt = (
            case_results[-1].attempt_id
            if attempt_id is None and case_results
            else attempt_id
        )
        if selected_attempt is None:
            return self.reconstruct_case_detail(evaluation_id, case_id)
        result = next(
            (item for item in case_results if item.attempt_id == selected_attempt),
            None,
        )
        if result is None:
            raise FileNotFoundError("selected Attempt is not terminal")
        path = self._attempt_root(root, case_id, selected_attempt) / "case_detail.json"
        payload = self._json(path, root)
        detail = decode_case_snapshot(
            payload,
            SnapshotIdentity(
                evaluation_id,
                manifest.manifest_id,
                case_id,
                selected_attempt,
            ),
            result,
        )
        return self._decorate_attempt_window(detail, root, manifest, case_id)

    def reconstruct_case_detail(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ) -> CaseDetail:
        """Expensive background-only reconstruction for legacy/damaged snapshots."""
        return self.get_case(
            evaluation_id,
            case_id,
            attempt_id=attempt_id,
            _authorize_attempt=False,
        )

    def persist_terminal_case_snapshot(self, detail: CaseDetail) -> None:
        """Atomically replace one validated accepted Case snapshot."""
        from rpent.ui.case_snapshot import (
            SnapshotIdentity,
            decode_case_snapshot,
            encode_case_snapshot,
        )

        if not isinstance(detail, CaseDetail) or detail.attempt_id is None:
            raise TypeError("terminal Case snapshot requires a CaseDetail Attempt")
        root, manifest = self._lookup(detail.evaluation_id)
        result = next(
            (
                item
                for item in self._indexed_case_results(
                    root, manifest, detail.case_id
                )
                if item.attempt_id == detail.attempt_id
            ),
            None,
        )
        if result is None:
            raise FileNotFoundError("selected Attempt is not terminal")
        payload = encode_case_snapshot(detail)
        decode_case_snapshot(
            payload,
            SnapshotIdentity(
                detail.evaluation_id,
                manifest.manifest_id,
                detail.case_id,
                detail.attempt_id,
            ),
            result,
        )
        media.atomic_write_json(
            self._attempt_root(root, detail.case_id, detail.attempt_id)
            / "case_detail.json",
            payload,
        )
        self.note_accepted_result(result, detail)

    def terminal_operation_sequence(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ) -> int:
        """Return the immutable accepted final sequence for one Attempt."""
        root, manifest = self._lookup(evaluation_id)
        case_results = self._indexed_case_results(root, manifest, case_id)
        selected_attempt = (
            case_results[-1].attempt_id
            if attempt_id is None and case_results
            else attempt_id
        )
        if selected_attempt is None or not any(
            result.attempt_id == selected_attempt for result in case_results
        ):
            raise FileNotFoundError("selected Attempt is not terminal")
        with self._result_index_lock:
            final_sequence = self._accepted_final_sequences.get(
                (evaluation_id, case_id, selected_attempt)
            )
        if final_sequence is None:
            raise ValueError("accepted final sequence is unavailable")
        return final_sequence

    def load_terminal_operation_snapshot(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None = None,
    ) -> OperationPage:
        """Load and verify one selected terminal operation snapshot."""
        from rpent.ui.case_snapshot import SnapshotIdentity
        from rpent.ui.operation_snapshot import decode_operation_snapshot

        root, manifest = self._lookup(evaluation_id)
        case_results = self._indexed_case_results(root, manifest, case_id)
        selected_attempt = (
            case_results[-1].attempt_id
            if attempt_id is None and case_results
            else attempt_id
        )
        result = next(
            (item for item in case_results if item.attempt_id == selected_attempt),
            None,
        )
        if result is None or selected_attempt is None:
            raise FileNotFoundError("selected Attempt is not terminal")
        final_sequence = self.terminal_operation_sequence(
            evaluation_id, case_id, attempt_id=selected_attempt
        )
        path = self._attempt_root(root, case_id, selected_attempt) / "operations.json"
        return decode_operation_snapshot(
            self._json(path, root),
            SnapshotIdentity(
                result.evaluation_id,
                result.manifest_id,
                result.case_id,
                result.attempt_id,
            ),
            final_sequence,
        )

    def persist_terminal_operation_snapshot(self, page: OperationPage) -> None:
        """Atomically replace one validated accepted operation snapshot."""
        from rpent.ui.case_snapshot import SnapshotIdentity
        from rpent.ui.operation_snapshot import (
            decode_operation_snapshot,
            encode_operation_snapshot,
        )

        if not isinstance(page, OperationPage):
            raise TypeError("terminal operation snapshot requires an OperationPage")
        root, manifest = self._lookup(page.evaluation_id)
        result = next(
            (
                item
                for item in self._indexed_case_results(
                    root, manifest, page.case_id
                )
                if item.attempt_id == page.attempt_id
            ),
            None,
        )
        if result is None:
            raise FileNotFoundError("selected Attempt is not terminal")
        final_sequence = self.terminal_operation_sequence(
            page.evaluation_id,
            page.case_id,
            attempt_id=page.attempt_id,
        )
        if page.incomplete_tail:
            raise ValueError("operation snapshot terminal tail is incomplete")
        payload = encode_operation_snapshot(page, manifest.manifest_id)
        decode_operation_snapshot(
            payload,
            SnapshotIdentity(
                result.evaluation_id,
                result.manifest_id,
                result.case_id,
                result.attempt_id,
            ),
            final_sequence,
        )
        media.atomic_write_json(
            self._attempt_root(root, page.case_id, page.attempt_id)
            / "operations.json",
            payload,
        )

    def _decorate_attempt_window(
        self,
        detail: CaseDetail,
        root: Path,
        manifest: EvaluationManifest,
        case_id: str,
    ) -> CaseDetail:
        key = (manifest.evaluation_id, case_id)
        with self._result_index_lock:
            items = self._attempt_summary_index.get(key, ())
            total = self._attempt_summary_totals.get(key, len(items))
        if detail.run_summary is not None:
            items = tuple(
                AttemptListItemView(
                    attempt_id=item.attempt_id,
                    summary=(
                        detail.run_summary
                        if item.attempt_id == detail.attempt_id
                        else item.summary
                    ),
                )
                for item in items
            )
        window = items[-256:]
        return detail.model_copy(
            update={
                "attempts": window,
                "attempts_total": total,
                "attempts_truncated": total > len(window),
            }
        )

    def decorate_case_detail(self, detail: CaseDetail) -> CaseDetail:
        """Decorate one display value from bounded in-memory Attempt summaries."""
        root, manifest = self._lookup(detail.evaluation_id)
        if detail.case_id not in {case.case_id for case in manifest.cases}:
            raise KeyError((detail.evaluation_id, detail.case_id))
        decorated = self._decorate_attempt_window(
            detail, root, manifest, detail.case_id
        )
        if detail.attempt_id is None or any(
            item.attempt_id == detail.attempt_id for item in decorated.attempts
        ):
            return decorated
        key = (detail.evaluation_id, detail.case_id)
        with self._result_index_lock:
            if any(
                result.attempt_id == detail.attempt_id
                for result in self._result_index.get(key, ())
            ):
                return decorated
        summary = detail.run_summary or AttemptSummaryView(
            started_at=detail.started_at,
            artifact_health=detail.artifact_health,
        )
        items = (
            *decorated.attempts,
            AttemptListItemView(attempt_id=detail.attempt_id, summary=summary),
        )[-256:]
        total = decorated.attempts_total + 1
        return decorated.model_copy(
            update={
                "attempts": items,
                "attempts_total": total,
                "attempts_truncated": total > len(items),
            }
        )

    def _data_schema_projection(
        self,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        cameras: tuple[CameraDescriptorView, ...],
    ) -> CaseDataSchemaView:
        declared = manifest.contracts.get("data_schema")
        if isinstance(declared, Mapping):
            try:
                return CaseDataSchemaView.model_validate(declared)
            except ValueError:
                pass
        descriptors = tuple(
            CameraDescriptor(
                camera.camera_id,
                camera.label,
                camera.role,
                camera.width,
                camera.height,
                camera.channels,
                camera.fps,
                camera.media_type,
            )
            for camera in cameras
        )
        try:
            from rpent.envs.base import get_env_spec

            provider = get_env_spec(case.environment).data_schema
            if provider is not None:
                return CaseDataSchemaView.model_validate(
                    provider(descriptors).to_payload()
                )
        except (ImportError, TypeError, ValueError):
            pass
        return CaseDataSchemaView.model_validate(
            build_data_schema(
                cameras=descriptors,
                action_shape=_contract_shape(manifest.contracts, "action"),
                state_shape=_contract_shape(manifest.contracts, "state"),
            ).to_payload()
        )

    def console_page(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        offset: int,
        limit: int,
        attempt_id: str | None = None,
    ) -> ConsolePageView:
        attempt_root, complete = self._selected_console_root(
            evaluation_id, case_id, attempt_id
        )
        if attempt_root is None:
            root, _ = self._lookup(evaluation_id)
            startup = _startup_console_text(root, case_id).encode("utf-8")
            if not startup:
                return _empty_console_page(offset=offset, complete=complete)
            selected_offset = min(offset, len(startup))
            raw = startup[selected_offset : selected_offset + limit]
            return ConsolePageView(
                offset=selected_offset,
                next_offset=selected_offset + len(raw),
                text=raw.decode("utf-8"),
                truncated=offset > len(startup),
                complete=False,
            )
        path = attempt_root / "console.log"
        try:
            return read_console_page(
                path,
                offset=offset,
                limit=limit,
                complete=complete,
                root=attempt_root,
            )
        except FileNotFoundError:
            return _empty_console_page(offset=offset, complete=complete)

    def console_text(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ) -> str:
        attempt_root, _ = self._selected_console_root(
            evaluation_id, case_id, attempt_id
        )
        if attempt_root is None:
            root, _ = self._lookup(evaluation_id)
            startup = _startup_console_text(root, case_id)
            if startup:
                return startup
            raise FileNotFoundError("console output is unavailable")
        return read_console_text(attempt_root / "console.log", root=attempt_root)

    def open_console_download(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[BinaryIO, int]:
        attempt_root, _ = self._selected_console_root(
            evaluation_id, case_id, attempt_id
        )
        if attempt_root is None:
            raise FileNotFoundError("console output is unavailable")
        return open_console_file(attempt_root / "console.log", root=attempt_root)

    def read_submission(self, evaluation_id: str) -> bytes:
        root, manifest = self._lookup(evaluation_id)
        official, _, qualification, submission = self._official_views(root, manifest)
        if not submission.available:
            raise KeyError(evaluation_id)
        return self._validated_submission_bytes(root, manifest, official, qualification)

    def _validated_submission_bytes(
        self,
        root: Path,
        manifest: EvaluationManifest,
        official: OfficialMetricsView,
        qualification: QualificationView,
    ) -> bytes:
        from robots.robocasa.submission import (
            SubmissionMetadata,
            validate_leaderboard_submission,
        )

        path = root / "leaderboard_submission.json"
        with _open_beneath(root, path) as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("leaderboard submission exceeds size limit")
        try:
            payload = json.loads(raw)
        except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("leaderboard submission is malformed") from exc
        validate_leaderboard_submission(payload)
        metadata = SubmissionMetadata.from_payload(
            manifest.qualification.get("submission_metadata")
        ).to_payload()
        if any(payload.get(name) != value for name, value in metadata.items()):
            raise ValueError("submission metadata drift")
        derived = {
            "atomic_seen_success": "atomic_seen",
            "composite_seen_success": "composite_seen",
            "composite_unseen_success": "composite_unseen",
        }
        if any(
            payload[field] != official.split_success_rates.get(split, -1) * 100
            for field, split in derived.items()
        ):
            raise ValueError("submission score drift")
        checkpoint = manifest.policy.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or payload["checkpoint_url"]
            != "https://huggingface.co/"
            f"{checkpoint.get('repo_id')}/tree/{checkpoint.get('revision')}"
        ):
            raise ValueError("submission checkpoint drift")
        dependency = next(
            check
            for check in qualification.checks
            if check.id == "installed_dependencies"
        ).evidence
        if not isinstance(dependency, dict) or payload[
            "robocasa_version"
        ] != dependency.get("robocasa_version"):
            raise ValueError("submission dependency drift")
        return raw

    def events_since(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        since: int,
        attempt_id: str | None = None,
    ) -> EventPage:
        _json_safe_index(since, "since")
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        page, incomplete, has_more, confirmed_sequence = self._event_page(
            root, evaluation_id, case_id, selected_attempt, since
        )
        return EventPage(
            evaluation_id=evaluation_id,
            case_id=case_id,
            attempt_id=selected_attempt,
            since=since,
            last_sequence=(page[-1]["sequence"] if page else confirmed_sequence),
            incomplete_tail=incomplete,
            has_more=has_more,
            events=page,
        )

    def operations(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str,
    ) -> OperationPage:
        return self.reconstruct_operations(
            evaluation_id, case_id, attempt_id=attempt_id
        )

    def reconstruct_operations(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ) -> OperationPage:
        """Expensive background-only operation reconstruction from the journal."""
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        projection = OperationProjection()
        since = 0
        incomplete_tail = False
        observed = 0
        while True:
            page = self.events_since(
                evaluation_id,
                case_id,
                since=since,
                attempt_id=selected_attempt,
            )
            for event in page.events:
                projection.consume(event)
            observed += len(page.events)
            if observed > self.max_jsonl_records:
                raise ValueError("operation projection record limit exceeded")
            incomplete_tail = incomplete_tail or page.incomplete_tail
            if not page.has_more:
                since = page.last_sequence
                break
            if page.last_sequence <= since:
                raise ValueError("operation projection did not advance")
            since = page.last_sequence
        artifacts = self._artifacts(root, evaluation_id, case_id, selected_attempt)
        valid_camera_clip_artifacts = frozenset(
            artifact.artifact_id
            for artifact in artifacts
            if artifact.kind == "action_camera_clip"
            and artifact.available
            and artifact.byte_size is not None
            and artifact.byte_size > 0
        )
        operations = projection.summaries(valid_camera_clip_artifacts)
        steps = [
            step
            for operation in operations
            for step in (operation.first_step_id, operation.last_step_id)
            if step is not None
        ]
        return OperationPage(
            evaluation_id=evaluation_id,
            case_id=case_id,
            attempt_id=selected_attempt,
            last_sequence=since,
            incomplete_tail=incomplete_tail,
            first_step_id=min(steps) if steps else None,
            latest_step_id=max(steps) if steps else None,
            operations=operations,
        )

    def timeline_window(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str,
        requested_step: int | None,
    ) -> TimelineWindow:
        if requested_step is not None:
            _json_safe_index(requested_step, "requested_step")
        root, manifest = self._lookup(evaluation_id)
        if not any(case.case_id == case_id for case in manifest.cases):
            raise KeyError((evaluation_id, case_id))
        self.authorize_attempt(evaluation_id, case_id, attempt_id)
        attempt_root = self._attempt_root(root, case_id, attempt_id)
        path = attempt_root / "events.jsonl"
        resolved = _resolved_beneath(root, path)
        try:
            stream = _open_beneath(root, path)
        except FileNotFoundError:
            window = TimelineWindow(
                evaluation_id=evaluation_id,
                case_id=case_id,
                attempt_id=attempt_id,
                requested_step=requested_step,
                selected_transition=None,
                first_transition=None,
                latest_transition=None,
                transition_count=0,
                event_window=None,
                gap=None,
                incomplete_tail=False,
            )
        else:
            with stream:
                window = self._timeline_index.locate(
                    stream,
                    path=resolved,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    attempt_id=attempt_id,
                    requested_step=requested_step,
                )
        if (
            len(window.model_dump_json().encode("utf-8"))
            > self.max_timeline_response_bytes
        ):
            raise ValueError("timeline response byte limit exceeded")
        return window

    def event_checkpoint(self, evaluation_id: str) -> dict[str, dict[str, Any]]:
        """Return authoritative current-attempt journal checkpoints per Case."""
        root, _ = self._lookup(evaluation_id)
        evaluation = self.get_evaluation(evaluation_id)
        checkpoints: dict[str, dict[str, Any]] = {}
        for case in evaluation.cases:
            sequence = 0
            if case.attempt_id is not None:
                _, _, _, sequence, _ = self._event_progress(
                    root, evaluation_id, case.case_id, case.attempt_id
                )
            checkpoints[case.case_id] = {
                "attempt_id": case.attempt_id,
                "sequence": sequence,
            }
        return checkpoints

    def resolve_artifact(
        self,
        evaluation_id: str,
        case_id: str,
        artifact_id: str,
        attempt_id: str | None = None,
    ) -> Path:
        """Resolve metadata only; API routes must use :meth:`open_artifact`."""
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        artifacts = self._artifacts(root, evaluation_id, case_id, selected_attempt)
        descriptor = next(
            (item for item in artifacts if item.artifact_id == artifact_id), None
        )
        if descriptor is None:
            raise KeyError((evaluation_id, case_id, artifact_id))
        attempt_root = self._attempt_root(root, case_id, selected_attempt)
        path = _safe_artifact_path(root, attempt_root, descriptor.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def open_artifact(
        self,
        evaluation_id: str,
        case_id: str,
        artifact_id: str,
        attempt_id: str | None = None,
    ) -> BinaryIO:
        """Safely open a declared regular file; API routes must use this boundary."""
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        artifacts = self._artifacts(root, evaluation_id, case_id, selected_attempt)
        descriptor = next(
            (item for item in artifacts if item.artifact_id == artifact_id), None
        )
        if descriptor is None:
            raise KeyError((evaluation_id, case_id, artifact_id))
        attempt_root = self._attempt_root(root, case_id, selected_attempt)
        try:
            path = _declared_artifact_path(attempt_root, descriptor.relative_path)
            _resolved_beneath(root, path)
        except ValueError as exc:
            raise OSError("artifact path is no longer safe to open") from exc
        stream = _open_beneath(root, path)
        try:
            canonical_health = _open_beneath(root, attempt_root / "health.jsonl")
        except FileNotFoundError:
            return stream
        except BaseException:
            stream.close()
            raise
        try:
            descriptor_stat = os.fstat(stream.fileno())
            health_stat = os.fstat(canonical_health.fileno())
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                health_stat.st_dev,
                health_stat.st_ino,
            ):
                return stream
            try:
                return _sanitized_health_journal(
                    stream,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    attempt_id=selected_attempt,
                    max_bytes=self.max_jsonl_bytes,
                    max_records=self.max_jsonl_records,
                    max_record_bytes=self.max_event_record_bytes,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise OSError("health journal unavailable") from exc
            finally:
                stream.close()
        except BaseException:
            stream.close()
            raise
        finally:
            canonical_health.close()

    def open_live_frame(
        self, evaluation_id: str, case_id: str, resource_id: str
    ) -> BinaryIO:
        """Open the current attempt's atomically replaced live PNG."""
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, None
        ).attempt_id
        if selected_attempt is None:
            raise FileNotFoundError("Case has no current attempt")
        _, cameras, _ = self._recording_projection(
            root,
            evaluation_id,
            case_id,
            selected_attempt,
            finalized=False,
        )
        selected_camera = _native_camera_id_from_projection(
            cameras, evaluation_id, case_id, resource_id
        )
        attempt_root = self._attempt_root(root, case_id, selected_attempt)
        component = media.camera_storage_component(selected_camera)
        try:
            return _open_beneath(root, attempt_root / "live" / f"{component}.png")
        except FileNotFoundError:
            legacy = media.legacy_camera_storage_component(selected_camera)
            return _open_beneath(root, attempt_root / "live" / f"{legacy}.png")

    def read_live_frame(
        self, evaluation_id: str, case_id: str, resource_id: str
    ) -> MediaBytes:
        """Read one current frame into an explicitly bounded immutable value."""
        return _read_media_bytes(
            self.open_live_frame(evaluation_id, case_id, resource_id)
        )

    def read_tool_image(
        self,
        evaluation_id: str,
        case_id: str,
        sequence: int,
        attempt_id: str | None = None,
    ) -> MediaBytes:
        """Read an image reference attested by one retained tool-result event."""
        _json_safe_index(sequence, "sequence")
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        if selected_attempt is None:
            raise FileNotFoundError("Case has no selected attempt")
        page = self.events_since(
            evaluation_id,
            case_id,
            since=max(0, sequence - 1),
            attempt_id=selected_attempt,
        )
        event = next(
            (item for item in page.events if item.get("sequence") == sequence),
            None,
        )
        reference = _tool_image_reference(event)
        attempt_root = self._attempt_root(root, case_id, selected_attempt)
        selected = _read_media_bytes(
            _open_beneath(
                attempt_root,
                _declared_artifact_path(attempt_root, reference),
            )
        )
        if selected.media_type not in {"image/png", "image/jpeg"}:
            raise ValueError("tool result reference is not an image")
        return selected

    def open_indexed_frame(
        self,
        evaluation_id: str,
        case_id: str,
        resource_id: str,
        frame_index: int,
        attempt_id: str | None = None,
    ) -> BinaryIO:
        """Open a frame only after validating its declared camera index."""
        _json_safe_index(frame_index, "frame_index")
        root, _ = self._lookup(evaluation_id)
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        attempt_root = self._attempt_root(root, case_id, selected_attempt)
        _, cameras, _ = self._recording_projection(
            root,
            evaluation_id,
            case_id,
            selected_attempt,
            finalized=(attempt_root / "artifacts.json").is_file(),
        )
        selected_camera = _native_camera_id_from_projection(
            cameras, evaluation_id, case_id, resource_id
        )
        expected_kind = f"camera_{selected_camera}_index"
        artifacts = self._artifacts(root, evaluation_id, case_id, selected_attempt)
        descriptor = next(
            (item for item in artifacts if item.kind == expected_kind), None
        )
        if descriptor is None:
            raise KeyError((evaluation_id, case_id, selected_camera, "index"))
        with self.open_artifact(
            evaluation_id,
            case_id,
            descriptor.artifact_id,
            attempt_id=selected_attempt,
        ) as index_stream:
            raw = index_stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("camera index byte limit exceeded")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid camera index JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"camera", "frames"}:
            raise ValueError("camera index must contain camera and frames")
        camera = CameraDescriptor.from_payload(payload["camera"])
        if camera.camera_id != selected_camera:
            raise ValueError("camera index identity mismatch")
        frames_payload = payload["frames"]
        if not isinstance(frames_payload, list):
            raise ValueError("camera index frames must be an array")
        frames = tuple(FrameReference.from_payload(item) for item in frames_payload)
        if any(frame.camera_id != selected_camera for frame in frames):
            raise ValueError("camera frame identity mismatch")
        if len({frame.frame_index for frame in frames}) != len(frames):
            raise ValueError("camera frame indexes must be unique")
        selected = next(
            (frame for frame in frames if frame.frame_index == frame_index), None
        )
        if selected is None:
            raise KeyError((evaluation_id, case_id, selected_camera, frame_index))
        path = _declared_artifact_path(attempt_root, selected.relative_path)
        return _open_beneath(root, path)

    def read_indexed_frame(
        self,
        evaluation_id: str,
        case_id: str,
        resource_id: str,
        frame_index: int,
        attempt_id: str | None = None,
    ) -> MediaBytes:
        """Read one validated retained frame into bounded immutable bytes."""
        return _read_media_bytes(
            self.open_indexed_frame(
                evaluation_id,
                case_id,
                resource_id,
                frame_index,
                attempt_id=attempt_id,
            )
        )

    def _discover(
        self, *, recover: bool = True
    ) -> tuple[
        dict[str, tuple[Path, EvaluationManifest]],
        dict[str, Path],
        dict[str, EvaluationListItem],
    ]:
        found: dict[str, tuple[Path, EvaluationManifest]] = {}
        owners: dict[str, Path] = {}
        list_items: dict[str, EvaluationListItem] = {}
        seen_roots: set[Path] = set()
        for configured_root in self.roots:
            with _locked_index(configured_root, exclusive=False):
                boundary_index, boundary_owners, boundary_items, dirty = (
                    self._discover_boundary(configured_root, recover=recover)
                )
            if dirty:
                with _locked_index(configured_root, exclusive=True):
                    boundary_index, boundary_owners, boundary_items, dirty = (
                        self._discover_boundary(configured_root, recover=recover)
                    )
                    if dirty:
                        self._write_boundary_index(
                            configured_root,
                            boundary_index,
                            boundary_owners,
                            boundary_items,
                        )
            for evaluation_id, indexed in boundary_index.items():
                root, _ = indexed
                if root in seen_roots:
                    raise ValueError("evaluation root is indexed more than once")
                if evaluation_id in found:
                    raise ValueError(f"duplicate evaluation_id {evaluation_id!r}")
                found[evaluation_id] = indexed
                owners[evaluation_id] = boundary_owners[evaluation_id]
                list_items[evaluation_id] = boundary_items[evaluation_id]
                seen_roots.add(root)
        return found, owners, list_items

    def _discover_boundary(
        self, configured_root: Path, *, recover: bool
    ) -> tuple[
        dict[str, tuple[Path, EvaluationManifest]],
        dict[str, Path],
        dict[str, EvaluationListItem],
        bool,
    ]:
        found: dict[str, tuple[Path, EvaluationManifest]] = {}
        owners: dict[str, Path] = {}
        list_items: dict[str, EvaluationListItem] = {}
        seen_roots: set[Path] = set()
        index_path = configured_root / _INDEX_FILENAME
        if not index_path.exists():
            return found, owners, list_items, False
        payload = self._cached(
            index_path,
            configured_root,
            json.loads,
            max_bytes=_MAX_JSON_BYTES,
        )
        version, entries = _persistent_index_entries(payload)
        dirty = False
        for entry in entries:
            root = _persistent_artifact_root(entry["artifact_root"])
            if root in seen_roots:
                raise ValueError("evaluation root is indexed more than once")
            manifest = self._cached(
                root / "evaluation_manifest.json",
                root,
                lambda raw: _manifest_from_payload(json.loads(raw)),
                max_bytes=_MAX_JSON_BYTES,
            )
            if manifest.evaluation_id != entry["evaluation_id"]:
                raise ValueError("evaluation index evaluation_id mismatch")
            if manifest.manifest_id != entry["manifest_id"]:
                raise ValueError("evaluation index manifest_id mismatch")
            if manifest.evaluation_id in found:
                raise ValueError(f"duplicate evaluation_id {manifest.evaluation_id!r}")
            try:
                item = EvaluationListItem.model_validate_json(
                    json.dumps(entry["list_item"])
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid evaluation index list item") from exc
            if item.evaluation_id != manifest.evaluation_id:
                raise ValueError("list item evaluation_id mismatch")
            if item.manifest_id != manifest.manifest_id:
                raise ValueError("list item manifest_id mismatch")
            _validate_persistent_list_item(item, manifest)
            if recover and item.execution_state in _RECOVERED_LIST_STATES:
                try:
                    recovered = self._evaluation_item(root, manifest)
                except (OSError, ValueError, CurrentAttemptUnavailable):
                    recovered = item
                if recovered != item:
                    item = recovered
                    dirty = True
            found[manifest.evaluation_id] = (root, manifest)
            owners[manifest.evaluation_id] = configured_root
            list_items[manifest.evaluation_id] = item
            seen_roots.add(root)
        return found, owners, list_items, dirty

    def _replace_boundary_discovery(
        self,
        boundary: Path,
        boundary_index: Mapping[str, tuple[Path, EvaluationManifest]],
        boundary_owners: Mapping[str, Path],
        boundary_items: Mapping[str, EvaluationListItem],
    ) -> tuple[
        dict[str, tuple[Path, EvaluationManifest]],
        dict[str, Path],
        dict[str, EvaluationListItem],
    ]:
        with self._index_lock:
            indexed = {
                evaluation_id: value
                for evaluation_id, value in self._index.items()
                if self._index_owners[evaluation_id] != boundary
            }
            owners = {
                evaluation_id: owner
                for evaluation_id, owner in self._index_owners.items()
                if owner != boundary
            }
            list_items = {
                evaluation_id: item
                for evaluation_id, item in self._list_items.items()
                if self._index_owners[evaluation_id] != boundary
            }
        seen_roots = {root for root, _ in indexed.values()}
        for evaluation_id, value in boundary_index.items():
            root, _ = value
            if root in seen_roots:
                raise ValueError("evaluation root is indexed more than once")
            if evaluation_id in indexed:
                raise ValueError(f"duplicate evaluation_id {evaluation_id!r}")
            indexed[evaluation_id] = value
            owners[evaluation_id] = boundary_owners[evaluation_id]
            list_items[evaluation_id] = boundary_items[evaluation_id]
            seen_roots.add(root)
        return indexed, owners, list_items

    def _lookup(self, evaluation_id: str) -> tuple[Path, EvaluationManifest]:
        with self._index_lock:
            found = self._index.get(evaluation_id)
        if found is None:
            raise KeyError(evaluation_id)
        return found

    def _results(
        self, root: Path, manifest: EvaluationManifest
    ) -> tuple[EpisodeResult, ...]:
        records, _ = self._jsonl(root / "results.jsonl", root)
        results = tuple(EpisodeResult.from_payload(record) for record in records)
        aggregate_results(manifest, results)
        return results

    def _evaluation_item(
        self,
        root: Path,
        manifest: EvaluationManifest,
        *,
        cases: tuple[CaseSummary, ...] | None = None,
        preparation: PreparationView | None = None,
    ) -> EvaluationListItem:
        if preparation is None:
            preparation = self._preparation(root, manifest)
        if cases is None:
            results = self._results(root, manifest)
            latest = _latest_results(results)
            attempts = _result_attempts(results)
            cases = tuple(
                self._case_summary(
                    root,
                    manifest,
                    case.to_payload(),
                    latest.get(case.case_id),
                    attempts.get(case.case_id, set()),
                )
                for case in manifest.cases
            )
        completed = sum(
            case.episode_result is not None
            and case.episode_result.get("status") != "cancelled"
            for case in cases
        )
        return EvaluationListItem(
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            created_at=manifest.created_at,
            environment=str(manifest.environment.get("name", "unknown")),
            task_selector=manifest.task_selector,
            execution_state=_evaluation_state_with_preparation(cases, preparation),
            scheduled=len(manifest.cases),
            completed=completed,
            metrics=aggregate_evaluation(cases),
        )

    def _preparation(
        self, root: Path, manifest: EvaluationManifest
    ) -> PreparationView | None:
        payload = self._optional_json(root / "preparation.json", root)
        if payload is None:
            return None
        snapshot = parse_preparation(
            payload,
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
        )
        return PreparationView(
            status=snapshot.status,
            current_phase=snapshot.current_phase,
            phases=tuple(
                {
                    "phase": phase.phase,
                    "status": phase.status,
                    "started_at": phase.started_at,
                    "finished_at": phase.finished_at,
                    "detail": phase.detail,
                }
                for phase in snapshot.phases
            ),
            error=(
                None
                if snapshot.error is None
                else {
                    "error_type": snapshot.error.error_type,
                    "message": snapshot.error.message,
                }
            ),
        )

    def _case_summary(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: dict[str, Any],
        result: EpisodeResult | None,
        result_attempts: set[str],
    ) -> CaseSummary:
        return self._case_summary_with_sequence(
            root, manifest, case, result, result_attempts
        )[0]

    def _case_summary_with_sequence(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: dict[str, Any],
        result: EpisodeResult | None,
        result_attempts: set[str],
        forced_attempt_id: str | None = None,
    ) -> tuple[CaseSummary, int]:
        if forced_attempt_id is not None:
            active_attempt = forced_attempt_id if result is None else None
            attempt_id = forced_attempt_id
        else:
            active_attempt = self._active_attempt(
                root, manifest, case, result_attempts, result
            )
            if result is not None and active_attempt == result.attempt_id:
                active_attempt = None
                attempt_id = result.attempt_id
            elif active_attempt is not None:
                attempt_id = active_attempt
                result = None
            else:
                attempt_id = result.attempt_id if result is not None else None
        started_at = result.started_at if result is not None else None
        finished_at = result.finished_at if result is not None else None
        if active_attempt is not None:
            projection = self._optional_json(
                self._attempt_root(root, case["case_id"], active_attempt) / "case.json",
                root,
            )
            if projection is not None and isinstance(projection.get("started_at"), str):
                _parse_timestamp(projection["started_at"], require_utc=True)
                started_at = projection["started_at"]
        has_event, current_step, _, last_sequence, usage = self._event_progress(
            root, manifest.evaluation_id, case["case_id"], attempt_id
        )
        health, _ = self._artifact_projection(
            root, manifest.evaluation_id, case["case_id"], attempt_id
        )
        planner_elapsed_seconds, vla_chunks, retained_frame_count = (
            self._attempt_runtime_metrics(
                root,
                manifest.evaluation_id,
                case["case_id"],
                attempt_id,
                finalized=result is not None,
            )
        )
        summary = CaseSummary(
            case_id=case["case_id"],
            ordinal=case["ordinal"],
            task_name=case["task_name"],
            task_index=case["task_index"],
            episode_index=case["episode_index"],
            seed=case["seed"],
            execution_state=_case_state(result, attempt_id, has_event, health),
            attempt_id=attempt_id,
            episode_result=result.to_payload() if result is not None else None,
            artifact_health=health,
            current_step=current_step,
            started_at=started_at,
            finished_at=finished_at,
            run_summary=build_attempt_summary(
                result.to_payload() if result is not None else None,
                artifact_health=health,
                usage=usage,
                planner_elapsed_seconds=planner_elapsed_seconds,
                vla_chunks=vla_chunks,
                retained_frame_count=retained_frame_count,
            ),
        )
        return summary, last_sequence

    def _attempt_items(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: dict[str, Any],
        results: tuple[EpisodeResult, ...],
    ) -> tuple[AttemptListItemView, ...]:
        items: list[AttemptListItemView] = []
        retained = set()
        for result in results:
            retained.add(result.attempt_id)
            health, _ = self._artifact_projection(
                root,
                manifest.evaluation_id,
                case["case_id"],
                result.attempt_id,
            )
            _, _, _, _, usage = self._event_progress(
                root,
                manifest.evaluation_id,
                case["case_id"],
                result.attempt_id,
            )
            planner_elapsed_seconds, vla_chunks, retained_frame_count = (
                self._attempt_runtime_metrics(
                    root,
                    manifest.evaluation_id,
                    case["case_id"],
                    result.attempt_id,
                    finalized=True,
                )
            )
            summary = build_attempt_summary(
                result.to_payload(),
                artifact_health=health,
                usage=usage,
                planner_elapsed_seconds=planner_elapsed_seconds,
                vla_chunks=vla_chunks,
                retained_frame_count=retained_frame_count,
            )
            if summary is not None:
                items.append(
                    AttemptListItemView(
                        attempt_id=result.attempt_id,
                        summary=summary,
                    )
                )
        active = self._active_attempt(
            root,
            manifest,
            case,
            retained,
            results[-1] if results else None,
        )
        if active is not None and active not in retained:
            health, _ = self._artifact_projection(
                root, manifest.evaluation_id, case["case_id"], active
            )
            _, _, _, _, usage = self._event_progress(
                root, manifest.evaluation_id, case["case_id"], active
            )
            planner_elapsed_seconds, vla_chunks, retained_frame_count = (
                self._attempt_runtime_metrics(
                    root,
                    manifest.evaluation_id,
                    case["case_id"],
                    active,
                    finalized=False,
                )
            )
            summary = build_attempt_summary(
                None,
                artifact_health=health,
                usage=usage,
                planner_elapsed_seconds=planner_elapsed_seconds,
                vla_chunks=vla_chunks,
                retained_frame_count=retained_frame_count,
            )
            if summary is not None:
                items.append(AttemptListItemView(attempt_id=active, summary=summary))
        return tuple(items)

    def _attempt_runtime_metrics(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
        *,
        finalized: bool,
    ) -> tuple[float | None, int | None, int | None]:
        if attempt_id is None:
            return None, None, None
        attempt_root = self._attempt_root(root, case_id, attempt_id)
        try:
            planner_elapsed_seconds, vla_chunks = self._runtime_event_metrics(
                root,
                evaluation_id,
                case_id,
                attempt_id,
                finalized=finalized,
            )
        except (OSError, TypeError, ValueError):
            planner_elapsed_seconds, vla_chunks = None, None
        try:
            retained_frame_count = self._retained_frame_count(
                root,
                evaluation_id,
                case_id,
                attempt_id,
                attempt_root,
                finalized=finalized,
            )
        except (OSError, TypeError, ValueError):
            retained_frame_count = None
        return planner_elapsed_seconds, vla_chunks, retained_frame_count

    def _runtime_event_metrics(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        *,
        finalized: bool,
    ) -> tuple[float | None, int | None]:
        path = self._attempt_root(root, case_id, attempt_id) / "events.jsonl"

        def load(raw: bytes) -> tuple[float | None, int | None, bool, int]:
            planner_elapsed_seconds = None
            completed_chunks = None
            completed_seen = False
            chunk_events = 0
            previous = 0
            stream = io.BytesIO(raw)
            for record_count, line in enumerate(stream, start=1):
                if record_count > self.max_jsonl_records:
                    raise ValueError("event journal record limit exceeded")
                if not line.endswith(b"\n"):
                    break
                if not line.strip():
                    continue
                if len(line) > self.max_event_record_bytes:
                    raise ValueError("event record byte limit exceeded")
                event = _decode_event(line, path)
                _validate_event_identity(event, evaluation_id, case_id, attempt_id)
                if event["sequence"] != previous + 1:
                    raise ValueError("event journal sequence mismatch")
                previous = event["sequence"]
                payload = event["payload"]
                if event["event_type"] == "planning_finished":
                    stats = payload.get("stats")
                    value = (
                        stats.get("elapsed_s") if isinstance(stats, Mapping) else None
                    )
                    planner_elapsed_seconds = _optional_non_negative_float(value)
                elif event["event_type"] == "vla_rollout_completed":
                    completed_seen = True
                    completed_chunks = _optional_non_negative_int(
                        payload.get("chunks_used")
                    )
                elif event["event_type"] == "vla_chunk_completed":
                    chunk_events += 1
            return (
                planner_elapsed_seconds,
                completed_chunks,
                completed_seen,
                chunk_events,
            )

        resolved = _resolved_beneath(root, path)
        with _open_beneath(root, path) as stream:
            metadata = os.fstat(stream.fileno())
            if metadata.st_size > self.max_jsonl_bytes:
                raise ValueError("event journal byte limit exceeded")
            key = (
                resolved,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime_ns,
                metadata.st_size,
            )
            with self._runtime_metrics_cache_lock:
                metrics = self._runtime_metrics_cache.get(key)
            if metrics is None:
                raw = stream.read(self.max_jsonl_bytes + 1)
                if len(raw) > self.max_jsonl_bytes:
                    raise ValueError("event journal byte limit exceeded")
                metrics = load(raw)
                with self._runtime_metrics_cache_lock:
                    cached = self._runtime_metrics_cache.get(key)
                    if cached is not None:
                        metrics = cached
                    else:
                        self._runtime_metrics_cache = {
                            existing: value
                            for existing, value in self._runtime_metrics_cache.items()
                            if existing[0] != resolved
                        }
                        self._runtime_metrics_cache[key] = metrics
        (
            planner_elapsed_seconds,
            completed_chunks,
            completed_seen,
            chunk_events,
        ) = metrics
        vla_chunks = (
            completed_chunks
            if completed_seen
            else chunk_events
            if not finalized and chunk_events
            else None
        )
        return planner_elapsed_seconds, vla_chunks

    def _retained_frame_count(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        attempt_root: Path,
        *,
        finalized: bool,
    ) -> int | None:
        if not finalized:
            return self._transition_frame_count(root, case_id, attempt_id, attempt_root)
        payload = self._json(attempt_root / "artifacts.json", root)
        for name, expected in {
            "schema_version": 1,
            "evaluation_id": evaluation_id,
            "case_id": case_id,
            "attempt_id": attempt_id,
        }.items():
            if payload.get(name) != expected:
                raise ValueError(f"artifact {name} identity mismatch")
        profile, cameras, retained_camera_ids = self._recording_projection(
            root, evaluation_id, case_id, attempt_id, finalized=True
        )
        if profile is None:
            return None
        declarations = payload.get("artifacts")
        if not isinstance(declarations, list):
            raise ValueError("artifact declarations must be an array")
        total = 0
        for camera_id in retained_camera_ids:
            declaration = next(
                (
                    item
                    for item in declarations
                    if isinstance(item, Mapping)
                    and item.get("kind") == f"camera_{camera_id}_index"
                ),
                None,
            )
            if declaration is None:
                raise ValueError("retained camera index is missing")
            relative_path = declaration.get("relative_path")
            if not isinstance(relative_path, str):
                raise ValueError("camera index path must be a string")
            index = self._json(
                _safe_artifact_path(root, attempt_root, relative_path), root
            )
            if not isinstance(index, dict) or set(index) != {"camera", "frames"}:
                raise ValueError("camera index must contain camera and frames")
            camera = CameraDescriptor.from_payload(index["camera"])
            if camera.camera_id != camera_id or not any(
                item.camera_id == camera_id for item in cameras
            ):
                raise ValueError("camera index identity mismatch")
            frames_payload = index["frames"]
            if not isinstance(frames_payload, list):
                raise ValueError("camera index frames must be an array")
            frames = tuple(FrameReference.from_payload(item) for item in frames_payload)
            if any(frame.camera_id != camera_id for frame in frames):
                raise ValueError("camera frame identity mismatch")
            if len({frame.frame_index for frame in frames}) != len(frames):
                raise ValueError("camera frame indexes must be unique")
            total += len(frames)
            if total > self.max_jsonl_records:
                raise ValueError("retained frame count limit exceeded")
        return total

    def _transition_frame_count(
        self, root: Path, case_id: str, attempt_id: str, attempt_root: Path
    ) -> int | None:
        path = attempt_root / "trajectory/transitions.jsonl"
        total = 0
        identities: set[tuple[str, int]] = set()
        records, _ = self._jsonl(path, root)
        if not records:
            return None
        for raw in records:
            transition = TransitionRecord.from_payload(raw)
            if transition.case_id != case_id or transition.attempt_id != attempt_id:
                raise ValueError("transition identity mismatch")
            for frame in transition.frames:
                identity = (frame.camera_id, frame.frame_index)
                if identity in identities:
                    raise ValueError("transition frame indexes must be unique")
                identities.add(identity)
                total += 1
                if total > self.max_jsonl_records:
                    raise ValueError("retained frame count limit exceeded")
        return total

    def _active_attempt(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: dict[str, Any],
        result_attempts: set[str],
        latest_result: EpisodeResult | None,
    ) -> str | None:
        del result_attempts
        selected_case = EpisodeCase.from_payload(case)
        return self._selected_current_attempt(
            root, manifest, selected_case, latest_result
        )

    def _artifacts(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
    ) -> tuple[ArtifactDescriptor, ...]:
        return self._artifact_projection(root, evaluation_id, case_id, attempt_id)[1]

    def _recording_projection(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
        *,
        finalized: bool,
        max_bytes: int = _MAX_JSON_BYTES,
    ) -> tuple[
        RecordingProfileView | None,
        tuple[CameraDescriptorView, ...],
        tuple[str, ...],
    ]:
        if attempt_id is None:
            return None, (), ()
        attempt_root = self._attempt_root(root, case_id, attempt_id)
        source = attempt_root / ("artifacts.json" if finalized else "recording.json")
        try:
            payload = self._json(source, root, max_bytes=max_bytes)
        except FileNotFoundError:
            return None, (), ()
        expected_identity = {
            "schema_version": RECORDING_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "case_id": case_id,
            "attempt_id": attempt_id,
        }
        for name, expected in expected_identity.items():
            if payload.get(name) != expected:
                raise ValueError(f"recording {name} identity mismatch")
        discovery_fields = {
            "episode_id",
            "profile",
            "cameras",
            "retained_camera_ids",
        }
        present_fields = discovery_fields.intersection(payload)
        if finalized and not present_fields:
            return None, (), ()
        if present_fields != discovery_fields:
            raise ValueError("recording discovery fields must be complete")
        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("recording episode_id must be non-empty")
        try:
            profile = RecordingProfile.from_payload(payload["profile"])
            cameras_payload = payload["cameras"]
            retained_payload = payload["retained_camera_ids"]
        except KeyError as exc:
            raise ValueError(f"recording discovery is missing {exc}") from exc
        if not isinstance(cameras_payload, list):
            raise ValueError("recording cameras must be an array")
        if not isinstance(retained_payload, list) or not all(
            isinstance(camera_id, str) for camera_id in retained_payload
        ):
            raise ValueError("retained_camera_ids must be a string array")
        cameras = tuple(CameraDescriptor.from_payload(item) for item in cameras_payload)
        retained = tuple(retained_payload)
        expected_retained = profile.retained_camera_ids(cameras)
        if retained != expected_retained:
            raise ValueError("retained_camera_ids do not match recording profile")
        return (
            RecordingProfileView(mode=profile.mode, camera_ids=profile.camera_ids),
            tuple(
                CameraDescriptorView(
                    resource_id=media.camera_storage_component(camera.camera_id),
                    **camera.to_payload(),
                )
                for camera in cameras
            ),
            retained,
        )

    def _artifact_projection(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
    ) -> tuple[ArtifactHealthView | None, tuple[ArtifactDescriptor, ...]]:
        if attempt_id is None:
            return None, ()
        attempt_root = self._attempt_root(root, case_id, attempt_id)
        manifest_path = attempt_root / "artifacts.json"
        if not manifest_path.is_file():
            recording = self._optional_json(attempt_root / "recording.json", root)
            if recording is not None:
                try:
                    expected_identity = {
                        "schema_version": RECORDING_SCHEMA_VERSION,
                        "evaluation_id": evaluation_id,
                        "case_id": case_id,
                        "attempt_id": attempt_id,
                    }
                    for name, expected in expected_identity.items():
                        if recording.get(name) != expected:
                            raise ValueError(f"recording {name} identity mismatch")
                    live = recording.get("artifacts", [])
                    if not isinstance(live, list):
                        raise ValueError("recording artifacts must be a JSON array")
                    declarations = [
                        item
                        for item in live
                        if isinstance(item, Mapping)
                        and item.get("kind") == "action_camera_clip"
                    ]
                    descriptors = tuple(
                        self._artifact_descriptor(root, attempt_root, item)
                        for item in declarations
                    )
                    if len({item.artifact_id for item in descriptors}) != len(
                        descriptors
                    ):
                        raise ValueError("artifact IDs must be unique")
                    if descriptors:
                        return ArtifactHealthView(status="partial"), descriptors
                except (KeyError, OSError, TypeError, ValueError):
                    # Active discovery cannot replace finalized artifact-health
                    # semantics when its identity or declarations are invalid.
                    pass
            qualification = self._optional_json(
                attempt_root / "qualification.json", root
            )
            if qualification is not None:
                try:
                    health = ArtifactHealth.from_payload(
                        qualification["artifact_health"]
                    )
                    return _health_view(health), ()
                except (KeyError, TypeError, ValueError):
                    pass
            return ArtifactHealthView(
                status="unavailable", issues=("artifacts.json is missing",)
            ), ()
        try:
            payload = self._json(manifest_path, root)
            expected_identity = {
                "schema_version": 1,
                "evaluation_id": evaluation_id,
                "case_id": case_id,
                "attempt_id": attempt_id,
            }
            for name, expected in expected_identity.items():
                if payload.get(name) != expected:
                    raise ValueError(f"artifact {name} identity mismatch")
            health = ArtifactHealth.from_payload(payload["health"])
            declared = payload.get("artifacts", [])
            if not isinstance(declared, list):
                raise ValueError("artifacts must be a JSON array")
            descriptors = tuple(
                self._artifact_descriptor(root, attempt_root, item) for item in declared
            )
            if len({item.artifact_id for item in descriptors}) != len(descriptors):
                raise ValueError("artifact IDs must be unique")
        except (KeyError, OSError, TypeError, ValueError):
            return ArtifactHealthView(
                status="unavailable", issues=("artifact metadata unavailable",)
            ), ()
        return _health_view(health), descriptors

    def _artifact_descriptor(
        self, root: Path, attempt_root: Path, item: Any
    ) -> ArtifactDescriptor:
        if not isinstance(item, Mapping):
            raise ValueError("artifact declaration must be an object")
        kind = item.get("kind")
        relative_path = item.get("relative_path")
        artifact_id = item.get("artifact_id")
        if artifact_id is None:
            artifact_id = _legacy_artifact_id(kind, relative_path)
        if not all(
            isinstance(value, str) and value.strip()
            for value in (artifact_id, kind, relative_path)
        ):
            raise ValueError("artifact declaration fields must be non-empty strings")
        try:
            artifact_path = _safe_artifact_path(root, attempt_root, relative_path)
            if kind == "action_camera_clip":
                with _open_beneath(attempt_root, artifact_path) as stream:
                    metadata = os.fstat(stream.fileno())
                    available = stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0
                    byte_size = metadata.st_size if available else None
            else:
                available = artifact_path.is_file()
                byte_size = artifact_path.stat().st_size if available else None
        except OSError:
            if kind != "action_camera_clip":
                raise
            available = False
            byte_size = None
        except ValueError:
            available = False
            byte_size = None
        return ArtifactDescriptor(
            artifact_id=artifact_id,
            kind=kind,
            relative_path=relative_path,
            available=available,
            byte_size=byte_size,
        )

    def _event_progress(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
    ) -> tuple[bool, int | None, bool, int, Mapping[str, Any] | None]:
        if attempt_id is None:
            return False, None, False, 0, None
        path = self._attempt_root(root, case_id, attempt_id) / "events.jsonl"
        try:
            with _open_beneath(root, path) as stream:
                size = os.fstat(stream.fileno()).st_size
                window = min(size, self.max_event_tail_bytes)
                stream.seek(size - window)
                raw = stream.read(window)
        except FileNotFoundError:
            return False, None, False, 0, None
        if size > window:
            newline = raw.find(b"\n")
            if newline < 0:
                return (
                    bool(size),
                    None,
                    bool(raw and not raw.endswith(b"\n")),
                    0,
                    None,
                )
            raw = raw[newline + 1 :]
        incomplete = bool(raw and not raw.endswith(b"\n"))
        complete = raw if not incomplete else raw[: raw.rfind(b"\n") + 1]
        has_event = False
        last_sequence = 0
        current_step = None
        latest_usage = None
        for line in reversed(complete.splitlines()):
            if not line.strip():
                continue
            if len(line) > self.max_event_record_bytes:
                return True, current_step, incomplete, last_sequence, latest_usage
            event = _decode_event(line, path)
            _validate_event_identity(event, evaluation_id, case_id, attempt_id)
            has_event = True
            if last_sequence == 0:
                last_sequence = event["sequence"]
            if current_step is None:
                current_step = _event_progress_step(event)
            if latest_usage is None and event.get("event_type") == "usage_updated":
                payload = event.get("payload")
                if isinstance(payload, Mapping):
                    latest_usage = payload
            if current_step is not None and latest_usage is not None:
                break
        return has_event, current_step, incomplete, last_sequence, latest_usage

    def _event_page(
        self,
        root: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None,
        since: int,
    ) -> tuple[tuple[dict[str, Any], ...], bool, bool, int]:
        if attempt_id is None:
            return (), False, False, 0
        path = self._attempt_root(root, case_id, attempt_id) / "events.jsonl"
        events: list[dict[str, Any]] = []
        scanned_bytes = 0
        scanned_records = 0
        previous = 0
        incomplete = False
        try:
            stream = _open_beneath(root, path)
        except FileNotFoundError:
            return (), False, False, 0
        with stream:
            metadata = os.fstat(stream.fileno())
            resolved = _resolved_beneath(root, path)
            generation = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                metadata.st_size,
            )
            start_offset = self._event_checkpoint(resolved, generation, since)
            if start_offset is not None and start_offset <= metadata.st_size:
                stream.seek(start_offset)
                previous = since
            else:
                while previous < since:
                    line = stream.readline()
                    if not line:
                        return (), incomplete, False, previous
                    if not line.endswith(b"\n"):
                        return (), True, False, previous
                    if not line.strip():
                        continue
                    if len(line) > self.max_event_record_bytes:
                        raise ValueError(f"event record byte limit exceeded in {path}")
                    event = _decode_event(line, path)
                    _validate_event_identity(event, evaluation_id, case_id, attempt_id)
                    if event["sequence"] != previous + 1:
                        raise ValueError("event journal sequence mismatch")
                    previous = event["sequence"]
                self._remember_event_checkpoint(
                    resolved,
                    generation,
                    since,
                    stream.tell(),
                )
            last_page_offset: int | None = None
            while True:
                line = stream.readline()
                if not line:
                    break
                scanned_bytes += len(line)
                scanned_records += 1
                if scanned_bytes > self.max_jsonl_bytes:
                    raise ValueError(f"event scan byte limit exceeded in {path}")
                if scanned_records > self.max_jsonl_records:
                    raise ValueError(f"event scan record limit exceeded in {path}")
                if not line.endswith(b"\n"):
                    incomplete = True
                    break
                if not line.strip():
                    continue
                if len(line) > self.max_event_record_bytes:
                    raise ValueError(f"event record byte limit exceeded in {path}")
                event = _decode_event(line, path)
                _validate_event_identity(event, evaluation_id, case_id, attempt_id)
                if event["sequence"] != previous + 1:
                    raise ValueError("event journal sequence mismatch")
                previous = event["sequence"]
                if event["sequence"] <= since:
                    continue
                if len(events) == self.max_event_page_size:
                    if events and last_page_offset is not None:
                        self._remember_event_checkpoint(
                            resolved,
                            generation,
                            events[-1]["sequence"],
                            last_page_offset,
                        )
                    return tuple(events), incomplete, True, events[-1]["sequence"]
                events.append(event)
                last_page_offset = stream.tell()
        if events and last_page_offset is not None:
            self._remember_event_checkpoint(
                resolved,
                generation,
                events[-1]["sequence"],
                last_page_offset,
            )
        return tuple(events), incomplete, False, previous

    def _event_checkpoint(
        self,
        resolved: Path,
        generation: _EventGeneration,
        sequence: int,
    ) -> int | None:
        with self._event_offsets_lock:
            for key in tuple(self._event_offsets):
                if key[0] == resolved and key[1:6] != generation:
                    del self._event_offsets[key]
            key = (resolved, *generation, sequence)
            offset = self._event_offsets.pop(key, None)
            if offset is None or offset > generation[-1]:
                return None
            self._event_offsets[key] = offset
            return offset

    def _remember_event_checkpoint(
        self,
        resolved: Path,
        generation: _EventGeneration,
        sequence: int,
        offset: int,
    ) -> None:
        with self._event_offsets_lock:
            key = (resolved, *generation, sequence)
            self._event_offsets.pop(key, None)
            self._event_offsets[key] = offset
            while len(self._event_offsets) > self.event_checkpoint_capacity:
                self._event_offsets.popitem(last=False)

    def _attempt_root(self, root: Path, case_id: str, attempt_id: str | None) -> Path:
        if (
            attempt_id is None
            or not isinstance(case_id, str)
            or not isinstance(attempt_id, str)
            or case_id in {".", ".."}
            or attempt_id in {".", ".."}
            or Path(case_id).name != case_id
            or Path(attempt_id).name != attempt_id
        ):
            raise ValueError("invalid Case attempt identity")
        path = (root / "cases" / case_id / "attempts" / attempt_id).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Case attempt escapes evaluation root")
        return path

    def _selected_console_root(
        self, evaluation_id: str, case_id: str, attempt_id: str | None
    ) -> tuple[Path | None, bool]:
        root, manifest = self._lookup(evaluation_id)
        manifest_case = next(
            (item for item in manifest.cases if item.case_id == case_id), None
        )
        if manifest_case is None:
            raise KeyError((evaluation_id, case_id))
        results = tuple(
            item for item in self._results(root, manifest) if item.case_id == case_id
        )
        selected_attempt = self.authorize_attempt(
            evaluation_id, case_id, attempt_id
        ).attempt_id
        if selected_attempt is None:
            return None, False
        result = next(
            (item for item in results if item.attempt_id == selected_attempt), None
        )
        complete = result is not None
        if result is not None and result.task_batch_attempt_id is not None:
            return (
                self._grouped_result_console_root(
                    root, manifest, manifest_case, result
                ),
                complete,
            )
        if result is None:
            grouped = self._active_grouped_console_root(
                root, manifest, manifest_case, selected_attempt
            )
            if grouped is not None:
                return grouped, False
        components = ("cases", case_id, "attempts", selected_attempt)
        return self._canonical_console_root(root, components), complete

    def _grouped_result_console_root(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        result: EpisodeResult,
    ) -> Path:
        batch_attempt_id = result.task_batch_attempt_id
        if batch_attempt_id is None:
            raise ValueError("selected grouped Console batch is unavailable")
        validate_task_batch_result(manifest, case, batch_attempt_id, result)
        return self._canonical_console_root(
            root, ("task_batches", case.task_name, batch_attempt_id)
        )

    def _active_grouped_console_root(
        self,
        root: Path,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        attempt_id: str,
    ) -> Path | None:
        suffix = f"-{case.episode_index:03d}-{case.case_id[:12]}"
        if not attempt_id.endswith(suffix):
            return None
        batch_attempt_id = attempt_id[: -len(suffix)]
        validate_task_batch_attempt_id(batch_attempt_id)
        if attempt_id != f"{batch_attempt_id}{suffix}":
            raise ValueError("active grouped Console attempt is not canonical")
        batch_root = self._canonical_console_root(
            root, ("task_batches", case.task_name, batch_attempt_id)
        )
        ledger = batch_root / "batch.jsonl"
        metadata = ledger.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or ledger.resolve() != ledger.absolute()
        ):
            raise ValueError("active grouped Console ledger is not canonical")
        attempts = load_task_batch_attempts(ledger, manifest)
        if len(attempts) not in {1, 2} or any(
            attempt.task_name != case.task_name
            or attempt.batch_attempt_id != batch_attempt_id
            or case.case_id not in attempt.expected_case_ids
            for attempt in attempts
        ):
            raise ValueError("active grouped Console ledger identity mismatch")
        return batch_root

    @staticmethod
    def _canonical_console_root(root: Path, components: tuple[str, ...]) -> Path:
        boundary = root.resolve()
        expected = boundary.joinpath(*components).absolute()
        resolved = expected.resolve()
        if resolved != expected or not expected.is_relative_to(boundary):
            raise ValueError("selected Console root is not canonical")
        if expected.exists() and not expected.is_dir():
            raise OSError("selected Console root is unavailable")
        return expected

    def _require_attempt(
        self, root: Path, case_id: str, attempt_id: str | None
    ) -> Path:
        attempt_root = self._attempt_root(root, case_id, attempt_id)
        if not attempt_root.is_dir():
            raise KeyError((case_id, attempt_id))
        return attempt_root

    def _summary(
        self, root: Path, manifest: EvaluationManifest
    ) -> EvaluationSummaryView | None:
        try:
            summary = self._json(root / "summary.json", root)
        except FileNotFoundError:
            return None
        if summary.get("manifest_id") != manifest.manifest_id or (
            "evaluation_id" in summary
            and summary["evaluation_id"] != manifest.evaluation_id
        ):
            raise ValueError("summary identity mismatch")
        try:
            status_counts = summary["status_counts"]
            per_task = summary["per_task"]
            if not isinstance(status_counts, dict) or not isinstance(per_task, dict):
                raise TypeError("summary mappings must be JSON objects")
            projected = EvaluationSummaryView(
                manifest_id=summary["manifest_id"],
                evaluation_id=summary.get("evaluation_id"),
                scheduled=summary["scheduled"],
                completed=summary["completed"],
                successful=summary["successful"],
                success_rate=summary["success_rate"],
                errors=summary["errors"],
                timeouts=summary["timeouts"],
                status_counts=tuple(
                    {"status": status, "count": count}
                    for status, count in sorted(status_counts.items())
                ),
                per_task=tuple(
                    {
                        **values,
                        "task_name": task_name,
                    }
                    for task_name, values in sorted(per_task.items())
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid summary: {exc}") from exc
        _validate_summary_manifest(projected, manifest)
        return projected

    def _official_views(
        self, root: Path, manifest: EvaluationManifest
    ) -> tuple[
        OfficialMetricsView,
        RPentDiagnosticsView,
        QualificationView,
        SubmissionView,
    ]:
        profile_id = manifest.qualification.get("official_profile")
        if not isinstance(profile_id, str) or not profile_id:
            return (
                OfficialMetricsView(),
                RPentDiagnosticsView(),
                QualificationView(),
                SubmissionView(),
            )

        official = OfficialMetricsView()
        diagnostics = RPentDiagnosticsView()
        qualification = QualificationView()
        try:
            payload = self._json(root / "official_metrics.json", root)
            retained = OfficialMetrics.from_payload(manifest, payload)
            candidate = OfficialMetricsView(
                available=True,
                task_success_rates=dict(retained.task_success_rates),
                split_success_rates=dict(retained.split_success_rates),
                overall_success_rate=retained.overall_success_rate,
                completed_rollouts=retained.completed_rollouts,
                scheduled_rollouts=retained.scheduled_rollouts,
                completed_tasks=retained.completed_tasks,
                scheduled_tasks=retained.scheduled_tasks,
                is_complete=retained.is_complete,
            )
            official = candidate
        except (KeyError, OSError, RecursionError, TypeError, ValueError):
            pass
        try:
            payload = self._json(root / "rpent_diagnostics.json", root)
            selected = payload["selected_cohort"]
            if (
                payload.get("schema_version") != 1
                or payload.get("manifest_id") != manifest.manifest_id
                or not isinstance(selected, dict)
            ):
                raise ValueError("RPent diagnostics identity mismatch")
            candidate = RPentDiagnosticsView(
                available=True,
                selected_rollouts=selected.get("rollouts"),
                scheduled_rollouts=selected.get("scheduled_rollouts"),
                contract_success_rate=selected.get("contract_success_rate"),
                system_success_rate=selected.get("system_success_rate"),
                selected_cohort=selected,
                all_retained_audit=payload["all_retained_audit"],
                structured_evidence=payload["structured_evidence"],
            )
            if (
                candidate.selected_rollouts is None
                or candidate.scheduled_rollouts != len(manifest.cases)
                or candidate.selected_rollouts > candidate.scheduled_rollouts
            ):
                raise ValueError("RPent diagnostics workload mismatch")
            diagnostics = candidate
        except (KeyError, OSError, RecursionError, TypeError, ValueError):
            pass
        try:
            payload = self._json(root / "qualification.json", root)
            checks = payload["checks"]
            if (
                payload.get("schema_version") != 1
                or payload.get("profile_id") != profile_id
                or not isinstance(checks, list)
            ):
                raise ValueError("qualification identity mismatch")
            candidate = QualificationView(
                available=True,
                profile_id=profile_id,
                status=payload["status"],
                accepted=payload["accepted"],
                checks=tuple(
                    QualificationCheckView.model_validate(item) for item in checks
                ),
            )
            from robots.robocasa.qualification import QUALIFICATION_CHECK_IDS

            if tuple(check.id for check in candidate.checks) != (
                QUALIFICATION_CHECK_IDS
            ) or candidate.status != (
                "qualified"
                if candidate.accepted
                else "partial"
                if not official.is_complete
                else "rejected"
            ):
                raise ValueError("qualification checks or status drift")
            qualification = candidate
        except (KeyError, OSError, RecursionError, TypeError, ValueError):
            pass

        submission = SubmissionView()
        if qualification.accepted and official.is_complete:
            try:
                retained_results = self._results(root, manifest)
                if self._summary(root, manifest) is None or tuple(
                    result.case_id for result in retained_results
                ) != tuple(case.case_id for case in manifest.cases):
                    raise ValueError("submission bundle result identity drift")
                self._validated_submission_bytes(
                    root, manifest, official, qualification
                )
                reproduction = root / "reproduction.md"
                with _open_beneath(root, reproduction) as stream:
                    retained = stream.read(_MAX_JSON_BYTES + 1)
                if (
                    len(retained) > _MAX_JSON_BYTES
                    or manifest.manifest_id.encode() not in retained
                ):
                    raise ValueError("reproduction artifact identity drift")
            except (OSError, RecursionError, TypeError, ValueError):
                pass
            else:
                submission = SubmissionView(
                    available=True,
                    download_url=(
                        "/api/evaluations/"
                        f"{quote(manifest.evaluation_id, safe='')}/submission"
                    ),
                )
        return official, diagnostics, qualification, submission

    def _optional_json(self, path: Path, boundary: Path) -> dict[str, Any] | None:
        try:
            return self._json(path, boundary)
        except FileNotFoundError:
            return None

    def _json(
        self,
        path: Path,
        boundary: Path,
        *,
        max_bytes: int = _MAX_JSON_BYTES,
    ) -> dict[str, Any]:
        def load(raw: bytes) -> dict[str, Any]:
            require_bounded_json_nesting(raw, source=path)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a JSON object")
            return value

        return self._cached(path, boundary, load, max_bytes=max_bytes)

    def _jsonl(
        self, path: Path, boundary: Path
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        def load(raw: bytes) -> tuple[tuple[dict[str, Any], ...], bool]:
            incomplete = bool(raw and not raw.endswith(b"\n"))
            complete = raw if not incomplete else raw[: raw.rfind(b"\n") + 1]
            records: list[dict[str, Any]] = []
            for number, line in enumerate(complete.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    require_bounded_json_nesting(line, source=path)
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSONL record {number} in {path}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"JSONL record {number} in {path} is not an object"
                    )
                records.append(value)
                if len(records) > self.max_jsonl_records:
                    raise ValueError(f"JSONL record limit exceeded in {path}")
            return tuple(records), incomplete

        try:
            return self._cached(
                path,
                boundary,
                load,
                max_bytes=self.max_jsonl_bytes,
            )
        except FileNotFoundError:
            return (), False

    def _cached(
        self,
        path: Path,
        boundary: Path,
        loader: Callable[[bytes], _T],
        *,
        max_bytes: int,
    ) -> _T:
        resolved = _resolved_beneath(boundary, path)
        with _open_beneath(boundary, path) as stream:
            metadata = os.fstat(stream.fileno())
            if metadata.st_size > max_bytes:
                raise ValueError(f"file byte limit exceeded for {path}")
            key = (
                resolved,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime_ns,
                metadata.st_size,
            )
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]
                observed_generation = next(
                    (existing for existing in self._cache if existing[0] == resolved),
                    None,
                )
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"file byte limit exceeded for {path}")
        try:
            loaded = loader(raw)
        except RecursionError as exc:
            raise ValueError(f"JSON nesting limit exceeded in {path}") from exc
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
            current_generation = next(
                (existing for existing in self._cache if existing[0] == resolved),
                None,
            )
            if (
                current_generation is not None
                and current_generation != observed_generation
            ):
                return self._cache[current_generation]
            self._cache = {
                existing: value
                for existing, value in self._cache.items()
                if existing[0] != resolved
            }
            self._cache[key] = loaded
            return loaded


def _legacy_artifact_id(kind: Any, relative_path: Any) -> Any:
    if not isinstance(kind, str) or not isinstance(relative_path, str):
        return kind
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}", kind) is not None:
        return kind
    digest = hashlib.sha256(
        b"rpent-legacy-artifact-v1\0"
        + kind.encode("utf-8")
        + b"\0"
        + relative_path.encode("utf-8")
    ).hexdigest()
    return f"artifact-sha256-{digest}"


def _tool_image_reference(event: Mapping[str, Any] | None) -> str:
    if event is None or event.get("event_type") != "tool_result":
        raise KeyError("tool result event not found")
    payload = event.get("payload")
    result = payload.get("result") if isinstance(payload, Mapping) else None
    references = result.get("observation_refs") if isinstance(result, Mapping) else None
    if not isinstance(references, (list, tuple)):
        raise ValueError("tool result has no retained image reference")
    images = [
        item
        for item in references
        if isinstance(item, str)
        and PurePosixPath(item).suffix.lower() in {".jpeg", ".jpg", ".png"}
    ]
    if not images:
        raise ValueError("tool result has no retained image reference")
    return next(
        (
            item
            for item in reversed(images)
            if "overlay" in PurePosixPath(item).name.lower()
        ),
        images[-1],
    )


def _read_media_bytes(stream: BinaryIO) -> MediaBytes:
    try:
        try:
            metadata = os.fstat(stream.fileno())
        except (AttributeError, OSError):
            metadata = None
        if metadata is not None:
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("media is not an exclusively retained regular file")
            if metadata.st_size > _MAX_MEDIA_BYTES:
                raise ValueError("camera media exceeds the 8 MiB size limit")
        data = stream.read(_MAX_MEDIA_BYTES + 1)
    finally:
        stream.close()
    if len(data) > _MAX_MEDIA_BYTES:
        raise ValueError("camera media exceeds the 8 MiB size limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"
    return MediaBytes(
        data=data,
        media_type=media_type,
        etag=f'"{hashlib.sha256(data).hexdigest()}"',
        size=len(data),
    )


def select_playback_artifact(
    *, artifacts, artifact_health, cameras, retained_camera_ids
) -> str | None:
    """Select one authoritative retained camera video in display order."""
    if artifact_health is None or artifact_health.status in {"partial", "unavailable"}:
        return None
    retained = set(retained_camera_ids)
    ordered = tuple(
        camera
        for camera in cameras
        if camera.camera_id in retained and camera.role == "primary"
    ) + tuple(
        camera
        for camera in cameras
        if camera.camera_id in retained and camera.role != "primary"
    )
    available = {
        artifact.kind: artifact for artifact in artifacts if artifact.available
    }
    for camera in ordered:
        artifact = available.get(f"camera_{camera.camera_id}_video")
        if artifact is not None:
            return artifact.artifact_id
    return None


def _latest_results(
    results: tuple[EpisodeResult, ...],
) -> dict[str, EpisodeResult]:
    latest: dict[str, EpisodeResult] = {}
    for result in results:
        latest[result.case_id] = result
    return latest


def _manifest_from_payload(payload: Any) -> EvaluationManifest:
    """Validate a securely-read manifest without reopening its path."""
    if not isinstance(payload, dict):
        raise ValueError("evaluation manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version not in (1, 2, EVALUATION_MANIFEST_SCHEMA_VERSION):
        raise ValueError("unsupported evaluation manifest schema_version")
    try:
        rpent = payload["rpent"]
        manifest = EvaluationManifest(
            rpent_commit=rpent["commit"],
            rpent_dirty=rpent["dirty"],
            cli_arguments=payload["cli_arguments"],
            environment=payload["environment"],
            simulator=payload["simulator"],
            policy=payload["policy"],
            task_selector=payload["task_selector"],
            cases=tuple(EpisodeCase.from_payload(item) for item in payload["cases"]),
            schedule=payload["schedule"],
            contracts=payload["contracts"],
            runtime=payload["runtime"],
            launch=payload.get("launch", {}) if schema_version >= 3 else {},
            recording=payload.get(
                "recording",
                {"schema_version": 1, "mode": "all", "camera_ids": []},
            ),
            qualification=payload.get(
                "qualification", {"require_complete_artifacts": False}
            ),
            evaluation_id=payload.get("evaluation_id", payload["manifest_id"]),
            created_at=payload["created_at"],
            schema_version=schema_version,
        )
        expected_id = payload["manifest_id"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"evaluation manifest is missing required data: {exc}"
        ) from exc
    if expected_id != manifest.manifest_id:
        raise ValueError(
            "manifest_id mismatch: "
            f"expected {manifest.manifest_id}, actual {expected_id}"
        )
    object.__setattr__(manifest, "source_schema_version", schema_version)
    return manifest


def _persistent_index_entries(
    payload: Any,
) -> tuple[int, tuple[Mapping[str, Any], ...]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "evaluations",
    }:
        raise ValueError("invalid evaluation index schema")
    version = payload["schema_version"]
    if type(version) is not int or version != _INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation index schema version")
    entries = payload["evaluations"]
    if not isinstance(entries, list):
        raise ValueError("evaluation index entries must be a list")
    required = {"evaluation_id", "manifest_id", "artifact_root"}
    expected = required | {"list_item"}
    parsed: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ValueError("invalid evaluation index entry")
        if any(
            not isinstance(entry[name], str) or not entry[name] for name in required
        ):
            raise ValueError("evaluation index fields must be non-empty strings")
        parsed.append(entry)
    return version, tuple(parsed)


def _persistent_artifact_root(value: str) -> Path:
    root = Path(value)
    if not root.is_absolute() or str(root.resolve()) != value:
        raise ValueError("invalid evaluation index artifact root")
    return root


def _persistent_index_payload(
    boundary: Path,
    index: Mapping[str, tuple[Path, EvaluationManifest]],
    owners: Mapping[str, Path],
    list_items: Mapping[str, EvaluationListItem],
) -> dict[str, Any]:
    entries = []
    for evaluation_id in sorted(index):
        if owners[evaluation_id] != boundary:
            continue
        root, manifest = index[evaluation_id]
        entries.append(
            {
                "evaluation_id": evaluation_id,
                "manifest_id": manifest.manifest_id,
                "artifact_root": str(root.resolve()),
                "list_item": list_items[evaluation_id].model_dump(mode="json"),
            }
        )
    return {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "evaluations": entries,
    }


def _result_attempts(results: tuple[EpisodeResult, ...]) -> dict[str, set[str]]:
    attempts: dict[str, set[str]] = {}
    for result in results:
        attempts.setdefault(result.case_id, set()).add(result.attempt_id)
    return attempts


def _health_view(health: ArtifactHealth) -> ArtifactHealthView:
    return ArtifactHealthView(status=health.status, issues=health.issues)


def _case_state(
    result: EpisodeResult | None,
    attempt_id: str | None,
    has_events: bool,
    health: ArtifactHealthView | None,
) -> ExecutionState:
    if result is not None:
        return "cancelled" if result.status == "cancelled" else "terminal"
    if attempt_id is None:
        return "queued"
    if health is not None and health.status != "unavailable":
        return "finalizing"
    return "running" if has_events else "starting"


def _evaluation_state(cases: tuple[CaseSummary, ...]) -> ExecutionState:
    states = {case.execution_state for case in cases}
    if states <= {"terminal"}:
        return "terminal"
    if "running" in states:
        return "running"
    if "finalizing" in states:
        return "finalizing"
    if "starting" in states:
        return "starting"
    if "cancelled" in states and states <= {"cancelled", "terminal"}:
        return "cancelled"
    return "queued"


def _evaluation_state_with_preparation(
    cases: tuple[CaseSummary, ...], preparation: PreparationView | None
) -> ExecutionState:
    case_state = _evaluation_state(cases)
    if preparation is None or preparation.status == "ready":
        return case_state
    if preparation.status == "preparing":
        return (
            case_state
            if case_state in {"terminal", "cancelled", "worker_lost"}
            else "starting"
        )
    if preparation.status == "cancelled":
        return "cancelled"
    return "terminal"


def _safe_artifact_path(root: Path, attempt_root: Path, relative_path: str) -> Path:
    path = _declared_artifact_path(attempt_root, relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes Evaluation root")
    return path


def _native_camera_id(case: CaseDetail, resource_id: str) -> str:
    return _native_camera_id_from_projection(
        case.cameras, case.evaluation_id, case.case_id, resource_id
    )


def _native_camera_id_from_projection(
    cameras: tuple[CameraDescriptorView, ...],
    evaluation_id: str,
    case_id: str,
    resource_id: str,
) -> str:
    if (
        not isinstance(resource_id, str)
        or re.fullmatch(r"camera-sha256-[0-9a-f]{64}", resource_id) is None
    ):
        raise ValueError("camera resource_id must be canonical")
    camera = next((item for item in cameras if item.resource_id == resource_id), None)
    if camera is None:
        raise KeyError((evaluation_id, case_id, resource_id))
    if media.camera_storage_component(camera.camera_id) != resource_id:
        raise ValueError("camera resource identity mismatch")
    return camera.camera_id


def _json_safe_index(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > JSON_SAFE_INTEGER_MAX
    ):
        raise ValueError(
            f"{name} must be a non-negative JSON safe integer "
            f"(at most {JSON_SAFE_INTEGER_MAX})"
        )
    return value


def _optional_non_negative_int(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= JSON_SAFE_INTEGER_MAX else None


def _optional_non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        return None
    if not finite:
        return None
    return float(value)


def _sanitized_health_journal(
    stream: BinaryIO,
    *,
    evaluation_id: str,
    case_id: str,
    attempt_id: str | None,
    max_bytes: int,
    max_records: int,
    max_record_bytes: int,
) -> io.BytesIO:
    raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes or (raw and not raw.endswith(b"\n")):
        raise ValueError("invalid health journal boundary")
    lines = raw.splitlines()
    if len(lines) > max_records:
        raise ValueError("health journal record limit exceeded")
    encoded: list[bytes] = []
    required = {"evaluation_id", "case_id", "attempt_id", "recorded_at", "issue"}
    for line in lines:
        if not line or len(line) > max_record_bytes:
            raise ValueError("invalid health journal record")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid health journal record") from exc
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("invalid health journal fields")
        if (
            row["evaluation_id"] != evaluation_id
            or row["case_id"] != case_id
            or row["attempt_id"] != attempt_id
        ):
            raise ValueError("health journal identity mismatch")
        _parse_timestamp(row["recorded_at"], require_utc=True)
        public_issue = ArtifactHealth("degraded", (row["issue"],)).issues[0]
        sanitized = {**row, "issue": public_issue}
        encoded.append(
            (
                json.dumps(
                    sanitized,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
    return io.BytesIO(b"".join(encoded))


def _declared_artifact_path(attempt_root: Path, relative_path: str) -> Path:
    declared = PurePosixPath(relative_path)
    if declared.is_absolute() or ".." in declared.parts:
        raise ValueError("artifact path must stay within its Evaluation")
    return attempt_root / Path(*declared.parts)


def _resolved_beneath(boundary: Path, path: Path) -> Path:
    root = boundary.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes configured root: {path}")
    try:
        path.absolute().relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path is not lexically below configured root: {path}"
        ) from exc
    return resolved


def _open_beneath(boundary: Path, path: Path) -> BinaryIO:
    """Open a regular file without following any path-component symlinks."""
    root = boundary.resolve()
    _resolved_beneath(root, path)
    relative = path.absolute().relative_to(root)
    if not relative.parts:
        raise OSError("expected a file below the configured root")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    descriptors: list[int] = [os.open(root, directory_flags)]
    try:
        for part in relative.parts[:-1]:
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | nofollow,
            dir_fd=descriptors[-1],
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("retained path is not a regular file")
            return os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _list_directory_beneath(boundary: Path, path: Path) -> tuple[str, ...]:
    root = boundary.resolve()
    _resolved_beneath(root, path)
    relative = path.absolute().relative_to(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    descriptors: list[int] = [os.open(root, flags)]
    try:
        for part in relative.parts:
            descriptors.append(os.open(part, flags, dir_fd=descriptors[-1]))
        return tuple(sorted(os.listdir(descriptors[-1])))
    except FileNotFoundError:
        return ()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _iter_directory_beneath(boundary: Path, path: Path):
    """Yield directory names without materializing an unbounded listing."""
    root = boundary.resolve()
    _resolved_beneath(root, path)
    relative = path.absolute().relative_to(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    descriptors: list[int] = [os.open(root, flags)]
    try:
        for part in relative.parts:
            descriptors.append(os.open(part, flags, dir_fd=descriptors[-1]))
        with os.scandir(descriptors[-1]) as entries:
            for entry in entries:
                yield entry.name
    except FileNotFoundError:
        return
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _file_mtime(path: Path, boundary: Path) -> datetime:
    with _open_beneath(boundary, path) as stream:
        return datetime.fromtimestamp(os.fstat(stream.fileno()).st_mtime, timezone.utc)


def _parse_timestamp(value: Any, *, require_utc: bool) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("attempt timestamp must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("attempt timestamp must be timezone-aware") from exc
    if parsed.utcoffset() is None or (
        require_utc and parsed.utcoffset() != timezone.utc.utcoffset(parsed)
    ):
        raise ValueError("attempt timestamp must be timezone-aware UTC")
    return parsed


def _decode_event(line: bytes, path: Path) -> dict[str, Any]:
    try:
        require_bounded_json_nesting(line, source=path)
        value = json.loads(line)
        return EventEnvelope.from_payload(value).to_payload()
    except RecursionError as exc:
        raise ValueError(f"JSON nesting limit exceeded in {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid event JSON in {path}") from exc


def _validate_event_identity(
    event: dict[str, Any], evaluation_id: str, case_id: str, attempt_id: str
) -> None:
    if (
        event["evaluation_id"] != evaluation_id
        or event["case_id"] != case_id
        or event["attempt_id"] != attempt_id
    ):
        raise ValueError("event journal identity mismatch")


def _event_progress_step(event: dict[str, Any]) -> int | None:
    step_id = event.get("step_id")
    if type(step_id) is int and step_id >= 0:
        return step_id
    if event.get("event_type") != "operation_completed":
        return None
    last_step_id = event.get("payload", {}).get("last_step_id")
    return last_step_id if type(last_step_id) is int and last_step_id >= 0 else None


def _validate_summary_manifest(
    summary: EvaluationSummaryView, manifest: EvaluationManifest
) -> None:
    expected: dict[str, int] = {}
    for case in manifest.cases:
        expected[case.task_name] = expected.get(case.task_name, 0) + 1
    actual = {item.task_name: item.scheduled for item in summary.per_task}
    if summary.scheduled != len(manifest.cases):
        raise ValueError("summary scheduled does not match manifest")
    if actual != expected:
        raise ValueError("summary per-task schedule does not match manifest")


__all__ = [
    "CurrentAttemptUnavailable",
    "EvaluationRepository",
    "MediaBytes",
]
