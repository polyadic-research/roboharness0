"""Immutable, per-Case runtime display projections."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Literal

from rpent.evaluation.attempt_registration import AttemptRegistration
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult
from rpent.recording.contracts import EventEnvelope
from rpent.recording.data_schema import build_data_schema
from rpent.ui.case_snapshot import SnapshotRebuilding, SnapshotUnavailable
from rpent.ui.operation_projection import OperationProjection
from rpent.ui.repository import CurrentAttemptIdentity
from rpent.ui.schemas import (
    ArtifactDescriptor,
    CameraDescriptorView,
    CaseDataSchemaView,
    CaseDetail,
    CaseSummary,
    OperationPage,
    RecordingProfileView,
)

RuntimePhase = Literal[
    "fresh", "repairing", "terminal_pending", "sealed", "unavailable"
]
_STALLED_REPAIR_SECONDS = 15.0
RuntimeDisposition = Literal["applied", "duplicate", "repairing", "rejected"]
RuntimeKey = tuple[str, str, str]
CaseKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeUpdate:
    evaluation_id: str
    case_id: str
    attempt_id: str
    runtime_revision: int
    event: EventEnvelope | None = None


@dataclass(frozen=True, slots=True)
class RuntimeReduction:
    disposition: RuntimeDisposition
    update: RuntimeUpdate | None = None
    reason: str | None = None
    sequence_gap: bool = False


@dataclass(frozen=True, slots=True)
class JournalRead:
    events: tuple[EventEnvelope, ...]
    high_water: int


@dataclass(frozen=True, slots=True)
class RecordingDeclaration:
    profile: RecordingProfileView | None
    cameras: tuple[CameraDescriptorView, ...]
    retained_camera_ids: tuple[str, ...]
    data_schema: CaseDataSchemaView
    artifacts: tuple[ArtifactDescriptor, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeEntry:
    value: CaseDetail
    operations: OperationProjection
    terminal_operations: OperationPage | None
    phase: RuntimePhase
    attempt_epoch: int
    runtime_revision: int
    last_sequence: int
    final_sequence: int | None
    observed_high_water: int
    unavailable_reason: str | None


class CaseRuntimeStore:
    """Own one immutable display snapshot and linearization lock per Case."""

    def __init__(
        self,
        *,
        capacity: int = 4_096,
        journal_reader: Callable[..., JournalRead] | None = None,
        high_water_reader: Callable[[CurrentAttemptIdentity], int] | None = None,
        update_publisher: Callable[[RuntimeUpdate], object] | None = None,
        repair_workers: int = 2,
        repair_capacity: int = 128,
        heartbeat_interval: float | None = None,
        snapshot_loader: Callable[..., CaseDetail] | None = None,
        snapshot_reconstructor: Callable[..., CaseDetail] | None = None,
        snapshot_persister: Callable[[CaseDetail], None] | None = None,
        operation_snapshot_loader: Callable[..., OperationPage] | None = None,
        operation_snapshot_reconstructor: Callable[..., OperationPage] | None = None,
        operation_snapshot_persister: Callable[[OperationPage], None] | None = None,
        operation_sequence_reader: Callable[..., int] | None = None,
        terminal_capacity: int = 1_024,
        snapshot_capacity: int = 128,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._meta_lock = threading.Lock()
        self._locks: dict[CaseKey, threading.Lock] = {}
        self._entries: dict[RuntimeKey, RuntimeEntry] = {}
        self._current: dict[CaseKey, RuntimeKey] = {}
        self._revisions: dict[CaseKey, int] = {}
        self._epochs: dict[CaseKey, int] = {}
        self._recordings: dict[RuntimeKey, RecordingDeclaration] = {}
        self._recording_flights: dict[
            RuntimeKey, Future[RecordingDeclaration]
        ] = {}
        self._identities: dict[RuntimeKey, CurrentAttemptIdentity] = {}
        self._journal_reader = journal_reader
        self._high_water_reader = high_water_reader
        self._update_publisher = update_publisher
        if repair_workers < 1 or repair_capacity < 1:
            raise ValueError("repair workers and capacity must be positive")
        self._repair_executor = ThreadPoolExecutor(
            max_workers=repair_workers, thread_name_prefix="rpent-case-repair"
        )
        self._repair_slots = threading.BoundedSemaphore(repair_capacity)
        self._repair_flights: dict[RuntimeKey, Future[None]] = {}
        self._repair_started: dict[RuntimeKey, float] = {}
        self._closed = False
        if terminal_capacity < 1 or snapshot_capacity < 1:
            raise ValueError("snapshot capacities must be positive")
        self._terminal_capacity = terminal_capacity
        self._snapshot_loader = snapshot_loader
        self._snapshot_reconstructor = snapshot_reconstructor
        self._snapshot_persister = snapshot_persister
        self._operation_snapshot_loader = operation_snapshot_loader
        self._operation_snapshot_reconstructor = operation_snapshot_reconstructor
        self._operation_snapshot_persister = operation_snapshot_persister
        self._operation_sequence_reader = operation_sequence_reader
        self._case_decorator: Callable[[CaseDetail], CaseDetail] | None = None
        self._terminal_cache: OrderedDict[RuntimeKey, CaseDetail] = OrderedDict()
        self._snapshot_current: dict[CaseKey, RuntimeKey] = {}
        self._snapshot_failures: set[tuple[str, str, str | None]] = set()
        self._snapshot_flights: dict[
            tuple[str, str, str | None], Future[None]
        ] = {}
        self._operation_terminal_cache: OrderedDict[RuntimeKey, OperationPage] = (
            OrderedDict()
        )
        self._operation_snapshot_current: dict[CaseKey, RuntimeKey] = {}
        self._operation_snapshot_failures: set[
            tuple[str, str, str | None]
        ] = set()
        self._operation_snapshot_flights: dict[
            tuple[str, str, str | None], Future[None]
        ] = {}
        self._snapshot_slots = threading.BoundedSemaphore(snapshot_capacity)
        self._startup_registration_reader: Callable[[], object] | None = None
        self._startup_recording_loader: Callable[[CurrentAttemptIdentity], object] | None = None
        self._startup_recovered = False
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "reducer_duplicates": 0,
            "reducer_rejected": 0,
            "sequence_gaps": 0,
            "high_water_mismatches": 0,
            "repair_passes": 0,
            "missing_snapshots": 0,
            "damaged_snapshots": 0,
            "snapshot_rebuilds": 0,
            "snapshot_write_failures": 0,
            "operation_cache_hits": 0,
            "operation_cache_misses": 0,
            "operation_cache_evictions": 0,
            "missing_operation_snapshots": 0,
            "damaged_operation_snapshots": 0,
            "operation_snapshot_rebuilds": 0,
        }
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        if heartbeat_interval is not None:
            self._start_heartbeat(heartbeat_interval)

    def configure_recovery(
        self,
        *,
        journal_reader: Callable[..., JournalRead],
        high_water_reader: Callable[[CurrentAttemptIdentity], int],
        update_publisher: Callable[[RuntimeUpdate], object],
        heartbeat_interval: float = 5.0,
    ) -> None:
        """Attach production journal authority before active recovery starts."""
        if not all(
            callable(item)
            for item in (journal_reader, high_water_reader, update_publisher)
        ):
            raise TypeError("recovery collaborators must be callable")
        with self._meta_lock:
            if self._heartbeat_thread is not None:
                raise RuntimeError("runtime recovery is already configured")
            self._journal_reader = journal_reader
            self._high_water_reader = high_water_reader
            self._update_publisher = update_publisher
        self._start_heartbeat(heartbeat_interval)

    def configure_snapshots(
        self,
        *,
        loader: Callable[..., CaseDetail],
        reconstructor: Callable[..., CaseDetail],
        persister: Callable[[CaseDetail], None] | None = None,
    ) -> None:
        if (
            not callable(loader)
            or not callable(reconstructor)
            or (persister is not None and not callable(persister))
        ):
            raise TypeError("snapshot collaborators must be callable")
        with self._meta_lock:
            self._snapshot_loader = loader
            self._snapshot_reconstructor = reconstructor
            self._snapshot_persister = persister

    def configure_operation_snapshots(
        self,
        *,
        loader: Callable[..., OperationPage],
        reconstructor: Callable[..., OperationPage],
        sequence_reader: Callable[..., int],
        persister: Callable[[OperationPage], None] | None = None,
    ) -> None:
        if (
            not callable(loader)
            or not callable(reconstructor)
            or not callable(sequence_reader)
            or (persister is not None and not callable(persister))
        ):
            raise TypeError("operation snapshot collaborators must be callable")
        with self._meta_lock:
            self._operation_snapshot_loader = loader
            self._operation_snapshot_reconstructor = reconstructor
            self._operation_sequence_reader = sequence_reader
            self._operation_snapshot_persister = persister

    def configure_startup(
        self,
        *,
        registration_reader: Callable[[], object],
        recording_loader: Callable[[CurrentAttemptIdentity], object],
    ) -> None:
        if not callable(registration_reader) or not callable(recording_loader):
            raise TypeError("startup collaborators must be callable")
        with self._meta_lock:
            if self._startup_recovered:
                raise RuntimeError("runtime startup recovery already ran")
            self._startup_registration_reader = registration_reader
            self._startup_recording_loader = recording_loader

    def configure_case_decorator(
        self, decorator: Callable[[CaseDetail], CaseDetail]
    ) -> None:
        if not callable(decorator):
            raise TypeError("Case decorator must be callable")
        with self._meta_lock:
            self._case_decorator = decorator

    def startup_recover(self) -> None:
        """Recover committed active Attempts once, isolating failures per Case."""
        with self._meta_lock:
            if self._startup_recovered:
                return
            reader = self._startup_registration_reader
            loader = self._startup_recording_loader
            self._startup_recovered = True
        if reader is None or loader is None:
            return
        self.recover_attempts(reader())

    def recover_attempts(
        self,
        registrations: Iterable[
            tuple[AttemptRegistration, EvaluationManifest, EpisodeCase]
        ],
    ) -> None:
        """Recover committed Attempts with startup's per-Case isolation."""
        for registration, manifest, case in registrations:
            try:
                self.recover_attempt(registration, manifest, case)
            except (OSError, TypeError, ValueError, RuntimeError):
                continue
        self.wait_for_repairs(timeout=15.0)

    def recover_attempt(
        self,
        registration: AttemptRegistration,
        manifest: EvaluationManifest,
        case: EpisodeCase,
    ) -> RuntimeUpdate:
        """Apply the shared startup sequence to one committed Attempt."""
        update = self.start_attempt(registration, manifest, case)
        with self._meta_lock:
            loader = self._startup_recording_loader
            high_water_reader = self._high_water_reader
        if loader is None:
            return update
        identity = CurrentAttemptIdentity(
            registration.evaluation_id,
            registration.manifest_id,
            registration.case_id,
            registration.attempt_id,
        )
        declaration = loader(identity)
        self.prime_recording_declaration(identity, declaration)
        if high_water_reader is not None:
            durable_sequence = high_water_reader(identity)
            if durable_sequence:
                self.observe_high_water(identity, durable_sequence)
        return update

    def start_attempt(
        self,
        registration: AttemptRegistration,
        manifest: EvaluationManifest,
        case: EpisodeCase,
    ) -> RuntimeUpdate:
        """Promote one committed Attempt and publish its base projection."""
        _validate_registration(registration, manifest, case)
        case_key = (registration.evaluation_id, registration.case_id)
        key = (*case_key, registration.attempt_id)
        lock = self._case_lock(case_key)
        with lock:
            existing = self._entries.get(key)
            if existing is not None and self._current.get(case_key) == key:
                return _update(existing, registration.attempt_id)
            with self._meta_lock:
                if key not in self._entries and len(self._entries) >= self._capacity:
                    raise RuntimeError("runtime entry capacity exceeded")
                revision = self._revisions.get(case_key, 0) + 1
                epoch = self._epochs.get(case_key, 0) + 1
                self._revisions[case_key] = revision
                self._epochs[case_key] = epoch
            value = _base_detail(registration, manifest, case)
            entry = RuntimeEntry(
                value=value,
                operations=OperationProjection(),
                terminal_operations=None,
                phase="fresh",
                attempt_epoch=epoch,
                runtime_revision=revision,
                last_sequence=0,
                final_sequence=None,
                observed_high_water=0,
                unavailable_reason=None,
            )
            self._entries[key] = entry
            self._current[case_key] = key
            self._identities[key] = CurrentAttemptIdentity(
                registration.evaluation_id,
                registration.manifest_id,
                registration.case_id,
                registration.attempt_id,
            )
            with self._meta_lock:
                self._snapshot_failures = {
                    request
                    for request in self._snapshot_failures
                    if request[:2] != case_key
                }
                self._operation_snapshot_failures = {
                    request
                    for request in self._operation_snapshot_failures
                    if request[:2] != case_key
                }
        return _update(entry, registration.attempt_id)

    def apply_event(
        self, event: EventEnvelope, recording: RecordingDeclaration
    ) -> RuntimeReduction:
        """Reduce one adjacent event, or enter/follow bounded repair."""
        if not isinstance(event, EventEnvelope):
            raise TypeError("event must be an EventEnvelope")
        if not isinstance(recording, RecordingDeclaration):
            raise TypeError("recording must be a RecordingDeclaration")
        case_key = (event.evaluation_id, event.case_id)
        lock = self._case_lock(case_key)
        with lock:
            key = self._current.get(case_key)
            if key is None or key[2] != event.attempt_id:
                self._increment_metric("reducer_rejected")
                return RuntimeReduction("rejected", reason="attempt is not current")
            entry = self._entries[key]
            if entry.phase == "sealed":
                if event.sequence <= (entry.final_sequence or 0):
                    self._increment_metric("reducer_duplicates")
                    return RuntimeReduction("duplicate")
                self._increment_metric("reducer_rejected")
                return RuntimeReduction("rejected", reason="Attempt is sealed")
            if entry.phase in {"terminal_pending", "unavailable"}:
                self._increment_metric("reducer_rejected")
                return RuntimeReduction("rejected", reason=f"Attempt is {entry.phase}")
            if event.sequence <= entry.last_sequence:
                if event.event_type == "action_camera_clips_ready":
                    value = _recording_value(entry.value, recording)
                    if value != entry.value:
                        revision = self._next_revision(case_key)
                        replacement = replace(
                            entry,
                            value=value,
                            runtime_revision=revision,
                        )
                        self._entries[key] = replacement
                        return RuntimeReduction(
                            "applied",
                            _update(replacement, event.attempt_id, event),
                        )
                self._increment_metric("reducer_duplicates")
                return RuntimeReduction("duplicate")
            if entry.phase == "repairing":
                sequence_gap = event.sequence != entry.last_sequence + 1
                self._entries[key] = replace(
                    entry,
                    observed_high_water=max(entry.observed_high_water, event.sequence),
                )
                return RuntimeReduction(
                    "repairing",
                    reason="event sequence gap" if sequence_gap else None,
                    sequence_gap=sequence_gap,
                )
            if event.sequence != entry.last_sequence + 1:
                self._increment_metric("sequence_gaps")
                replacement = replace(
                    entry,
                    phase="repairing",
                    observed_high_water=max(entry.observed_high_water, event.sequence),
                )
                self._entries[key] = replacement
                repair_epoch = replacement.attempt_epoch
                reduction = RuntimeReduction(
                    "repairing",
                    reason="event sequence gap",
                    sequence_gap=True,
                )
            else:
                repair_epoch = None
                revision = self._next_revision(case_key)
                operations = entry.operations.copy()
                operations.consume(event.to_payload())
                value = _reduce_value(entry.value, event, recording)
                replacement = replace(
                    entry,
                    value=value,
                    operations=operations,
                    phase="fresh",
                    runtime_revision=revision,
                    last_sequence=event.sequence,
                    observed_high_water=max(entry.observed_high_water, event.sequence),
                )
                self._entries[key] = replacement
                reduction = RuntimeReduction(
                    "applied", _update(replacement, event.attempt_id, event)
                )
        if repair_epoch is not None:
            with self._meta_lock:
                self._repair_started.setdefault(key, time.monotonic())
            self._schedule_repair(key, repair_epoch)
        return reduction

    def observe_high_water(
        self, identity: CurrentAttemptIdentity, durable_sequence: int
    ) -> RuntimeReduction:
        """Schedule replay when durable journal state advances without ingress."""
        if not isinstance(identity, CurrentAttemptIdentity):
            raise TypeError("identity must be a CurrentAttemptIdentity")
        if (
            isinstance(durable_sequence, bool)
            or not isinstance(durable_sequence, int)
            or durable_sequence < 0
        ):
            raise ValueError("durable_sequence must be non-negative")
        key = (identity.evaluation_id, identity.case_id, identity.attempt_id or "")
        case_key = key[:2]
        lock = self._case_lock(case_key)
        with lock:
            entry = self._entries.get(key)
            if (
                entry is None
                or self._current.get(case_key) != key
                or self._identities.get(key) != identity
                or entry.phase in {"sealed", "terminal_pending"}
            ):
                self._increment_metric("reducer_rejected")
                return RuntimeReduction("rejected", reason="Attempt is not active")
            if durable_sequence <= entry.last_sequence:
                self._increment_metric("reducer_duplicates")
                return RuntimeReduction("duplicate")
            if entry.phase == "unavailable" and durable_sequence <= entry.observed_high_water:
                self._increment_metric("reducer_rejected")
                return RuntimeReduction("rejected", reason="repair remains unavailable")
            replacement = replace(
                entry,
                phase="repairing",
                observed_high_water=max(entry.observed_high_water, durable_sequence),
                unavailable_reason=None,
            )
            self._entries[key] = replacement
            epoch = replacement.attempt_epoch
        with self._meta_lock:
            self._repair_started.setdefault(key, time.monotonic())
        self._schedule_repair(key, epoch)
        return RuntimeReduction("repairing")

    async def recording_declaration(
        self,
        key: RuntimeKey,
        loader: Callable[[], Awaitable[RecordingDeclaration]],
    ) -> RecordingDeclaration:
        """Load and cache one Attempt declaration with a single async flight."""
        if not callable(loader):
            raise TypeError("loader must be callable")
        owner = False
        with self._meta_lock:
            retained = self._recordings.get(key)
            if retained is not None:
                return retained
            flight = self._recording_flights.get(key)
            if flight is None:
                flight = Future()
                self._recording_flights[key] = flight
                owner = True
        if not owner:
            return await asyncio.wrap_future(flight)
        try:
            loaded = await loader()
            if not isinstance(loaded, RecordingDeclaration):
                raise TypeError("recording loader returned an invalid declaration")
        except BaseException as exc:
            with self._meta_lock:
                self._recording_flights.pop(key, None)
                flight.set_exception(exc)
            raise
        with self._meta_lock:
            self._recordings[key] = loaded
            self._recording_flights.pop(key, None)
            flight.set_result(loaded)
        return loaded

    def prime_recording_declaration(
        self,
        identity: CurrentAttemptIdentity,
        declaration: RecordingDeclaration,
    ) -> None:
        """Install one startup-validated declaration without reducer I/O."""
        if not isinstance(identity, CurrentAttemptIdentity) or not isinstance(
            declaration, RecordingDeclaration
        ):
            raise TypeError("identity and declaration contracts are required")
        if identity.attempt_id is None:
            raise ValueError("recording declaration requires an Attempt")
        key = (identity.evaluation_id, identity.case_id, identity.attempt_id)
        case_key = key[:2]
        lock = self._case_lock(case_key)
        with lock:
            if self._current.get(case_key) != key or self._identities.get(key) != identity:
                raise ValueError("recording declaration Attempt is not current")
            with self._meta_lock:
                self._recordings[key] = declaration

    def mark_terminal_pending(self, result: EpisodeResult) -> RuntimeUpdate:
        """Fence ordinary reduction before terminal snapshot assembly."""
        key, case_key, lock = self._result_entry(result)
        with lock:
            entry = self._entries[key]
            if entry.phase == "sealed":
                return _update(entry, result.attempt_id)
            epoch = self._next_epoch(case_key)
            revision = self._next_revision(case_key)
            replacement = replace(
                entry,
                phase="terminal_pending",
                attempt_epoch=epoch,
                runtime_revision=revision,
            )
            self._entries[key] = replacement
        return _update(replacement, result.attempt_id)

    def publish_terminal(
        self,
        detail: CaseDetail,
        result: EpisodeResult,
        operation_page: OperationPage | None = None,
    ) -> RuntimeUpdate:
        """Permanently seal an accepted result and its display snapshot."""
        if not isinstance(detail, CaseDetail):
            raise TypeError("detail must be a CaseDetail")
        key, case_key, lock = self._result_entry(result)
        if (
            detail.evaluation_id != result.evaluation_id
            or detail.case_id != result.case_id
            or detail.attempt_id != result.attempt_id
        ):
            raise ValueError("terminal snapshot identity mismatch")
        with lock:
            entry = self._entries[key]
            if entry.phase == "sealed":
                return _update(entry, result.attempt_id)
            if entry.phase != "terminal_pending":
                raise RuntimeError("Attempt is not terminal-pending")
            revision = self._next_revision(case_key)
            final_sequence = detail.last_sequence
            value = detail.model_copy(
                update={
                    "episode_result": result.to_payload(),
                    "execution_state": (
                        "cancelled" if result.status == "cancelled" else "terminal"
                    ),
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                }
            )
            terminal_operations = operation_page or _operation_page(
                entry.operations, value, final_sequence
            )
            _validate_operation_page(
                terminal_operations,
                result.evaluation_id,
                result.case_id,
                result.attempt_id,
                expected_sequence=final_sequence,
            )
            replacement = replace(
                entry,
                value=value,
                terminal_operations=terminal_operations,
                phase="sealed",
                runtime_revision=revision,
                last_sequence=final_sequence,
                final_sequence=final_sequence,
                observed_high_water=max(entry.observed_high_water, final_sequence),
                unavailable_reason=None,
            )
            self._entries[key] = replacement
        return _update(replacement, result.attempt_id)

    def get_operations(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None = None,
    ) -> OperationPage:
        """Return a hot projection, one strict cold snapshot, or a rebuild state."""
        case_key = (evaluation_id, case_id)
        lock = self._case_lock(case_key)
        with lock:
            key = (
                self._current.get(case_key)
                if attempt_id is None
                else (evaluation_id, case_id, attempt_id)
            )
            entry = None if key is None else self._entries.get(key)
            if entry is not None:
                if entry.terminal_operations is not None:
                    return entry.terminal_operations
                return _operation_page(
                    entry.operations, entry.value, entry.last_sequence
                )
        request_key = (evaluation_id, case_id, attempt_id)
        retained = None
        with self._meta_lock:
            cache_key = (
                self._operation_snapshot_current.get(case_key)
                if attempt_id is None
                else (evaluation_id, case_id, attempt_id)
            )
            if cache_key is not None:
                retained = self._operation_terminal_cache.get(cache_key)
                if retained is not None:
                    self._operation_terminal_cache.move_to_end(cache_key)
                    self._metrics["operation_cache_hits"] += 1
            if retained is None:
                self._metrics["operation_cache_misses"] += 1
            if request_key in self._operation_snapshot_failures:
                raise SnapshotUnavailable("operation snapshot rebuild failed")
            loader = self._operation_snapshot_loader
            sequence_reader = self._operation_sequence_reader
        if retained is not None:
            return retained
        if loader is None or sequence_reader is None:
            raise KeyError(request_key)
        try:
            expected_sequence = sequence_reader(
                evaluation_id, case_id, attempt_id=attempt_id
            )
            page = loader(evaluation_id, case_id, attempt_id)
            _validate_operation_page(
                page,
                evaluation_id,
                case_id,
                attempt_id,
                expected_sequence=expected_sequence,
            )
        except FileNotFoundError:
            self._increment_metric("missing_operation_snapshots")
            self._schedule_operation_snapshot_rebuild(request_key)
            raise SnapshotRebuilding("operation snapshot rebuild started") from None
        except (OSError, TypeError, ValueError):
            self._increment_metric("damaged_operation_snapshots")
            self._schedule_operation_snapshot_rebuild(request_key)
            raise SnapshotRebuilding("operation snapshot rebuild started") from None
        self._cache_terminal_operations(page, request_key, expected_sequence)
        return page

    def finalize_operations(self, detail: CaseDetail) -> OperationPage:
        """Apply final artifact authority without changing runtime state."""
        if not isinstance(detail, CaseDetail):
            raise TypeError("detail must be a CaseDetail")
        case_key = (detail.evaluation_id, detail.case_id)
        key = (*case_key, detail.attempt_id or "")
        lock = self._case_lock(case_key)
        with lock:
            entry = self._entries.get(key)
            if (
                detail.attempt_id is None
                or self._current.get(case_key) != key
                or entry is None
                or entry.value.evaluation_id != detail.evaluation_id
                or entry.value.manifest_id != detail.manifest_id
                or entry.value.case_id != detail.case_id
                or entry.value.attempt_id != detail.attempt_id
            ):
                raise ValueError("operation finalization identity mismatch")
            if entry.phase != "terminal_pending":
                raise RuntimeError("Attempt is not terminal-pending")
            if entry.last_sequence != detail.last_sequence:
                raise ValueError("operation finalization sequence mismatch")
            return _operation_page(entry.operations, detail, entry.last_sequence)

    def mark_unavailable(
        self, key: RuntimeKey, epoch: int, reason: str
    ) -> RuntimeUpdate | None:
        """Publish an unavailable phase only for the captured Attempt epoch."""
        case_key = key[:2]
        lock = self._case_lock(case_key)
        with lock:
            entry = self._entries.get(key)
            if entry is None or entry.attempt_epoch != epoch or entry.phase == "sealed":
                return None
            revision = self._next_revision(case_key)
            replacement = replace(
                entry,
                phase="unavailable",
                runtime_revision=revision,
                unavailable_reason=reason,
            )
            self._entries[key] = replacement
        return _update(replacement, key[2])

    def get_case(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None = None,
    ) -> CaseDetail:
        """Return a hot value, one strict cold snapshot, or a rebuild state."""
        entry = self.get_entry(evaluation_id, case_id, attempt_id)
        if entry is not None:
            if entry.phase in {"fresh", "sealed"}:
                return self._decorate_case(entry.value)
            if entry.phase == "unavailable":
                raise SnapshotUnavailable(entry.unavailable_reason or "unavailable")
            raise SnapshotRebuilding(entry.phase)
        request_key = (evaluation_id, case_id, attempt_id)
        case_key = (evaluation_id, case_id)
        retained = None
        with self._meta_lock:
            cache_key = (
                self._snapshot_current.get(case_key)
                if attempt_id is None
                else (evaluation_id, case_id, attempt_id)
            )
            if cache_key is not None:
                retained = self._terminal_cache.get(cache_key)
                if retained is not None:
                    self._terminal_cache.move_to_end(cache_key)
                    self._metrics["cache_hits"] += 1
            if retained is None:
                self._metrics["cache_misses"] += 1
            if request_key in self._snapshot_failures:
                raise SnapshotUnavailable("terminal snapshot rebuild failed")
            loader = self._snapshot_loader
        if retained is not None:
            return self._decorate_case(retained)
        if loader is None:
            raise KeyError(request_key)
        try:
            detail = loader(evaluation_id, case_id, attempt_id)
            _validate_snapshot_detail(detail, evaluation_id, case_id, attempt_id)
        except FileNotFoundError:
            self._increment_metric("missing_snapshots")
            self._schedule_snapshot_rebuild(request_key)
            raise SnapshotRebuilding("terminal snapshot rebuild started") from None
        except (OSError, TypeError, ValueError):
            self._increment_metric("damaged_snapshots")
            self._schedule_snapshot_rebuild(request_key)
            raise SnapshotRebuilding("terminal snapshot rebuild started") from None
        if detail.attempt_id is None:
            return self._decorate_case(detail)
        self._cache_terminal(detail)
        return self._decorate_case(detail)

    def wait_for_snapshots(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._meta_lock:
                futures = (
                    *self._snapshot_flights.values(),
                    *self._operation_snapshot_flights.values(),
                )
            if not futures:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait(futures, timeout=remaining)

    def get_entry(
        self,
        evaluation_id: str,
        case_id: str,
        attempt_id: str | None = None,
    ) -> RuntimeEntry | None:
        case_key = (evaluation_id, case_id)
        lock = self._case_lock(case_key)
        with lock:
            key = (
                self._current.get(case_key)
                if attempt_id is None
                else (evaluation_id, case_id, attempt_id)
            )
            entry = None if key is None else self._entries.get(key)
            return (
                None
                if entry is None
                else replace(entry, operations=entry.operations.copy())
            )

    def case_summary(self, evaluation_id: str, case_id: str) -> CaseSummary | None:
        """Return the last immutable summary even while its Case is repairing."""
        entry = self.get_entry(evaluation_id, case_id)
        if entry is not None:
            return CaseSummary(
                **entry.value.model_dump(include=set(CaseSummary.model_fields))
            )
        with self._meta_lock:
            key = self._snapshot_current.get((evaluation_id, case_id))
            detail = None if key is None else self._terminal_cache.get(key)
        if detail is None:
            return None
        return CaseSummary(
            **detail.model_dump(include=set(CaseSummary.model_fields))
        )

    @property
    def repair_flight_count(self) -> int:
        with self._meta_lock:
            return len(self._repair_flights)

    def wait_for_repairs(self, *, timeout: float) -> bool:
        """Wait for the currently owned bounded repair set (test/shutdown helper)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._meta_lock:
                futures = tuple(self._repair_flights.values())
            if not futures:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait(futures, timeout=remaining)

    def probe_active_attempts(self) -> None:
        """Probe known active identities without discovering filesystem paths."""
        if self._high_water_reader is None:
            return
        with self._meta_lock:
            identities = tuple(self._identities.items())
        for key, identity in identities:
            case_key = key[:2]
            lock = self._case_lock(case_key)
            with lock:
                entry = self._entries.get(key)
                if (
                    entry is None
                    or self._current.get(case_key) != key
                    or entry.phase in {"sealed", "terminal_pending"}
                ):
                    continue
                epoch = entry.attempt_epoch
            try:
                high_water = self._high_water_reader(identity)
                self.observe_high_water(identity, high_water)
            except (OSError, TypeError, ValueError):
                update = self.mark_unavailable(
                    key, epoch, "journal_high_water_failed"
                )
                self._publish_update(update)

    def close(self) -> None:
        """Stop heartbeat ownership and join all bounded repair work."""
        with self._meta_lock:
            if self._closed:
                return
            self._closed = True
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=15)
        self._repair_executor.shutdown(wait=True, cancel_futures=False)

    def snapshot(self) -> dict[str, int]:
        with self._meta_lock:
            entries = len(self._entries)
            recordings = len(self._recordings)
            repair_queue = len(self._recording_flights)
            repairs = len(self._repair_flights)
            terminal_cache = len(self._terminal_cache)
            snapshot_flights = len(self._snapshot_flights)
            operation_terminal_cache = len(self._operation_terminal_cache)
            operation_snapshot_flights = len(self._operation_snapshot_flights)
            metrics = dict(self._metrics)
            terminal_entries = sum(
                entry.phase == "sealed" for entry in self._entries.values()
            )
            now = time.monotonic()
            stalled = sum(
                now - started >= _STALLED_REPAIR_SECONDS
                for started in self._repair_started.values()
            )
        return {
            "active_entries": entries - terminal_entries,
            "terminal_entries": terminal_entries,
            "capacity": self._capacity,
            "recording_declarations": recordings,
            "recording_flights": repair_queue,
            "repair_flights": repairs,
            "terminal_cache": terminal_cache,
            "snapshot_flights": snapshot_flights,
            "operation_terminal_cache": operation_terminal_cache,
            "operation_snapshot_flights": operation_snapshot_flights,
            "stalled_active_repairs": stalled,
            **metrics,
        }

    def note_snapshot_write_failure(self) -> None:
        self._increment_metric("snapshot_write_failures")

    def _decorate_case(self, detail: CaseDetail) -> CaseDetail:
        with self._meta_lock:
            decorator = self._case_decorator
        return detail if decorator is None else decorator(detail)

    def _schedule_snapshot_rebuild(
        self, request_key: tuple[str, str, str | None]
    ) -> None:
        with self._meta_lock:
            if (
                self._closed
                or request_key in self._snapshot_flights
                or request_key in self._snapshot_failures
                or self._snapshot_reconstructor is None
            ):
                return
            if not self._snapshot_slots.acquire(blocking=False):
                self._snapshot_failures.add(request_key)
                return
            self._metrics["snapshot_rebuilds"] += 1
            captured_current = self._current.get(request_key[:2])
            future = self._repair_executor.submit(
                self._rebuild_snapshot, request_key, captured_current
            )
            self._snapshot_flights[request_key] = future
        future.add_done_callback(
            lambda completed, selected=request_key: self._snapshot_finished(
                selected, completed
            )
        )

    def _rebuild_snapshot(
        self,
        request_key: tuple[str, str, str | None],
        captured_current: RuntimeKey | None,
    ) -> None:
        evaluation_id, case_id, attempt_id = request_key
        try:
            assert self._snapshot_reconstructor is not None
            detail = self._snapshot_reconstructor(
                evaluation_id, case_id, attempt_id=attempt_id
            )
            _validate_snapshot_detail(detail, evaluation_id, case_id, attempt_id)
        except (OSError, TypeError, ValueError, KeyError):
            with self._meta_lock:
                self._snapshot_failures.add(request_key)
            return
        with self._meta_lock:
            if (
                attempt_id is None
                and self._current.get(request_key[:2]) != captured_current
            ):
                return
        if self._snapshot_persister is not None:
            try:
                self._snapshot_persister(detail)
            except (OSError, TypeError, ValueError, KeyError):
                self._increment_metric("snapshot_write_failures")
                return
        self._cache_terminal(detail)

    def _snapshot_finished(
        self,
        request_key: tuple[str, str, str | None],
        _future: Future[None],
    ) -> None:
        with self._meta_lock:
            self._snapshot_flights.pop(request_key, None)
        self._snapshot_slots.release()

    def _cache_terminal(self, detail: CaseDetail) -> None:
        assert detail.attempt_id is not None
        key = (detail.evaluation_id, detail.case_id, detail.attempt_id)
        case_key = key[:2]
        with self._meta_lock:
            self._terminal_cache[key] = detail
            self._terminal_cache.move_to_end(key)
            self._snapshot_current[case_key] = key
            self._snapshot_failures.discard((*case_key, None))
            self._snapshot_failures.discard(key)
            while len(self._terminal_cache) > self._terminal_capacity:
                evicted, _detail = self._terminal_cache.popitem(last=False)
                self._metrics["cache_evictions"] += 1
                if self._snapshot_current.get(evicted[:2]) == evicted:
                    self._snapshot_current.pop(evicted[:2], None)

    def _schedule_operation_snapshot_rebuild(
        self, request_key: tuple[str, str, str | None]
    ) -> None:
        with self._meta_lock:
            if (
                self._closed
                or request_key in self._operation_snapshot_flights
                or request_key in self._operation_snapshot_failures
                or self._operation_snapshot_reconstructor is None
                or self._operation_sequence_reader is None
            ):
                return
            if not self._snapshot_slots.acquire(blocking=False):
                self._operation_snapshot_failures.add(request_key)
                return
            self._metrics["operation_snapshot_rebuilds"] += 1
            captured_current = self._current.get(request_key[:2])
            future = self._repair_executor.submit(
                self._rebuild_operation_snapshot,
                request_key,
                captured_current,
            )
            self._operation_snapshot_flights[request_key] = future
        future.add_done_callback(
            lambda completed, selected=request_key: self._operation_snapshot_finished(
                selected, completed
            )
        )

    def _rebuild_operation_snapshot(
        self,
        request_key: tuple[str, str, str | None],
        captured_current: RuntimeKey | None,
    ) -> None:
        evaluation_id, case_id, attempt_id = request_key
        try:
            assert self._operation_snapshot_reconstructor is not None
            assert self._operation_sequence_reader is not None
            try:
                expected_sequence = self._operation_sequence_reader(
                    evaluation_id, case_id, attempt_id=attempt_id
                )
            except ValueError:
                expected_sequence = self._recover_operation_sequence(request_key)
                if expected_sequence is None:
                    return
            page = None
            if self._operation_snapshot_loader is not None:
                try:
                    loaded = self._operation_snapshot_loader(
                        evaluation_id, case_id, attempt_id
                    )
                    _validate_operation_page(
                        loaded,
                        evaluation_id,
                        case_id,
                        attempt_id,
                        expected_sequence=expected_sequence,
                    )
                    page = loaded
                except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
                    pass
            if page is None:
                page = self._operation_snapshot_reconstructor(
                    evaluation_id, case_id, attempt_id=attempt_id
                )
            _validate_operation_page(
                page,
                evaluation_id,
                case_id,
                attempt_id,
                expected_sequence=expected_sequence,
            )
        except (OSError, TypeError, ValueError, KeyError):
            with self._meta_lock:
                self._operation_snapshot_failures.add(request_key)
            return
        with self._meta_lock:
            if (
                attempt_id is None
                and self._current.get(request_key[:2]) != captured_current
            ):
                return
        if self._operation_snapshot_persister is not None:
            try:
                self._operation_snapshot_persister(page)
            except (OSError, TypeError, ValueError, KeyError):
                self._increment_metric("snapshot_write_failures")
                return
        self._cache_terminal_operations(page, request_key, expected_sequence)

    def _recover_operation_sequence(
        self, request_key: tuple[str, str, str | None]
    ) -> int | None:
        if self._snapshot_reconstructor is None:
            raise ValueError("accepted final sequence is unavailable")
        evaluation_id, case_id, attempt_id = request_key
        detail = self._snapshot_reconstructor(
            evaluation_id, case_id, attempt_id=attempt_id
        )
        _validate_snapshot_detail(detail, evaluation_id, case_id, attempt_id)
        if detail.attempt_id is None:
            raise ValueError("accepted final sequence is unavailable")
        if self._snapshot_persister is not None:
            try:
                self._snapshot_persister(detail)
            except (OSError, TypeError, ValueError, KeyError):
                self._increment_metric("snapshot_write_failures")
                return None
        self._cache_terminal(detail)
        assert self._operation_sequence_reader is not None
        return self._operation_sequence_reader(
            evaluation_id, case_id, attempt_id=detail.attempt_id
        )

    def _operation_snapshot_finished(
        self,
        request_key: tuple[str, str, str | None],
        _future: Future[None],
    ) -> None:
        with self._meta_lock:
            self._operation_snapshot_flights.pop(request_key, None)
        self._snapshot_slots.release()

    def _cache_terminal_operations(
        self,
        page: OperationPage,
        request_key: tuple[str, str, str | None],
        expected_sequence: int,
    ) -> None:
        _validate_operation_page(
            page,
            request_key[0],
            request_key[1],
            request_key[2],
            expected_sequence=expected_sequence,
        )
        key = (page.evaluation_id, page.case_id, page.attempt_id)
        case_key = key[:2]
        with self._meta_lock:
            self._operation_terminal_cache[key] = page
            self._operation_terminal_cache.move_to_end(key)
            self._operation_snapshot_current[case_key] = key
            self._operation_snapshot_failures.discard((*case_key, None))
            self._operation_snapshot_failures.discard(key)
            while len(self._operation_terminal_cache) > self._terminal_capacity:
                evicted, _page = self._operation_terminal_cache.popitem(last=False)
                self._metrics["operation_cache_evictions"] += 1
                if self._operation_snapshot_current.get(evicted[:2]) == evicted:
                    self._operation_snapshot_current.pop(evicted[:2], None)

    def _schedule_repair(self, key: RuntimeKey, epoch: int) -> None:
        if self._journal_reader is None or self._high_water_reader is None:
            return
        with self._meta_lock:
            if self._closed or key in self._repair_flights:
                return
            if not self._repair_slots.acquire(blocking=False):
                overflow = True
            else:
                overflow = False
                future = self._repair_executor.submit(self._repair, key, epoch)
                self._repair_flights[key] = future
                future.add_done_callback(
                    lambda completed, selected=key: self._repair_finished(
                        selected, completed
                    )
                )
        if overflow:
            with self._meta_lock:
                self._repair_started.pop(key, None)
            update = self.mark_unavailable(key, epoch, "repair_capacity_exceeded")
            self._publish_update(update)

    def _repair_finished(self, key: RuntimeKey, _future: Future[None]) -> None:
        with self._meta_lock:
            self._repair_flights.pop(key, None)
            self._repair_started.pop(key, None)
        self._repair_slots.release()

    def _repair(self, key: RuntimeKey, epoch: int) -> None:
        case_key = key[:2]
        identity = self._identities[key]
        while True:
            self._increment_metric("repair_passes")
            lock = self._case_lock(case_key)
            with lock:
                entry = self._entries.get(key)
                if (
                    entry is None
                    or self._current.get(case_key) != key
                    or entry.phase != "repairing"
                    or entry.attempt_epoch != epoch
                ):
                    return
                base_value = entry.value
                base_sequence = entry.last_sequence
                observed = entry.observed_high_water
            with self._meta_lock:
                recording = self._recordings.get(key)
            if recording is None:
                self._repair_failed(key, epoch, "recording_declaration_unavailable")
                return
            try:
                assert self._journal_reader is not None
                journal = self._journal_reader(
                    identity, through_sequence=observed
                )
                if not isinstance(journal, JournalRead):
                    raise TypeError("journal reader returned an invalid value")
                value = base_value
                operations = OperationProjection()
                sequence = base_sequence
                for event in journal.events:
                    operations.consume(event.to_payload())
                    if event.sequence <= sequence:
                        continue
                    if event.sequence != sequence + 1:
                        raise ValueError("journal sequence is not contiguous")
                    value = _reduce_value(value, event, recording)
                    sequence = event.sequence
                if journal.high_water != sequence or journal.high_water < base_sequence:
                    self._increment_metric("high_water_mismatches")
                    raise ValueError("journal high-water mismatch")
            except (OSError, TypeError, ValueError):
                self._repair_failed(key, epoch, "journal_read_failed")
                return
            with lock:
                retained = self._entries.get(key)
                if (
                    retained is None
                    or self._current.get(case_key) != key
                    or retained.phase != "repairing"
                    or retained.attempt_epoch != epoch
                ):
                    return
                with self._meta_lock:
                    latest_recording = self._recordings.get(key)
                if latest_recording != recording:
                    continue
                self._entries[key] = replace(
                    retained,
                    value=value,
                    operations=operations,
                    last_sequence=sequence,
                )
            try:
                assert self._high_water_reader is not None
                durable = self._high_water_reader(identity)
                if isinstance(durable, bool) or not isinstance(durable, int) or durable < 0:
                    raise ValueError("journal high-water is invalid")
            except (OSError, TypeError, ValueError):
                self._repair_failed(key, epoch, "journal_high_water_failed")
                return
            if durable > sequence:
                with lock:
                    retained = self._entries.get(key)
                    if retained is None or retained.attempt_epoch != epoch:
                        return
                    self._entries[key] = replace(
                        retained,
                        observed_high_water=max(
                            retained.observed_high_water, durable
                        ),
                    )
                continue
            with lock:
                retained = self._entries.get(key)
                if (
                    retained is None
                    or self._current.get(case_key) != key
                    or retained.phase != "repairing"
                    or retained.attempt_epoch != epoch
                ):
                    return
                with self._meta_lock:
                    latest_recording = self._recordings.get(key)
                if latest_recording != recording:
                    continue
                revision = self._next_revision(case_key)
                replacement = replace(
                    retained,
                    phase="fresh",
                    runtime_revision=revision,
                    unavailable_reason=None,
                )
                self._entries[key] = replacement
            self._publish_update(_update(replacement, key[2]))
            return

    def _repair_failed(self, key: RuntimeKey, epoch: int, reason: str) -> None:
        update = self.mark_unavailable(key, epoch, reason)
        self._publish_update(update)

    def _publish_update(self, update: RuntimeUpdate | None) -> None:
        if update is None or self._update_publisher is None:
            return
        try:
            self._update_publisher(update)
        except Exception:
            pass

    def _increment_metric(self, name: str) -> None:
        with self._meta_lock:
            self._metrics[name] += 1

    def _heartbeat(self, interval: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            self.probe_active_attempts()

    def _start_heartbeat(self, interval: float) -> None:
        if not 0 < interval <= 15:
            raise ValueError("heartbeat_interval must be between 0 and 15 seconds")
        if self._high_water_reader is None:
            raise ValueError("heartbeat requires a high-water reader")
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            args=(interval,),
            name="rpent-case-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _case_lock(self, key: CaseKey) -> threading.Lock:
        with self._meta_lock:
            return self._locks.setdefault(key, threading.Lock())

    def _next_revision(self, key: CaseKey) -> int:
        with self._meta_lock:
            value = self._revisions.get(key, 0) + 1
            self._revisions[key] = value
            return value

    def _next_epoch(self, key: CaseKey) -> int:
        with self._meta_lock:
            value = self._epochs.get(key, 0) + 1
            self._epochs[key] = value
            return value

    def _result_entry(
        self, result: EpisodeResult
    ) -> tuple[RuntimeKey, CaseKey, threading.Lock]:
        if not isinstance(result, EpisodeResult):
            raise TypeError("result must be an EpisodeResult")
        case_key = (result.evaluation_id, result.case_id)
        key = (*case_key, result.attempt_id)
        lock = self._case_lock(case_key)
        if self._current.get(case_key) != key or key not in self._entries:
            raise ValueError("result Attempt is not current")
        return key, case_key, lock


def _validate_registration(
    registration: AttemptRegistration,
    manifest: EvaluationManifest,
    case: EpisodeCase,
) -> None:
    if not isinstance(registration, AttemptRegistration):
        raise TypeError("registration must be an AttemptRegistration")
    if not isinstance(manifest, EvaluationManifest) or not isinstance(case, EpisodeCase):
        raise TypeError("manifest and case must use evaluation contracts")
    if (
        registration.evaluation_id != manifest.evaluation_id
        or registration.manifest_id != manifest.manifest_id
        or registration.case_id != case.case_id
        or case not in manifest.cases
    ):
        raise ValueError("Attempt registration scope mismatch")


def _plain_schema_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_schema_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_schema_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_schema_payload(item) for item in value)
    return value


def _base_detail(
    registration: AttemptRegistration,
    manifest: EvaluationManifest,
    case: EpisodeCase,
) -> CaseDetail:
    declared = manifest.contracts.get("data_schema")
    if isinstance(declared, Mapping):
        schema = CaseDataSchemaView.model_validate(_plain_schema_payload(declared))
    else:
        schema = CaseDataSchemaView.model_validate(
            build_data_schema(cameras=(), action_shape=(), state_shape=()).to_payload()
        )
    return CaseDetail(
        case_id=case.case_id,
        ordinal=case.ordinal,
        task_name=case.task_name,
        task_index=case.task_index,
        episode_index=case.episode_index,
        seed=case.seed,
        execution_state="starting",
        attempt_id=registration.attempt_id,
        episode_result=None,
        artifact_health=None,
        current_step=None,
        started_at=registration.registered_at,
        finished_at=None,
        run_summary=None,
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        last_sequence=0,
        case=case.to_payload(),
        data_schema=schema,
    )


def _reduce_value(
    value: CaseDetail,
    event: EventEnvelope,
    recording: RecordingDeclaration,
) -> CaseDetail:
    return _recording_value(value, recording).model_copy(
        update={
            "execution_state": "running",
            "last_sequence": event.sequence,
            "current_step": event.step_id,
        }
    )


def _recording_value(
    value: CaseDetail, recording: RecordingDeclaration
) -> CaseDetail:
    return value.model_copy(
        update={
            "recording_profile": recording.profile,
            "cameras": recording.cameras,
            "retained_camera_ids": recording.retained_camera_ids,
            "data_schema": recording.data_schema,
            "artifacts": recording.artifacts,
        }
    )


def _operation_page(
    projection: OperationProjection,
    detail: CaseDetail,
    last_sequence: int,
) -> OperationPage:
    valid_camera_clip_artifacts = frozenset(
        artifact.artifact_id
        for artifact in detail.artifacts
        if artifact.kind == "action_camera_clip"
        and artifact.available
        and artifact.byte_size is not None
        and artifact.byte_size > 0
    )
    operations = projection.copy().summaries(valid_camera_clip_artifacts)
    steps = [
        step
        for operation in operations
        for step in (operation.first_step_id, operation.last_step_id)
        if step is not None
    ]
    assert detail.attempt_id is not None
    return OperationPage(
        evaluation_id=detail.evaluation_id,
        case_id=detail.case_id,
        attempt_id=detail.attempt_id,
        last_sequence=last_sequence,
        incomplete_tail=False,
        first_step_id=min(steps) if steps else None,
        latest_step_id=max(steps) if steps else None,
        operations=operations,
    )


def _validate_snapshot_detail(
    detail: CaseDetail,
    evaluation_id: str,
    case_id: str,
    attempt_id: str | None,
) -> None:
    if not isinstance(detail, CaseDetail):
        raise TypeError("snapshot collaborator returned an invalid CaseDetail")
    if attempt_id is None and detail.attempt_id is None:
        if detail.episode_result is not None:
            raise ValueError("snapshot collaborator identity mismatch")
        return
    if (
        detail.evaluation_id != evaluation_id
        or detail.case_id != case_id
        or detail.attempt_id is None
        or (attempt_id is not None and detail.attempt_id != attempt_id)
        or detail.episode_result is None
    ):
        raise ValueError("snapshot collaborator identity mismatch")


def _validate_operation_page(
    page: OperationPage,
    evaluation_id: str,
    case_id: str,
    attempt_id: str | None,
    *,
    expected_sequence: int | None = None,
) -> None:
    if not isinstance(page, OperationPage):
        raise TypeError("snapshot collaborator returned an invalid OperationPage")
    if (
        page.evaluation_id != evaluation_id
        or page.case_id != case_id
        or (attempt_id is not None and page.attempt_id != attempt_id)
    ):
        raise ValueError("operation snapshot collaborator identity mismatch")
    if expected_sequence is not None and (
        page.last_sequence != expected_sequence or page.incomplete_tail
    ):
        raise ValueError("operation snapshot collaborator terminal mismatch")


def _update(
    entry: RuntimeEntry,
    attempt_id: str,
    event: EventEnvelope | None = None,
) -> RuntimeUpdate:
    return RuntimeUpdate(
        entry.value.evaluation_id,
        entry.value.case_id,
        attempt_id,
        entry.runtime_revision,
        event,
    )


__all__ = [
    "CaseKey",
    "CaseRuntimeStore",
    "JournalRead",
    "RecordingDeclaration",
    "RuntimeDisposition",
    "RuntimeEntry",
    "RuntimeKey",
    "RuntimePhase",
    "RuntimeReduction",
    "RuntimeUpdate",
]
