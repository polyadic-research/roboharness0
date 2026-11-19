from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

import rpent.dashboard.attachment as attachment
from rpent.dashboard.attachment import (
    DashboardAlreadyRunning,
    DashboardDescriptor,
    DashboardLease,
)
from rpent.evaluation.attempt_registration import register_attempt
from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.repository import EvaluationRepository
from tests.test_case_runtime import _operation_events
from tests.test_ui_repository import (
    _append_jsonl,
    _event,
    _result,
    _write_evaluation,
    _write_recording_discovery,
    _write_result,
)


def _valid_payload(runs_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance_id": uuid4().hex,
        "pid": os.getpid(),
        "origin": "http://127.0.0.1:43123",
        "runs_root": str(runs_root.resolve()),
        "notification_token": "worker-secret",
        "started_at": "2026-08-14T03:04:05+00:00",
    }


def _write_descriptor(path: Path, payload: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(mode)


def _attempt_hint() -> attachment.LifecycleHint:
    return attachment.LifecycleHint(
        schema_version=1,
        kind="attempt_registered",
        evaluation_id="evaluation-1",
        manifest_id="manifest-1",
        case_id="case-1",
        attempt_id="attempt-1",
    )


def test_refresher_initialization_performs_correlated_durable_refresh() -> None:
    class Repository:
        def __init__(self) -> None:
            self.refreshes = 0

        def index_fingerprint(self):
            return ((1, 2, 3, 4),)

        def refresh_index(self, *, force: bool, recover: bool) -> None:
            assert force is True
            assert recover is False
            self.refreshes += 1

        def startup_runtime_registrations(self):
            return ()

    repository = Repository()

    attachment.DashboardAttachmentRefresher(repository, object())

    assert repository.refreshes == 1


def test_refresher_bounds_changing_fingerprints_then_retries_next_tick() -> None:
    class Repository:
        def __init__(self) -> None:
            self.generation = 1
            self.changing = False
            self.observations = 0
            self.observation_limit: int | None = None
            self.refreshes = 0

        def index_fingerprint(self):
            self.observations += 1
            if (
                self.observation_limit is not None
                and self.observations > self.observation_limit
            ):
                raise AssertionError("correlated refresh did not stop")
            if self.changing:
                self.generation += 1
            return ((1, self.generation, 1, self.generation),)

        def refresh_index(self, *, force: bool, recover: bool) -> None:
            assert force is True
            assert recover is False
            self.refreshes += 1

        def startup_runtime_registrations(self):
            return ()

    repository = Repository()
    refresher = attachment.DashboardAttachmentRefresher(repository, object())
    retained = refresher._fingerprint
    initial_refreshes = repository.refreshes
    repository.changing = True
    repository.observations = 0
    repository.observation_limit = 5

    refresher.refresh()

    assert attachment._MAX_CORRELATED_REFRESH_ATTEMPTS == 2
    assert repository.refreshes - initial_refreshes == 2
    assert refresher._fingerprint == retained

    repository.changing = False
    repository.generation = 99
    repository.observations = 0
    repository.observation_limit = None
    exhausted_refreshes = repository.refreshes

    refresher.refresh_if_changed()

    assert repository.refreshes - exhausted_refreshes == 1
    assert refresher._fingerprint == ((1, 99, 1, 99),)


def test_refresher_retries_external_index_replace_during_refresh(
    tmp_path: Path,
) -> None:
    repository = EvaluationRepository((tmp_path,))
    loaded = threading.Event()
    release = threading.Event()

    class BarrierRepository:
        def __init__(self) -> None:
            self.block_next = False

        def index_fingerprint(self):
            return repository.index_fingerprint()

        def refresh_index(self, *, force: bool, recover: bool) -> None:
            repository.refresh_index(force=force, recover=recover)
            if self.block_next:
                self.block_next = False
                loaded.set()
                assert release.wait(2)

        def startup_runtime_registrations(self):
            return repository.startup_runtime_registrations()

    barrier_repository = BarrierRepository()
    runtime_store = CaseRuntimeStore()
    repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        barrier_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    first_root = tmp_path / "external-one"
    first, _ = _write_evaluation(
        first_root,
        evaluation_id="evaluation-one",
        persist_index=False,
    )
    external_repository.register_evaluation(first_root, first)
    barrier_repository.block_next = True
    errors = []

    def refresh() -> None:
        try:
            refresher.refresh_if_changed()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=refresh)
    thread.start()
    assert loaded.wait(2)
    second_root = tmp_path / "external-two"
    second, _ = _write_evaluation(
        second_root,
        evaluation_id="evaluation-two",
        persist_index=False,
    )
    external_repository.register_evaluation(second_root, second)
    release.set()
    thread.join(2)

    try:
        assert not thread.is_alive()
        assert errors == []
        refresher.refresh_if_changed()
        assert [
            item.evaluation_id for item in repository.list_evaluations()
        ] == ["evaluation-one", "evaluation-two"]
    finally:
        release.set()
        runtime_store.close()


def test_refresher_recovers_external_registration_after_missed_hint(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-external",
        "2026-08-14T03:04:05+00:00",
    )
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        assert dashboard_repository.list_evaluations() == ()
        assert runtime_store.get_entry(manifest.evaluation_id, case.case_id) is None

        refresher.refresh_if_changed()

        assert [
            item.evaluation_id for item in dashboard_repository.list_evaluations()
        ] == [manifest.evaluation_id]
        entry = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert entry is not None
        assert entry.value.attempt_id == registration.attempt_id
    finally:
        runtime_store.close()


def test_refresher_reprobes_active_case_when_locator_arrives_after_index(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore(
        journal_reader=dashboard_repository.read_event_journal,
        high_water_reader=dashboard_repository.read_event_high_water,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    dashboard_repository.bind_runtime_store(runtime_store)
    runtime_store.configure_startup(
        registration_reader=dashboard_repository.startup_runtime_registrations,
        recording_loader=dashboard_repository.load_recording_declaration,
    )
    runtime_store.startup_recover()
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)

    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )
    refresher.refresh_if_changed()
    assert runtime_store.get_entry(manifest.evaluation_id, case.case_id) is None

    registration = register_attempt(
        manifest,
        case,
        "attempt-after-index",
        "2026-08-14T03:04:05+00:00",
    )
    attempt = root / "cases" / case.case_id / "attempts" / registration.attempt_id
    _write_recording_discovery(attempt, manifest, case)
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, registration.attempt_id, 1, 9),
    )

    try:
        # The attempt hint is deliberately dropped and the index is unchanged.
        refresher.refresh_if_changed()

        entry = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert entry is not None
        assert entry.value.attempt_id == registration.attempt_id
        assert entry.last_sequence == 1
        assert entry.value.current_step == 9
    finally:
        runtime_store.close()


def test_refresher_keeps_running_retry_unresolved_after_cancelled_result(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore(
        journal_reader=dashboard_repository.read_event_journal,
        high_water_reader=dashboard_repository.read_event_high_water,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    dashboard_repository.bind_runtime_store(runtime_store)
    runtime_store.configure_startup(
        registration_reader=dashboard_repository.startup_runtime_registrations,
        recording_loader=dashboard_repository.load_recording_declaration,
    )
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    old = register_attempt(
        manifest,
        case,
        "attempt-old",
        "2026-08-06T00:00:00+00:00",
    )
    _write_result(root, _result(manifest, case, old.attempt_id, status="cancelled"))
    external_repository = EvaluationRepository((tmp_path,))
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        refresher.refresh_if_changed()
        observed_state = dashboard_repository.list_evaluations()[0].execution_state

        new = register_attempt(
            manifest,
            case,
            "attempt-new",
            "2026-08-14T03:04:05+00:00",
        )
        attempt = root / "cases" / case.case_id / "attempts" / new.attempt_id
        _write_recording_discovery(attempt, manifest, case)

        # The new-attempt hint is deliberately dropped and the index is unchanged.
        refresher.refresh_if_changed()

        entry = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert observed_state == "running"
        assert dashboard_repository.list_evaluations()[0].execution_state == "running"
        assert entry is not None
        assert entry.value.attempt_id == new.attempt_id
    finally:
        runtime_store.close()


def test_refresher_immediately_recovers_external_durable_high_water(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore(
        journal_reader=dashboard_repository.read_event_journal,
        high_water_reader=dashboard_repository.read_event_high_water,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    dashboard_repository.bind_runtime_store(runtime_store)
    runtime_store.configure_startup(
        registration_reader=dashboard_repository.startup_runtime_registrations,
        recording_loader=dashboard_repository.load_recording_declaration,
    )
    runtime_store.startup_recover()
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-external",
        "2026-08-14T03:04:05+00:00",
    )
    attempt = (
        root
        / "cases"
        / case.case_id
        / "attempts"
        / registration.attempt_id
    )
    _write_recording_discovery(attempt, manifest, case)
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, registration.attempt_id, 1, 7),
    )
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        refresher.refresh_if_changed()

        entry = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert entry is not None
        assert entry.phase == "fresh"
        assert entry.last_sequence == 1
        assert entry.value.current_step == 7
    finally:
        runtime_store.close()


def test_refresher_duplicate_hint_preserves_external_attempt_epoch(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-external",
        "2026-08-14T03:04:05+00:00",
    )
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )
    hint = attachment.LifecycleHint(
        1,
        "attempt_registered",
        manifest.evaluation_id,
        manifest.manifest_id,
        case.case_id,
        registration.attempt_id,
    )

    try:
        refresher.accept(hint)
        initial = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert initial is not None

        refresher.accept(hint)
        refresher.accept(hint)

        retained = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert retained is not None
        assert retained.attempt_epoch == initial.attempt_epoch
        assert retained.runtime_revision == initial.runtime_revision
    finally:
        runtime_store.close()


def test_refresher_does_not_restart_high_water_recovery_for_live_hint_target(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    high_water_reads = []
    runtime_store = CaseRuntimeStore(
        journal_reader=dashboard_repository.read_event_journal,
        high_water_reader=lambda identity: (
            high_water_reads.append(identity)
            or dashboard_repository.read_event_high_water(identity)
        ),
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    dashboard_repository.bind_runtime_store(runtime_store)
    runtime_store.configure_startup(
        registration_reader=dashboard_repository.startup_runtime_registrations,
        recording_loader=dashboard_repository.load_recording_declaration,
    )
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-live",
        "2026-08-14T03:04:05+00:00",
    )
    attempt = root / "cases" / case.case_id / "attempts" / registration.attempt_id
    _write_recording_discovery(attempt, manifest, case)
    EvaluationRepository((tmp_path,)).register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        refresher.accept(
            attachment.LifecycleHint(
                1,
                "attempt_registered",
                manifest.evaluation_id,
                manifest.manifest_id,
                case.case_id,
                registration.attempt_id,
            )
        )
        refresher.refresh_if_changed()

        entry = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert entry is not None
        assert entry.value.attempt_id == registration.attempt_id
        assert high_water_reads == []
    finally:
        runtime_store.close()


def test_result_hint_rejects_unknown_identity_without_loading_other_result(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    result = _result(manifest, case, "attempt-durable")
    _write_result(root, result)
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="terminal",
    )

    try:
        with pytest.raises(KeyError):
            refresher.accept(
                attachment.LifecycleHint(
                    1,
                    "result_accepted",
                    "untrusted-evaluation",
                    "untrusted-manifest",
                    "untrusted-case",
                    "untrusted-attempt",
                )
            )

        assert tuple(
            item.evaluation_id for item in dashboard_repository.list_evaluations()
        ) == (manifest.evaluation_id,)
        assert runtime_store.get_entry(manifest.evaluation_id, case.case_id) is None
    finally:
        runtime_store.close()


def test_attempt_hint_rejects_manifest_and_attempt_identity_mismatch(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-durable",
        "2026-08-14T03:04:05+00:00",
    )
    EvaluationRepository((tmp_path,)).register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        for manifest_id, attempt_id in (
            ("wrong-manifest", registration.attempt_id),
            (manifest.manifest_id, "wrong-attempt"),
        ):
            with pytest.raises(ValueError):
                refresher.accept(
                    attachment.LifecycleHint(
                        1,
                        "attempt_registered",
                        manifest.evaluation_id,
                        manifest_id,
                        case.case_id,
                        attempt_id,
                    )
                )
        assert runtime_store.get_entry(manifest.evaluation_id, case.case_id) is None
    finally:
        runtime_store.close()


def test_attempt_hint_recovers_only_its_durable_target(tmp_path: Path) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    registrations = []
    for index in (1, 2):
        root = tmp_path / f"external-{index}"
        manifest, case = _write_evaluation(
            root,
            evaluation_id=f"evaluation-{index}",
            persist_index=False,
        )
        registration = register_attempt(
            manifest,
            case,
            f"attempt-{index}",
            "2026-08-14T03:04:05+00:00",
        )
        external_repository.register_evaluation(
            root,
            manifest,
            execution_state="running",
        )
        registrations.append((registration, manifest, case))
    selected, selected_manifest, selected_case = registrations[0]
    other, other_manifest, other_case = registrations[1]

    try:
        refresher.accept(
            attachment.LifecycleHint(
                1,
                "attempt_registered",
                selected.evaluation_id,
                selected.manifest_id,
                selected.case_id,
                selected.attempt_id,
            )
        )

        assert runtime_store.get_entry(
            selected_manifest.evaluation_id,
            selected_case.case_id,
        ) is not None
        assert runtime_store.get_entry(
            other_manifest.evaluation_id,
            other_case.case_id,
        ) is None
        assert other.attempt_id == "attempt-2"
    finally:
        runtime_store.close()


def test_result_hint_requires_durable_result_and_seals_idempotently(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-durable",
        "2026-08-14T03:04:05+00:00",
    )
    EvaluationRepository((tmp_path,)).register_evaluation(
        root,
        manifest,
        execution_state="running",
    )
    attempt_hint = attachment.LifecycleHint(
        1,
        "attempt_registered",
        manifest.evaluation_id,
        manifest.manifest_id,
        case.case_id,
        registration.attempt_id,
    )
    result_hint = attachment.LifecycleHint(
        1,
        "result_accepted",
        manifest.evaluation_id,
        manifest.manifest_id,
        case.case_id,
        registration.attempt_id,
    )

    try:
        refresher.accept(attempt_hint)
        with pytest.raises(ValueError):
            refresher.accept(result_hint)

        result = _result(manifest, case, registration.attempt_id)
        attempt = root / "cases" / case.case_id / "attempts" / registration.attempt_id
        _append_jsonl(
            attempt / "events.jsonl",
            *(event.to_payload() for event in _operation_events(registration)),
        )
        _write_result(root, result)
        refresher.accept(result_hint)
        sealed = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert sealed is not None and sealed.phase == "sealed"
        assert sealed.value.episode_result == result.to_payload()
        assert [
            operation.operation_id
            for operation in runtime_store.get_operations(
                manifest.evaluation_id,
                case.case_id,
                registration.attempt_id,
            ).operations
        ] == ["operation-1"]

        refresher.accept(result_hint)
        duplicate = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert duplicate is not None
        assert duplicate.attempt_epoch == sealed.attempt_epoch
        assert duplicate.runtime_revision == sealed.runtime_revision
    finally:
        runtime_store.close()


def test_delayed_historical_result_hint_preserves_durable_current_attempt(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    old = register_attempt(
        manifest,
        case,
        "attempt-old",
        "2026-08-06T00:00:00+00:00",
    )
    _write_result(root, _result(manifest, case, old.attempt_id))
    new = register_attempt(
        manifest,
        case,
        "attempt-new",
        "2026-08-14T03:04:05+00:00",
    )
    EvaluationRepository((tmp_path,)).register_evaluation(
        root,
        manifest,
        execution_state="running",
    )
    hint = attachment.LifecycleHint(
        1,
        "result_accepted",
        manifest.evaluation_id,
        manifest.manifest_id,
        case.case_id,
        old.attempt_id,
    )

    try:
        refresher.accept(hint)
        first = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert first is not None
        assert first.value.attempt_id == new.attempt_id
        assert dashboard_repository.authorize_attempt(
            manifest.evaluation_id,
            case.case_id,
            old.attempt_id,
        ).attempt_id == old.attempt_id

        refresher.accept(hint)
        duplicate = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert duplicate is not None
        assert duplicate.value.attempt_id == new.attempt_id
        assert duplicate.attempt_epoch == first.attempt_epoch
        assert duplicate.runtime_revision == first.runtime_revision
    finally:
        runtime_store.close()


def test_refresher_seals_active_runtime_from_durable_terminal_result(
    tmp_path: Path,
) -> None:
    dashboard_repository = EvaluationRepository((tmp_path,))
    runtime_store = CaseRuntimeStore()
    dashboard_repository.bind_runtime_store(runtime_store)
    refresher = attachment.DashboardAttachmentRefresher(
        dashboard_repository,
        runtime_store,
    )
    external_repository = EvaluationRepository((tmp_path,))
    root = tmp_path / "external-evaluation"
    manifest, case = _write_evaluation(root, persist_index=False)
    registration = register_attempt(
        manifest,
        case,
        "attempt-external",
        "2026-08-14T03:04:05+00:00",
    )
    external_repository.register_evaluation(
        root,
        manifest,
        execution_state="running",
    )

    try:
        refresher.refresh_if_changed()
        active = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert active is not None and active.phase == "fresh"

        result = _result(manifest, case, registration.attempt_id)
        _write_result(root, result)
        external_repository.update_evaluation_list_item(
            manifest.evaluation_id,
            execution_state="terminal",
        )
        refresher.refresh_if_changed()

        terminal = runtime_store.get_entry(manifest.evaluation_id, case.case_id)
        assert terminal is not None
        assert terminal.phase == "sealed"
        assert terminal.value.execution_state == "terminal"
        assert terminal.value.episode_result == result.to_payload()
    finally:
        runtime_store.close()


def test_external_fingerprint_watcher_retries_and_closes_idempotently() -> None:
    refreshed = threading.Event()

    class Refresher:
        def __init__(self) -> None:
            self.calls = 0

        def refresh_if_changed(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("transient index failure")
            refreshed.set()

    refresher = Refresher()
    watcher = attachment.IndexFingerprintWatcher(refresher)
    watcher.start()
    thread = watcher._thread

    assert thread is not None and thread.daemon
    assert refreshed.wait(1.5)
    watcher.start()
    assert watcher._thread is thread

    started = time.monotonic()
    watcher.close()
    watcher.close()

    assert time.monotonic() - started < 2.1
    assert not thread.is_alive()


def test_lifecycle_hint_is_immutable_and_serializes_only_identity() -> None:
    hint = _attempt_hint()

    assert hint.to_payload() == {
        "schema_version": 1,
        "kind": "attempt_registered",
        "evaluation_id": "evaluation-1",
        "manifest_id": "manifest-1",
        "case_id": "case-1",
        "attempt_id": "attempt-1",
    }
    with pytest.raises(FrozenInstanceError):
        hint.attempt_id = "replacement"  # type: ignore[misc]


def test_lifecycle_publisher_returns_after_bounded_enqueue(tmp_path: Path) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(descriptor_path, _valid_payload(tmp_path))
    entered = threading.Event()
    release = threading.Event()

    async def post(*_args, **_kwargs):
        entered.set()
        await asyncio.to_thread(release.wait)
        return httpx.Response(204)

    publisher = attachment.LifecycleHintPublisher(
        descriptor_path,
        expected_runs_root=tmp_path,
        post=post,
    )
    started = time.monotonic()

    assert publisher.publish(_attempt_hint()) is True
    assert time.monotonic() - started < 0.05
    assert entered.wait(1)
    release.set()
    publisher.close()


def test_lifecycle_publisher_reloads_private_descriptor_for_every_delivery(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    calls = []
    delivered = threading.Event()

    async def post(url, **kwargs):
        calls.append((url, kwargs))
        delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    first = _valid_payload(tmp_path)
    first["origin"] = "http://127.0.0.1:8001"
    first["notification_token"] = "secret-one"
    _write_descriptor(descriptor_path, first)
    publisher = attachment.LifecycleHintPublisher(
        descriptor_path,
        expected_runs_root=tmp_path,
        post=post,
    )

    assert publisher.publish(_attempt_hint()) is True
    assert delivered.wait(1)
    delivered.clear()
    second = _valid_payload(tmp_path)
    second["origin"] = "http://127.0.0.1:8002"
    second["notification_token"] = "secret-two"
    _write_descriptor(descriptor_path, second)
    assert publisher.publish(_attempt_hint()) is True
    assert delivered.wait(1)

    assert [(url, kwargs["headers"]) for url, kwargs in calls] == [
        (
            "http://127.0.0.1:8001/api/internal/lifecycle",
            {"X-RPent-Notification-Token": "secret-one"},
        ),
        (
            "http://127.0.0.1:8002/api/internal/lifecycle",
            {"X-RPent-Notification-Token": "secret-two"},
        ),
    ]
    assert [kwargs["json"] for _, kwargs in calls] == [
        _attempt_hint().to_payload(),
        _attempt_hint().to_payload(),
    ]
    publisher.close()


def test_lifecycle_publisher_reports_network_failure_without_raising(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(descriptor_path, _valid_payload(tmp_path))
    failed = threading.Event()

    async def post(*_args, **_kwargs):
        raise httpx.ConnectError("injected connection failure")

    publisher = attachment.LifecycleHintPublisher(
        descriptor_path,
        expected_runs_root=tmp_path,
        post=post,
        on_failure=failed.set,
    )

    assert publisher.publish(_attempt_hint()) is True
    assert failed.wait(1)
    publisher.close()
    assert not publisher._worker.is_alive()


def test_lifecycle_publisher_close_is_idempotent_and_terminates_worker(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(descriptor_path, _valid_payload(tmp_path))
    delivered = threading.Event()

    async def post(url, **_kwargs):
        delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    publisher = attachment.LifecycleHintPublisher(
        descriptor_path,
        expected_runs_root=tmp_path,
        post=post,
    )

    assert publisher.publish(_attempt_hint()) is True
    assert delivered.wait(1)
    publisher.close()
    publisher.close()

    assert not publisher._worker.is_alive()
    assert publisher.publish(_attempt_hint()) is False


def test_lifecycle_publisher_close_is_bounded_during_network_delivery(
    tmp_path: Path,
) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(descriptor_path, _valid_payload(tmp_path))
    entered = threading.Event()

    async def post(*_args, **_kwargs):
        entered.set()
        await asyncio.sleep(10)
        return httpx.Response(204)

    publisher = attachment.LifecycleHintPublisher(
        descriptor_path,
        expected_runs_root=tmp_path,
        post=post,
    )
    assert publisher.publish(_attempt_hint()) is True
    assert entered.wait(1)

    enqueue_started = time.monotonic()
    queued = [publisher.publish(_attempt_hint()) for _index in range(17)]
    assert time.monotonic() - enqueue_started < 0.05
    assert queued.count(True) == 16
    assert queued.count(False) == 1

    started = time.monotonic()
    publisher.close()

    assert time.monotonic() - started < 1.0
    assert not publisher._worker.is_alive()


def test_dashboard_lease_is_singleton_and_publishes_private_descriptor(
    tmp_path: Path,
) -> None:
    first = DashboardLease.acquire(tmp_path)
    try:
        with pytest.raises(DashboardAlreadyRunning, match="already running"):
            DashboardLease.acquire(tmp_path)
        descriptor = first.publish("http://127.0.0.1:43123", "worker-secret")
        assert descriptor.runs_root == tmp_path.resolve()
        assert descriptor.pid == os.getpid()
        assert descriptor.origin == "http://127.0.0.1:43123"
        assert stat.S_IMODE(descriptor.path.stat().st_mode) == 0o600
        assert DashboardDescriptor.load(
            descriptor.path, expected_runs_root=tmp_path
        ) == descriptor
    finally:
        first.close()

    assert not descriptor.path.exists()

    replacement = DashboardLease.acquire(tmp_path)
    replacement.close()


def test_dashboard_lease_can_store_runtime_metadata_outside_runs_root(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    lease = DashboardLease.acquire(runs_root, runtime_dir=runtime_root)
    try:
        descriptor = lease.publish(
            "http://127.0.0.1:43123",
            "worker-secret",
        )

        assert descriptor.path == (
            runtime_root / ".rpent-dashboard.json"
        ).resolve()
        assert descriptor.runs_root == runs_root.resolve()
        assert stat.S_IMODE(descriptor.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(
            (runtime_root / ".rpent-dashboard.lock").stat().st_mode
        ) == 0o600
    finally:
        lease.close()

    assert not descriptor.path.exists()


@pytest.mark.parametrize("failure_stage", ("fchmod", "flock"))
def test_dashboard_lease_acquire_closes_stream_on_setup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    opened = []
    original_open = Path.open

    def tracked_open(path: Path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracked_open)

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_stage} failure")

    if failure_stage == "fchmod":
        monkeypatch.setattr(attachment.os, "fchmod", fail)
    else:
        monkeypatch.setattr(attachment.fcntl, "flock", fail)

    with pytest.raises(OSError, match=f"injected {failure_stage} failure") as error:
        DashboardLease.acquire(tmp_path)

    assert type(error.value) is OSError
    assert len(opened) == 1
    assert opened[0].closed is True


def test_dashboard_lease_rejects_a_competing_process(tmp_path: Path) -> None:
    first = DashboardLease.acquire(tmp_path)
    child = """
import sys
from rpent.dashboard.attachment import DashboardAlreadyRunning, DashboardLease

try:
    lease = DashboardLease.acquire(sys.argv[1])
except DashboardAlreadyRunning:
    raise SystemExit(23)
else:
    lease.close()
    raise SystemExit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", child, str(tmp_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        first.close()

    assert result.returncode == 23, result.stderr


def test_dashboard_lease_close_is_idempotent_and_preserves_replacement_descriptor(
    tmp_path: Path,
) -> None:
    lease = DashboardLease.acquire(tmp_path)
    descriptor = lease.publish("http://127.0.0.1:43123", "worker-secret")
    replacement = _valid_payload(tmp_path)
    _write_descriptor(descriptor.path, replacement)

    lease.close()
    lease.close()

    assert json.loads(descriptor.path.read_bytes()) == replacement


def test_closed_dashboard_lease_cannot_overwrite_replacement_descriptor(
    tmp_path: Path,
) -> None:
    stale = DashboardLease.acquire(tmp_path)
    stale.close()
    replacement = DashboardLease.acquire(tmp_path)
    try:
        descriptor = replacement.publish(
            "http://127.0.0.1:43123", "replacement-secret"
        )

        with pytest.raises(RuntimeError, match="closed"):
            stale.publish("http://127.0.0.1:43124", "stale-secret")

        assert DashboardDescriptor.load(
            descriptor.path, expected_runs_root=tmp_path
        ) == descriptor
    finally:
        replacement.close()


def test_dashboard_descriptor_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / ".rpent-dashboard.json"
    path.write_text("{", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema"),
        ("instance_id", "not-a-uuid", "instance"),
        ("pid", True, "pid"),
        ("pid", 0, "pid"),
        ("started_at", "2026-08-14T03:04:05", "UTC"),
        ("started_at", "2026-08-14T03:04:05+10:00", "UTC"),
        ("started_at", "2026-08-14T03:04:05-00:00", "UTC"),
        ("origin", "http://user:secret@127.0.0.1:43123", "origin"),
        ("notification_token", "worker\nsecret", "header"),
    ],
)
def test_dashboard_descriptor_rejects_invalid_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _valid_payload(tmp_path)
    payload[field] = value
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    with pytest.raises(ValueError, match=message):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


@pytest.mark.parametrize(
    "origin",
    (
        "https://127.0.0.1:43123",
        "http://0.0.0.0:43123",
        "http://[::]:43123",
        "http://127.0.0.2:43123",
        "http://192.0.2.1:43123",
        "http://example.com:43123",
        "http://localhost.example:43123",
        "http://127.0.0.1:43123/path",
        "http://127.0.0.1:43123?query=1",
        "http://127.0.0.1:43123#fragment",
    ),
)
def test_dashboard_descriptor_rejects_non_loopback_plain_http_origin(
    tmp_path: Path,
    origin: str,
) -> None:
    payload = _valid_payload(tmp_path)
    payload["origin"] = origin
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    with pytest.raises(ValueError, match="loopback HTTP origin"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


@pytest.mark.parametrize(
    "origin",
    (
        "http://127.0.0.1:43123",
        "http://[::1]:43123",
    ),
)
def test_dashboard_descriptor_accepts_canonical_loopback_origins(
    tmp_path: Path,
    origin: str,
) -> None:
    payload = _valid_payload(tmp_path)
    payload["origin"] = origin
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    assert DashboardDescriptor.load(path, expected_runs_root=tmp_path).origin == origin


def test_dashboard_descriptor_rejects_localhost_hostname(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["origin"] = "http://localhost:43123"
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    with pytest.raises(ValueError, match="loopback HTTP origin"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_dashboard_descriptor_requires_exact_fields(
    tmp_path: Path, change: str
) -> None:
    payload = _valid_payload(tmp_path)
    if change == "missing":
        del payload["notification_token"]
    else:
        payload["unexpected"] = "value"
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    with pytest.raises(ValueError, match="fields"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


def test_dashboard_descriptor_rejects_wrong_runs_root(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected"
    expected_root.mkdir()
    payload = _valid_payload(tmp_path / "other")
    path = expected_root / ".rpent-dashboard.json"
    _write_descriptor(path, payload)

    with pytest.raises(ValueError, match="workspace mismatch"):
        DashboardDescriptor.load(path, expected_runs_root=expected_root)


def test_dashboard_descriptor_rejects_path_outside_workspace(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected"
    expected_root.mkdir()
    path = tmp_path / "outside" / ".rpent-dashboard.json"
    _write_descriptor(path, _valid_payload(expected_root))

    with pytest.raises(ValueError, match="workspace mismatch"):
        DashboardDescriptor.load(path, expected_runs_root=expected_root)


@pytest.mark.parametrize(
    "relative_path",
    ("other.json", "nested/.rpent-dashboard.json"),
)
def test_dashboard_descriptor_requires_canonical_workspace_path(
    tmp_path: Path,
    relative_path: str,
) -> None:
    path = tmp_path / relative_path
    _write_descriptor(path, _valid_payload(tmp_path))

    with pytest.raises(ValueError, match="workspace mismatch"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


def test_dashboard_descriptor_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_descriptor(target, _valid_payload(tmp_path))
    selected = tmp_path / ".rpent-dashboard.json"
    selected.symlink_to(target)

    with pytest.raises(ValueError, match="private regular file"):
        DashboardDescriptor.load(selected, expected_runs_root=tmp_path)


def test_dashboard_descriptor_rejects_over_permissive_mode(tmp_path: Path) -> None:
    path = tmp_path / ".rpent-dashboard.json"
    _write_descriptor(path, _valid_payload(tmp_path), mode=0o644)

    with pytest.raises(ValueError, match="private regular file"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)


def test_dashboard_descriptor_rejects_files_larger_than_64_kib(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".rpent-dashboard.json"
    path.write_bytes(b" " * (64 * 1024 + 1))
    path.chmod(0o600)

    with pytest.raises(ValueError, match="64 KiB"):
        DashboardDescriptor.load(path, expected_runs_root=tmp_path)
