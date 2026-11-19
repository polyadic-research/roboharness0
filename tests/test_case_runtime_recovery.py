from __future__ import annotations

import asyncio
import json
import threading

import pytest

from rpent.evaluation.attempt_registration import register_attempt
from rpent.ui.case_runtime import (
    CaseRuntimeStore,
    JournalRead,
    RuntimeUpdate,
)
from rpent.ui.repository import CurrentAttemptIdentity
from tests.test_case_runtime import _event, _recording, _registration
from tests.test_episode_coordinator import _manifest, _result
from tests.test_ui_repository import _write_evaluation


def _cache_recording(store, manifest, case, attempt_id="attempt-1") -> None:
    async def load():
        return _recording()

    asyncio.run(
        store.recording_declaration(
            (manifest.evaluation_id, case.case_id, attempt_id), load
        )
    )


def test_one_repair_flight_freezes_normal_reduction_and_wakes_on_completion(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    updates: list[RuntimeUpdate] = []

    def read(_identity, *, through_sequence=None):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return JournalRead(
            (_event(manifest, case, 1), _event(manifest, case, 2)), 2
        )

    store = CaseRuntimeStore(
        journal_reader=read,
        high_water_reader=lambda _identity: 2,
        update_publisher=updates.append,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)
    assert store.apply_event(_event(manifest, case, 1), _recording()).disposition == "applied"

    assert store.apply_event(_event(manifest, case, 3), _recording()).disposition == "repairing"
    assert entered.wait(timeout=1)
    assert store.apply_event(_event(manifest, case, 2), _recording()).disposition == "repairing"
    assert store.repair_flight_count == 1
    release.set()
    assert store.wait_for_repairs(timeout=2)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert calls == 1
    assert entry is not None and entry.phase == "fresh"
    assert entry.last_sequence == 2
    assert len(updates) == 1 and updates[0].event is None


def test_repair_revalidates_attempt_epoch_before_candidate_publication(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    entered = threading.Event()
    release = threading.Event()
    updates: list[RuntimeUpdate] = []

    def read(_identity, *, through_sequence=None):
        entered.set()
        assert release.wait(timeout=2)
        return JournalRead((_event(manifest, case, 1),), 1)

    store = CaseRuntimeStore(
        journal_reader=read,
        high_water_reader=lambda _identity: 1,
        update_publisher=updates.append,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)
    store.apply_event(_event(manifest, case, 2), _recording())
    assert entered.wait(timeout=1)

    store.start_attempt(
        _registration(manifest, case, "attempt-2"), manifest, case
    )
    release.set()
    assert store.wait_for_repairs(timeout=2)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.value.attempt_id == "attempt-2"
    assert entry.last_sequence == 0
    assert updates == []


def test_terminal_pending_discards_obsolete_replay_candidate(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    entered = threading.Event()
    release = threading.Event()
    updates: list[RuntimeUpdate] = []

    def read(_identity, *, through_sequence=None):
        entered.set()
        assert release.wait(timeout=2)
        return JournalRead((_event(manifest, case, 1),), 1)

    store = CaseRuntimeStore(
        journal_reader=read,
        high_water_reader=lambda _identity: 1,
        update_publisher=updates.append,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)
    store.apply_event(_event(manifest, case, 2), _recording())
    assert entered.wait(timeout=1)

    pending = store.mark_terminal_pending(
        _result(manifest, case, attempt_id="attempt-1")
    )
    release.set()
    assert store.wait_for_repairs(timeout=2)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.phase == "terminal_pending"
    assert entry.runtime_revision == pending.runtime_revision
    assert updates == []


def test_journal_growth_requires_another_bounded_pass_then_converges(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    passes = 0

    def read(_identity, *, through_sequence=None):
        nonlocal passes
        passes += 1
        high = 2 if passes == 1 else 3
        return JournalRead(
            tuple(_event(manifest, case, sequence) for sequence in range(1, high + 1)),
            high,
        )

    high_water_calls = 0

    def high_water(_identity):
        nonlocal high_water_calls
        high_water_calls += 1
        return 3

    store = CaseRuntimeStore(
        journal_reader=read,
        high_water_reader=high_water,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)
    store.apply_event(_event(manifest, case, 2), _recording())
    assert store.wait_for_repairs(timeout=2)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert passes == 2
    assert high_water_calls >= 2
    assert entry is not None and entry.phase == "fresh"
    assert entry.last_sequence == 3


def test_repair_read_failure_marks_only_matching_attempt_unavailable(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=2)
    case, other = manifest.cases
    updates: list[RuntimeUpdate] = []

    def fail(_identity, *, through_sequence=None):
        raise OSError("journal unreadable")

    store = CaseRuntimeStore(
        journal_reader=fail,
        high_water_reader=lambda _identity: 2,
        update_publisher=updates.append,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    store.start_attempt(
        _registration(manifest, other, "attempt-other"), manifest, other
    )
    _cache_recording(store, manifest, case)
    store.apply_event(_event(manifest, case, 2), _recording())
    assert store.wait_for_repairs(timeout=2)

    failed = store.get_entry(manifest.evaluation_id, case.case_id)
    healthy = store.get_entry(manifest.evaluation_id, other.case_id)
    assert failed is not None and failed.phase == "unavailable"
    assert failed.unavailable_reason == "journal_read_failed"
    assert healthy is not None and healthy.phase == "fresh"
    assert len(updates) == 1


def test_unavailable_repair_restarts_only_after_authoritative_advance(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    identity = CurrentAttemptIdentity(
        manifest.evaluation_id, manifest.manifest_id, case.case_id, "attempt-1"
    )
    failed = True

    def read(_identity, *, through_sequence=None):
        if failed:
            raise OSError("journal unavailable")
        return JournalRead(
            tuple(_event(manifest, case, sequence) for sequence in (1, 2, 3)),
            3,
        )

    store = CaseRuntimeStore(
        journal_reader=read,
        high_water_reader=lambda _identity: 3,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)
    store.apply_event(_event(manifest, case, 2), _recording())
    assert store.wait_for_repairs(timeout=2)

    assert store.observe_high_water(identity, 2).disposition == "rejected"
    assert store.repair_flight_count == 0
    failed = False
    assert store.observe_high_water(identity, 3).disposition == "repairing"
    assert store.wait_for_repairs(timeout=2)
    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.phase == "fresh"
    assert entry.last_sequence == 3


def test_forged_observed_high_water_cannot_hold_repair_forever(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    store = CaseRuntimeStore(
        journal_reader=lambda _identity, **_kwargs: JournalRead(
            (_event(manifest, case, 1),), 1
        ),
        high_water_reader=lambda _identity: 1,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)

    store.apply_event(_event(manifest, case, 59), _recording())
    assert store.wait_for_repairs(timeout=2)

    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.phase == "fresh"
    assert entry.last_sequence == 1
    assert entry.observed_high_water == 59


def test_observe_high_water_repairs_lost_notification_without_enumeration(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    identity = CurrentAttemptIdentity(
        manifest.evaluation_id, manifest.manifest_id, case.case_id, "attempt-1"
    )
    store = CaseRuntimeStore(
        journal_reader=lambda _identity, **_kwargs: JournalRead(
            (_event(manifest, case, 1),), 1
        ),
        high_water_reader=lambda _identity: 1,
        update_publisher=lambda _update: None,
        heartbeat_interval=None,
    )
    store.start_attempt(_registration(manifest, case), manifest, case)
    _cache_recording(store, manifest, case)

    reduction = store.observe_high_water(identity, 1)
    assert reduction.disposition == "repairing"
    assert store.wait_for_repairs(timeout=2)
    entry = store.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None and entry.last_sequence == 1
    assert entry.phase == "fresh"


def test_repository_journal_and_high_water_are_bounded_strict_and_incomplete_safe(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    register_attempt(
        manifest, case, "attempt-1", "2026-08-13T00:00:00+00:00"
    )
    identity = CurrentAttemptIdentity(
        manifest.evaluation_id, manifest.manifest_id, case.case_id, "attempt-1"
    )
    path = root / "cases" / case.case_id / "attempts" / "attempt-1" / "events.jsonl"
    path.write_text(
        "".join(
            json.dumps(_event(manifest, case, sequence).to_payload()) + "\n"
            for sequence in (1, 2)
        )
        + '{"incomplete":',
        encoding="utf-8",
    )
    from rpent.ui.repository import EvaluationRepository

    repository = EvaluationRepository((root.parent,), max_event_tail_bytes=4096)
    monkeypatch.setattr(
        "rpent.ui.repository._iter_directory_beneath",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("heartbeat must not enumerate Attempts")
        ),
    )

    assert repository.read_event_high_water(identity) == 2
    journal = repository.read_event_journal(identity, through_sequence=1)
    assert journal.high_water == 1
    assert [event.sequence for event in journal.events] == [1]

    path.write_text(
        json.dumps(_event(manifest, case, 2).to_payload()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        repository.read_event_journal(identity)
