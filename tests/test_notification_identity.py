from __future__ import annotations

import asyncio
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rpent.evaluation.attempt_registration import register_attempt
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.recording.contracts import EventEnvelope
from rpent.ui.app import create_evaluation_app
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout, RepositoryReadGate
from rpent.ui.repository import EvaluationRepository
from rpent.ui.routes.internal import EventRequest, publish_event
from rpent.ui.service import ConflictError


def _write_running_evaluation(root: Path) -> tuple[EvaluationManifest, EpisodeCase]:
    case = EpisodeCase(
        ordinal=0,
        environment="robocasa",
        preset_name="robocasa-smoke",
        task_name="pick mug",
        task_index=0,
        instruction="pick up the mug",
        seed=0,
        episode_index=0,
        horizon=100,
        checkpoint_repo_id="org/policy",
        checkpoint_revision="a" * 40,
        binding_name="robocasa-v1",
    )
    manifest = EvaluationManifest(
        evaluation_id="evaluation-1",
        rpent_commit="b" * 40,
        rpent_dirty=False,
        cli_arguments={"task": "0"},
        environment={"name": "robocasa", "preset_name": "robocasa-smoke"},
        simulator={"name": "robocasa"},
        policy={"name": "smolvla"},
        task_selector="0",
        cases=(case,),
        schedule={"episodes_per_task": 1},
        contracts={},
        runtime={"evaluation_root": str(root)},
        recording={"schema_version": 1, "mode": "all", "camera_ids": []},
        qualification={"require_complete_artifacts": False},
        created_at="2026-08-11T00:00:00+00:00",
    )
    manifest.write_new(root / "evaluation_manifest.json")
    register_attempt(
        manifest,
        case,
        "attempt-current",
        "2026-08-11T00:00:01+00:00",
    )
    return manifest, case


def test_current_attempt_identity_avoids_full_case_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_running_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    repository.register_evaluation(root, manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("full Case projection must not run")

    for name in (
        "_case_summary_with_sequence",
        "_event_progress",
        "_artifact_projection",
        "_runtime_event_metrics",
        "_transition_frame_count",
        "_recording_projection",
    ):
        monkeypatch.setattr(repository, name, forbidden)

    identity = repository.current_attempt_identity(manifest.evaluation_id, case.case_id)

    assert identity.evaluation_id == manifest.evaluation_id
    assert identity.manifest_id == manifest.manifest_id
    assert identity.case_id == case.case_id
    assert identity.attempt_id == "attempt-current"


def _event(
    evaluation_id: str = "evaluation-1",
    case_id: str = "case-1",
    attempt_id: str = "attempt-1",
) -> EventEnvelope:
    return EventEnvelope(
        schema_version=1,
        evaluation_id=evaluation_id,
        case_id=case_id,
        attempt_id=attempt_id,
        sequence=1,
        recorded_at="2026-08-11T00:00:01+00:00",
        event_type="transition_recorded",
        step_id=0,
        operation_id=None,
        execution_id=None,
        payload={},
    )


class _ImmediateGate:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], float | None]] = []

    async def run(self, key, function, *, timeout_s=None):
        self.calls.append((key, timeout_s))
        return function()


class _DelayedGate:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls: list[tuple[tuple[str, ...], float | None]] = []
        self.started = asyncio.Event()

    async def run(self, key, function, *, timeout_s=None):
        self.calls.append((key, timeout_s))
        self.started.set()
        try:
            async with asyncio.timeout(timeout_s):
                await asyncio.sleep(self.delay)
        except TimeoutError as exc:
            raise ReadGateTimeout(key) from exc
        return function()


class _SelectiveFailureGate(_ImmediateGate):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def run(self, key, function, *, timeout_s=None):
        self.calls.append((key, timeout_s))
        if key[-1] == "case-foreign":
            raise self.error
        return function()


class _IdentityRepository:
    def __init__(self, attempt_id: str = "attempt-1") -> None:
        self.attempt_id = attempt_id
        self.identity_calls = 0
        self.get_case_calls = 0

    def current_attempt_identity(self, evaluation_id: str, case_id: str):
        self.identity_calls += 1
        return SimpleNamespace(
            evaluation_id=evaluation_id,
            manifest_id=f"manifest-{evaluation_id}",
            case_id=case_id,
            attempt_id=self.attempt_id,
        )

    def get_case(self, evaluation_id: str, case_id: str):
        self.get_case_calls += 1
        raise AssertionError("notification ingress must not project CaseDetail")


class _MappingIdentityRepository:
    def __init__(self) -> None:
        self.identities: dict[tuple[str, str], str | None] = {}
        self.identity_calls: dict[tuple[str, str], int] = {}

    def current_attempt_identity(self, evaluation_id: str, case_id: str):
        key = (evaluation_id, case_id)
        self.identity_calls[key] = self.identity_calls.get(key, 0) + 1
        try:
            attempt_id = self.identities[key]
        except KeyError:
            raise KeyError(key) from None
        return SimpleNamespace(
            evaluation_id=evaluation_id,
            manifest_id=f"manifest-{evaluation_id}",
            case_id=case_id,
            attempt_id=attempt_id,
        )


class _SlowIdentityRepository(_IdentityRepository):
    def __init__(
        self, *, delay: float = 0.3, error: BaseException | None = None
    ) -> None:
        super().__init__()
        self.delay = delay
        self.error = error
        self.started = threading.Event()

    def current_attempt_identity(self, evaluation_id: str, case_id: str):
        self.identity_calls += 1
        self.started.set()
        time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            evaluation_id=evaluation_id,
            manifest_id=f"manifest-{evaluation_id}",
            case_id=case_id,
            attempt_id=self.attempt_id,
        )


async def _wait_for_shared_caller(gate: RepositoryReadGate) -> None:
    async with asyncio.timeout(1):
        while gate.snapshot().shared_callers == 0:
            await asyncio.sleep(0.005)


def test_one_thousand_matching_notifications_use_one_minimal_projection() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _ImmediateGate()
    index = NotificationIdentityIndex(repository, gate)
    event = _event()

    async def run() -> None:
        for _ in range(1_000):
            await index.accept(event)

    asyncio.run(run())

    assert repository.identity_calls == 1
    assert repository.get_case_calls == 0
    assert len(gate.calls) == 1
    assert gate.calls[0][0] == ("notification-identity", "evaluation-1", "case-1")
    assert 0 < gate.calls[0][1] <= 0.5


def test_prime_committed_identity_avoids_repository_read() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex
    from rpent.ui.repository import CurrentAttemptIdentity

    repository = _IdentityRepository()
    gate = _ImmediateGate()
    index = NotificationIdentityIndex(repository, gate)
    index.prime(
        CurrentAttemptIdentity(
            evaluation_id="evaluation-1",
            manifest_id="manifest-evaluation-1",
            case_id="case-1",
            attempt_id="attempt-1",
        )
    )

    asyncio.run(index.accept(_event()))

    assert repository.identity_calls == 0
    assert gate.calls == []


@pytest.mark.parametrize("primed_attempt", [None, "attempt-old"])
def test_concurrent_cold_and_mismatched_notifications_coalesce_refresh(
    primed_attempt: str | None,
) -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    class BlockingRepository(_IdentityRepository):
        def __init__(self) -> None:
            super().__init__(primed_attempt or "attempt-new")
            self.block = False
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            identity = super().current_attempt_identity(evaluation_id, case_id)
            if self.block:
                self.started.set()
                assert self.release.wait(timeout=2)
            return identity

    repository = BlockingRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        if primed_attempt is not None:
            await index.accept(_event(attempt_id=primed_attempt))
            repository.attempt_id = "attempt-new"
        repository.block = True
        tasks = [
            asyncio.create_task(index.accept(_event(attempt_id="attempt-new")))
            for _ in range(32)
        ]
        try:
            async with asyncio.timeout(1):
                while not repository.started.is_set():
                    await asyncio.sleep(0.005)
            repository.release.set()
            await asyncio.gather(*tasks)
        finally:
            repository.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        asyncio.run(run())
        assert repository.identity_calls == (1 if primed_attempt is None else 2)
        assert repository.get_case_calls == 0
    finally:
        repository.release.set()
        gate.close()


@pytest.mark.parametrize("cancelled_index", [0, 1])
def test_cancelled_co_waiter_does_not_restart_shared_cold_read(
    cancelled_index: int,
) -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _SlowIdentityRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)
    event = _event()

    async def run() -> None:
        tasks = [asyncio.create_task(index.accept(event))]
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        tasks.append(asyncio.create_task(index.accept(event)))
        await _wait_for_shared_caller(gate)
        tasks[cancelled_index].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[cancelled_index]
        await tasks[1 - cancelled_index]
        await index.accept(event)

    try:
        asyncio.run(run())
        assert repository.identity_calls == 1
        entry = index._entries[("evaluation-1", "case-1")]
        assert entry.waiters == 0
    finally:
        gate.close()


def test_both_cancelled_cold_waiters_remove_placeholder_after_last_waiter() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    class BlockingRepository(_IdentityRepository):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            self.identity_calls += 1
            self.started.set()
            assert self.release.wait(timeout=2)
            return SimpleNamespace(
                evaluation_id=evaluation_id,
                manifest_id=f"manifest-{evaluation_id}",
                case_id=case_id,
                attempt_id=self.attempt_id,
            )

    repository = BlockingRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        tasks = [asyncio.create_task(index.accept(_event()))]
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        tasks.append(asyncio.create_task(index.accept(_event())))
        await _wait_for_shared_caller(gate)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert ("evaluation-1", "case-1") not in index._entries
        repository.release.set()
        async with asyncio.timeout(1):
            while gate.snapshot().flights:
                await asyncio.sleep(0.005)
        await index.accept(_event())

    try:
        asyncio.run(run())
        assert repository.identity_calls == 2
        assert index._entries[("evaluation-1", "case-1")].waiters == 0
    finally:
        repository.release.set()
        gate.close()


def test_shared_cold_read_error_removes_placeholder_after_all_waiters() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _SlowIdentityRepository(delay=0.1, error=KeyError("missing"))
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        tasks = [asyncio.create_task(index.accept(_event()))]
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        tasks.append(asyncio.create_task(index.accept(_event())))
        await _wait_for_shared_caller(gate)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, KeyError) for result in results)

    try:
        asyncio.run(run())
        assert repository.identity_calls == 1
        assert ("evaluation-1", "case-1") not in index._entries
    finally:
        gate.close()


def test_cancelled_co_waiter_with_repository_error_leaves_no_placeholder() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _SlowIdentityRepository(delay=0.1, error=KeyError("missing"))
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        first = asyncio.create_task(index.accept(_event()))
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        second = asyncio.create_task(index.accept(_event()))
        await _wait_for_shared_caller(gate)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(KeyError):
            await second

    try:
        asyncio.run(run())
        assert repository.identity_calls == 1
        assert ("evaluation-1", "case-1") not in index._entries
    finally:
        gate.close()


def test_concurrent_baseline_notifications_remain_dirty_until_successor() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    class BlockingRepository(_IdentityRepository):
        def __init__(self) -> None:
            super().__init__("attempt-old")
            self.block = False
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            identity = super().current_attempt_identity(evaluation_id, case_id)
            if self.block:
                self.started.set()
                assert self.release.wait(timeout=2)
            return identity

    repository = BlockingRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        await index.accept(_event(attempt_id="attempt-old"))
        index.begin_transition("evaluation-1", {"case-1": "attempt-old"})
        repository.block = True
        tasks = [
            asyncio.create_task(index.accept(_event(attempt_id="attempt-old")))
            for _ in range(32)
        ]
        try:
            async with asyncio.timeout(1):
                while not repository.started.is_set():
                    await asyncio.sleep(0.005)
            repository.release.set()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert results == [None] * 32
            repository.block = False
            repository.attempt_id = "attempt-new"
            with pytest.raises(NotificationAttemptConflict):
                await index.accept(_event(attempt_id="attempt-old"))
            await index.accept(_event(attempt_id="attempt-new"))
        finally:
            repository.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        asyncio.run(run())
        assert repository.identity_calls == 3
    finally:
        repository.release.set()
        gate.close()


def test_no_successor_keeps_baseline_authoritative_but_never_fast_cached() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    repository = _IdentityRepository("attempt-old")
    index = NotificationIdentityIndex(repository, _ImmediateGate())
    index.begin_transition("evaluation-1", {"case-1": "attempt-old"})

    async def run() -> None:
        for _ in range(3):
            await index.accept(_event(attempt_id="attempt-old"))
        repository.attempt_id = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(attempt_id="attempt-old"))
        await index.accept(_event(attempt_id="attempt-new"))

    asyncio.run(run())

    assert repository.identity_calls == 4


def test_mismatch_requires_repository_confirmation_and_invalidation_refreshes() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    repository = _IdentityRepository("attempt-old")
    gate = _ImmediateGate()
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        await index.accept(_event(attempt_id="attempt-old"))
        repository.attempt_id = "attempt-new"
        with pytest.raises(NotificationAttemptConflict, match="attempt is not current"):
            await index.accept(_event(attempt_id="attempt-unconfirmed"))
        index.invalidate("evaluation-1", "case-1")
        await index.accept(_event(attempt_id="attempt-new"))

    asyncio.run(run())

    assert repository.identity_calls == 3


def test_identity_cache_is_lru_bounded_to_4096_case_keys() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _ImmediateGate()
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        for value in range(4_096):
            await index.accept(_event(case_id=f"case-{value}"))
        await index.accept(_event(case_id="case-0"))
        await index.accept(_event(case_id="case-4096"))
        await index.accept(_event(case_id="case-1"))

    asyncio.run(run())

    assert repository.identity_calls == 4_098


def test_identity_cache_rejects_capacity_above_hard_bound() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    with pytest.raises(ValueError, match="between 1 and 4096"):
        NotificationIdentityIndex(
            _IdentityRepository(), _ImmediateGate(), capacity=4_097
        )


def test_large_transition_admission_evicts_normal_entries_in_one_pass() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    index = NotificationIdentityIndex(repository, _ImmediateGate())

    async def prime() -> None:
        for value in range(4_096):
            await index.accept(_event(case_id=f"case-{value}"))

    asyncio.run(prime())
    eviction_scans = 0
    original = index._evict_normal

    def counted_evict_normal(*, excluding=None):
        nonlocal eviction_scans
        eviction_scans += 1
        return original(excluding=excluding)

    index._evict_normal = counted_evict_normal
    attempts = {
        **{f"case-{value}": "attempt-1" for value in range(2_048, 4_096)},
        **{f"case-{value}": "attempt-1" for value in range(4_096, 6_144)},
    }

    index.begin_transition("evaluation-1", attempts)

    assert eviction_scans <= 1
    assert len(index._entries) == 4_096
    assert all(index._entries[("evaluation-1", case_id)].dirty for case_id in attempts)


@pytest.mark.parametrize(
    "error",
    [
        ReadGateBusy(("notification-identity", "foreign", "case-foreign")),
        ReadGateTimeout(("notification-identity", "foreign", "case-foreign")),
    ],
)
def test_failed_refresh_placeholder_does_not_displace_a_cached_identity(
    error: BaseException,
) -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _MappingIdentityRepository()
    stable_key = ("stable", "case-stable")
    repository.identities[stable_key] = "attempt-stable"
    index = NotificationIdentityIndex(
        repository, _SelectiveFailureGate(error), capacity=1
    )

    async def run() -> None:
        await index.accept(_event(*stable_key, attempt_id="attempt-stable"))
        with pytest.raises(type(error)):
            await index.accept(_event("foreign", "case-foreign"))
        await index.accept(_event(*stable_key, attempt_id="attempt-stable"))

    asyncio.run(run())

    assert repository.identity_calls == {stable_key: 1}
    assert tuple(index._entries) == (stable_key,)


def test_foreign_404_placeholder_does_not_displace_a_cached_identity() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _MappingIdentityRepository()
    stable_key = ("stable", "case-stable")
    repository.identities[stable_key] = "attempt-stable"
    index = NotificationIdentityIndex(repository, _ImmediateGate(), capacity=1)

    async def run() -> None:
        await index.accept(_event(*stable_key, attempt_id="attempt-stable"))
        with pytest.raises(KeyError):
            await index.accept(_event("foreign", "case-foreign"))
        await index.accept(_event(*stable_key, attempt_id="attempt-stable"))

    asyncio.run(run())

    assert repository.identity_calls[stable_key] == 1
    assert ("foreign", "case-foreign") not in index._entries


@pytest.mark.parametrize(
    "error",
    [
        ReadGateBusy(("notification-identity", "foreign", "case-foreign")),
        ReadGateTimeout(("notification-identity", "foreign", "case-foreign")),
    ],
)
def test_failed_unrelated_refresh_cannot_displace_dirty_transition(
    error: BaseException,
) -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    repository = _MappingIdentityRepository()
    target_key = ("target", "case-target")
    repository.identities[target_key] = "attempt-old"
    index = NotificationIdentityIndex(
        repository, _SelectiveFailureGate(error), capacity=1
    )
    index.begin_transition("target", {"case-target": "attempt-old"})

    async def run() -> None:
        with pytest.raises(type(error)):
            await index.accept(_event("foreign", "case-foreign"))
        await index.accept(_event(*target_key, attempt_id="attempt-old"))
        repository.identities[target_key] = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(*target_key, attempt_id="attempt-old"))
        await index.accept(_event(*target_key, attempt_id="attempt-new"))

    asyncio.run(run())

    assert repository.identity_calls[target_key] == 2


def test_transition_over_capacity_uses_bounded_evaluation_barrier() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    repository = _MappingIdentityRepository()
    attempts = {f"case-{value}": "attempt-old" for value in range(4_097)}
    repository.identities.update(
        (("evaluation-large", case_id), attempt_id)
        for case_id, attempt_id in attempts.items()
    )
    index = NotificationIdentityIndex(repository, _ImmediateGate())
    index.begin_transition("evaluation-large", attempts)
    target_key = ("evaluation-large", "case-0")

    async def run() -> None:
        await index.accept(_event(*target_key, attempt_id="attempt-old"))
        repository.identities[target_key] = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(*target_key, attempt_id="attempt-old"))
        await index.accept(_event(*target_key, attempt_id="attempt-new"))

    asyncio.run(run())

    assert repository.identity_calls[target_key] == 3
    assert len(index._entries) + len(index._dirty_evaluations) <= 4_096


def test_overflow_barrier_count_has_global_fail_closed_bound() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _MappingIdentityRepository()
    index = NotificationIdentityIndex(repository, _ImmediateGate(), capacity=2)

    for value in range(3):
        index.begin_transition(
            f"evaluation-{value}",
            {"case-1": "attempt-old", "case-2": "attempt-old", "case-3": None},
        )

    assert index._dirty_all is True
    assert len(index._entries) + len(index._dirty_evaluations) == 0


def test_global_verify_only_read_ignores_unrelated_invalidation() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _DelayedGate(0.3)
    index = NotificationIdentityIndex(repository, gate, capacity=1)
    index.begin_transition(
        "evaluation-overflow-1",
        {"case-1": "attempt-old", "case-2": "attempt-old"},
    )
    index.begin_transition(
        "evaluation-overflow-2",
        {"case-1": "attempt-old", "case-2": "attempt-old"},
    )
    assert index._dirty_all is True

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event()))
        await gate.started.wait()
        await asyncio.sleep(0.2)
        index.invalidate("evaluation-unrelated")
        await task

    asyncio.run(run())

    assert repository.identity_calls == 1
    assert len(gate.calls) == 1
    assert index._entries == {}


@pytest.mark.parametrize("activation", ["capacity", "authority-history"])
def test_dirty_all_activation_accepts_inflight_confirmation_uncached(
    activation: str,
) -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _SlowIdentityRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    capacity = 1 if activation == "capacity" else 4_096
    index = NotificationIdentityIndex(repository, gate, capacity=capacity)
    if activation == "authority-history":
        for value in range(4_096):
            index.invalidate(f"authority-{value}")
    hub = EvaluationEventHub(secret=b"dirty-all-test-secret")
    cursor = hub.cursor("evaluation-1")
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                notification_identities=index,
                evaluation_event_hub=hub,
            )
        )
    )

    async def notify(event: EventEnvelope):
        payload = EventRequest.model_validate(event.to_payload())
        return await publish_event(payload, request)

    async def run() -> None:
        callers = [asyncio.create_task(notify(_event()))]
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        callers.append(asyncio.create_task(notify(_event())))
        await _wait_for_shared_caller(gate)
        await asyncio.sleep(0.2)
        if activation == "capacity":
            index.begin_transition(
                "evaluation-overflow",
                {"case-1": "attempt-old", "case-2": "attempt-old"},
            )
        else:
            index.invalidate("authority-overflow")
        assert index._dirty_all is True

        responses = await asyncio.gather(*callers)
        assert [response.status_code for response in responses] == [204, 204]
        assert repository.identity_calls == 1
        assert index._entries == {}

        repository.attempt_id = "attempt-new"
        with pytest.raises(HTTPException) as stale:
            await notify(_event())
        assert stale.value.status_code == 409
        assert repository.identity_calls == 2
        assert index._entries == {}
        assert [
            event.attempt_id
            for event in hub.read(cursor, evaluation_id="evaluation-1").events
        ] == ["attempt-1"]

        current = await notify(_event(attempt_id="attempt-new"))
        assert current.status_code == 204
        assert repository.identity_calls == 3
        assert index._entries == {}

    try:
        asyncio.run(run())
    finally:
        gate.close()


@pytest.mark.parametrize(
    ("evaluation_id", "case_id"),
    [("evaluation-other", None), ("evaluation-1", "case-other")],
)
def test_unrelated_invalidation_does_not_restart_an_inflight_identity_read(
    evaluation_id: str, case_id: str | None
) -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _DelayedGate(0.3)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event()))
        await gate.started.wait()
        index.invalidate(evaluation_id, case_id)
        await task

    asyncio.run(run())

    assert repository.identity_calls == 1
    assert len(gate.calls) == 1


def test_unrelated_invalidation_does_not_restart_uncached_protected_read() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _DelayedGate(0.3)
    index = NotificationIdentityIndex(repository, gate, capacity=1)
    index.begin_transition("protected", {"case-protected": "attempt-old"})

    async def run() -> None:
        task = asyncio.create_task(
            index.accept(_event("uncached", "case-uncached", "attempt-1"))
        )
        await gate.started.wait()
        index.invalidate("unrelated")
        await task

    asyncio.run(run())

    assert repository.identity_calls == 1
    assert len(gate.calls) == 1


def test_same_key_invalidation_retries_within_one_outer_500ms_budget() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()
    gate = _DelayedGate(0.3)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event()))
        await gate.started.wait()
        index.invalidate("evaluation-1", "case-1")
        with pytest.raises(ReadGateTimeout):
            await task

    asyncio.run(run())

    assert repository.identity_calls == 1
    assert len(gate.calls) == 2
    assert 0 < gate.calls[0][1] <= 0.5
    assert 0 < gate.calls[1][1] < 0.25


def test_repeated_same_key_invalidation_cannot_extend_the_outer_deadline() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _IdentityRepository()

    class InvalidatingGate(_DelayedGate):
        def __init__(self) -> None:
            super().__init__(0.11)
            self.index: NotificationIdentityIndex | None = None

        async def run(self, key, function, *, timeout_s=None):
            result = await super().run(key, function, timeout_s=timeout_s)
            assert self.index is not None
            if len(self.calls) <= 6:
                self.index.invalidate("evaluation-1", "case-1")
            return result

    gate = InvalidatingGate()
    index = NotificationIdentityIndex(repository, gate)
    gate.index = index

    with pytest.raises(ReadGateTimeout):
        asyncio.run(index.accept(_event()))

    assert 4 <= len(gate.calls) <= 5
    assert gate.calls[-1][1] < 0.1


def test_caller_cancellation_does_not_seed_the_identity_cache() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    class BlockingRepository(_IdentityRepository):
        def __init__(self) -> None:
            super().__init__("attempt-old")
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            self.started.set()
            assert self.release.wait(timeout=2)
            return super().current_attempt_identity(evaluation_id, case_id)

    repository = BlockingRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event(attempt_id="attempt-old")))
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        repository.release.set()
        async with asyncio.timeout(1):
            while gate.snapshot().flights:
                await asyncio.sleep(0.005)
        repository.attempt_id = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(attempt_id="attempt-old"))

    try:
        asyncio.run(run())
        assert repository.identity_calls == 2
    finally:
        repository.release.set()
        gate.close()


def test_cancelled_unrelated_refresh_cannot_displace_dirty_transition() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    class BlockingRepository(_MappingIdentityRepository):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            if case_id == "case-foreign":
                self.started.set()
                assert self.release.wait(timeout=2)
            return super().current_attempt_identity(evaluation_id, case_id)

    repository = BlockingRepository()
    target_key = ("target", "case-target")
    repository.identities[target_key] = "attempt-old"
    repository.identities[("foreign", "case-foreign")] = "attempt-foreign"
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate, capacity=1)
    index.begin_transition("target", {"case-target": "attempt-old"})

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event("foreign", "case-foreign")))
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        repository.release.set()
        async with asyncio.timeout(1):
            while gate.snapshot().flights:
                await asyncio.sleep(0.005)
        await index.accept(_event(*target_key, attempt_id="attempt-old"))
        repository.identities[target_key] = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(*target_key, attempt_id="attempt-old"))

    try:
        asyncio.run(run())
        assert repository.identity_calls[target_key] == 2
    finally:
        repository.release.set()
        gate.close()


def test_transition_started_during_refresh_prevents_stale_inflight_cache() -> None:
    from rpent.ui.notification_identity import (
        NotificationAttemptConflict,
        NotificationIdentityIndex,
    )

    class BlockingRepository(_IdentityRepository):
        def __init__(self) -> None:
            super().__init__("attempt-old")
            self.block_once = True
            self.started = threading.Event()
            self.release = threading.Event()

        def current_attempt_identity(self, evaluation_id: str, case_id: str):
            identity = super().current_attempt_identity(evaluation_id, case_id)
            if self.block_once:
                self.block_once = False
                self.started.set()
                assert self.release.wait(timeout=2)
            return identity

    repository = BlockingRepository()
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    index = NotificationIdentityIndex(repository, gate)

    async def run() -> None:
        task = asyncio.create_task(index.accept(_event(attempt_id="attempt-old")))
        async with asyncio.timeout(1):
            while not repository.started.is_set():
                await asyncio.sleep(0.005)
        index.begin_transition("evaluation-1", {"case-1": "attempt-old"})
        repository.release.set()
        await task
        repository.attempt_id = "attempt-new"
        with pytest.raises(NotificationAttemptConflict):
            await index.accept(_event(attempt_id="attempt-old"))

    try:
        asyncio.run(run())
        assert repository.identity_calls == 3
    finally:
        repository.release.set()
        gate.close()


class _RouteIdentityRepository(_IdentityRepository):
    def __init__(self) -> None:
        super().__init__()
        self.identities = {("evaluation-1", "case-1"): "attempt-1"}
        self.case_states = {("evaluation-1", "case-1"): "cancelled"}
        self.error: BaseException | None = None

    def current_attempt_identity(self, evaluation_id: str, case_id: str):
        self.identity_calls += 1
        if self.error is not None:
            raise self.error
        try:
            attempt_id = self.identities[(evaluation_id, case_id)]
        except KeyError:
            raise KeyError((evaluation_id, case_id)) from None
        return SimpleNamespace(
            evaluation_id=evaluation_id,
            manifest_id=f"manifest-{evaluation_id}",
            case_id=case_id,
            attempt_id=attempt_id,
        )

    def get_evaluation(self, evaluation_id: str):
        cases = tuple(
            SimpleNamespace(
                case_id=case_id,
                attempt_id=attempt_id,
                execution_state=self.case_states[(selected_evaluation_id, case_id)],
            )
            for (selected_evaluation_id, case_id), attempt_id in self.identities.items()
            if selected_evaluation_id == evaluation_id
        )
        if not cases:
            raise KeyError(evaluation_id)
        return SimpleNamespace(
            cases=cases,
            qualification=SimpleNamespace(accepted=False),
            official_metrics=SimpleNamespace(is_complete=False),
        )

    def evaluation_metadata(self, evaluation_id: str):
        case_ids = frozenset(
            case_id
            for selected_evaluation_id, case_id in self.identities
            if selected_evaluation_id == evaluation_id
        )
        if not case_ids:
            raise KeyError(evaluation_id)
        return SimpleNamespace(evaluation_id=evaluation_id, case_ids=case_ids)


class _ErrorGate(_ImmediateGate):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def run(self, key, function, *, timeout_s=None):
        self.calls.append((key, timeout_s))
        raise self.error

    def snapshot(self):
        return {"saturated": 0}


def _notification_client(
    repository: _RouteIdentityRepository,
    *,
    gate=None,
    service=None,
    notification_identities=None,
) -> tuple[TestClient, EvaluationEventHub]:
    selected_gate = _ImmediateGate() if gate is None else gate
    if not hasattr(selected_gate, "snapshot"):
        selected_gate.snapshot = lambda: {"saturated": 0}
    hub = EvaluationEventHub(secret=b"notification-test-secret")
    app = create_evaluation_app(
        repository=repository,
        service=SimpleNamespace() if service is None else service,
        event_hub=hub,
        notification_token="notification-secret",
        repository_read_gate=selected_gate,
        notification_identities=notification_identities,
    )
    return (
        TestClient(
            app,
            headers={"X-RPent-Notification-Token": "notification-secret"},
            raise_server_exceptions=False,
        ),
        hub,
    )


def test_internal_notification_handler_is_async_and_never_projects_case() -> None:
    repository = _RouteIdentityRepository()
    client, _ = _notification_client(repository)

    response = client.post("/api/internal/events", json=_event().to_payload())

    assert inspect.iscoroutinefunction(publish_event)
    assert response.status_code == 204
    assert repository.identity_calls == 1
    assert repository.get_case_calls == 0


def test_internal_notifications_refresh_new_attempt_and_reject_stale_or_foreign() -> (
    None
):
    repository = _RouteIdentityRepository()
    client, hub = _notification_client(repository)
    cursor = hub.cursor("evaluation-1")

    first = client.post("/api/internal/events", json=_event().to_payload())
    repository.identities[("evaluation-1", "case-1")] = "attempt-2"
    current = _event(attempt_id="attempt-2")
    refreshed = client.post("/api/internal/events", json=current.to_payload())
    cached = client.post("/api/internal/events", json=current.to_payload())
    stale = client.post("/api/internal/events", json=_event().to_payload())
    foreign_evaluation = client.post(
        "/api/internal/events", json=_event(evaluation_id="foreign").to_payload()
    )
    foreign_case = client.post(
        "/api/internal/events", json=_event(case_id="foreign").to_payload()
    )

    assert [first.status_code, refreshed.status_code, cached.status_code] == [204] * 3
    assert stale.status_code == 409
    assert foreign_evaluation.status_code == 404
    assert foreign_case.status_code == 404
    assert repository.identity_calls == 5
    assert repository.get_case_calls == 0
    assert len(hub.read(cursor, evaluation_id="evaluation-1").events) == 2


@pytest.mark.parametrize(
    ("error", "status_code", "retry_after"),
    [
        (
            ReadGateBusy(("notification-identity", "evaluation-1", "case-1")),
            503,
            "1",
        ),
        (
            ReadGateTimeout(("notification-identity", "evaluation-1", "case-1")),
            504,
            None,
        ),
    ],
)
def test_identity_gate_failure_never_publishes(
    error: BaseException, status_code: int, retry_after: str | None
) -> None:
    repository = _RouteIdentityRepository()
    client, hub = _notification_client(repository, gate=_ErrorGate(error))
    cursor = hub.cursor("evaluation-1")

    response = client.post("/api/internal/events", json=_event().to_payload())

    assert response.status_code == status_code
    assert response.headers.get("retry-after") == retry_after
    assert repository.get_case_calls == 0
    assert hub.read(cursor, evaluation_id="evaluation-1").events == ()


def test_repository_identity_error_never_publishes() -> None:
    repository = _RouteIdentityRepository()
    repository.error = RuntimeError("identity read failed")
    client, hub = _notification_client(repository)
    cursor = hub.cursor("evaluation-1")

    response = client.post("/api/internal/events", json=_event().to_payload())

    assert response.status_code == 500
    assert repository.identity_calls == 1
    assert repository.get_case_calls == 0
    assert hub.read(cursor, evaluation_id="evaluation-1").events == ()


def test_repository_timeout_error_is_a_generic_500_and_never_publishes() -> None:
    repository = _RouteIdentityRepository()
    repository.error = TimeoutError("repository read failed")
    gate = RepositoryReadGate(workers=1, pending=1, default_timeout_s=1)
    client, hub = _notification_client(repository, gate=gate)
    cursor = hub.cursor("evaluation-1")

    try:
        response = client.post("/api/internal/events", json=_event().to_payload())

        assert response.status_code == 500
        assert repository.identity_calls == 1
        assert hub.read(cursor, evaluation_id="evaluation-1").events == ()

        repository.error = None
        recovered = client.post("/api/internal/events", json=_event().to_payload())
        assert recovered.status_code == 204
        assert repository.identity_calls == 2
    finally:
        gate.close()


def test_foreign_404_cannot_evict_transition_and_publish_later_stale_event() -> None:
    from rpent.ui.notification_identity import NotificationIdentityIndex

    repository = _RouteIdentityRepository()
    gate = _ImmediateGate()
    index = NotificationIdentityIndex(repository, gate, capacity=1)
    index.begin_transition("evaluation-1", {"case-1": "attempt-1"})
    client, hub = _notification_client(
        repository,
        gate=gate,
        notification_identities=index,
    )
    cursor = hub.cursor("evaluation-1")

    assert (
        client.post(
            "/api/internal/events", json=_event("foreign", "case-foreign").to_payload()
        ).status_code
        == 404
    )
    assert (
        client.post("/api/internal/events", json=_event().to_payload()).status_code
        == 204
    )
    repository.identities[("evaluation-1", "case-1")] = "attempt-2"
    assert (
        client.post("/api/internal/events", json=_event().to_payload()).status_code
        == 409
    )

    assert [
        event.attempt_id
        for event in hub.read(cursor, evaluation_id="evaluation-1").events
    ] == ["attempt-1"]


@pytest.mark.parametrize("prime_cache", [False, True])
def test_resume_transition_verifies_baseline_until_materialized_successor(
    prime_cache: bool,
) -> None:
    repository = _RouteIdentityRepository()
    service = _ControlService()
    client, hub = _notification_client(repository, service=service)

    if prime_cache:
        assert (
            client.post(
                "/api/internal/events", json=_event(attempt_id="attempt-1").to_payload()
            ).status_code
            == 204
        )
    cursor = hub.cursor("evaluation-1")

    resumed = client.post("/api/evaluations/evaluation-1/resume")
    old_while_pending = client.post(
        "/api/internal/events", json=_event(attempt_id="attempt-1").to_payload()
    )
    repository.identities[("evaluation-1", "case-1")] = "attempt-2"
    stale_after_transition = client.post(
        "/api/internal/events", json=_event(attempt_id="attempt-1").to_payload()
    )
    current = client.post(
        "/api/internal/events", json=_event(attempt_id="attempt-2").to_payload()
    )
    cached = client.post(
        "/api/internal/events", json=_event(attempt_id="attempt-2").to_payload()
    )

    assert resumed.status_code == 202
    assert [
        old_while_pending.status_code,
        stale_after_transition.status_code,
        current.status_code,
        cached.status_code,
    ] == [204, 409, 204, 204]
    assert [
        event.attempt_id
        for event in hub.read(cursor, evaluation_id="evaluation-1").events
    ] == (["attempt-2"] if prime_cache else ["attempt-1", "attempt-2"])


def test_resume_transition_does_not_invalidate_unaffected_case_identity() -> None:
    repository = _RouteIdentityRepository()
    repository.identities[("evaluation-1", "case-terminal")] = "attempt-terminal"
    repository.case_states[("evaluation-1", "case-terminal")] = "terminal"
    client, _ = _notification_client(repository, service=_ControlService())
    terminal = _event(case_id="case-terminal", attempt_id="attempt-terminal")

    assert (
        client.post("/api/internal/events", json=terminal.to_payload()).status_code
        == 204
    )
    calls_before = repository.identity_calls
    assert client.post("/api/evaluations/evaluation-1/resume").status_code == 202
    assert (
        client.post("/api/internal/events", json=terminal.to_payload()).status_code
        == 204
    )

    assert repository.identity_calls == calls_before


class _RecordingNotificationIdentities:
    def __init__(self) -> None:
        self.invalidations: list[tuple[str, str | None]] = []
        self.transitions: list[tuple[str, dict[str, str | None] | None]] = []
        self.closed = 0

    async def accept(self, event: EventEnvelope) -> None:
        return None

    def invalidate(self, evaluation_id: str, case_id: str | None = None) -> None:
        self.invalidations.append((evaluation_id, case_id))

    def begin_transition(
        self,
        evaluation_id: str,
        attempts: dict[str, str | None] | None = None,
    ) -> None:
        self.transitions.append((evaluation_id, attempts))

    def close(self) -> None:
        self.closed += 1


class _ControlService:
    def __init__(self) -> None:
        self.fail_cancel = False
        self.fail_resume = False

    def start(self, request):
        return SimpleNamespace(
            evaluation_id="evaluation-start", operation="start", state="queued"
        )

    def cancel(self, evaluation_id: str, reason: str):
        if self.fail_cancel:
            raise ConflictError("not active")
        return SimpleNamespace(
            evaluation_id=evaluation_id, operation="cancel", state="running"
        )

    def resume(self, evaluation_id: str):
        if self.fail_resume:
            raise ConflictError("not resumable")
        return SimpleNamespace(
            evaluation_id=evaluation_id, operation="resume", state="queued"
        )

    def retry(self, evaluation_id: str, request):
        return SimpleNamespace(
            evaluation_id="evaluation-retry", operation="retry", state="queued"
        )

    def rerun(self, evaluation_id: str, case_id: str):
        return SimpleNamespace(
            evaluation_id="evaluation-rerun", operation="rerun", state="queued"
        )


class _ControlRepository:
    def get_evaluation(self, evaluation_id: str):
        assert evaluation_id == "evaluation-1"
        return SimpleNamespace(
            cases=(
                SimpleNamespace(
                    case_id="case-cancelled",
                    attempt_id="attempt-old",
                    execution_state="cancelled",
                ),
                SimpleNamespace(
                    case_id="case-queued",
                    attempt_id=None,
                    execution_state="queued",
                ),
                SimpleNamespace(
                    case_id="case-terminal",
                    attempt_id="attempt-terminal",
                    execution_state="terminal",
                ),
            )
        )

    def evaluation_metadata(self, evaluation_id: str):
        case_ids = {
            "evaluation-start": frozenset({"case-start"}),
            "evaluation-retry": frozenset({"case-retry"}),
            "evaluation-rerun": frozenset({"case-rerun"}),
            "evaluation-replay": frozenset({"case-replay"}),
        }
        return SimpleNamespace(case_ids=case_ids[evaluation_id])


def test_successful_controls_invalidate_affected_keys_but_failures_do_not() -> None:
    identities = _RecordingNotificationIdentities()
    service = _ControlService()
    app = create_evaluation_app(
        repository=_ControlRepository(),
        service=service,
        event_hub=EvaluationEventHub(),
        notification_identities=identities,
    )

    app.state.evaluation_service.start(object())
    app.state.evaluation_service.cancel("evaluation-1", "user_requested")
    app.state.evaluation_service.resume("evaluation-1")
    app.state.evaluation_service.retry("evaluation-1", object())
    app.state.evaluation_service.rerun("evaluation-1", "case-1")
    service.fail_cancel = True
    with pytest.raises(ConflictError):
        app.state.evaluation_service.cancel("evaluation-failed", "user_requested")
    service.fail_resume = True
    with pytest.raises(ConflictError):
        app.state.evaluation_service.resume("evaluation-1")

    assert identities.invalidations == [("evaluation-1", None)]
    assert identities.transitions == [
        ("evaluation-start", None),
        (
            "evaluation-1",
            {"case-cancelled": "attempt-old", "case-queued": None},
        ),
        ("evaluation-retry", None),
        ("evaluation-rerun", None),
    ]


def test_successful_callback_controls_invalidate_only_after_return() -> None:
    identities = _RecordingNotificationIdentities()

    def force_stop(evaluation_id: str):
        if evaluation_id == "evaluation-failed":
            raise ConflictError("not active")
        return SimpleNamespace(evaluation_id=evaluation_id)

    def action_replay(
        evaluation_id: str, case_id: str, attempt_id: str, recipe_id: str
    ):
        return SimpleNamespace(evaluation_id="evaluation-replay")

    app = create_evaluation_app(
        repository=_ControlRepository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(),
        notification_identities=identities,
        force_stop_handler=force_stop,
        action_replay_handler=action_replay,
    )

    app.state.force_stop_handler("evaluation-1")
    app.state.action_replay_handler("evaluation-1", "case-1", "attempt-1", "recipe-1")
    with pytest.raises(ConflictError):
        app.state.force_stop_handler("evaluation-failed")

    assert identities.invalidations == [("evaluation-1", None)]
    assert identities.transitions == [("evaluation-replay", None)]


def test_same_evaluation_action_replay_transitions_the_source_attempt() -> None:
    identities = _RecordingNotificationIdentities()

    def action_replay(
        evaluation_id: str, case_id: str, attempt_id: str, recipe_id: str
    ):
        return SimpleNamespace(evaluation_id=evaluation_id)

    app = create_evaluation_app(
        repository=_ControlRepository(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(),
        notification_identities=identities,
        action_replay_handler=action_replay,
    )

    app.state.action_replay_handler("evaluation-1", "case-1", "attempt-1", "recipe-1")

    assert identities.transitions == [("evaluation-1", {"case-1": "attempt-1"})]


def test_app_does_not_close_injected_notification_identity_index() -> None:
    identities = _RecordingNotificationIdentities()
    app = create_evaluation_app(
        repository=SimpleNamespace(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(),
        notification_identities=identities,
    )

    app.state.close_owned_read_gates()

    assert app.state.owns_notification_identities is False
    assert identities.closed == 0


def test_app_closes_its_owned_notification_identity_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rpent.ui.app as app_module

    identities = _RecordingNotificationIdentities()
    monkeypatch.setattr(
        app_module,
        "NotificationIdentityIndex",
        lambda repository, read_gate: identities,
        raising=False,
    )
    app = create_evaluation_app(
        repository=SimpleNamespace(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(),
    )

    app.state.close_owned_read_gates()

    assert app.state.owns_notification_identities is True
    assert identities.closed == 1
