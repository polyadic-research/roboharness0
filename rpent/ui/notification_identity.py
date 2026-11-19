"""Bounded authoritative identity checks for worker notifications."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from rpent.recording.contracts import EventEnvelope
from rpent.ui.io_gate import ReadGateTimeout
from rpent.ui.repository import CurrentAttemptIdentity

_IDENTITY_CAPACITY = 4_096
_IDENTITY_TIMEOUT_SECONDS = 0.5
_CaseKey = tuple[str, str]


class NotificationAttemptConflict(RuntimeError):
    """The notification does not belong to the repository's current attempt."""


@dataclass(slots=True)
class _IdentityEntry:
    identity: CurrentAttemptIdentity | None = None
    epoch: int = 0
    dirty: bool = False
    transition_attempt_id: str | None = None
    waiters: int = 0


class NotificationIdentityIndex:
    """Cache repository-confirmed current attempts by Evaluation and Case."""

    def __init__(
        self,
        repository: Any,
        read_gate: Any,
        *,
        capacity: int = _IDENTITY_CAPACITY,
    ) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 1 <= capacity <= _IDENTITY_CAPACITY
        ):
            raise ValueError("capacity must be an integer between 1 and 4096")
        self._repository = repository
        self._read_gate = read_gate
        self._capacity = capacity
        self._lock = threading.Lock()
        self._closed = False
        self._entries: OrderedDict[_CaseKey, _IdentityEntry] = OrderedDict()
        self._dirty_evaluations: OrderedDict[str, None] = OrderedDict()
        self._evaluation_epochs: OrderedDict[str, int] = OrderedDict()
        self._dirty_all = False

    def prime(self, identity: CurrentAttemptIdentity) -> None:
        """Cache an identity only after its registration locator is committed."""
        key = (identity.evaluation_id, identity.case_id)
        confirmed = _confirmed_identity(identity, key)
        with self._lock:
            if self._closed:
                raise RuntimeError("notification identity index is closed")
            if self._dirty_all:
                self._dirty_all = False
            self._dirty_evaluations.pop(identity.evaluation_id, None)
            entry = self._entries.get(key)
            if entry is None:
                entry = self._store_entry(key)
            if entry is None:
                return
            entry.epoch += 1
            entry.identity = confirmed
            entry.dirty = False
            entry.transition_attempt_id = None
            self._entries.move_to_end(key)

    async def accept(self, event: EventEnvelope) -> None:
        """Confirm one event while preserving the legacy no-value contract."""
        await self.accept_identity(event)

    async def accept_identity(
        self, event: EventEnvelope
    ) -> CurrentAttemptIdentity:
        """Confirm that one validated event belongs to the current attempt."""
        key = (event.evaluation_id, event.case_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _IDENTITY_TIMEOUT_SECONDS
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("notification identity index is closed")
                force_verify = (
                    self._dirty_all or event.evaluation_id in self._dirty_evaluations
                )
                entry = None if force_verify else self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                elif not force_verify and self._state_size() < self._capacity:
                    entry = _IdentityEntry()
                    self._entries[key] = entry
                cached = None if entry is None else entry.identity
                if (
                    entry is not None
                    and not entry.dirty
                    and cached is not None
                    and cached.attempt_id == event.attempt_id
                ):
                    return cached
                epoch = None if entry is None else entry.epoch
                if entry is not None:
                    entry.waiters += 1
                authority_token = self._authority_token(event.evaluation_id)
            timeout_s = max(0.0, deadline - loop.time())
            if timeout_s <= 0:
                with self._lock:
                    self._leave_entry(
                        key,
                        entry,
                        epoch,
                        discard_blank=True,
                    )
                raise ReadGateTimeout(("notification-identity", *key))
            try:
                selected = await self._read_gate.run(
                    ("notification-identity", *key),
                    partial(self._repository.current_attempt_identity, *key),
                    timeout_s=timeout_s,
                )
                confirmed = _confirmed_identity(selected, key)
            except BaseException:
                with self._lock:
                    self._leave_entry(
                        key,
                        entry,
                        epoch,
                        discard_blank=True,
                    )
                raise
            with self._lock:
                self._leave_entry(key, entry, epoch, discard_blank=False)
                if self._closed:
                    continue
                if self._dirty_all:
                    entry = None
                elif entry is not None:
                    if self._entries.get(key) is not entry or entry.epoch != epoch:
                        continue
                elif authority_token != self._authority_token(event.evaluation_id):
                    continue
                elif self._entries.get(key) is not None:
                    continue
                elif not (
                    self._dirty_all or event.evaluation_id in self._dirty_evaluations
                ):
                    entry = self._store_entry(key)
                if entry is not None:
                    if (
                        entry.dirty
                        and confirmed.attempt_id != entry.transition_attempt_id
                    ):
                        entry.dirty = False
                        entry.transition_attempt_id = None
                    entry.identity = confirmed
                    self._entries.move_to_end(key)
            if confirmed.attempt_id != event.attempt_id:
                raise NotificationAttemptConflict("attempt is not current")
            return confirmed

    def begin_transition(
        self,
        evaluation_id: str,
        attempts: Mapping[str, str | None] | None = None,
    ) -> None:
        """Verify affected Cases until the repository observes successors."""
        if attempts is None:
            try:
                metadata = self._repository.evaluation_metadata(evaluation_id)
            except (AttributeError, KeyError):
                self.invalidate(evaluation_id)
                return
            attempts = dict.fromkeys(metadata.case_ids)
        selected = _transition_attempts(attempts)
        if not selected:
            return
        selected_keys = {(evaluation_id, case_id) for case_id in selected}
        with self._lock:
            if self._closed:
                raise RuntimeError("notification identity index is closed")
            self._advance_authority(evaluation_id)
            if self._dirty_all:
                return
            if evaluation_id in self._dirty_evaluations:
                self._dirty_evaluations.move_to_end(evaluation_id)
                return
            protected = sum(
                entry.dirty or entry.waiters > 0 for entry in self._entries.values()
            )
            added = sum(
                self._entries.get(key) is None or not self._entries[key].dirty
                for key in selected_keys
            )
            if protected + added + len(self._dirty_evaluations) > self._capacity:
                self._set_evaluation_dirty(evaluation_id)
                return
            missing = sum(key not in self._entries for key in selected_keys)
            evictions = max(0, self._state_size() + missing - self._capacity)
            removed = self._evict_normals(evictions, excluding=selected_keys)
            if removed != evictions:
                self._set_evaluation_dirty(evaluation_id)
                return
            for case_id, attempt_id in selected.items():
                key = (evaluation_id, case_id)
                entry = self._entries.get(key)
                if entry is None:
                    entry = _IdentityEntry()
                    self._entries[key] = entry
                entry.epoch += 1
                entry.identity = None
                entry.dirty = True
                entry.transition_attempt_id = attempt_id
                self._entries.move_to_end(key)

    def invalidate(self, evaluation_id: str, case_id: str | None = None) -> None:
        """Forget one Case or every cached Case for an Evaluation."""
        with self._lock:
            self._advance_authority(evaluation_id)
            if case_id is not None:
                key = (evaluation_id, case_id)
                entry = self._entries.get(key)
                if entry is not None and entry.dirty:
                    entry.epoch += 1
                    entry.identity = None
                else:
                    self._entries.pop(key, None)
                return
            for key in tuple(self._entries):
                if key[0] == evaluation_id:
                    entry = self._entries[key]
                    if entry.dirty:
                        entry.epoch += 1
                        entry.identity = None
                    else:
                        self._entries.pop(key)

    def close(self) -> None:
        """Reject future checks and release cached identities."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._entries.clear()
            self._dirty_evaluations.clear()
            self._evaluation_epochs.clear()

    def _state_size(self) -> int:
        return len(self._entries) + len(self._dirty_evaluations)

    def _leave_entry(
        self,
        key: _CaseKey,
        entry: _IdentityEntry | None,
        epoch: int | None,
        *,
        discard_blank: bool,
    ) -> None:
        if entry is None:
            return
        entry.waiters -= 1
        if (
            discard_blank
            and self._entries.get(key) is entry
            and entry.epoch == epoch
            and entry.identity is None
            and not entry.dirty
            and entry.waiters == 0
        ):
            self._entries.pop(key)

    def _authority_token(self, evaluation_id: str) -> tuple[bool, int]:
        if self._dirty_all:
            return True, 0
        return False, self._evaluation_epochs.get(evaluation_id, 0)

    def _advance_authority(self, evaluation_id: str) -> None:
        if self._dirty_all:
            return
        epoch = self._evaluation_epochs.get(evaluation_id)
        if epoch is not None:
            self._evaluation_epochs[evaluation_id] = epoch + 1
            self._evaluation_epochs.move_to_end(evaluation_id)
            return
        if len(self._evaluation_epochs) >= _IDENTITY_CAPACITY:
            self._activate_dirty_all()
            return
        self._evaluation_epochs[evaluation_id] = 1

    def _store_entry(self, key: _CaseKey) -> _IdentityEntry | None:
        while self._state_size() >= self._capacity:
            if not self._evict_normal():
                return None
        entry = _IdentityEntry()
        self._entries[key] = entry
        return entry

    def _evict_normal(self, *, excluding: set[_CaseKey] | None = None) -> bool:
        return self._evict_normals(1, excluding=excluding) == 1

    def _evict_normals(
        self,
        count: int,
        *,
        excluding: set[_CaseKey] | None = None,
    ) -> int:
        if count <= 0:
            return 0
        selected_exclusions = set() if excluding is None else excluding
        removed = 0
        for key in tuple(self._entries):
            entry = self._entries[key]
            if (
                not entry.dirty
                and entry.waiters == 0
                and key not in selected_exclusions
            ):
                self._entries.pop(key)
                removed += 1
                if removed == count:
                    break
        return removed

    def _set_evaluation_dirty(self, evaluation_id: str) -> None:
        for key in tuple(self._entries):
            if key[0] == evaluation_id:
                self._entries.pop(key)
        while self._state_size() >= self._capacity:
            if not self._evict_normal():
                self._activate_dirty_all()
                return
        self._dirty_evaluations[evaluation_id] = None

    def _activate_dirty_all(self) -> None:
        self._dirty_all = True
        self._entries.clear()
        self._dirty_evaluations.clear()
        self._evaluation_epochs.clear()


def _confirmed_identity(selected: Any, key: _CaseKey) -> CurrentAttemptIdentity:
    evaluation_id, case_id = key
    if (
        getattr(selected, "evaluation_id", None) != evaluation_id
        or getattr(selected, "case_id", None) != case_id
    ):
        raise ValueError("notification identity scope mismatch")
    manifest_id = getattr(selected, "manifest_id", None)
    attempt_id = getattr(selected, "attempt_id", None)
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("notification manifest identity is invalid")
    if attempt_id is not None and (not isinstance(attempt_id, str) or not attempt_id):
        raise ValueError("notification attempt identity is invalid")
    return CurrentAttemptIdentity(
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
        case_id=case_id,
        attempt_id=attempt_id,
    )


def _transition_attempts(
    attempts: Mapping[str, str | None],
) -> dict[str, str | None]:
    selected: dict[str, str | None] = {}
    for case_id, attempt_id in attempts.items():
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("transition Case identity is invalid")
        if attempt_id is not None and (
            not isinstance(attempt_id, str) or not attempt_id
        ):
            raise ValueError("transition attempt identity is invalid")
        selected[case_id] = attempt_id
    return selected


__all__ = ["NotificationAttemptConflict", "NotificationIdentityIndex"]
