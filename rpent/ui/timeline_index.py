"""Bounded sparse indexes for authoritative retained Case timelines."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from rpent.recording.contracts import EventEnvelope
from rpent.ui.json_safety import require_bounded_json_nesting
from rpent.ui.schemas import (
    SequenceWindowView,
    StepGapView,
    TimelineWindow,
    TransitionAnchorView,
)

_WINDOW_RADIUS = 128


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    record_count: int
    offset: int
    sequence: int
    latest: TransitionAnchorView | None


@dataclass(slots=True)
class _JournalIndex:
    device: int
    inode: int
    observed_size: int = 0
    complete_offset: int = 0
    modified_ns: int = 0
    changed_ns: int = 0
    last_record_offset: int | None = None
    last_record_digest: bytes | None = None
    record_count: int = 0
    last_sequence: int = 0
    incomplete_tail: bool = False
    first: TransitionAnchorView | None = None
    latest: TransitionAnchorView | None = None
    transition_count: int = 0
    stride: int = 1
    checkpoints: list[_Checkpoint] = field(
        default_factory=lambda: [_Checkpoint(0, 0, 0, None)]
    )


class RetainedTimelineIndex:
    """Incrementally locate transition steps without retaining event payloads."""

    def __init__(
        self,
        *,
        max_records: int,
        max_bytes: int,
        max_record_bytes: int,
        index_capacity: int,
        cache_capacity: int,
        max_seek_records: int,
        max_seek_bytes: int,
    ) -> None:
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.max_record_bytes = max_record_bytes
        self.index_capacity = index_capacity
        self.cache_capacity = cache_capacity
        self.max_seek_records = max_seek_records
        self.max_seek_bytes = max_seek_bytes
        self._cache: OrderedDict[tuple[str, str, str, Path], _JournalIndex] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self.last_update_records = 0
        self.last_seek_records = 0
        self.last_seek_bytes = 0

    @property
    def checkpoint_count(self) -> int:
        return max((len(item.checkpoints) for item in self._cache.values()), default=0)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def locate(
        self,
        stream: BinaryIO,
        *,
        path: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        requested_step: int | None,
    ) -> TimelineWindow:
        key = (evaluation_id, case_id, attempt_id, path)
        with self._lock:
            metadata = os.fstat(stream.fileno())
            if metadata.st_size > self.max_bytes:
                raise ValueError(f"timeline journal byte limit exceeded in {path}")
            index = self._cache.pop(key, None)
            rebuild = (
                index is None
                or index.device != metadata.st_dev
                or index.inode != metadata.st_ino
                or metadata.st_size < index.observed_size
            )
            if rebuild:
                index = _JournalIndex(metadata.st_dev, metadata.st_ino)
                self._scan_update(
                    stream,
                    index,
                    path=path,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    attempt_id=attempt_id,
                    start=0,
                )
            elif metadata.st_size > index.observed_size:
                self._verify_append_seam(
                    stream,
                    index,
                    path=path,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    attempt_id=attempt_id,
                )
                self._scan_update(
                    stream,
                    index,
                    path=path,
                    evaluation_id=evaluation_id,
                    case_id=case_id,
                    attempt_id=attempt_id,
                    start=index.complete_offset,
                )
            elif metadata.st_size == index.observed_size and (
                metadata.st_mtime_ns != index.modified_ns
                or metadata.st_ctime_ns != index.changed_ns
            ):
                raise ValueError("timeline journal changed in place")
            else:
                self.last_update_records = 0
            self._cache[key] = index
            while len(self._cache) > self.cache_capacity:
                self._cache.popitem(last=False)
            return self._window(
                stream,
                index,
                path=path,
                evaluation_id=evaluation_id,
                case_id=case_id,
                attempt_id=attempt_id,
                requested_step=requested_step,
            )

    def _scan_update(
        self,
        stream: BinaryIO,
        index: _JournalIndex,
        *,
        path: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        start: int,
    ) -> None:
        stream.seek(start)
        update_records = 0
        while True:
            record_start = stream.tell()
            line = stream.readline(self.max_record_bytes + 2)
            if not line:
                index.incomplete_tail = False
                index.complete_offset = record_start
                break
            if len(line) > self.max_record_bytes:
                raise ValueError(f"event record byte limit exceeded in {path}")
            if not line.endswith(b"\n"):
                index.incomplete_tail = True
                index.complete_offset = record_start
                break
            if not line.strip():
                index.complete_offset = stream.tell()
                continue
            event = _event(line, path, evaluation_id, case_id, attempt_id)
            if event.sequence != index.last_sequence + 1:
                raise ValueError("event journal sequence mismatch")
            index.record_count += 1
            update_records += 1
            if index.record_count > self.max_records:
                raise ValueError(f"timeline journal record limit exceeded in {path}")
            index.last_sequence = event.sequence
            index.last_record_offset = record_start
            index.last_record_digest = hashlib.sha256(line).digest()
            if event.event_type == "transition_recorded":
                if event.step_id is None:
                    raise ValueError("transition event requires step_id")
                if index.latest is not None and event.step_id <= index.latest.step_id:
                    raise ValueError("transition steps must be strictly increasing")
                anchor = TransitionAnchorView(
                    step_id=event.step_id, sequence=event.sequence
                )
                index.first = index.first or anchor
                index.latest = anchor
                index.transition_count += 1
            index.complete_offset = stream.tell()
            if index.record_count % index.stride == 0:
                index.checkpoints.append(
                    _Checkpoint(
                        index.record_count,
                        index.complete_offset,
                        index.last_sequence,
                        index.latest,
                    )
                )
                self._bound_checkpoints(index)
        metadata = os.fstat(stream.fileno())
        index.observed_size = metadata.st_size
        index.modified_ns = metadata.st_mtime_ns
        index.changed_ns = metadata.st_ctime_ns
        self.last_update_records = update_records

    def _verify_append_seam(
        self,
        stream: BinaryIO,
        index: _JournalIndex,
        *,
        path: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
    ) -> None:
        if index.last_record_offset is None or index.last_record_digest is None:
            return
        stream.seek(index.last_record_offset)
        line = stream.readline(self.max_record_bytes + 2)
        if (
            not line.endswith(b"\n")
            or hashlib.sha256(line).digest() != index.last_record_digest
        ):
            raise ValueError("timeline journal append seam changed")
        event = _event(line, path, evaluation_id, case_id, attempt_id)
        if event.sequence != index.last_sequence:
            raise ValueError("timeline journal append seam changed")

    def _bound_checkpoints(self, index: _JournalIndex) -> None:
        while len(index.checkpoints) > self.index_capacity:
            index.stride *= 2
            if index.stride > self.max_seek_records:
                raise ValueError("timeline index cannot satisfy seek record budget")
            index.checkpoints = [
                item
                for item in index.checkpoints
                if item.record_count == 0 or item.record_count % index.stride == 0
            ]

    def _window(
        self,
        stream: BinaryIO,
        index: _JournalIndex,
        *,
        path: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        requested_step: int | None,
    ) -> TimelineWindow:
        if index.latest is None:
            self.last_seek_records = 0
            self.last_seek_bytes = 0
            return TimelineWindow(
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
                incomplete_tail=index.incomplete_tail,
            )
        if requested_step is None:
            selected = index.latest
            predecessor = index.latest
            successor = None
            self.last_seek_records = 0
            self.last_seek_bytes = 0
        else:
            selected, predecessor, successor = self._seek(
                stream,
                index,
                requested_step,
                path=path,
                evaluation_id=evaluation_id,
                case_id=case_id,
                attempt_id=attempt_id,
            )
        pivot = (
            selected.sequence
            if selected is not None
            else successor.sequence
            if successor is not None
            else index.last_sequence
        )
        event_window = SequenceWindowView(
            from_sequence=max(1, pivot - _WINDOW_RADIUS),
            to_sequence=min(index.last_sequence, pivot + _WINDOW_RADIUS),
        )
        gap = None
        if requested_step is not None and selected is None:
            gap = StepGapView(
                from_step=(predecessor.step_id + 1 if predecessor else requested_step),
                to_step=(successor.step_id - 1 if successor else requested_step),
            )
        return TimelineWindow(
            evaluation_id=evaluation_id,
            case_id=case_id,
            attempt_id=attempt_id,
            requested_step=requested_step,
            selected_transition=selected,
            first_transition=index.first,
            latest_transition=index.latest,
            transition_count=index.transition_count,
            event_window=event_window,
            gap=gap,
            incomplete_tail=index.incomplete_tail,
        )

    def _seek(
        self,
        stream: BinaryIO,
        index: _JournalIndex,
        requested_step: int,
        *,
        path: Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
    ) -> tuple[
        TransitionAnchorView | None,
        TransitionAnchorView | None,
        TransitionAnchorView | None,
    ]:
        checkpoint = index.checkpoints[0]
        for candidate in index.checkpoints:
            if (
                candidate.latest is not None
                and candidate.latest.step_id >= requested_step
            ):
                break
            checkpoint = candidate
        stream.seek(checkpoint.offset)
        previous_sequence = checkpoint.sequence
        predecessor = checkpoint.latest
        scanned_records = 0
        scanned_bytes = 0
        while stream.tell() < index.complete_offset:
            line = stream.readline(self.max_record_bytes + 2)
            scanned_records += 1
            scanned_bytes += len(line)
            if (
                scanned_records > self.max_seek_records
                or scanned_bytes > self.max_seek_bytes
            ):
                raise ValueError("timeline seek scan budget exceeded")
            event = _event(line, path, evaluation_id, case_id, attempt_id)
            if event.sequence != previous_sequence + 1:
                raise ValueError("event journal sequence mismatch")
            previous_sequence = event.sequence
            if event.event_type != "transition_recorded":
                continue
            assert event.step_id is not None
            anchor = TransitionAnchorView(
                step_id=event.step_id, sequence=event.sequence
            )
            if event.step_id == requested_step:
                self.last_seek_records = scanned_records
                self.last_seek_bytes = scanned_bytes
                return anchor, anchor, None
            if event.step_id > requested_step:
                self.last_seek_records = scanned_records
                self.last_seek_bytes = scanned_bytes
                return None, predecessor, anchor
            predecessor = anchor
        self.last_seek_records = scanned_records
        self.last_seek_bytes = scanned_bytes
        return None, predecessor, None


def _event(
    line: bytes,
    path: Path,
    evaluation_id: str,
    case_id: str,
    attempt_id: str,
) -> EventEnvelope:
    try:
        require_bounded_json_nesting(line, source=path)
        value = json.loads(line)
        event = EventEnvelope.from_payload(value)
    except RecursionError as exc:
        raise ValueError(f"JSON nesting limit exceeded in {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid event JSON in {path}") from exc
    if (
        event.evaluation_id != evaluation_id
        or event.case_id != case_id
        or event.attempt_id != attempt_id
    ):
        raise ValueError("event journal identity mismatch")
    return event


__all__ = ["RetainedTimelineIndex"]
