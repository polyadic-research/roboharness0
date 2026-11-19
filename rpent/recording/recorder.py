"""Recoverable single-writer recording for one evaluation Case attempt."""

from __future__ import annotations

import fcntl
import json
import math
import re
import stat
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rpent.episode.contracts import StepResult
from rpent.evaluation.results import EpisodeResult
from rpent.recording import media
from rpent.recording.contracts import (
    RECORDING_SCHEMA_VERSION,
    ArtifactHealth,
    EventEnvelope,
    FrameReference,
    RecordingProfile,
    TransitionRecord,
    sanitize_artifact_issues,
)
from rpent.recording.display import DisplayAdapter
from rpent.recording.json_safe import json_safe
from rpent.recording.live_events import record_episode_video_available

NotificationSink = Callable[[EventEnvelope], None]


@dataclass(frozen=True, slots=True)
class _OperationClipJob:
    operation_id: str
    step_ids: tuple[int, ...]


class CaseRecorder:
    """Own all journals and camera artifacts for one Case attempt."""

    def __init__(
        self,
        *,
        root: str | Path,
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        episode_id: str,
        display: DisplayAdapter,
        profile: RecordingProfile,
        notify: NotificationSink | None = None,
        llm_inputs_only: bool = False,
    ) -> None:
        for name, value in (
            ("evaluation_id", evaluation_id),
            ("case_id", case_id),
            ("attempt_id", attempt_id),
            ("episode_id", episode_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        self.root = Path(root)
        if not isinstance(llm_inputs_only, bool) or (llm_inputs_only and profile.mode != "none"):
            raise ValueError("LLM-only images require a boolean flag and recording mode none")
        self.evaluation_id = evaluation_id
        self.case_id = case_id
        self.attempt_id = attempt_id
        self.episode_id = episode_id
        self.display = display
        self.cameras = tuple(display.descriptors)
        self.profile = RecordingProfile.validate(
            profile.mode, profile.camera_ids, self.cameras
        )
        self.notify = notify
        self.llm_inputs_only = llm_inputs_only
        self._mutex = threading.RLock()
        self._issues: list[str] = []
        self._persisted_issues: set[str] = set()
        self._pending_issues: list[str] = []
        self._persisting_health = False
        self._health_poisoned = False
        self._poisoned: str | None = None
        self._closed = False
        self._final_manifest: dict[str, Any] | None = None
        self._final_result: EpisodeResult | None = None
        self._action_camera_artifacts: list[dict[str, str]] = []
        self._submitted_clip_operations: set[str] = set()
        self._clip_executor: ThreadPoolExecutor | None = None

        media.ensure_directory(self.root)
        self._finalized_on_open = (self.root / "artifacts.json").is_file()
        lock_mode = "rb" if self._finalized_on_open else "a+b"
        self._lock_stream = (self.root / ".recorder.lock").open(lock_mode)
        try:
            lock_type = fcntl.LOCK_SH if self._finalized_on_open else fcntl.LOCK_EX
            fcntl.flock(self._lock_stream.fileno(), lock_type | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise RuntimeError("Case attempt already has an active writer") from exc

        try:
            self._recover()
            self._load_final_manifest()
            if self._final_manifest is None:
                self._write_recording_discovery()
                self._clip_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="rpent-action-clips",
                )
        except BaseException:
            self.close()
            raise

    @property
    def _events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def _transitions_path(self) -> Path:
        return self.root / "trajectory/transitions.jsonl"

    @property
    def _health_path(self) -> Path:
        return self.root / "health.jsonl"

    def _recover(self) -> None:
        self._recover_health()
        events, event_tail = media.read_complete_jsonl(self._events_path)
        transitions, transition_tail = media.read_complete_jsonl(self._transitions_path)
        envelopes = [EventEnvelope.from_payload(row) for row in events]
        records = [TransitionRecord.from_payload(row) for row in transitions]
        self._validate_recovered(envelopes, records)
        self._sequence = envelopes[-1].sequence if envelopes else 0
        self._transition_count = len(records)
        self._last_step_id = records[-1].step_id if records else -1
        self._next_frame_index = (
            max((record.frame_index for record in records), default=-1) + 1
        )
        if event_tail and not self._finalized_on_open:
            media.replace_jsonl(self._events_path, events)
        if event_tail:
            self._degrade("ignored incomplete events.jsonl tail during recovery")
        if transition_tail and not self._finalized_on_open:
            media.replace_jsonl(self._transitions_path, transitions)
        if transition_tail:
            self._degrade("ignored incomplete transitions.jsonl tail during recovery")
        self._repair_transition_evidence(envelopes, records)

    def _recover_health(self) -> None:
        rows, incomplete = media.read_complete_jsonl(self._health_path)
        seen: set[str] = set()
        for row in rows:
            if (
                row.get("evaluation_id") != self.evaluation_id
                or row.get("case_id") != self.case_id
                or row.get("attempt_id") != self.attempt_id
            ):
                raise ValueError("recovered health identity does not match recorder")
            issue = row.get("issue")
            if not isinstance(issue, str) or not issue.strip():
                raise ValueError("health issue must be a non-empty string")
            public_issue = sanitize_artifact_issues((issue,))[0]
            if public_issue in seen:
                continue
            seen.add(public_issue)
            self._issues.append(public_issue)
            self._persisted_issues.add(public_issue)
        if incomplete and not self._finalized_on_open:
            media.replace_jsonl(self._health_path, rows)
        if incomplete:
            self._degrade("ignored incomplete health.jsonl tail during recovery")

    def _repair_transition_evidence(
        self,
        events: list[EventEnvelope],
        transitions: list[TransitionRecord],
    ) -> None:
        event_transitions = {
            json.dumps(event.payload.get("transition"), sort_keys=True)
            for event in events
            if event.event_type == "transition_recorded"
        }
        root = self.root.resolve()
        retained = set(self._retained_camera_ids())
        for transition in transitions:
            payload = transition.to_payload()
            present = {frame.camera_id for frame in transition.frames}
            for camera_id in sorted(retained.difference(present)):
                self._degrade("retained camera frame unavailable")
            for frame in transition.frames:
                path = (self.root / frame.relative_path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    self._degrade("retained camera frame unavailable")
            signature = json.dumps(payload, sort_keys=True)
            if signature in event_transitions:
                continue
            if self._finalized_on_open:
                self._degrade("transition event unavailable")
                continue
            self._degrade("transition event recovered")
            self._ensure_writable()
            event = self._make_event(
                "transition_recorded",
                {"transition": payload},
                step_id=transition.step_id,
                operation_id=transition.operation_id,
                execution_id=transition.execution_id,
            )
            try:
                self._commit_event(event, notify=False)
            except media.JsonlRollbackError as exc:
                self._poison("transition event repair", exc)
                raise
            except OSError:
                self._degrade("transition event repair failed")

    def _load_final_manifest(self) -> None:
        path = self.root / "artifacts.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid finalized artifacts manifest") from exc
        if not isinstance(payload, dict):
            raise ValueError("finalized artifacts manifest must be an object")
        identity = (
            payload.get("evaluation_id"),
            payload.get("case_id"),
            payload.get("attempt_id"),
            payload.get("episode_id"),
        )
        expected = (
            self.evaluation_id,
            self.case_id,
            self.attempt_id,
            self.episode_id,
        )
        if identity != expected:
            raise ValueError("finalized attempt identity does not match recorder")
        if payload.get("profile") != self.profile.to_payload() or payload.get(
            "cameras"
        ) != [camera.to_payload() for camera in self.cameras]:
            raise ValueError(
                "finalized attempt recording identity does not match recorder"
            )
        try:
            result = EpisodeResult.from_payload(payload["episode_result"])
        except KeyError as exc:
            raise ValueError("finalized artifacts manifest is missing result") from exc
        if (
            result.evaluation_id != self.evaluation_id
            or result.case_id != self.case_id
            or result.attempt_id != self.attempt_id
        ):
            raise ValueError("finalized result identity does not match recorder")
        stored_health = payload.get("health", {})
        projected_status = "complete"
        if isinstance(stored_health, dict):
            parsed_health = ArtifactHealth.from_payload(stored_health)
            projected_status = parsed_health.status
            for issue in parsed_health.issues:
                if issue not in self._issues:
                    self._issues.append(issue)
                self._persisted_issues.add(issue)
        if projected_status == "complete" and self._issues:
            projected_status = "degraded"
        declarations = payload.get("artifacts")
        if isinstance(declarations, list):
            self._action_camera_artifacts = [
                dict(item)
                for item in declarations
                if isinstance(item, dict)
                and item.get("kind") == "action_camera_clip"
                and all(
                    isinstance(item.get(name), str)
                    for name in ("artifact_id", "relative_path")
                )
            ]
        payload["health"] = ArtifactHealth(
            projected_status, tuple(self._issues)
        ).to_payload()
        payload["transition_count"] = self._transition_count
        payload["event_count"] = self._sequence
        payload["artifacts"] = self._declared_artifacts()
        self._final_manifest = payload
        self._final_result = result

    def _validate_recovered(
        self,
        events: list[EventEnvelope],
        transitions: list[TransitionRecord],
    ) -> None:
        previous_sequence = 0
        for event in events:
            self._validate_identity(event.case_id, event.attempt_id)
            if event.evaluation_id != self.evaluation_id:
                raise ValueError("recovered event identity does not match recorder")
            if event.sequence != previous_sequence + 1:
                raise ValueError("recovered event sequences must be contiguous")
            previous_sequence = event.sequence
        previous_step = -1
        for transition in transitions:
            self._validate_identity(transition.case_id, transition.attempt_id)
            if transition.episode_id != self.episode_id:
                raise ValueError(
                    "recovered transition episode_id does not match recorder"
                )
            if transition.step_id <= previous_step:
                raise ValueError("recovered step IDs must be strictly increasing")
            previous_step = transition.step_id

    def _validate_identity(self, case_id: str, attempt_id: str) -> None:
        if case_id != self.case_id or attempt_id != self.attempt_id:
            raise ValueError("recovered Case/attempt identity does not match recorder")

    def _degrade(self, issue: str) -> None:
        with self._mutex:
            issue = sanitize_artifact_issues((issue,))[0]
            if issue not in self._issues:
                self._issues.append(issue)
            if issue not in self._persisted_issues and issue not in self._pending_issues:
                self._pending_issues.append(issue)
            self._retry_health()

    def _set_poison(self, context: str, _error: BaseException) -> None:
        if self._poisoned is None:
            self._poisoned = f"recorder poisoned by {context}"
        issue = "recorder poisoned"
        if issue not in self._issues:
            self._issues.append(issue)
        if issue not in self._persisted_issues and issue not in self._pending_issues:
            self._pending_issues.append(issue)

    def _poison(self, context: str, error: BaseException) -> None:
        self._set_poison(context, error)
        self._retry_health()

    def _retry_health(self) -> None:
        if self._finalized_on_open or self._persisting_health or self._health_poisoned:
            return
        self._persisting_health = True
        try:
            for issue in tuple(self._pending_issues):
                try:
                    media.append_jsonl(
                        self._health_path,
                        {
                            "evaluation_id": self.evaluation_id,
                            "case_id": self.case_id,
                            "attempt_id": self.attempt_id,
                            "recorded_at": datetime.now(timezone.utc).isoformat(),
                            "issue": issue,
                        },
                    )
                except media.JsonlRollbackError as exc:
                    self._health_poisoned = True
                    self._set_poison("health journal rollback failure", exc)
                    break
                except Exception:
                    break
                self._pending_issues.remove(issue)
                self._persisted_issues.add(issue)
        finally:
            self._persisting_health = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CaseRecorder is closed")

    def _ensure_writable(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError(self._poisoned)
        if self._final_manifest is not None:
            raise RuntimeError("CaseRecorder attempt is finalized")
        self._ensure_open()

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        """Implement the planning EventRecorder protocol."""
        return self.record_event(event, **payload).to_payload()

    def record_event(self, event_type: str, **payload: Any) -> EventEnvelope:
        with self._mutex:
            self._ensure_writable()
            self._retry_health()
            self._ensure_writable()
            event = self._make_event(event_type, payload)
            try:
                self._commit_event(event)
            except media.JsonlRollbackError as exc:
                self._poison("event journal rollback failure", exc)
                raise
            except OSError:
                self._degrade("event journal write failed")
                raise
            return event

    def retain_observation_artifacts(
        self, result: Mapping[str, Any]
    ) -> tuple[Path, ...]:
        """Persist camera bytes returned by an environment observation tool."""
        if self.llm_inputs_only:
            return ()
        with self._mutex:
            self._ensure_writable()
            explicit = result.get("_image_camera_ids")
            # Derived images (zoom crops) are not camera frames.
            derived = result.get("_image_derivation")
            derived = set(derived) if isinstance(derived, Mapping) else set()
            if isinstance(explicit, Mapping):
                known = {camera.camera_id for camera in self.cameras}
                assigned = {
                    camera_id: field
                    for camera_id, field in explicit.items()
                    if camera_id in known
                    and isinstance(field, str)
                    and field.startswith("_image_")
                    and field.endswith("_bytes")
                    and field not in derived
                    and isinstance(result.get(field), bytes)
                    and result[field]
                }
            else:
                fields = {
                    "primary": "_image_bytes",
                    "wrist": "_image_wrist_bytes",
                }
                assigned = {
                    camera.camera_id: fields[camera.role]
                    for camera in self.cameras
                    if camera.role in fields
                }
                remaining_fields = [
                    key
                    for key, value in result.items()
                    if key.startswith("_image_")
                    and key.endswith("_bytes")
                    and key not in assigned.values()
                    and isinstance(value, bytes)
                    and value
                ]
                remaining_cameras = [
                    camera.camera_id
                    for camera in self.cameras
                    if camera.camera_id not in assigned
                ]
                assigned.update(
                    zip(remaining_cameras, remaining_fields, strict=False)
                )

            paths: list[Path] = []
            for camera_id in self._retained_camera_ids():
                field = assigned.get(camera_id)
                value = result.get(field) if field is not None else None
                if not isinstance(value, bytes) or not value:
                    continue
                try:
                    frame = self.display.frame(camera_id, {camera_id: value})
                except Exception:
                    self._degrade("camera extraction failed")
                    continue
                path = self.root / "live" / (
                    f"{media.camera_storage_component(camera_id)}.png"
                )
                try:
                    media.atomic_write_bytes(path, frame)
                except OSError:
                    self._degrade("live camera write failed")
                    continue
                paths.append(path.resolve())
            return tuple(paths)

    def _make_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        step_id: int | None = None,
        operation_id: str | None = None,
        execution_id: str | None = None,
    ) -> EventEnvelope:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty")
        safe = json_safe(dict(payload))
        return EventEnvelope(
            schema_version=RECORDING_SCHEMA_VERSION,
            evaluation_id=self.evaluation_id,
            case_id=self.case_id,
            attempt_id=self.attempt_id,
            sequence=self._sequence + 1,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            step_id=step_id if step_id is not None else safe.get("step_id"),
            operation_id=(
                operation_id if operation_id is not None else safe.get("operation_id")
            ),
            execution_id=(
                execution_id if execution_id is not None else safe.get("execution_id")
            ),
            payload=safe,
        )

    def _commit_event(self, event: EventEnvelope, *, notify: bool = True) -> None:
        with self._mutex:
            media.append_jsonl(self._events_path, event.to_payload())
            self._sequence = event.sequence
            if notify and self.notify is not None:
                try:
                    # False means a disposable live hint was dropped; the durable
                    # journal remains authoritative and artifact health is unchanged.
                    self.notify(event)
                except Exception:
                    self._degrade("notification delivery failed")

    def record_transition(
        self,
        *,
        action: np.ndarray,
        result: StepResult,
        timestamp_s: float,
        operation_id: str | None,
        execution_id: str | None,
    ) -> EventEnvelope:
        with self._mutex:
            self._ensure_writable()
            self._retry_health()
            self._ensure_writable()
            record = self._build_transition(
                action, result, timestamp_s, operation_id, execution_id
            )
            try:
                media.append_jsonl(self._transitions_path, record.to_payload())
            except media.JsonlRollbackError as exc:
                self._poison("transition journal rollback failure", exc)
                raise
            except OSError:
                self._degrade("transition journal write failed")
                raise
            self._transition_count += 1
            self._last_step_id = record.step_id
            self._next_frame_index += 1
            event = self._make_event(
                "transition_recorded",
                {"transition": record.to_payload()},
                step_id=record.step_id,
                operation_id=record.operation_id,
                execution_id=record.execution_id,
            )
            try:
                self._commit_event(event)
            except media.JsonlRollbackError as exc:
                self._poison("transition event journal rollback failure", exc)
                raise
            except OSError:
                self._degrade("transition event journal write failed")
            return event

    def _build_transition(
        self,
        action: np.ndarray,
        result: StepResult,
        timestamp_s: float,
        operation_id: str | None,
        execution_id: str | None,
    ) -> TransitionRecord:
        if not isinstance(result, StepResult):
            raise ValueError("result must be a StepResult")
        if result.episode_id != self.episode_id:
            raise ValueError("result episode_id does not match recorder")
        if result.step_id <= self._last_step_id:
            raise ValueError("step IDs must be strictly increasing")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)):
            raise ValueError("timestamp_s must be a non-negative finite number")
        timestamp_s = float(timestamp_s)
        if timestamp_s < 0 or not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be a non-negative finite number")
        action_array = np.asarray(action)
        if (
            action_array.ndim != 1
            or not np.issubdtype(action_array.dtype, np.number)
            or np.issubdtype(action_array.dtype, np.complexfloating)
            or not np.isfinite(action_array).all()
        ):
            raise ValueError("action must be a one-dimensional numeric array")
        action_values = json_safe(action_array)
        if not isinstance(action_values, list):
            raise ValueError("action must be a one-dimensional numeric array")

        camera_ids = {camera.camera_id for camera in self.cameras}
        state = json_safe(
            {
                key: value
                for key, value in result.observation.items()
                if key not in camera_ids
            }
        )
        record = TransitionRecord(
            case_id=self.case_id,
            attempt_id=self.attempt_id,
            episode_id=self.episode_id,
            step_id=result.step_id,
            frame_index=self._next_frame_index,
            timestamp_s=timestamp_s,
            state=state,
            action=action_values,
            reward=result.reward,
            success=result.success,
            terminated=result.terminated,
            truncated=result.truncated,
            operation_id=operation_id,
            execution_id=execution_id,
            frames=(),
        )
        frames = self._write_frames(result.observation, result.step_id, timestamp_s)
        return replace(record, frames=frames)

    def _write_frames(
        self, observation: Mapping[str, Any], step_id: int, timestamp_s: float
    ) -> tuple[FrameReference, ...]:
        if self.llm_inputs_only:
            return ()
        retained = set(self._retained_camera_ids())
        references: list[FrameReference] = []
        for camera in self.cameras:
            camera_id = camera.camera_id
            component = media.camera_storage_component(camera_id)
            try:
                value = self.display.frame(camera_id, observation)
            except Exception:
                self._degrade("camera extraction failed")
                continue
            try:
                media.atomic_write_bytes(self.root / "live" / f"{component}.png", value)
            except OSError:
                self._degrade("live camera write failed")
            if camera_id not in retained:
                continue
            relative = Path(
                "trajectory",
                "cameras",
                component,
                "segments",
                f"{self._next_frame_index:08d}.png",
            )
            try:
                media.atomic_write_bytes(self.root / relative, value)
            except OSError:
                self._degrade("camera segment write failed")
                continue
            references.append(
                FrameReference(
                    camera_id=camera_id,
                    frame_index=self._next_frame_index,
                    step_id=step_id,
                    timestamp_s=timestamp_s,
                    relative_path=relative.as_posix(),
                )
            )
        return tuple(references)

    def _retained_camera_ids(self) -> tuple[str, ...]:
        return self.profile.retained_camera_ids(self.cameras)

    def _write_recording_discovery(self) -> None:
        snapshot = self._artifact_snapshot()
        media.atomic_write_json(
            self.root / "recording.json",
            {
                name: snapshot[name]
                for name in (
                    "schema_version",
                    "evaluation_id",
                    "case_id",
                    "attempt_id",
                    "episode_id",
                    "profile",
                    "cameras",
                    "retained_camera_ids",
                )
            }
            | {"artifacts": list(self._action_camera_artifacts)},
        )

    def snapshot(self) -> dict[str, Any]:
        with self._mutex:
            if self._final_manifest is not None:
                return self._final_manifest
            return self._artifact_snapshot()

    def _artifact_snapshot(self) -> dict[str, Any]:
        health = ArtifactHealth(
            "degraded" if self._issues else "complete", tuple(self._issues)
        )
        return {
            "schema_version": RECORDING_SCHEMA_VERSION,
            "evaluation_id": self.evaluation_id,
            "case_id": self.case_id,
            "attempt_id": self.attempt_id,
            "episode_id": self.episode_id,
            "profile": self.profile.to_payload(),
            "cameras": [camera.to_payload() for camera in self.cameras],
            "retained_camera_ids": list(self._retained_camera_ids()),
            "transition_count": self._transition_count,
            "event_count": self._sequence,
            "health": health.to_payload(),
        }

    def submit_operation_clips(
        self, operation_id: str, step_ids: tuple[int, ...]
    ) -> None:
        """Queue one completed operation; never wait for video encoding."""
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be non-empty")
        if (
            not isinstance(step_ids, tuple)
            or not step_ids
            or any(
                isinstance(step_id, bool)
                or not isinstance(step_id, int)
                or step_id < 0
                for step_id in step_ids
            )
            or any(left >= right for left, right in zip(step_ids, step_ids[1:]))
        ):
            raise ValueError("step_ids must be strictly increasing non-negative integers")
        with self._mutex:
            self._ensure_writable()
            if operation_id in self._submitted_clip_operations:
                return
            executor = self._clip_executor
            if executor is None:
                raise RuntimeError("operation clip encoder is unavailable")
            self._submitted_clip_operations.add(operation_id)
            executor.submit(
                self._run_operation_clip_job,
                _OperationClipJob(operation_id, step_ids),
            )

    def _run_operation_clip_job(self, job: _OperationClipJob) -> None:
        try:
            with self._mutex:
                rows, incomplete = media.read_complete_jsonl(self._transitions_path)
                if incomplete:
                    self._degrade("action camera clip unavailable")
                selected = [
                    row
                    for row in rows
                    if row.get("operation_id") == job.operation_id
                ]
                if [row.get("step_id") for row in selected] != list(job.step_ids):
                    selected = []
                    self._degrade("action camera clip unavailable")
            artifacts = self._finalize_action_camera_clips(
                selected,
                publish_ready=False,
                notify=True,
            )
            with self._mutex:
                known = {
                    item["artifact_id"] for item in self._action_camera_artifacts
                }
                self._action_camera_artifacts.extend(
                    item for item in artifacts if item["artifact_id"] not in known
                )
                self._write_recording_discovery()
            self._publish_action_camera_ready(
                job.operation_id,
                tuple(artifacts),
                notify=True,
            )
        except Exception:
            self._degrade("action camera clip unavailable")
            try:
                self._publish_action_camera_ready(job.operation_id, (), notify=True)
            except Exception:
                self._degrade("action camera clip unavailable")

    def _drain_operation_clip_worker(self) -> None:
        with self._mutex:
            executor = self._clip_executor
            self._clip_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def finalize(self, result: EpisodeResult) -> dict[str, Any]:
        self._drain_operation_clip_worker()
        with self._mutex:
            if not isinstance(result, EpisodeResult):
                raise ValueError("result must be an EpisodeResult")
            if self._final_manifest is not None:
                if self._final_result is None or result != self._final_result:
                    raise ValueError("result does not match finalized attempt result")
                return self._final_manifest
            self._ensure_writable()
            self._retry_health()
            self._ensure_writable()
            if (
                result.evaluation_id != self.evaluation_id
                or result.case_id != self.case_id
                or result.attempt_id != self.attempt_id
            ):
                raise ValueError(
                    "result evaluation_id/case_id/attempt_id does not match recorder"
                )
            transitions, incomplete = media.read_complete_jsonl(self._transitions_path)
            if incomplete:
                self._degrade("ignored incomplete transitions.jsonl tail at finalize")
            try:
                media.write_parquet(
                    self.root / "trajectory/transitions.parquet", transitions
                )
            except Exception:
                self._degrade("Parquet finalization failed")

            episode_video_published = False
            for camera in self.cameras:
                if camera.camera_id not in self._retained_camera_ids():
                    continue
                references = [
                    frame
                    for row in transitions
                    for frame in row.get("frames", [])
                    if frame.get("camera_id") == camera.camera_id
                ]
                component = media.camera_storage_component(camera.camera_id)
                camera_root = self.root / "trajectory/cameras" / component
                try:
                    media.atomic_write_json(
                        camera_root / "index.json",
                        {"camera": camera.to_payload(), "frames": references},
                    )
                except Exception:
                    self._degrade("camera index finalization failed")
                try:
                    video_path = camera_root / "video.mp4"
                    media.finalize_video(
                        video_path,
                        (self.root / frame["relative_path"] for frame in references),
                        camera.fps,
                    )
                    if (
                        not episode_video_published
                        and video_path.is_file()
                        and video_path.stat().st_size > 0
                    ):
                        record_episode_video_available(self, f"{component}-video")
                        episode_video_published = True
                except Exception:
                    self._degrade("camera video finalization failed")

            finalized_clips = self._finalize_action_camera_clips(
                transitions, publish_ready=True
            )
            known_clips = {
                item["artifact_id"] for item in self._action_camera_artifacts
            }
            self._action_camera_artifacts.extend(
                item
                for item in finalized_clips
                if item["artifact_id"] not in known_clips
            )

            retained = set(self._retained_camera_ids())
            for camera in self.cameras:
                if camera.camera_id not in retained:
                    component = media.camera_storage_component(camera.camera_id)
                    try:
                        (self.root / "live" / f"{component}.png").unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        self._degrade("live camera cleanup failed")

            self._ensure_writable()
            manifest = {
                **self._artifact_snapshot(),
                "episode_result": result.to_payload(),
                "artifacts": self._declared_artifacts(),
            }
            media.atomic_write_json(self.root / "artifacts.json", manifest)
            self._final_manifest = manifest
            self._final_result = result
            self.close()
            return manifest

    def _declared_artifacts(self) -> list[dict[str, str]]:
        camera_artifact_ids = {
            f"camera_{camera_id}_{suffix}": (
                f"{media.camera_storage_component(camera_id)}-{suffix}"
            )
            for camera_id in self._retained_camera_ids()
            for suffix in ("index", "video")
        }
        candidates = (
            ("events", self._events_path),
            ("health", self._health_path),
            ("transitions_jsonl", self._transitions_path),
            ("transitions_parquet", self.root / "trajectory/transitions.parquet"),
            *(
                (
                    f"camera_{camera_id}_index",
                    self.root
                    / "trajectory/cameras"
                    / media.camera_storage_component(camera_id)
                    / "index.json",
                )
                for camera_id in self._retained_camera_ids()
            ),
            *(
                (
                    f"camera_{camera_id}_video",
                    self.root
                    / "trajectory/cameras"
                    / media.camera_storage_component(camera_id)
                    / "video.mp4",
                )
                for camera_id in self._retained_camera_ids()
            ),
        )
        declared = []
        for kind, path in candidates:
            if not path.exists():
                continue
            artifact = {
                "kind": kind,
                "relative_path": path.relative_to(self.root).as_posix(),
            }
            artifact_id = camera_artifact_ids.get(kind)
            if artifact_id is not None:
                artifact["artifact_id"] = artifact_id
            declared.append(artifact)
        declared.extend(self._action_camera_artifacts)
        action_root = self.root / "action_videos"
        if action_root.is_dir():
            for path in sorted(action_root.glob("step_*.mp4")):
                match = re.fullmatch(r"step_(\d+)_[-A-Za-z0-9_.]+\.mp4", path.name)
                if match is None or not path.is_file() or path.stat().st_size == 0:
                    continue
                declared.append(
                    {
                        "artifact_id": f"action-step-{int(match.group(1))}",
                        "kind": "action_clip",
                        "relative_path": path.relative_to(self.root).as_posix(),
                    }
                )
        return declared

    def _finalize_action_camera_clips(
        self,
        rows: list[dict[str, Any]],
        *,
        publish_ready: bool = False,
        notify: bool = False,
        ready_operation_id: str | None = None,
    ) -> list[dict[str, str]]:
        transitions: list[TransitionRecord] = []
        for row in rows:
            try:
                transition = TransitionRecord.from_payload(row)
            except (TypeError, ValueError):
                self._degrade("action camera clip unavailable")
                continue
            if transition.operation_id:
                transitions.append(transition)
        by_operation: dict[str, list[TransitionRecord]] = {}
        for transition in transitions:
            by_operation.setdefault(transition.operation_id or "", []).append(transition)
        if not by_operation:
            if publish_ready and ready_operation_id is not None:
                self._publish_action_camera_ready(
                    ready_operation_id, (), notify=notify
                )
            return []

        primary = next(camera for camera in self.cameras if camera.role == "primary")
        event_rows, _ = media.read_complete_jsonl(self._events_path)
        exact_events: set[tuple[str, str, str]] = set()
        ready_operations: set[str] = set()
        pair_artifacts: dict[tuple[str, str], set[str]] = {}
        artifact_pairs: dict[str, set[tuple[str, str]]] = {}
        for event in event_rows:
            if (
                event.get("event_type") == "action_camera_clips_ready"
                and isinstance(event.get("operation_id"), str)
            ):
                ready_operations.add(event["operation_id"])
                continue
            if event.get("event_type") != "action_camera_clip_available":
                continue
            payload = event.get("payload")
            operation_id = event.get("operation_id")
            if not isinstance(payload, dict) or not isinstance(operation_id, str):
                continue
            camera_id = payload.get("camera_id")
            artifact_id = payload.get("artifact_id")
            if not isinstance(camera_id, str) or not isinstance(artifact_id, str):
                continue
            pair = (operation_id, camera_id)
            exact_events.add((operation_id, camera_id, artifact_id))
            pair_artifacts.setdefault(pair, set()).add(artifact_id)
            artifact_pairs.setdefault(artifact_id, set()).add(pair)

        retained = self._retained_camera_ids()
        declarations: list[dict[str, str]] = []
        seen_targets: dict[Path, tuple[str, str]] = {}
        seen_artifacts: dict[str, tuple[str, str]] = {}
        for operation_id, operation_transitions in by_operation.items():
            expected_steps = [item.step_id for item in operation_transitions]
            operation_declarations: list[dict[str, str]] = []
            for camera_id in retained:
                pair = (operation_id, camera_id)
                artifact_id = media.action_camera_artifact_id(*pair)
                target = (
                    self.root
                    / "action_videos"
                    / media.operation_storage_component(operation_id)
                    / f"{media.camera_storage_component(camera_id)}.mp4"
                )
                if (
                    target in seen_targets and seen_targets[target] != pair
                    or artifact_id in seen_artifacts
                    and seen_artifacts[artifact_id] != pair
                    or len(pair_artifacts.get(pair, {artifact_id})) != 1
                    or len(artifact_pairs.get(artifact_id, {pair})) != 1
                ):
                    self._degrade("action camera clip unavailable")
                    continue
                seen_targets[target] = pair
                seen_artifacts[artifact_id] = pair
                references = [
                    frame
                    for transition in operation_transitions
                    for frame in transition.frames
                    if frame.camera_id == camera_id
                ]
                if [frame.step_id for frame in references] != expected_steps:
                    self._degrade("action camera clip unavailable")
                    continue
                frame_paths: list[Path] = []
                if any(
                    (path := self.root / frame.relative_path) is None
                    or not self._safe_action_clip_path(path, require_file=True)
                    for frame in references
                ):
                    self._degrade("action camera clip unavailable")
                    continue
                frame_paths = [self.root / frame.relative_path for frame in references]
                reusable = self._safe_action_clip_path(target, require_file=True)
                if reusable:
                    try:
                        reusable = media.video_frame_count(target) == len(expected_steps)
                    except Exception:
                        reusable = False
                    if not reusable:
                        self._degrade("action camera clip unavailable")
                elif target.exists() or target.is_symlink():
                    try:
                        mode = target.lstat().st_mode
                    except OSError:
                        self._degrade("action camera clip unavailable")
                        continue
                    if not stat.S_ISREG(mode):
                        self._degrade("action camera clip unavailable")
                        continue
                if not reusable:
                    if not self._safe_action_clip_path(target.parent, require_file=False):
                        self._degrade("action camera clip unavailable")
                        continue
                    try:
                        media.finalize_validated_video(
                            target,
                            frame_paths,
                            primary.fps,
                            expected_frames=len(expected_steps),
                        )
                    except Exception:
                        self._degrade("action camera clip unavailable")
                        continue
                event_identity = (operation_id, camera_id, artifact_id)
                if event_identity not in exact_events:
                    try:
                        with self._mutex:
                            event = self._make_event(
                                "action_camera_clip_available",
                                {"camera_id": camera_id, "artifact_id": artifact_id},
                                operation_id=operation_id,
                            )
                            self._commit_event(event, notify=notify)
                    except media.JsonlRollbackError as exc:
                        self._poison("action camera event rollback failure", exc)
                        raise
                    except OSError:
                        self._degrade("action camera clip unavailable")
                        continue
                    exact_events.add(event_identity)
                declaration = {
                    "artifact_id": artifact_id,
                    "kind": "action_camera_clip",
                    "relative_path": target.relative_to(self.root).as_posix(),
                }
                declarations.append(declaration)
                operation_declarations.append(declaration)
            if publish_ready and operation_id not in ready_operations:
                self._publish_action_camera_ready(
                    operation_id,
                    tuple(operation_declarations),
                    notify=notify,
                )
                ready_operations.add(operation_id)
        if (
            publish_ready
            and ready_operation_id is not None
            and ready_operation_id not in ready_operations
        ):
            self._publish_action_camera_ready(ready_operation_id, (), notify=notify)
        return declarations

    def _publish_action_camera_ready(
        self,
        operation_id: str,
        declarations: tuple[dict[str, str], ...],
        *,
        notify: bool,
    ) -> None:
        clips = [
            {
                "camera_id": camera_id,
                "artifact_id": declaration["artifact_id"],
            }
            for declaration in declarations
            for camera_id in self._retained_camera_ids()
            if declaration["relative_path"].endswith(
                f"/{media.camera_storage_component(camera_id)}.mp4"
            )
        ]
        with self._mutex:
            event = self._make_event(
                "action_camera_clips_ready",
                {"clips": clips},
                operation_id=operation_id,
            )
            self._commit_event(event, notify=notify)

    def _safe_action_clip_path(self, path: Path, *, require_file: bool) -> bool:
        try:
            root = self.root.resolve(strict=True)
            relative = path.relative_to(self.root)
            current = self.root
            for part in relative.parts:
                current /= part
                try:
                    metadata = current.lstat()
                except FileNotFoundError:
                    return not require_file
                if stat.S_ISLNK(metadata.st_mode):
                    return False
            resolved = path.resolve(strict=require_file)
            if not resolved.is_relative_to(root):
                return False
            if require_file:
                metadata = path.lstat()
                return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0
            return not path.exists() or stat.S_ISDIR(path.lstat().st_mode)
        except (OSError, ValueError):
            return False

    def close(self) -> None:
        self._drain_operation_clip_worker()
        with self._mutex:
            if self._closed:
                return
            self._retry_health()
            self._closed = True
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()

    def __enter__(self) -> CaseRecorder:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CaseRecorder"]
