"""Private workspace discovery for the single RPent Dashboard process."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import queue
import stat
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx

from rpent.recording import media

_DESCRIPTOR_FIELDS = {
    "schema_version",
    "instance_id",
    "pid",
    "origin",
    "runs_root",
    "notification_token",
    "started_at",
}
_MAX_DESCRIPTOR_BYTES = 64 * 1024
LifecycleHintKind = Literal[
    "evaluation_persisted",
    "attempt_registered",
    "result_accepted",
]
logger = logging.getLogger(__name__)
_MAX_CORRELATED_REFRESH_ATTEMPTS = 2
_PUBLISHER_STOP = object()


@dataclass(frozen=True, slots=True)
class LifecycleHint:
    """Identity-only notice that durable Evaluation state may have changed."""

    schema_version: int
    kind: LifecycleHintKind
    evaluation_id: str
    manifest_id: str
    case_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("invalid lifecycle hint schema")
        if self.kind not in {
            "evaluation_persisted",
            "attempt_registered",
            "result_accepted",
        }:
            raise ValueError("invalid lifecycle hint kind")
        for name in ("evaluation_id", "manifest_id"):
            _identity(getattr(self, name), name)
        if self.kind == "evaluation_persisted":
            if self.case_id is not None or self.attempt_id is not None:
                raise ValueError("evaluation-persisted hint must not identify an attempt")
        else:
            _identity(self.case_id, "case_id")
            _identity(self.attempt_id, "attempt_id")

    def to_payload(self) -> dict[str, int | str]:
        payload: dict[str, int | str] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "evaluation_id": self.evaluation_id,
            "manifest_id": self.manifest_id,
        }
        if self.case_id is not None:
            payload["case_id"] = self.case_id
        if self.attempt_id is not None:
            payload["attempt_id"] = self.attempt_id
        return payload


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not 1 <= len(value) <= 512:
        raise ValueError(f"{name} must be a string between 1 and 512 characters")
    return value


_AsyncPost = Callable[..., Awaitable[httpx.Response]]


class LifecycleHintPublisher:
    """Best-effort bounded delivery to the current workspace Dashboard."""

    def __init__(
        self,
        descriptor_path: str | Path,
        *,
        expected_runs_root: str | Path,
        post: _AsyncPost | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        on_failure: Callable[[], object] | None = None,
    ) -> None:
        if post is not None and transport is not None:
            raise ValueError("provide post or transport, not both")
        self._descriptor_path = Path(descriptor_path)
        self._expected_runs_root = Path(expected_runs_root).resolve()
        self._post = post
        self._transport = transport
        self._on_failure = on_failure
        self._timeout = httpx.Timeout(0.25, connect=0.1)
        self._hints: queue.Queue[LifecycleHint | object] = queue.Queue(16)
        self._stop = threading.Event()
        self._close_lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name=f"rpent-lifecycle-hint-{id(self):x}",
            daemon=True,
        )
        self._worker.start()

    def publish(self, hint: LifecycleHint) -> bool:
        if not isinstance(hint, LifecycleHint):
            raise TypeError("hint must be a LifecycleHint")
        with self._close_lock:
            if self._closed:
                return False
            try:
                self._hints.put_nowait(LifecycleHint(**hint.to_payload()))
            except queue.Full:
                return False
        return True

    def close(self) -> None:
        """Stop delivery and boundedly join the one owned worker thread."""
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self._stop.set()
                try:
                    self._hints.put_nowait(_PUBLISHER_STOP)
                except queue.Full:
                    pass
            worker = self._worker
        if worker is not threading.current_thread():
            worker.join(timeout=0.75)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                hint = self._hints.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if hint is _PUBLISHER_STOP or self._stop.is_set():
                    return
                assert isinstance(hint, LifecycleHint)
                try:
                    asyncio.run(self._send(hint))
                except BaseException:
                    self._notify_failure()
            finally:
                self._hints.task_done()

    def _notify_failure(self) -> None:
        callback = self._on_failure
        if callback is None:
            return
        try:
            callback()
        except BaseException:
            pass

    async def _send(self, hint: LifecycleHint) -> bool:
        return await asyncio.wait_for(self._request(hint), timeout=0.25)

    async def _request(self, hint: LifecycleHint) -> bool:
        descriptor = DashboardDescriptor.load(
            self._descriptor_path,
            expected_runs_root=self._expected_runs_root,
            expected_descriptor_path=self._descriptor_path,
        )
        response: httpx.Response | None = None
        async with httpx.AsyncClient(
            timeout=self._timeout,
            transport=self._transport,
            trust_env=False,
        ) as client:
            try:
                post = self._post or client.post
                response = await post(
                    f"{descriptor.origin}/api/internal/lifecycle",
                    json=hint.to_payload(),
                    headers={
                        "X-RPent-Notification-Token": descriptor.notification_token
                    },
                    timeout=self._timeout,
                )
                await response.aread()
                response.raise_for_status()
                return True
            finally:
                if response is not None:
                    await response.aclose()


class DashboardAttachmentRefresher:
    """Reload durable external Evaluation state into one live Dashboard."""

    def __init__(self, repository: Any, runtime_store: Any) -> None:
        self._repository = repository
        self._runtime_store = runtime_store
        self._lock = threading.Lock()
        self._fingerprint: tuple[Any, ...] = ()
        self._retry_unresolved = False
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    def accept(self, hint: LifecycleHint) -> None:
        if not isinstance(hint, LifecycleHint):
            raise TypeError("hint must be a LifecycleHint")
        with self._lock:
            fingerprint = self._reload_index_locked()
            if fingerprint is None:
                raise ValueError("lifecycle index changed during refresh")
            metadata = self._repository.evaluation_metadata(hint.evaluation_id)
            if metadata.manifest_id != hint.manifest_id:
                raise ValueError("lifecycle manifest identity mismatch")
            if hint.kind == "evaluation_persisted":
                self._fingerprint = fingerprint
                self._retry_unresolved = True
                return
            assert hint.case_id is not None and hint.attempt_id is not None
            if hint.case_id not in metadata.case_ids:
                raise KeyError((hint.evaluation_id, hint.case_id))
            if hint.kind == "attempt_registered":
                registration, manifest, case = (
                    self._repository.load_runtime_registration(
                        hint.evaluation_id,
                        hint.manifest_id,
                        hint.case_id,
                        hint.attempt_id,
                    )
                )
                self._runtime_store.start_attempt(registration, manifest, case)
                self._fingerprint = fingerprint
                self._retry_unresolved = True
                return
            self._accept_result_locked(hint)
            self._fingerprint = fingerprint

    def refresh_if_changed(self) -> None:
        with self._lock:
            fingerprint = tuple(self._repository.index_fingerprint())
            if fingerprint == self._fingerprint and not self._retry_unresolved:
                return
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        fingerprint = self._reload_index_locked()
        if fingerprint is None:
            return
        self._recover_snapshot_locked(fingerprint)

    def _reload_index_locked(self) -> tuple[Any, ...] | None:
        for _attempt in range(_MAX_CORRELATED_REFRESH_ATTEMPTS):
            before = tuple(self._repository.index_fingerprint())
            self._repository.refresh_index(force=True, recover=False)
            fingerprint = tuple(self._repository.index_fingerprint())
            if before == fingerprint:
                return fingerprint
        return None

    def _recover_snapshot_locked(self, fingerprint: tuple[Any, ...]) -> None:
        snapshot = getattr(
            self._repository,
            "startup_runtime_registration_snapshot",
            None,
        )
        if callable(snapshot):
            registrations, unresolved = snapshot()
        else:
            registrations = self._repository.startup_runtime_registrations()
            unresolved = False
        if registrations:
            get_entry = getattr(self._runtime_store, "get_entry", None)
            missing = (
                tuple(
                    (registration, manifest, case)
                    for registration, manifest, case in registrations
                    if get_entry(
                        registration.evaluation_id,
                        registration.case_id,
                        registration.attempt_id,
                    )
                    is None
                )
                if callable(get_entry)
                else registrations
            )
            if missing:
                self._runtime_store.recover_attempts(missing)
        list_evaluations = getattr(self._repository, "list_evaluations", None)
        if callable(list_evaluations):
            for item in list_evaluations():
                if item.completed == 0:
                    continue
                evaluation = self._repository.get_evaluation(item.evaluation_id)
                for case in evaluation.cases:
                    entry = self._runtime_store.get_entry(
                        item.evaluation_id,
                        case.case_id,
                    )
                    if entry is None or entry.phase == "sealed":
                        continue
                    detail = self._repository.reconstruct_case_detail(
                        item.evaluation_id,
                        case.case_id,
                        attempt_id=entry.value.attempt_id,
                    )
                    if detail.episode_result is None:
                        continue
                    from rpent.evaluation.results import EpisodeResult

                    result = EpisodeResult.from_payload(detail.episode_result)
                    operation_page = self._terminal_operation_page(detail)
                    self._runtime_store.mark_terminal_pending(result)
                    self._runtime_store.publish_terminal(
                        detail, result, operation_page
                    )
        self._fingerprint = fingerprint
        self._retry_unresolved = unresolved

    def _accept_result_locked(self, hint: LifecycleHint) -> None:
        assert hint.case_id is not None and hint.attempt_id is not None
        try:
            detail = self._repository.reconstruct_case_detail(
                hint.evaluation_id,
                hint.case_id,
                attempt_id=hint.attempt_id,
            )
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError("lifecycle result is not durably accepted") from exc
        payload = detail.episode_result
        if payload is None:
            raise ValueError("lifecycle result is not durably accepted")
        from rpent.evaluation.results import EpisodeResult

        result = EpisodeResult.from_payload(payload)
        if (
            detail.manifest_id != hint.manifest_id
            or result.evaluation_id != hint.evaluation_id
            or result.manifest_id != hint.manifest_id
            or result.case_id != hint.case_id
            or result.attempt_id != hint.attempt_id
        ):
            raise ValueError("lifecycle result identity mismatch")
        self._repository.note_accepted_result(result, detail)
        durable_current = self._repository.current_attempt_identity(
            hint.evaluation_id,
            hint.case_id,
        )
        if durable_current.attempt_id != hint.attempt_id:
            if durable_current.attempt_id is not None:
                registration, manifest, case = (
                    self._repository.load_runtime_registration(
                        hint.evaluation_id,
                        hint.manifest_id,
                        hint.case_id,
                        durable_current.attempt_id,
                    )
                )
                self._runtime_store.start_attempt(registration, manifest, case)
            return
        entry = self._runtime_store.get_entry(
            hint.evaluation_id,
            hint.case_id,
            hint.attempt_id,
        )
        if entry is not None and entry.phase == "sealed":
            if entry.value.episode_result != result.to_payload():
                raise ValueError("lifecycle result identity drift")
            return
        current = self._runtime_store.get_entry(
            hint.evaluation_id,
            hint.case_id,
        )
        if current is not None and current.value.attempt_id != hint.attempt_id:
            raise ValueError("lifecycle result Attempt is not current")
        if entry is None:
            registration, manifest, case = (
                self._repository.load_runtime_registration(
                    hint.evaluation_id,
                    hint.manifest_id,
                    hint.case_id,
                    hint.attempt_id,
                    require_current=False,
                )
            )
            self._runtime_store.start_attempt(registration, manifest, case)
        operation_page = self._terminal_operation_page(detail)
        self._runtime_store.mark_terminal_pending(result)
        self._runtime_store.publish_terminal(detail, result, operation_page)

    def _terminal_operation_page(self, detail: Any) -> Any:
        loader = getattr(
            self._repository, "load_terminal_operation_snapshot", None
        )
        reconstructor = getattr(self._repository, "reconstruct_operations", None)
        if not callable(loader) or not callable(reconstructor):
            raise TypeError("repository must expose terminal operation authority")
        try:
            return loader(
                detail.evaluation_id,
                detail.case_id,
                detail.attempt_id,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
            page = reconstructor(
                detail.evaluation_id,
                detail.case_id,
                attempt_id=detail.attempt_id,
            )
            persister = getattr(
                self._repository, "persist_terminal_operation_snapshot", None
            )
            if callable(persister):
                persister(page)
            return page


class IndexFingerprintWatcher:
    """Poll one repository index fingerprint without scanning Evaluation roots."""

    def __init__(self, refresher: Any) -> None:
        self._refresher = refresher
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="rpent-index-fingerprint",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self._refresher.refresh_if_changed()
            except Exception:
                logger.exception("dashboard Evaluation index refresh failed")

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)


class DashboardAlreadyRunning(RuntimeError):
    """Raised when another Dashboard owns the workspace lease."""


def validate_loopback_host(value: object) -> str:
    """Return one canonical host that cannot bind outside loopback."""
    if type(value) is not str:
        raise ValueError("dashboard host must be a canonical loopback address")
    if value in {"127.0.0.1", "::1"}:
        return value
    raise ValueError("dashboard host must be a canonical loopback address")


def _http_origin(value: object) -> str:
    if type(value) is not str or any(
        ord(character) < 0x21 or ord(character) == 0x7F for character in value
    ):
        raise ValueError("dashboard origin must be a plain loopback HTTP origin")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "dashboard origin must be a plain loopback HTTP origin"
        ) from exc
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("dashboard origin must be a plain loopback HTTP origin")
    try:
        host = validate_loopback_host(parsed.hostname)
    except ValueError as exc:
        raise ValueError(
            "dashboard origin must be a plain loopback HTTP origin"
        ) from exc
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority = f"{authority}:{port}"
    if value != f"http://{authority}":
        raise ValueError("dashboard origin must be a plain loopback HTTP origin")
    return value


def _header_value(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise ValueError("invalid dashboard notification header value")
    return value


def _utc_timestamp(value: object) -> str:
    if type(value) is not str or not value.endswith(("+00:00", "Z")):
        raise ValueError("dashboard started_at must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(
            f"{value[:-1]}+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError("dashboard started_at must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("dashboard started_at must be a UTC timestamp")
    return value


@dataclass(frozen=True, slots=True)
class DashboardDescriptor:
    """Validated connection details for one live workspace Dashboard."""

    schema_version: int
    instance_id: str
    pid: int
    origin: str
    runs_root: Path
    notification_token: str
    started_at: str
    path: Path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_runs_root: str | Path | None = None,
        expected_descriptor_path: str | Path | None = None,
    ) -> DashboardDescriptor:
        selected = Path(path)
        expected_root = (
            Path(expected_runs_root).resolve()
            if expected_runs_root is not None
            else None
        )
        info = selected.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("dashboard descriptor must be a private regular file")
        if info.st_size > _MAX_DESCRIPTOR_BYTES:
            raise ValueError("dashboard descriptor must not exceed 64 KiB")

        resolved_path = selected.resolve()
        expected_path = (
            Path(expected_descriptor_path).resolve()
            if expected_descriptor_path is not None
            else (
                expected_root / ".rpent-dashboard.json"
                if expected_root is not None
                else resolved_path
            )
        )
        if (
            resolved_path != expected_path
            or resolved_path.name != ".rpent-dashboard.json"
        ):
            raise ValueError("dashboard descriptor workspace mismatch")

        payload = json.loads(selected.read_bytes())
        if not isinstance(payload, dict) or set(payload) != _DESCRIPTOR_FIELDS:
            raise ValueError("invalid dashboard descriptor fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("invalid dashboard descriptor schema")

        instance_id = payload["instance_id"]
        if type(instance_id) is not str:
            raise ValueError("invalid dashboard instance ID")
        try:
            UUID(instance_id)
        except ValueError as exc:
            raise ValueError("invalid dashboard instance ID") from exc

        pid = payload["pid"]
        if type(pid) is not int or pid <= 0:
            raise ValueError("invalid dashboard pid")

        raw_runs_root = payload["runs_root"]
        if type(raw_runs_root) is not str or not Path(raw_runs_root).is_absolute():
            raise ValueError("dashboard descriptor workspace mismatch")
        runs_root = Path(raw_runs_root).resolve()
        if expected_root is not None and runs_root != expected_root:
            raise ValueError("dashboard descriptor workspace mismatch")

        origin = _http_origin(payload["origin"])
        token = _header_value(payload["notification_token"])
        started_at = _utc_timestamp(payload["started_at"])
        return cls(
            1,
            instance_id,
            pid,
            origin,
            runs_root,
            token,
            started_at,
            resolved_path,
        )


class DashboardLease:
    """Own the advisory singleton lock for one resolved runs root."""

    def __init__(
        self,
        runs_root: Path,
        runtime_dir: Path,
        stream: BinaryIO,
        instance_id: str,
    ) -> None:
        self.runs_root = runs_root
        self._stream = stream
        self.instance_id = instance_id
        self.descriptor_path = runtime_dir / ".rpent-dashboard.json"

    @classmethod
    def acquire(
        cls,
        runs_root: str | Path,
        *,
        runtime_dir: str | Path | None = None,
    ) -> DashboardLease:
        root = Path(runs_root).resolve()
        metadata_root = Path(runtime_dir).resolve() if runtime_dir else root
        root.mkdir(parents=True, exist_ok=True)
        metadata_root.mkdir(parents=True, exist_ok=True)
        stream = (metadata_root / ".rpent-dashboard.lock").open("a+b")
        try:
            os.fchmod(stream.fileno(), 0o600)
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DashboardAlreadyRunning(
                    f"Dashboard already running for {root}"
                ) from exc
            return cls(root, metadata_root, stream, uuid4().hex)
        except BaseException:
            stream.close()
            raise

    def publish(self, origin: str, token: str) -> DashboardDescriptor:
        if self._stream.closed:
            raise RuntimeError("dashboard lease is closed")
        payload = {
            "schema_version": 1,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "origin": _http_origin(origin),
            "runs_root": str(self.runs_root),
            "notification_token": _header_value(token),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        media.atomic_write_json(self.descriptor_path, payload)
        os.chmod(self.descriptor_path, 0o600)
        return DashboardDescriptor.load(
            self.descriptor_path,
            expected_runs_root=self.runs_root,
            expected_descriptor_path=self.descriptor_path,
        )

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            try:
                descriptor = DashboardDescriptor.load(
                    self.descriptor_path,
                    expected_runs_root=self.runs_root,
                    expected_descriptor_path=self.descriptor_path,
                )
            except (OSError, ValueError):
                pass
            else:
                if descriptor.instance_id == self.instance_id:
                    self.descriptor_path.unlink()
        finally:
            try:
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()


__all__ = [
    "DashboardAttachmentRefresher",
    "DashboardAlreadyRunning",
    "DashboardDescriptor",
    "DashboardLease",
    "IndexFingerprintWatcher",
    "LifecycleHint",
    "LifecycleHintPublisher",
    "validate_loopback_host",
]
