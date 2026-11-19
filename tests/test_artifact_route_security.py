from __future__ import annotations

import asyncio
import gc
import hashlib
import io
import json
import os
import re
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from rpent.episode.contracts import StepResult
from rpent.evaluation.attempt_registration import register_attempt
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult
from rpent.recording import media as recording_media
from rpent.recording.contracts import CameraDescriptor, RecordingProfile
from rpent.recording.display import png_bytes
from rpent.recording.recorder import CaseRecorder
from rpent.ui.app import create_evaluation_app
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout, RepositoryReadGate
from rpent.ui.repository import EvaluationRepository, MediaBytes
from rpent.ui.routes.cases import (
    _artifact_response,
    _close_media_stream,
    _MediaStreamOwner,
    _OwnedStreamingResponse,
)
from tests.test_ui_repository import (
    _append_jsonl,
    _attempt_root,
    _result,
    _write_artifacts,
    _write_evaluation,
    _write_result,
)


def _indexed_repository(
    root: Path,
    *,
    boundary: Path | None = None,
    **kwargs,
) -> EvaluationRepository:
    repository = EvaluationRepository(
        ((root.parent if boundary is None else boundary),),
        **kwargs,
    )
    repository.register_evaluation(
        root,
        EvaluationManifest.load(root / "evaluation_manifest.json"),
    )
    return repository


class _ArtifactRepository:
    def __init__(self, streams, *, artifact_kinds=None):
        self.streams = streams
        self.artifact_kinds = artifact_kinds or {}
        self.calls = []

    def open_artifact(self, evaluation_id, case_id, artifact_id, attempt_id=None):
        self.calls.append(
            (evaluation_id, case_id, artifact_id)
            if attempt_id is None
            else (evaluation_id, case_id, artifact_id, attempt_id)
        )
        value = self.streams.get((evaluation_id, case_id, artifact_id))
        if value is None:
            raise KeyError(artifact_id)
        if isinstance(value, BaseException):
            raise value
        return value() if callable(value) else value

    def open_live_frame(self, evaluation_id, case_id, camera_id):
        raise AssertionError("camera routes must use bounded byte reads")

    def read_live_frame(self, evaluation_id, case_id, camera_id):
        self.calls.append(("live", evaluation_id, case_id, camera_id))
        return self._media_bytes(
            self.streams[(evaluation_id, case_id, f"live:{camera_id}")]
        )

    def open_indexed_frame(
        self, evaluation_id, case_id, camera_id, frame_index, attempt_id=None
    ):
        raise AssertionError("camera routes must use bounded byte reads")

    def read_indexed_frame(
        self, evaluation_id, case_id, camera_id, frame_index, attempt_id=None
    ):
        self.calls.append(
            ("frame", evaluation_id, case_id, camera_id, frame_index)
            if attempt_id is None
            else (
                "frame",
                evaluation_id,
                case_id,
                camera_id,
                frame_index,
                attempt_id,
            )
        )
        return self._media_bytes(
            self.streams[(evaluation_id, case_id, f"frame:{camera_id}:{frame_index}")]
        )

    def read_tool_image(
        self, evaluation_id, case_id, sequence, attempt_id=None
    ):
        self.calls.append(
            ("tool-image", evaluation_id, case_id, sequence)
            if attempt_id is None
            else ("tool-image", evaluation_id, case_id, sequence, attempt_id)
        )
        value = self.streams.get(
            (evaluation_id, case_id, f"tool-image:{sequence}")
        )
        if value is None:
            raise KeyError(sequence)
        if isinstance(value, BaseException):
            raise value
        return self._media_bytes(value)

    @staticmethod
    def _media_bytes(value):
        stream = value() if callable(value) else value
        try:
            data = stream.read()
        finally:
            stream.close()
        media_type = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        return MediaBytes(
            data=data,
            media_type=media_type,
            etag=f'"{hashlib.sha256(data).hexdigest()}"',
            size=len(data),
        )

    def get_case(self, evaluation_id, case_id, *, attempt_id=None):
        if (evaluation_id, case_id) != ("evaluation-1", "case-1"):
            raise KeyError((evaluation_id, case_id))
        return SimpleNamespace(
            artifacts=tuple(
                SimpleNamespace(artifact_id=artifact_id, kind=kind)
                for artifact_id, kind in self.artifact_kinds.items()
            )
        )


class _NamedBytes(io.BytesIO):
    def __init__(self, value: bytes, *, name: str, nlink: int = 1):
        super().__init__(value)
        self.name = name
        self.nlink = nlink

    def fileno(self):
        raise OSError


class _ThreadObservedBytes(_NamedBytes):
    def __init__(self, value: bytes, *, name: str):
        super().__init__(value, name=name)
        self.threads = []

    def _record(self) -> None:
        self.threads.append(threading.current_thread().name)

    def fileno(self):
        self._record()
        return super().fileno()

    def read(self, *args, **kwargs):
        self._record()
        return super().read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        self._record()
        return super().seek(*args, **kwargs)

    def tell(self):
        self._record()
        return super().tell()

    def close(self):
        self._record()
        return super().close()


class _CountingBytes(_NamedBytes):
    def __init__(self, value: bytes, *, name: str):
        super().__init__(value, name=name)
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        return super().close()


class _ErrorReadGate:
    def __init__(self, error: BaseException):
        self.error = error

    async def run(self, key, function, *, timeout_s=None):
        raise self.error


class _LateTimeoutReadGate:
    def __init__(self):
        self.threads = []

    async def run(self, key, function, *, timeout_s=None):
        thread = threading.Thread(target=function)
        self.threads.append(thread)
        thread.start()
        await asyncio.sleep(0)
        raise ReadGateTimeout(key)


class _OpenThenBusyReadGate:
    def __init__(self):
        self.calls = 0

    async def run(self, key, function, *, timeout_s=None):
        self.calls += 1
        if self.calls == 1:
            return function()
        raise ReadGateBusy(key)


class _BlockingReadBytes(_NamedBytes):
    def __init__(self, value: bytes, *, name: str):
        super().__init__(value, name=name)
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def read(self, *args, **kwargs):
        if self.block:
            self.started.set()
            self.release.wait(1)
        return super().read(*args, **kwargs)


class _OpenThenTimedOutReadGate:
    def __init__(self, stream: _BlockingReadBytes):
        self.stream = stream
        self.calls = 0
        self.threads = []

    async def run(self, key, function, *, timeout_s=None):
        self.calls += 1
        if self.calls == 1:
            result = function()
            self.stream.block = True
            return result
        if self.calls == 2:
            thread = threading.Thread(target=function)
            self.threads.append(thread)
            thread.start()
            while not self.stream.started.is_set():
                await asyncio.sleep(0)
            raise ReadGateTimeout(key)
        raise ReadGateBusy(key)


class _BlockingCloseGate:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_attempts = 0

    async def run(self, key, function, *, timeout_s=None):
        if key[0] == "artifact-close":
            self.close_attempts += 1
            self.started.set()
            await self.release.wait()
        return function()


class _ShortDeadlineGate:
    def __init__(self, gate: RepositoryReadGate):
        self.gate = gate

    async def run(self, key, function, *, timeout_s=None):
        return await self.gate.run(key, function, timeout_s=0.02)


def _client(repository, *, repository_read_gate=None, media_read_gate=None):
    return TestClient(
        create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
            **(
                {}
                if repository_read_gate is None
                else {"repository_read_gate": repository_read_gate}
            ),
            **({} if media_read_gate is None else {"media_read_gate": media_read_gate}),
        )
    )


async def _wait_for_gate(gate: RepositoryReadGate, predicate) -> None:
    async with asyncio.timeout(1):
        while not predicate(gate.snapshot()):
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("method_name", ["read_live_frame", "read_indexed_frame"])
def test_repository_camera_byte_reads_are_immutable_and_bounded(method_name) -> None:
    png = b"\x89PNG\r\n\x1a\nbody"
    repository = EvaluationRepository(())
    if method_name == "read_live_frame":
        repository.open_live_frame = lambda *args, **kwargs: io.BytesIO(png)
        result = repository.read_live_frame("evaluation-1", "case-1", "camera")
    else:
        repository.open_indexed_frame = lambda *args, **kwargs: io.BytesIO(png)
        result = repository.read_indexed_frame("evaluation-1", "case-1", "camera", 3)

    assert isinstance(result, MediaBytes)
    assert result.data == png
    assert result.media_type == "image/png"
    assert result.etag.startswith('"')
    assert result.size == len(png)

    oversized = b"x" * (8 * 1024 * 1024 + 1)
    if method_name == "read_live_frame":
        repository.open_live_frame = lambda *args, **kwargs: io.BytesIO(oversized)

        def call():
            return repository.read_live_frame("evaluation-1", "case-1", "camera")

    else:
        repository.open_indexed_frame = lambda *args, **kwargs: io.BytesIO(oversized)

        def call():
            return repository.read_indexed_frame("evaluation-1", "case-1", "camera", 3)

    with pytest.raises(ValueError, match="8 MiB|size limit"):
        call()


def test_artifact_streams_the_repository_opened_fd_without_reopening(monkeypatch):
    stream = _NamedBytes(b"\x89PNG\r\n\x1a\nbody", name="frame.png")
    repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
    opened = []
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: opened.append(args))

    response = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
    )

    assert response.status_code == 200
    assert response.content.startswith(b"\x89PNG")
    assert response.headers["content-type"] == "image/png"
    assert opened == []
    assert stream.closed


def test_large_artifact_open_read_and_close_use_the_owned_media_gate() -> None:
    stream = _ThreadObservedBytes(
        b"\x1aE\xdf\xa3" + b"video" * 20_000,
        name="retained.webm",
    )
    repository = _ArtifactRepository({("evaluation-1", "case-1", "video"): stream})

    with _client(repository) as client:
        response = client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/video"
        )

    assert response.status_code == 200
    assert response.content.startswith(b"\x1aE\xdf\xa3")
    assert stream.closed
    assert stream.threads
    assert all(name.startswith("rpent-ui-media") for name in stream.threads)


@pytest.mark.parametrize(
    "failed_message_type",
    ["http.response.start", "http.response.body"],
)
def test_artifact_send_failure_closes_retained_stream(
    failed_message_type: str,
) -> None:
    async def run() -> None:
        stream = _NamedBytes(b"\x89PNG\r\n\x1a\nbody", name="frame.png")
        repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
        app = create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "asgi": {"spec_version": "2.4"},
            "app": app,
        }
        request = Request(scope)
        try:
            response = await _artifact_response(
                request,
                "evaluation-1",
                "case-1",
                "frame",
                cache_control="no-store",
            )
            assert stream.closed is False

            async def receive():
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == failed_message_type:
                    raise OSError("send failed")

            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
            assert stream.closed
        finally:
            app.state.close_owned_read_gates()

    asyncio.run(run())


def test_artifact_close_submission_failure_keeps_ownership_retryable() -> None:
    async def run() -> None:
        stream = _CountingBytes(b"payload", name="artifact.bin")
        owner = _MediaStreamOwner()
        assert owner.publish(stream)
        operation_lock = threading.Lock()
        gate = RepositoryReadGate(workers=1, pending=0)
        request = Request(
            {
                "type": "http",
                "app": SimpleNamespace(state=SimpleNamespace(media_read_gate=gate)),
            }
        )

        def reject_submission(*args, **kwargs):
            gate.close()
            raise RuntimeError("cannot schedule new futures after shutdown")

        gate._executor.submit = reject_submission
        operation_lock.acquire()
        try:
            await _close_media_stream(request, object(), operation_lock, owner)
            assert stream.closed is False
        finally:
            operation_lock.release()

        await _close_media_stream(request, object(), operation_lock, owner)
        assert stream.closed
        assert stream.close_calls == 1

    asyncio.run(run())


def test_timed_out_queued_artifact_close_survives_gate_shutdown() -> None:
    async def run() -> None:
        gate = RepositoryReadGate(workers=1, pending=1)
        blocker_started = threading.Event()
        blocker_release = threading.Event()

        def blocked() -> None:
            blocker_started.set()
            assert blocker_release.wait(1)

        blocker = asyncio.create_task(gate.run(("blocker",), blocked))
        stream = _CountingBytes(b"payload", name="artifact.bin")
        owner = _MediaStreamOwner()
        assert owner.publish(stream)
        operation_lock = threading.Lock()
        request = Request(
            {
                "type": "http",
                "app": SimpleNamespace(
                    state=SimpleNamespace(media_read_gate=_ShortDeadlineGate(gate))
                ),
            }
        )
        try:
            await _wait_for_gate(gate, lambda snapshot: snapshot.running == 1)
            assert blocker_started.is_set()
            await _close_media_stream(request, object(), operation_lock, owner)
            assert gate.snapshot().queued == 1
            gate.close()
            await _wait_for_gate(gate, lambda snapshot: snapshot.queued == 0)
            assert stream.closed
            assert stream.close_calls == 1
        finally:
            blocker_release.set()
            await asyncio.gather(blocker, return_exceptions=True)
            gate.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failed_message_type",
    ["http.response.start", "http.response.body"],
)
def test_transport_error_survives_cancelled_queued_artifact_cleanup(
    failed_message_type: str,
) -> None:
    async def run() -> None:
        gate = RepositoryReadGate(workers=1, pending=1)
        stream = _CountingBytes(
            b"\x89PNG\r\n\x1a\nbody",
            name="frame.png",
        )
        repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
        app = create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
            media_read_gate=gate,
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "asgi": {"spec_version": "2.4"},
            "app": app,
        }
        response = await _artifact_response(
            Request(scope),
            "evaluation-1",
            "case-1",
            "frame",
            cache_control="no-store",
        )
        blocker_started = threading.Event()
        blocker_release = threading.Event()
        blocker = None

        def blocked() -> None:
            blocker_started.set()
            assert blocker_release.wait(1)

        async def start_blocker() -> None:
            nonlocal blocker
            blocker = asyncio.create_task(gate.run(("blocker",), blocked))
            await _wait_for_gate(gate, lambda snapshot: snapshot.running == 1)

        if failed_message_type == "http.response.start":
            await start_blocker()

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] != failed_message_type:
                return
            if blocker is None:
                await start_blocker()
            raise OSError("send failed")

        response_task = asyncio.create_task(response(scope, receive, send))
        try:
            await _wait_for_gate(gate, lambda snapshot: snapshot.queued == 1)
            assert blocker_started.is_set()
            gate.close()
            with pytest.raises(ClientDisconnect):
                await response_task
            assert stream.closed
            assert stream.close_calls == 1
        finally:
            blocker_release.set()
            await asyncio.gather(
                response_task,
                *([] if blocker is None else [blocker]),
                return_exceptions=True,
            )
            gate.close()

    asyncio.run(run())


def test_artifact_response_cancellation_retries_interrupted_finalizer() -> None:
    async def run() -> None:
        stream = _CountingBytes(
            b"\x89PNG\r\n\x1a\nbody",
            name="frame.png",
        )
        gate = _BlockingCloseGate()
        repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
        app = create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
            media_read_gate=gate,
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "asgi": {"spec_version": "2.4"},
            "app": app,
        }
        response = await _artifact_response(
            Request(scope),
            "evaluation-1",
            "case-1",
            "frame",
            cache_control="no-store",
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        response_task = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(gate.started.wait(), 1)
        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task
        assert stream.closed
        assert stream.close_calls == 1
        gate.release.set()

    asyncio.run(run())


def test_artifact_response_retains_detached_finalizer_until_completion() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop_errors = []
        previous_handler = loop.get_exception_handler()
        finalizer_ref = None
        waiter_ref = None
        attempts = 0

        def capture_loop_error(_loop, context) -> None:
            loop_errors.append((context.get("message"), context.get("exception")))

        async def fail_finalizer() -> None:
            nonlocal attempts, finalizer_ref, waiter_ref
            attempts += 1
            finalizer = asyncio.current_task()
            assert finalizer is not None
            waiter = loop.create_future()
            finalizer_ref = weakref.ref(finalizer)
            waiter_ref = weakref.ref(waiter)
            await waiter
            raise RuntimeError("cleanup failed")

        response = _OwnedStreamingResponse(iter([b"frame"]), fail_finalizer)

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "asgi": {"spec_version": "2.4"},
        }
        loop.set_exception_handler(capture_loop_error)
        response_task = asyncio.create_task(response(scope, receive, send))
        try:
            async with asyncio.timeout(1):
                while waiter_ref is None:
                    await asyncio.sleep(0)
            response_task.cancel()
            await asyncio.sleep(0)
            assert not response_task.done()

            response_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(response_task, 1)

            gc.collect()
            await asyncio.sleep(0)
            assert finalizer_ref is not None
            assert waiter_ref is not None
            finalizer = finalizer_ref()
            waiter = waiter_ref()
            assert finalizer is not None, loop_errors
            assert waiter is not None
            assert not finalizer.done()

            waiter.set_result(None)
            del waiter
            async with asyncio.timeout(1):
                while not finalizer.done():
                    await asyncio.sleep(0)
            del finalizer
            async with asyncio.timeout(1):
                while finalizer_ref() is not None:
                    gc.collect()
                    await asyncio.sleep(0)

            assert attempts == 1
            assert waiter_ref() is None
            assert loop_errors == []
        finally:
            await asyncio.gather(response_task, return_exceptions=True)
            finalizer = None if finalizer_ref is None else finalizer_ref()
            waiter = None if waiter_ref is None else waiter_ref()
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
            if finalizer is not None:
                await asyncio.gather(finalizer, return_exceptions=True)
            gc.collect()
            await asyncio.sleep(0)
            loop.set_exception_handler(previous_handler)

    asyncio.run(run())


def test_queued_artifact_read_and_cleanup_close_stream_exactly_once() -> None:
    async def run() -> None:
        gate = RepositoryReadGate(workers=1, pending=2)
        stream = _CountingBytes(
            b"\x89PNG\r\n\x1a\nbody",
            name="frame.png",
        )
        repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
        app = create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
            media_read_gate=gate,
        )
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "asgi": {"spec_version": "2.4"},
            "app": app,
        }
        response = await _artifact_response(
            Request(scope),
            "evaluation-1",
            "case-1",
            "frame",
            cache_control="no-store",
        )
        blocker_release = threading.Event()

        def blocked() -> None:
            assert blocker_release.wait(1)

        blocker = asyncio.create_task(gate.run(("blocker",), blocked))
        await _wait_for_gate(gate, lambda snapshot: snapshot.running == 1)

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            return None

        response_task = asyncio.create_task(response(scope, receive, send))
        try:
            await _wait_for_gate(gate, lambda snapshot: snapshot.queued == 1)
            response_task.cancel()
            await _wait_for_gate(gate, lambda snapshot: snapshot.queued == 2)
            blocker_release.set()
            with pytest.raises(asyncio.CancelledError):
                await response_task
            assert stream.closed
            assert stream.close_calls == 1
        finally:
            blocker_release.set()
            await asyncio.gather(blocker, response_task, return_exceptions=True)
            gate.close()

    asyncio.run(run())


def test_artifact_response_preserves_transport_error_if_finalizer_raises() -> None:
    async def run() -> None:
        finalized = False

        async def fail_finalizer() -> None:
            nonlocal finalized
            finalized = True
            raise RuntimeError("cleanup failed")

        response = _OwnedStreamingResponse(iter([b"frame"]), fail_finalizer)

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("send failed")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "asgi": {"spec_version": "2.4"},
        }
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
        assert finalized

    asyncio.run(run())


def test_successful_artifact_response_closes_stream_exactly_once() -> None:
    stream = _CountingBytes(
        b"\x89PNG\r\n\x1a\nbody",
        name="frame.png",
    )
    repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})

    with _client(repository) as client:
        response = client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
        )

    assert response.status_code == 200
    assert stream.closed
    assert stream.close_calls == 1


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (ReadGateBusy(("artifact-open",)), 503),
        (ReadGateTimeout(("artifact-open",)), 504),
    ],
)
def test_large_artifact_media_gate_has_explicit_overload_status(
    error, status_code
) -> None:
    stream = _NamedBytes(b"\x89PNG\r\n\x1a\nbody", name="frame.png")
    repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})

    response = _client(repository, media_read_gate=_ErrorReadGate(error)).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
    )

    assert response.status_code == status_code
    assert response.headers.get("retry-after") == ("1" if status_code == 503 else None)


def test_timed_out_artifact_open_closes_the_late_file() -> None:
    stream = _NamedBytes(b"\x89PNG\r\n\x1a\nbody", name="frame.png")
    started = threading.Event()
    release = threading.Event()

    def delayed_stream():
        started.set()
        assert release.wait(1)
        return stream

    gate = _LateTimeoutReadGate()
    repository = _ArtifactRepository(
        {("evaluation-1", "case-1", "frame"): delayed_stream}
    )

    response = _client(repository, media_read_gate=gate).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
    )
    assert started.is_set()
    assert response.status_code == 504

    release.set()
    for thread in gate.threads:
        thread.join(1)
    assert stream.closed


def test_busy_artifact_close_uses_a_safe_fallback() -> None:
    stream = _NamedBytes(b"\x89PNG\r\n\x1a\nbody", name="frame.png")
    gate = _OpenThenBusyReadGate()
    repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})

    response = _client(repository, media_read_gate=gate).head(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
    )

    assert response.status_code == 200
    assert stream.closed


def test_timed_out_artifact_read_has_bounded_cleanup_and_eventual_close() -> None:
    async def run() -> None:
        stream = _BlockingReadBytes(
            b"\x89PNG\r\n\x1a\nbody",
            name="frame.png",
        )
        gate = _OpenThenTimedOutReadGate(stream)
        repository = _ArtifactRepository({("evaluation-1", "case-1", "frame"): stream})
        app = create_evaluation_app(
            repository=repository,
            service=SimpleNamespace(),
            event_hub=EvaluationEventHub(secret=b"fixed-test-secret"),
            simulator_catalog={},
            media_read_gate=gate,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                outcome = await asyncio.wait_for(
                    asyncio.gather(
                        client.get(
                            "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
                        ),
                        return_exceptions=True,
                    ),
                    0.2,
                )
            assert isinstance(outcome[0], BaseException)
        finally:
            stream.release.set()
            for thread in gate.threads:
                thread.join(1)
            app.state.close_owned_read_gates()
        assert stream.closed

    asyncio.run(run())


def test_retained_media_routes_forward_safe_attempt_scope():
    resource_id = f"camera-sha256-{'a' * 64}"
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "frame"): _NamedBytes(
                b"\x89PNG\r\n\x1a\nartifact", name="frame.png"
            ),
            ("evaluation-1", "case-1", f"frame:{resource_id}:3"): _NamedBytes(
                b"\x89PNG\r\n\x1a\nframe", name="frame.png"
            ),
        }
    )
    client = _client(repository)

    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame",
            params={"attempt_id": "attempt-old"},
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/cameras/"
            f"{resource_id}/frames/3",
            params={"attempt_id": "attempt-old"},
        ).status_code
        == 200
    )
    assert ("evaluation-1", "case-1", "frame", "attempt-old") in repository.calls
    assert (
        "frame",
        "evaluation-1",
        "case-1",
        resource_id,
        3,
        "attempt-old",
    ) in repository.calls


@pytest.mark.parametrize("attempt_id", ["../escape", "%2e%2e%2fescape", ""])
def test_retained_media_rejects_unsafe_attempt_before_repository_lookup(attempt_id):
    repository = _ArtifactRepository({})

    response = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame",
        params={"attempt_id": attempt_id},
    )

    assert response.status_code == 422
    assert repository.calls == []


@pytest.mark.parametrize(
    "artifact_id",
    ["../secret", "/etc/passwd", "%2e%2e", "%252e%252e%252fsecret"],
)
def test_artifact_route_rejects_traversal_before_repository_lookup(artifact_id):
    repository = _ArtifactRepository({})

    response = _client(repository).get(
        f"/api/evaluations/evaluation-1/cases/case-1/artifacts/{artifact_id}"
    )

    assert response.status_code in {404, 422}
    assert repository.calls == []


def test_artifact_identity_cannot_cross_case_or_evaluation():
    repository = _ArtifactRepository(
        {("evaluation-1", "case-1", "frame"): _NamedBytes(b"ok", name="frame.png")}
    )
    client = _client(repository)

    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-2/artifacts/frame"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/evaluations/evaluation-2/cases/case-1/artifacts/frame"
        ).status_code
        == 404
    )


def test_resource_ids_are_bounded_ascii_and_canonicalized_before_lookup():
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "video"): _NamedBytes(
                b"\x89PNG\r\n\x1a\nbody", name="ignored"
            )
        }
    )
    client = _client(repository)

    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/%2576ideo"
        ).status_code
        == 200
    )
    assert repository.calls[-1] == ("evaluation-1", "case-1", "video")
    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/%E9%9B%AA"
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/" + "a" * 129
        ).status_code
        == 422
    )


def test_artifact_rejects_unsafe_open_and_unknown_mime():
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "unsafe"): OSError("unsafe"),
            ("evaluation-1", "case-1", "unknown"): _NamedBytes(
                b"not media", name="payload.bin"
            ),
        }
    )
    client = _client(repository)

    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/unsafe"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/evaluations/evaluation-1/cases/case-1/artifacts/unknown"
        ).status_code
        == 415
    )


def test_declared_utf8_log_supports_bounded_range_preview():
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "console-log"): _NamedBytes(
                b"INFO fixture log ready\n", name="console.log"
            )
        },
        artifact_kinds={"console-log": "console_log"},
    )

    response = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/console-log",
        headers={"Range": "bytes=0-7"},
    )

    assert response.status_code == 206
    assert response.content == b"INFO fix"
    assert response.headers["content-type"].startswith("text/plain")


def test_artifact_rejects_hardlinked_file(tmp_path):
    original = tmp_path / "outside.png"
    linked = tmp_path / "inside.png"
    original.write_bytes(b"\x89PNG\r\n\x1a\nbody")
    os.link(original, linked)
    repository = _ArtifactRepository(
        {("evaluation-1", "case-1", "frame"): linked.open("rb")}
    )

    response = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/frame"
    )

    assert response.status_code == 404


def test_single_range_streams_from_the_open_fd_and_invalid_multi_range_is_416():
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "video"): lambda: _NamedBytes(
                b"\x00\x00\x00\x18ftypisom-data", name="video.mp4"
            ),
        }
    )

    response = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/video",
        headers={"Range": "bytes=4-7"},
    )

    assert response.status_code == 206
    assert response.content == b"ftyp"
    assert response.headers["content-range"] == "bytes 4-7/17"
    assert response.headers["content-length"] == "4"
    assert response.headers["content-disposition"] == 'inline; filename="video.mp4"'
    invalid = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/video",
        headers={"Range": "bytes=0-1,4-5"},
    )
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == "bytes */17"
    suffix = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/video",
        headers={"Range": "bytes=-4"},
    )
    assert suffix.status_code == 206
    assert suffix.content == b"data"
    opened = _client(repository).get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/video",
        headers={"Range": "bytes=8-"},
    )
    assert opened.content == b"isom-data"
    head = _client(repository).head(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/video",
        headers={"Range": "bytes=0-3"},
    )
    assert head.status_code == 206
    assert head.content == b""
    assert head.headers["content-range"] == "bytes 0-3/17"


def test_legacy_health_download_get_head_etag_and_range_use_sanitized_bytes(tmp_path):
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "id-alias",
                "kind": "health",
                "relative_path": "health.jsonl",
            },
            {
                "artifact_id": "health",
                "kind": "kind-alias",
                "relative_path": "health.jsonl",
            },
            {
                "artifact_id": "both-alias",
                "kind": "kind-alias",
                "relative_path": "health.jsonl",
            },
            {
                "artifact_id": "duplicate-alias",
                "kind": "other-kind",
                "relative_path": "health.jsonl",
            },
        ],
    )
    secret = "/Users/alice/private/model-key"
    source = attempt / "health.jsonl"
    for recorded_at in ("2026-08-07T00:00:00+00:00", "2026-08-07T00:00:01+00:00"):
        _append_jsonl(
            source,
            {
                "evaluation_id": manifest.evaluation_id,
                "case_id": case.case_id,
                "attempt_id": result.attempt_id,
                "recorded_at": recorded_at,
                "issue": f"legacy error: {secret}",
            },
        )
    original = source.read_bytes()
    client = _client(EvaluationRepository((root.parent,)))
    base = f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}/artifacts"

    for artifact_id in ("id-alias", "health", "both-alias", "duplicate-alias"):
        url = f"{base}/{artifact_id}"
        response = client.get(url)
        head = client.head(url)
        ranged = client.get(url, headers={"Range": "bytes=0-15"})
        cached = client.get(url, headers={"If-None-Match": response.headers["etag"]})

        assert response.status_code == 200
        assert secret.encode() not in response.content
        assert b"recording artifact issue" in response.content
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["etag"] == response.headers["etag"]
        assert head.headers["content-length"] == str(len(response.content))
        assert ranged.status_code == 206
        assert ranged.content == response.content[:16]
        assert ranged.headers["content-range"] == f"bytes 0-15/{len(response.content)}"
        assert cached.status_code == 304
    assert source.read_bytes() == original


def test_malformed_health_alias_download_fails_closed_for_all_methods(tmp_path):
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "health-alias",
                "kind": "not-health",
                "relative_path": "health.jsonl",
            }
        ],
    )
    source = attempt / "health.jsonl"
    source.write_bytes(b'{"issue":"/Users/alice/private/model-key"}\n')
    original = source.read_bytes()
    client = _client(EvaluationRepository((root.parent,)))
    url = f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}/artifacts/health-alias"

    assert client.get(url).status_code == 404
    assert client.head(url).status_code == 404
    assert client.get(url, headers={"Range": "bytes=0-3"}).status_code == 404
    assert source.read_bytes() == original


@pytest.mark.parametrize("malformed", [False, True])
def test_filesystem_case_alias_uses_health_fd_identity(tmp_path, malformed):
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "case-alias",
                "kind": "ordinary-label",
                "relative_path": "HEALTH.JSONL",
            }
        ],
    )
    source = attempt / "health.jsonl"
    secret = "/Users/alice/private/model-key"
    if malformed:
        source.write_bytes(json.dumps({"issue": secret}).encode() + b"\n")
    else:
        _append_jsonl(
            source,
            {
                "evaluation_id": manifest.evaluation_id,
                "case_id": case.case_id,
                "attempt_id": result.attempt_id,
                "recorded_at": "2026-08-07T00:00:00+00:00",
                "issue": secret,
            },
        )
    alias = attempt / "HEALTH.JSONL"
    try:
        aliases_same_file = (
            alias.stat().st_dev,
            alias.stat().st_ino,
        ) == (source.stat().st_dev, source.stat().st_ino)
    except FileNotFoundError:
        aliases_same_file = False
    if not aliases_same_file:
        pytest.skip("filesystem is case-sensitive")
    original = source.read_bytes()
    repository = EvaluationRepository((root.parent,))
    client = _client(repository)
    url = f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}/artifacts/case-alias"

    if malformed:
        with pytest.raises(OSError, match="health journal unavailable"):
            repository.open_artifact(manifest.evaluation_id, case.case_id, "case-alias")
        assert client.get(url).status_code == 404
        assert client.head(url).status_code == 404
        assert client.get(url, headers={"Range": "bytes=0-3"}).status_code == 404
    else:
        with repository.open_artifact(
            manifest.evaluation_id, case.case_id, "case-alias"
        ) as stream:
            sanitized = stream.read()
        response = client.get(url)
        head = client.head(url)
        ranged = client.get(url, headers={"Range": "bytes=0-15"})
        assert secret.encode() not in sanitized + response.content
        assert response.status_code == head.status_code == 200
        assert head.headers["etag"] == response.headers["etag"]
        assert ranged.status_code == 206
        assert ranged.content == response.content[:16]
    assert source.read_bytes() == original


def test_unicode_normalized_root_alias_keeps_health_sanitization_when_supported(
    tmp_path,
):
    composed = tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    decomposed = tmp_path / "cafe\N{COMBINING ACUTE ACCENT}"
    composed.mkdir()
    try:
        aliases_same_directory = (
            decomposed.stat().st_dev,
            decomposed.stat().st_ino,
        ) == (composed.stat().st_dev, composed.stat().st_ino)
    except FileNotFoundError:
        aliases_same_directory = False
    if not aliases_same_directory:
        pytest.skip("filesystem does not normalize Unicode path aliases")

    root = composed / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "unicode-root-alias",
                "kind": "alias",
                "relative_path": "health.jsonl",
            }
        ],
    )
    source = attempt / "health.jsonl"
    secret = "/Users/alice/private/model-key"
    _append_jsonl(
        source,
        {
            "evaluation_id": manifest.evaluation_id,
            "case_id": case.case_id,
            "attempt_id": result.attempt_id,
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": secret,
        },
    )
    repository = EvaluationRepository((decomposed / "evaluations",))

    with repository.open_artifact(
        manifest.evaluation_id, case.case_id, "unicode-root-alias"
    ) as stream:
        assert secret.encode() not in stream.read()


def test_health_labels_do_not_sanitize_an_ordinary_file(tmp_path):
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "health",
                "kind": "health",
                "relative_path": "notes.json",
            }
        ],
    )
    source = attempt / "notes.json"
    source.write_bytes(b'{"note":"ordinary"}')
    _append_jsonl(
        attempt / "health.jsonl",
        {
            "evaluation_id": manifest.evaluation_id,
            "case_id": case.case_id,
            "attempt_id": result.attempt_id,
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": "/Users/alice/private/model-key",
        },
    )
    repository = EvaluationRepository((root.parent,))

    with repository.open_artifact(
        manifest.evaluation_id, case.case_id, "health"
    ) as stream:
        assert stream.read() == source.read_bytes()


def test_generic_webm_is_inline_but_structured_data_remains_attachment():
    repository = _ArtifactRepository(
        {
            ("evaluation-1", "case-1", "webm"): lambda: _NamedBytes(
                b"\x1aE\xdf\xa3video", name="ignored.bin"
            ),
            ("evaluation-1", "case-1", "data"): lambda: _NamedBytes(
                b'{"ok":true}', name="ignored.bin"
            ),
        }
    )
    client = _client(repository)

    video = client.get("/api/evaluations/evaluation-1/cases/case-1/artifacts/webm")
    data = client.get("/api/evaluations/evaluation-1/cases/case-1/artifacts/data")

    assert video.headers["content-type"] == "video/webm"
    assert video.headers["content-disposition"] == 'inline; filename="video.webm"'
    assert data.headers["content-type"] == "application/json"
    assert data.headers["content-disposition"] == 'attachment; filename="artifact.bin"'


def test_head_if_none_match_and_attachment_headers_close_every_fd():
    streams = []

    def opened():
        stream = _NamedBytes(b"\x89PNG\r\n\x1a\npayload", name="misleading.exe")
        streams.append(stream)
        return stream

    repository = _ArtifactRepository({("evaluation-1", "case-1", "download"): opened})
    client = _client(repository)

    head = client.head("/api/evaluations/evaluation-1/cases/case-1/artifacts/download")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == "15"
    assert head.headers["content-type"] == "image/png"
    assert head.headers["content-disposition"] == 'attachment; filename="artifact.bin"'
    assert streams[-1].closed

    cached = client.get(
        "/api/evaluations/evaluation-1/cases/case-1/artifacts/download",
        headers={"If-None-Match": head.headers["etag"]},
    )
    assert cached.status_code == 304
    assert cached.content == b""
    assert streams[-1].closed


def test_camera_latest_is_uncached_and_retained_frame_is_immutable():
    camera_id = "a" * 61
    resource_id = recording_media.camera_storage_component(camera_id)
    boundary_camera_id = "b" * 512
    boundary_resource_id = recording_media.camera_storage_component(boundary_camera_id)
    repository = _ArtifactRepository(
        {
            (
                "evaluation-1",
                "case-1",
                f"live:{resource_id}",
            ): _NamedBytes(b"\x89PNG\r\n\x1a\nlive", name="ignored.bin"),
            (
                "evaluation-1",
                "case-1",
                f"frame:{resource_id}:3",
            ): _NamedBytes(b"\x89PNG\r\n\x1a\nretained", name="ignored.bin"),
            (
                "evaluation-1",
                "case-1",
                f"live:{boundary_resource_id}",
            ): _NamedBytes(b"\x89PNG\r\n\x1a\nboundary", name="ignored.bin"),
        }
    )
    client = _client(repository)

    latest = client.get(
        f"/api/evaluations/evaluation-1/cases/case-1/cameras/{resource_id}/latest"
    )
    retained = client.get(
        f"/api/evaluations/evaluation-1/cases/case-1/cameras/{resource_id}/frames/3"
    )
    boundary = client.get(
        f"/api/evaluations/evaluation-1/cases/case-1/cameras/{boundary_resource_id}/latest"
    )

    assert latest.status_code == retained.status_code == boundary.status_code == 200
    assert latest.headers["cache-control"] == "no-store"
    assert "immutable" in retained.headers["cache-control"]
    assert retained.headers["etag"].startswith('"')
    assert latest.headers["content-type"] == "image/png"
    assert repository.calls == [
        ("live", "evaluation-1", "case-1", resource_id),
        ("frame", "evaluation-1", "case-1", resource_id, 3),
        ("live", "evaluation-1", "case-1", boundary_resource_id),
    ]


def test_tool_image_route_is_attempt_scoped_and_immutable():
    payload = b"\x89PNG\r\n\x1a\nretained-tool-overlay"
    repository = _ArtifactRepository(
        {
            (
                "evaluation-1",
                "case-1",
                "tool-image:3",
            ): _NamedBytes(payload, name="ignored.bin"),
        }
    )
    client = _client(repository)

    response = client.get(
        "/api/evaluations/evaluation-1/cases/case-1/tool-results/3/image",
        params={"attempt_id": "attempt-old"},
    )

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]
    assert repository.calls == [
        ("tool-image", "evaluation-1", "case-1", 3, "attempt-old")
    ]


def test_tool_image_route_maps_missing_and_invalid_evidence():
    repository = _ArtifactRepository(
        {
            (
                "evaluation-1",
                "case-1",
                "tool-image:4",
            ): ValueError("tool result reference is not an image"),
        }
    )
    client = _client(repository)

    assert client.get(
        "/api/evaluations/evaluation-1/cases/case-1/tool-results/3/image"
    ).status_code == 404
    assert client.get(
        "/api/evaluations/evaluation-1/cases/case-1/tool-results/4/image"
    ).status_code == 422


def test_camera_byte_responses_preserve_head_etag_and_single_range_semantics():
    resource_id = f"camera-sha256-{'a' * 64}"
    payload = b"\x89PNG\r\n\x1a\npayload"
    repository = _ArtifactRepository(
        {("evaluation-1", "case-1", f"live:{resource_id}"): lambda: io.BytesIO(payload)}
    )
    client = _client(repository)
    path = f"/api/evaluations/evaluation-1/cases/case-1/cameras/{resource_id}/latest"

    head = client.head(path)
    ranged = client.get(path, headers={"Range": "bytes=4-7"})
    cached = client.get(path, headers={"If-None-Match": head.headers["etag"]})
    invalid = client.get(path, headers={"Range": "bytes=0-1,4-5"})

    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == str(len(payload))
    assert ranged.status_code == 206
    assert ranged.content == payload[4:8]
    assert ranged.headers["content-range"] == f"bytes 4-7/{len(payload)}"
    assert cached.status_code == 304
    assert cached.content == b""
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == f"bytes */{len(payload)}"


def test_camera_byte_reader_preserves_unsupported_media_status() -> None:
    resource_id = f"camera-sha256-{'a' * 64}"
    repository = EvaluationRepository(())
    repository.open_live_frame = lambda *args, **kwargs: io.BytesIO(b"not-media")

    response = _client(repository).get(
        f"/api/evaluations/evaluation-1/cases/case-1/cameras/{resource_id}/latest"
    )

    assert response.status_code == 415


@pytest.mark.parametrize(
    "resource_id",
    [
        "camera-sha256-" + "A" * 64,
        "camera-sha256-" + "a" * 63,
        "camera-sha256-" + "g" * 64,
        "camera-sha256-",
        "camera-deadbeef",
    ],
)
def test_camera_route_rejects_noncanonical_resource_id(resource_id):
    repository = _ArtifactRepository({})

    response = _client(repository).get(
        f"/api/evaluations/evaluation-1/cases/case-1/cameras/{resource_id}/latest"
    )

    assert response.status_code == 422
    assert repository.calls == []


def test_real_case_recorder_camera_frames_flow_through_repository_and_api(tmp_path):
    evaluation_root = tmp_path / "evaluations/run"
    case = EpisodeCase(
        ordinal=0,
        environment="robocasa",
        preset_name="preset",
        task_name="task",
        task_index=0,
        instruction="do it",
        seed=0,
        episode_index=0,
        horizon=10,
        checkpoint_repo_id="repo/model",
        checkpoint_revision="a" * 40,
        binding_name="binding",
    )
    manifest = EvaluationManifest(
        evaluation_id="evaluation-camera-api",
        rpent_commit="b" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "robocasa"},
        simulator={"name": "robocasa"},
        policy={"name": "test"},
        task_selector="0",
        cases=(case,),
        schedule={"episodes_per_task": 1},
        contracts={},
        runtime={"evaluation_root": str(evaluation_root)},
        recording={"schema_version": 1, "mode": "all", "camera_ids": []},
        qualification={"require_complete_artifacts": False},
        created_at="2026-08-07T00:00:00+00:00",
    )
    manifest.write_new(evaluation_root / "evaluation_manifest.json")
    attempt_id = "attempt-camera-api"
    register_attempt(
        manifest,
        case,
        attempt_id,
        "2026-08-07T00:00:00+00:00",
    )
    attempt = evaluation_root / "cases" / case.case_id / "attempts" / attempt_id

    native_camera_id = "视" * 170 + "ab"

    class Display:
        descriptors = (
            CameraDescriptor(native_camera_id, "Agent view", "primary", 2, 2, 3, 20.0),
        )

        def frame(self, camera_id, observation):
            return png_bytes(observation[camera_id])

    recorder = CaseRecorder(
        root=attempt,
        evaluation_id=manifest.evaluation_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        episode_id="episode-camera-api",
        display=Display(),
        profile=RecordingProfile("all"),
    )
    repository = _indexed_repository(evaluation_root)
    client = _client(repository)
    base = f"/api/evaluations/{manifest.evaluation_id}/cases/{case.case_id}"

    assert (attempt / "recording.json").is_file()
    active = client.get(base)
    assert active.status_code == 200
    assert active.json()["recording_profile"] == {
        "mode": "all",
        "camera_ids": [],
    }
    assert active.json()["cameras"] == [
        {
            "resource_id": recording_media.camera_storage_component(native_camera_id),
            "camera_id": native_camera_id,
            "label": "Agent view",
            "role": "primary",
            "width": 2,
            "height": 2,
            "channels": 3,
            "fps": 20.0,
            "media_type": "rgb",
        }
    ]
    assert active.json()["retained_camera_ids"] == [native_camera_id]

    recorder.record_transition(
        action=np.zeros(1),
        result=StepResult(
            episode_id="episode-camera-api",
            step_id=1,
            observation={native_camera_id: np.ones((2, 2, 3), dtype=np.uint8)},
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
        ),
        timestamp_s=0.1,
        operation_id=None,
        execution_id=None,
    )
    resource_id = active.json()["cameras"][0]["resource_id"]
    live = client.get(f"{base}/cameras/{resource_id}/latest")
    assert live.status_code == 200
    assert live.content.startswith(b"\x89PNG")
    forged = "camera-sha256-" + "0" * 64
    assert client.get(f"{base}/cameras/{forged}/latest").status_code == 404

    result = EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        status="terminal_failure",
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=os.getpid(),
        worker_exit_code=0,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at="2026-08-07T00:00:01+00:00",
        artifact_dir=str(attempt),
        simulator_success=False,
        contract_success=False,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )
    artifact_manifest = recorder.finalize(result)
    declared_video = next(
        artifact
        for artifact in artifact_manifest["artifacts"]
        if artifact["kind"] == f"camera_{native_camera_id}_video"
    )
    assert re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}", declared_video["artifact_id"]
    )
    (attempt / "recording.json").write_text(
        json.dumps({"evaluation_id": "stale-discovery"})
    )
    (evaluation_root / "results.jsonl").write_text(
        json.dumps(result.to_payload()) + "\n"
    )

    finalized = client.get(base)
    assert finalized.status_code == 200
    assert finalized.json()["recording_profile"]["mode"] == "all"
    assert finalized.json()["cameras"][0]["camera_id"] == native_camera_id
    assert finalized.json()["cameras"][0]["resource_id"] == resource_id
    assert finalized.json()["retained_camera_ids"] == [native_camera_id]
    assert finalized.json()["playback_artifact_id"] == declared_video["artifact_id"]

    video = client.get(
        f"{base}/artifacts/{declared_video['artifact_id']}",
        params={"attempt_id": attempt_id},
    )
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"

    legacy_manifest = json.loads((attempt / "artifacts.json").read_text())
    legacy_video = next(
        artifact
        for artifact in legacy_manifest["artifacts"]
        if artifact["kind"] == f"camera_{native_camera_id}_video"
    )
    legacy_video.pop("artifact_id")
    recording_media.atomic_write_json(attempt / "artifacts.json", legacy_manifest)
    legacy_client = _client(_indexed_repository(evaluation_root))
    legacy_detail = legacy_client.get(base)
    legacy_playback_id = legacy_detail.json()["playback_artifact_id"]
    assert re.fullmatch(r"artifact-sha256-[0-9a-f]{64}", legacy_playback_id)
    assert (
        legacy_client.get(
            f"{base}/artifacts/{legacy_playback_id}",
            params={"attempt_id": attempt_id},
        ).status_code
        == 200
    )

    retained = client.get(f"{base}/cameras/{resource_id}/frames/0")
    assert retained.status_code == 200
    assert retained.content == live.content
