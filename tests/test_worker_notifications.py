from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from test_episode_worker_recording import _Episode, _manifest

import rpent.evaluation.worker as evaluation_worker
import rpent.recording.notifications as notifications
from rpent.dashboard.attachment import (
    DashboardLease,
    LifecycleHint,
    LifecycleHintPublisher,
)
from rpent.evaluation.worker import PendingCaseRecorder, _notification_sink
from rpent.recording.contracts import CameraDescriptor, EventEnvelope
from rpent.recording.display import png_bytes
from rpent.recording.notifications import HttpNotificationSink, notification_sink


def _event() -> EventEnvelope:
    return EventEnvelope(
        schema_version=1,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        sequence=1,
        recorded_at="2026-08-07T00:00:00+00:00",
        event_type="configured",
        step_id=None,
        operation_id=None,
        execution_id=None,
        payload={"ready": True},
    )


def test_descriptor_notifications_bypass_proxy_and_reject_hostname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_requests: list[tuple[str, str | None]] = []
    proxy_requests: list[tuple[str, str | None]] = []
    delivered = threading.Event()

    def handler_for(records: list[tuple[str, str | None]]):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                records.append(
                    (self.path, self.headers.get("X-RPent-Notification-Token"))
                )
                if len(target_requests) + len(proxy_requests) == 2:
                    delivered.set()
                self.send_response(204)
                self.end_headers()

            def log_message(self, _format: str, *_args) -> None:
                pass

        return Handler

    target = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(target_requests)
    )
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(proxy_requests))
    servers = (target, proxy)
    threads = tuple(
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in servers
    )
    for thread in threads:
        thread.start()
    proxy_origin = f"http://127.0.0.1:{proxy.server_port}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, proxy_origin)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")

    lease = DashboardLease.acquire(tmp_path)
    publisher = None
    sink = None
    try:
        lease.publish(
            f"http://127.0.0.1:{target.server_port}", "never-proxy-secret"
        )
        descriptor = tmp_path / ".rpent-dashboard.json"
        publisher = LifecycleHintPublisher(
            descriptor,
            expected_runs_root=tmp_path,
        )
        sink = notification_sink(
            descriptor_path=descriptor,
            expected_runs_root=tmp_path,
        )

        assert publisher.publish(
            LifecycleHint(
                1,
                "evaluation_persisted",
                "evaluation-1",
                "manifest-1",
            )
        )
        assert sink(_event()) is True
        assert delivered.wait(2)

        assert proxy_requests == []
        assert set(target_requests) == {
            ("/api/internal/lifecycle", "never-proxy-secret"),
            ("/api/internal/events", "never-proxy-secret"),
        }
        with pytest.raises(ValueError, match="loopback"):
            lease.publish(
                f"http://localhost:{target.server_port}", "never-proxy-secret"
            )
    finally:
        if sink is not None:
            sink.close()
        if publisher is not None:
            publisher.close()
        lease.close()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(2)


def test_http_notification_returns_after_enqueue_without_waiting_for_http() -> None:
    entered = threading.Event()
    release = threading.Event()

    async def post(*_args, **_kwargs):
        entered.set()
        await asyncio.to_thread(release.wait)
        return httpx.Response(204)

    sink = HttpNotificationSink(
        target=lambda: notifications.NotificationTarget(
            "one", "http://127.0.0.1:8000", "secret"
        ),
        post=post,
        workers=1,
        queue_capacity=1,
    )
    started = time.monotonic()

    assert sink(_event()) is True
    assert time.monotonic() - started < 0.05
    assert entered.wait(1)
    release.set()


def test_http_notification_close_stops_workers_and_rejects_new_delivery() -> None:
    delivered = threading.Event()

    async def post(url, **_kwargs):
        delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    sink = HttpNotificationSink(
        "http://127.0.0.1:8000",
        post=post,
        workers=2,
    )
    assert sink(_event()) is True
    assert delivered.wait(1)

    sink.close()
    sink.close()

    assert all(worker.is_alive() is False for worker in sink._workers)
    assert sink(replace(_event(), sequence=2)) is False


@pytest.mark.parametrize("failure", (None, KeyboardInterrupt("worker interrupted")))
def test_cli_harness_episode_closes_its_notification_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None,
) -> None:
    class OwnedSink:
        def __init__(self) -> None:
            self.close_calls = 0

        def __call__(self, _event: EventEnvelope) -> bool:
            return True

        def close(self) -> None:
            self.close_calls += 1

    sink = OwnedSink()
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    artifact_dir = tmp_path / "attempt-owned-sink"
    artifact_dir.mkdir()

    def run_cli(_argv, *, worker_recording) -> int:
        assert worker_recording.notify is sink
        if failure is not None:
            raise failure
        (artifact_dir / "transcript_owned.json").write_text(
            json.dumps(
                {
                    "terminal_classification": "SUCCESS",
                    "harness": {"state": "complete"},
                    "simulator_success": True,
                    "contract_success": True,
                    "episode_metrics": {
                        "terminated": True,
                        "truncated": False,
                        "step_count": 0,
                        "action_count": 0,
                    },
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setenv("RPENT_SHARED_POLICY_ENDPOINT", "http://127.0.0.1:19000")
    monkeypatch.setattr(evaluation_worker, "_notification_sink", lambda _manifest: sink)
    monkeypatch.setattr(
        evaluation_worker,
        "_with_preparation_environment",
        lambda _manifest, function: function(),
    )
    monkeypatch.setattr(
        evaluation_worker,
        "advance_preparation_from_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluation_worker,
        "_fail_preparation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("rpent.cli.main.main", run_cli)

    if failure is None:
        report = evaluation_worker._run_cli_harness_episode(
            case, manifest, artifact_dir
        )
        assert report.terminal_classification == "SUCCESS"
    else:
        with pytest.raises(KeyboardInterrupt, match="worker interrupted"):
            evaluation_worker._run_cli_harness_episode(
                case, manifest, artifact_dir
            )
    assert sink.close_calls == 1


def test_descriptor_target_is_reloaded_for_each_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_path = tmp_path / ".rpent-dashboard.json"
    calls: list[tuple[str, dict[str, str] | None]] = []
    delivered = threading.Event()

    def write_descriptor(instance_id: str, origin: str, token: str) -> None:
        descriptor_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "instance_id": instance_id,
                    "pid": 1,
                    "origin": origin,
                    "runs_root": str(tmp_path.resolve()),
                    "notification_token": token,
                    "started_at": "2026-08-14T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        descriptor_path.chmod(0o600)

    async def post(_client, url, **kwargs):
        calls.append((url, kwargs["headers"]))
        delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    write_descriptor(
        "11111111-1111-4111-8111-111111111111",
        "http://127.0.0.1:8001",
        "secret-one",
    )
    sink = notification_sink(
        descriptor_path=descriptor_path,
        expected_runs_root=tmp_path,
    )

    assert sink(_event()) is True
    assert delivered.wait(1)
    delivered.clear()
    write_descriptor(
        "22222222-2222-4222-8222-222222222222",
        "http://127.0.0.1:8002",
        "secret-two",
    )
    assert sink(replace(_event(), sequence=2)) is True
    assert delivered.wait(1)

    assert calls == [
        (
            "http://127.0.0.1:8001/api/internal/events",
            {"X-RPent-Notification-Token": "secret-one"},
        ),
        (
            "http://127.0.0.1:8002/api/internal/events",
            {"X-RPent-Notification-Token": "secret-two"},
        ),
    ]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("offline"),
        httpx.ReadTimeout("slow"),
    ],
)
def test_http_notification_outages_are_contained(error: httpx.HTTPError) -> None:
    attempted = threading.Event()

    async def post(*_args, **_kwargs):
        attempted.set()
        raise error

    sink = HttpNotificationSink("http://127.0.0.1:8000", post=post)

    assert sink(_event()) is True
    assert attempted.wait(1)


def test_http_notification_posts_strict_envelope_with_short_timeouts() -> None:
    calls = []
    delivered = threading.Event()

    async def post(url, **kwargs):
        calls.append((url, kwargs))
        delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    event = _event()
    sink = HttpNotificationSink(
        "http://127.0.0.1:8000/", notification_token="worker-secret", post=post
    )

    assert sink(event) is True
    assert delivered.wait(1)
    assert calls[0][0] == "http://127.0.0.1:8000/api/internal/events"
    assert calls[0][1]["json"] == event.to_payload()
    assert calls[0][1]["headers"] == {"X-RPent-Notification-Token": "worker-secret"}
    timeout = calls[0][1]["timeout"]
    assert timeout.connect == pytest.approx(0.1)
    assert timeout.read == pytest.approx(0.25)


def test_http_status_failure_is_contained() -> None:
    attempted = threading.Event()

    async def post(url, **_kwargs):
        attempted.set()
        return httpx.Response(503, request=httpx.Request("POST", url))

    assert HttpNotificationSink("http://localhost:8000", post=post)(_event()) is True
    assert attempted.wait(1)


def test_cancel_suppressing_delivery_cannot_block_caller_or_second_worker() -> None:
    calls = 0
    post_started = threading.Event()
    cancelled = threading.Event()
    cancellation_finished = threading.Event()
    second_finished = threading.Event()

    async def post(url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            post_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await asyncio.sleep(0.35)
                cancellation_finished.set()
        else:
            second_finished.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    sink = HttpNotificationSink("http://localhost:8000", post=post, workers=2)
    started = time.monotonic()
    assert sink(_event()) is True
    assert time.monotonic() - started < 0.05
    assert post_started.wait(1)
    assert cancelled.wait(1)

    assert cancellation_finished.is_set() is False
    assert sink(replace(_event(), sequence=2)) is True
    assert second_finished.wait(1)


def test_unexpected_delivery_error_does_not_break_next_delivery() -> None:
    calls = 0
    first_attempted = threading.Event()
    second_delivered = threading.Event()

    async def post(url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_attempted.set()
            raise RuntimeError("client bug")
        second_delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    sink = HttpNotificationSink("http://localhost:8000", post=post, workers=1)

    assert sink(_event()) is True
    assert first_attempted.wait(1)
    assert sink(replace(_event(), sequence=2)) is True
    assert second_delivered.wait(1)


def test_two_permanent_hangs_keep_workers_and_queue_bounded() -> None:
    calls = 0
    both_started = threading.Event()

    async def post(_url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()

    sink = HttpNotificationSink(
        "http://localhost:8000", post=post, workers=2, queue_capacity=2
    )
    assert sink(_event()) is True
    assert sink(replace(_event(), sequence=2)) is True
    assert both_started.wait(1)
    assert sink(replace(_event(), sequence=3)) is True
    assert sink(replace(_event(), sequence=4)) is True
    started = time.monotonic()
    assert sink(replace(_event(), sequence=5)) is False

    assert time.monotonic() - started < 0.05
    assert calls == 2
    assert len(sink._workers) == 2
    assert all(worker.daemon for worker in sink._workers)
    assert sink._deliveries.maxsize == 2
    assert sink._deliveries.qsize() == 2


def test_timed_out_delivery_does_not_prevent_later_queued_delivery() -> None:
    calls = 0
    first_started = threading.Event()
    second_delivered = threading.Event()

    async def post(url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.35)
        else:
            second_delivered.set()
        return httpx.Response(204, request=httpx.Request("POST", url))

    sink = HttpNotificationSink(
        "http://localhost:8000", post=post, workers=1, queue_capacity=1
    )

    assert sink(_event()) is True
    assert first_started.wait(1)
    assert sink(replace(_event(), sequence=2)) is True
    assert second_delivered.wait(2)
    assert calls == 2


class _TrickleStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        for chunk in (b"a", b"b", b"c"):
            await asyncio.sleep(0.11)
            yield chunk


class _CloseTrackingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False
        self.closed_event = threading.Event()

    async def __aiter__(self):
        yield b"ok"

    async def aclose(self) -> None:
        self.closed = True
        self.closed_event.set()


def test_custom_post_response_is_always_closed() -> None:
    stream = _CloseTrackingStream()

    async def post(url, **_kwargs):
        return httpx.Response(
            204,
            request=httpx.Request("POST", url),
            stream=stream,
        )

    sink = HttpNotificationSink("http://localhost:8000", post=post)

    assert sink(_event()) is True
    assert stream.closed_event.wait(1)
    assert stream.closed is True


def test_response_body_trickle_does_not_block_enqueue() -> None:
    attempted = threading.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        attempted.set()
        return httpx.Response(200, request=request, stream=_TrickleStream())

    sink = HttpNotificationSink(
        "http://localhost:8000", transport=httpx.MockTransport(handler)
    )
    started = time.monotonic()

    assert sink(_event()) is True
    assert time.monotonic() - started < 0.05
    assert attempted.wait(1)


def test_sync_sink_rejects_use_inside_running_event_loop() -> None:
    async def invoke() -> bool:
        return HttpNotificationSink("http://localhost:8000")(_event())

    assert asyncio.run(invoke()) is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workers": 0}, "workers"),
        ({"workers": True}, "workers"),
        ({"queue_capacity": 0}, "queue_capacity"),
        ({"queue_capacity": True}, "queue_capacity"),
    ],
)
def test_http_notification_validates_fixed_resource_bounds(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HttpNotificationSink("http://localhost:8000", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://localhost:8000",
        "http://user:password@localhost:8000",
        "http://localhost:8000/base",
        "http://localhost:8000?query=yes",
        "http://localhost:8000#fragment",
        "http:///missing-host",
    ],
)
def test_http_notification_rejects_non_origin_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        HttpNotificationSink(endpoint)


def test_no_endpoint_uses_noop_sink() -> None:
    assert notification_sink(None)(_event()) is True
    assert notification_sink("  ")(_event()) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"image_bytes": "abc"},
        {"nested": [{"frame_bytes": "abc"}]},
        {"nested": {"image_data": "abc"}},
        {"data_url": "data:image/png;base64,abc"},
        {"base64": "abc"},
        {"png_bytes": "abc"},
        {"jpeg_bytes": "abc"},
    ],
)
def test_notification_sink_rejects_media_payloads_before_http(payload) -> None:
    calls = []

    async def post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("must not post")

    event = replace(_event(), payload=payload)
    sink = HttpNotificationSink(
        "http://localhost:8000", notification_token="secret", post=post
    )

    assert sink(event) is False
    assert calls == []


def test_notification_sink_rejects_envelopes_over_64_kib_before_http() -> None:
    calls = []

    async def post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("must not post")

    event = replace(_event(), payload={"message": "x" * (64 * 1024)})
    sink = HttpNotificationSink(
        "http://localhost:8000", notification_token="secret", post=post
    )

    assert sink(event) is False
    assert calls == []


class _Display:
    descriptors = (CameraDescriptor("agent", "Agent", "primary", 2, 2, 3, 20.0),)

    def frame(self, camera_id, observation):
        assert camera_id == "agent"
        return png_bytes(observation[camera_id])

    def frames(self, observation):
        return {"agent": self.frame("agent", observation)}


def test_worker_sink_uses_manifest_endpoint_then_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    assert _notification_sink(manifest)(_event()) is True

    configured = replace(
        manifest,
        runtime={**manifest.runtime, "notification_endpoint": "http://manifest:8000"},
    )
    monkeypatch.setenv("RPENT_NOTIFICATION_TOKEN", "worker-secret")
    assert _notification_sink(configured).endpoint == "http://manifest:8000"


def test_worker_sink_prefers_dashboard_descriptor_over_legacy_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    configured = replace(
        manifest,
        runtime={
            **manifest.runtime,
            "evaluation_catalog_root": str(tmp_path),
            "notification_endpoint": "http://legacy:8000",
        },
    )
    monkeypatch.setenv(
        "RPENT_DASHBOARD_DESCRIPTOR", str(tmp_path / ".rpent-dashboard.json")
    )
    monkeypatch.setenv("RPENT_NOTIFICATION_TOKEN", "legacy-secret")

    assert _notification_sink(configured).endpoint is None


def test_worker_sink_accepts_descriptor_outside_catalog_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_root = tmp_path / "catalog"
    runtime_root = tmp_path / "runtime"
    lease = DashboardLease.acquire(catalog_root, runtime_dir=runtime_root)
    descriptor = lease.publish(
        "http://127.0.0.1:43123",
        "worker-secret",
    )
    monkeypatch.setenv(
        "RPENT_DASHBOARD_DESCRIPTOR",
        str(descriptor.path),
    )

    sink = _notification_sink(_manifest(catalog_root))
    try:
        assert sink.endpoint is None
        assert sink._target().origin == "http://127.0.0.1:43123"
    finally:
        sink.close()
        lease.close()


def test_worker_notification_occurs_after_durable_acceptance_and_failure_is_safe(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    root = tmp_path / "attempt"
    observed = []

    def notify(event: EventEnvelope) -> bool:
        retained = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        observed.append((event.to_payload(), retained))
        return False

    pending = PendingCaseRecorder(manifest, case, "attempt-1", root, notify=notify)
    pending.record("startup", buffered=True)
    pending.bind(type("Bootstrap", (), {"display_adapter": _Display()})(), _Episode())

    assert observed[0][0] == observed[0][1][-1]
    assert pending.recorder.snapshot()["health"]["status"] == "complete"
