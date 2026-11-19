"""Dashboard HTTP server for the live monitor.

The FastAPI app runs on an OS-picked free port inside a daemon thread, so it can
sit alongside the agent run loop and keep serving after the run finishes, until
the process is stopped.

Routes mirror the fixed frontend contract in ``rpent/dashboard/index.html``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from rpent.dashboard.attachment import (
    DashboardAttachmentRefresher,
    DashboardLease,
    IndexFingerprintWatcher,
    validate_loopback_host,
)
from rpent.dashboard.state import State
from rpent.ui.diagnostics import install_thread_dump_signal
from rpent.ui.schemas import LaunchAcceptedView


class DashboardServer:
    """Threaded FastAPI server exposing the dashboard API for registered runs."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        runs_dir: str = "",
        runtime_dir: str | None = None,
        language: str = "en",
        evaluation_repository: Any | None = None,
        evaluation_service: Any | None = None,
        evaluation_event_hub: Any | None = None,
        event_hub_secret: bytes | None = None,
        notification_token: str | None = None,
        start_request_builder: Any | None = None,
        simulator_catalog: Any | None = None,
        sse_manager: Any | None = None,
    ) -> None:
        self.host = _validated_bind_host(host)
        self.port = int(port)
        self.runs_dir = runs_dir
        from rpent.evaluation.result_acceptance import ResultAcceptance
        from rpent.ui.app import build_start_request
        from rpent.ui.case_runtime import CaseRuntimeStore
        from rpent.ui.event_hub import EvaluationEventHub
        from rpent.ui.repository import EvaluationRepository
        from rpent.ui.service import EvaluationService
        from rpent.ui.sse import SSEManager
        from rpent.utils.config import get_repo_root

        evaluation_roots = (
            (Path(runs_dir),) if runs_dir else (get_repo_root() / "logs",)
        )
        self._runs_root = Path(evaluation_roots[0]).resolve()
        self._runtime_dir = Path(runtime_dir).resolve() if runtime_dir else None
        self._descriptor_path = (
            self._runtime_dir or self._runs_root
        ) / ".rpent-dashboard.json"
        self._evaluation_repository = (
            evaluation_repository
            if evaluation_repository is not None
            else EvaluationRepository(evaluation_roots)
        )
        self._runtime_enabled = isinstance(
            self._evaluation_repository, EvaluationRepository
        )
        self._case_runtime_store = CaseRuntimeStore()
        if self._runtime_enabled:
            self._evaluation_repository.bind_runtime_store(
                self._case_runtime_store
            )
        self._owns_evaluation_service = evaluation_service is None
        self._evaluation_service = (
            evaluation_service
            if evaluation_service is not None
            else EvaluationService(
                on_persisted=lambda context: (
                    self._evaluation_repository.register_evaluation(
                        context.root,
                        context.manifest,
                        execution_state="running",
                    )
                ),
                on_finished=lambda context: (
                    self._evaluation_repository.update_evaluation_list_item(
                        context.manifest.evaluation_id
                    )
                ),
                attempt_registered=lambda registration, manifest, case: (
                    self._case_runtime_store.start_attempt(
                        registration, manifest, case
                    )
                ),
                result_acceptance_factory=lambda manifest, store: ResultAcceptance(
                    manifest=manifest,
                    store=store,
                    runtime_store=self._case_runtime_store,
                    assembler=lambda result: (
                        self._evaluation_repository.reconstruct_case_detail(
                            result.evaluation_id,
                            result.case_id,
                            attempt_id=result.attempt_id,
                        )
                    ),
                    operation_reconstructor=lambda result: (
                        self._evaluation_repository.reconstruct_operations(
                            result.evaluation_id,
                            result.case_id,
                            attempt_id=result.attempt_id,
                        )
                    ),
                    accepted_observer=self._evaluation_repository.note_accepted_result,
                    update_publisher=lambda update: (
                        self._evaluation_event_hub.publish_runtime_update(update)
                    ),
                ),
            )
        )
        self._evaluation_event_hub = (
            evaluation_event_hub
            if evaluation_event_hub is not None
            else EvaluationEventHub(
                secret=(
                    event_hub_secret
                    if event_hub_secret is not None
                    else secrets.token_bytes(32)
                )
            )
        )
        if isinstance(self._evaluation_repository, EvaluationRepository) and isinstance(
            self._evaluation_event_hub, EvaluationEventHub
        ):
            self._case_runtime_store.configure_recovery(
                journal_reader=self._evaluation_repository.read_event_journal,
                high_water_reader=self._evaluation_repository.read_event_high_water,
                update_publisher=self._evaluation_event_hub.publish_runtime_update,
                heartbeat_interval=1.0,
            )
            self._case_runtime_store.configure_snapshots(
                loader=self._evaluation_repository.load_terminal_case_snapshot,
                reconstructor=self._evaluation_repository.reconstruct_case_detail,
                persister=self._evaluation_repository.persist_terminal_case_snapshot,
            )
            self._case_runtime_store.configure_case_decorator(
                self._evaluation_repository.decorate_case_detail
            )
            self._case_runtime_store.configure_startup(
                registration_reader=(
                    self._evaluation_repository.startup_runtime_registrations
                ),
                recording_loader=(
                    self._evaluation_repository.load_recording_declaration
                ),
            )
            self._case_runtime_store.startup_recover()
        self._dashboard_attachment_refresher = (
            DashboardAttachmentRefresher(
                self._evaluation_repository,
                self._case_runtime_store,
            )
            if self._runtime_enabled
            else None
        )
        self._attachment_watcher = (
            IndexFingerprintWatcher(self._dashboard_attachment_refresher)
            if self._dashboard_attachment_refresher is not None
            else _NoopIndexFingerprintWatcher()
        )
        self._owns_sse_manager = sse_manager is None
        self._sse_manager = SSEManager() if sse_manager is None else sse_manager
        self.notification_token = (
            notification_token
            if notification_token is not None
            else secrets.token_urlsafe(32)
        )
        self._start_request_builder = start_request_builder or (
            lambda payload, _origin: build_start_request(
                payload,
                self._descriptor_path,
                output_root=evaluation_roots[0],
            )
        )
        self._simulator_catalog = simulator_catalog
        self._index_file = Path(__file__).parent / (
            "index.zh-cn.html" if language == "zh-cn" else "index.html"
        )
        self._index_csp = _content_security_policy(self._index_file)
        self._runs: dict[str, State] = {}
        self._start_stop_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_state = "new"
        self._owned_read_gates_closed = False
        self._sse_manager_closed = False
        self._evaluation_service_closed = False
        self._case_runtime_closed = False
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._dashboard_lease: DashboardLease | None = None
        self.origin: str | None = None
        self.browser_url: str | None = None

        # -- legacy launcher route state -----------------------------------
        self._launch_enabled = False
        self._launch_defaults: dict[str, Any] = {}
        self._launch_config: dict[str, Any] | None = None
        self._launch_event = threading.Event()

    def register(self, run: State) -> None:
        self._runs[run.run_id] = run
        if not self.runs_dir:
            self.runs_dir = str(run.output_dir.parent)

    def start(self) -> str:
        """Launch uvicorn in a daemon thread; return the base URL once serving."""
        with self._start_stop_lock:
            return self._start()

    def wait(self) -> None:
        """Block until the started uvicorn thread exits."""
        thread = self._thread
        if thread is None:
            raise RuntimeError("dashboard server is not started")
        while thread.is_alive():
            thread.join(timeout=0.5)

    def _start(self) -> str:
        with self._lifecycle_lock:
            if self._lifecycle_state == "stopped":
                raise RuntimeError("dashboard server is stopped")
            if self._lifecycle_state != "new" or self._dashboard_lease is not None:
                raise RuntimeError("dashboard server is already started")
            self._lifecycle_state = "starting"
        try:
            lease = DashboardLease.acquire(
                self._runs_root,
                runtime_dir=self._runtime_dir,
            )
        except BaseException:
            with self._lifecycle_lock:
                self._lifecycle_state = "new"
            raise
        with self._lifecycle_lock:
            self._dashboard_lease = lease
        try:
            port = self.port or _free_port(self.host)
            config = uvicorn.Config(
                self._app, host=self.host, port=port, log_level="warning"
            )
            self._server = uvicorn.Server(config)
            install_thread_dump_signal()
            self._thread = threading.Thread(target=self._server.run, daemon=True)
            self._thread.start()

            t0 = time.time()
            while not self._server.started and time.time() - t0 < 10:
                time.sleep(0.05)
            if not self._server.started:
                raise RuntimeError(
                    f"dashboard server did not start on {self.host}:{port}"
                )
            self.port = port
            origin_host = f"[{self.host}]" if ":" in self.host else self.host
            origin = f"http://{origin_host}:{port}"
            self._app.state.control_plane_origin = origin
            self.origin = origin
            self.browser_url = f"{origin}/run"
            assert self._dashboard_lease is not None
            self._dashboard_lease.publish(origin, self.notification_token)
            self._attachment_watcher.start()
            with self._lifecycle_lock:
                self._lifecycle_state = "running"
            return self.browser_url
        except BaseException:
            self._stop()
            raise

    def stop(self) -> None:
        """Idempotently stop uvicorn and release only owned UI resources."""
        with self._start_stop_lock:
            self._stop()

    def _stop(self) -> None:
        self._app.state.dashboard_diagnostics.set_component(
            "lifecycle", {"shutting_down": 1}
        )
        with self._lifecycle_lock:
            if self._lifecycle_state == "stopped":
                return
            self._lifecycle_state = "stopping"
            server = self._server
            thread = self._thread
            lease = self._dashboard_lease
        first_error: BaseException | None = None
        try:
            self._attachment_watcher.close()
        except BaseException as exc:
            first_error = exc
        try:
            if server is not None:
                server.should_exit = True
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if self._owns_sse_manager and not self._sse_manager_closed:
            try:
                self._sse_manager.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._sse_manager_closed = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        watcher_thread = getattr(self._attachment_watcher, "_thread", None)
        if (
            (thread is not None and thread.is_alive())
            or (watcher_thread is not None and watcher_thread.is_alive())
        ):
            if first_error is not None:
                raise first_error
            return
        if first_error is not None:
            raise first_error
        if not self._owned_read_gates_closed:
            close_owned_read_gates = getattr(
                self._app.state, "close_owned_read_gates", None
            )
            if callable(close_owned_read_gates):
                close_owned_read_gates()
            self._owned_read_gates_closed = True
        self._close_evaluation_service()
        self._close_case_runtime()
        if lease is not None:
            lease.close()
        with self._lifecycle_lock:
            if self._dashboard_lease is lease:
                self._dashboard_lease = None
            self._lifecycle_state = "stopped"
            self._stopped = True

    def _close_evaluation_service(self) -> None:
        with self._lifecycle_lock:
            if not self._owns_evaluation_service or self._evaluation_service_closed:
                return
        self._evaluation_service.close()
        with self._lifecycle_lock:
            self._evaluation_service_closed = True

    def _close_case_runtime(self) -> None:
        with self._lifecycle_lock:
            if self._case_runtime_closed:
                return
        self._case_runtime_store.close()
        with self._lifecycle_lock:
            self._case_runtime_closed = True

    # -- routes ------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RPent dashboard")

        @app.get("/")
        def index() -> Response:
            return RedirectResponse("/run", status_code=307)

        @app.get("/legacy")
        def legacy_index() -> Response:
            return FileResponse(
                self._index_file,
                media_type="text/html",
                headers={"Content-Security-Policy": self._index_csp},
            )

        @app.get("/healthz")
        async def healthz() -> dict[str, bool]:
            return {"ok": True}

        @app.get("/readyz")
        async def readyz(request: Request) -> JSONResponse:
            state = request.app.state.dashboard_diagnostics.readiness()
            return JSONResponse(
                {"ok": state.ready, "reasons": state.reasons},
                status_code=200 if state.ready else 503,
            )

        @app.get("/api/launch/state")
        def api_launch_state() -> JSONResponse:
            return JSONResponse(
                {
                    "enabled": self._launch_enabled,
                    "pending": self._launch_enabled and not self._launch_event.is_set(),
                    "defaults": self._launch_defaults,
                }
            )

        @app.post("/api/launch/run", response_model=LaunchAcceptedView)
        def api_launch_run(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
            if not self._launch_enabled:
                return JSONResponse({"error": "launcher not armed"}, status_code=409)
            if self._launch_event.is_set():
                return JSONResponse({"error": "already launched"}, status_code=409)
            self._launch_config = dict(payload or {})
            self._launch_event.set()
            return JSONResponse({"ok": True})

        @app.get("/api/runs")
        def api_runs() -> JSONResponse:
            return JSONResponse(
                {
                    "runs_dir": self.runs_dir,
                    "runs": [r.run_info() for r in self._runs.values()],
                }
            )

        @app.get("/api/run")
        def api_run(run: str) -> JSONResponse:
            live = self._runs.get(run)
            if live is None:
                return JSONResponse({"error": "unknown run"}, status_code=404)
            return JSONResponse(live.run_detail())

        @app.get("/api/run/transcript")
        def api_transcript(run: str, since: int = 0) -> JSONResponse:
            live = self._runs.get(run)
            events = live.events_since(since) if live else []
            return JSONResponse({"events": events})

        @app.get("/api/run/frame")
        def api_frame(run: str, kind: str = "agent", t: str = "") -> Response:
            live = self._runs.get(run)
            png = live.frame(kind) if live else None
            if png is None:
                return Response(status_code=404)
            return Response(png, media_type="image/png")

        @app.get("/api/run/video")
        def api_video(run: str) -> Response:
            live = self._runs.get(run)
            if live is None or not live.has_video():
                return Response(status_code=404)
            return FileResponse(
                live.video_path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/run/action-video")
        def api_action_video(run: str, step: int) -> Response:
            live = self._runs.get(run)
            path = live.action_video_path(step) if live else None
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/stream")
        def api_stream(run: str) -> StreamingResponse:
            async def gen():
                while True:
                    live = self._runs.get(run)
                    if live is not None:
                        yield f"data: {json.dumps(live.snapshot())}\n\n"
                    else:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(gen(), media_type="text/event-stream")

        from rpent.ui.app import create_evaluation_app

        create_evaluation_app(
            app=app,
            repository=self._evaluation_repository,
            service=self._evaluation_service,
            event_hub=self._evaluation_event_hub,
            start_request_builder=self._start_request_builder,
            control_plane_origin=None,
            notification_token=self.notification_token,
            simulator_catalog=self._simulator_catalog,
            sse_manager=self._sse_manager,
            case_runtime_store=(
                self._case_runtime_store if self._runtime_enabled else None
            ),
            dashboard_attachment_refresher=(
                self._dashboard_attachment_refresher
            ),
        )
        app.router.add_event_handler("shutdown", self._close_evaluation_service)
        app.router.add_event_handler("shutdown", self._close_case_runtime)

        return app


class _NoopIndexFingerprintWatcher:
    _thread = None

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _validated_bind_host(host: object) -> str:
    return validate_loopback_host(host)


def _content_security_policy(path: Path) -> str:
    source = path.read_text(encoding="utf-8")

    def hashes(tag: str) -> str:
        values = re.findall(
            rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>",
            source,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return " ".join(
            f"'sha256-{base64.b64encode(hashlib.sha256(value.encode()).digest()).decode()}'"
            for value in values
        )

    script_hashes = hashes("script")
    style_hashes = hashes("style")
    return "; ".join(
        (
            "default-src 'none'",
            f"script-src 'self' {script_hashes}".rstrip(),
            f"style-src 'self' {style_hashes}".rstrip(),
            "img-src 'self' data: blob:",
            "media-src 'self' blob:",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "frame-ancestors 'none'",
            "form-action 'none'",
        )
    )
