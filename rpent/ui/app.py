"""FastAPI assembly for the multi-Simulator Evaluation control plane."""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import secrets
import sysconfig
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles

from rpent.cli.main import _build_argparser
from rpent.envs.base import get_env_spec
from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.diagnostics import DashboardDiagnostics, RequestObservationMiddleware
from rpent.ui.io_gate import RepositoryReadGate
from rpent.ui.notification_identity import NotificationIdentityIndex
from rpent.ui.repository import _read_media_bytes, select_playback_artifact
from rpent.ui.routes import include_evaluation_routes
from rpent.ui.routes.evaluations import StartEvaluationRequest
from rpent.ui.routes.simulators import build_simulator_catalog
from rpent.ui.schemas import CommandView
from rpent.ui.service import StartEvaluation
from rpent.ui.sse import SSEManager

StartRequestBuilder = Callable[[StartEvaluationRequest, str], StartEvaluation]
ActionReplayHandler = Callable[[str, str, str, str], CommandView]
ForceStopHandler = Callable[[str], CommandView]
_MAX_API_BODY_BYTES = 1024 * 1024
_NOTIFICATION_PATHS = frozenset(
    {
        "/api/internal/events",
        "/api/internal/lifecycle",
    }
)
logger = logging.getLogger(__name__)


def create_evaluation_app(
    *,
    repository: Any,
    service: Any,
    event_hub: Any,
    start_request_builder: StartRequestBuilder | None = None,
    action_replay_handler: ActionReplayHandler | None = None,
    force_stop_handler: ForceStopHandler | None = None,
    control_plane_origin: str | None = None,
    notification_token: str | None = None,
    simulator_catalog: Mapping[str, Mapping[str, Any]] | None = None,
    sse_manager: SSEManager | None = None,
    repository_read_gate: RepositoryReadGate | None = None,
    media_read_gate: RepositoryReadGate | None = None,
    notification_identities: NotificationIdentityIndex | None = None,
    case_runtime_store: CaseRuntimeStore | None = None,
    dashboard_attachment_refresher: Any | None = None,
    app: FastAPI | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    selected = app or FastAPI(title="RPent Evaluation UI")
    selected_notification_token = _secret(notification_token, "notification_token")
    selected.state.evaluation_repository = _CapabilityProjectionRepository(
        repository,
        service,
        action_replay=callable(action_replay_handler),
        force_stop=callable(force_stop_handler),
    )
    selected.state.evaluation_event_hub = event_hub
    selected.state.case_runtime_store = case_runtime_store
    selected.state.runtime_repository = repository
    owns_sse_manager = sse_manager is None
    selected.state.sse_manager = SSEManager() if sse_manager is None else sse_manager
    selected.state.owns_sse_manager = owns_sse_manager
    owns_repository_read_gate = repository_read_gate is None
    selected.state.repository_read_gate = (
        RepositoryReadGate() if repository_read_gate is None else repository_read_gate
    )
    selected.state.owns_repository_read_gate = owns_repository_read_gate
    owns_media_read_gate = media_read_gate is None
    selected.state.media_read_gate = (
        RepositoryReadGate(
            workers=4,
            pending=4,
            thread_name_prefix="rpent-ui-media",
        )
        if media_read_gate is None
        else media_read_gate
    )
    selected.state.owns_media_read_gate = owns_media_read_gate
    owns_notification_identities = notification_identities is None
    selected.state.notification_identities = (
        NotificationIdentityIndex(repository, selected.state.repository_read_gate)
        if notification_identities is None
        else notification_identities
    )
    selected.state.owns_notification_identities = owns_notification_identities
    selected.state.evaluation_service = _IndexingEvaluationService(
        service,
        repository,
        selected.state.notification_identities,
    )
    selected.state.start_request_builder = start_request_builder
    selected.state.action_replay_handler = _invalidating_action_replay_handler(
        action_replay_handler,
        selected.state.notification_identities,
    )
    selected.state.force_stop_handler = _invalidating_force_stop_handler(
        force_stop_handler,
        selected.state.notification_identities,
    )
    selected.state.action_replay_sources = set()
    selected.state.action_replay_lock = threading.Lock()
    selected.state.control_plane_origin = control_plane_origin
    selected.state.notification_token = selected_notification_token
    selected_refresher = (
        _NoopDashboardAttachmentRefresher()
        if dashboard_attachment_refresher is None
        else dashboard_attachment_refresher
    )
    if not callable(getattr(selected_refresher, "accept", None)):
        raise TypeError("dashboard_attachment_refresher must expose accept")
    selected.state.dashboard_attachment_refresher = selected_refresher
    selected.state.dashboard_diagnostics = DashboardDiagnostics()
    _register_component_snapshot(
        selected.state.dashboard_diagnostics,
        "sse",
        selected.state.sse_manager,
    )
    _register_component_snapshot(
        selected.state.dashboard_diagnostics,
        "repository_read_gate",
        selected.state.repository_read_gate,
    )
    _register_component_snapshot(
        selected.state.dashboard_diagnostics,
        "media_read_gate",
        selected.state.media_read_gate,
    )
    if case_runtime_store is not None:
        _register_component_snapshot(
            selected.state.dashboard_diagnostics,
            "case_runtime",
            case_runtime_store,
        )
        _register_component_snapshot(
            selected.state.dashboard_diagnostics,
            "case_repository",
            repository,
        )
    selected.state.simulator_catalog = dict(
        build_simulator_catalog() if simulator_catalog is None else simulator_catalog
    )
    include_evaluation_routes(selected)

    @selected.get("/api/internal/diagnostics", include_in_schema=False)
    async def internal_diagnostics(request: Request) -> dict[str, Any]:
        return await request.app.state.dashboard_diagnostics.snapshot()

    selected.add_middleware(
        _ApiBoundaryMiddleware,
        notification_token=selected_notification_token,
    )
    selected.add_middleware(
        RequestObservationMiddleware,
        diagnostics=selected.state.dashboard_diagnostics,
    )
    _mount_static_ui(selected, _resolve_static_dir(static_dir))
    if owns_sse_manager:
        selected.router.add_event_handler("shutdown", selected.state.sse_manager.close)

    def close_owned_read_gates() -> None:
        if owns_notification_identities:
            selected.state.notification_identities.close()
        if owns_repository_read_gate:
            selected.state.repository_read_gate.close()
        if owns_media_read_gate:
            selected.state.media_read_gate.close()

    selected.state.close_owned_read_gates = close_owned_read_gates
    selected.router.add_event_handler("shutdown", close_owned_read_gates)
    return selected


def _register_component_snapshot(
    diagnostics: DashboardDiagnostics,
    name: str,
    component: Any,
) -> None:
    snapshot = getattr(component, "snapshot", None)
    if not callable(snapshot):
        return

    def provider() -> Mapping[str, int | float | str]:
        values = snapshot()
        if isinstance(values, Mapping):
            return values
        return asdict(values)

    diagnostics.register_component(name, provider)


class _CapabilityProjectionRepository:
    """Add current UI capabilities without weakening repository authority."""

    def __init__(
        self,
        repository: Any,
        service: Any,
        *,
        action_replay: bool,
        force_stop: bool,
    ) -> None:
        self._repository = repository
        self._service = service
        self._action_replay = action_replay
        self._force_stop = force_stop

    def __getattr__(self, name: str):
        return getattr(self._repository, name)

    def get_evaluation(self, evaluation_id: str):
        detail = self._repository.get_evaluation(evaluation_id)
        if not callable(getattr(detail, "model_copy", None)):
            return detail
        allowed = ()
        preparation = getattr(detail, "preparation", None)
        if preparation is not None and preparation.status in {"failed", "cancelled"}:
            allowed = ("retry",)
        elif detail.execution_state in {
            "queued",
            "starting",
            "running",
            "finalizing",
        }:
            allowed = ("cancel", "force_stop") if self._force_stop else ("cancel",)
        elif detail.execution_state in {"cancelled", "worker_lost"}:
            allowed = ("resume",)
        if detail.qualification.accepted or detail.official_metrics.is_complete:
            allowed = ()
        return detail.model_copy(
            update={"available_commands": self._commands(evaluation_id, None, allowed)}
        )

    def get_case(
        self,
        evaluation_id: str,
        case_id: str,
        *,
        attempt_id: str | None = None,
    ):
        detail = (
            self._repository.get_case(evaluation_id, case_id)
            if attempt_id is None
            else self._repository.get_case(
                evaluation_id,
                case_id,
                attempt_id=attempt_id,
            )
        )
        if not callable(getattr(detail, "model_copy", None)):
            return detail
        playback = select_playback_artifact(
            artifacts=detail.artifacts,
            artifact_health=detail.artifact_health,
            cameras=detail.cameras,
            retained_camera_ids=detail.retained_camera_ids,
        )
        metadata = getattr(self._repository, "evaluation_metadata", None)
        rerun_allowed = (
            metadata(evaluation_id).allow_selective_case_rerun
            if callable(metadata)
            else True
        )
        replay_allowed = (
            attempt_id is None
            and self._action_replay
            and detail.execution_state
            not in {"queued", "starting", "running", "finalizing"}
            and len(_recipe_artifacts(detail)) == 1
        )
        allowed = (("rerun",) if rerun_allowed else ()) + (
            ("action_replay",) if replay_allowed else ()
        )
        return detail.model_copy(
            update={
                "available_commands": self._commands(evaluation_id, case_id, allowed),
                "playback_artifact_id": playback,
            }
        )

    def read_live_frame(self, evaluation_id: str, case_id: str, resource_id: str):
        reader = getattr(self._repository, "read_live_frame", None)
        if callable(reader):
            return reader(evaluation_id, case_id, resource_id)
        return _read_media_bytes(
            self._repository.open_live_frame(evaluation_id, case_id, resource_id)
        )

    def read_indexed_frame(
        self,
        evaluation_id: str,
        case_id: str,
        resource_id: str,
        frame_index: int,
        *,
        attempt_id: str | None = None,
    ):
        reader = getattr(self._repository, "read_indexed_frame", None)
        kwargs = {} if attempt_id is None else {"attempt_id": attempt_id}
        if callable(reader):
            return reader(
                evaluation_id,
                case_id,
                resource_id,
                frame_index,
                **kwargs,
            )
        return _read_media_bytes(
            self._repository.open_indexed_frame(
                evaluation_id,
                case_id,
                resource_id,
                frame_index,
                **kwargs,
            )
        )

    def _commands(
        self,
        evaluation_id: str,
        case_id: str | None,
        allowed: tuple[str, ...],
    ) -> tuple[str, ...]:
        projection = getattr(self._service, "available_commands", None)
        advertised = (
            tuple(projection(evaluation_id, case_id)) if callable(projection) else ()
        )
        callback_commands = set()
        if self._action_replay:
            callback_commands.add("action_replay")
        if self._force_stop:
            callback_commands.add("force_stop")
        return tuple(
            command
            for command in allowed
            if command in advertised or command in callback_commands
        )


class _IndexingEvaluationService:
    """Index successful controls and update notification identity authority."""

    def __init__(self, service: Any, repository: Any, identities: Any) -> None:
        self._service = service
        self._repository = repository
        self._identities = identities
        self._list_item_lock = threading.Lock()
        self._list_item_generations: dict[str, int] = {}

    def __getattr__(self, name: str):
        return getattr(self._service, name)

    def start(self, *args, **kwargs):
        run = self._register(self._service.start(*args, **kwargs))
        self._identities.begin_transition(run.evaluation_id)
        return run

    def cancel(self, evaluation_id: str, *args, **kwargs):
        run = self._service.cancel(evaluation_id, *args, **kwargs)
        self._identities.invalidate(evaluation_id)
        return run

    def resume(self, evaluation_id: str, *args, **kwargs):
        attempts = _resumable_attempts(self._repository.get_evaluation(evaluation_id))
        run = self._register(self._service.resume(evaluation_id, *args, **kwargs))
        self._identities.begin_transition(evaluation_id, attempts)
        return run

    def retry(self, evaluation_id: str, *args, **kwargs):
        run = self._register(self._service.retry(evaluation_id, *args, **kwargs))
        self._identities.begin_transition(run.evaluation_id)
        return run

    def rerun(self, evaluation_id: str, case_id: str, *args, **kwargs):
        run = self._register(
            self._service.rerun(evaluation_id, case_id, *args, **kwargs)
        )
        self._identities.begin_transition(run.evaluation_id)
        return run

    def _register(self, run: Any):
        register = getattr(self._repository, "register_evaluation", None)
        context_getter = getattr(type(self._service), "_get_context", None)
        with self._list_item_lock:
            generation = self._list_item_generations.get(run.evaluation_id, 0) + 1
            if callable(register) and callable(context_getter):
                context = context_getter(self._service, run.evaluation_id)
                register(context.root, context.manifest, execution_state="running")
            self._list_item_generations[run.evaluation_id] = generation
        self._attach_list_item_update(run, generation)
        return run

    def _attach_list_item_update(self, run: Any, generation: int) -> None:
        future = getattr(run, "future", None)
        update = getattr(self._repository, "update_evaluation_list_item", None)
        if future is None or not callable(getattr(future, "add_done_callback", None)):
            return
        if not callable(update):
            return

        def finished(completed) -> None:
            try:
                outcome = completed.result()
            except BaseException:
                execution_state = "terminal"
            else:
                execution_state = (
                    "cancelled"
                    if getattr(outcome, "interrupted", False)
                    else "terminal"
                )
            with self._list_item_lock:
                if self._list_item_generations.get(run.evaluation_id) != generation:
                    return
                try:
                    update(run.evaluation_id, execution_state=execution_state)
                except Exception:
                    logger.exception(
                        "failed to publish Evaluation list item evaluation_id=%s",
                        run.evaluation_id,
                    )

        future.add_done_callback(finished)


def _invalidating_force_stop_handler(handler: Any, identities: Any):
    if not callable(handler):
        return handler

    def selected(evaluation_id: str):
        result = handler(evaluation_id)
        identities.invalidate(evaluation_id)
        return result

    return selected


def _invalidating_action_replay_handler(handler: Any, identities: Any):
    if not callable(handler):
        return handler

    def selected(
        evaluation_id: str,
        case_id: str,
        attempt_id: str,
        recipe_id: str,
    ):
        result = handler(evaluation_id, case_id, attempt_id, recipe_id)
        attempts = (
            {case_id: attempt_id} if result.evaluation_id == evaluation_id else None
        )
        identities.begin_transition(result.evaluation_id, attempts)
        return result

    return selected


def _resumable_attempts(detail: Any) -> dict[str, str | None]:
    cases = tuple(getattr(detail, "cases", ()))
    manifest = getattr(detail, "manifest", {})
    schedule = manifest.get("schedule", {}) if isinstance(manifest, Mapping) else {}
    grouping = schedule.get("worker_grouping", "case")
    if grouping == "task":
        pending_tasks = {
            case.task_name
            for case in cases
            if getattr(case, "execution_state", None) != "terminal"
        }
        selected = (case for case in cases if case.task_name in pending_tasks)
    else:
        selected = (
            case
            for case in cases
            if getattr(case, "execution_state", None) != "terminal"
        )
    return {case.case_id: case.attempt_id for case in selected}


def _recipe_artifacts(detail: Any) -> tuple[Any, ...]:
    return tuple(
        artifact
        for artifact in getattr(detail, "artifacts", ())
        if artifact.kind == "action_recipe" and artifact.available
    )


class _ImmutableStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code in {200, 304}:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def _resolve_static_dir(configured: str | Path | None) -> Path | None:
    candidates = (
        (Path(configured),)
        if configured is not None
        else (
            Path(__file__).resolve().parents[2] / "ui" / "dist",
            Path(sysconfig.get_path("data")) / "share" / "rpent" / "ui" / "dist",
        )
    )
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "index.html").is_file() and (root / "assets").is_dir():
            return root
    return None


def _mount_static_ui(app: FastAPI, root: Path | None) -> None:
    if root is not None:
        app.mount(
            "/assets",
            _ImmutableStaticFiles(directory=root / "assets", follow_symlink=False),
            name="evaluation-ui-assets",
        )

    @app.api_route("/{spa_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa_fallback(spa_path: str, request: Request):
        first = spa_path.split("/", 1)[0].lower()
        if first in {
            "api",
            "assets",
            "docs",
            "internal",
            "openapi.json",
            "redoc",
        }:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if root is None:
            return PlainTextResponse(
                "RPent UI build is unavailable; run npm --prefix ui run build.",
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        return FileResponse(
            root / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )


class _ApiBoundaryMiddleware:
    """Authenticate mutations and bound POST bodies before framework parsing."""

    def __init__(self, app, *, notification_token: str) -> None:
        self.app = app
        self.notification_token = notification_token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method")
        if method == "GET" and path == "/api/internal/diagnostics":
            headers = {name.lower(): value for name, value in scope.get("headers", ())}
            presented = headers.get(b"authorization", b"")
            expected = f"Bearer {self.notification_token}".encode()
            if not hmac.compare_digest(presented, expected):
                await _asgi_error(send, 401, "authentication required")
                return
        if method != "POST":
            await self.app(scope, receive, send)
            return
        if path in _NOTIFICATION_PATHS:
            headers = {name.lower(): value for name, value in scope.get("headers", ())}
            presented = headers.get(b"x-rpent-notification-token")
            if presented is None:
                await _asgi_error(send, 401, "authentication required")
                return
            try:
                decoded = presented.decode("utf-8")
            except UnicodeDecodeError:
                decoded = ""
            if not hmac.compare_digest(decoded, self.notification_token):
                await _asgi_error(send, 403, "authentication failed")
                return
        headers = {name.lower(): value for name, value in scope.get("headers", ())}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await _asgi_error(send, 400, "invalid content-length")
                return
            if declared_size > _MAX_API_BODY_BYTES:
                await _asgi_error(send, 413, "request body too large")
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > _MAX_API_BODY_BYTES:
                await _asgi_error(send, 413, "request body too large")
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return {"type": "http.request", "body": b"", "more_body": False}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


class _NoopDashboardAttachmentRefresher:
    def accept(self, hint: Any) -> None:
        del hint

    def refresh_if_changed(self) -> None:
        pass


async def _asgi_error(send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _secret(value: str | None, name: str) -> str:
    if value is None:
        return secrets.token_urlsafe(32)
    if not isinstance(value, str) or not value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} must be a non-empty header value")
    return value


def build_start_request(
    payload: StartEvaluationRequest,
    dashboard_descriptor_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> StartEvaluation:
    """Build through the existing CLI/EnvSpec boundary without route-owned rules."""
    spec = get_env_spec(payload.simulator_id)
    if spec.launch_preset is None:
        raise ValueError(f"simulator {spec.name!r} has no launch preset")
    preset = spec.launch_preset()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    submitted = dict(payload.config)
    if payload.recording is not None:
        submitted["recording_mode"] = payload.recording.mode
        submitted["recording_camera"] = list(payload.recording.camera_ids)
    effective = preset.normalize(payload.execution_mode, submitted)
    prerequisites = preset.preflight(payload.execution_mode, effective)
    assembler_owned = {
        "reuse_inference",
        "reuse_policy_service",
        "vla_max_chunks",
    }
    runtime_values = {
        name: value for name, value in effective.items() if name not in assembler_owned
    }
    argv = ["--env", spec.name, *_config_argv(parser, runtime_values)]
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ValueError("invalid simulator launch configuration") from exc
    args.execution_mode = payload.execution_mode
    for name in assembler_owned:
        if name in effective:
            setattr(args, name, effective[name])
    if output_root is not None and args.output_dir is None:
        default_output = spec.parse_config(args).output_dir
        args.output_dir = Path(output_root) / default_output.name

    resolved_tasks = None
    if spec.name == "robocasa" and getattr(args, "evaluation_profile", None) in {
        None,
        "development",
    }:
        from robots.robocasa.tasks import resolve_robocasa_tasks_from_contract

        resolved_tasks = resolve_robocasa_tasks_from_contract(str(args.task))

    return StartEvaluation(
        args=args,
        env_spec=spec,
        argv=tuple(argv),
        resolved_tasks=resolved_tasks,
        worker_environment={
            "RPENT_DASHBOARD_DESCRIPTOR": str(
                Path(dashboard_descriptor_path).resolve()
            )
        },
        launch={
            "execution_mode": payload.execution_mode,
            "preset_version": preset.version,
            "runner": f"{spec.name}-{payload.execution_mode}-v1",
            "submitted": submitted,
            "effective": effective,
        },
        prerequisites=prerequisites,
    )


def _config_argv(parser: argparse.ArgumentParser, values: dict[str, Any]) -> list[str]:
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "env_name"}
    }
    unknown = sorted(set(values).difference(actions))
    if unknown:
        raise ValueError(f"unknown launch fields: {unknown}")
    argv: list[str] = []
    for name, value in values.items():
        action = actions[name]
        option = next(
            (item for item in action.option_strings if item.startswith("--")), None
        )
        if option is None:
            raise ValueError(f"launch field {name!r} has no public option")
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"launch field {name!r} must be boolean")
            if value:
                argv.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if type(value) is not bool:
                raise ValueError(f"launch field {name!r} must be boolean")
            if not value:
                argv.append(option)
        elif isinstance(action, argparse._AppendAction):
            if not isinstance(value, list) or not all(
                _matches_converter(action.type, item) for item in value
            ):
                raise ValueError(f"launch field {name!r} has invalid array items")
            for item in value:
                argv.extend((option, str(item)))
        else:
            if not _matches_converter(action.type, value):
                raise ValueError(f"launch field {name!r} has an invalid type")
            argv.extend((option, str(value)))
    return argv


def _matches_converter(converter: Any, value: Any) -> bool:
    if converter is int or getattr(converter, "__name__", None) in {
        "_positive_integer", "_nonnegative_integer",
    }:
        return type(value) is int
    if converter is float:
        return type(value) in {int, float}
    return isinstance(value, str)


__all__ = ["build_start_request", "create_evaluation_app"]
