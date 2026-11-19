"""Nested Case resources and safe retained-media delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import Annotated, Any, BinaryIO
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from rpent.envs.launch import LaunchValidationError
from rpent.recording.contracts import JSON_SAFE_INTEGER_MAX
from rpent.ui.case_snapshot import SnapshotRebuilding, SnapshotUnavailable
from rpent.ui.console import MAX_CONSOLE_PAGE_BYTES, iter_console_download
from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout
from rpent.ui.repository import MediaBytes
from rpent.ui.routes.evaluations import (
    _repository_read,
    _retry_request,
    _run_payload,
    _sse,
    _sse_capacity_response,
    _sse_repository_call,
)
from rpent.ui.schemas import (
    CaseDetail,
    CommandView,
    ConsolePageView,
    EventPage,
    OperationPage,
    TimelineWindow,
)
from rpent.ui.service import ConflictError, EvaluationNotFoundError
from rpent.ui.sse import (
    DEFAULT_POLL_INTERVAL,
    ReservedStreamingResponse,
    SSECapacityError,
    case_event_stream,
)

router = APIRouter(prefix="/api/evaluations/{evaluation_id}/cases", tags=["cases"])


class _MediaStreamOwner:
    """Transfer one late-opened stream to either the response or cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._close_requested = threading.Event()
        self._stream: BinaryIO | None = None
        self._active_operations = 0
        self._close_claim: object | None = None
        self._close_started = False
        self._close_complete = False

    def publish(self, stream: BinaryIO) -> bool:
        with self._lock:
            if self._close_requested.is_set():
                self._close_claim = None
                self._close_started = False
                self._close_complete = True
                return False
            self._stream = stream
            return True

    def begin_operation(self) -> bool:
        with self._lock:
            if self._close_requested.is_set():
                return False
            self._active_operations += 1
            return True

    def finish_operation(self) -> object | None:
        with self._lock:
            self._active_operations -= 1
            if (
                self._close_requested.is_set()
                and self._active_operations == 0
                and self._close_started
            ):
                return self._close_claim
            return None

    def request_close(self) -> tuple[BinaryIO | None, bool, object | None]:
        self._close_requested.set()
        with self._lock:
            if self._close_complete or self._close_claim is not None:
                return None, False, None
            if self._stream is None:
                self._close_complete = True
                return None, True, None
            if getattr(self._stream, "closed", False):
                self._close_complete = True
                return None, False, None
            claim = object()
            self._close_claim = claim
            worker_owns_close = self._active_operations > 0
            self._close_started = worker_owns_close
            return self._stream, worker_owns_close, claim

    def start_close(self, claim: object) -> bool:
        with self._lock:
            if (
                self._close_complete
                or self._close_claim is not claim
                or self._close_started
            ):
                return False
            self._close_started = True
            return True

    def claim_read_close(self) -> object | None:
        with self._lock:
            if self._close_complete or self._close_started:
                return None
            claim = object()
            self._close_claim = claim
            self._close_started = True
            return claim

    def complete_close(self, claim: object) -> None:
        with self._lock:
            if self._close_claim is claim:
                self._close_claim = None
                self._close_started = False
                self._close_complete = True

    def fail_close(self, claim: object) -> None:
        with self._lock:
            if self._close_claim is not claim:
                return
            if self._stream is not None and getattr(self._stream, "closed", False):
                self._close_complete = True
            self._close_claim = None
            self._close_started = False

    def abandon_close(self, claim: object) -> bool:
        with self._lock:
            if self._close_claim is not claim or self._close_started:
                return False
            self._close_claim = None
            return True


class _OwnedStreamingResponse(StreamingResponse):
    """Finalize one retained media owner across every ASGI response exit."""

    # asyncio holds tasks weakly; detached cleanup stays rooted only until done.
    _detached_finalizers: set[asyncio.Future[None]] = set()
    _detached_finalizers_lock = threading.Lock()

    def __init__(
        self,
        content,
        finalizer: Callable[[], Awaitable[None]],
        **kwargs,
    ) -> None:
        super().__init__(content, **kwargs)
        self._finalizer = finalizer

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        except BaseException:
            await self._run_finalizer(suppress_errors=True)
            raise
        await self._run_finalizer(suppress_errors=False)

    @classmethod
    def _retain_detached_finalizer(cls, finalizer: asyncio.Future[None]) -> None:
        if finalizer.done():
            cls._finish_detached_finalizer(finalizer)
            return
        with cls._detached_finalizers_lock:
            cls._detached_finalizers.add(finalizer)
        finalizer.add_done_callback(cls._finish_detached_finalizer)

    @classmethod
    def _finish_detached_finalizer(cls, finalizer: asyncio.Future[None]) -> None:
        try:
            if not finalizer.cancelled():
                finalizer.exception()
        finally:
            with cls._detached_finalizers_lock:
                cls._detached_finalizers.discard(finalizer)

    async def _run_finalizer(self, *, suppress_errors: bool) -> None:
        finalizer = asyncio.ensure_future(self._finalizer())
        try:
            await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if finalizer.done() and not (current and current.cancelling()):
                if not suppress_errors:
                    raise
                return
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                self._retain_detached_finalizer(finalizer)
            except BaseException:
                pass
            raise
        except BaseException:
            if not suppress_errors:
                raise


@router.get("/{case_id}", response_model=CaseDetail)
async def get_case(
    evaluation_id: str,
    case_id: str,
    request: Request,
    attempt_id: str | None = None,
):
    return await _read_case(
        request,
        evaluation_id,
        case_id,
        attempt_id=None if attempt_id is None else _safe_id(attempt_id),
    )


@router.get("/{case_id}/events", response_model=EventPage)
async def get_events(
    evaluation_id: str,
    case_id: str,
    request: Request,
    since: Annotated[int, Query(ge=0, le=JSON_SAFE_INTEGER_MAX)] = 0,
    attempt_id: str | None = None,
):
    try:
        repository = request.app.state.evaluation_repository
        selected_attempt = None if attempt_id is None else _safe_id(attempt_id)
        return await _repository_read(
            request,
            ("events", evaluation_id, case_id, selected_attempt, since),
            partial(
                repository.events_since,
                evaluation_id,
                case_id,
                since=since,
                **(
                    {} if selected_attempt is None else {"attempt_id": selected_attempt}
                ),
            ),
            timeout_s=2.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{case_id}/operations", response_model=OperationPage)
async def get_operations(
    evaluation_id: str,
    case_id: str,
    request: Request,
    attempt_id: str,
):
    try:
        selected_attempt = _safe_id(attempt_id)
        runtime_store = getattr(request.app.state, "case_runtime_store", None)
        reader = (
            request.app.state.evaluation_repository.operations
            if runtime_store is None
            else runtime_store.get_operations
        )
        return await _repository_read(
            request,
            ("operations", evaluation_id, case_id, selected_attempt),
            partial(
                reader,
                evaluation_id,
                case_id,
                *(() if runtime_store is None else (selected_attempt,)),
                **({"attempt_id": selected_attempt} if runtime_store is None else {}),
            ),
            timeout_s=2.0,
        )
    except SnapshotRebuilding as exc:
        raise HTTPException(
            status_code=503,
            detail="operation_snapshot_rebuilding",
            headers={"Retry-After": "1"},
        ) from exc
    except SnapshotUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="operation_snapshot_unavailable",
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{case_id}/console", response_model=ConsolePageView)
async def get_console(
    evaluation_id: str,
    case_id: str,
    request: Request,
    offset: Annotated[int, Query(ge=0, le=JSON_SAFE_INTEGER_MAX)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_CONSOLE_PAGE_BYTES)] = 64 * 1024,
    attempt_id: str | None = None,
):
    try:
        selected_attempt = None if attempt_id is None else _safe_id(attempt_id)
        return await _repository_read(
            request,
            ("console", evaluation_id, case_id, selected_attempt, offset, limit),
            partial(
                request.app.state.evaluation_repository.console_page,
                evaluation_id,
                case_id,
                offset=offset,
                limit=limit,
                **(
                    {} if selected_attempt is None else {"attempt_id": selected_attempt}
                ),
            ),
            timeout_s=2.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="console unavailable") from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="console unavailable") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="console output is unavailable"
        ) from exc


@router.get("/{case_id}/console/download", response_class=Response)
def download_console(
    evaluation_id: str,
    case_id: str,
    request: Request,
    attempt_id: str | None = None,
    inline: bool = False,
):
    try:
        selected_attempt = None if attempt_id is None else _safe_id(attempt_id)
        stream, size = request.app.state.evaluation_repository.open_console_download(
            evaluation_id,
            case_id,
            **({} if selected_attempt is None else {"attempt_id": selected_attempt}),
        )
    except (KeyError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="console unavailable") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="console output is unavailable"
        ) from exc
    return StreamingResponse(
        content=iter_console_download(stream, size),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": (
                'inline; filename="console.log"'
                if inline
                else 'attachment; filename="console.log"'
            ),
            "Content-Length": str(size),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{case_id}/timeline", response_model=TimelineWindow)
async def get_timeline(
    evaluation_id: str,
    case_id: str,
    request: Request,
    attempt_id: str,
    step: Annotated[int | None, Query(ge=0, le=JSON_SAFE_INTEGER_MAX)] = None,
):
    try:
        selected_attempt = _safe_id(attempt_id)
        return await _repository_read(
            request,
            ("timeline", evaluation_id, case_id, selected_attempt, step),
            partial(
                request.app.state.evaluation_repository.timeline_window,
                evaluation_id,
                case_id,
                attempt_id=selected_attempt,
                requested_step=step,
            ),
            timeout_s=2.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{case_id}/stream")
async def stream_case(
    evaluation_id: str,
    case_id: str,
    request: Request,
    since: Annotated[int | None, Query(ge=0, le=JSON_SAFE_INTEGER_MAX)] = None,
    cursor: str | None = None,
    timeout: Annotated[float | None, Query(ge=0, le=30)] = None,
):
    repository = request.app.state.evaluation_repository
    runtime_store = getattr(request.app.state, "case_runtime_store", None)
    case_reader = (
        repository.get_case if runtime_store is None else runtime_store.get_case
    )
    manager = request.app.state.sse_manager
    try:
        current = await _sse_repository_call(
            manager,
            case_reader,
            evaluation_id,
            case_id,
        )
    except SnapshotRebuilding as exc:
        raise HTTPException(
            status_code=503,
            detail="case_snapshot_rebuilding",
            headers={"Retry-After": "1"},
        ) from exc
    except SnapshotUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="case_snapshot_unavailable"
        ) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    hub = request.app.state.evaluation_event_hub
    retained = cursor or request.headers.get("last-event-id")
    position = None
    reset_attempt = False
    if retained is not None:
        try:
            position = hub.case_position(retained, evaluation_id, case_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        reset_attempt = position.attempt_id != current.attempt_id
        selected = 0 if reset_attempt else position.sequence
    else:
        selected = 0 if since is None else since
    if selected < 0:
        raise HTTPException(status_code=422, detail="since must be non-negative")
    try:
        connection = manager.reserve()
    except SSECapacityError as exc:
        raise _sse_capacity_response(exc) from exc
    if timeout is None:
        try:
            stream = case_event_stream(
                request,
                request.app.state.evaluation_repository,
                hub,
                manager,
                evaluation_id,
                case_id,
                since=since,
                cursor=retained,
                heartbeat_interval=getattr(
                    request.app.state, "sse_heartbeat_interval", 15.0
                ),
                poll_interval=getattr(
                    request.app.state, "sse_poll_interval", DEFAULT_POLL_INTERVAL
                ),
                case_reader=case_reader,
            )
            return ReservedStreamingResponse(
                stream,
                connection,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )
        except BaseException:
            connection.release()
            raise
    try:
        page = await _sse_repository_call(
            manager,
            repository.events_since,
            evaluation_id,
            case_id,
            since=selected,
        )
        if position is not None and position.attempt_id != page.attempt_id:
            reset_attempt = True
            if selected != 0:
                page = await _sse_repository_call(
                    manager,
                    repository.events_since,
                    evaluation_id,
                    case_id,
                    since=0,
                )
    except KeyError as exc:
        connection.release()
        raise HTTPException(status_code=404, detail="case not found") from exc
    except ValueError as exc:
        connection.release()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BaseException:
        connection.release()
        raise

    def body():
        if reset_attempt:
            yield _sse(
                "snapshot",
                {
                    "evaluation_id": evaluation_id,
                    "case_id": case_id,
                    "attempt_id": page.attempt_id,
                    "last_sequence": 0,
                    "reset_reason": "attempt_changed",
                },
            )
        for event in page.events:
            yield _sse(
                "case_event",
                event,
                event_id=hub.case_cursor(
                    evaluation_id,
                    case_id,
                    page.attempt_id,
                    event["sequence"],
                ),
            )
        if not page.events and not reset_attempt:
            yield _sse(
                "snapshot",
                {
                    "evaluation_id": evaluation_id,
                    "case_id": case_id,
                    "attempt_id": page.attempt_id,
                    "last_sequence": page.last_sequence,
                },
            )
        yield _sse(
            "checkpoint",
            {
                "attempt_id": page.attempt_id,
                "last_sequence": page.last_sequence,
                "has_more": page.has_more,
                "incomplete_tail": page.incomplete_tail,
            },
            event_id=hub.case_cursor(
                evaluation_id,
                case_id,
                page.attempt_id,
                page.last_sequence,
            ),
        )

    return ReservedStreamingResponse(
        body(),
        connection,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{case_id}/cameras/{resource_id}/latest")
@router.head("/{case_id}/cameras/{resource_id}/latest", include_in_schema=False)
async def latest_frame(
    evaluation_id: str, case_id: str, resource_id: str, request: Request
):
    selected_resource = _safe_camera_resource_id(resource_id)
    try:
        selected = await _repository_read(
            request,
            ("live-frame", evaluation_id, case_id, selected_resource),
            partial(
                request.app.state.evaluation_repository.read_live_frame,
                evaluation_id,
                case_id,
                selected_resource,
            ),
            timeout_s=1.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="artifact unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _media_bytes_response(request, selected, cache_control="no-store")


@router.get("/{case_id}/cameras/{resource_id}/frames/{frame_index}")
@router.head(
    "/{case_id}/cameras/{resource_id}/frames/{frame_index}",
    include_in_schema=False,
)
async def retained_frame(
    evaluation_id: str,
    case_id: str,
    resource_id: str,
    frame_index: Annotated[int, Path(ge=0, le=JSON_SAFE_INTEGER_MAX)],
    request: Request,
    attempt_id: str | None = None,
):
    selected_resource = _safe_camera_resource_id(resource_id)
    selected_attempt = None if attempt_id is None else _safe_id(attempt_id)
    try:
        selected = await _repository_read(
            request,
            (
                "indexed-frame",
                evaluation_id,
                case_id,
                selected_resource,
                frame_index,
                selected_attempt,
            ),
            partial(
                request.app.state.evaluation_repository.read_indexed_frame,
                evaluation_id,
                case_id,
                selected_resource,
                frame_index,
                **(
                    {} if selected_attempt is None else {"attempt_id": selected_attempt}
                ),
            ),
            timeout_s=1.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="artifact unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _media_bytes_response(
        request,
        selected,
        cache_control="public, max-age=31536000, immutable",
    )


@router.get("/{case_id}/tool-results/{sequence}/image")
@router.head(
    "/{case_id}/tool-results/{sequence}/image",
    include_in_schema=False,
)
async def tool_result_image(
    evaluation_id: str,
    case_id: str,
    sequence: Annotated[int, Path(ge=1, le=JSON_SAFE_INTEGER_MAX)],
    request: Request,
    attempt_id: str | None = None,
):
    selected_attempt = None if attempt_id is None else _safe_id(attempt_id)
    try:
        selected = await _repository_read(
            request,
            (
                "tool-image",
                evaluation_id,
                case_id,
                sequence,
                selected_attempt,
            ),
            partial(
                request.app.state.evaluation_repository.read_tool_image,
                evaluation_id,
                case_id,
                sequence,
                **(
                    {} if selected_attempt is None else {"attempt_id": selected_attempt}
                ),
            ),
            timeout_s=1.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="artifact unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _media_bytes_response(
        request,
        selected,
        cache_control="public, max-age=31536000, immutable",
    )


@router.get("/{case_id}/artifacts/{artifact_id}")
@router.head("/{case_id}/artifacts/{artifact_id}", include_in_schema=False)
async def artifact(
    evaluation_id: str,
    case_id: str,
    artifact_id: str,
    request: Request,
    attempt_id: str | None = None,
):
    return await _artifact_response(
        request,
        evaluation_id,
        case_id,
        _safe_id(artifact_id),
        attempt_id=(None if attempt_id is None else _safe_id(attempt_id)),
        cache_control="public, max-age=31536000, immutable",
        allow_declared_text=True,
    )


@router.post(
    "/{case_id}/rerun",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def rerun_case(evaluation_id: str, case_id: str, request: Request):
    try:
        evaluation = request.app.state.evaluation_repository.get_evaluation(
            evaluation_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    qualification = evaluation.manifest.get("qualification")
    if (
        isinstance(qualification, dict)
        and qualification.get("allow_selective_case_rerun") is False
    ):
        raise HTTPException(
            status_code=409,
            detail="selective Case rerun is disabled for this profile",
        )
    _case(request, evaluation_id, case_id)
    try:
        run = request.app.state.evaluation_service.rerun(evaluation_id, case_id)
    except EvaluationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case not found") from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run)


@router.post(
    "/{case_id}/action-replay",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def action_replay(evaluation_id: str, case_id: str, request: Request):
    handler = request.app.state.action_replay_handler
    if not callable(handler):
        raise HTTPException(
            status_code=501, detail="action replay capability is unavailable"
        )
    try:
        evaluation = request.app.state.evaluation_repository.get_evaluation(
            evaluation_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    detail = _case(request, evaluation_id, case_id)
    if detail.execution_state in {"queued", "starting", "running", "finalizing"}:
        raise HTTPException(status_code=409, detail="source Episode is still active")
    recipes = tuple(
        artifact
        for artifact in detail.artifacts
        if artifact.kind == "action_recipe" and artifact.available
    )
    if len(recipes) != 1:
        raise HTTPException(
            status_code=409, detail="one retained action recipe is required"
        )
    attempt_id = detail.attempt_id
    if attempt_id is None:
        raise HTTPException(
            status_code=409, detail="source Episode attempt is unavailable"
        )
    builder = request.app.state.start_request_builder
    origin = request.app.state.control_plane_origin
    if builder is None or origin is None:
        raise HTTPException(status_code=501, detail="launch capability is unavailable")
    try:
        assembled = builder(_retry_request(evaluation), origin)
    except LaunchValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="source Episode launch configuration is no longer valid",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    blocked = [
        item
        for item in getattr(assembled, "prerequisites", ())
        if item.status == "blocked"
    ]
    if blocked:
        raise HTTPException(
            status_code=422,
            detail="source Episode prerequisites are not ready",
        )
    recipe_id = recipes[0].artifact_id
    source = (evaluation_id, case_id, attempt_id, recipe_id)
    lock = request.app.state.action_replay_lock
    with lock:
        if source in request.app.state.action_replay_sources:
            raise HTTPException(
                status_code=409, detail="action replay already requested"
            )
        request.app.state.action_replay_sources.add(source)
    try:
        current = _case(request, evaluation_id, case_id)
        current_recipes = tuple(
            artifact
            for artifact in current.artifacts
            if artifact.kind == "action_recipe" and artifact.available
        )
        if (
            current.attempt_id != attempt_id
            or current.execution_state
            in {"queued", "starting", "running", "finalizing"}
            or len(current_recipes) != 1
            or current_recipes[0].artifact_id != recipe_id
        ):
            with lock:
                request.app.state.action_replay_sources.discard(source)
            raise HTTPException(
                status_code=409,
                detail="source Episode attempt changed before replay",
            )
        return handler(evaluation_id, case_id, attempt_id, recipe_id)
    except EvaluationNotFoundError as exc:
        with lock:
            request.app.state.action_replay_sources.discard(source)
        raise HTTPException(status_code=404, detail="source Episode not found") from exc
    except ConflictError as exc:
        with lock:
            request.app.state.action_replay_sources.discard(source)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        with lock:
            request.app.state.action_replay_sources.discard(source)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        with lock:
            request.app.state.action_replay_sources.discard(source)
        raise


async def _read_case(
    request: Request,
    evaluation_id: str,
    case_id: str,
    *,
    attempt_id: str | None = None,
):
    runtime_store = getattr(request.app.state, "case_runtime_store", None)
    if runtime_store is not None:
        try:
            return runtime_store.get_case(evaluation_id, case_id, attempt_id)
        except SnapshotRebuilding as exc:
            raise HTTPException(
                status_code=503,
                detail="case_snapshot_rebuilding",
                headers={"Retry-After": "1"},
            ) from exc
        except SnapshotUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="case_snapshot_unavailable",
            ) from exc
        except KeyError:
            pass
    try:
        repository = request.app.state.evaluation_repository
        return await _repository_read(
            request,
            ("case", evaluation_id, case_id, attempt_id),
            partial(
                repository.get_case,
                evaluation_id,
                case_id,
                **({} if attempt_id is None else {"attempt_id": attempt_id}),
            ),
            timeout_s=2.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _case(
    request: Request,
    evaluation_id: str,
    case_id: str,
    *,
    attempt_id: str | None = None,
):
    try:
        repository = request.app.state.evaluation_repository
        if attempt_id is None:
            return repository.get_case(evaluation_id, case_id)
        return repository.get_case(
            evaluation_id,
            case_id,
            attempt_id=attempt_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case attempt not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=422, detail="resource ID must be non-empty")
    decoded = value
    for _ in range(3):
        expanded = unquote(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}", decoded) is None:
        raise HTTPException(status_code=422, detail="invalid resource ID")
    return decoded


def _safe_camera_resource_id(value: str) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail="invalid camera resource ID")
    decoded = value
    for _ in range(3):
        expanded = unquote(decoded)
        if expanded == decoded:
            break
        decoded = expanded
    if re.fullmatch(r"camera-sha256-[0-9a-f]{64}", decoded) is None:
        raise HTTPException(status_code=422, detail="invalid camera resource ID")
    return decoded


def _media_bytes_response(
    request: Request,
    selected: MediaBytes,
    *,
    cache_control: str,
) -> Response:
    if selected.size != len(selected.data):
        raise HTTPException(status_code=422, detail="camera media size is invalid")
    if selected.media_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="unsupported artifact media type")
    headers = {
        "Cache-Control": cache_control,
        "ETag": selected.etag,
        "Accept-Ranges": "bytes",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": 'inline; filename="preview.bin"',
    }
    if _etag_matches(request.headers.get("if-none-match"), selected.etag):
        return Response(status_code=304, headers=headers)
    selected_range = request.headers.get("range")
    if selected_range is None:
        start, end, status_code = 0, max(0, selected.size - 1), 200
    else:
        try:
            start, end = _parse_range(selected_range, selected.size)
        except ValueError:
            return Response(
                status_code=416,
                headers={
                    **headers,
                    "Content-Range": f"bytes */{selected.size}",
                },
            )
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{selected.size}"
    length = 0 if selected.size == 0 else end - start + 1
    headers["Content-Length"] = str(length)
    content = b"" if request.method == "HEAD" else selected.data[start : end + 1]
    return Response(
        content=content,
        status_code=status_code,
        media_type=selected.media_type,
        headers=headers,
    )


async def _artifact_response(
    request: Request,
    evaluation_id: str,
    case_id: str,
    artifact_id: str | None,
    *,
    attempt_id: str | None = None,
    cache_control: str,
    inline: bool = False,
    allow_declared_text: bool = False,
):
    case = await _read_case(
        request,
        evaluation_id,
        case_id,
        attempt_id=attempt_id,
    )
    text_allowed = allow_declared_text and _is_declared_text_artifact(case, artifact_id)
    token = object()
    operation_lock = threading.Lock()
    owner = _MediaStreamOwner()

    def open_selected():
        stream: BinaryIO | None = None
        try:
            repository = request.app.state.evaluation_repository
            stream = repository.open_artifact(
                evaluation_id,
                case_id,
                artifact_id,
                **({} if attempt_id is None else {"attempt_id": attempt_id}),
            )
            etag, size = _stream_metadata(stream)
            prefix = stream.read(min(size, 1024 * 1024 + 1))
            media_type = _sniff_media_type(prefix, size=size, allow_text=text_allowed)
            if media_type is None:
                raise HTTPException(
                    status_code=415, detail="unsupported artifact media type"
                )
            stream.seek(0)
            if not owner.publish(stream):
                stream.close()
            return stream, etag, size, media_type
        except BaseException:
            if stream is not None:
                stream.close()
            raise

    try:
        try:
            stream, etag, size, media_type = await _media_gate_call(
                request,
                ("artifact-open", token),
                open_selected,
            )
        except BaseException:
            await _close_media_stream(request, token, operation_lock, owner)
            raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=404, detail="artifact unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if inline:
        disposition = 'inline; filename="preview.bin"'
    elif media_type == "video/mp4":
        disposition = 'inline; filename="video.mp4"'
    elif media_type == "video/webm":
        disposition = 'inline; filename="video.webm"'
    else:
        disposition = 'attachment; filename="artifact.bin"'
    headers = {
        "Cache-Control": cache_control,
        "ETag": etag,
        "Accept-Ranges": "bytes",
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": disposition,
    }
    if _etag_matches(request.headers.get("if-none-match"), etag):
        await _close_media_stream(request, token, operation_lock, owner)
        return Response(status_code=304, headers=headers)
    selected_range = request.headers.get("range")
    if selected_range is None:
        start, end, status_code = 0, max(0, size - 1), 200
    else:
        try:
            start, end = _parse_range(selected_range, size)
        except ValueError:
            await _close_media_stream(request, token, operation_lock, owner)
            return Response(
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = 0 if size == 0 else end - start + 1
    headers["Content-Length"] = str(length)
    if request.method == "HEAD":
        await _close_media_stream(request, token, operation_lock, owner)
        return Response(status_code=status_code, media_type=media_type, headers=headers)
    return _OwnedStreamingResponse(
        _stream_and_close_async(
            request,
            token,
            operation_lock,
            owner,
            stream,
            start,
            length,
        ),
        partial(_close_media_stream, request, token, operation_lock, owner),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


def _stream_metadata(stream: BinaryIO) -> tuple[str, int]:
    try:
        metadata = os.fstat(stream.fileno())
    except (AttributeError, OSError):
        position = stream.tell()
        stream.seek(0)
        value = stream.read()
        stream.seek(position)
        return f'"{hashlib.sha256(value).hexdigest()}"', len(value)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OSError("artifact is not an exclusively retained regular file")
    identity = (
        f"{metadata.st_dev}:{metadata.st_ino}:{metadata.st_mtime_ns}:{metadata.st_size}"
    ).encode()
    return f'"{hashlib.sha256(identity).hexdigest()}"', metadata.st_size


def _is_declared_text_artifact(case: CaseDetail, artifact_id: str | None) -> bool:
    if artifact_id is None:
        return False
    for artifact in case.artifacts:
        if artifact.artifact_id != artifact_id:
            continue
        kind_tokens = set(re.split(r"[^a-z0-9]+", artifact.kind.lower()))
        return bool(kind_tokens & {"log", "logs", "text"})
    return False


def _sniff_media_type(
    prefix: bytes, *, size: int, allow_text: bool = False
) -> str | None:
    stripped = prefix.lstrip()
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        return "video/mp4"
    if prefix.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    if prefix.startswith(b"PAR1"):
        return "application/vnd.apache.parquet"
    if stripped.startswith((b"{", b"[")):
        if size > 1024 * 1024:
            return "application/octet-stream"
        try:
            parsed = json.loads(prefix)
        except UnicodeDecodeError:
            return None
        except json.JSONDecodeError:
            try:
                rows = [
                    json.loads(line) for line in prefix.splitlines() if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return (
                "application/x-ndjson"
                if rows and all(isinstance(row, dict) for row in rows)
                else None
            )
        return "application/json" if isinstance(parsed, (dict, list)) else None
    if allow_text:
        try:
            prefix.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if any(byte < 32 and byte not in (9, 10, 13) for byte in prefix):
            return None
        return "text/plain"
    return None


def _etag_matches(header: str | None, etag: str) -> bool:
    if header is None:
        return False
    return any(
        candidate == "*" or candidate.removeprefix("W/") == etag
        for candidate in (item.strip() for item in header.split(","))
    )


def _parse_range(value: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if match is None or size <= 0:
        raise ValueError("invalid byte range")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("invalid byte range")
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("invalid byte range")
        return max(0, size - length), size - 1
    start = int(first)
    end = size - 1 if not last else int(last)
    if start >= size or end < start:
        raise ValueError("invalid byte range")
    return start, min(end, size - 1)


async def _media_gate_call(
    request: Request,
    key: object,
    function: Callable[[], Any],
    *,
    timeout_s: float = 2.0,
):
    try:
        return await request.app.state.media_read_gate.run(
            key,
            function,
            timeout_s=timeout_s,
        )
    except ReadGateBusy as exc:
        raise HTTPException(
            status_code=503,
            detail="dashboard media capacity is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except ReadGateTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail="dashboard media read timed out",
        ) from exc


def _complete_stream_close(
    owner: _MediaStreamOwner,
    claim: object,
    stream: BinaryIO,
) -> None:
    try:
        stream.close()
    except BaseException:
        owner.fail_close(claim)
        raise
    else:
        owner.complete_close(claim)


def _close_claimed_stream(
    owner: _MediaStreamOwner,
    claim: object,
    operation_lock: threading.Lock,
    stream: BinaryIO,
) -> None:
    if not owner.start_close(claim):
        return
    with operation_lock:
        _complete_stream_close(owner, claim, stream)


def _fallback_media_close(
    owner: _MediaStreamOwner,
    operation_lock: threading.Lock,
) -> None:
    stream, worker_owns_close, claim = owner.request_close()
    if stream is None or worker_owns_close or claim is None:
        return
    if not operation_lock.acquire(blocking=False):
        owner.abandon_close(claim)
        return
    try:
        if owner.start_close(claim):
            _complete_stream_close(owner, claim, stream)
    finally:
        operation_lock.release()


async def _close_media_stream(
    request: Request,
    token: object,
    operation_lock: threading.Lock,
    owner: _MediaStreamOwner,
) -> None:
    stream, worker_owns_close, claim = owner.request_close()
    if stream is None or worker_owns_close or claim is None:
        return
    try:
        await request.app.state.media_read_gate.run(
            ("artifact-close", token),
            partial(
                _close_claimed_stream,
                owner,
                claim,
                operation_lock,
                stream,
            ),
            timeout_s=2.0,
        )
    except ReadGateTimeout:
        if owner.abandon_close(claim):
            _fallback_media_close(owner, operation_lock)
    except asyncio.CancelledError:
        caller_cancelled = bool(
            asyncio.current_task() and asyncio.current_task().cancelling()
        )
        if owner.abandon_close(claim):
            _fallback_media_close(owner, operation_lock)
        if caller_cancelled:
            raise
    except Exception:
        owner.abandon_close(claim)
        _fallback_media_close(owner, operation_lock)


async def _stream_and_close_async(
    request: Request,
    token: object,
    operation_lock: threading.Lock,
    owner: _MediaStreamOwner,
    stream: BinaryIO,
    start: int,
    remaining: int,
) -> AsyncIterator[bytes]:
    position = start
    operation = 0
    try:
        while remaining > 0:
            size = min(64 * 1024, remaining)

            def read() -> bytes:
                with operation_lock:
                    if not owner.begin_operation():
                        claim = owner.claim_read_close()
                        if claim is not None:
                            _complete_stream_close(owner, claim, stream)
                        return b""
                    try:
                        stream.seek(position)
                        return stream.read(size)
                    finally:
                        claim = owner.finish_operation()
                        if claim is not None:
                            _complete_stream_close(owner, claim, stream)

            chunk = await _media_gate_call(
                request,
                ("artifact-read", token, operation),
                read,
            )
            if not chunk:
                break
            position += len(chunk)
            remaining -= len(chunk)
            operation += 1
            yield chunk
    finally:
        await _close_media_stream(request, token, operation_lock, owner)


__all__ = ["router"]
