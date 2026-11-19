"""Evaluation collection, lifecycle, and bounded notification routes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Hashable, Mapping
from dataclasses import replace
from functools import partial
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.envs.base import get_env_spec
from rpent.envs.launch import LaunchValidationError
from rpent.evaluation.front_controller import build_evaluation_manifest
from rpent.evaluation.manifest import EvaluationManifest
from rpent.planning.manifest import content_hash
from rpent.recording.json_safe import json_safe
from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout
from rpent.ui.schemas import (
    CommandView,
    EvaluationDetail,
    EvaluationListItem,
    LaunchPreflightView,
    LaunchPreviewView,
)
from rpent.ui.service import (
    ConflictError,
    EvaluationNotFoundError,
    ManifestConfirmationError,
)
from rpent.ui.sse import (
    DEFAULT_POLL_INTERVAL,
    SSE_PAGE_MAX_BYTES,
    SSE_PAGE_MAX_EVENTS,
    ReservedStreamingResponse,
    SSECapacityError,
    encode_sse,
    evaluation_event_stream,
)

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])
MAX_PREVIEW_CASES = 10_000
_STABLE_RUNTIME_KEYS = (
    "main_interpreter",
    "policy_interpreter",
    "simulator_interpreter",
    "simulator_endpoint",
    "simulator_endpoint_argument",
    "policy_endpoint",
    "policy_endpoint_argument",
    "sam3_endpoint",
    "sam3_endpoint_argument",
    "rendering_backend",
    "shared_policy_required",
    "grouped_worker_module",
    "qualification_module",
    "worker_module",
    "runner_module",
)


class RecordingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mode: str
    camera_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profile(self) -> RecordingRequest:
        if self.mode not in {"none", "primary", "selected", "all"}:
            raise ValueError("unknown recording mode")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        if any(not camera_id.strip() for camera_id in self.camera_ids):
            raise ValueError("camera_ids must be non-empty")
        if self.mode == "selected" and not self.camera_ids:
            raise ValueError("selected recording requires camera_ids")
        if self.mode != "selected" and self.camera_ids:
            raise ValueError("camera_ids are valid only for selected recording")
        return self


class StartEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    simulator_id: str = Field(min_length=1)
    execution_mode: str = Field(default="hybrid", min_length=1)
    config: dict[str, Any]
    recording: RecordingRequest | None = None
    workload_confirmation_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )


class LaunchPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    simulator_id: str = Field(min_length=1)
    execution_mode: str = Field(default="hybrid", min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reason: Literal["user_requested"] = "user_requested"


def _run_payload(run: Any, *, operation: str | None = None) -> dict[str, Any]:
    return {
        "evaluation_id": run.evaluation_id,
        "operation": run.operation if operation is None else operation,
        "state": run.state,
    }


@router.get("", response_model=list[EvaluationListItem])
async def list_evaluations(request: Request):
    return await _repository_read(
        request,
        ("evaluations",),
        request.app.state.evaluation_repository.list_evaluations,
        timeout_s=5.0,
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=CommandView)
def start_evaluation(payload: StartEvaluationRequest, request: Request):
    builder = request.app.state.start_request_builder
    origin = request.app.state.control_plane_origin
    if builder is None or origin is None:
        raise HTTPException(status_code=501, detail="launch capability is unavailable")
    try:
        assembled = builder(payload, origin)
    except LaunchValidationError as exc:
        return _launch_validation_response(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    blocked = [
        item
        for item in getattr(assembled, "prerequisites", ())
        if item.status == "blocked"
    ]
    if blocked:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Launch prerequisites are not ready.",
                "field_errors": {
                    "prerequisites": " ".join(item.message for item in blocked)
                },
            },
        )
    if hasattr(assembled, "args") and _requires_confirmation_projection(
        assembled, payload
    ):
        try:
            manifest = _preview_manifest(assembled)
            projection = _workload_projection(manifest)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if projection["requires_workload_confirmation"]:
            confirmed = payload.workload_confirmation_digest
            expected = projection["preview_digest"]
            if confirmed is None:
                return _confirmation_error(
                    "Confirm the current official workload preview."
                )
            if confirmed != expected:
                return _confirmation_error("The confirmed workload preview is stale.")
            assembled = replace(
                assembled,
                manifest_validator=_confirmed_workload(expected),
            )
    try:
        return _run_payload(request.app.state.evaluation_service.start(assembled))
    except ManifestConfirmationError:
        return _confirmation_error("The confirmed workload preview is stale.")
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvaluationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _requires_confirmation_projection(
    assembled: Any, payload: StartEvaluationRequest
) -> bool:
    if payload.workload_confirmation_digest is not None:
        return True
    profile = getattr(getattr(assembled, "args", None), "evaluation_profile", None)
    return isinstance(profile, str) and bool(profile) and profile != "development"


@router.post(
    "/preflight",
    response_model=LaunchPreflightView,
    response_model_exclude_defaults=True,
)
def preflight_evaluation(payload: LaunchPreflightRequest):
    spec = get_env_spec(payload.simulator_id)
    if spec.launch_preset is None:
        raise HTTPException(status_code=501, detail="launch capability is unavailable")
    try:
        prerequisites = spec.launch_preset().partial_preflight(
            payload.execution_mode, payload.config
        )
    except LaunchValidationError as exc:
        return _launch_validation_response(exc)
    return LaunchPreflightView(
        prerequisites=tuple(
            {
                "id": item.id,
                "label": item.label,
                "status": item.status,
                "message": item.message,
            }
            for item in prerequisites
        )
    )


@router.post(
    "/preview",
    response_model=LaunchPreviewView,
    response_model_exclude_defaults=True,
)
def preview_evaluation(payload: StartEvaluationRequest, request: Request):
    builder = request.app.state.start_request_builder
    origin = request.app.state.control_plane_origin
    if builder is None or origin is None:
        raise HTTPException(status_code=501, detail="launch capability is unavailable")
    try:
        assembled = builder(payload, origin)
        manifest = _preview_manifest(assembled)
    except LaunchValidationError as exc:
        return _launch_validation_response(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    identities = {(case.task_name, case.task_index) for case in manifest.cases}
    estimate = manifest.recording["storage_estimate"]
    task_resolution = str(estimate["basis"]["task_resolution"])
    projection = _workload_projection(manifest)
    return LaunchPreviewView(
        case_count=len(manifest.cases),
        task_count=len(identities),
        episodes_per_task=int(manifest.schedule["episodes_per_task"]),
        total_horizon_steps=sum(case.horizon for case in manifest.cases),
        recording_recommendation=("all" if len(manifest.cases) == 1 else "primary"),
        storage_estimate={
            **dict(estimate),
            "basis": dict(estimate["basis"]),
        },
        task_resolution=task_resolution,
        effective_config=json_safe(manifest.launch.get("effective", {})),
        prerequisites=tuple(
            {
                "id": item.id,
                "label": item.label,
                "status": item.status,
                "message": item.message,
            }
            for item in getattr(assembled, "prerequisites", ())
        ),
        **projection,
    )


def _preview_manifest(assembled: Any) -> EvaluationManifest:
    preview_args = copy.copy(assembled.args)
    preview_args.dry_run = True
    manifest, _ = build_evaluation_manifest(
        preview_args,
        assembled.env_spec,
        assembled.argv,
        resolved_tasks=assembled.resolved_tasks,
        git_identity=assembled.git_identity or (lambda: ("preview", False)),
        max_cases=MAX_PREVIEW_CASES,
        launch=getattr(assembled, "launch", {}),
    )
    return manifest


def _workload_projection(manifest: EvaluationManifest) -> dict[str, Any]:
    task_count = len({(case.task_name, case.task_index) for case in manifest.cases})
    case_count = len(manifest.cases)
    rollouts = int(manifest.schedule["episodes_per_task"])
    grouping = manifest.schedule.get("worker_grouping")
    batch_count = task_count if grouping == "task" else case_count
    control_hz = manifest.contracts.get("control_hz")
    estimated_runtime = (
        sum(case.horizon for case in manifest.cases) / control_hz
        if isinstance(control_hz, (int, float))
        and not isinstance(control_hz, bool)
        and control_hz > 0
        else None
    )
    effective = manifest.schedule.get("effective_inputs")
    if not isinstance(effective, Mapping):
        effective = manifest.launch.get("effective", {})
    locked_inputs = manifest.schedule.get("locked_inputs")
    if not isinstance(locked_inputs, Mapping):
        locked_inputs = {}
    submitted = manifest.launch.get("submitted")
    if not isinstance(submitted, Mapping):
        submitted = {}
    launch_effective = manifest.launch.get("effective")
    if not isinstance(launch_effective, Mapping):
        launch_effective = {}
    pool_managed_policy_endpoint = (
        manifest.environment.get("name") == "robocasa"
        and launch_effective.get("reuse_policy_service") is True
        and not launch_effective.get("policy_endpoint")
        and not submitted.get("policy_endpoint")
    )
    profile_id = manifest.schedule.get("evaluation_profile")
    locked_field_names = tuple(
        str(name)
        for name, value in locked_inputs.items()
        if not (
            isinstance(profile_id, str)
            and value == profile_id
            and submitted.get(name) == profile_id
        )
    )
    workload = {
        "profile_id": manifest.schedule.get("evaluation_profile"),
        "task_count": task_count,
        "rollouts_per_task": rollouts,
        "case_count": case_count,
        "batch_count": batch_count,
        "scene_split": manifest.schedule.get("scene_split")
        or manifest.simulator.get("scene_split"),
        "seed": manifest.schedule.get("base_seed"),
        "horizon_source": manifest.schedule.get("horizon_source"),
        "estimated_runtime_seconds": estimated_runtime,
        "locked_effective_facts": json_safe(effective),
        "locked_field_names": locked_field_names,
    }
    digest = "sha256:" + content_hash(
        {
            "domain": "rpent.launch-workload-preview.v1",
            "environment": manifest.environment,
            "simulator": {
                key: value
                for key, value in manifest.simulator.items()
                if key != "task_resolution"
            },
            "execution_mode": manifest.launch.get("execution_mode"),
            "effective_config": manifest.launch.get("effective", {}),
            "cases": [case.to_payload() for case in manifest.cases],
            "schedule": manifest.schedule,
            "recording": _stable_recording_identity(manifest.recording),
            "policy": manifest.policy,
            "contracts": manifest.contracts,
            "runtime_identity": {
                key: json_safe(manifest.runtime[key])
                for key in _STABLE_RUNTIME_KEYS
                if key in manifest.runtime
                and not (key == "policy_endpoint" and pool_managed_policy_endpoint)
            },
            "workload": workload,
        }
    )
    return {
        "workload": workload,
        "requires_workload_confirmation": manifest.qualification.get(
            "require_workload_confirmation"
        )
        is True,
        "preview_digest": digest,
    }


def _confirmed_workload(expected_digest: str):
    def validate(manifest: EvaluationManifest) -> None:
        if _workload_projection(manifest)["preview_digest"] != expected_digest:
            raise ManifestConfirmationError(
                "authoritative manifest does not match confirmed workload"
            )

    return validate


def _stable_recording_identity(recording: Mapping[str, Any]) -> dict[str, Any]:
    payload = json_safe(recording)
    estimate = payload.get("storage_estimate")
    if isinstance(estimate, dict):
        basis = estimate.get("basis")
        if isinstance(basis, dict):
            basis.pop("task_resolution", None)
    return payload


def _confirmation_error(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Official workload confirmation is required.",
            "field_errors": {"workload_confirmation_digest": message},
        },
    )


def _launch_validation_response(exc: LaunchValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid launch configuration.",
            "field_errors": exc.errors,
        },
    )


@router.get("/{evaluation_id}", response_model=EvaluationDetail)
async def get_evaluation(evaluation_id: str, request: Request):
    try:
        return await _repository_read(
            request,
            ("evaluation", evaluation_id),
            partial(
                request.app.state.evaluation_repository.get_evaluation,
                evaluation_id,
            ),
            timeout_s=2.0,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc


async def _repository_read(
    request: Request,
    key: Hashable,
    function: Callable[[], Any],
    *,
    timeout_s: float,
):
    try:
        return await request.app.state.repository_read_gate.run(
            key,
            function,
            timeout_s=timeout_s,
        )
    except ReadGateBusy as exc:
        raise HTTPException(
            status_code=503,
            detail="dashboard read capacity is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except ReadGateTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail="dashboard read timed out",
        ) from exc


@router.post(
    "/{evaluation_id}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def cancel_evaluation(
    evaluation_id: str,
    request: Request,
    payload: Annotated[CancelRequest, Body()] = CancelRequest(),
):
    try:
        return _run_payload(
            request.app.state.evaluation_service.cancel(evaluation_id, payload.reason),
            operation="cancel",
        )
    except EvaluationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{evaluation_id}/force-stop",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def force_stop_evaluation(evaluation_id: str, request: Request):
    handler = request.app.state.force_stop_handler
    if not callable(handler):
        raise HTTPException(
            status_code=501, detail="force stop capability is unavailable"
        )
    try:
        detail = request.app.state.evaluation_repository.get_evaluation(evaluation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    if detail.execution_state not in {"queued", "starting", "running", "finalizing"}:
        raise HTTPException(status_code=409, detail="Episode is not active")
    try:
        return handler(evaluation_id)
    except EvaluationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/{evaluation_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def resume_evaluation(evaluation_id: str, request: Request):
    try:
        detail = request.app.state.evaluation_repository.get_evaluation(evaluation_id)
        if detail.qualification.accepted or detail.official_metrics.is_complete:
            raise HTTPException(
                status_code=409,
                detail="completed official evaluation cannot be resumed",
            )
        return _run_payload(request.app.state.evaluation_service.resume(evaluation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    except EvaluationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{evaluation_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=CommandView,
)
def retry_evaluation(evaluation_id: str, request: Request):
    try:
        detail = request.app.state.evaluation_repository.get_evaluation(evaluation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    if detail.qualification.accepted or detail.official_metrics.is_complete:
        raise HTTPException(
            status_code=409,
            detail="completed official evaluation cannot be retried",
        )
    preparation = detail.preparation
    if preparation is None or preparation.status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=409,
            detail="evaluation preparation is not retryable",
        )
    try:
        payload = _retry_request(detail)
        builder = request.app.state.start_request_builder
        origin = request.app.state.control_plane_origin
        if builder is None or origin is None:
            raise HTTPException(
                status_code=501, detail="launch capability is unavailable"
            )
        assembled = builder(payload, origin)
        run = request.app.state.evaluation_service.retry(evaluation_id, assembled)
    except LaunchValidationError as exc:
        return _launch_validation_response(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _run_payload(run, operation="retry")


@router.get("/{evaluation_id}/submission", response_class=Response)
def download_submission(evaluation_id: str, request: Request):
    try:
        payload = request.app.state.evaluation_repository.read_submission(evaluation_id)
    except (KeyError, OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="submission unavailable") from exc
    return Response(
        payload,
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="leaderboard_submission.json"',
            "Cache-Control": "no-store",
        },
    )


def _retry_request(detail: EvaluationDetail) -> StartEvaluationRequest:
    try:
        environment = detail.manifest["environment"]
        launch = detail.manifest["launch"]
        recording = detail.manifest["recording"]
        if not all(
            isinstance(value, dict) for value in (environment, launch, recording)
        ):
            raise ValueError("historical launch identity must contain objects")
        simulator_id = environment["name"]
        execution_mode = launch["execution_mode"]
        submitted = launch["submitted"]
        mode = recording["mode"]
        camera_ids = recording.get("camera_ids", [])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"historical launch identity is incomplete: {exc}") from exc
    if not isinstance(submitted, dict):
        raise ValueError("historical submitted launch config must be an object")
    return StartEvaluationRequest(
        simulator_id=simulator_id,
        execution_mode=execution_mode,
        config=dict(submitted),
        recording=RecordingRequest(mode=mode, camera_ids=camera_ids),
    )


@router.get("/{evaluation_id}/stream")
async def stream_evaluation(
    evaluation_id: str,
    request: Request,
    cursor: str | None = None,
    timeout: Annotated[float | None, Query(ge=0, le=30)] = None,
):
    repository = request.app.state.evaluation_repository
    manager = request.app.state.sse_manager
    try:
        await _sse_repository_call(manager, repository.get_evaluation, evaluation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evaluation not found") from exc
    hub = request.app.state.evaluation_event_hub
    try:
        selected = (
            cursor or request.headers.get("last-event-id") or hub.cursor(evaluation_id)
        )
        hub.read(
            selected,
            evaluation_id=evaluation_id,
            max_events=1,
            max_bytes=256 * 1024,
        )
        connection = manager.reserve()
    except SSECapacityError as exc:
        raise _sse_capacity_response(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        if timeout is None:
            stream = evaluation_event_stream(
                request,
                request.app.state.evaluation_repository,
                hub,
                manager,
                evaluation_id,
                cursor=selected,
                heartbeat_interval=getattr(
                    request.app.state, "sse_heartbeat_interval", 15.0
                ),
                poll_interval=getattr(
                    request.app.state, "sse_poll_interval", DEFAULT_POLL_INTERVAL
                ),
            )
            return ReservedStreamingResponse(
                stream,
                connection,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                },
            )
        batch = await _finite_evaluation_batch(
            hub,
            selected,
            evaluation_id,
            timeout,
        )
    except ValueError as exc:
        connection.release()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BaseException:
        connection.release()
        raise

    expired_payload = None
    try:
        if batch.expired:
            evaluation = await _sse_repository_call(
                manager,
                repository.get_evaluation,
                evaluation_id,
            )
            checkpoint = await _sse_repository_call(
                manager,
                repository.event_checkpoint,
                evaluation_id,
            )
            expired_payload = {
                "cursor": batch.next_cursor,
                "expired_reason": batch.expired_reason,
                "evaluation": evaluation.model_dump(),
                "last_sequences": checkpoint,
            }
    except BaseException:
        connection.release()
        raise

    def body():
        if expired_payload is not None:
            yield _sse("snapshot", expired_payload)
        for event in batch.events:
            yield _sse("evaluation_event", event.to_payload())
        if not batch.events and not batch.expired:
            yield _sse("heartbeat", {"cursor": batch.next_cursor})
        yield _sse("cursor", {"cursor": batch.next_cursor}, event_id=batch.next_cursor)

    return ReservedStreamingResponse(
        body(),
        connection,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


async def _finite_evaluation_batch(
    hub: Any,
    cursor: str,
    evaluation_id: str,
    timeout: float,
):
    subscription = hub.open_async_subscription(evaluation_id)
    try:
        batch = hub.read(
            cursor,
            evaluation_id=evaluation_id,
            max_events=SSE_PAGE_MAX_EVENTS,
            max_bytes=SSE_PAGE_MAX_BYTES,
        )
        if not batch.events and not batch.expired and timeout > 0:
            await subscription.wait(timeout)
            batch = hub.read(
                cursor,
                evaluation_id=evaluation_id,
                max_events=SSE_PAGE_MAX_EVENTS,
                max_bytes=SSE_PAGE_MAX_BYTES,
            )
        return batch
    finally:
        subscription.close()


async def _sse_repository_call(
    manager: Any, function: Callable[..., Any], *args, **kwargs
):
    try:
        return await manager.repository_call(function, *args, **kwargs)
    except SSECapacityError as exc:
        raise _sse_capacity_response(exc) from exc


def _sse_capacity_response(exc: SSECapacityError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=str(exc),
        headers={"Retry-After": "2"},
    )


def _sse(
    name: str, payload: dict[str, Any], *, event_id: str | int | None = None
) -> str:
    return encode_sse(
        name, payload, event_id=None if event_id is None else str(event_id)
    )


__all__ = [
    "CancelRequest",
    "RecordingRequest",
    "StartEvaluationRequest",
    "router",
]
