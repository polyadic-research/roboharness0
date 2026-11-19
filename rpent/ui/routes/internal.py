"""Best-effort worker notification ingress."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rpent.dashboard.attachment import LifecycleHint
from rpent.recording.contracts import JSON_SAFE_INTEGER_MAX, EventEnvelope
from rpent.recording.notifications import (
    NotificationTooLargeError,
    validate_notification,
)
from rpent.ui.io_gate import ReadGateBusy, ReadGateTimeout
from rpent.ui.notification_identity import NotificationAttemptConflict

router = APIRouter(prefix="/api/internal", tags=["internal"])


class EventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int
    evaluation_id: str
    case_id: str
    attempt_id: str
    sequence: int = Field(ge=0, le=JSON_SAFE_INTEGER_MAX)
    recorded_at: str
    event_type: str
    step_id: int | None = Field(ge=0, le=JSON_SAFE_INTEGER_MAX)
    operation_id: str | None
    execution_id: str | None
    payload: dict[str, Any]


class LifecycleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    kind: Literal["evaluation_persisted", "attempt_registered", "result_accepted"]
    evaluation_id: str = Field(min_length=1, max_length=512)
    manifest_id: str = Field(min_length=1, max_length=512)
    case_id: str | None = Field(default=None, min_length=1, max_length=512)
    attempt_id: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_identity_scope(self) -> "LifecycleRequest":
        has_attempt_identity = self.case_id is not None and self.attempt_id is not None
        if self.kind == "evaluation_persisted":
            if self.case_id is not None or self.attempt_id is not None:
                raise ValueError(
                    "evaluation_persisted must not include Case or Attempt identity"
                )
        elif not has_attempt_identity:
            raise ValueError(f"{self.kind} requires Case and Attempt identity")
        return self


@router.post("/lifecycle", status_code=204)
def publish_lifecycle(payload: LifecycleRequest, request: Request) -> Response:
    try:
        request.app.state.dashboard_attachment_refresher.accept(
            LifecycleHint(**payload.model_dump())
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="lifecycle identity not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail="lifecycle durable state mismatch"
        ) from exc
    return Response(status_code=204)


@router.post("/events", status_code=204)
async def publish_event(payload: EventRequest, request: Request) -> Response:
    try:
        event = EventEnvelope.from_payload(payload.model_dump())
        validate_notification(event)
    except NotificationTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        identities = request.app.state.notification_identities
        accept_identity = getattr(identities, "accept_identity", None)
        identity = (
            await accept_identity(event)
            if callable(accept_identity)
            else await identities.accept(event)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="case not found") from exc
    except NotificationAttemptConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    runtime_store = getattr(request.app.state, "case_runtime_store", None)
    if runtime_store is None:
        request.app.state.evaluation_event_hub.publish(event)
        return Response(status_code=204)
    if identity is None:
        try:
            identity = await request.app.state.repository_read_gate.run(
                ("notification-identity", event.evaluation_id, event.case_id),
                partial(
                    request.app.state.runtime_repository.current_attempt_identity,
                    event.evaluation_id,
                    event.case_id,
                ),
                timeout_s=0.5,
            )
        except (ReadGateBusy, ReadGateTimeout) as exc:
            raise HTTPException(
                status_code=503,
                detail="recording_declaration_unavailable",
                headers={"Retry-After": "1"},
            ) from exc
    async def load_recording():
        return await request.app.state.repository_read_gate.run(
            ("recording-declaration", *runtime_key),
            partial(
                request.app.state.runtime_repository.load_recording_declaration,
                identity,
            ),
            timeout_s=0.5,
        )

    runtime_key = (event.evaluation_id, event.case_id, event.attempt_id)
    try:
        if event.event_type == "action_camera_clips_ready":
            recording = await load_recording()
            runtime_store.prime_recording_declaration(identity, recording)
        else:
            recording = await runtime_store.recording_declaration(
                runtime_key, load_recording
            )
    except (FileNotFoundError, OSError, TypeError, ValueError, ReadGateBusy) as exc:
        raise HTTPException(
            status_code=503,
            detail="recording_declaration_unavailable",
            headers={"Retry-After": "1"},
        ) from exc
    except ReadGateTimeout as exc:
        raise HTTPException(
            status_code=503,
            detail="recording_declaration_unavailable",
            headers={"Retry-After": "1"},
        ) from exc
    reduction = runtime_store.apply_event(event, recording)
    if reduction.disposition == "rejected":
        raise HTTPException(status_code=409, detail=reduction.reason or "rejected")
    if reduction.update is not None:
        request.app.state.evaluation_event_hub.publish_runtime_update(
            reduction.update
        )
    elif reduction.disposition == "duplicate" or (
        reduction.disposition == "repairing" and not reduction.sequence_gap
    ):
        request.app.state.evaluation_event_hub.publish(event)
    return Response(status_code=204)


__all__ = ["router"]
