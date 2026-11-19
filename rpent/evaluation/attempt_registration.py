"""Authoritative, versioned registration for one current Case Attempt."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rpent.evaluation.batches import validate_case_attempt_id
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.planning.manifest import canonical_json
from rpent.recording import media

CURRENT_ATTEMPT_SCHEMA_VERSION = 1
MAX_CURRENT_ATTEMPT_BYTES = 16 * 1024
_REGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "evaluation_id",
        "manifest_id",
        "case_id",
        "attempt_id",
        "registered_at",
    }
)


@dataclass(frozen=True, slots=True)
class AttemptRegistration:
    schema_version: int
    evaluation_id: str
    manifest_id: str
    case_id: str
    attempt_id: str
    registered_at: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def register_attempt(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
    registered_at: str,
) -> AttemptRegistration:
    """Write immutable Case metadata, then atomically commit its locator."""
    if not isinstance(manifest, EvaluationManifest):
        raise TypeError("manifest must be an EvaluationManifest")
    if not isinstance(case, EpisodeCase) or case not in manifest.cases:
        raise ValueError("case does not belong to manifest")
    validate_case_attempt_id(attempt_id)
    _timestamp(registered_at)
    registration = AttemptRegistration(
        schema_version=CURRENT_ATTEMPT_SCHEMA_VERSION,
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        registered_at=registered_at,
    )
    root = Path(str(manifest.runtime.get("evaluation_root", "."))).resolve()
    attempt_root = root / "cases" / case.case_id / "attempts" / attempt_id
    if not attempt_root.resolve().is_relative_to(root):
        raise ValueError("Attempt registration escapes evaluation root")
    case_path = attempt_root / "case.json"
    case_payload = _json_value(
        {
            **registration.to_payload(),
            "started_at": registered_at,
            "case": case.to_payload(),
            "recording": manifest.recording,
            "qualification": manifest.qualification,
        }
    )
    if case_path.exists():
        if _load_json(case_path, MAX_CURRENT_ATTEMPT_BYTES) != case_payload:
            raise ValueError("existing Attempt registration mismatch")
    else:
        media.atomic_write_json(case_path, case_payload)
    locator = root / "cases" / case.case_id / "current_attempt.json"
    media.atomic_write_json(locator, registration.to_payload())
    return registration


def load_attempt_registration(
    path: Path,
    *,
    evaluation_id: str,
    manifest_id: str,
    case_id: str,
    attempt_id: str,
    max_bytes: int = MAX_CURRENT_ATTEMPT_BYTES,
) -> AttemptRegistration:
    """Strictly load one bounded current-Attempt locator."""
    payload = _load_json(Path(path), max_bytes)
    if set(payload) != _REGISTRATION_FIELDS:
        raise ValueError("Attempt registration fields are invalid")
    if payload.get("schema_version") != CURRENT_ATTEMPT_SCHEMA_VERSION:
        raise ValueError("Attempt registration schema_version is unsupported")
    expected = {
        "evaluation_id": evaluation_id,
        "manifest_id": manifest_id,
        "case_id": case_id,
        "attempt_id": attempt_id,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Attempt registration {name} mismatch")
    registered_at = payload.get("registered_at")
    if not isinstance(registered_at, str):
        raise ValueError("Attempt registration registered_at is invalid")
    _timestamp(registered_at)
    return AttemptRegistration(
        schema_version=CURRENT_ATTEMPT_SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
        case_id=case_id,
        attempt_id=attempt_id,
        registered_at=registered_at,
    )


def load_current_attempt(
    path: Path,
    *,
    evaluation_id: str,
    manifest_id: str,
    case_id: str,
    max_bytes: int = MAX_CURRENT_ATTEMPT_BYTES,
) -> AttemptRegistration:
    """Strictly load a locator whose Attempt identity is inside the payload."""
    payload = _load_json(Path(path), max_bytes)
    if set(payload) != _REGISTRATION_FIELDS:
        raise ValueError("Attempt registration fields are invalid")
    attempt_id = payload.get("attempt_id")
    validate_case_attempt_id(attempt_id)
    return _registration_from_mapping(
        payload,
        evaluation_id=evaluation_id,
        manifest_id=manifest_id,
        case_id=case_id,
        attempt_id=attempt_id,
    )


def load_case_registration(
    path: Path,
    *,
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
    max_bytes: int = MAX_CURRENT_ATTEMPT_BYTES,
) -> AttemptRegistration:
    """Load a locator target and validate all immutable Case registration data."""
    payload = _load_json(Path(path), max_bytes)
    expected_fields = _REGISTRATION_FIELDS | {
        "started_at",
        "case",
        "recording",
        "qualification",
    }
    if set(payload) != expected_fields:
        raise ValueError("Attempt case registration fields are invalid")
    registration = _registration_from_mapping(
        payload,
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
    )
    if (
        payload["started_at"] != registration.registered_at
        or payload["case"] != case.to_payload()
        or payload["recording"] != _json_value(manifest.recording)
        or payload["qualification"] != _json_value(manifest.qualification)
    ):
        raise ValueError("Attempt case registration mismatch")
    return registration


def _registration_from_mapping(
    payload: Mapping[str, Any],
    *,
    evaluation_id: str,
    manifest_id: str,
    case_id: str,
    attempt_id: str,
) -> AttemptRegistration:
    if payload.get("schema_version") != CURRENT_ATTEMPT_SCHEMA_VERSION:
        raise ValueError("Attempt registration schema_version is unsupported")
    for name, expected in {
        "evaluation_id": evaluation_id,
        "manifest_id": manifest_id,
        "case_id": case_id,
        "attempt_id": attempt_id,
    }.items():
        if payload.get(name) != expected:
            raise ValueError(f"Attempt registration {name} mismatch")
    registered_at = payload.get("registered_at")
    if not isinstance(registered_at, str):
        raise ValueError("Attempt registration registered_at is invalid")
    _timestamp(registered_at)
    return AttemptRegistration(
        CURRENT_ATTEMPT_SCHEMA_VERSION,
        evaluation_id,
        manifest_id,
        case_id,
        attempt_id,
        registered_at,
    )


def _load_json(path: Path, max_bytes: int) -> dict[str, Any]:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Attempt registration must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError("Attempt registration byte limit exceeded")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(raw) > max_bytes:
        raise ValueError("Attempt registration byte limit exceeded")
    try:
        payload = json.loads(raw)
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt registration is malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError("Attempt registration must be a JSON object")
    return payload


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registered_at must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("registered_at must be timezone-aware") from exc
    if parsed.utcoffset() is None:
        raise ValueError("registered_at must be timezone-aware")
    return parsed


def _json_value(value: Any) -> Any:
    return json.loads(canonical_json(value))


__all__ = [
    "AttemptRegistration",
    "CURRENT_ATTEMPT_SCHEMA_VERSION",
    "MAX_CURRENT_ATTEMPT_BYTES",
    "load_attempt_registration",
    "load_current_attempt",
    "load_case_registration",
    "register_attempt",
]
