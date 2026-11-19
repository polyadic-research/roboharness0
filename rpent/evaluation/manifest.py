"""Immutable, content-addressed evaluation manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from rpent.evaluation.cases import EpisodeCase
from rpent.planning.manifest import canonical_json, content_hash

EVALUATION_MANIFEST_SCHEMA_VERSION = 3
_SUPPORTED_SCHEMA_VERSIONS = (1, 2, EVALUATION_MANIFEST_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """All immutable inputs needed to reproduce an evaluation schedule."""

    rpent_commit: str
    rpent_dirty: bool
    cli_arguments: Mapping[str, Any]
    environment: Mapping[str, Any]
    simulator: Mapping[str, Any]
    policy: Mapping[str, Any]
    task_selector: str
    cases: tuple[EpisodeCase, ...]
    schedule: Mapping[str, Any]
    contracts: Mapping[str, Any]
    runtime: Mapping[str, Any]
    launch: Mapping[str, Any] = field(default_factory=dict)
    recording: Mapping[str, Any] = field(
        default_factory=lambda: {
            "schema_version": 1,
            "mode": "all",
            "camera_ids": (),
        }
    )
    qualification: Mapping[str, Any] = field(
        default_factory=lambda: {"require_complete_artifacts": False}
    )
    evaluation_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    manifest_id: str = field(init=False)
    source_schema_version: int = field(
        init=False, default=EVALUATION_MANIFEST_SCHEMA_VERSION
    )
    schema_version: int = field(
        default=EVALUATION_MANIFEST_SCHEMA_VERSION,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.rpent_commit, str) or not self.rpent_commit.strip():
            raise ValueError("rpent_commit must be a non-empty string")
        if not isinstance(self.rpent_dirty, bool):
            raise ValueError("rpent_dirty must be a boolean")
        if not isinstance(self.task_selector, str) or not self.task_selector.strip():
            raise ValueError("task_selector must be a non-empty string")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty timestamp")
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise ValueError("evaluation_id must be a non-empty string")
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported evaluation manifest schema_version")
        cases = tuple(self.cases)
        if not cases or not all(isinstance(case, EpisodeCase) for case in cases):
            raise ValueError("cases must contain at least one EpisodeCase")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("manifest cases must have unique case IDs")
        object.__setattr__(self, "cases", cases)
        for name in (
            "cli_arguments",
            "environment",
            "simulator",
            "policy",
            "schedule",
            "contracts",
            "runtime",
            "launch",
            "recording",
            "qualification",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze_mapping(value))
        if self.schema_version >= 2:
            _validate_recording(self.recording)
            _validate_qualification(self.qualification)
        if self.schema_version >= 3:
            _validate_launch(self.launch)
        object.__setattr__(self, "manifest_id", content_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        """Return every identity field; creation time is intentionally absent."""
        payload = {
            "schema_version": self.schema_version,
            "rpent": {"commit": self.rpent_commit, "dirty": self.rpent_dirty},
            "cli_arguments": _plain(self.cli_arguments),
            "environment": _plain(self.environment),
            "simulator": _plain(self.simulator),
            "policy": _plain(self.policy),
            "task_selector": self.task_selector,
            "cases": [case.to_payload() for case in self.cases],
            "schedule": _plain(self.schedule),
            "contracts": _plain(self.contracts),
            "runtime": _plain(
                self.runtime
                if self.schema_version == 1
                else _identity_runtime(self.runtime, self.schema_version)
            ),
        }
        if self.schema_version >= 2:
            payload["recording"] = _plain(self.recording)
            payload["qualification"] = _plain(self.qualification)
        if self.schema_version >= 3:
            payload["launch"] = _plain(self.launch)
        return payload

    @property
    def legacy_identity(self) -> bool:
        """Whether this instance was verified from a schema-v1 manifest."""
        return self.source_schema_version == 1

    def to_payload(self) -> dict[str, Any]:
        """Return the complete persisted representation."""
        persisted = self.identity_payload()
        persisted["runtime"] = _plain(self.runtime)
        return {
            **persisted,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            **(
                {"evaluation_id": self.evaluation_id}
                if self.schema_version >= 2
                else {}
            ),
        }

    def write_new(self, path: str | Path) -> EvaluationManifest:
        """Create once, or verify and reuse an identical existing manifest."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = (canonical_json(self.to_payload()) + "\n").encode("utf-8")
        try:
            with target.open("xb") as stream:
                stream.write(encoded)
        except FileExistsError:
            existing = self.load(target)
            if existing.manifest_id != self.manifest_id:
                raise ValueError(
                    f"refusing to overwrite different manifest at {target}"
                ) from None
            return existing
        return self.load(target)

    @classmethod
    def load(cls, path: str | Path) -> EvaluationManifest:
        """Load and verify manifest and case content identities."""
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load evaluation manifest {target}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("evaluation manifest must be a JSON object")
        schema_version = payload.get("schema_version")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported evaluation manifest schema_version")
        try:
            rpent = payload["rpent"]
            manifest = cls(
                rpent_commit=rpent["commit"],
                rpent_dirty=rpent["dirty"],
                cli_arguments=payload["cli_arguments"],
                environment=payload["environment"],
                simulator=payload["simulator"],
                policy=payload["policy"],
                task_selector=payload["task_selector"],
                cases=tuple(
                    EpisodeCase.from_payload(item) for item in payload["cases"]
                ),
                schedule=payload["schedule"],
                contracts=payload["contracts"],
                runtime=payload["runtime"],
                launch=payload.get("launch", {}) if schema_version >= 3 else {},
                recording=payload.get(
                    "recording",
                    {"schema_version": 1, "mode": "all", "camera_ids": []},
                ),
                qualification=payload.get(
                    "qualification", {"require_complete_artifacts": False}
                ),
                evaluation_id=payload.get("evaluation_id", payload["manifest_id"]),
                created_at=payload["created_at"],
                schema_version=schema_version,
            )
            expected_id = payload["manifest_id"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"evaluation manifest is missing required data: {exc}"
            ) from exc
        if expected_id != manifest.manifest_id:
            raise ValueError(
                "manifest_id mismatch: "
                f"expected {manifest.manifest_id}, actual {expected_id}"
            )
        object.__setattr__(manifest, "source_schema_version", schema_version)
        return manifest


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze(item) for key, item in dict(value).items()}
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _identity_runtime(
    runtime: Mapping[str, Any], schema_version: int
) -> dict[str, Any]:
    excluded = {"evaluation_root", "output_dir"}
    if schema_version >= 3:
        excluded.update({
            "policy_endpoint",
            "simulator_endpoint",
            "sam3_endpoint",
            "notification_endpoint",
        })
    return {
        str(key): value
        for key, value in runtime.items()
        if key not in excluded
    }


def _validate_recording(recording: Mapping[str, Any]) -> None:
    if recording.get("schema_version") != 1:
        raise ValueError("recording.schema_version must be 1")
    mode = recording.get("mode")
    only_inputs = recording.get("llm_inputs_only", False)
    if not isinstance(only_inputs, bool):
        raise ValueError("recording.llm_inputs_only must be a boolean")
    if only_inputs and mode != "none":
        raise ValueError("LLM-only images require control-step recording mode none")
    if mode not in {"none", "primary", "selected", "all"}:
        raise ValueError(f"unknown recording mode: {mode}")
    camera_ids = recording.get("camera_ids")
    if not isinstance(camera_ids, tuple) or not all(
        isinstance(camera_id, str) and camera_id.strip() for camera_id in camera_ids
    ):
        raise ValueError("recording.camera_ids must contain non-empty strings")
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("recording.camera_ids must be unique")
    if mode == "selected" and not camera_ids:
        raise ValueError("selected recording requires camera IDs")
    if mode != "selected" and camera_ids:
        raise ValueError("camera IDs are valid only for selected recording")
    estimate = recording.get("storage_estimate")
    if estimate is not None:
        _validate_storage_estimate(estimate)


def _validate_storage_estimate(estimate: Any) -> None:
    if not isinstance(estimate, Mapping):
        raise ValueError("recording.storage_estimate must be a mapping")
    minimum = estimate.get("minimum_bytes")
    maximum = estimate.get("maximum_bytes")
    if not _non_negative_integer(minimum) or not _non_negative_integer(maximum):
        raise ValueError("recording.storage_estimate byte bounds must be non-negative")
    if minimum > maximum:
        raise ValueError("recording.storage_estimate byte bounds are reversed")
    basis = estimate.get("basis")
    if not isinstance(basis, Mapping):
        raise ValueError("recording.storage_estimate basis must be a mapping")
    for name in (
        "total_horizon_steps",
        "retained_camera_count_min",
        "retained_camera_count_max",
    ):
        if not _non_negative_integer(basis.get(name)):
            raise ValueError(f"recording.storage_estimate basis {name} is invalid")
    for name in ("bytes_per_camera_step_min", "bytes_per_camera_step_max"):
        if not _positive_integer(basis.get(name)):
            raise ValueError(f"recording.storage_estimate basis {name} is invalid")
    if basis["retained_camera_count_min"] > basis["retained_camera_count_max"]:
        raise ValueError("recording.storage_estimate camera bounds are reversed")
    contract_camera_count = basis.get("contract_camera_count")
    if contract_camera_count is not None and not _non_negative_integer(
        contract_camera_count
    ):
        raise ValueError(
            "recording.storage_estimate basis contract_camera_count is invalid"
        )
    if (
        contract_camera_count is not None
        and basis["retained_camera_count_max"] > contract_camera_count
    ):
        raise ValueError("recording.storage_estimate exceeds contract camera count")
    if basis["bytes_per_camera_step_min"] > basis["bytes_per_camera_step_max"]:
        raise ValueError("recording.storage_estimate per-step bounds are reversed")
    if not isinstance(basis.get("note"), str) or not basis["note"].strip():
        raise ValueError("recording.storage_estimate basis note is invalid")
    task_resolution = basis.get("task_resolution")
    if task_resolution is not None and (
        not isinstance(task_resolution, str) or not task_resolution.strip()
    ):
        raise ValueError("recording.storage_estimate basis task_resolution is invalid")


def _non_negative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_qualification(qualification: Mapping[str, Any]) -> None:
    if not isinstance(qualification.get("require_complete_artifacts"), bool):
        raise ValueError("qualification.require_complete_artifacts must be boolean")


def _validate_launch(launch: Mapping[str, Any]) -> None:
    if not launch:
        return
    required = {
        "execution_mode", "preset_version", "runner", "submitted", "effective"
    }
    if set(launch) != required:
        raise ValueError("launch must contain the canonical schema-v3 fields")
    if not isinstance(launch["execution_mode"], str) or not launch[
        "execution_mode"
    ].strip():
        raise ValueError("launch.execution_mode must be a non-empty string")
    if (
        not isinstance(launch["preset_version"], int)
        or isinstance(launch["preset_version"], bool)
        or launch["preset_version"] < 1
    ):
        raise ValueError("launch.preset_version must be a positive integer")
    if not isinstance(launch["runner"], str) or not launch["runner"].strip():
        raise ValueError("launch.runner must be a non-empty string")
    for name in ("submitted", "effective"):
        if not isinstance(launch[name], Mapping):
            raise ValueError(f"launch.{name} must be a mapping")


__all__ = ["EVALUATION_MANIFEST_SCHEMA_VERSION", "EvaluationManifest"]
