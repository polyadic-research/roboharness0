"""Validate pinned LeRobot processor artifacts against RPent bindings.

This module is imported only by the isolated policy process and artifact
qualification tests. The coordinator does not import Hugging Face or LeRobot.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from rpent.episode.binding import PolicyBinding
from rpent.policies.smolvla_catalog import (
    SMOLVLA_CHECKPOINTS,
    SmolVLACheckpoint,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
# Keep the evaluation inputs aligned with the pinned checkpoint's serialized
# preprocessor. Generic LeRobot examples are not part of this checkpoint contract.
_ROBOCASA_EVALUATION_RENAME_MAP = MappingProxyType(
    {
        "observation.images.robot0_agentview_left": "observation.images.camera1",
        "observation.images.robot0_agentview_right": "observation.images.camera2",
        "observation.images.robot0_eye_in_hand": "observation.images.camera3",
    }
)


class ArtifactValidationError(ValueError):
    """Raised when pinned checkpoint artifacts and a binding disagree."""


@dataclass(frozen=True, slots=True)
class CheckpointArtifactContract:
    """Small immutable contract extracted from one pinned checkpoint."""

    repo_id: str
    revision: str
    binding_name: str
    rename_map: Mapping[str, str]
    evaluation_rename_map: Mapping[str, str]
    normalizer_shapes: Mapping[str, tuple[int, ...]]
    action_shape: tuple[int, ...]
    empty_camera_count: int
    allowed_missing_policy_inputs: frozenset[str]
    model_config_state_shape: tuple[int, ...]
    file_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rename_map", MappingProxyType(dict(self.rename_map)))
        object.__setattr__(
            self,
            "evaluation_rename_map",
            MappingProxyType(dict(self.evaluation_rename_map)),
        )
        object.__setattr__(
            self,
            "normalizer_shapes",
            MappingProxyType(
                {key: tuple(shape) for key, shape in self.normalizer_shapes.items()}
            ),
        )
        object.__setattr__(
            self, "file_digests", MappingProxyType(dict(self.file_digests))
        )
        object.__setattr__(self, "action_shape", tuple(self.action_shape))
        object.__setattr__(
            self, "model_config_state_shape", tuple(self.model_config_state_shape)
        )
        object.__setattr__(
            self,
            "allowed_missing_policy_inputs",
            frozenset(self.allowed_missing_policy_inputs),
        )
        if self.empty_camera_count < 0:
            raise ValueError("empty_camera_count must be non-negative")
        missing_files = set(_ARTIFACT_FILES) - set(self.file_digests)
        if missing_files:
            raise ValueError(f"artifact digests missing {sorted(missing_files)}")
        for filename, digest in self.file_digests.items():
            if not _DIGEST.fullmatch(digest):
                raise ValueError(f"invalid artifact digest for {filename}")

    @property
    def checkpoint(self) -> SmolVLACheckpoint:
        return SmolVLACheckpoint(
            self.repo_id,
            self.revision,
            self.binding_name,
        )

    @property
    def identity_digest(self) -> str:
        payload = {
            "checkpoint": self.checkpoint.to_payload(),
            "rename_map": dict(self.rename_map),
            "evaluation_rename_map": dict(self.evaluation_rename_map),
            "normalizer_shapes": {
                key: list(value) for key, value in self.normalizer_shapes.items()
            },
            "action_shape": list(self.action_shape),
            "empty_camera_count": self.empty_camera_count,
            "allowed_missing_policy_inputs": sorted(
                self.allowed_missing_policy_inputs
            ),
            "file_digests": dict(self.file_digests),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate_binding(self, binding: PolicyBinding) -> None:
        """Fail closed when the RPent binding drifts from processor artifacts."""
        released = next(
            (
                item
                for item in SMOLVLA_CHECKPOINTS.values()
                if item.repo_id == self.repo_id
            ),
            None,
        )
        if released is None or released.revision != self.revision:
            self._error(
                "checkpoint provenance",
                "revision",
                released.revision if released is not None else "a released checkpoint",
                self.revision,
            )
        if binding.name != self.binding_name:
            self._error(
                "policy_preprocessor.json",
                "binding_name",
                self.binding_name,
                binding.name,
            )

        state = next(
            (
                feature
                for feature in binding.inputs
                if feature.policy_key == "observation.state"
            ),
            None,
        )
        expected_state = self.normalizer_shapes.get("observation.state")
        actual_state = state.shape if state is not None else None
        if actual_state != expected_state:
            self._error(
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "observation.state",
                expected_state,
                actual_state,
            )
        if binding.action_shape != self.action_shape:
            self._error(
                "policy_postprocessor.json",
                "action_shape",
                self.action_shape,
                binding.action_shape,
            )
        if binding.action_dtype != np.dtype(np.float32):
            self._error(
                "policy_postprocessor.json",
                "action_dtype",
                "float32",
                binding.action_dtype.name,
            )

        actual_rename = {
            _processor_image_key(feature.native_key): feature.policy_key
            for feature in binding.inputs
            if feature.policy_key.startswith("observation.images.")
        }
        if actual_rename != dict(self.evaluation_rename_map):
            self._error(
                "LeRobot evaluation contract",
                "rename_map",
                dict(self.evaluation_rename_map),
                actual_rename,
            )
        if (
            binding.allowed_missing_policy_inputs
            != self.allowed_missing_policy_inputs
        ):
            self._error(
                "config.json",
                "allowed_missing_policy_inputs",
                sorted(self.allowed_missing_policy_inputs),
                sorted(binding.allowed_missing_policy_inputs),
            )
        if self.empty_camera_count != len(self.allowed_missing_policy_inputs):
            self._error(
                "config.json",
                "empty_cameras",
                self.empty_camera_count,
                len(self.allowed_missing_policy_inputs),
            )

    def _error(
        self,
        filename: str,
        field: str,
        expected: Any,
        actual: Any,
    ) -> None:
        raise ArtifactValidationError(
            f"checkpoint {self.repo_id}@{self.revision} artifact {filename} "
            f"binding {self.binding_name} field {field}: "
            f"expected {expected!r}, actual {actual!r}"
        )


def binding_identity(binding: PolicyBinding) -> str:
    """Return a stable identity for the complete RPent feature/action binding."""
    payload = {
        "name": binding.name,
        "inputs": [
            {
                "native_key": feature.native_key,
                "policy_key": feature.policy_key,
                "shape": list(feature.shape),
                "dtype": feature.dtype.name,
            }
            for feature in binding.inputs
        ],
        "action_shape": list(binding.action_shape),
        "action_dtype": binding.action_dtype.name,
        "action_low": binding.action_low,
        "action_high": binding.action_high,
        "allowed_missing_policy_inputs": sorted(
            binding.allowed_missing_policy_inputs
        ),
        "clip_actions_to_bounds": binding.clip_actions_to_bounds,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _processor_image_key(native_key: str) -> str:
    if native_key.startswith("observation.images."):
        return native_key
    return f"observation.images.{native_key}"


def load_fixture_contract(name: str) -> CheckpointArtifactContract:
    """Load one provenance-marked, weight-free checkpoint test fixture."""
    if name not in SMOLVLA_CHECKPOINTS:
        raise ValueError(f"unknown SmolVLA fixture: {name!r}")
    root = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "smolvla"
        / name
    )
    processor = _read_json(root / "policy_preprocessor.json")
    shapes = _read_json(root / "normalizer_shapes.json")
    checkpoint = SMOLVLA_CHECKPOINTS[name]
    for payload, filename in (
        (processor, "policy_preprocessor.json"),
        (shapes, "normalizer_shapes.json"),
    ):
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"fixture {filename} missing provenance")
        if (
            provenance.get("source_repo_id") != checkpoint.repo_id
            or provenance.get("revision") != checkpoint.revision
            or not provenance.get("original_filename")
            or not _DIGEST.fullmatch(str(provenance.get("sha256", "")))
            or not provenance.get("extraction_date")
        ):
            raise ValueError(f"fixture {filename} has invalid provenance")
    contract = CheckpointArtifactContract(
        repo_id=checkpoint.repo_id,
        revision=checkpoint.revision,
        binding_name=checkpoint.binding_name,
        rename_map=processor["rename_map"],
        evaluation_rename_map=processor.get(
            "evaluation_rename_map", processor["rename_map"]
        ),
        normalizer_shapes={
            key: tuple(value)
            for key, value in shapes["normalizer_shapes"].items()
        },
        action_shape=tuple(processor["declared_action_shape"]),
        empty_camera_count=int(processor["empty_camera_count"]),
        allowed_missing_policy_inputs=frozenset(
            processor["allowed_missing_policy_inputs"]
        ),
        model_config_state_shape=tuple(processor["model_config_state_shape"]),
        file_digests=processor["file_digests"],
    )
    if (
        shapes["provenance"]["sha256"]
        != contract.file_digests[
            "policy_preprocessor_step_5_normalizer_processor.safetensors"
        ]
    ):
        raise ValueError("normalizer fixture digest does not match file_digests")
    if (
        processor["provenance"]["sha256"]
        != contract.file_digests["policy_preprocessor.json"]
    ):
        raise ValueError("preprocessor fixture digest does not match file_digests")
    if contract.normalizer_shapes.get("action") != contract.action_shape:
        raise ValueError("normalizer action shape does not match declared action shape")
    return contract


DownloadFile = Callable[[str, str, str], str]


def load_checkpoint_artifacts(
    checkpoint: SmolVLACheckpoint,
    binding: PolicyBinding,
    *,
    download_file: DownloadFile | None = None,
) -> CheckpointArtifactContract:
    """Download exact pinned files, inspect metadata, and validate a binding."""
    released = SMOLVLA_CHECKPOINTS.get(
        next(
            (
                name
                for name, candidate in SMOLVLA_CHECKPOINTS.items()
                if candidate.repo_id == checkpoint.repo_id
            ),
            "",
        )
    )
    if released != checkpoint:
        expected_revision = (
            released.revision if released is not None else "a released checkpoint"
        )
        raise ArtifactValidationError(
            f"checkpoint {checkpoint.repo_id}@{checkpoint.revision} artifact "
            f"checkpoint provenance binding {binding.name} field revision: "
            f"expected {expected_revision!r}, actual {checkpoint.revision!r}"
        )
    if download_file is None:
        from huggingface_hub import hf_hub_download

        def download_file(repo_id: str, filename: str, revision: str) -> str:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )

    paths: dict[str, Path] = {}
    raw: dict[str, bytes] = {}
    for filename in _ARTIFACT_FILES:
        try:
            path = Path(
                download_file(checkpoint.repo_id, filename, checkpoint.revision)
            )
            content = path.read_bytes()
        except Exception as exc:
            raise ArtifactValidationError(
                f"checkpoint {checkpoint.repo_id}@{checkpoint.revision} artifact "
                f"{filename} binding {binding.name}: expected accessible pinned "
                f"artifact, actual {type(exc).__name__}: {exc}"
            ) from exc
        paths[filename] = path
        raw[filename] = content
    digests = {
        filename: hashlib.sha256(content).hexdigest()
        for filename, content in raw.items()
    }
    config = _decode_json(raw["config.json"], "config.json")
    preprocessor = _decode_json(
        raw["policy_preprocessor.json"], "policy_preprocessor.json"
    )
    postprocessor = _decode_json(
        raw["policy_postprocessor.json"], "policy_postprocessor.json"
    )
    rename_step = _processor_step(preprocessor, "rename_observations_processor")
    normalizer_step = _processor_step(preprocessor, "normalizer_processor")
    unnormalizer_step = _processor_step(postprocessor, "unnormalizer_processor")
    tensor_shapes = _safetensor_shapes(
        raw["policy_preprocessor_step_5_normalizer_processor.safetensors"]
    )
    state_shape = tensor_shapes.get("observation.state.mean")
    action_shape = tensor_shapes.get("action.mean")
    declared_action = tuple(
        unnormalizer_step["config"]["features"]["action"]["shape"]
    )
    if action_shape != declared_action:
        raise ArtifactValidationError(
            f"checkpoint {checkpoint.repo_id}@{checkpoint.revision} artifact "
            "policy_postprocessor.json binding "
            f"{binding.name} field action_shape: expected {action_shape!r}, "
            f"actual {declared_action!r}"
        )
    configured_inputs = config.get("input_features", {})
    rename_map = dict(rename_step.get("config", {}).get("rename_map", {}))
    renamed_policy_keys = set(rename_map.values())
    missing_policy_inputs = frozenset(
        key
        for key, feature in configured_inputs.items()
        if isinstance(feature, Mapping)
        and feature.get("type") == "VISUAL"
        and not key.startswith("observation.images.empty_camera_")
        and key not in renamed_policy_keys
    )
    contract = CheckpointArtifactContract(
        repo_id=checkpoint.repo_id,
        revision=checkpoint.revision,
        binding_name=checkpoint.binding_name,
        rename_map=rename_map,
        evaluation_rename_map=(
            _ROBOCASA_EVALUATION_RENAME_MAP
            if checkpoint == SMOLVLA_CHECKPOINTS["robocasa"]
            else rename_map
        ),
        normalizer_shapes={
            "observation.state": state_shape,
            "action": action_shape,
        },
        action_shape=declared_action,
        empty_camera_count=int(config.get("empty_cameras", 0)),
        allowed_missing_policy_inputs=missing_policy_inputs,
        model_config_state_shape=tuple(
            configured_inputs["observation.state"]["shape"]
        ),
        file_digests=digests,
    )
    # Ensure the named normalizer step actually points at the inspected file.
    if normalizer_step.get("state_file") != (
        "policy_preprocessor_step_5_normalizer_processor.safetensors"
    ):
        contract._error(
            "policy_preprocessor.json",
            "normalizer state_file",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            normalizer_step.get("state_file"),
        )
    contract.validate_binding(binding)
    return contract


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_json(raw: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact {filename}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON artifact {filename} must be an object")
    return value


def _processor_step(payload: Mapping[str, Any], registry_name: str) -> dict[str, Any]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ArtifactValidationError("processor artifact missing steps")
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("registry_name") == registry_name
    ]
    if len(matches) != 1:
        raise ArtifactValidationError(
            f"processor artifact expected one {registry_name!r} step, "
            f"actual {len(matches)}"
        )
    return matches[0]


def _safetensor_shapes(raw: bytes) -> dict[str, tuple[int, ...]]:
    if len(raw) < 8:
        raise ArtifactValidationError("safetensors artifact is truncated")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size <= 0 or 8 + header_size > len(raw):
        raise ArtifactValidationError("safetensors artifact has invalid header size")
    header = _decode_json(raw[8 : 8 + header_size], "safetensors header")
    shapes: dict[str, tuple[int, ...]] = {}
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("shape"), list
        ):
            raise ArtifactValidationError(
                f"safetensors tensor {key!r} has invalid shape metadata"
            )
        shapes[key] = tuple(int(value) for value in metadata["shape"])
    return shapes


__all__ = [
    "ArtifactValidationError",
    "CheckpointArtifactContract",
    "binding_identity",
    "load_checkpoint_artifacts",
    "load_fixture_contract",
]
