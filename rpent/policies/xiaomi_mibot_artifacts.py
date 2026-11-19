"""Fail-closed validation for the released Xiaomi MiBoT checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XiaomiMiBoTCheckpoint,
)

_REQUIRED_FILES = (
    "config.json",
    "configuration_mibot.py",
    "modeling_mibot.py",
    "processing_mibot.py",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


class ArtifactValidationError(ValueError):
    """Raised when checkpoint files or serialized identities drift."""


@dataclass(frozen=True, slots=True)
class XiaomiArtifactManifest:
    """Static identity extracted from a validated Xiaomi snapshot."""

    checkpoint: XiaomiMiBoTCheckpoint
    model_class: str
    processor_class: str
    state_dim: int
    action_dim: int
    state_length: int
    file_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "file_digests", MappingProxyType(dict(self.file_digests))
        )

    @property
    def identity_digest(self) -> str:
        payload = {
            "checkpoint": self.checkpoint.to_payload(),
            "model_class": self.model_class,
            "processor_class": self.processor_class,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_length": self.state_length,
            "file_digests": dict(self.file_digests),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ArtifactValidationError(
            f"{field} mismatch: expected {expected!r}, actual {actual!r}"
        )


def inspect_xiaomi_artifacts(
    snapshot_path: str | Path,
    checkpoint: XiaomiMiBoTCheckpoint,
) -> XiaomiArtifactManifest:
    """Validate required serialized Xiaomi identities without loading weights."""
    if checkpoint != XIAOMI_MIBOT_CHECKPOINT:
        raise ArtifactValidationError("checkpoint is not the released checkpoint identity")
    snapshot = Path(snapshot_path)
    missing = [name for name in _REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise ArtifactValidationError(f"required artifact files are missing: {missing}")

    config = _read_json(snapshot / "config.json")
    processor = _read_json(snapshot / "processor_config.json")
    preprocessor = _read_json(snapshot / "preprocessor_config.json")
    video_processor = _read_json(snapshot / "video_preprocessor_config.json")
    weights = _read_json(snapshot / "model.safetensors.index.json")

    _require(config.get("architectures"), ["MiBoTForActionGeneration"], "architectures")
    _require(
        config.get("auto_map", {}).get("AutoConfig"),
        "configuration_mibot.MiBoTConfig",
        "AutoConfig remote class",
    )
    _require(
        config.get("auto_map", {}).get("AutoModel"),
        "modeling_mibot.MiBoTForActionGeneration",
        "MiBoTForActionGeneration remote class",
    )
    _require(config.get("model_type"), "mibot", "model_type")
    _require(config.get("state_dim"), 60, "state_dim")
    _require(config.get("action_dim"), 60, "action_dim")
    _require(config.get("state_length"), 4, "state_length")
    _require(processor.get("processor_class"), "MiBotProcessor", "processor_class")
    _require(
        processor.get("auto_map", {}).get("AutoProcessor"),
        "processing_mibot.MiBotProcessor",
        "processing_mibot.MiBotProcessor remote class",
    )
    _require(
        video_processor.get("auto_map", {}).get("AutoProcessor"),
        "processing_mibot.MiBotProcessor",
        "video processor remote class",
    )
    action_config = preprocessor.get("action_config")
    if not isinstance(action_config, dict) or checkpoint.robot_type not in action_config:
        raise ArtifactValidationError(
            f"preprocessor_config.json is missing {checkpoint.robot_type!r}"
        )
    if not isinstance(weights.get("weight_map"), dict) or not weights["weight_map"]:
        raise ArtifactValidationError("model.safetensors.index.json has no weight_map")

    return XiaomiArtifactManifest(
        checkpoint=checkpoint,
        model_class="MiBoTForActionGeneration",
        processor_class="MiBotProcessor",
        state_dim=60,
        action_dim=60,
        state_length=4,
        file_digests={name: _sha256(snapshot / name) for name in _REQUIRED_FILES},
    )


def ensure_xiaomi_snapshot(
    checkpoint: XiaomiMiBoTCheckpoint = XIAOMI_MIBOT_CHECKPOINT,
) -> XiaomiArtifactManifest:
    """Download the immutable snapshot and validate its static contract."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(repo_id=checkpoint.repo_id, revision=checkpoint.revision)
    )
    if checkpoint.revision not in snapshot.parts:
        raise ArtifactValidationError(
            "snapshot_download did not return the requested revision snapshot"
        )
    return inspect_xiaomi_artifacts(snapshot, checkpoint)


__all__ = [
    "ArtifactValidationError",
    "XiaomiArtifactManifest",
    "ensure_xiaomi_snapshot",
    "inspect_xiaomi_artifacts",
]
