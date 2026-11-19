"""Strict RoboCasa leaderboard submission metadata and exporter."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robots.robocasa.dependencies import RoboCasaDependencyReport
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import OfficialMetrics

_MAX_METADATA_BYTES = 64 * 1024
_FIELDS = {
    "model_name",
    "policy_family",
    "date",
    "submission_source",
    "code_url",
    "commit_hash",
    "paper_link",
    "open_source",
    "wandb",
    "training_config",
    "notes",
}
_DERIVED_FIELDS = {
    "robocasa_version",
    "atomic_seen_success",
    "composite_seen_success",
    "composite_unseen_success",
    "checkpoint_url",
}
LEADERBOARD_SUBMISSION_SCHEMA = MappingProxyType(
    {
        "type": "object",
        "additionalProperties": False,
        "required": tuple(sorted((_FIELDS | _DERIVED_FIELDS) - {"wandb"})),
        "properties": {
            **{
                name: {"type": "string", "minLength": 1}
                for name in (
                    "model_name",
                    "policy_family",
                    "date",
                    "robocasa_version",
                    "code_url",
                    "commit_hash",
                    "checkpoint_url",
                    "paper_link",
                    "wandb",
                    "notes",
                )
            },
            "submission_source": {
                "type": "string",
                "enum": ("external", "internal"),
            },
            "open_source": {"type": "string", "enum": ("yes", "no")},
            "atomic_seen_success": {"type": "number"},
            "composite_seen_success": {"type": "number"},
            "composite_unseen_success": {"type": "number"},
            "training_config": {
                "type": "object",
                "additionalProperties": False,
                "required": ("batch_size", "num_training_steps"),
                "properties": {
                    "batch_size": {"type": "integer", "minimum": 1},
                    "num_training_steps": {"type": "integer", "minimum": 1},
                },
            },
        },
    }
)


@dataclass(frozen=True, slots=True)
class SubmissionMetadata:
    model_name: str
    policy_family: str
    date: str
    submission_source: str
    code_url: str
    commit_hash: str
    paper_link: str
    open_source: str
    training_config: Mapping[str, int]
    notes: str
    wandb: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_name",
            "policy_family",
            "date",
            "submission_source",
            "code_url",
            "commit_hash",
            "paper_link",
            "notes",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
                raise ValueError(f"submission metadata {name} is invalid")
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date) is None:
                raise ValueError
            date.fromisoformat(self.date)
        except ValueError as exc:
            raise ValueError("submission metadata date must be YYYY-MM-DD") from exc
        if self.submission_source not in {"external", "internal"}:
            raise ValueError("submission_source is not an allowed value")
        if self.open_source not in {"yes", "no"}:
            raise ValueError("open_source must be 'yes' or 'no'")
        training = dict(self.training_config)
        if set(training) != {"batch_size", "num_training_steps"} or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in training.values()
        ):
            raise ValueError("training_config must contain positive integer fields")
        object.__setattr__(self, "training_config", MappingProxyType(training))
        if self.wandb is not None and (
            not isinstance(self.wandb, str) or not self.wandb.strip()
        ):
            raise ValueError("wandb must be a non-empty string or omitted")

    @classmethod
    def from_payload(cls, payload: Any) -> SubmissionMetadata:
        if not isinstance(payload, Mapping):
            raise ValueError("submission metadata must be a JSON object")
        unknown = set(payload).difference(_FIELDS)
        if unknown:
            raise ValueError(f"unknown submission metadata fields: {sorted(unknown)}")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise ValueError(f"invalid submission metadata fields: {exc}") from exc

    @classmethod
    def from_json_path(cls, path: str | Path) -> SubmissionMetadata:
        target = Path(path)
        data = target.read_bytes()
        if len(data) > _MAX_METADATA_BYTES:
            raise ValueError("submission metadata JSON exceeds 64 KiB")
        try:
            return cls.from_payload(json.loads(data))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("submission metadata JSON is invalid") from exc

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "policy_family": self.policy_family,
            "date": self.date,
            "submission_source": self.submission_source,
            "code_url": self.code_url,
            "commit_hash": self.commit_hash,
            "paper_link": self.paper_link,
            "open_source": self.open_source,
            "training_config": dict(self.training_config),
            "notes": self.notes,
        }
        if self.wandb is not None:
            payload["wandb"] = self.wandb
        return payload


def build_leaderboard_submission(
    manifest: EvaluationManifest,
    metrics: OfficialMetrics,
    metadata: SubmissionMetadata,
    dependencies: RoboCasaDependencyReport,
) -> dict[str, Any]:
    if not metrics.is_complete:
        raise ValueError("partial official metrics cannot be submitted")
    checkpoint = manifest.policy.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("manifest checkpoint identity is unavailable")
    split = metrics.split_success_rates
    payload = {
        **metadata.to_payload(),
        "robocasa_version": dependencies.robocasa_version,
        "atomic_seen_success": split["atomic_seen"] * 100,
        "composite_seen_success": split["composite_seen"] * 100,
        "composite_unseen_success": split["composite_unseen"] * 100,
        "checkpoint_url": (
            f"https://huggingface.co/{checkpoint['repo_id']}/tree/"
            f"{checkpoint['revision']}"
        ),
    }
    validate_leaderboard_submission(payload)
    return payload


def validate_leaderboard_submission(payload: Any) -> None:
    """Validate the locked upstream leaderboard JSON contract without dependencies."""
    if not isinstance(payload, Mapping):
        raise ValueError("official leaderboard schema requires an object")
    expected = _FIELDS | _DERIVED_FIELDS
    required = expected - {"wandb"}
    if set(payload).difference(expected) or not required.issubset(payload):
        raise ValueError("official leaderboard schema fields do not match")
    strings = required - {
        "atomic_seen_success",
        "composite_seen_success",
        "composite_unseen_success",
        "training_config",
    }
    if any(not isinstance(payload[name], str) or not payload[name] for name in strings):
        raise ValueError("official leaderboard schema requires non-empty strings")
    if payload["submission_source"] not in {"external", "internal"}:
        raise ValueError("official leaderboard schema submission_source is invalid")
    if payload["open_source"] not in {"yes", "no"}:
        raise ValueError("official leaderboard schema open_source is invalid")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["date"]) is None:
            raise ValueError
        date.fromisoformat(payload["date"])
    except ValueError as exc:
        raise ValueError("official leaderboard schema date is invalid") from exc
    for name in (
        "atomic_seen_success",
        "composite_seen_success",
        "composite_unseen_success",
    ):
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 100
        ):
            raise ValueError("official leaderboard schema score is invalid")
    training = payload["training_config"]
    if (
        not isinstance(training, Mapping)
        or set(training)
        != {
            "batch_size",
            "num_training_steps",
        }
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in training.values()
        )
    ):
        raise ValueError("official leaderboard schema training_config is invalid")
    if "wandb" in payload and (
        not isinstance(payload["wandb"], str) or not payload["wandb"]
    ):
        raise ValueError("official leaderboard schema wandb is invalid")


def write_submission_bundle(
    root: Path,
    manifest: EvaluationManifest,
    metrics: OfficialMetrics,
    qualification: Any,
    dependencies: RoboCasaDependencyReport,
) -> None:
    if not qualification.accepted or not metrics.is_complete:
        raise ValueError("only qualified terminal runs may be exported")
    raw = manifest.qualification.get("submission_metadata")
    metadata = SubmissionMetadata.from_payload(raw)
    payload = build_leaderboard_submission(manifest, metrics, metadata, dependencies)
    reproduction = _reproduction(manifest, metrics, metadata, dependencies)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    staged = _stage_bytes(
        root,
        {
            "reproduction.md": reproduction.encode("utf-8"),
            "leaderboard_submission.json": encoded,
        },
    )
    _publish_staged(root, staged)


def _reproduction(manifest, metrics, metadata, dependencies) -> str:
    checkpoint = manifest.policy["checkpoint"]
    return (
        "# RoboCasa365 reproduction\n\n"
        f"- Model: {metadata.model_name}\n"
        f"- Overall native success: {metrics.overall_success_rate * 100:.6g}\n"
        f"- RPent commit: {manifest.rpent_commit}\n"
        f"- VLA checkpoint: {checkpoint['repo_id']}@{checkpoint['revision']}\n"
        f"- RoboCasa: {dependencies.robocasa_version} ({dependencies.robocasa_commit})\n"
        f"- Manifest: `{manifest.manifest_id}`\n"
    )


def _stage_bytes(root: Path, values: Mapping[str, bytes]) -> dict[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, str] = {}
    try:
        for name, value in values.items():
            descriptor, temporary = tempfile.mkstemp(dir=root, prefix=f".{name}.")
            staged[root / name] = temporary
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for temporary in staged.values():
            Path(temporary).unlink(missing_ok=True)
        raise
    return staged


def _publish_staged(root: Path, staged: Mapping[Path, str]) -> None:
    previous: dict[Path, bytes | None] = {}
    published: list[Path] = []
    for target in staged:
        if target.exists() and target.resolve() != target:
            raise ValueError("submission bundle target cannot be a symlink")
        previous[target] = target.read_bytes() if target.exists() else None
    try:
        for target, temporary in staged.items():
            os.replace(temporary, target)
            published.append(target)
        _fsync_directory(root)
    except BaseException:
        for target in published:
            value = previous[target]
            if value is None:
                target.unlink(missing_ok=True)
            else:
                with target.open("wb") as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
        _fsync_directory(root)
        raise
    finally:
        for temporary in staged.values():
            Path(temporary).unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SubmissionMetadata",
    "LEADERBOARD_SUBMISSION_SCHEMA",
    "build_leaderboard_submission",
    "validate_leaderboard_submission",
    "write_submission_bundle",
]
