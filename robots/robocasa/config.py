"""Immutable configuration for one RoboCasa365 runtime case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.policies.robocasa_pi05_catalog import RoboCasaPi05Checkpoint
from rpent.policies.smolvla_catalog import SmolVLACheckpoint
from rpent.policies.xiaomi_mibot_catalog import XiaomiMiBoTCheckpoint

OBJECT_REGISTRIES = ("lightwheel", "objaverse", "aigen")
_ALLOWED_REGISTRIES = frozenset(OBJECT_REGISTRIES)


def parse_object_registries(value: Any) -> tuple[str, ...]:
    """Parse RPent's comma-list and LeRobot-compatible bracket syntax."""
    if isinstance(value, (tuple, list)):
        parts = tuple(str(item).strip() for item in value)
    else:
        rendered = str(value).strip()
        if rendered.startswith("[") and rendered.endswith("]"):
            rendered = rendered[1:-1]
        parts = tuple(part.strip() for part in rendered.split(","))
    if not parts or any(not part for part in parts):
        raise ValueError("RoboCasa object registries contain an empty entry")
    if len(parts) != len(set(parts)):
        raise ValueError("RoboCasa object registries contain duplicates")
    unknown = set(parts).difference(_ALLOWED_REGISTRIES)
    if unknown:
        raise ValueError(f"unknown RoboCasa object registries: {sorted(unknown)}")
    return parts


@dataclass(frozen=True, slots=True)
class RoboCasaConfig:
    task: str
    seed: int
    checkpoint: SmolVLACheckpoint | RoboCasaPi05Checkpoint | XiaomiMiBoTCheckpoint
    object_registries: tuple[str, ...]
    env_endpoint: str | None
    policy_endpoint: str | None
    policy_device: str
    policy_dtype: str
    safety_ceiling: int | None
    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    dashboard_state: Any | None
    task_desc: dict[str, Any]
    vla: str = "smolvla"
    control_hz: int = 20
    image_size: int = 256
    require_environment_success: bool = True


__all__ = ["OBJECT_REGISTRIES", "RoboCasaConfig", "parse_object_registries"]
