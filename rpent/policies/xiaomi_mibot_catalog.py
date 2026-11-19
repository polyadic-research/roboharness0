"""Immutable Xiaomi MiBoT checkpoint and official-source identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")

XIAOMI_OFFICIAL_SOURCE_REVISION = "556cca33963a2b36d835a40374c3b4c8eef68401"


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTCheckpoint:
    """Pinned Xiaomi checkpoint and the RPent binding it may serve."""

    repo_id: str
    revision: str
    binding_name: str
    robot_type: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repo_id, str)
            or not self.repo_id.strip()
            or any(character.isspace() for character in self.repo_id)
            or "/" not in self.repo_id
        ):
            raise ValueError("repo_id must be a non-empty owner/model identifier")
        if not isinstance(self.revision, str) or not _COMMIT_REVISION.fullmatch(
            self.revision
        ):
            raise ValueError("revision must be a full commit revision")
        if not isinstance(self.binding_name, str) or not self.binding_name.strip():
            raise ValueError("binding_name must be non-empty")
        if not isinstance(self.robot_type, str) or not self.robot_type.strip():
            raise ValueError("robot_type must be non-empty")

    def to_payload(self) -> dict[str, str]:
        """Return the stable wire and manifest representation."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "binding_name": self.binding_name,
            "robot_type": self.robot_type,
        }

    @property
    def identity_digest(self) -> str:
        """Hash every immutable identity field over canonical JSON."""
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


XIAOMI_MIBOT_CHECKPOINT = XiaomiMiBoTCheckpoint(
    repo_id="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
    revision="0d1aa76d0d82debc9b611e4d1e231096434d5be4",
    binding_name="robocasa-xiaomi-mibot-v1",
    robot_type="robocasa365",
)


@dataclass(frozen=True, slots=True)
class XiaomiMiBoTPolicySpec:
    """Official inference contract, independent of the planning profile."""

    observation_history: int = 4
    observation_interval: int = 2
    crop_ratio: float = 0.95
    generated_chunk_size: int = 16
    default_actions_per_request: int = 16
    native_action_shape: tuple[int, ...] = (12,)
    input_camera_ids: tuple[str, ...] = (
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    )
    decode_order: str = "processor.decode_action_then_slice_12"
    preset_name: str = "robocasa365-pandaomron-xiaomi-mibot-v1"
    instruction_suffix: str = " /no_cot"
    default_dtype: str = "bfloat16"


XIAOMI_MIBOT_POLICY_SPEC = XiaomiMiBoTPolicySpec()

# SDS two-4090 D01/D04 smoke, 2026-10-11: simulator + policy peaks were
# 12383/12200 MiB. Round the maximum up to a GiB and add one GiB headroom.
# allocate_replicas additionally reserves its configured 1536 MiB per GPU.
XIAOMI_REPLICA_MEMORY_MIB = 14336


__all__ = [
    "XIAOMI_MIBOT_CHECKPOINT",
    "XIAOMI_MIBOT_POLICY_SPEC",
    "XIAOMI_OFFICIAL_SOURCE_REVISION",
    "XiaomiMiBoTCheckpoint",
]
