"""Immutable released-checkpoint identities for isolated SmolVLA inference."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")
SMOLVLA_LEROBOT_REVISION = "64b23178d5348609c266250d3e1f511eba4c33ff"


@dataclass(frozen=True, slots=True)
class SmolVLACheckpoint:
    """Pinned checkpoint and the RPent binding it is allowed to serve."""

    repo_id: str
    revision: str
    binding_name: str

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

    def to_payload(self) -> dict[str, str]:
        """Return the stable wire/manifest representation."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "binding_name": self.binding_name,
        }

    @property
    def identity_digest(self) -> str:
        """Hash every immutable checkpoint identity field over canonical JSON."""
        encoded = json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


SMOLVLA_CHECKPOINTS: Mapping[str, SmolVLACheckpoint] = MappingProxyType(
    {
        "robocerebra": SmolVLACheckpoint(
            repo_id="lerobot/smolvla_robocerebra",
            revision="7ff416240ff73bda10a2b5dbd4245f72eaa959d0",
            binding_name="robocerebra-smolvla-v1",
        ),
        "robocasa": SmolVLACheckpoint(
            repo_id="lerobot/smolvla_robocasa",
            revision="38d38bf58e1217092c5a81418fcc8eef1a7c3a6f",
            binding_name="robocasa-smolvla-v1",
        ),
    }
)

SMOLVLA_PROCESSOR_DIGESTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "robocerebra": MappingProxyType(
            {
                "config.json": "88a0eea27345ae7f7e623312df97ba3737a625c219e029ec079109b7a571bc5e",
                "policy_preprocessor.json": "eccf07bf3cc0cdec8b549c1e8465ff87d2a4827fb2b6593844f0029d034a0afb",
                "policy_postprocessor.json": "3b51f092c70c710ce0213ee1b63bf51b4878ec67828dd2d67daf9ef51081a41a",
                "policy_preprocessor_step_5_normalizer_processor.safetensors": "4362b66c6861c40b1fc1e3a41b474a6ee5850a240ee93260c70e09ce1b04ee5a",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "4362b66c6861c40b1fc1e3a41b474a6ee5850a240ee93260c70e09ce1b04ee5a",
            }
        ),
        "robocasa": MappingProxyType(
            {
                "config.json": "4887a2d06c915af3acd8516e914b9b4200477c2e8e899676d55f6c83316b1834",
                "policy_preprocessor.json": "85629923264df28c5ff4e339dd9a7965f6db7960f9dcd69899da103d6adb3139",
                "policy_postprocessor.json": "293bd851e37ae71678b45c7e540240815d90fd01fe674833b1d29b8ab1ae9ad2",
                "policy_preprocessor_step_5_normalizer_processor.safetensors": "fa18666b4621709db49c7d29108801e4ddd11c666609323413148a6c2328a620",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "fa18666b4621709db49c7d29108801e4ddd11c666609323413148a6c2328a620",
            }
        ),
    }
)


__all__ = [
    "SMOLVLA_CHECKPOINTS",
    "SMOLVLA_LEROBOT_REVISION",
    "SMOLVLA_PROCESSOR_DIGESTS",
    "SmolVLACheckpoint",
]
