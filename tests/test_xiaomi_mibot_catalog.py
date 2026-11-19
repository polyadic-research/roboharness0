from __future__ import annotations

from dataclasses import replace

import pytest

from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_OFFICIAL_SOURCE_REVISION,
    XiaomiMiBoTCheckpoint,
)


def test_released_xiaomi_checkpoint_is_immutable_and_pinned() -> None:
    assert XIAOMI_OFFICIAL_SOURCE_REVISION == (
        "556cca33963a2b36d835a40374c3b4c8eef68401"
    )
    assert XIAOMI_MIBOT_CHECKPOINT.repo_id == (
        "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
    )
    assert XIAOMI_MIBOT_CHECKPOINT.revision == (
        "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
    )
    assert XIAOMI_MIBOT_CHECKPOINT.binding_name == "robocasa-xiaomi-mibot-v1"
    assert XIAOMI_MIBOT_CHECKPOINT.robot_type == "robocasa365"
    assert len(XIAOMI_MIBOT_CHECKPOINT.identity_digest) == 64


@pytest.mark.parametrize("revision", ["main", "v1", "A" * 40, "abc", "g" * 40])
def test_xiaomi_checkpoint_rejects_symbolic_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="full commit revision"):
        XiaomiMiBoTCheckpoint("owner/model", revision, "binding", "robot")


def test_xiaomi_checkpoint_payload_and_digest_cover_every_field() -> None:
    checkpoint = XIAOMI_MIBOT_CHECKPOINT

    assert checkpoint.to_payload() == {
        "repo_id": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
        "revision": "0d1aa76d0d82debc9b611e4d1e231096434d5be4",
        "binding_name": "robocasa-xiaomi-mibot-v1",
        "robot_type": "robocasa365",
    }
    variants = (
        replace(checkpoint, repo_id="other/model"),
        replace(checkpoint, revision="0" * 40),
        replace(checkpoint, binding_name="other-binding"),
        replace(checkpoint, robot_type="other-robot"),
    )
    assert all(item.identity_digest != checkpoint.identity_digest for item in variants)


@pytest.mark.parametrize(
    ("repo_id", "binding_name", "robot_type"),
    [
        ("", "binding", "robot"),
        ("owner/model", "", "robot"),
        ("owner model", "binding", "robot"),
        ("owner/model", "binding", ""),
    ],
)
def test_xiaomi_checkpoint_rejects_invalid_identity(
    repo_id: str, binding_name: str, robot_type: str
) -> None:
    with pytest.raises(ValueError):
        XiaomiMiBoTCheckpoint(repo_id, "0" * 40, binding_name, robot_type)
