from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.policies.xiaomi_mibot_artifacts import (
    ArtifactValidationError,
    ensure_xiaomi_snapshot,
    inspect_xiaomi_artifacts,
)
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XiaomiMiBoTCheckpoint,
)

REQUIRED_FILES = (
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


def _snapshot(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payloads = {
        "config.json": {
            "architectures": ["MiBoTForActionGeneration"],
            "auto_map": {
                "AutoConfig": "configuration_mibot.MiBoTConfig",
                "AutoModel": "modeling_mibot.MiBoTForActionGeneration",
            },
            "model_type": "mibot",
            "state_dim": 60,
            "action_dim": 60,
            "state_length": 4,
        },
        "processor_config.json": {
            "processor_class": "MiBotProcessor",
            "auto_map": {"AutoProcessor": "processing_mibot.MiBotProcessor"},
        },
        "preprocessor_config.json": {
            "processor_class": "MiBotProcessor",
            "action_config": {
                "robocasa365": {
                    "mean": [[0.0] * 60 for _ in range(16)],
                    "std": [[1.0] * 60 for _ in range(16)],
                }
            },
        },
        "video_preprocessor_config.json": {
            "processor_class": "MiBotProcessor",
            "auto_map": {"AutoProcessor": "processing_mibot.MiBotProcessor"},
            "do_resize": False,
            "min_frames": 4,
        },
        "model.safetensors.index.json": {
            "metadata": {"total_size": 1},
            "weight_map": {"weight": "model-00001-of-00001.safetensors"},
        },
    }
    for filename in REQUIRED_FILES:
        content = payloads.get(filename, f"# official {filename}\n")
        target = path / filename
        target.write_text(
            json.dumps(content) if isinstance(content, dict) else content,
            encoding="utf-8",
        )
    return path


def test_official_artifacts_produce_hashed_immutable_manifest(tmp_path: Path) -> None:
    manifest = inspect_xiaomi_artifacts(_snapshot(tmp_path), XIAOMI_MIBOT_CHECKPOINT)

    assert manifest.checkpoint == XIAOMI_MIBOT_CHECKPOINT
    assert manifest.model_class == "MiBoTForActionGeneration"
    assert manifest.processor_class == "MiBotProcessor"
    assert manifest.state_dim == 60
    assert manifest.action_dim == 60
    assert manifest.state_length == 4
    assert set(manifest.file_digests) == set(REQUIRED_FILES)
    assert all(len(digest) == 64 for digest in manifest.file_digests.values())
    assert len(manifest.identity_digest) == 64


@pytest.mark.parametrize(
    "missing", ["configuration_mibot.py", "modeling_mibot.py", "processing_mibot.py"]
)
def test_missing_remote_code_is_rejected(tmp_path: Path, missing: str) -> None:
    _snapshot(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(ArtifactValidationError, match=missing):
        inspect_xiaomi_artifacts(tmp_path, XIAOMI_MIBOT_CHECKPOINT)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("architectures", ["OtherModel"], "MiBoTForActionGeneration"),
        ("state_dim", 14, "state_dim"),
        ("action_dim", 12, "action_dim"),
        ("state_length", 1, "state_length"),
    ],
)
def test_wrong_model_contract_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _snapshot(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=message):
        inspect_xiaomi_artifacts(tmp_path, XIAOMI_MIBOT_CHECKPOINT)


def test_wrong_processor_remote_class_is_rejected(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    path = tmp_path / "processor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["auto_map"]["AutoProcessor"] = "other.CustomProcessor"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="processing_mibot.MiBotProcessor"):
        inspect_xiaomi_artifacts(tmp_path, XIAOMI_MIBOT_CHECKPOINT)


def test_missing_robot_action_config_is_rejected(tmp_path: Path) -> None:
    _snapshot(tmp_path)
    path = tmp_path / "preprocessor_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["action_config"] = {}
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="robocasa365"):
        inspect_xiaomi_artifacts(tmp_path, XIAOMI_MIBOT_CHECKPOINT)


def test_only_released_checkpoint_identity_is_accepted(tmp_path: Path) -> None:
    checkpoint = XiaomiMiBoTCheckpoint(
        XIAOMI_MIBOT_CHECKPOINT.repo_id,
        "0" * 40,
        XIAOMI_MIBOT_CHECKPOINT.binding_name,
        XIAOMI_MIBOT_CHECKPOINT.robot_type,
    )

    with pytest.raises(ArtifactValidationError, match="released checkpoint"):
        inspect_xiaomi_artifacts(_snapshot(tmp_path), checkpoint)


def test_snapshot_download_uses_pinned_revision_and_validates_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path / XIAOMI_MIBOT_CHECKPOINT.revision)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    manifest = ensure_xiaomi_snapshot(XIAOMI_MIBOT_CHECKPOINT)

    assert manifest.checkpoint == XIAOMI_MIBOT_CHECKPOINT
    assert calls == [
        {
            "repo_id": XIAOMI_MIBOT_CHECKPOINT.repo_id,
            "revision": XIAOMI_MIBOT_CHECKPOINT.revision,
        }
    ]
