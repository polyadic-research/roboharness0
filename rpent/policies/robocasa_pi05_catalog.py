"""Identity of the released RoboCasa OpenPI pi0.5 checkpoint."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

OPENPI_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
PI05_CONFIG = "pi05_pretrain_human300"
PI05_SUBFOLDER = "pi05_pretrain_human300/multitask_learning/75000"
PI05_ACTIONS_PER_REQUEST = 5
PI05_PROTOCOL = "rpent-robocasa-openpi/1"


@dataclass(frozen=True)
class RoboCasaPi05Checkpoint:
    repo_id: str = "robocasa/robocasa365_checkpoints"
    revision: str = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
    binding_name: str = "robocasa-pi05-v1"
    subfolder: str = PI05_SUBFOLDER
    config_name: str = PI05_CONFIG

    def to_payload(self):
        return asdict(self)

    @property
    def identity_digest(self):
        return hashlib.sha256(
            json.dumps(self.to_payload(), sort_keys=True).encode()
        ).hexdigest()


PI05_CHECKPOINT = RoboCasaPi05Checkpoint()


def artifact_manifest():
    return json.loads(
        Path(__file__).with_name("robocasa_pi05_artifacts.json").read_text()
    )


def verify_pi05_artifacts(directory):
    """Check every inference file against the immutable HF git/LFS object IDs."""
    root = Path(directory).expanduser()
    manifest = artifact_manifest()
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Pi0.5 artifact missing or wrong size: {record['path']}")
        digest = hashlib.sha256() if record["algorithm"] == "sha256" else hashlib.sha1()
        if record["algorithm"] == "git-blob-sha1":
            digest.update(f"blob {record['size']}\0".encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record["digest"]:
            raise ValueError(f"Pi0.5 artifact digest mismatch: {record['path']}")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
