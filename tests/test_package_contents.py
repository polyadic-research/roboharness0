from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def test_wheel_preserves_retained_robocasa_runtime_and_excludes_other_integrations(tmp_path) -> None:
    wheel_dir = tmp_path / "wheel"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = tuple(wheel_dir.glob("rpent-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        manifest = json.loads(Path("publication/source-manifest.json").read_text())
        excluded = set(manifest.get("excluded_frozen_files", []))
        for relative, expected in manifest["frozen_files"].items():
            if relative.startswith(("rpent/", "robots/")) and relative not in excluded:
                assert hashlib.sha256(archive.read(relative)).hexdigest() == expected, relative
        assert not names.intersection(excluded)
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        assert metadata["License-Expression"] == "Apache-2.0"
        license_files = {
            "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
            "LICENSES/RPent-Apache-2.0.txt", "LICENSES/RoboCasa-MIT.txt",
            "LICENSES/robosuite-MIT.txt",
        }
        assert set(metadata.get_all("License-File", [])) == license_files
        for relative in license_files:
            packaged = next(name for name in names if name.endswith(f"/licenses/{relative}"))
            assert archive.read(packaged) == Path(relative).read_bytes(), relative
    required = {
        "rpent/cli/main.py",
        "rpent/harness/lifecycle.py",
        "rpent/runtime/codexgateway.py",
        "rpent/policies/xiaomi_mibot_server.py",
        "robots/robocasa/__init__.py",
        "robots/robocasa/competition_worker.py",
        "robots/robocasa/policy_skills/xiaomi-mibot.capabilities.json",
        "robots/robocasa/task_groups.json",
    }
    assert required.issubset(names), sorted(required - names)
    assert not any(name.startswith(("robots/libero/", "robots/robocerebra/", "ui/")) for name in names)
    sdists = tuple(wheel_dir.glob("rpent-*.tar.gz"))
    assert len(sdists) == 1
    with tarfile.open(sdists[0]) as archive:
        prefix = f"rpent-{metadata['Version']}/"
        for relative in license_files:
            packaged_file = archive.extractfile(prefix + relative)
            assert packaged_file is not None, relative
            assert packaged_file.read() == Path(relative).read_bytes(), relative
        packaged_metadata = archive.extractfile(prefix + "PKG-INFO")
        assert packaged_metadata is not None
        source_metadata = BytesParser().parsebytes(packaged_metadata.read())
        assert source_metadata["License-Expression"] == "Apache-2.0"
        assert set(source_metadata.get_all("License-File", [])) == license_files


def test_package_configuration_does_not_add_heavy_backends_to_base_dependencies() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["rpent*", "robots*"]' in text
    dependencies = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    for heavy in ("lerobot", "robocasa", "robosuite", "mujoco"):
        assert heavy not in dependencies.lower()


def test_confirmed_project_license_is_recorded_in_release_metadata() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    provenance = json.loads(Path("SOURCE_PROVENANCE.json").read_text())
    candidate = json.loads(Path("evaluations/robocasa365/2.0/candidate.json").read_text())
    assert project["license"] == "Apache-2.0"
    assert provenance["metadata"]["project_license"] == "Apache-2.0"
    assert candidate["project_license"] == "Apache-2.0"
    assert provenance["metadata"]["project_license_file"] == "LICENSE"
    assert Path("LICENSE").read_bytes() == Path("LICENSES/RPent-Apache-2.0.txt").read_bytes()
