"""The Xiaomi-selectable release must agree across source and installed metadata."""

import importlib.metadata
import tomllib
from pathlib import Path

from packaging.version import Version

from rpent import __version__


def test_xiaomi_release_version_and_installed_metadata_agree():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert Version(__version__) >= Version("1.2.0")
    assert project["project"]["version"] == __version__
    assert importlib.metadata.version("rpent") == __version__
