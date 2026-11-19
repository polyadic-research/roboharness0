"""Resolve source identity for Git checkouts and rsync deployments."""

from __future__ import annotations

import re
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def deployed_source_commit(repo_root: Path) -> str | None:
    """Read the commit recorded by the rsync deployment protocol, if valid."""
    try:
        lines = (repo_root / ".rpent-release").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if key == "SOURCE_COMMIT" and separator and _COMMIT_PATTERN.fullmatch(value):
            return value
    return None
