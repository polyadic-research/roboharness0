"""Attach task-scoped static acceptance notes without simulator access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SKILL_DIRECTORY = Path(__file__).resolve().parent / "robocasa-acceptance"


def load_acceptance_skill(task: str) -> dict[str, Any]:
    """Load only exact task matches; return advisory text and frozen provenance."""
    root = SKILL_DIRECTORY
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = {name.strip() for name in task.split(",")}
    matched = sorted(selected & manifest["tasks"].keys())
    paths = (["SKILL.md", *sorted({manifest["tasks"][name]["reference"] for name in matched})]
             if matched else [])
    text_parts = []
    hashes = {"manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}
    for relative in paths:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("acceptance reference must stay inside its skill")
        content = path.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        text_parts.append(content.decode("utf-8").strip())
    text = "\n\n".join(text_parts)
    return {
        "text": text,
        "metadata": {
            "id": manifest["id"],
            "version": manifest["version"],
            "matched_tasks": matched,
            "conditions": sorted({manifest["tasks"][name]["condition"] for name in matched}),
            "advisory_only": True,
            "source_revision": manifest["source_revision"],
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "files_sha256": hashes,
        },
    }
