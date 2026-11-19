"""Build content-addressed manifests for reproducible planning runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rpent.planning.stack import PlanningStack
from rpent.source_identity import deployed_source_commit


def content_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for text or JSON-serializable content."""
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON-like values with the repository's one canonical form."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def build_experiment_manifest(
    *,
    config: PlanningStack,
    planner_backend: str,
    model: str | None,
    max_tokens: int,
    no_images: bool,
    system_prompt: str,
    user_message: str,
    tool_schemas: list[dict[str, Any]],
    repo_root: str | Path,
    skill_outcome_adapter_id: str = "unknown-physical-outcome-v1",
    runtime_metadata: dict[str, Any] | None = None,
    volatile_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture every material planning input that can be pinned cheaply."""
    normalized_system_prompt = _replace_volatile(system_prompt, volatile_values)
    normalized_tools = _replace_volatile(tool_schemas, volatile_values)
    execution = {
        "skill_outcome_adapter_id": skill_outcome_adapter_id,
    }
    if runtime_metadata is not None:
        execution["runtime"] = runtime_metadata
    return {
        **config.metadata(),
        "backend": {
            "id": planner_backend,
            "model": model,
            "max_tokens": max_tokens,
            "no_images": no_images,
        },
        "execution": execution,
        "artifacts": {
            "system_prompt_sha256": content_hash(system_prompt),
            "policy_prompt_sha256": content_hash(normalized_system_prompt),
            "user_message_sha256": content_hash(user_message),
            "tool_schema_sha256": content_hash(normalized_tools),
            "tool_names": [schema.get("name") for schema in tool_schemas],
        },
        "code": {
            "git_commit": _git_commit(Path(repo_root)),
        },
    }


def _replace_volatile(value: Any, substitutions: dict[str, str] | None) -> Any:
    """Replace run-specific values before computing a policy identity hash."""
    if not substitutions:
        return value
    if isinstance(value, str):
        for actual, placeholder in substitutions.items():
            if actual:
                value = value.replace(actual, placeholder)
        return value
    if isinstance(value, dict):
        return {
            str(key): _replace_volatile(item, substitutions)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_replace_volatile(item, substitutions) for item in value]
    return value


def _git_commit(repo_root: Path) -> str | None:
    deployed_commit = deployed_source_commit(repo_root)
    if deployed_commit is not None:
        return deployed_commit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
