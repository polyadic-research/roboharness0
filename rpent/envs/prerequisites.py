"""Fast side-effect-free launch prerequisite projections."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rpent.envs.launch import LaunchPrerequisite
from rpent.planning import (
    AUTHORITATIVE_PLANNING_PROFILE,
    LEGACY_AUTHORITATIVE_PLANNING_PROFILE,
    ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE,
    ROBOCASA_PRECISE_PLANNING_PROFILE,
)
from rpent.runtime.factory import authoritative_runtime_options


def runtime_prerequisite(
    identifier: str,
    label: str,
    values: Mapping[str, object],
    *,
    endpoint_field: str,
    env_var: str,
    executable_suffix: Sequence[str] = (),
) -> LaunchPrerequisite:
    """Project endpoint or local-executable readiness without importing it."""
    if str(values.get(endpoint_field) or "").strip():
        return LaunchPrerequisite(
            identifier, label, "ready", f"Configured through {endpoint_field}."
        )
    configured = os.environ.get(env_var)
    executable = (
        Path(configured).expanduser().joinpath(*executable_suffix)
        if configured
        else None
    )
    ready = bool(executable and executable.is_file() and os.access(executable, os.X_OK))
    return LaunchPrerequisite(
        identifier,
        label,
        "ready" if ready else "blocked",
        (
            f"{env_var} names an executable runtime."
            if ready
            else f"Configure {env_var} or provide {endpoint_field}."
        ),
    )


def planner_prerequisite(mode: str, values: Mapping[str, object]) -> LaunchPrerequisite:
    """Project selected local planner command availability."""
    if mode != "hybrid":
        return LaunchPrerequisite(
            "planner", "Planner", "not-applicable", "This mode does not use a planner."
        )
    planner = str(values.get("planner", "codex"))
    if values.get("planning_profile") in {
        AUTHORITATIVE_PLANNING_PROFILE,
        LEGACY_AUTHORITATIVE_PLANNING_PROFILE,
        ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE,
        ROBOCASA_PRECISE_PLANNING_PROFILE,
    }:
        try:
            authoritative_runtime_options(planner)
        except ValueError as exc:
            return LaunchPrerequisite(
                "planner", "Planner", "blocked", str(exc)
            )
    command = {"codex": "codex", "claude_code": "claude"}.get(planner)
    if planner == "codex":
        has_base_url = bool(os.environ.get("CODEX_BASE_URL", "").strip())
        has_api_key = bool(os.environ.get("CODEX_API_KEY", "").strip())
        if has_base_url != has_api_key:
            return LaunchPrerequisite(
                "planner",
                "Planner",
                "blocked",
                "Configure both CODEX_BASE_URL and CODEX_API_KEY, or neither.",
            )
        if has_base_url:
            return LaunchPrerequisite(
                "planner", "Planner", "ready", "Codex API authentication is configured."
            )
        rpent_home = os.environ.get("RPENT_HOME", "").strip()
        current_auth = Path.home() / ".codex" / "auth.json"
        expected_auth = (
            Path(rpent_home).expanduser() / ".codex" / "auth.json"
            if rpent_home
            else None
        )
        if (
            expected_auth
            and expected_auth != current_auth
            and expected_auth.is_file()
            and not current_auth.is_file()
        ):
            return LaunchPrerequisite(
                "planner",
                "Planner",
                "blocked",
                f"HOME={Path.home()} cannot see Codex authentication stored under "
                f"RPENT_HOME={Path(rpent_home).expanduser()}; start RPent through the "
                "canonical deployment entrypoint.",
            )
        configured = os.environ.get("CODEX_BIN", "").strip()
        candidate = Path(configured).expanduser() if configured else None
        executable = (
            str(candidate)
            if candidate
            and candidate.is_absolute()
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
            else shutil.which(configured or "codex")
        )
        if executable:
            try:
                status = subprocess.run(
                    [executable, "login", "status"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                output = f"{status.stdout}\n{status.stderr}".lower()
                if status.returncode == 0 and "not logged in" not in output:
                    return LaunchPrerequisite(
                        "planner", "Planner", "ready", "Codex CLI is logged in."
                    )
            except (OSError, subprocess.TimeoutExpired):
                pass
        return LaunchPrerequisite(
            "planner",
            "Planner",
            "blocked",
            "Configure both CODEX_BASE_URL and CODEX_API_KEY, or run codex login.",
        )
    if command is None or shutil.which(command):
        return LaunchPrerequisite(
            "planner", "Planner", "ready", "The selected planner is configured."
        )
    return LaunchPrerequisite(
        "planner",
        "Planner",
        "warning",
        f"{command} was not found on PATH; verify the worker environment.",
    )


__all__ = ["planner_prerequisite", "runtime_prerequisite"]
