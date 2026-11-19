"""Static env-extension descriptor.

Lives in :mod:`rpent.envs` alongside
:class:`~rpent.envs.prompt_bundle.PromptBundle` so envs
and planners can both import it without pulling in
:mod:`rpent.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`rpent.tools.toolkit.Toolkit` and its env subclasses —
``EnvSpec`` carries the env identity, the prompt bundle, and
the three runner hooks (``add_cli_args`` / ``parse_config`` /
``init_runtime``) that keep ``rpent/cli/main.py`` env-agnostic.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.envs.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from rpent.dashboard.state import State
    from rpent.envs.launch import LaunchPreset
    from rpent.episode.runtime import EpisodeBootstrap
    from rpent.evaluation.preparation import PolicyDaemon, PreparedEvaluation
    from rpent.recording.contracts import CameraDescriptor
    from rpent.recording.data_schema import CaseDataSchema
    from rpent.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`EnvSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    dashboard_state: "State | None"
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for RPent."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    build_episode_bootstrap: Callable[
        [argparse.Namespace, Path, Any],
        "EpisodeBootstrap",
    ]
    prepare_evaluation: Callable[
        [argparse.Namespace, RunConfig, Iterable[Any] | None],
        "PreparedEvaluation",
    ]
    build_policy_daemon: Callable[
        [argparse.Namespace, Path],
        "PolicyDaemon | None",
    ]
    resolve_planning_profile: Callable[[argparse.Namespace], str]
    data_schema: Callable[
        [tuple["CameraDescriptor", ...]], "CaseDataSchema"
    ] | None = None
    coordinate_evaluation_by_default: bool = False
    launch_preset: Callable[[], "LaunchPreset"] | None = None
    validate_evaluation_launch: Callable[[argparse.Namespace], None] | None = None
    evaluation_runners: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
