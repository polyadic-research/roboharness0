"""Immutable RoboCasa365 evaluation profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robots.robocasa.dependencies import ROBOCASA_COMMIT
from robots.robocasa.tasks import packaged_registry_snapshot

DEVELOPMENT_PROFILE_ID = "development"
LEADERBOARD_PROFILE_ID = "robocasa365-leaderboard-v1"

_OFFICIAL_TASK_SETS = (
    "atomic_seen",
    "composite_seen",
    "composite_unseen",
)
_OFFICIAL_GROUP_SIZES = (18, 16, 16)
_PINNED_REGISTRY_DIGEST = str(packaged_registry_snapshot()["registry_digest"])


@dataclass(frozen=True, slots=True)
class RoboCasaEvaluationProfile:
    profile_id: str
    official: bool
    task_sets: tuple[str, ...]
    task_names: tuple[str, ...]
    scene_split: str | None
    rollouts_per_task: int | None
    scenario_seed: int | None
    scenario_sampling: str
    success_source: str
    horizon_source: str
    worker_grouping: str


def _official_task_names(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    provenance = snapshot.get("provenance")
    groups = snapshot.get("groups")
    tasks = snapshot.get("tasks")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("commit") != ROBOCASA_COMMIT
    ):
        raise RuntimeError("official RoboCasa registry provenance is inconsistent")
    if not isinstance(groups, Mapping) or not isinstance(tasks, Mapping):
        raise RuntimeError("official RoboCasa registry snapshot is incomplete")
    encoded = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if snapshot.get("registry_digest") != digest or digest != _PINNED_REGISTRY_DIGEST:
        raise RuntimeError("official RoboCasa registry provenance digest is invalid")

    selected: list[str] = []
    for group_name, expected_size in zip(
        _OFFICIAL_TASK_SETS,
        _OFFICIAL_GROUP_SIZES,
        strict=True,
    ):
        group = groups.get(group_name)
        if not isinstance(group, Mapping):
            raise RuntimeError(f"official RoboCasa task set {group_name} is missing")
        names = tuple(str(name) for name in group.get("tasks", ()))
        if len(names) != expected_size or len(names) != len(set(names)):
            raise RuntimeError(
                f"official RoboCasa task set {group_name} has invalid membership"
            )
        selected.extend(names)
    if len(selected) != 50 or len(selected) != len(set(selected)):
        raise RuntimeError("official RoboCasa task sets overlap")
    if any(name not in tasks for name in selected):
        raise RuntimeError("official RoboCasa registry is missing task metadata")
    return tuple(selected)


def resolve_evaluation_profile(
    profile_id: str | None,
    registry_snapshot: Mapping[str, Any],
) -> RoboCasaEvaluationProfile:
    """Resolve one stable profile and validate official facts together."""

    selected = DEVELOPMENT_PROFILE_ID if profile_id is None else profile_id
    if selected == DEVELOPMENT_PROFILE_ID:
        return RoboCasaEvaluationProfile(
            profile_id=DEVELOPMENT_PROFILE_ID,
            official=False,
            task_sets=(),
            task_names=(),
            scene_split=None,
            rollouts_per_task=None,
            scenario_seed=None,
            scenario_sampling="base_seed_plus_episode_index",
            success_source="robocasa_native_task_predicate",
            horizon_source="upstream_or_explicit_min",
            worker_grouping="case",
        )
    if selected != LEADERBOARD_PROFILE_ID:
        raise ValueError(f"unknown RoboCasa evaluation profile {selected!r}")
    return RoboCasaEvaluationProfile(
        profile_id=LEADERBOARD_PROFILE_ID,
        official=True,
        task_sets=_OFFICIAL_TASK_SETS,
        task_names=_official_task_names(registry_snapshot),
        scene_split="pretrain",
        rollouts_per_task=50,
        scenario_seed=7,
        scenario_sampling="seeded_task_environment_sequential_resets",
        success_source="robocasa_native_task_predicate",
        horizon_source="robocasa_1_0_1_upstream",
        worker_grouping="task",
    )


__all__ = [
    "DEVELOPMENT_PROFILE_ID",
    "LEADERBOARD_PROFILE_ID",
    "RoboCasaEvaluationProfile",
    "resolve_evaluation_profile",
]
