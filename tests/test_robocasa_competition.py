from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa.competition import (
    DEVELOPMENT_PROFILE_ID,
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.evaluation import prepare_evaluation
from robots.robocasa.tasks import TASK_GROUP_FIXTURE
from rpent.evaluation.front_controller import build_evaluation_manifest


def _registry_snapshot() -> dict:
    return json.loads(TASK_GROUP_FIXTURE.read_text(encoding="utf-8"))


def _official_build_inputs(
    tmp_path: Path,
    planning_profile: str = "robocasa-authoritative-gated-v1",
):
    profile = resolve_evaluation_profile(
        LEADERBOARD_PROFILE_ID,
        _registry_snapshot(),
    )
    canonical_selector = ",".join(profile.task_names)
    args = SimpleNamespace(
        env_name="robocasa",
        dry_run=True,
        task=canonical_selector,
        evaluation_profile=LEADERBOARD_PROFILE_ID,
        planning_profile=planning_profile,
        episodes=None,
        seed=0,
        scene_split=None,
        max_episode_steps=None,
        policy_endpoint=None,
        env_endpoint=None,
        recording_mode=None,
        recording_camera=(),
        require_complete_artifacts=False,
    )
    run_config = SimpleNamespace(
        object_registries=("lightwheel",),
        output_dir=tmp_path,
        safety_ceiling=None,
    )
    env_spec = SimpleNamespace(
        parse_config=lambda _args: run_config,
        prepare_evaluation=prepare_evaluation,
    )
    launch = {
        "execution_mode": "hybrid",
        "preset_version": 6,
        "runner": "robocasa365-pandaomron-smolvla-v1",
        "submitted": {
            "evaluation_profile": LEADERBOARD_PROFILE_ID,
            "planning_profile": planning_profile,
        },
        "effective": {"planning_profile": planning_profile},
    }
    return profile, canonical_selector, args, env_spec, launch


def test_leaderboard_profile_locks_official_protocol_and_task_membership() -> None:
    snapshot = _registry_snapshot()

    profile = resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)

    expected_sets = ("atomic_seen", "composite_seen", "composite_unseen")
    assert DEVELOPMENT_PROFILE_ID == "development"
    assert LEADERBOARD_PROFILE_ID == "robocasa365-leaderboard-v1"
    assert profile.profile_id == LEADERBOARD_PROFILE_ID
    assert profile.official is True
    assert profile.task_sets == expected_sets
    assert profile.task_names == tuple(
        name
        for group_name in expected_sets
        for name in snapshot["groups"][group_name]["tasks"]
    )
    assert [len(snapshot["groups"][name]["tasks"]) for name in expected_sets] == [
        18,
        16,
        16,
    ]
    assert len(profile.task_names) == len(set(profile.task_names)) == 50
    assert profile.scene_split == "pretrain"
    assert profile.rollouts_per_task == 50
    assert profile.scenario_seed == 7
    assert profile.scenario_sampling == "seeded_task_environment_sequential_resets"
    assert profile.success_source == "robocasa_native_task_predicate"
    assert profile.horizon_source == "robocasa_1_0_1_upstream"
    assert profile.worker_grouping == "task"


def test_development_profile_is_default_and_keeps_protocol_inputs_unlocked() -> None:
    profile = resolve_evaluation_profile(None, _registry_snapshot())

    assert profile.profile_id == DEVELOPMENT_PROFILE_ID
    assert profile.official is False
    assert profile.task_sets == ()
    assert profile.task_names == ()
    assert profile.scene_split is None
    assert profile.rollouts_per_task is None
    assert profile.scenario_seed is None
    assert profile.worker_grouping == "case"


def test_competition_profile_is_frozen() -> None:
    profile = resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, _registry_snapshot())

    with pytest.raises(FrozenInstanceError):
        profile.scene_split = "target"  # type: ignore[misc]


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown RoboCasa evaluation profile"):
        resolve_evaluation_profile("leaderboard-latest", _registry_snapshot())


@pytest.mark.parametrize("profile_id", ["", 0, False])
def test_explicit_falsey_profile_id_is_rejected(profile_id: object) -> None:
    with pytest.raises(ValueError, match="unknown RoboCasa evaluation profile"):
        resolve_evaluation_profile(profile_id, _registry_snapshot())  # type: ignore[arg-type]


@pytest.mark.parametrize("conflict", ["duplicate", "wrong_size", "missing_task"])
def test_official_profile_rejects_conflicting_registry_membership(
    conflict: str,
) -> None:
    snapshot = _registry_snapshot()
    tasks = snapshot["groups"]["atomic_seen"]["tasks"]
    if conflict == "duplicate":
        tasks[-1] = tasks[0]
    elif conflict == "wrong_size":
        tasks.pop()
    else:
        snapshot["tasks"].pop(tasks[0])

    with pytest.raises(RuntimeError, match="official RoboCasa"):
        resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)


def test_official_profile_rejects_inconsistent_registry_provenance() -> None:
    snapshot = _registry_snapshot()
    snapshot["provenance"]["commit"] = "0" * 40

    with pytest.raises(RuntimeError, match="provenance"):
        resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)


def test_official_profile_rejects_self_consistent_replacement_membership() -> None:
    snapshot = _registry_snapshot()
    official_names = {
        name
        for group_name in ("atomic_seen", "composite_seen", "composite_unseen")
        for name in snapshot["groups"][group_name]["tasks"]
    }
    replacement = next(name for name in snapshot["tasks"] if name not in official_names)
    snapshot["groups"]["atomic_seen"]["tasks"][-1] = replacement
    encoded = json.dumps(
        snapshot["groups"], sort_keys=True, separators=(",", ":")
    ).encode()
    snapshot["registry_digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    with pytest.raises(RuntimeError, match="official RoboCasa"):
        resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)


def test_prepare_evaluation_records_task_sets_and_effective_scene_split(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dry_run=True,
        task=None,
        evaluation_profile=LEADERBOARD_PROFILE_ID,
        planning_profile="authoritative-gated-v1",
        episodes=None,
        seed=0,
        max_episode_steps=None,
        actions_per_policy_request=25,
        policy_endpoint=None,
        env_endpoint=None,
    )
    run_config = SimpleNamespace(
        object_registries=("lightwheel",),
        output_dir=tmp_path,
        safety_ceiling=None,
    )

    prepared = prepare_evaluation(args, run_config)

    assert len(prepared.tasks) == 50
    assert prepared.simulator["task_sets"] == (
        "atomic_seen",
        "composite_seen",
        "composite_unseen",
    )
    assert prepared.simulator["scene_split"] == "pretrain"
    assert {task.split for task in prepared.tasks} == {"pretrain"}




def test_prepare_evaluation_preserves_selected_development_planning_profile(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dry_run=True,
        task="CloseFridge",
        evaluation_profile=DEVELOPMENT_PROFILE_ID,
        planning_profile="libero-hooked-gated-v3",
        episodes=1,
        seed=0,
        max_episode_steps=None,
        policy_endpoint=None,
        env_endpoint=None,
    )
    run_config = SimpleNamespace(
        object_registries=("lightwheel",),
        output_dir=tmp_path,
        safety_ceiling=None,
    )

    prepared = prepare_evaluation(args, run_config)

    assert prepared.policy["planning_profile"] == "libero-hooked-gated-v3"
    assert prepared.policy["noise_mode"] == "episode_generator"
    assert prepared.policy["noise_seed_policy"] == "case_seed"
    assert prepared.policy["policy_method"] == "predict_action_chunk"
    assert prepared.policy["checkpoint_chunk_size"] == 50
    assert prepared.policy["checkpoint_n_action_steps"] == 50
    assert prepared.policy["executed_actions_per_request"] == 25
    assert prepared.schedule == {}


@pytest.mark.parametrize(
    "planning_profile",
    ["authoritative-gated-v1", "libero-hooked-gated-v3"],
)
def test_official_manifest_preserves_user_selected_planning_profile(
    tmp_path: Path,
    planning_profile: str,
) -> None:
    _, _, args, env_spec, launch = _official_build_inputs(tmp_path, planning_profile)

    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )

    assert manifest.policy["planning_profile"] == planning_profile
    assert manifest.schedule["effective_inputs"]["planning_profile"] == planning_profile
    assert manifest.launch["submitted"]["planning_profile"] == planning_profile
    assert manifest.launch["effective"]["planning_profile"] == planning_profile
    assert "planning_profile" not in manifest.schedule["locked_inputs"]


def test_official_manifest_planning_identity_changes_manifest_id(tmp_path: Path) -> None:
    _, _, authoritative_args, env_spec, authoritative_launch = _official_build_inputs(
        tmp_path, "authoritative-gated-v1"
    )
    _, _, hooked_args, _, hooked_launch = _official_build_inputs(
        tmp_path, "libero-hooked-gated-v3"
    )

    authoritative, _ = build_evaluation_manifest(
        authoritative_args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=authoritative_launch,
    )
    hooked, _ = build_evaluation_manifest(
        hooked_args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=hooked_launch,
    )

    assert authoritative.manifest_id != hooked.manifest_id


def test_official_manifest_materializes_locked_2500_case_schedule(
    tmp_path: Path,
) -> None:
    profile, canonical_selector, args, env_spec, launch = _official_build_inputs(
        tmp_path
    )

    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )

    assert len(manifest.simulator["tasks"]) == 50
    assert len(manifest.cases) == 2_500
    assert [case.task_name for case in manifest.cases[0:50]] == [
        profile.task_names[0]
    ] * 50
    assert [case.task_name for case in manifest.cases[50:100]] == [
        profile.task_names[1]
    ] * 50
    assert [case.episode_index for case in manifest.cases[:50]] == list(range(50))
    assert {case.seed for case in manifest.cases} == {7}
    assert len({case.case_id for case in manifest.cases}) == 2_500
    assert all(
        case.horizon == task["upstream_horizon"]
        for task, case in zip(
            manifest.simulator["tasks"], manifest.cases[::50], strict=True
        )
    )
    assert dict(manifest.schedule) == {
        "worker_timeout": {
            "mode": "horizon_tiers",
            "steps_per_tier": 750,
            "seconds_per_tier": 1800,
        },
        "evaluation_profile": LEADERBOARD_PROFILE_ID,
        "episodes_per_task": 50,
        "base_seed": 7,
        "seed_policy": "shared_environment_sequence",
        "ordering": "task_registry_then_episode_index",
        "task_count": 50,
        "task_sets": ("atomic_seen", "composite_seen", "composite_unseen"),
        "task_set_sizes": {
            "atomic_seen": 18,
            "composite_seen": 16,
            "composite_unseen": 16,
        },
        "task_set_membership": {
            name: tuple(_registry_snapshot()["groups"][name]["tasks"])
            for name in ("atomic_seen", "composite_seen", "composite_unseen")
        },
        "task_registry_digest": _registry_snapshot()["registry_digest"],
        "scene_split": "pretrain",
        "scenario_sampling": "seeded_task_environment_sequential_resets",
        "worker_grouping": "task",
        "success_source": "robocasa_native_task_predicate",
        "horizon_source": "robocasa_1_0_1_upstream",
        "locked_inputs": {
            "evaluation_profile": LEADERBOARD_PROFILE_ID,
            "task": canonical_selector,
            "seed": 7,
            "episodes": 50,
            "scene_split": "pretrain",
            "max_episode_steps": None,
        },
        "effective_inputs": {
            "evaluation_profile": LEADERBOARD_PROFILE_ID,
            "task": canonical_selector,
            "seed": 7,
            "episodes": 50,
            "scene_split": "pretrain",
            "max_episode_steps": None,
            "planning_profile": "robocasa-authoritative-gated-v1",
            "task_count": 50,
            "case_count": 2_500,
            "batch_count": 50,
            "worker_grouping": "task",
            "scenario_sampling": "seeded_task_environment_sequential_resets",
            "success_source": "robocasa_native_task_predicate",
            "horizon_source": "robocasa_1_0_1_upstream",
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
        },
    }
    assert manifest.environment["evaluation_profile"] == LEADERBOARD_PROFILE_ID
    assert manifest.simulator["evaluation_profile"] == LEADERBOARD_PROFILE_ID
    assert (
        manifest.policy["planning_profile"]
        == "robocasa-authoritative-gated-v1"
    )
    assert manifest.launch["effective"]["evaluation_profile"] == (
        LEADERBOARD_PROFILE_ID
    )
    assert (
        manifest.launch["effective"]["planning_profile"]
        == "robocasa-authoritative-gated-v1"
    )
    assert manifest.runtime["grouped_worker_module"] == (
        "robots.robocasa.competition_worker"
    )
    assert dict(manifest.launch["effective"]) == dict(
        manifest.schedule["effective_inputs"]
    )
    assert manifest.qualification["official_profile"] == LEADERBOARD_PROFILE_ID
    assert manifest.qualification["require_exact_task_membership"] is True
    assert manifest.qualification["require_complete_task_batches"] is True
    assert manifest.qualification["require_workload_confirmation"] is True
    assert manifest.qualification["allow_selective_case_rerun"] is False
    assert manifest.task_selector == canonical_selector


def test_implicit_default_seed_is_accepted_and_locked_to_seven(
    tmp_path: Path,
) -> None:
    _, _, args, env_spec, launch = _official_build_inputs(tmp_path)

    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )

    assert args.seed == 0
    assert manifest.schedule["base_seed"] == 7
    assert manifest.launch["effective"]["seed"] == 7


def test_explicit_conflicting_cli_seed_is_rejected(tmp_path: Path) -> None:
    _, _, args, env_spec, launch = _official_build_inputs(tmp_path)

    with pytest.raises(ValueError, match="locked input.*seed"):
        build_evaluation_manifest(
            args,
            env_spec,
            ["--env", "robocasa", "--seed", "0"],
            git_identity=lambda: ("a" * 40, False),
            launch=launch,
        )


def test_explicit_matching_cli_seed_is_accepted(tmp_path: Path) -> None:
    _, _, args, env_spec, launch = _official_build_inputs(tmp_path)
    args.seed = 7

    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa", "--seed", "7"],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )

    assert manifest.schedule["base_seed"] == 7


@pytest.mark.parametrize("section", ["submitted", "effective"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "CloseFridge"),
        ("seed", 0),
        ("episodes", 1),
        ("scene_split", "target"),
        ("max_episode_steps", 1_000),
    ],
)
def test_official_manifest_rejects_conflicting_launch_audit_inputs(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    _, _, args, env_spec, launch = _official_build_inputs(tmp_path)
    launch[section][field] = value

    with pytest.raises(ValueError, match=f"launch.{section}.*{field}"):
        build_evaluation_manifest(
            args,
            env_spec,
            ["--env", "robocasa"],
            git_identity=lambda: ("a" * 40, False),
            launch=launch,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "CloseFridge"),
        ("seed", 8),
        ("episodes", 49),
        ("scene_split", "target"),
        ("max_episode_steps", 1_000),
    ],
)
def test_official_preparation_rejects_conflicting_overrides(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    args = SimpleNamespace(
        dry_run=True,
        task=None,
        evaluation_profile=LEADERBOARD_PROFILE_ID,
        planning_profile="authoritative-gated-v1",
        episodes=None,
        seed=0,
        scene_split=None,
        max_episode_steps=None,
        policy_endpoint=None,
        env_endpoint=None,
    )
    setattr(args, field, value)
    run_config = SimpleNamespace(
        object_registries=("lightwheel",),
        output_dir=tmp_path,
        safety_ceiling=None,
    )

    with pytest.raises(ValueError, match=f"official.*{field}"):
        prepare_evaluation(args, run_config)
