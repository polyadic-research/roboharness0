from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import rpent.cli.main as cli
from robots.robocasa.competition import (
    DEVELOPMENT_PROFILE_ID,
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.tasks import packaged_registry_snapshot
from rpent.envs import get_env_spec


def _parse(argv: list[str]):
    with patch.object(sys, "argv", ["rpent", *argv]):
        parser = cli._build_argparser()
        early, _ = parser.parse_known_args()
        spec = get_env_spec(early.env_name)
        spec.add_cli_args(parser, use_dashboard=False)
        return parser.parse_args()


def test_robocasa_cli_defaults_to_pinned_safe_preset() -> None:
    args = _parse(["--env", "robocasa", "--task", "CloseFridge"])
    config = get_env_spec("robocasa").parse_config(args)

    assert config.task == "CloseFridge"
    assert config.object_registries == ("lightwheel", "objaverse")
    assert config.safety_ceiling is None
    assert config.checkpoint.repo_id == "lerobot/smolvla_robocasa"
    assert config.checkpoint.revision == "38d38bf58e1217092c5a81418fcc8eef1a7c3a6f"
    assert config.control_hz == 20
    assert config.image_size == 256
    assert config.require_environment_success is True
    assert config.task_desc["execution_mode"] == "rpent_harness"
    assert args.evaluation_profile == DEVELOPMENT_PROFILE_ID


def test_robocasa_cli_accepts_pinned_xiaomi_policy_path() -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--task",
            "KettleBoiling",
            "--policy.path",
            "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
        ]
    )

    assert args.policy_path == "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"


def test_robocasa_defaults_to_upstream_horizon_profile() -> None:
    args = _parse(["--env", "robocasa", "--task", "CloseFridge"])

    assert (
        get_env_spec("robocasa").resolve_planning_profile(args)
        == "robocasa-horizon-gated-v1"
    )


def test_robocasa_official_profile_supplies_task_and_locked_cli_values() -> None:
    args = _parse(["--env", "robocasa", "--evaluation-profile", LEADERBOARD_PROFILE_ID])

    config = get_env_spec("robocasa").parse_config(args)
    profile = resolve_evaluation_profile(
        LEADERBOARD_PROFILE_ID, packaged_registry_snapshot()
    )

    assert config.task == ",".join(profile.task_names)
    assert config.seed == 7
    assert config.safety_ceiling is None


def test_robocasa_official_profile_rejects_conflicting_task() -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--evaluation-profile",
            LEADERBOARD_PROFILE_ID,
            "--task",
            "CloseFridge",
        ]
    )

    with pytest.raises(ValueError, match="owns task selector"):
        get_env_spec("robocasa").parse_config(args)


@pytest.mark.parametrize(
    "override",
    [
        ["--seed", "0"],
        ["--episodes", "1"],
        ["--max-episode-steps", "900"],
    ],
)
def test_robocasa_official_profile_rejects_conflicting_cli_overrides(
    override: list[str],
) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--evaluation-profile",
            LEADERBOARD_PROFILE_ID,
            *override,
        ]
    )

    with pytest.raises(ValueError, match="official evaluation profile owns"):
        get_env_spec("robocasa").parse_config(args)


def test_robocasa_cli_rejects_unknown_evaluation_profile() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--env",
                "robocasa",
                "--evaluation-profile",
                "future-profile",
            ]
        )


@pytest.mark.parametrize("foreground_flag", ["--interactive", "--dashboard"])
def test_robocasa_official_profile_rejects_foreground_modes(
    foreground_flag: str,
) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--evaluation-profile",
            LEADERBOARD_PROFILE_ID,
            foreground_flag,
        ]
    )

    with pytest.raises(ValueError, match="coordinated evaluation"):
        get_env_spec("robocasa").parse_config(args)


@pytest.mark.parametrize("foreground_flag", ["--interactive", "--dashboard"])
def test_robocasa_development_profile_keeps_foreground_modes(
    foreground_flag: str,
) -> None:
    args = _parse(["--env", "robocasa", "--task", "CloseFridge", foreground_flag])

    assert get_env_spec("robocasa").parse_config(args).task == "CloseFridge"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--env",
            "robocasa",
            "--task",
            "CloseFridge",
            "--object-registries",
            "objaverse,lightwheel",
        ],
        [
            "--env",
            "robocasa",
            "--task",
            "CloseFridge",
            "--env.obj_registries=[objaverse,lightwheel]",
        ],
    ],
)
def test_robocasa_accepts_explicit_registry_syntax(argv) -> None:
    config = get_env_spec("robocasa").parse_config(_parse(argv))
    assert config.object_registries == ("objaverse", "lightwheel")


@pytest.mark.parametrize(
    "registries",
    ["", "[lightwheel,lightwheel]", "unknown", "[objaverse,]"],
)
def test_robocasa_rejects_invalid_registry_configuration(registries) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--task",
            "CloseFridge",
            f"--object-registries={registries}",
        ]
    )
    with pytest.raises(ValueError, match="registr"):
        get_env_spec("robocasa").parse_config(args)


def test_robocasa_requires_task_and_positive_safety_ceiling() -> None:
    with pytest.raises(ValueError, match="--task is required"):
        get_env_spec("robocasa").parse_config(_parse(["--env", "robocasa"]))
    with pytest.raises(SystemExit):
        _parse(
            [
                "--env",
                "robocasa",
                "--task",
                "CloseFridge",
                "--max-episode-steps",
                "0",
            ]
        )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("libero-hooked-gated-v3", "libero-hooked-gated-v3"),
        ("episode-hooked-gated-v1", "episode-hooked-gated-v1"),
        ("libero-reactive-v1", "libero-reactive-v1"),
        ("libero-authoritative-gated-v1", "authoritative-gated-v1"),
    ],
)
def test_robocasa_resolves_registered_planning_profile(
    requested: str,
    expected: str,
) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--task",
            "CloseFridge",
            "--planning-profile",
            requested,
        ]
    )

    assert get_env_spec("robocasa").resolve_planning_profile(args) == expected


def test_robocasa_parser_rejects_unknown_planning_profile() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--env",
                "robocasa",
                "--task",
                "CloseFridge",
                "--planning-profile",
                "unknown-planning-profile",
            ]
        )


def test_robocasa_parser_does_not_import_upstream_or_lerobot() -> None:
    before = {
        name
        for name in sys.modules
        if name == "robocasa" or name.startswith(("robocasa.", "lerobot"))
    }
    _parse(["--env", "robocasa", "--task", "atomic_seen"])
    after = {
        name
        for name in sys.modules
        if name == "robocasa" or name.startswith(("robocasa.", "lerobot"))
    }
    assert after == before
