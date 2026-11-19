from __future__ import annotations

import json

import pytest

import rpent.evaluation.front_controller as front_controller
from rpent import __version__
from rpent.cli.main import main


def test_xiaomi_dry_run_retains_selected_policy_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_PYTHON", "/env/xiaomi/bin/python")
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/models/xiaomi")
    assert main(["--env", "robocasa", "--task", "SlideDishwasherRack",
                 "--vla", "xiaomi-mibot", "--dry-run", "--output-dir", str(tmp_path)]) == 0
    xiaomi = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert xiaomi["policy"]["instruction_source"] == "environment_language"
    assert xiaomi["policy"]["forward_step_guidance"] is True
    assert xiaomi["policy"]["executed_actions_per_request"] == 16
    assert xiaomi["runtime"]["policy_interpreter"] == "/env/xiaomi/bin/python"
    assert xiaomi["runtime"]["policy_snapshot_path"] == "/models/xiaomi"
    assert xiaomi["environment"]["preset_name"] == "robocasa365-pandaomron-xiaomi-mibot-v1"


@pytest.mark.parametrize(
    ("environment", "task", "episodes", "expected_cases"),
    [
        ("robocasa", "atomic_seen", 20, 360),
    ],
)
def test_dry_run_resolves_from_packaged_contract_without_simulator_dependencies(
    monkeypatch,
    tmp_path,
    environment: str,
    task: str,
    episodes: int,
    expected_cases: int,
) -> None:
    monkeypatch.delenv("RPENT_ROBOCASA_PYTHON", raising=False)
    monkeypatch.delenv("RPENT_SMOLVLA_PYTHON", raising=False)
    output_dir = tmp_path / environment

    exit_code = main(
        [
            "--env",
            environment,
            "--task",
            task,
            "--episodes",
            str(episodes),
            "--planning-profile",
            "authoritative-gated-v1",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(
        (output_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["cases"]) == expected_cases
    assert manifest["environment"]["name"] == environment


def test_installed_wheel_identity_does_not_require_a_git_worktree(
    monkeypatch,
    tmp_path,
) -> None:
    installed_module = (
        tmp_path / "site-packages/rpent/evaluation/front_controller.py"
    )
    monkeypatch.setattr(front_controller, "__file__", str(installed_module))

    commit, dirty = front_controller._git_identity()

    assert commit == f"installed-wheel:rpent@{__version__}"
    assert dirty is False


def test_deployment_release_identity_takes_precedence_over_git(
    monkeypatch,
    tmp_path,
) -> None:
    deployed_commit = "1" * 40
    installed_module = tmp_path / "repo/rpent/evaluation/front_controller.py"
    installed_module.parent.mkdir(parents=True)
    (tmp_path / "repo/.rpent-release").write_text(
        f"SOURCE_COMMIT={deployed_commit}\n"
        "DEPLOYED_AT_UTC=2026-08-18T00:00:00Z\n"
        "DEPLOY_METHOD=rsync-clean-tree\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(front_controller, "__file__", str(installed_module))

    commit, dirty = front_controller._git_identity()

    assert commit == deployed_commit
    assert dirty is False
