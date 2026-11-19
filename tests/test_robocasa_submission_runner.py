"""Reviewer planning stays offline and cannot masquerade as historical replay."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import pytest

from tools import run_robocasa_submission as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def planner_path(tmp_path):
    path = tmp_path / "planner.json"
    path.write_text((ROOT / runner.PLANNER).read_text())
    return path


def test_default_entry_only_plans_and_does_not_read_or_print_credentials(tmp_path, planner_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-sentinel")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://private-test.invalid/v1")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Planning started a process"))
    code = runner.main([
        "--runtime-prefix", str(tmp_path / "runtime not installed"),
        "--output-dir", str(tmp_path / "new evaluation"),
        "--planner-config", str(planner_path),
        "--api-settings-confirmed",
        "--env-file", str(tmp_path / "private file does not exist.env"),
    ])
    output = capsys.readouterr().out
    plan = json.loads(output)
    assert code == 0 and plan["dry_run"] is True and plan["executed"] is False
    assert plan["scheduled_rollouts"] == 2500
    assert plan["historical_replay"] is False
    assert plan["private_environment_file_supplied"] is True
    assert "private-test-sentinel" not in output and "private-test.invalid" not in output
    assert not (tmp_path / "new evaluation").exists()


def test_argument_array_uses_frozen_workload_and_keeps_whole_solve_timeout_unset(tmp_path, planner_path):
    plan = runner.build_plan(ROOT, tmp_path / "runtime with spaces", tmp_path / "output with spaces", planner_path=planner_path)
    argv = plan["argv"]
    assert argv[argv.index("--evaluation-profile") + 1] == "robocasa365-leaderboard-v1"
    assert argv[argv.index("--planning-profile") + 1] == "robocasa-horizon-gated-v1"
    assert argv[argv.index("--model") + 1] == "codexgateway:gpt-6-astra"
    assert argv[argv.index("--plan-generation-timeout-s") + 1] == "180"
    assert "--planner-timeout-s" not in argv and "--base-url" not in argv
    assert "--max-episode-steps" not in argv and "--task" not in argv
    assert plan == runner.build_plan(ROOT, tmp_path / "runtime with spaces", tmp_path / "output with spaces", planner_path=planner_path)


def test_plan_arguments_are_accepted_by_the_actual_frozen_cli_parser(tmp_path, planner_path):
    from robots.robocasa import _add_cli_args
    from rpent.cli.main import _build_argparser

    plan = runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", planner_path=planner_path)
    parser = _build_argparser()
    _add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(plan["argv"][3:])
    assert args.evaluation_profile == "robocasa365-leaderboard-v1"
    assert args.vla == "xiaomi-mibot" and args.jobs == 1
    assert args.seed is None and args.episodes is None and args.max_episode_steps is None


@pytest.mark.parametrize("change", [
    lambda data: data["workload"].update(rollouts_per_task=4),
    lambda data: data["workload"].update(scenario_seed=0),
    lambda data: data["workload"]["tasks"].pop(),
    lambda data: data.update(kind="historical_selected200_reference"),
    lambda data: data["execution"].update(protocol="new_protocol"),
    lambda data: data["simulator"].update(robocasa_commit="main"),
])
def test_changed_profile_or_historical_replay_is_rejected(tmp_path, planner_path, change):
    data = json.loads((ROOT / runner.PROFILE).read_text())
    change(data)
    profile = tmp_path / "changed.json"
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", profile_path=profile, planner_path=planner_path)


def test_execution_requires_explicit_api_settings_declaration_before_any_process(tmp_path, planner_path, monkeypatch):
    plan = runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", planner_path=planner_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Execution reached a process"))
    with pytest.raises(ValueError, match="api-settings-confirmed"):
        runner.execute_plan(plan, gateway_settings_confirmed=False, environment={})


def test_previous_output_is_refused_instead_of_retried_or_overwritten(tmp_path, planner_path, monkeypatch):
    output = tmp_path / "previous-result"
    output.mkdir()
    plan = runner.build_plan(ROOT, tmp_path / "runtime", output, planner_path=planner_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Previous result was relaunched"))
    with pytest.raises(ValueError, match="output.*exists"):
        runner.execute_plan(plan, gateway_settings_confirmed=True, environment={})


def test_execution_checks_private_environment_before_starting_any_process(tmp_path, planner_path, monkeypatch):
    plan = runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", planner_path=planner_path)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Missing credentials started a process"))
    with pytest.raises(ValueError, match="missing private environment"):
        runner.execute_plan(plan, gateway_settings_confirmed=True, environment={})


def test_execution_rejects_embedded_endpoint_credentials_without_echoing_them(tmp_path, planner_path, monkeypatch):
    plan = runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", planner_path=planner_path)
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
    environment = {"OPENAI_BASE_URL": "https://reviewer:password-sentinel@example.invalid/v1", "OPENAI_API_KEY": "key-sentinel"}
    with pytest.raises(ValueError) as raised:
        runner.execute_plan(plan, gateway_settings_confirmed=True, environment=environment)
    assert "password-sentinel" not in str(raised.value) and "key-sentinel" not in str(raised.value)


def test_injected_outer_timeout_cannot_change_planner_semantics(tmp_path, planner_path):
    config = json.loads(planner_path.read_text())
    config["cli"]["planner_timeout_s"] = 330
    planner_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="timeout"):
        runner.build_plan(ROOT, tmp_path / "runtime", tmp_path / "output", planner_path=planner_path)


def test_historical_archive_is_deterministic_bounded_and_preserves_canonical_bytes(tmp_path):
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    receipt = runner.archive_historical(ROOT, first)
    assert receipt["raw_archives_included"] is False and receipt["cases"] == 200
    runner.archive_historical(ROOT, second)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        names = archive.getnames()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
        assert all(not name.startswith((".git", "outputs/", "logs/")) for name in names)
        for name in runner.HISTORICAL_RECORDS:
            relative = "results/2.0-selected200/" + name
            assert archive.extractfile(relative).read() == (ROOT / relative).read_bytes()
        assert "evaluations/robocasa365/2.0/planner.json" in names
        assert "evaluations/robocasa365/2.0/planner-evidence.json" in names
    with pytest.raises(FileExistsError):
        runner.archive_historical(ROOT, first)


def test_tampered_historical_record_cannot_be_archived(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / "results/2.0-selected200", source / "results/2.0-selected200")
    (source / "results/2.0-selected200/version-identity.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.archive_historical(source, tmp_path / "bad.tar.gz")


def test_maintained_historical_manifest_cannot_silently_drop_overrides(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / "results/2.0-selected200", source / "results/2.0-selected200")
    folder = source / "evaluations/robocasa365/2.0"
    folder.mkdir(parents=True)
    for name in ("historical-selection.json", "historical-parameters.json"):
        shutil.copyfile(ROOT / "evaluations/robocasa365/2.0" / name, folder / name)
    parameters = json.loads((folder / "historical-parameters.json").read_text())
    parameters["selected_overrides"].pop()
    (folder / "historical-parameters.json").write_text(json.dumps(parameters))
    with pytest.raises(ValueError, match="historical parameter"):
        runner.verify_historical_configuration(source)
