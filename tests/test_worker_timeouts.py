"""Task-specific deadlines without running models or waiting for real timeouts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rpent.envs import get_env_spec
from rpent.evaluation.parallel import PolicyLane
from tests.test_episode_coordinator import _coordinator, _manifest, _Process
from tests.test_evaluation_cli import _parse, _robocasa_tasks


def _robocasa_manifest(tmp_path, horizons, *, ceilings=None, schedule=None):
    original = _manifest(tmp_path, task_count=len(horizons))
    return replace(
        original,
        environment={"name": "robocasa"},
        simulator={
            "tasks": [
                {"name": case.task_name, "upstream_horizon": horizon}
                for case, horizon in zip(original.cases, horizons, strict=True)
            ]
        },
        cases=tuple(
            replace(case, environment="robocasa", horizon=horizon)
            for case, horizon in zip(
                original.cases, horizons if ceilings is None else ceilings, strict=True
            )
        ),
        schedule=original.schedule if schedule is None else schedule,
    )


@pytest.mark.parametrize(
    ("horizon", "seconds"),
    [
        (1, 1800),
        (450, 1800),
        (750, 1800),
        (751, 3600),
        (1500, 3600),
        (1501, 5400),
        (2250, 5400),
        (2251, 7200),
        (2400, 7200),
        (3000, 7200),
        (3001, 9000),
        (3750, 9000),
        (3751, 10800),
        (4350, 10800),
        (4500, 10800),
        (4501, 12600),
    ],
)
def test_official_horizon_timeout_tier_boundaries(tmp_path, horizon, seconds):
    from rpent.evaluation.timeouts import resolve_worker_timeout_s

    manifest = _robocasa_manifest(tmp_path, [horizon])
    assert resolve_worker_timeout_s(manifest, manifest.cases[0]) == seconds


def test_deadline_uses_official_horizon_even_with_development_ceiling(tmp_path):
    from rpent.evaluation.timeouts import resolve_worker_timeout_s

    manifest = _robocasa_manifest(tmp_path, [2400], ceilings=[450])
    assert resolve_worker_timeout_s(manifest, manifest.cases[0]) == 7200
    assert resolve_worker_timeout_s(manifest, manifest.cases[0], override=37) == 37


def test_non_robocasa_retains_fixed_default(tmp_path):
    from rpent.evaluation.timeouts import resolve_worker_timeout_s

    manifest = _manifest(tmp_path)
    assert resolve_worker_timeout_s(manifest, manifest.cases[0]) == 1800


def test_older_manifest_falls_back_to_case_horizon(tmp_path):
    from rpent.evaluation.timeouts import resolve_worker_timeout_s

    manifest = replace(_robocasa_manifest(tmp_path, [4350]), simulator={})
    assert resolve_worker_timeout_s(manifest, manifest.cases[0]) == 10800


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "1800"])
def test_invalid_explicit_worker_deadlines_are_rejected(value):
    from rpent.evaluation.timeouts import normalize_worker_timeout_s

    with pytest.raises(ValueError, match="finite positive"):
        normalize_worker_timeout_s(value)


def test_grouped_worker_sums_automatic_case_budgets(tmp_path):
    from rpent.evaluation.timeouts import resolve_group_worker_timeout_s

    manifest = _robocasa_manifest(tmp_path, [2400, 2400])
    assert resolve_group_worker_timeout_s(manifest, manifest.cases) == 14400
    assert resolve_group_worker_timeout_s(manifest, manifest.cases, override=17) == 17
    fixed = replace(
        manifest, schedule={"worker_timeout": {"mode": "fixed", "seconds": 42}}
    )
    assert resolve_group_worker_timeout_s(fixed, fixed.cases) == 42


def test_manifest_timeout_policy_is_reused_and_explicit_override_wins(tmp_path):
    from rpent.evaluation.timeouts import resolve_worker_timeout_s

    manifest = _robocasa_manifest(
        tmp_path, [4350], schedule={"worker_timeout": {"mode": "fixed", "seconds": 42}}
    )
    manifest = manifest.write_new(tmp_path / "persisted.json")
    assert resolve_worker_timeout_s(manifest, manifest.cases[0]) == 42
    assert resolve_worker_timeout_s(manifest, manifest.cases[0], override=17) == 17


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("override", [None, 37])
def test_mixed_case_deadlines_are_dispatched_individually(
    tmp_path, monkeypatch, jobs, override
):
    manifest = _robocasa_manifest(tmp_path, [450, 2400, 4350])
    deadlines = {}

    def wait(process, *, timeout, cancellation):
        deadlines[process.case.horizon] = timeout
        return process.wait()

    monkeypatch.setattr("rpent.evaluation.coordinator._wait_for_worker", wait)

    def launch(*, command, case, result_path, env):
        return _Process(
            pid=2000 + case.ordinal,
            events=[],
            case=case,
            result_path=result_path,
            manifest=manifest,
            behavior="success",
            attempt_id=command[command.index("--attempt-id") + 1],
        )

    daemon = SimpleNamespace(
        endpoint="http://127.0.0.1:19000",
        start=lambda: None,
        stop=lambda: None,
        lanes=tuple(
            PolicyLane(f"http://127.0.0.1:{19000 + i}", {}) for i in range(jobs)
        ),
    )
    outcome = _coordinator(
        tmp_path, manifest, daemon, launch, jobs=jobs, worker_timeout_s=override
    ).run()
    assert outcome.completed == 3
    assert outcome.exit_code == 0
    assert deadlines == (
        {450: 1800, 2400: 7200, 4350: 10800}
        if override is None
        else {450: override, 2400: override, 4350: override}
    )


def test_timeout_result_reports_resolved_deadline(tmp_path, monkeypatch):
    import subprocess

    from rpent.evaluation.results import ResultStore

    manifest = _robocasa_manifest(tmp_path, [2400])

    def wait(process, *, timeout, cancellation):
        raise subprocess.TimeoutExpired("worker", timeout)

    monkeypatch.setattr("rpent.evaluation.coordinator._wait_for_worker", wait)

    def launch(*, command, case, result_path, env):
        return _Process(
            pid=2000,
            events=[],
            case=case,
            result_path=result_path,
            manifest=manifest,
            behavior="timeout",
            attempt_id=command[command.index("--attempt-id") + 1],
        )

    daemon = SimpleNamespace(
        endpoint="http://127.0.0.1:19000",
        start=lambda: None,
        stop=lambda: None,
    )
    _coordinator(tmp_path, manifest, daemon, launch, worker_timeout_s=None).run()
    result = ResultStore(tmp_path / "results.jsonl", manifest).load()[0]
    assert result.status == "timeout"
    assert result.error_message == "worker exceeded timeout 7200s"


@pytest.mark.parametrize("override", [None, 1234])
def test_cli_records_effective_timeout_policy(tmp_path, override):
    from rpent.evaluation.front_controller import build_evaluation_manifest

    argv = ["--env", "robocasa", "--task", "CloseFridge", "--output-dir", str(tmp_path)]
    if override is not None:
        argv.extend(["--worker-timeout-s", str(override)])
    args = _parse(argv)
    assert args.worker_timeout_s == override
    manifest, _ = build_evaluation_manifest(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=_robocasa_tasks(1),
        git_identity=lambda: ("a" * 40, False),
    )
    assert dict(manifest.schedule["worker_timeout"]) == (
        {"mode": "horizon_tiers", "steps_per_tier": 750, "seconds_per_tier": 1800}
        if override is None
        else {"mode": "fixed", "seconds": override}
    )


def test_robocasa_launch_preset_defaults_to_automatic_timeout():
    preset = get_env_spec("robocasa").launch_preset()
    field = next(field for field in preset.fields if field.name == "worker_timeout_s")
    assert not field.has_default
    assert not field.requires_value


def test_dashboard_launch_uses_automatic_worker_deadline():
    from rpent.ui.app import build_start_request
    from rpent.ui.routes.evaluations import StartEvaluationRequest

    request = build_start_request(
        StartEvaluationRequest(
            simulator_id="robocasa",
            config={"task": "CloseFridge", "planner": "api", "model": "openai:gpt-5.5"},
        ),
        "http://control-plane.test:8123",
    )
    assert request.args.worker_timeout_s is None
    assert "--worker-timeout-s" not in request.argv




def test_automatic_timeout_preparation_failure_remains_readable():
    from rpent.ui.service import _incomplete_preparation_failure

    error = _incomplete_preparation_failure(
        SimpleNamespace(summary=SimpleNamespace(timeouts=1)), None
    )
    assert isinstance(error, TimeoutError)
    assert "task-specific timeout" in str(error)
