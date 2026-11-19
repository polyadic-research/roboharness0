from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.competition import (
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.competition_worker import main, run_task_batch
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.tasks import (
    packaged_registry_snapshot,
    resolve_robocasa_tasks_from_contract,
)
from rpent.evaluation.batches import load_task_batch_attempts
from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.front_controller import build_evaluation_manifest
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult, ResultStore
from tests.test_robocasa_competition import _official_build_inputs


def _manifest(tmp_path: Path) -> EvaluationManifest:
    snapshot = packaged_registry_snapshot()
    profile = resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)
    tasks = resolve_robocasa_tasks_from_contract(
        ",".join(profile.task_names), scene_split=profile.scene_split
    )
    protocol = EpisodeProtocol(
        environment="robocasa",
        preset_name="robocasa365-pandaomron-smolvla-v1",
        horizon_ceiling=None,
        checkpoint_repo_id="lerobot/smolvla_base",
        checkpoint_revision="c" * 40,
        binding_name="robocasa-smolvla-v1",
    )
    cases = expand_episode_cases(
        tasks,
        protocol,
        episodes=50,
        base_seed=7,
        seed_policy="shared_environment_sequence",
    )
    return EvaluationManifest(
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "robocasa", "evaluation_profile": profile.profile_id},
        simulator={
            "name": "robocasa365",
            "evaluation_profile": profile.profile_id,
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "scene_split": "pretrain",
            "split": "pretrain",
            "robot": "PandaOmron",
            "object_registries": ("lightwheel",),
            "task_registry_digest": snapshot["registry_digest"],
            "tasks": tuple(
                {
                    "name": task.name,
                    "split": task.split,
                    "upstream_horizon": task.upstream_horizon,
                }
                for task in tasks
            ),
        },
        policy={"checkpoint": "fixed"},
        task_selector=",".join(profile.task_names),
        cases=cases,
        schedule={
            "evaluation_profile": profile.profile_id,
            "worker_grouping": "task",
            "base_seed": 7,
            "episodes_per_task": 50,
            "seed_policy": "shared_environment_sequence",
            "scene_split": "pretrain",
            "scenario_sampling": profile.scenario_sampling,
            "success_source": profile.success_source,
            "horizon_source": profile.horizon_source,
            "task_count": 50,
            "task_sets": profile.task_sets,
            "task_set_membership": {
                name: tuple(snapshot["groups"][name]["tasks"])
                for name in profile.task_sets
            },
            "task_registry_digest": snapshot["registry_digest"],
        },
        contracts={"action": {"shape": [12]}},
        runtime={
            "evaluation_root": str(tmp_path),
            "simulator_interpreter": "/opt/robocasa/bin/python",
            "simulator_endpoint": None,
            "rendering_backend": "egl",
            "policy_endpoint": "http://127.0.0.1:18000",
            "policy_endpoint_argument": "--policy-endpoint",
            "shared_policy_required": True,
        },
        created_at="2026-08-09T00:00:00+00:00",
    )


class _Daemon:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.endpoint = None
        self.start_calls = 0
        self.stop_calls = 0
        self.instances.append(self)

    def start(self):
        self.start_calls += 1
        self.endpoint = "http://127.0.0.1:19001"
        return object()

    def stop(self):
        self.stop_calls += 1


def _result(case, manifest, attempt_id, task_batch_attempt_id):
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        task_batch_attempt_id=task_batch_attempt_id,
        status="success",
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=123,
        worker_exit_code=0,
        started_at="2026-08-09T00:00:00+00:00",
        finished_at="2026-08-09T00:01:00+00:00",
        artifact_dir=str(
            Path(manifest.runtime["evaluation_root"])
            / "cases"
            / case.case_id
            / "attempts"
            / attempt_id
        ),
        simulator_success=True,
        contract_success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )


def _batch_paths(manifest, task_name, batch_attempt_id):
    root = (
        Path(manifest.runtime["evaluation_root"])
        / "task_batches"
        / task_name
        / batch_attempt_id
    )
    return root / "results.jsonl", root / "batch.jsonl"


def test_task_batch_reuses_one_environment_and_runs_fifty_fresh_cases(
    monkeypatch, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    results_path, ledger_path = _batch_paths(manifest, "CloseFridge", "batch-001")
    _Daemon.instances = []
    calls = []
    monkeypatch.setenv("RPENT_SHARED_POLICY_ENDPOINT", "http://127.0.0.1:18000")

    def case_runner(case, selected_manifest, **kwargs):
        calls.append(
            (
                case.case_id,
                kwargs["attempt_id"],
                kwargs["task_batch_attempt_id"],
                os.environ["RPENT_SHARED_ENV_ENDPOINT"],
                os.environ["RPENT_SHARED_ENV_SCENE_SPLIT"],
                os.environ["RPENT_SHARED_POLICY_ENDPOINT"],
            )
        )
        return _result(case, selected_manifest, **kwargs)

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-001",
        results_path,
        batch_ledger=ledger_path,
        daemon_factory=_Daemon,
        case_runner=case_runner,
    )

    assert exit_code == 0
    assert len(_Daemon.instances) == 1
    daemon = _Daemon.instances[0]
    assert daemon.start_calls == daemon.stop_calls == 1
    selected_fact = next(
        fact for fact in manifest.simulator["tasks"] if fact["name"] == "CloseFridge"
    )
    assert daemon.kwargs["config"] == {
        "task": "CloseFridge",
        "split": "pretrain",
        "seed": 7,
        "robot": "PandaOmron",
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_size": 256,
        "object_registries": ["lightwheel"],
        "rendering_mode": "headless",
        "upstream_horizon": selected_fact["upstream_horizon"],
        "safety_ceiling": selected_fact["upstream_horizon"],
        "task_registry_commit": ROBOCASA_COMMIT,
        "task_registry_digest": packaged_registry_snapshot()["registry_digest"],
    }
    selected_cases = tuple(
        case for case in manifest.cases if case.task_name == "CloseFridge"
    )
    assert [call[0] for call in calls] == [case.case_id for case in selected_cases]
    assert len({call[1] for call in calls}) == 50
    assert {call[2] for call in calls} == {"batch-001"}
    assert {call[3] for call in calls} == {"http://127.0.0.1:19001"}
    assert {call[4] for call in calls} == {"pretrain"}
    assert {call[5] for call in calls} == {"http://127.0.0.1:18000"}
    assert "RPENT_SHARED_ENV_ENDPOINT" not in os.environ
    assert "RPENT_SHARED_ENV_SCENE_SPLIT" not in os.environ
    results = ResultStore(results_path, manifest).load()
    assert len(results) == 50
    records = load_task_batch_attempts(ledger_path, manifest)
    assert [record.status for record in records] == ["running", "complete"]
    assert records[-1].completed_case_ids == tuple(
        case.case_id for case in selected_cases
    )


def test_external_batch_registers_every_case_before_daemon_creation(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    results_path, ledger_path = _batch_paths(
        manifest, "CloseFridge", "batch-registration"
    )
    selected = tuple(
        case for case in manifest.cases if case.task_name == "CloseFridge"
    )
    observed = False

    def daemon_factory(**_kwargs):
        nonlocal observed
        for case in selected:
            attempt_id = (
                f"batch-registration-{case.episode_index:03d}-{case.case_id[:12]}"
            )
            case_root = tmp_path / "cases" / case.case_id
            registration = json.loads(
                (case_root / "attempts" / attempt_id / "case.json").read_text()
            )
            locator = json.loads((case_root / "current_attempt.json").read_text())
            assert locator["attempt_id"] == attempt_id
            assert locator["registered_at"] == registration["started_at"]
        observed = True
        raise RuntimeError("registration observed")

    assert run_task_batch(
        manifest,
        "CloseFridge",
        "batch-registration",
        results_path,
        batch_ledger=ledger_path,
        daemon_factory=daemon_factory,
    ) == 1
    assert observed is True


def test_max_length_batch_id_derives_a_safe_case_attempt_id(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    batch_id = "b" * 128
    results, ledger = _batch_paths(manifest, "CloseFridge", batch_id)
    seen = []

    def stop_after_first(case, selected, **kwargs):
        del case, selected
        seen.append(kwargs["attempt_id"])
        raise RuntimeError("stop")

    assert (
        run_task_batch(
            manifest,
            "CloseFridge",
            batch_id,
            results,
            batch_ledger=ledger,
            daemon_factory=_Daemon,
            case_runner=stop_after_first,
        )
        != 0
    )
    assert seen and seen[0].startswith(batch_id + "-000-")

    too_long = "b" * 129
    bad_results, bad_ledger = _batch_paths(manifest, "CloseFridge", too_long)
    with pytest.raises(ValueError, match="task_batch_attempt_id"):
        run_task_batch(
            manifest,
            "CloseFridge",
            too_long,
            bad_results,
            batch_ledger=bad_ledger,
            daemon_factory=_Daemon,
            case_runner=stop_after_first,
        )


def test_case_seventeen_crash_retains_prefix_and_fails_batch(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    results_path, ledger_path = _batch_paths(manifest, "CloseFridge", "batch-crash")
    _Daemon.instances = []
    calls = 0

    def case_runner(case, selected_manifest, **kwargs):
        nonlocal calls
        if calls == 17:
            raise RuntimeError("worker crashed")
        calls += 1
        return _result(case, selected_manifest, **kwargs)

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-crash",
        results_path,
        batch_ledger=ledger_path,
        daemon_factory=_Daemon,
        case_runner=case_runner,
    )

    assert exit_code != 0
    assert len(ResultStore(results_path, manifest).load()) == 17
    records = load_task_batch_attempts(ledger_path, manifest)
    assert [record.status for record in records] == ["running", "failed"]
    selected_cases = tuple(
        case for case in manifest.cases if case.task_name == "CloseFridge"
    )
    assert records[-1].completed_case_ids == tuple(
        case.case_id for case in selected_cases[:17]
    )
    assert _Daemon.instances[0].stop_calls == 1


def test_cancellation_retains_only_real_results_and_marks_cancelled(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    results_path, ledger_path = _batch_paths(manifest, "CloseFridge", "batch-cancel")
    _Daemon.instances = []
    calls = 0

    def case_runner(case, selected_manifest, **kwargs):
        nonlocal calls
        if calls == 2:
            raise KeyboardInterrupt
        calls += 1
        return _result(case, selected_manifest, **kwargs)

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-cancel",
        results_path,
        batch_ledger=ledger_path,
        daemon_factory=_Daemon,
        case_runner=case_runner,
    )

    assert exit_code == 130
    results = ResultStore(results_path, manifest).load()
    assert len(results) == 2
    assert all(result.simulator_success for result in results)
    records = load_task_batch_attempts(ledger_path, manifest)
    assert records[-1].status == "cancelled"
    selected_cases = tuple(
        case for case in manifest.cases if case.task_name == "CloseFridge"
    )
    assert records[-1].completed_case_ids == tuple(
        case.case_id for case in selected_cases[:2]
    )
    assert _Daemon.instances[0].stop_calls == 1


def test_cancelled_episode_evidence_is_appended_but_not_completed(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-cancel-evidence")

    def runner(case, selected, **kwargs):
        result = replace(
            _result(case, selected, **kwargs),
            status="cancelled",
            worker_exit_code=130,
            simulator_success=False,
            contract_success=False,
            terminated=False,
            error_type="KeyboardInterrupt",
            error_message="cancelled",
        )
        error = KeyboardInterrupt()
        error.rpent_result = result
        raise error

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-cancel-evidence",
        results,
        batch_ledger=ledger,
        daemon_factory=_Daemon,
        case_runner=runner,
    )

    assert exit_code == 130
    retained = ResultStore(results, manifest).load()
    assert len(retained) == 1 and retained[0].status == "cancelled"
    terminal = load_task_batch_attempts(ledger, manifest)[-1]
    assert terminal.status == "cancelled"
    assert terminal.completed_case_ids == ()


def test_daemon_construction_failure_still_writes_failed_terminal_record(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path)
    results_path, ledger_path = _batch_paths(manifest, "CloseFridge", "batch-startup")

    def fail_factory(**_kwargs):
        raise RuntimeError("cannot construct daemon")

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-startup",
        results_path,
        batch_ledger=ledger_path,
        daemon_factory=fail_factory,
    )

    assert exit_code != 0
    records = load_task_batch_attempts(ledger_path, manifest)
    assert [record.status for record in records] == ["running", "failed"]
    assert records[-1].completed_case_ids == ()


def test_worker_cli_loads_manifest_and_forwards_all_batch_paths(
    monkeypatch, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.write_new(manifest_path)
    calls = []

    def fake_run(loaded, task, batch_attempt_id, results, *, batch_ledger):
        calls.append(
            (loaded.manifest_id, task, batch_attempt_id, results, batch_ledger)
        )
        return 23

    monkeypatch.setattr(
        "robots.robocasa.competition_worker.run_task_batch",
        fake_run,
    )
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-cli")

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--task",
            "CloseFridge",
            "--batch-attempt-id",
            "batch-cli",
            "--results",
            str(results),
            "--batch-ledger",
            str(ledger),
        ]
    )

    assert exit_code == 23
    assert calls == [
        (
            manifest.manifest_id,
            "CloseFridge",
            "batch-cli",
            results,
            ledger,
        )
    ]


def test_missing_required_shared_policy_fails_before_ledger_or_daemon(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    manifest = replace(
        manifest,
        runtime={**dict(manifest.runtime), "policy_endpoint": None},
    )
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-no-policy")
    _Daemon.instances = []

    with pytest.raises(ValueError, match="shared policy"):
        run_task_batch(
            manifest,
            "CloseFridge",
            "batch-no-policy",
            results,
            batch_ledger=ledger,
            daemon_factory=_Daemon,
        )

    assert not ledger.exists()
    assert not results.exists()
    assert _Daemon.instances == []


@pytest.mark.parametrize("location", ["central", "external"])
def test_batch_output_paths_are_attempt_local_before_any_read(
    location, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    expected_results, expected_ledger = _batch_paths(
        manifest, "CloseFridge", "batch-path"
    )
    if location == "central":
        results = tmp_path / "results.jsonl"
        ledger = expected_ledger
    else:
        results = expected_results
        ledger = tmp_path.parent / "outside-batch.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text("do-not-read", encoding="utf-8")
    _Daemon.instances = []

    with pytest.raises(ValueError, match="attempt-local"):
        run_task_batch(
            manifest,
            "CloseFridge",
            "batch-path",
            results,
            batch_ledger=ledger,
            daemon_factory=_Daemon,
        )

    assert results.read_text(encoding="utf-8") == "do-not-read"
    assert not ledger.exists()
    assert _Daemon.instances == []


@pytest.mark.parametrize(
    ("status", "error_type"),
    [("error", "RuntimeError"), ("timeout", "TimeoutError")],
)
def test_infrastructure_result_is_retained_then_fails_without_continuing(
    status, error_type, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    batch_id = f"batch-{status}"
    results, ledger = _batch_paths(manifest, "CloseFridge", batch_id)
    calls = []

    def runner(case, selected, **kwargs):
        calls.append(case.case_id)
        return replace(
            _result(case, selected, **kwargs),
            status=status,
            worker_exit_code=1,
            simulator_success=False,
            contract_success=False,
            terminated=False,
            error_type=error_type,
            error_message="transport died",
        )

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        batch_id,
        results,
        batch_ledger=ledger,
        daemon_factory=_Daemon,
        case_runner=runner,
    )

    assert exit_code != 0
    assert len(calls) == 1
    retained = ResultStore(results, manifest).load()
    assert len(retained) == 1 and retained[0].status == status
    terminal = load_task_batch_attempts(ledger, manifest)[-1]
    assert terminal.status == "failed"
    assert terminal.completed_case_ids == ()


@pytest.mark.parametrize(
    ("simulator_success", "contract_success"),
    [(True, False), (False, True)],
)
def test_success_authority_divergence_is_still_a_completed_rollout(
    simulator_success, contract_success, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    batch_id = f"batch-divergence-{int(simulator_success)}"
    results, ledger = _batch_paths(manifest, "CloseFridge", batch_id)
    first = True

    def runner(case, selected, **kwargs):
        nonlocal first
        result = _result(case, selected, **kwargs)
        if first:
            first = False
            return replace(
                result,
                status="error",
                simulator_success=simulator_success,
                contract_success=contract_success,
                terminated=True,
            )
        return result

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        batch_id,
        results,
        batch_ledger=ledger,
        daemon_factory=_Daemon,
        case_runner=runner,
    )

    assert exit_code == 0
    assert len(ResultStore(results, manifest).load()) == 50
    assert load_task_batch_attempts(ledger, manifest)[-1].status == "complete"


def test_fatal_base_exception_propagates_with_running_only_evidence(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-fatal")

    class FatalDaemon(_Daemon):
        def start(self):
            raise SystemExit(99)

    with pytest.raises(SystemExit):
        run_task_batch(
            manifest,
            "CloseFridge",
            "batch-fatal",
            results,
            batch_ledger=ledger,
            daemon_factory=FatalDaemon,
        )

    assert [record.status for record in load_task_batch_attempts(ledger, manifest)] == [
        "running"
    ]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("environment", "evaluation_profile", "development"),
        ("simulator", "evaluation_profile", "development"),
        ("schedule", "episodes_per_task", 49),
        ("schedule", "seed_policy", "base_seed_plus_episode_index"),
        ("schedule", "scenario_sampling", "wrong"),
        ("schedule", "base_seed", 8),
        ("schedule", "scene_split", "target"),
        ("schedule", "success_source", "contract"),
        ("schedule", "horizon_source", "ceiling"),
        ("schedule", "worker_grouping", "case"),
        ("schedule", "task_set_membership", {}),
        ("schedule", "task_registry_digest", "sha256:" + "0" * 64),
        ("runtime", "shared_policy_required", False),
    ],
)
def test_official_worker_rejects_mutated_profile_facts_before_daemon(
    section, field, value, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    changed = {**dict(getattr(manifest, section)), field: value}
    manifest = replace(manifest, **{section: changed})
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-mutated")
    _Daemon.instances = []

    with pytest.raises(ValueError, match="official|profile|schedule|registry"):
        run_task_batch(
            manifest,
            "CloseFridge",
            "batch-mutated",
            results,
            batch_ledger=ledger,
            daemon_factory=_Daemon,
        )

    assert _Daemon.instances == []
    assert not ledger.exists()


def test_official_worker_rejects_reordered_task_membership(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    simulator = {
        **dict(manifest.simulator),
        "tasks": tuple(reversed(manifest.simulator["tasks"])),
    }
    manifest = replace(manifest, simulator=simulator)
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-reordered")

    with pytest.raises(ValueError, match="exact 50-task membership"):
        run_task_batch(
            manifest,
            "CloseFridge",
            "batch-reordered",
            results,
            batch_ledger=ledger,
            daemon_factory=_Daemon,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_id", "foreign-evaluation"),
        ("manifest_id", "f" * 64),
        ("task_name", "foreign-task"),
        ("seed", 8),
        ("episode_index", 49),
        ("checkpoint_revision", "f" * 40),
        ("binding_name", "foreign-binding"),
        ("artifact_dir", "/tmp/foreign-artifact"),
    ],
)
def test_batch_result_rejects_every_drifting_case_identity(
    field, value, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    batch_id = f"batch-drift-{field.replace('_', '-')}"
    results, ledger = _batch_paths(manifest, "CloseFridge", batch_id)

    def runner(case, selected, **kwargs):
        return replace(_result(case, selected, **kwargs), **{field: value})

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        batch_id,
        results,
        batch_ledger=ledger,
        daemon_factory=_Daemon,
        case_runner=runner,
    )

    assert exit_code != 0
    assert ResultStore(results, manifest).load() == ()
    assert load_task_batch_attempts(ledger, manifest)[-1].status == "failed"


def test_default_worker_cli_recreates_episode_and_policy_session_fifty_times(
    monkeypatch, tmp_path
) -> None:
    import importlib

    from robots.robocasa import runtime as robocasa_runtime
    from robots.robocasa.binding import make_robocasa_binding

    cli = importlib.import_module("rpent.cli.main")
    metadata = tmp_path / "submission-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "model_name": "RPent SmolVLA",
                "policy_family": "SmolVLA",
                "date": "2026-08-09",
                "submission_source": "external",
                "code_url": "N/A",
                "commit_hash": "N/A",
                "paper_link": "N/A",
                "open_source": "yes",
                "training_config": {
                    "batch_size": 64,
                    "num_training_steps": 100_000,
                },
                "notes": "N/A",
            }
        ),
        encoding="utf-8",
    )
    _profile, _selector, args, env_spec, launch = _official_build_inputs(tmp_path)
    args.submission_metadata_json = str(metadata)
    manifest, _run_config = build_evaluation_manifest(
        args,
        env_spec,
        [
            "--env",
            "robocasa",
            "--evaluation-profile",
            LEADERBOARD_PROFILE_ID,
            "--submission-metadata-json",
            str(metadata),
        ],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )
    manifest = replace(
        manifest,
        recording={"schema_version": 1, "mode": "none", "camera_ids": ()},
    )
    metadata.unlink()
    assert str(metadata) not in manifest.cli_arguments["argv"]
    assert "--submission-metadata-json" not in manifest.cli_arguments["argv"]
    monkeypatch.setenv("RPENT_SHARED_POLICY_ENDPOINT", "http://127.0.0.1:18000")
    results, ledger = _batch_paths(manifest, "CloseFridge", "batch-production")
    bootstraps = []
    simulators = []
    policy_sessions = []
    harnesses = []
    endpoints = []

    observation = {
        "state": np.zeros(16, dtype=np.float32),
        "robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
    }

    class Simulator:
        def __init__(self, episode_index):
            self.episode_id = f"episode-{episode_index}"
            self.instruction = "close the fridge"
            self.effective_horizon = 900
            self.reset_calls = []

        def reset(self, task, seed):
            self.reset_calls.append((task, seed))
            return observation

        def close(self):
            return None

    class PolicySession:
        def __init__(self):
            self.reset_calls = []

        def reset(self, episode_id):
            self.reset_calls.append(episode_id)

        def close(self):
            return None

    class Dependency:
        def __init__(self, resource):
            self.resource = resource

        def start(self):
            return self.resource

        def stop(self):
            return None

    class Toolkit:
        skill_outcome_adapter_id = "test-adapter"
        evaluation_metrics = {
            "terminated": True,
            "truncated": False,
            "step_count": 1,
            "action_count": 1,
        }

        def attach_transition_observer(self, observer):
            self.observer = observer

        def attach_event_recorder(self, recorder):
            self.recorder = recorder

        def set_context_policy(self, policy):
            self.context_policy = policy

        def get_tools_spec(self):
            return []

        def write_recipe(self, recipe_tag):
            return tmp_path / f"{recipe_tag}.json"

        def close(self):
            return None

    class HarnessResult:
        def snapshot(self):
            return {
                "state": "terminal_failure",
                "contract_success": False,
                "environment_success": False,
            }

    original_build = robocasa_runtime.build_episode_bootstrap

    def build_bootstrap(args, output_dir, dashboard):
        def environment_builder(**kwargs):
            assert kwargs["task"].split == "pretrain"
            simulator = Simulator(len(simulators))
            simulators.append(simulator)
            endpoints.append(
                (
                    os.environ["RPENT_SHARED_ENV_ENDPOINT"],
                    os.environ["RPENT_SHARED_ENV_SCENE_SPLIT"],
                )
            )
            dependency = Dependency(simulator)
            return dependency, dependency.start, {}

        def policy_builder(**_kwargs):
            policy = PolicySession()
            policy_sessions.append(policy)
            return Dependency(policy)

        bootstrap = original_build(
            args,
            output_dir,
            dashboard,
            task_resolver=resolve_robocasa_tasks_from_contract,
            environment_builder=environment_builder,
            policy_daemon_builder=policy_builder,
            toolkit_builder=lambda *_args, **_kwargs: Toolkit(),
        )
        assert bootstrap.binding == make_robocasa_binding()
        bootstraps.append(bootstrap)
        return bootstrap

    class Orchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, **_kwargs):
            return SimpleNamespace(
                finish_result=None,
                messages=[],
                stats={},
                error=None,
            )

    def attach_harness(**_kwargs):
        harness = HarnessResult()
        harnesses.append(harness)
        return harness, None

    monkeypatch.setattr(robocasa_runtime, "build_episode_bootstrap", build_bootstrap)
    monkeypatch.setattr(cli, "validate_runtime_requirements", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "ensure_resources", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "_build_planning_runtime_bundle",
        lambda **_kwargs: cli.PlanningRuntimeBundle(
            executor=object(),
            plan_generator=None,
            predicate_verifier=None,
            metadata={},
        ),
    )
    monkeypatch.setattr(cli, "PlanningOrchestrator", Orchestrator)
    monkeypatch.setattr(cli, "_attach_planning_harness", attach_harness)
    monkeypatch.setattr(cli, "build_experiment_manifest", lambda **_kwargs: {})

    exit_code = run_task_batch(
        manifest,
        "CloseFridge",
        "batch-production",
        results,
        batch_ledger=ledger,
        daemon_factory=_Daemon,
    )

    assert exit_code == 0
    assert len(bootstraps) == len({id(item) for item in bootstraps}) == 50
    assert len(harnesses) == len({id(item) for item in harnesses}) == 50
    assert len(simulators) == len({id(item) for item in simulators}) == 50
    assert [simulator.reset_calls for simulator in simulators] == [
        [("CloseFridge", 7)] for _ in range(50)
    ]
    assert len(policy_sessions) == len({id(item) for item in policy_sessions}) == 50
    assert [session.reset_calls for session in policy_sessions] == [
        [f"episode-{index}"] for index in range(50)
    ]
    assert set(endpoints) == {("http://127.0.0.1:19001", "pretrain")}
    assert len(ResultStore(results, manifest).load()) == 50
    assert load_task_batch_attempts(ledger, manifest)[-1].status == "complete"
