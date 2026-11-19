from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from rpent.evaluation.batches import (
    TaskBatchAttempt,
    TaskBatchProcessOutcome,
    append_task_batch_attempt,
    append_task_batch_process_outcome,
)
from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.coordinator import (
    EvaluationCoordinator,
    _default_worker_launcher,
    _validate_worker_case,
    _write_qualification_projection,
)
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.result_acceptance import (
    FinalizedResultCandidate,
    ReapedWorker,
)
from rpent.evaluation.results import EpisodeResult, ResultStore, aggregate_results
from rpent.planning.manifest import canonical_json
from tests.synthetic_tasks import SyntheticTask


def _manifest(tmp_path: Path, task_count: int = 3, episodes: int = 1, **overrides):
    tasks = tuple(
        SyntheticTask(
            suite="libero_10",
            index=index,
            name=f"task_{index}",
            language=f"instruction {index}",
            bddl_identifier=f"task_{index}.bddl",
            upstream_horizon=500,
        )
        for index in range(task_count)
    )
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="robocerebra-libero10-smolvla-v1",
        horizon_ceiling=20,
        checkpoint_repo_id="lerobot/smolvla_robocerebra",
        checkpoint_revision="7" * 40,
        binding_name="robocerebra-smolvla-v1",
    )
    values = {
        "rpent_commit": "a" * 40,
        "rpent_dirty": False,
        "cli_arguments": {"argv": ["--planner", "api"]},
        "environment": {"name": "synthetic"},
        "simulator": {"revision": "b" * 40},
        "policy": {"lerobot_revision": "c" * 40},
        "task_selector": "all",
        "cases": expand_episode_cases(tasks, protocol, episodes=episodes, base_seed=0),
        "schedule": {"episodes_per_task": episodes, "base_seed": 0},
        "contracts": {"action": {"shape": [7]}},
        "runtime": {
            "main_interpreter": "/opt/rpent/bin/python",
            "evaluation_root": str(tmp_path),
        },
        "created_at": "2026-08-05T12:00:00+10:00",
    }
    values.update(overrides)
    return EvaluationManifest(**values)


def _result(
    manifest,
    case,
    *,
    status="terminal_failure",
    pid=1000,
    attempt_id=None,
):
    success = status == "success"
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id or f"attempt-{pid}",
        status=status,
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=pid,
        worker_exit_code=0,
        started_at="2026-08-05T12:00:00+10:00",
        finished_at="2026-08-05T12:01:00+10:00",
        artifact_dir=str(
            Path(manifest.runtime["evaluation_root"])
            / "cases"
            / case.case_id
            / "attempts"
            / (attempt_id or f"attempt-{pid}")
        ),
        simulator_success=success,
        contract_success=success,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )


class _PolicyDaemon:
    name = "smolvla"
    endpoint = "http://127.0.0.1:19000"

    def __init__(self, events, manifest_path):
        self.events = events
        self.manifest_path = manifest_path
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        assert self.manifest_path.exists()
        self.start_calls += 1
        self.events.append("policy:start")

    def stop(self):
        self.stop_calls += 1
        self.events.append("policy:stop")


class _Process:
    def __init__(
        self,
        *,
        pid,
        events,
        case,
        result_path,
        manifest,
        behavior,
        attempt_id,
    ):
        self.pid = pid
        self.events = events
        self.case = case
        self.result_path = result_path
        self.manifest = manifest
        self.behavior = behavior
        self.attempt_id = attempt_id
        self.terminated = False
        self.killed = False
        self.reaped = False
        self.wait_calls = 0
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        self.events.append(f"wait:{self.case.ordinal}")
        if self.killed:
            if self.behavior == "interrupt_with_success_cleanup_reap_error":
                raise OSError("wait after kill failed")
            self.reaped = True
            return -9
        if self.behavior == "timeout":
            raise subprocess.TimeoutExpired("worker", timeout)
        if self.behavior == "interrupt_with_result":
            if self.wait_calls == 1:
                result = _result(
                    self.manifest,
                    self.case,
                    status="cancelled",
                    pid=self.pid,
                    attempt_id=self.attempt_id,
                )
                self.result_path.parent.mkdir(parents=True, exist_ok=True)
                self.result_path.write_text(
                    canonical_json(result.to_payload()) + "\n",
                    encoding="utf-8",
                )
                raise KeyboardInterrupt
            return 0
        if self.behavior in {
            "interrupt_with_success",
            "interrupt_with_terminal_failure",
        }:
            if self.wait_calls == 1:
                result = _result(
                    self.manifest,
                    self.case,
                    status=self.behavior.removeprefix("interrupt_with_"),
                    pid=self.pid,
                    attempt_id=self.attempt_id,
                )
                self.result_path.parent.mkdir(parents=True, exist_ok=True)
                self.result_path.write_text(
                    canonical_json(result.to_payload()) + "\n",
                    encoding="utf-8",
                )
                raise KeyboardInterrupt
            return 0
        if self.behavior in {
            "interrupt_with_success_cleanup_timeout",
            "interrupt_with_success_cleanup_reap_error",
        }:
            if self.wait_calls == 1:
                result = _result(
                    self.manifest,
                    self.case,
                    status="success",
                    pid=self.pid,
                    attempt_id=self.attempt_id,
                )
                self.result_path.parent.mkdir(parents=True, exist_ok=True)
                self.result_path.write_text(
                    canonical_json(result.to_payload()) + "\n",
                    encoding="utf-8",
                )
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired("worker", timeout)
        if self.behavior == "interrupt_then_success":
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            result = _result(
                self.manifest,
                self.case,
                status="success",
                pid=self.pid,
                attempt_id=self.attempt_id,
            )
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                canonical_json(result.to_payload()) + "\n",
                encoding="utf-8",
            )
            return 0
        if self.behavior == "interrupt":
            raise KeyboardInterrupt
        if self.behavior != "missing":
            result = _result(
                self.manifest,
                self.case,
                status=(
                    "success" if self.behavior == "success" else "terminal_failure"
                ),
                pid=self.pid,
                attempt_id=self.attempt_id,
            )
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.write_text(
                canonical_json(result.to_payload()) + "\n",
                encoding="utf-8",
            )
        return 3 if self.behavior == "missing" else 0

    def terminate(self):
        self.terminated = True
        self.events.append(f"terminate:{self.case.ordinal}")

    def kill(self):
        self.killed = True
        self.events.append(f"kill:{self.case.ordinal}")


class _Launcher:
    def __init__(self, manifest, events, behaviors, store=None):
        self.manifest = manifest
        self.events = events
        self.behaviors = iter(behaviors)
        self.processes = []
        self.store = store
        self.environments = []

    def __call__(self, *, command, case, result_path, env):
        if self.store is not None:
            assert len(self.store.load()) == len(self.processes)
        self.events.append(f"launch:{case.ordinal}")
        self.environments.append(dict(env))
        assert command[:3] == [
            "/opt/rpent/bin/python",
            "-m",
            "rpent.evaluation.worker",
        ]
        assert command[command.index("--case-id") + 1] == case.case_id
        attempt_id = command[command.index("--attempt-id") + 1]
        attempt_root = (
            Path(self.manifest.runtime["evaluation_root"])
            / "cases"
            / case.case_id
            / "attempts"
            / attempt_id
        )
        registration = json.loads((attempt_root / "case.json").read_text())
        locator = json.loads(
            (attempt_root.parents[1] / "current_attempt.json").read_text()
        )
        assert registration["started_at"] == locator["registered_at"]
        assert locator["attempt_id"] == attempt_id
        assert result_path.name.endswith(f"-{attempt_id}.json")
        assert env["RPENT_SHARED_POLICY_ENDPOINT"] == "http://127.0.0.1:19000"
        assert env["RPENT_SHARED_POLICY_LOADED"] == "1"
        process = _Process(
            pid=2000 + case.ordinal,
            events=self.events,
            case=case,
            result_path=result_path,
            manifest=self.manifest,
            behavior=next(self.behaviors),
            attempt_id=attempt_id,
        )
        self.processes.append(process)
        return process


def _coordinator(tmp_path, manifest, daemon, launcher, **overrides):
    values = {
        "manifest": manifest,
        "manifest_path": tmp_path / "manifest.json",
        "results_path": tmp_path / "results.jsonl",
        "summary_path": tmp_path / "summary.json",
        "policy_daemon": daemon,
        "worker_launcher": launcher,
        "worker_timeout_s": 30,
        "jobs": 1,
        "progress": lambda *_args, **_kwargs: None,
    }
    values.update(overrides)
    return EvaluationCoordinator(**values)


@pytest.mark.parametrize(
    ("behavior", "expected_status"),
    [("success", "success"), ("missing", "error"), ("timeout", "timeout")],
)
def test_coordinator_reaps_before_acceptance_for_normal_timeout_and_error(
    tmp_path, behavior, expected_status
):
    events = []
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest, events, [behavior])
    accepted = []

    def factory(_manifest, store):
        class Acceptance:
            def accept(self, candidate):
                assert isinstance(candidate, FinalizedResultCandidate)
                assert isinstance(candidate.completion, ReapedWorker)
                process = launcher.processes[0]
                assert process.wait_calls > 0
                if behavior == "timeout":
                    assert process.reaped is True
                store.append_or_confirm(candidate.result)
                accepted.append(candidate)
                return candidate.result

        return Acceptance()

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
        worker_timeout_s=0.001 if behavior == "timeout" else 30,
        result_acceptance_factory=factory,
    ).run()

    assert outcome.completed == 1
    assert [item.result.status for item in accepted] == [expected_status]


def test_qualification_projection_health_does_not_expose_artifact_paths(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-secret")
    output = tmp_path / "qualification_summary.json"

    _write_qualification_projection(output, manifest, (result,))

    payload = json.loads(output.read_text())
    issues = payload["cases"][0]["artifact_health"]["issues"]
    assert issues == ["qualification projection unavailable"]
    assert str(tmp_path) not in json.dumps(issues)


def test_coordinator_writes_manifest_starts_policy_once_and_runs_in_order(tmp_path):
    events = []
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    daemon = _PolicyDaemon(events, manifest_path)
    store = ResultStore(tmp_path / "results.jsonl", manifest)
    launcher = _Launcher(
        manifest,
        events,
        ["success", "terminal_failure", "success"],
        store=store,
    )
    coordinator = _coordinator(tmp_path, manifest, daemon, launcher)

    outcome = coordinator.run()

    assert outcome.exit_code == 0
    assert outcome.completed == 3
    assert daemon.start_calls == 1
    assert daemon.stop_calls == 1
    assert events == [
        "policy:start",
        "launch:0",
        "wait:0",
        "launch:1",
        "wait:1",
        "launch:2",
        "wait:2",
        "policy:stop",
    ]
    assert [result.status for result in store.load()] == [
        "success",
        "terminal_failure",
        "success",
    ]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["scheduled"] == 3
    assert summary["successful"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)


def test_case_worker_receives_its_internal_attempt_console_path(tmp_path):
    events = []
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest, events, ["success"])

    _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    attempt_id = launcher.processes[0].attempt_id
    assert launcher.environments[0]["RPENT_WORKER_LOG_PATH"] == str(
        (
            tmp_path
            / "cases"
            / manifest.cases[0].case_id
            / "attempts"
            / attempt_id
            / "console.log"
        ).resolve()
    )


def test_default_launcher_merges_output_without_leaking_internal_env(
    tmp_path, monkeypatch
):
    console = tmp_path / "attempt" / "console.log"
    console.parent.mkdir()
    observed = {}

    def popen(command, **kwargs):
        observed.update(kwargs)
        assert kwargs["stdout"].closed is False
        kwargs["stdout"].write(b"stdout\nstderr\n")
        return object()

    monkeypatch.setattr(subprocess, "Popen", popen)

    _default_worker_launcher(
        command=["python", "worker.py"],
        case=_manifest(tmp_path, task_count=1).cases[0],
        result_path=tmp_path / "result.json",
        env={"VISIBLE": "yes", "RPENT_WORKER_LOG_PATH": str(console)},
    )

    assert observed["env"] == {"VISIBLE": "yes"}
    assert observed["stderr"] is subprocess.STDOUT
    assert observed["stdout"].closed is True
    assert console.read_bytes() == b"stdout\nstderr\n"


def test_default_launcher_closes_console_when_process_start_fails(
    tmp_path, monkeypatch
):
    console = tmp_path / "attempt" / "console.log"
    console.parent.mkdir()
    observed = {}

    def popen(_command, **kwargs):
        observed["stdout"] = kwargs["stdout"]
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "Popen", popen)

    with pytest.raises(OSError, match="cannot start"):
        _default_worker_launcher(
            command=["python", "worker.py"],
            case=_manifest(tmp_path, task_count=1).cases[0],
            result_path=tmp_path / "result.json",
            env={"RPENT_WORKER_LOG_PATH": str(console)},
        )

    assert observed["stdout"].closed is True


def test_coordinator_merges_only_allowlisted_ephemeral_worker_environment(
    tmp_path, monkeypatch
):
    events = []
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest, events, ["success"])
    monkeypatch.setenv("RPENT_EVALUATION_ID", "stale-evaluation")
    coordinator = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
        worker_environment={
            "RPENT_NOTIFICATION_TOKEN": "worker-secret",
            "RPENT_PREPARATION_PATH": str(tmp_path / "preparation.json"),
            "RPENT_EVALUATION_ID": manifest.evaluation_id,
            "RPENT_MANIFEST_ID": manifest.manifest_id,
        },
    )

    coordinator.run()

    assert launcher.environments[0]["RPENT_NOTIFICATION_TOKEN"] == "worker-secret"
    assert launcher.environments[0]["RPENT_PREPARATION_PATH"] == str(
        tmp_path / "preparation.json"
    )
    assert launcher.environments[0]["RPENT_EVALUATION_ID"] == manifest.evaluation_id
    assert launcher.environments[0]["RPENT_MANIFEST_ID"] == manifest.manifest_id
    assert "worker-secret" not in (tmp_path / "manifest.json").read_text()
    with pytest.raises(ValueError, match="worker environment"):
        _coordinator(
            tmp_path,
            manifest,
            None,
            launcher,
            worker_environment={"UNSAFE": "value"},
        )


def test_timeout_and_missing_result_are_retained_then_scheduling_continues(tmp_path):
    events = []
    manifest = _manifest(tmp_path)
    daemon = _PolicyDaemon(events, tmp_path / "manifest.json")
    launcher = _Launcher(manifest, events, ["timeout", "missing", "success"])

    outcome = _coordinator(
        tmp_path,
        manifest,
        daemon,
        launcher,
        worker_timeout_s=0.001,
    ).run()

    assert outcome.exit_code != 0
    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert [result.status for result in results] == ["timeout", "error", "success"]
    timeout_process = launcher.processes[0]
    assert timeout_process.wait_timeouts[0] <= 0.001
    assert timeout_process.wait_timeouts[-2:] == [5.0, 5.0]
    assert timeout_process.killed is True
    assert timeout_process.reaped is True
    timeout = results[0]
    assert timeout.attempt_id == launcher.processes[0].attempt_id
    assert Path(timeout.artifact_dir).name == timeout.attempt_id
    assert Path(timeout.artifact_dir).is_dir()
    assert launcher.processes[0].terminated is True
    assert "launch:2" in events


def test_resume_skips_only_valid_retained_cases(tmp_path):
    events = []
    manifest = _manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    store = ResultStore(tmp_path / "results.jsonl", manifest)
    store.append(_result(manifest, manifest.cases[0], status="success"))
    daemon = _PolicyDaemon(events, tmp_path / "manifest.json")
    launcher = _Launcher(manifest, events, ["success", "success"])

    outcome = _coordinator(tmp_path, manifest, daemon, launcher).run(resume=True)

    assert outcome.completed == 3
    assert [process.case.ordinal for process in launcher.processes] == [1, 2]
    assert len(store.load()) == 3


def test_resume_with_already_accepted_result_repairs_missing_snapshot(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    manifest.write_new(tmp_path / "manifest.json")
    result = _result(manifest, manifest.cases[0], status="success")
    ResultStore(tmp_path / "results.jsonl", manifest).append(result)
    snapshot = Path(result.artifact_dir) / "case_detail.json"

    outcome = _coordinator(
        tmp_path,
        manifest,
        None,
        lambda **_kwargs: pytest.fail("completed Case must not launch"),
    ).run(resume=True)

    assert outcome.exit_code == 0
    assert snapshot.is_file()


def test_resume_refuses_changed_manifest(tmp_path):
    original = _manifest(tmp_path)
    original.write_new(tmp_path / "manifest.json")
    changed = _manifest(tmp_path, rpent_commit="f" * 40)
    daemon = _PolicyDaemon([], tmp_path / "manifest.json")

    with pytest.raises(ValueError, match="different manifest"):
        _coordinator(
            tmp_path,
            changed,
            daemon,
            _Launcher(changed, [], ["success"] * 3),
        ).run(resume=True)
    assert daemon.start_calls == 0


def test_existing_same_digest_manifest_is_authoritative_without_resume(tmp_path):
    persisted = _manifest(tmp_path, task_count=1, evaluation_id="persisted-eval")
    persisted.write_new(tmp_path / "manifest.json")
    submitted = _manifest(tmp_path, task_count=1, evaluation_id="submitted-eval")
    events = []
    launcher = _Launcher(persisted, events, ["success"])
    daemon = _PolicyDaemon(events, tmp_path / "manifest.json")
    coordinator = _coordinator(tmp_path, submitted, daemon, launcher)

    outcome = coordinator.run()

    assert outcome.exit_code == 0
    assert coordinator.manifest.evaluation_id == "persisted-eval"
    result = ResultStore(tmp_path / "results.jsonl", persisted).load()[0]
    assert result.evaluation_id == "persisted-eval"


def test_coordinator_first_schema_v1_write_uses_normalized_disk_identity(tmp_path):
    submitted = replace(
        _manifest(tmp_path, task_count=1),
        schema_version=1,
        evaluation_id="not-persisted",
    )
    expected = submitted.write_new(tmp_path / "expected-legacy.json")
    events = []
    launcher = _Launcher(expected, events, ["success"])
    coordinator = _coordinator(
        tmp_path,
        submitted,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    )

    outcome = coordinator.run()

    assert outcome.exit_code == 0
    assert coordinator.manifest == EvaluationManifest.load(tmp_path / "manifest.json")
    assert coordinator.manifest.evaluation_id == submitted.manifest_id
    result = ResultStore(tmp_path / "results.jsonl", coordinator.manifest).load()[0]
    assert result.evaluation_id == submitted.manifest_id


def test_interrupt_stops_active_worker_and_preserves_prior_results(tmp_path):
    events = []
    manifest = _manifest(tmp_path)
    daemon = _PolicyDaemon(events, tmp_path / "manifest.json")
    launcher = _Launcher(manifest, events, ["success", "interrupt", "success"])

    outcome = _coordinator(tmp_path, manifest, daemon, launcher).run()

    assert outcome.exit_code != 0
    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert [result.case_id for result in results] == [
        manifest.cases[0].case_id,
        manifest.cases[1].case_id,
    ]
    assert results[-1].status == "cancelled"
    assert results[-1].attempt_id == launcher.processes[1].attempt_id
    assert launcher.processes[1].terminated is True
    assert len(launcher.processes) == 2
    assert daemon.stop_calls == 1

    resume_launcher = _Launcher(manifest, [], ["success", "success"])
    resumed = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        resume_launcher,
    ).run(resume=True)
    assert resumed.completed == 3
    assert resume_launcher.processes[0].case.ordinal == 1


def test_interrupt_retains_worker_written_cancelled_result(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    events = []
    launcher = _Launcher(manifest, events, ["interrupt_with_result"])

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    result = ResultStore(tmp_path / "results.jsonl", manifest).load()[0]
    assert outcome.interrupted is True
    assert result.status == "cancelled"
    assert result.attempt_id == launcher.processes[0].attempt_id
    assert launcher.processes[0].wait_timeouts == [0.25, 0.25]


@pytest.mark.parametrize("status", ["success", "terminal_failure"])
def test_interrupt_retains_worker_written_terminal_physical_result(
    tmp_path,
    status,
):
    manifest = _manifest(tmp_path, task_count=1)
    events = []
    launcher = _Launcher(manifest, events, [f"interrupt_with_{status}"])

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert len(results) == 1
    assert outcome.interrupted is True
    assert results[0].status == status
    assert results[0].simulator_success is (status == "success")
    assert results[0].terminated is True
    assert results[0].step_count == 1
    assert launcher.processes[0].wait_timeouts == [0.25, 0.25]


def test_interrupt_cleanup_timeout_terminates_without_replacing_valid_result(
    tmp_path,
):
    manifest = _manifest(tmp_path, task_count=1)
    events = []
    launcher = _Launcher(
        manifest,
        events,
        ["interrupt_with_success_cleanup_timeout"],
    )

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    process = launcher.processes[0]
    assert outcome.interrupted is True
    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].simulator_success is True
    assert process.wait_timeouts == [0.25, 0.25, 5.0, 5.0]
    assert process.terminated is True
    assert process.killed is True
    assert process.reaped is True


def test_interrupt_cleanup_reap_error_refuses_unproven_result(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    events = []
    launcher = _Launcher(
        manifest,
        events,
        ["interrupt_with_success_cleanup_reap_error"],
    )

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    process = launcher.processes[0]
    assert outcome.interrupted is True
    assert outcome.infrastructure_error is True
    assert results == ()
    assert process.wait_timeouts == [0.25, 0.25, 5.0, 5.0]
    assert process.killed is True


def test_interrupt_gives_worker_bounded_grace_to_write_terminal_result(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)
    events = []
    launcher = _Launcher(manifest, events, ["interrupt_then_success"])

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon(events, tmp_path / "manifest.json"),
        launcher,
    ).run()

    result = ResultStore(tmp_path / "results.jsonl", manifest).load()[0]
    process = launcher.processes[0]
    assert outcome.interrupted is True
    assert result.status == "success"
    assert process.wait_timeouts == [0.25, 0.25]
    assert process.terminated is False


@pytest.mark.parametrize(
    "artifact_dir",
    ["/tmp/foreign-attempt", "../../outside-attempt"],
)
def test_worker_result_rejects_foreign_attempt_artifact_path(
    tmp_path,
    artifact_dir,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-path")

    with pytest.raises(ValueError, match="artifact"):
        _validate_worker_case(
            manifest,
            case,
            "attempt-path",
            replace(result, artifact_dir=artifact_dir),
        )


def test_jobs_other_than_one_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="jobs=1"):
        EvaluationCoordinator(
            manifest=manifest,
            manifest_path=tmp_path / "manifest.json",
            results_path=tmp_path / "results.jsonl",
            summary_path=tmp_path / "summary.json",
            policy_daemon=None,
            jobs=2,
        )


def test_pre_cancelled_run_does_not_start_policy_daemon(tmp_path):
    from rpent.evaluation.control import CancellationToken

    manifest = _manifest(
        tmp_path,
        task_count=1,
        runtime={
            "main_interpreter": "/opt/rpent/bin/python",
            "evaluation_root": str(tmp_path),
            "shared_policy_required": True,
        },
    )
    events = []
    daemon = _PolicyDaemon(events, tmp_path / "manifest.json")
    daemon.endpoint = ""
    token = CancellationToken()
    token.cancel("queued")

    outcome = _coordinator(
        tmp_path,
        manifest,
        daemon,
        _Launcher(manifest, events, ["success"]),
        cancellation=token,
    ).run()

    assert outcome.interrupted is True
    assert daemon.start_calls == 0
    assert daemon.stop_calls == 0
    retained = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert len(retained) == 1
    assert retained[0].status == "cancelled"
    assert retained[0].worker_pid is None


def test_policy_daemon_stop_failure_marks_infrastructure_without_losing_result(
    tmp_path,
):
    class StopFails(_PolicyDaemon):
        def stop(self):
            super().stop()
            raise OSError("daemon stop failed")

    events = []
    manifest = _manifest(tmp_path, task_count=1)
    daemon = StopFails(events, tmp_path / "manifest.json")
    launcher = _Launcher(manifest, events, ["success"])

    outcome = _coordinator(tmp_path, manifest, daemon, launcher).run()

    assert outcome.infrastructure_error is True
    assert outcome.exit_code != 0
    assert ResultStore(tmp_path / "results.jsonl", manifest).load()[0].status == (
        "success"
    )


def test_policy_stop_failure_does_not_replace_projected_coordinator_error(tmp_path):
    class StopFails(_PolicyDaemon):
        endpoint = ""

        def stop(self):
            super().stop()
            raise OSError("secondary stop failure")

    manifest = _manifest(
        tmp_path,
        task_count=1,
        runtime={
            "main_interpreter": "/opt/rpent/bin/python",
            "evaluation_root": str(tmp_path),
            "shared_policy_required": True,
        },
    )
    events = []

    outcome = _coordinator(
        tmp_path,
        manifest,
        StopFails(events, tmp_path / "manifest.json"),
        _Launcher(manifest, events, ["success"]),
    ).run()

    result = ResultStore(tmp_path / "results.jsonl", manifest).load()[0]
    assert outcome.infrastructure_error is True
    assert outcome.exit_code == 1
    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert "shared policy validation" in result.error_message


def _grouped_manifest(tmp_path: Path, *, episodes: int = 3) -> EvaluationManifest:
    manifest = _manifest(tmp_path, task_count=2, episodes=episodes)
    return replace(
        manifest,
        schedule={
            **dict(manifest.schedule),
            "worker_grouping": "task",
        },
        runtime={
            **dict(manifest.runtime),
            "grouped_worker_module": "tests.fake_task_worker",
        },
    )


class _GroupedProcess:
    def __init__(
        self,
        *,
        manifest,
        task_name,
        batch_attempt_id,
        results_path,
        ledger_path,
        behavior,
    ):
        self.pid = 9001
        self.manifest = manifest
        self.task_name = task_name
        self.batch_attempt_id = batch_attempt_id
        self.results_path = results_path
        self.ledger_path = ledger_path
        self.behavior = behavior
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        del timeout
        cases = tuple(
            case for case in self.manifest.cases if case.task_name == self.task_name
        )
        expected = tuple(case.case_id for case in cases)
        started_at = "2026-08-09T00:00:00+00:00"
        append_task_batch_attempt(
            self.ledger_path,
            self.manifest,
            TaskBatchAttempt(
                manifest_id=self.manifest.manifest_id,
                task_name=self.task_name,
                batch_attempt_id=self.batch_attempt_id,
                status="running",
                expected_case_ids=expected,
                completed_case_ids=(),
                started_at=started_at,
                finished_at=None,
            ),
        )
        count = (
            len(cases)
            if self.behavior.startswith("complete")
            else 17
            if self.behavior == "crash17"
            else 2
        )
        local = ResultStore(self.results_path, self.manifest)
        completed = []
        for index, case in enumerate(cases[:count]):
            attempt_id = f"{self.batch_attempt_id}-{index:03d}-{case.case_id[:12]}"
            local.append(
                replace(
                    _result(
                        self.manifest,
                        case,
                        status="success",
                        pid=self.pid,
                        attempt_id=attempt_id,
                    ),
                    task_batch_attempt_id=self.batch_attempt_id,
                )
            )
            completed.append(case.case_id)
        append_task_batch_attempt(
            self.ledger_path,
            self.manifest,
            TaskBatchAttempt(
                manifest_id=self.manifest.manifest_id,
                task_name=self.task_name,
                batch_attempt_id=self.batch_attempt_id,
                status=(
                    "complete" if self.behavior.startswith("complete") else "failed"
                ),
                expected_case_ids=expected,
                completed_case_ids=tuple(completed),
                started_at=started_at,
                finished_at="2026-08-09T00:01:00+00:00",
            ),
        )
        return (
            0
            if self.behavior == "complete"
            else 9
            if self.behavior == "complete-nonzero"
            else 3
        )

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _GroupedLauncher:
    def __init__(self, manifest, behaviors, central_store):
        self.manifest = manifest
        self.behaviors = iter(behaviors)
        self.central_store = central_store
        self.calls = []
        self.environments = []

    def __call__(self, *, command, case, result_path, env):
        self.environments.append(dict(env))
        assert env["RPENT_SHARED_POLICY_ENDPOINT"] == "http://127.0.0.1:19000"
        task_name = command[command.index("--task") + 1]
        batch_attempt_id = command[command.index("--batch-attempt-id") + 1]
        for selected_case in (
            item for item in self.manifest.cases if item.task_name == task_name
        ):
            attempt_id = (
                f"{batch_attempt_id}-{selected_case.episode_index:03d}-"
                f"{selected_case.case_id[:12]}"
            )
            case_root = (
                Path(self.manifest.runtime["evaluation_root"])
                / "cases"
                / selected_case.case_id
            )
            registration = json.loads(
                (case_root / "attempts" / attempt_id / "case.json").read_text()
            )
            locator = json.loads((case_root / "current_attempt.json").read_text())
            assert locator["attempt_id"] == attempt_id
            assert locator["registered_at"] == registration["started_at"]
        ledger_path = Path(command[command.index("--batch-ledger") + 1])
        assert command[:3] == [
            "/opt/rpent/bin/python",
            "-m",
            "tests.fake_task_worker",
        ]
        assert case.task_name == task_name
        assert result_path == Path(command[command.index("--results") + 1])
        assert result_path.parent == (
            Path(self.manifest.runtime["evaluation_root"])
            / "task_batches"
            / task_name
            / batch_attempt_id
        )
        self.calls.append((task_name, batch_attempt_id))
        return _GroupedProcess(
            manifest=self.manifest,
            task_name=task_name,
            batch_attempt_id=batch_attempt_id,
            results_path=result_path,
            ledger_path=ledger_path,
            behavior=next(self.behaviors),
        )


def test_task_grouping_dispatches_manifest_task_order_and_imports_results(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    launcher = _GroupedLauncher(manifest, ["complete", "complete"], central)

    coordinator = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launcher,
    )
    outcome = coordinator.run()

    assert outcome.exit_code == 0
    assert [task for task, _attempt in launcher.calls] == ["task_0", "task_1"]
    assert [result.case_id for result in central.load()] == [
        case.case_id for case in manifest.cases
    ]
    assert all(result.task_batch_attempt_id for result in central.load())
    for (task_name, batch_attempt_id), environment in zip(
        launcher.calls, launcher.environments, strict=True
    ):
        assert environment["RPENT_WORKER_LOG_PATH"] == str(
            (
                tmp_path
                / "task_batches"
                / task_name
                / batch_attempt_id
                / "console.log"
            ).resolve()
        )


def test_grouped_worker_reuses_one_reaped_proof_for_all_case_acceptances(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    launcher = _GroupedLauncher(manifest, ["complete", "complete"], central)
    proofs = []

    def factory(_manifest, store):
        class Acceptance:
            def accept(self, candidate):
                assert isinstance(candidate.completion, ReapedWorker)
                assert candidate.completion.scope_id == (
                    candidate.result.task_batch_attempt_id
                )
                proofs.append(candidate.completion)
                store.append_or_confirm(candidate.result)
                return candidate.result

        return Acceptance()

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launcher,
        result_acceptance_factory=factory,
    ).run()

    assert outcome.exit_code == 0
    for task_name in ("task_0", "task_1"):
        selected = [
            proof
            for proof, result in zip(proofs, central.load(), strict=True)
            if result.task_name == task_name
        ]
        assert len({id(proof) for proof in selected}) == 1


def test_official_task_grouping_writes_separate_native_and_system_projections(
    tmp_path,
):
    base = _grouped_manifest(tmp_path)
    manifest = replace(
        base,
        schedule={
            **dict(base.schedule),
            "evaluation_profile": "test-official",
            "task_sets": ("seen", "unseen"),
            "task_set_membership": {
                "seen": ("task_0",),
                "unseen": ("task_1",),
            },
        },
    )
    launcher = _GroupedLauncher(
        manifest,
        ["complete", "complete"],
        ResultStore(tmp_path / "results.jsonl", manifest),
    )

    coordinator = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launcher,
    )
    outcome = coordinator.run()

    official = json.loads((tmp_path / "official_metrics.json").read_text())
    diagnostics = json.loads((tmp_path / "rpent_diagnostics.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert outcome.exit_code == 0
    assert official["available"] is True
    assert official["is_complete"] is True
    assert official["overall_success_rate"] == 1.0
    assert diagnostics["selected_cohort"]["contract_successes"] == 6
    assert summary["success_rate"] == 1.0

    deep_json = "[" * 2_000 + "0" + "]" * 2_000
    retained = ResultStore(tmp_path / "results.jsonl", manifest).load()
    for result in retained:
        artifact = Path(result.artifact_dir)
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "events.jsonl").write_text(deep_json + "\n", encoding="utf-8")
        (artifact / "qualification.json").write_text(deep_json + "\n", encoding="utf-8")
    coordinator._write_projections(aggregate_results(manifest, retained), retained)

    after_official = json.loads((tmp_path / "official_metrics.json").read_text())
    after_diagnostics = json.loads((tmp_path / "rpent_diagnostics.json").read_text())
    assert after_official["overall_success_rate"] == 1.0
    assert after_official["is_complete"] is True
    assert after_diagnostics["schema_version"] == 1
    assert after_diagnostics["selected_cohort"]["rollouts"] == 6
    assert (
        "event evidence unavailable"
        in after_diagnostics["structured_evidence"]["issues"]
    )


def test_development_projection_marks_official_metrics_unavailable(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)

    _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        _Launcher(manifest, [], ["success"]),
    ).run()

    official = json.loads((tmp_path / "official_metrics.json").read_text())
    assert official["available"] is False
    assert "overall_success_rate" not in official


def test_official_authority_failure_does_not_overwrite_prior_projections(tmp_path):
    base = _grouped_manifest(tmp_path)
    manifest = replace(
        base,
        schedule={**dict(base.schedule), "evaluation_profile": "test-official"},
    )
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    coordinator = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        _GroupedLauncher(manifest, ["complete", "complete"], central),
    )
    coordinator.run()
    official_path = tmp_path / "official_metrics.json"
    diagnostics_path = tmp_path / "rpent_diagnostics.json"
    before = (official_path.read_bytes(), diagnostics_path.read_bytes())
    duplicate = _write_grouped_attempt(
        manifest, "task_0", "second-complete", "complete"
    )
    for result in duplicate:
        central.append(result)

    with pytest.raises(ValueError, match="multiple complete"):
        coordinator._write_projections(
            aggregate_results(manifest, central.load()), central.load()
        )

    assert (official_path.read_bytes(), diagnostics_path.read_bytes()) == before


def test_grouped_crash_imports_only_real_prefix_without_synthetic_cases(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    launcher = _GroupedLauncher(manifest, ["crash", "complete"], central)

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launcher,
    ).run()

    retained = central.load()
    assert outcome.infrastructure_error is True
    assert len(retained) == 5
    task_zero = [result for result in retained if result.task_name == "task_0"]
    assert [result.episode_index for result in task_zero] == [0, 1]
    assert all(result.error_type is None for result in retained)


def test_grouped_resume_reruns_failed_task_from_episode_zero(tmp_path):
    manifest = _grouped_manifest(tmp_path, episodes=50)
    first_store = ResultStore(tmp_path / "results.jsonl", manifest)
    first_launcher = _GroupedLauncher(manifest, ["crash17", "complete"], first_store)
    _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        first_launcher,
    ).run()

    resumed_launcher = _GroupedLauncher(manifest, ["complete"], first_store)
    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        resumed_launcher,
    ).run(resume=True)

    assert outcome.completed == 100
    assert [task for task, _attempt in resumed_launcher.calls] == ["task_0"]
    task_zero_attempt = resumed_launcher.calls[0][1]
    task_zero_results = [
        result
        for result in first_store.load()
        if result.task_batch_attempt_id == task_zero_attempt
    ]
    assert [result.episode_index for result in task_zero_results] == list(range(50))


def test_unknown_worker_grouping_is_rejected_without_side_effects(tmp_path):
    manifest = _manifest(
        tmp_path,
        schedule={
            "episodes_per_task": 1,
            "base_seed": 0,
            "worker_grouping": "vector",
        },
    )

    with pytest.raises(ValueError, match="worker_grouping"):
        _coordinator(tmp_path, manifest, None, lambda **_kwargs: None).run()

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "results.jsonl").exists()


def _write_grouped_attempt(manifest, task_name, batch_attempt_id, behavior):
    attempt_root = (
        Path(manifest.runtime["evaluation_root"])
        / "task_batches"
        / task_name
        / batch_attempt_id
    )
    process = _GroupedProcess(
        manifest=manifest,
        task_name=task_name,
        batch_attempt_id=batch_attempt_id,
        results_path=attempt_root / "results.jsonl",
        ledger_path=attempt_root / "batch.jsonl",
        behavior=behavior,
    )
    exit_code = process.wait()
    append_task_batch_process_outcome(
        Path(manifest.runtime["evaluation_root"]) / "coordinator_batch_outcomes.jsonl",
        manifest,
        TaskBatchProcessOutcome(
            manifest_id=manifest.manifest_id,
            task_name=task_name,
            batch_attempt_id=batch_attempt_id,
            worker_exit_code=exit_code,
        ),
    )
    return ResultStore(attempt_root / "results.jsonl", manifest).load()


def test_grouped_resume_idempotently_finishes_interrupted_central_import(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    local = _write_grouped_attempt(manifest, "task_0", "old-failed", "crash")
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    central.append(local[0])
    launcher = _GroupedLauncher(manifest, ["complete", "complete"], central)

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launcher,
    ).run(resume=True)

    retained = central.load()
    assert outcome.exit_code == 0
    assert sum(result.attempt_id == local[0].attempt_id for result in retained) == 1
    assert local[1] in retained
    assert [task for task, _attempt in launcher.calls] == ["task_0", "task_1"]


def test_grouped_resume_rejects_same_attempt_payload_drift(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    local = _write_grouped_attempt(manifest, "task_0", "old-failed", "crash")
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    central.append(replace(local[0], step_count=local[0].step_count + 1))

    with pytest.raises(ValueError, match="attempt identity drift"):
        _coordinator(
            tmp_path,
            manifest,
            _PolicyDaemon([], tmp_path / "manifest.json"),
            lambda **_kwargs: pytest.fail("worker must not launch"),
        ).run(resume=True)


def test_grouped_resume_rejects_multiple_complete_attempts(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    _write_grouped_attempt(manifest, "task_0", "valid-complete", "complete")
    _write_grouped_attempt(manifest, "task_1", "complete-one", "complete")
    _write_grouped_attempt(manifest, "task_1", "complete-two", "complete")

    with pytest.raises(ValueError, match="multiple complete"):
        _coordinator(
            tmp_path,
            manifest,
            _PolicyDaemon([], tmp_path / "manifest.json"),
            lambda **_kwargs: pytest.fail("worker must not launch"),
        ).run(resume=True)

    assert not (tmp_path / "results.jsonl").exists()


def test_grouped_launcher_failure_does_not_synthesize_central_results(tmp_path):
    manifest = _grouped_manifest(tmp_path)

    def fail_launch(**_kwargs):
        raise OSError("worker launcher unavailable")

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        fail_launch,
    ).run()

    assert outcome.infrastructure_error is True
    assert outcome.exit_code != 0
    assert ResultStore(tmp_path / "results.jsonl", manifest).load() == ()


def test_local_launcher_failure_accepts_error_for_registered_attempt(tmp_path):
    manifest = _manifest(tmp_path, task_count=1)

    def fail_launch(**_kwargs):
        raise OSError("worker launcher unavailable")

    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        fail_launch,
    ).run()

    locator = json.loads(
        (
            tmp_path
            / "cases"
            / manifest.cases[0].case_id
            / "current_attempt.json"
        ).read_text()
    )
    retained = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert outcome.infrastructure_error is True
    assert len(retained) == 1
    assert retained[0].status == "error"
    assert retained[0].attempt_id == locator["attempt_id"]


@pytest.mark.parametrize("child_claimed_exit", [0, 9])
def test_grouped_nonzero_complete_is_durably_rejected_on_resume(
    tmp_path, child_claimed_exit
):
    manifest = _grouped_manifest(tmp_path)
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    first = _GroupedLauncher(manifest, ["complete-nonzero", "complete"], central)
    first_outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        first,
    ).run()
    rejected_batch = first.calls[0][1]
    forged = TaskBatchProcessOutcome(
        manifest.manifest_id, "task_0", rejected_batch, child_claimed_exit
    )
    forged_path = (
        tmp_path / "task_batches" / "task_0" / rejected_batch / "process_outcome.json"
    )
    forged_path.write_text(canonical_json(forged.to_payload()) + "\n", encoding="utf-8")

    resumed = _GroupedLauncher(manifest, ["complete"], central)
    resumed_outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        resumed,
    ).run(resume=True)

    assert first_outcome.infrastructure_error is True
    assert resumed_outcome.exit_code == 0
    assert [task for task, _attempt in resumed.calls] == ["task_0"]
    assert resumed.calls[0][1] != first.calls[0][1]


def test_grouped_wait_exception_reaps_before_reading_evidence(tmp_path, monkeypatch):
    manifest = _grouped_manifest(tmp_path)
    central = ResultStore(tmp_path / "results.jsonl", manifest)
    process = None

    class WaitFails(_GroupedProcess):
        waited = False
        reaped = False

        def wait(self, timeout=None):
            if not self.waited:
                self.waited = True
                super().wait(timeout)
                raise OSError("wait transport failed")
            assert self.terminated
            self.reaped = True
            return 1

    def launch(*, command, case, result_path, env):
        nonlocal process
        del env
        task_name = command[command.index("--task") + 1]
        batch_attempt_id = command[command.index("--batch-attempt-id") + 1]
        process = WaitFails(
            manifest=manifest,
            task_name=task_name,
            batch_attempt_id=batch_attempt_id,
            results_path=result_path,
            ledger_path=Path(command[command.index("--batch-ledger") + 1]),
            behavior="crash",
        )
        assert case.task_name == task_name
        return process

    original = EvaluationCoordinator._load_task_evidences

    def assert_reaped(self, task_name):
        assert process is not None and process.terminated and process.reaped
        return original(self, task_name)

    monkeypatch.setattr(EvaluationCoordinator, "_load_task_evidences", assert_reaped)
    outcome = _coordinator(
        tmp_path,
        manifest,
        _PolicyDaemon([], tmp_path / "manifest.json"),
        launch,
    ).run()

    assert outcome.infrastructure_error is True
    assert len(central.load()) == 4


def test_grouped_fresh_run_rejects_old_local_evidence_before_import(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    _write_grouped_attempt(manifest, "task_0", "old-failed", "crash")

    with pytest.raises(ValueError, match="existing task batch"):
        _coordinator(
            tmp_path,
            manifest,
            None,
            lambda **_kwargs: pytest.fail("worker must not launch"),
        ).run()

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "results.jsonl").exists()


def test_grouped_fresh_run_rejects_central_outcome_evidence(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    append_task_batch_process_outcome(
        tmp_path / "coordinator_batch_outcomes.jsonl",
        manifest,
        TaskBatchProcessOutcome(manifest.manifest_id, "task_0", "old-batch", 3),
    )

    with pytest.raises(ValueError, match="existing task batch"):
        _coordinator(tmp_path, manifest, None, lambda **_kwargs: None).run()

    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("grouping", ["", False, 0, [], "vector"])
def test_invalid_worker_grouping_has_no_side_effects(tmp_path, grouping):
    manifest = _manifest(
        tmp_path,
        schedule={"episodes_per_task": 1, "base_seed": 0, "worker_grouping": grouping},
    )

    with pytest.raises(ValueError, match="worker_grouping"):
        _coordinator(tmp_path, manifest, None, lambda **_kwargs: None).run()

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("module", ["", "robocasa-worker", ".worker", 7])
def test_invalid_grouped_worker_module_fails_before_policy_or_projection(
    tmp_path, module
):
    manifest = replace(
        _grouped_manifest(tmp_path),
        runtime={
            **dict(_grouped_manifest(tmp_path).runtime),
            "grouped_worker_module": module,
        },
    )
    events = []

    with pytest.raises(ValueError, match="grouped_worker_module"):
        _coordinator(
            tmp_path,
            manifest,
            _PolicyDaemon(events, tmp_path / "manifest.json"),
            lambda **_kwargs: pytest.fail("worker must not launch"),
        ).run()

    assert events == []
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("drift", ["attempt", "artifact"])
def test_grouped_resume_rejects_noncanonical_local_result_identity(tmp_path, drift):
    manifest = _grouped_manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    local = _write_grouped_attempt(manifest, "task_0", "old-failed", "crash")
    path = tmp_path / "task_batches/task_0/old-failed/results.jsonl"
    changed = (
        replace(local[0], attempt_id="safe-but-wrong")
        if drift == "attempt"
        else replace(local[0], artifact_dir=str(tmp_path / "outside"))
    )
    payloads = [changed, *local[1:]]
    path.write_text(
        "".join(canonical_json(result.to_payload()) + "\n" for result in payloads),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="attempt|artifact"):
        _coordinator(tmp_path, manifest, None, lambda **_kwargs: None).run(resume=True)

    assert not (tmp_path / "results.jsonl").exists()


def test_grouped_resume_rejects_symlink_attempt_directory_escape(tmp_path):
    manifest = _grouped_manifest(tmp_path)
    manifest.write_new(tmp_path / "manifest.json")
    _write_grouped_attempt(manifest, "task_0", "escaped", "crash")
    attempt = tmp_path / "task_batches/task_0/escaped"
    outside = tmp_path / "outside-attempt"
    attempt.rename(outside)
    attempt.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="attempt directory"):
        _coordinator(tmp_path, manifest, None, lambda **_kwargs: None).run(resume=True)
