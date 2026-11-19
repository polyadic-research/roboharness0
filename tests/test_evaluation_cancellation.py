from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.control import CancellationToken, EvaluationCancelled
from rpent.evaluation.coordinator import EvaluationCoordinator
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult, ResultStore
from rpent.planning.manifest import canonical_json
from tests.synthetic_tasks import SyntheticTask


def _manifest(root: Path, task_count: int = 2) -> EvaluationManifest:
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
    return EvaluationManifest(
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "synthetic"},
        simulator={"revision": "b" * 40},
        policy={"lerobot_revision": "c" * 40},
        task_selector="all",
        cases=expand_episode_cases(tasks, protocol, base_seed=0),
        schedule={"episodes_per_task": 1, "base_seed": 0},
        contracts={"action": {"shape": [7]}},
        runtime={
            "main_interpreter": "/opt/rpent/bin/python",
            "evaluation_root": str(root),
            "policy_endpoint": "http://127.0.0.1:19000",
        },
        created_at="2026-08-07T00:00:00+00:00",
    )


def _result(manifest, case, attempt_id, *, status="success") -> EpisodeResult:
    success = status == "success"
    root = Path(manifest.runtime["evaluation_root"])
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        status=status,
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=123,
        worker_exit_code=0,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at="2026-08-07T00:00:01+00:00",
        artifact_dir=str(root / "cases" / case.case_id / "attempts" / attempt_id),
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


class _Process:
    def __init__(self, manifest, case, attempt_id, result_path, *, stubborn=False):
        self.manifest = manifest
        self.case = case
        self.attempt_id = attempt_id
        self.result_path = result_path
        self.stubborn = stubborn
        self.started = threading.Event()
        self.release = threading.Event()
        self.terminated = False
        self.killed = False
        self.wait_timeouts = []
        self.pid = 123

    def wait(self, timeout=None):
        self.started.set()
        self.wait_timeouts.append(timeout)
        if self.release.wait(timeout=0):
            self._write_result()
            return 0
        if self.killed or (self.terminated and not self.stubborn):
            return -9 if self.killed else -15
        raise subprocess.TimeoutExpired("worker", timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def _write_result(self):
        result = _result(self.manifest, self.case, self.attempt_id)
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(
            canonical_json(result.to_payload()) + "\n", encoding="utf-8"
        )


class _Launcher:
    def __init__(self, manifest, *, stubborn=False):
        self.manifest = manifest
        self.stubborn = stubborn
        self.processes = []

    def __call__(self, *, command, case, result_path, env):
        del command, env
        attempt_id = result_path.stem.rsplit("-", 1)[-1]
        # UUID attempt IDs contain dashes; the command is the authoritative source.
        attempt_id = result_path.name.removesuffix(".json").split(
            f"-{case.case_id}-", 1
        )[1]
        process = _Process(
            self.manifest,
            case,
            attempt_id,
            result_path,
            stubborn=self.stubborn,
        )
        self.processes.append(process)
        return process


def _coordinator(root, manifest, launcher, token=None):
    return EvaluationCoordinator(
        manifest=manifest,
        manifest_path=root / "evaluation_manifest.json",
        results_path=root / "results.jsonl",
        summary_path=root / "summary.json",
        policy_daemon=None,
        worker_launcher=launcher,
        worker_timeout_s=30,
        cancellation=token,
        progress=lambda *_: None,
    )


@pytest.mark.parametrize("timeout", [True, float("nan"), float("inf"), -float("inf")])
def test_worker_timeout_requires_a_finite_non_boolean_number(tmp_path, timeout):
    manifest = _manifest(tmp_path, task_count=1)
    with pytest.raises(ValueError, match="finite positive"):
        EvaluationCoordinator(
            manifest=manifest,
            manifest_path=tmp_path / "evaluation_manifest.json",
            results_path=tmp_path / "results.jsonl",
            summary_path=tmp_path / "summary.json",
            policy_daemon=None,
            worker_timeout_s=timeout,
        )


def test_cancellation_token_retains_first_reason_and_raises() -> None:
    token = CancellationToken()

    assert token.cancel("user_requested") is True
    assert token.cancel("later_reason") is False
    assert token.is_cancelled() is True
    assert token.reason == "user_requested"
    with pytest.raises(EvaluationCancelled, match="user_requested"):
        token.raise_if_cancelled()


def test_cancel_before_first_launch_accepts_cancelled_results_without_workers(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path)
    launcher = _Launcher(manifest)
    token = CancellationToken()
    token.cancel("user_requested")

    outcome = _coordinator(tmp_path, manifest, launcher, token).run()

    assert outcome.interrupted is True
    assert launcher.processes == []
    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert len(results) == len(manifest.cases)
    assert {result.status for result in results} == {"cancelled"}
    assert all(result.worker_pid is None for result in results)
    assert all(result.worker_exit_code is None for result in results)


def test_cancel_active_worker_stops_future_launches_and_records_attempt(tmp_path):
    manifest = _manifest(tmp_path)
    launcher = _Launcher(manifest)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, launcher, token)
    outcome_holder = []
    thread = threading.Thread(target=lambda: outcome_holder.append(coordinator.run()))
    thread.start()
    while not launcher.processes:
        thread.join(0.01)
    assert launcher.processes[0].started.wait(1)

    coordinator.cancel("user_requested")
    thread.join(2)

    assert not thread.is_alive()
    assert outcome_holder[0].interrupted is True
    assert len(launcher.processes) == 1
    assert launcher.processes[0].terminated is True
    assert max(launcher.processes[0].wait_timeouts[:1]) <= 0.25
    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert len(results) == 1
    assert results[0].status == "cancelled"
    assert results[0].attempt_id == launcher.processes[0].attempt_id


def test_cancel_does_not_replace_worker_terminal_result(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, launcher, token)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(coordinator.run()))
    thread.start()
    while not launcher.processes:
        thread.join(0.01)
    process = launcher.processes[0]
    assert process.started.wait(1)
    process.release.set()
    token.cancel("user_requested")
    thread.join(2)

    retained = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert outcomes[0].interrupted is True
    assert len(retained) == 1
    assert retained[0].status == "success"
    assert outcomes[0].summary.scheduled == 1
    assert outcomes[0].summary.completed == 1
    assert outcomes[0].summary.successful == 1
    assert outcomes[0].summary.success_rate == 1.0
    assert outcomes[0].summary.status_counts["success"] == 1
    assert outcomes[0].summary.status_counts["cancelled"] == 0


def test_cancel_keeps_completed_case_and_only_cancels_active_attempt(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    launcher = _Launcher(manifest)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, launcher, token)
    thread = threading.Thread(target=coordinator.run)
    thread.start()
    while not launcher.processes:
        thread.join(0.01)
    launcher.processes[0].release.set()
    while len(launcher.processes) < 2:
        thread.join(0.01)

    token.cancel("user_requested")
    thread.join(2)

    retained = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert [result.status for result in retained] == ["success", "cancelled"]
    assert retained[0].attempt_id == launcher.processes[0].attempt_id
    assert launcher.processes[1].terminated is True


def test_cancel_stubborn_worker_terminates_then_kills_and_reaps(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest, stubborn=True)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, launcher, token)
    thread = threading.Thread(target=coordinator.run)
    thread.start()
    while not launcher.processes:
        thread.join(0.01)
    assert launcher.processes[0].started.wait(1)

    token.cancel("user_requested")
    thread.join(2)

    process = launcher.processes[0]
    assert not thread.is_alive()
    assert process.terminated is True
    assert process.killed is True
    assert 5.0 in process.wait_timeouts


def test_cancel_is_retained_when_process_terminate_raises_oserror(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    launcher = _Launcher(manifest)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, launcher, token)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(coordinator.run()))
    thread.start()
    while not launcher.processes:
        thread.join(0.01)
    process = launcher.processes[0]
    assert process.started.wait(1)

    def unavailable():
        raise OSError("process already gone")

    process._write_result()
    process.terminate = unavailable
    token.cancel("user_requested")
    thread.join(2)

    assert not thread.is_alive()
    assert outcomes[0].interrupted is True
    assert (
        ResultStore(tmp_path / "results.jsonl", manifest).load()[0].status == "success"
    )


def test_resume_retries_cancelled_and_unstarted_with_new_attempts(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=3)
    cancelled_launcher = _Launcher(manifest)
    token = CancellationToken()
    coordinator = _coordinator(tmp_path, manifest, cancelled_launcher, token)
    thread = threading.Thread(target=coordinator.run)
    thread.start()
    while not cancelled_launcher.processes:
        thread.join(0.01)
    cancelled_launcher.processes[0].release.set()
    while len(cancelled_launcher.processes) < 2:
        thread.join(0.01)
    token.cancel("user_requested")
    thread.join(2)
    completed_attempt = cancelled_launcher.processes[0].attempt_id
    cancelled_attempt = cancelled_launcher.processes[1].attempt_id

    resumed_launcher = _Launcher(manifest)
    resumed = _coordinator(tmp_path, manifest, resumed_launcher)
    resume_thread = threading.Thread(target=lambda: resumed.run(resume=True))
    resume_thread.start()
    while len(resumed_launcher.processes) < 1:
        resume_thread.join(0.01)
    resumed_launcher.processes[0].release.set()
    while len(resumed_launcher.processes) < 2:
        resume_thread.join(0.01)
    resumed_launcher.processes[1].release.set()
    resume_thread.join(2)

    results = ResultStore(tmp_path / "results.jsonl", manifest).load()
    assert [result.status for result in results] == [
        "success",
        "cancelled",
        "success",
        "success",
    ]
    assert results[0].attempt_id == completed_attempt
    assert results[2].attempt_id != cancelled_attempt
    assert len({result.artifact_dir for result in results}) == 4
