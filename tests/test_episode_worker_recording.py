from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rpent.episode.contracts import StepResult
from rpent.episode.runtime import EpisodeBootstrap, open_episode
from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation_progress import (
    PREPARATION_PHASES,
    PreparationError,
    PreparationProgress,
    load_preparation,
)
from rpent.evaluation.results import EpisodeResult, ResultStore, aggregate_results
from rpent.evaluation.worker import (
    HarnessRunReport,
    PendingCaseRecorder,
    run_episode_case,
)
from rpent.evaluation.worker import main as worker_main
from rpent.recording.contracts import CameraDescriptor
from rpent.recording.display import png_bytes
from tests.synthetic_tasks import SyntheticTask


def test_xiaomi_recorder_persists_loaded_policy_health(tmp_path):
    from types import SimpleNamespace

    from robots.robocasa.display import RoboCasaDisplayAdapter

    manifest = replace(_manifest(tmp_path),
                       policy={"policy_family": "xiaomi-mibot", "protocol_version": 1},
                       recording={"schema_version": 1, "mode": "none", "camera_ids": []})
    health = {"artifact_identity": "a" * 64, "remote_code_revision": "b" * 40,
              "server_version": "rpent-xiaomi-mibot/1"}
    bootstrap = SimpleNamespace(display_adapter=RoboCasaDisplayAdapter(),
                                runtime_metadata={"policy_family": "xiaomi-mibot", "policy_health": health})
    pending = PendingCaseRecorder(manifest, manifest.cases[0], "attempt-1", tmp_path)
    episode = SimpleNamespace(episode_id="episode-1", observation=None,
                              set_transition_observer=lambda observer: None)
    recorder = pending.bind(bootstrap, episode)
    try:
        policy = json.loads((tmp_path / "policy.json").read_text())
        assert policy["health"] == health
        assert policy["artifact_identity"] == health["artifact_identity"]
        rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert any(row.get("event_type") == "policy_identity_verified" for row in rows)
    finally:
        recorder.close()


def _manifest(tmp_path: Path) -> EvaluationManifest:
    task = SyntheticTask(
        suite="libero_10",
        index=0,
        name="task_0",
        language="instruction",
        bddl_identifier="task_0.bddl",
        upstream_horizon=20,
    )
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="preset",
        horizon_ceiling=20,
        checkpoint_repo_id="repo/checkpoint",
        checkpoint_revision="7" * 40,
        binding_name="binding-v1",
    )
    return EvaluationManifest(
        evaluation_id="evaluation-1",
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "synthetic"},
        simulator={},
        policy={},
        task_selector="0",
        cases=expand_episode_cases((task,), protocol, base_seed=0),
        schedule={"episodes_per_task": 1},
        contracts={},
        runtime={"evaluation_root": str(tmp_path)},
        recording={"schema_version": 1, "mode": "all", "camera_ids": []},
        qualification={"require_complete_artifacts": True},
    )


def _report(**overrides) -> HarnessRunReport:
    values = {
        "exit_code": 0,
        "terminal_classification": "SUCCESS",
        "harness_state": "complete",
        "simulator_success": True,
        "contract_success": True,
        "terminated": True,
        "truncated": False,
        "step_count": 1,
        "action_count": 1,
        "error_type": None,
        "error_message": None,
    }
    values.update(overrides)
    return HarnessRunReport(**values)


class _Recorder:
    def __init__(self, events):
        self.events = events

    def finalize(self, result):
        assert isinstance(result, EpisodeResult)
        self.events.append(("finalize", result.attempt_id))
        return {"health": {"status": "complete", "issues": []}}


class _FinalizeAndCloseFailRecorder:
    def finalize(self, result):
        assert isinstance(result, EpisodeResult)
        raise RuntimeError("/Users/alice/private/model-key")

    def close(self):
        raise OSError(36, "File name too long", "/Users/alice/private/model-key")


def test_worker_writes_case_projection_before_runner_and_finalizes_after_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    events = []

    def runner(_case, _manifest, attempt_root):
        projection = json.loads((attempt_root / "case.json").read_text())
        events.append(("runner", projection["attempt_id"], projection["started_at"]))
        return replace(_report(), recorder=_Recorder(events))

    result = run_episode_case(
        case,
        manifest,
        harness_runner=runner,
        attempt_id_factory=lambda: "attempt-1",
        clock=iter(["2026-08-07T10:00:00+10:00", "2026-08-07T10:00:01+10:00"]).__next__,
    )

    assert result.evaluation_id == manifest.evaluation_id
    assert result.started_at == "2026-08-07T00:00:00+00:00"
    assert result.finished_at == "2026-08-07T00:00:01+00:00"
    assert result.attempt_id == "attempt-1"
    assert Path(result.artifact_dir) == (
        tmp_path / "cases" / case.case_id / "attempts" / "attempt-1"
    )
    assert events == [
        ("runner", "attempt-1", result.started_at),
        ("finalize", "attempt-1"),
    ]
    qualification = json.loads(
        (Path(result.artifact_dir) / "qualification.json").read_text()
    )
    assert qualification["episode_result"]["status"] == "success"
    assert qualification["accepted"] is True


def test_worker_rejects_naive_started_at_clock(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        run_episode_case(
            manifest.cases[0],
            manifest,
            harness_runner=lambda *_args: _report(),
            attempt_id_factory=lambda: "attempt-naive",
            clock=iter(["2026-08-07T00:00:00"]).__next__,
        )


def test_worker_finalizes_error_result_without_event_hub(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    events = []

    def runner(*_args):
        report = _report(
            exit_code=1,
            terminal_classification="FAILED",
            simulator_success=False,
            contract_success=False,
            error_type="RuntimeError",
            error_message="failed",
        )
        return replace(report, recorder=_Recorder(events))

    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=runner,
        attempt_id_factory=lambda: "attempt-error",
    )

    assert result.status == "error"
    assert events == [("finalize", "attempt-error")]


def test_qualification_write_failure_does_not_discard_physical_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    original_write_text = Path.write_text

    def fail_qualification(path, *args, **kwargs):
        if path.name == "qualification.json":
            raise OSError(36, "File name too long", "/Users/alice/private/model-key")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_qualification)
    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=lambda *_args: replace(_report(), recorder=_Recorder([])),
        attempt_id="attempt-qualification-write-error",
    )

    assert result.status == "success"
    assert result.simulator_success is True
    assert result.contract_success is True
    root = Path(result.artifact_dir)
    assert not (root / "qualification.json").exists()
    fallback = (root / "qualification_error.txt").read_text()
    assert "qualification write failed" in fallback
    assert "/Users/alice/private/model-key" not in fallback
    assert "File name too long" not in fallback


@pytest.mark.parametrize(
    ("report", "expected_status", "expected_simulator_success"),
    [
        (_report(), "success", True),
        (
            _report(
                exit_code=1,
                terminal_classification="FAILED",
                harness_state="failed",
                simulator_success=False,
                contract_success=False,
                error_type="RuntimeError",
                error_message="physical failure",
            ),
            "error",
            False,
        ),
    ],
)
def test_finalize_and_close_failures_do_not_replace_physical_result(
    tmp_path,
    report,
    expected_status,
    expected_simulator_success,
):
    manifest = _manifest(tmp_path)
    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=lambda *_args: replace(
            report, recorder=_FinalizeAndCloseFailRecorder()
        ),
        attempt_id=f"attempt-{expected_status}-recorder-errors",
    )

    assert result.status == expected_status
    assert result.simulator_success is expected_simulator_success
    assert result.error_message == report.error_message
    store = ResultStore(tmp_path / f"{expected_status}-results.jsonl", manifest)
    store.append(result)
    assert store.load() == (result,)
    qualification = json.loads(
        (Path(result.artifact_dir) / "qualification.json").read_text()
    )
    assert qualification["artifact_health"]["status"] == "unavailable"
    issues = qualification["artifact_health"]["issues"]
    assert issues == ["recording finalization failed", "recording close failed"]


def _result(manifest, attempt_id, status, finished_at):
    case = manifest.cases[0]
    success = status == "success"
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
        worker_pid=1,
        worker_exit_code=0,
        started_at="2026-08-07T00:00:00+00:00",
        finished_at=finished_at,
        artifact_dir=f"attempts/{attempt_id}",
        simulator_success=success,
        contract_success=success,
        terminated=success,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )


def test_result_store_retains_attempt_history_and_aggregate_uses_append_latest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    store = ResultStore(tmp_path / "results.jsonl", manifest)
    first = _result(manifest, "attempt-1", "terminal_failure", "2026-08-07T02:00:00Z")
    latest = _result(manifest, "attempt-2", "success", "2026-08-07T01:00:00Z")
    store.append(first)
    store.append(latest)

    assert store.load() == (first, latest)
    assert aggregate_results(manifest, store.load()).successful == 1


def test_cancelled_latest_attempt_is_retained_but_pending_for_resume(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    completed = _result(manifest, "attempt-1", "success", "2026-08-07T01:00:00Z")
    cancelled = replace(
        completed,
        attempt_id="attempt-2",
        status="cancelled",
        simulator_success=False,
        contract_success=False,
        terminated=False,
    )

    summary = aggregate_results(manifest, (completed, cancelled))

    assert summary.completed == 0
    assert summary.successful == 0
    assert summary.status_counts["cancelled"] == 1


class _Display:
    descriptors = (CameraDescriptor("agent", "Agent", "primary", 2, 2, 3, 20.0),)

    def frame(self, camera_id, observation):
        assert camera_id == "agent"
        return observation[camera_id]

    def frames(self, observation):
        return {"agent": self.frame("agent", observation)}


class _Episode:
    episode_id = "episode-1"

    def __init__(self):
        self.observer = None

    def set_transition_observer(self, observer):
        self.observer = observer


def test_pending_recorder_buffers_planning_events_and_installs_observer(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    attempt_root = tmp_path / "attempt"
    pending = PendingCaseRecorder(manifest, case, "attempt-1", attempt_root)
    pending.record(
        "configured",
        image=png_bytes(__import__("numpy").zeros((2, 2, 3), dtype="uint8")),
    )
    episode = _Episode()
    bootstrap = type("Bootstrap", (), {"display_adapter": _Display()})()

    recorder = pending.bind(bootstrap, episode)

    assert episode.observer is not None
    assert pending.recorder is recorder
    event = json.loads((attempt_root / "events.jsonl").read_text().splitlines()[0])
    assert event["event_type"] == "configured"


def test_open_episode_recording_timestamp_uses_simulator_reset_boundary(
    tmp_path: Path,
) -> None:
    # Reset, toolkit setup, measured step start/end, confirmed transition.
    times = iter((10.0, 13.0, 14.0, 14.5, 15.0))

    def monotonic():
        return next(times)

    observation = {
        "state": np.zeros(1, dtype=np.float32),
        "agent": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    class Simulator:
        episode_id = "episode-1"

        def reset(self, task, seed):
            del task, seed
            return observation

        def step(self, action):
            del action
            return StepResult("episode-1", 0, observation, 0.0, False, False, False)

        def close(self):
            pass

    class Policy:
        def reset(self, episode_id):
            del episode_id

        def close(self):
            pass

    class Binding:
        def bind_observation(self, value):
            return value

    class Toolkit:
        def attach_transition_observer(self, observer):
            self.observer = observer

        def close(self):
            pass

    def toolkit_factory(controller, initial_observation):
        del controller, initial_observation
        assert monotonic() == 13.0
        return Toolkit()

    bootstrap = EpisodeBootstrap(
        preset_name="test",
        task="task",
        seed=0,
        instruction="instruction",
        simulator_factory=Simulator,
        policy_factory=Policy,
        binding=Binding(),
        toolkit_factory=toolkit_factory,
        display_adapter=_Display(),
    )
    manifest = _manifest(tmp_path)
    root = tmp_path / "attempt-reset-time"
    pending = PendingCaseRecorder(
        manifest,
        manifest.cases[0],
        "attempt-reset-time",
        root,
        monotonic=monotonic,
    )

    with open_episode(bootstrap, monotonic=monotonic) as runtime:
        pending.bind(bootstrap, runtime)
        runtime.controller.execute_action(np.array([0.0], dtype=np.float32))

    pending.recorder.close()
    transition = json.loads(
        (root / "trajectory/transitions.jsonl").read_text().splitlines()[0]
    )
    assert transition["timestamp_s"] == pytest.approx(5.0)


def _raise_after_confirmed_transition(
    manifest: EvaluationManifest,
    case,
    attempt_root: Path,
    error: BaseException,
) -> None:
    pending = PendingCaseRecorder(manifest, case, attempt_root.name, attempt_root)
    episode = _Episode()
    pending.bind(type("Bootstrap", (), {"display_adapter": _Display()})(), episode)
    episode.observer.confirmed(
        np.array([0.25], dtype=np.float32),
        StepResult(
            episode_id="episode-1",
            step_id=0,
            observation={
                "state": np.array([1.0], dtype=np.float32),
                "agent": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
        ),
    )
    error.rpent_recorder = pending.recorder
    raise error


def _running_preparation(manifest: EvaluationManifest) -> Path:
    path = Path(manifest.runtime["evaluation_root"]) / "preparation.json"
    progress = PreparationProgress.create(
        path, manifest.evaluation_id, manifest.manifest_id
    )
    for phase in PREPARATION_PHASES[1:-1]:
        progress.advance(phase, "loading")
        progress.advance(phase, "ready")
    return path


@pytest.mark.parametrize("with_preparation", [False, True])
def test_runtime_error_after_transition_finalizes_partial_non_success(
    tmp_path: Path,
    with_preparation: bool,
) -> None:
    manifest = _manifest(tmp_path)
    if with_preparation:
        _running_preparation(manifest)
    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=lambda case, selected, root: _raise_after_confirmed_transition(
            selected, case, root, RuntimeError("planner failed")
        ),
        attempt_id="attempt-runtime-error",
    )

    assert result.status == "error"
    assert result.step_count == 1
    artifacts = json.loads((Path(result.artifact_dir) / "artifacts.json").read_text())
    assert artifacts["episode_result"]["status"] == "error"
    assert artifacts["transition_count"] == 1
    assert artifacts["health"]["status"] in {"complete", "degraded", "partial"}


@pytest.mark.parametrize("with_preparation", [False, True])
def test_keyboard_interrupt_after_transition_finalizes_cancelled_then_reraises(
    tmp_path: Path,
    with_preparation: bool,
) -> None:
    manifest = _manifest(tmp_path)
    if with_preparation:
        _running_preparation(manifest)
    root = (
        tmp_path / "cases" / manifest.cases[0].case_id / "attempts" / "attempt-keyboard"
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        run_episode_case(
            manifest.cases[0],
            manifest,
            harness_runner=lambda case, selected, attempt_root: (
                _raise_after_confirmed_transition(
                    selected, case, attempt_root, KeyboardInterrupt()
                )
            ),
            attempt_id="attempt-keyboard",
        )

    assert caught.value.rpent_result.status == "cancelled"
    assert caught.value.rpent_result.step_count == 1
    if with_preparation:
        progress = load_preparation(
            tmp_path / "preparation.json",
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
        )
        assert progress.phases[-1].status == "ready"
    artifacts = json.loads((root / "artifacts.json").read_text())
    assert artifacts["episode_result"]["status"] == "cancelled"
    assert artifacts["episode_result"]["simulator_success"] is False
    assert artifacts["transition_count"] == 1
    qualification = json.loads((root / "qualification.json").read_text())
    assert qualification["episode_result"]["status"] == "cancelled"


def test_worker_main_persists_finalized_cancellation_before_reraising(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    manifest.write_new(manifest_path)
    cancelled = _result(
        manifest,
        "attempt-main-interrupt",
        "cancelled",
        "2026-08-07T00:00:01Z",
    )

    def interrupt(*_args, **_kwargs):
        error = KeyboardInterrupt()
        error.rpent_result = cancelled
        raise error

    monkeypatch.setattr("rpent.evaluation.worker.run_episode_case", interrupt)

    with pytest.raises(KeyboardInterrupt):
        worker_main(
            [
                "--manifest",
                str(manifest_path),
                "--case-id",
                manifest.cases[0].case_id,
                "--attempt-id",
                cancelled.attempt_id,
                "--result",
                str(result_path),
            ]
        )

    assert EpisodeResult.from_payload(json.loads(result_path.read_text())) == cancelled


def test_worker_main_retains_cancelled_steps_with_running_preparation(
    monkeypatch, tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _running_preparation(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest.write_new(manifest_path)
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        "rpent.evaluation.worker._runner_for_manifest",
        lambda *_: lambda case, selected, root: _raise_after_confirmed_transition(
            selected, case, root, KeyboardInterrupt()
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        worker_main([
            "--manifest", str(manifest_path),
            "--case-id", manifest.cases[0].case_id,
            "--attempt-id", "cancelled-with-progress",
            "--result", str(result_path),
        ])

    result = EpisodeResult.from_payload(json.loads(result_path.read_text()))
    assert result.status == "cancelled"
    assert result.step_count == 1
    assert result.simulator_success is False


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled"])
def test_startup_error_retains_preparation_status(tmp_path: Path, status: str) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "preparation.json"
    progress = PreparationProgress.create(
        path, manifest.evaluation_id, manifest.manifest_id
    )
    if status != "pending":
        progress.advance("sam3_loading", "loading")
        progress.advance(
            "sam3_loading", status,
            error=PreparationError("RuntimeError", "startup failed")
            if status == "failed" else None,
        )
    original_progress = path.read_bytes()

    def fail(*_):
        raise RuntimeError("startup failed")

    result = run_episode_case(
        manifest.cases[0], manifest, harness_runner=fail,
        attempt_id="startup-error",
    )

    assert result.status == "error"
    assert result.error_message == "startup failed"
    assert result.step_count == 0
    assert path.read_bytes() == original_progress
