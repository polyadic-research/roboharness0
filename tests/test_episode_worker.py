from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.worker import (
    HarnessRunReport,
    _read_harness_report,
    load_case,
    main,
    run_episode_case,
)
from tests.synthetic_tasks import SyntheticTask


class _SnapshotRecorder:
    def __init__(self, transition_count):
        self.transition_count = transition_count

    def snapshot(self):
        return {"transition_count": self.transition_count}


def _write_transcript(root: Path, *, step_count=0, action_count=0) -> None:
    (root / "transcript_test.json").write_text(json.dumps({
        "terminal_classification": "SUCCESS",
        "harness": {"state": "complete"},
        "simulator_success": True,
        "contract_success": True,
        "episode_metrics": {
            "terminated": True,
            "truncated": False,
            "step_count": step_count,
            "action_count": action_count,
        },
        "error": None,
    }))


def test_harness_report_uses_recorded_transitions_as_counter_lower_bound(
    tmp_path,
) -> None:
    _write_transcript(tmp_path, step_count=0, action_count=0)

    report = _read_harness_report(0, tmp_path, _SnapshotRecorder(271))

    assert report.step_count == 271
    assert report.action_count == 271


@pytest.mark.parametrize("transition_count", [True, -1, 1.5, "3", None])
def test_harness_report_ignores_invalid_recorded_transition_counts(
    tmp_path, transition_count
) -> None:
    _write_transcript(tmp_path, step_count=2, action_count=1)

    report = _read_harness_report(
        0, tmp_path, _SnapshotRecorder(transition_count)
    )

    assert report.step_count == 2
    assert report.action_count == 1


def test_harness_report_keeps_genuine_zero_transition_episode(tmp_path) -> None:
    _write_transcript(tmp_path)

    report = _read_harness_report(0, tmp_path, _SnapshotRecorder(0))

    assert report.step_count == 0
    assert report.action_count == 0


def _manifest(tmp_path: Path) -> EvaluationManifest:
    task = SyntheticTask(
        suite="libero_10",
        index=3,
        name="task_3",
        language="instruction 3",
        bddl_identifier="libero_10/task_3.bddl",
        upstream_horizon=500,
    )
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="robocerebra-libero10-smolvla-v1",
        horizon_ceiling=480,
        checkpoint_repo_id="lerobot/smolvla_robocerebra",
        checkpoint_revision="7" * 40,
        binding_name="robocerebra-smolvla-v1",
    )
    return EvaluationManifest(
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={
            "argv": [
                "--planner",
                "api",
                "--planning-profile",
                "libero-authoritative-gated-v1",
            ]
        },
        environment={"name": "synthetic"},
        simulator={"revision": "b" * 40},
        policy={"lerobot_revision": "c" * 40},
        task_selector="3",
        cases=expand_episode_cases([task], protocol, base_seed=0),
        schedule={"episodes_per_task": 1, "base_seed": 0},
        contracts={"action": {"shape": [7]}},
        runtime={
            "main_interpreter": "/opt/python",
            "evaluation_root": str(tmp_path),
        },
        created_at="2026-08-05T12:00:00+10:00",
    )


def _success_report() -> HarnessRunReport:
    return HarnessRunReport(
        exit_code=0,
        terminal_classification="SUCCESS",
        harness_state="complete",
        simulator_success=True,
        contract_success=True,
        terminated=True,
        truncated=False,
        step_count=4,
        action_count=4,
        error_type=None,
        error_message=None,
    )


def test_load_case_verifies_named_case_before_execution(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    case = load_case(manifest, manifest.cases[0].case_id)
    assert case is manifest.cases[0]

    with pytest.raises(ValueError, match="unknown case"):
        load_case(manifest, "f" * 64)


def test_run_episode_case_calls_one_harness_entrypoint_and_scopes_artifacts(
    tmp_path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    calls = []

    def runner(selected_case, selected_manifest, artifact_dir):
        locator = json.loads(
            (artifact_dir.parents[1] / "current_attempt.json").read_text()
        )
        registration = json.loads((artifact_dir / "case.json").read_text())
        assert locator["attempt_id"] == artifact_dir.name
        assert locator["registered_at"] == registration["started_at"]
        calls.append((selected_case, selected_manifest, artifact_dir))
        return _success_report()

    result = run_episode_case(
        case,
        manifest,
        harness_runner=runner,
        worker_pid=4321,
        clock=iter(["2026-08-05T02:00:00+00:00", "2026-08-05T02:01:00+00:00"]).__next__,
    )

    assert len(calls) == 1
    assert calls[0][0] is case
    assert calls[0][1] is manifest
    assert calls[0][2].parent == tmp_path / "cases" / case.case_id / "attempts"
    assert calls[0][2].name == result.attempt_id
    assert result.status == "success"
    assert result.worker_pid == 4321
    assert result.simulator_success is True
    assert result.contract_success is True
    assert result.step_count == 4
    assert result.action_count == 4
    assert result.started_at == "2026-08-05T02:00:00+00:00"


def test_module_worker_accepts_report_from_imported_runner(
    monkeypatch, tmp_path
) -> None:
    import argparse

    import rpent.evaluation.worker as imported_worker

    class StopWorkerMain(Exception):
        pass

    spec = spec_from_file_location("__main__", imported_worker.__file__)
    assert spec is not None and spec.loader is not None
    module_worker = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "__main__", module_worker)
    monkeypatch.setitem(sys.modules, "rpent.evaluation.worker", imported_worker)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopWorkerMain),
    )
    with pytest.raises(StopWorkerMain):
        spec.loader.exec_module(module_worker)

    report_type = sys.modules["rpent.evaluation.worker"].HarnessRunReport
    report = report_type(
        exit_code=0,
        terminal_classification="SUCCESS",
        harness_state="complete",
        simulator_success=True,
        contract_success=True,
        terminated=True,
        truncated=False,
        step_count=4,
        action_count=4,
        error_type=None,
        error_message=None,
    )
    manifest = _manifest(tmp_path)
    result = module_worker.run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=lambda *_args: report,
    )

    assert result.status == "success"


@pytest.mark.parametrize(
    ("report", "status"),
    [
        (
            replace(
                _success_report(),
                exit_code=1,
                terminal_classification="FAILED",
                harness_state="terminal_failure",
                simulator_success=False,
                contract_success=False,
            ),
            "terminal_failure",
        ),
        (
            # 1.0.11 H2: a planner-declared failure is a failed episode.
            replace(
                _success_report(),
                exit_code=1,
                terminal_classification="FAILED",
                harness_state="needs_replan",
                simulator_success=False,
                contract_success=False,
            ),
            "terminal_failure",
        ),
        (
            replace(
                _success_report(),
                exit_code=1,
                terminal_classification="FAILED",
                simulator_success=False,
                contract_success=False,
                terminated=False,
                truncated=True,
            ),
            "truncated",
        ),
        (
            replace(
                _success_report(),
                exit_code=1,
                terminal_classification="FAILED",
                simulator_success=False,
                contract_success=False,
                error_type="TimeoutError",
            ),
            "timeout",
        ),
        (
            replace(
                _success_report(),
                exit_code=1,
                terminal_classification="FAILED",
                simulator_success=False,
                contract_success=False,
                error_type="RuntimeError",
                error_message="startup failed",
            ),
            "error",
        ),
    ],
)
def test_run_episode_case_maps_terminal_outcomes(tmp_path, report, status) -> None:
    manifest = _manifest(tmp_path)
    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=lambda *_args: report,
    )

    assert result.status == status


def test_startup_exception_becomes_one_error_result(tmp_path) -> None:
    manifest = _manifest(tmp_path)

    def fail(*_args):
        raise RuntimeError("policy preflight failed")

    result = run_episode_case(
        manifest.cases[0],
        manifest,
        harness_runner=fail,
    )

    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert result.error_message == "policy preflight failed"


def test_keyboard_interrupt_is_never_converted_to_success(tmp_path) -> None:
    manifest = _manifest(tmp_path)

    def interrupt(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_episode_case(
            manifest.cases[0],
            manifest,
            harness_runner=interrupt,
        )


def test_worker_main_writes_one_result_without_mutating_manifest(
    monkeypatch, tmp_path
) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "worker-result.json"
    manifest.write_new(manifest_path)
    before = manifest_path.read_bytes()
    calls = []

    def fake_run(case, loaded_manifest, *, attempt_id):
        calls.append((case.case_id, attempt_id))
        return run_episode_case(
            case,
            loaded_manifest,
            harness_runner=lambda *_args: _success_report(),
            attempt_id=attempt_id,
        )

    monkeypatch.setattr("rpent.evaluation.worker.run_episode_case", fake_run)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--case-id",
            manifest.cases[0].case_id,
            "--attempt-id",
            "attempt-main",
            "--result",
            str(result_path),
        ]
    )

    assert exit_code == 0
    assert calls == [(manifest.cases[0].case_id, "attempt-main")]
    assert manifest_path.read_bytes() == before
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert not (tmp_path / "summary.json").exists()


def test_worker_main_rejects_foreign_case_before_runner(monkeypatch, tmp_path) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest.write_new(manifest_path)

    def runner(*_args):
        pytest.fail("runner must not start")

    monkeypatch.setattr("rpent.evaluation.worker.run_episode_case", runner)

    with pytest.raises(ValueError, match="unknown case"):
        main(
            [
                "--manifest",
                str(manifest_path),
                "--case-id",
                "f" * 64,
                "--attempt-id",
                "attempt-foreign",
                "--result",
                str(tmp_path / "result.json"),
            ]
        )
