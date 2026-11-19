from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import rpent.cli.main as cli
import rpent.evaluation.front_controller as front_controller
from rpent.dashboard.attachment import DashboardLease
from rpent.envs import get_env_spec
from rpent.envs.launch import LaunchPrerequisite
from rpent.evaluation.attempt_registration import register_attempt
from rpent.evaluation.cases import EpisodeProtocol
from rpent.evaluation.front_controller import (
    build_evaluation_manifest,
    run_evaluation_front_controller,
    should_coordinate,
)
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation import PreparedEvaluation
from rpent.evaluation.preparation_progress import load_preparation
from rpent.evaluation.result_acceptance import (
    FinalizedResultCandidate,
    ReapedWorker,
)
from rpent.evaluation.results import ResultStore
from rpent.evaluation.startup_log import read_startup_events
from rpent.recording import media as recording_media
from rpent.ui.repository import EvaluationRepository
from tests.test_episode_coordinator import _result


def _parse(argv: list[str]):
    with patch.object(sys, "argv", ["rpent", *argv]):
        parser = cli._build_argparser()
        early, _ = parser.parse_known_args()
        spec = get_env_spec(early.env_name)
        spec.add_cli_args(parser, use_dashboard=False)
        return parser.parse_args()


def _tasks(count=10):
    return tuple(
        SimpleNamespace(
            suite="libero_10",
            index=index,
            name=f"task_{index}",
            language=f"instruction {index}",
            bddl_identifier=f"task_{index}.bddl",
            upstream_horizon=500,
        )
        for index in range(count)
    )


def _robocasa_tasks(count=18):
    return tuple(
        SimpleNamespace(
            name=f"Task{index}",
            instruction=f"upstream instruction {index}",
            upstream_horizon=500 + index,
            family="atomic",
            category="fixtures",
            split="target",
            registry_commit="456174f62b89b8fca99eaaf33949c29fec9cfc2a",
            registry_digest="sha256:" + "1" * 64,
        )
        for index in range(count)
    )


def _synthetic_args(tmp_path):
    return SimpleNamespace(
        env_name="synthetic",
        task="synthetic_task",
        episodes=1,
        seed=7,
        resume=False,
        output_dir=tmp_path,
        dry_run=False,
        worker_timeout_s=30.0,
        jobs=1,
    )


def _synthetic_spec(tmp_path, *, daemon=None, daemon_calls=None):
    task = SimpleNamespace(
        index=0,
        name="synthetic_task",
        language="perform synthetic task",
        upstream_horizon=5,
    )
    prepared = PreparedEvaluation(
        tasks=(task,),
        protocol=EpisodeProtocol(
            environment="synthetic",
            preset_name="synthetic-v1",
            horizon_ceiling=10,
            checkpoint_repo_id="local/synthetic",
            checkpoint_revision="revision",
            binding_name="synthetic-binding-v1",
        ),
        environment={"name": "synthetic", "protocol": "rpent-episode-v1"},
        simulator={"name": "synthetic-simulator"},
        policy={"checkpoint": {"repo_id": "local/synthetic"}},
        contracts={"action": {"shape": [1]}},
        runtime={
            "main_interpreter": sys.executable,
            "evaluation_root": str(tmp_path),
            "policy_endpoint": None,
            "shared_policy_required": False,
        },
    )

    def build_policy_daemon(args, output_dir):
        if daemon_calls is not None:
            daemon_calls.append((args.env_name, Path(output_dir)))
        return daemon

    return SimpleNamespace(
        name="synthetic",
        parse_config=lambda _args: SimpleNamespace(output_dir=Path(tmp_path)),
        prepare_evaluation=lambda _args, _config, _tasks: prepared,
        build_policy_daemon=build_policy_daemon,
        resolve_planning_profile=lambda _args: "libero-reactive-v1",
    )


def test_synthetic_env_spec_builds_manifest_without_core_registration(
    tmp_path,
) -> None:
    args = _synthetic_args(tmp_path)
    manifest, _config = build_evaluation_manifest(
        args,
        _synthetic_spec(tmp_path),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
    )

    assert manifest.environment["name"] == "synthetic"
    assert manifest.simulator["name"] == "synthetic-simulator"
    assert manifest.cases[0].task_name == "synthetic_task"


@pytest.mark.parametrize("episodes", [10_001, (1 << 53) - 1])
def test_preview_case_limit_rejects_before_case_materialization(
    tmp_path, episodes
) -> None:
    args = _synthetic_args(tmp_path)
    args.episodes = episodes
    with patch(
        "rpent.evaluation.front_controller.expand_episode_cases"
    ) as expand:
        with pytest.raises(ValueError, match="at most 10000 Cases"):
            build_evaluation_manifest(
                args,
                _synthetic_spec(tmp_path),
                ["--env", "synthetic", "--episodes", str(episodes)],
                git_identity=lambda: ("a" * 40, False),
                max_cases=10_000,
            )

    expand.assert_not_called()


def test_preview_case_limit_allows_exact_boundary_to_expand(tmp_path) -> None:
    args = _synthetic_args(tmp_path)
    args.episodes = 10_000
    with patch(
        "rpent.evaluation.front_controller.expand_episode_cases",
        side_effect=RuntimeError("boundary expanded"),
    ) as expand:
        with pytest.raises(RuntimeError, match="boundary expanded"):
            build_evaluation_manifest(
                args,
                _synthetic_spec(tmp_path),
                ["--env", "synthetic", "--episodes", "10000"],
                git_identity=lambda: ("a" * 40, False),
                max_cases=10_000,
            )

    expand.assert_called_once()


def test_synthetic_env_spec_owns_policy_daemon_without_core_registration(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
    args = _synthetic_args(tmp_path)
    daemon = object()
    daemon_calls = []
    coordinators = []

    class RecordingCoordinator:
        def __init__(self, **kwargs):
            coordinators.append(kwargs)

        def run(self, *, resume):
            assert resume is False
            return SimpleNamespace(exit_code=0, interrupted=False)

    exit_code = run_evaluation_front_controller(
        args,
        _synthetic_spec(
            tmp_path,
            daemon=daemon,
            daemon_calls=daemon_calls,
        ),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
        coordinator_factory=RecordingCoordinator,
    )

    assert exit_code == 0
    assert daemon_calls == [("synthetic", Path(tmp_path))]
    assert coordinators[0]["policy_daemon"] is daemon
    manifest = coordinators[0]["manifest"]
    preparation = load_preparation(
        tmp_path / "preparation.json",
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    assert preparation is not None
    assert [phase.status for phase in preparation.phases[:4]] == [
        "ready",
        "skipped",
        "skipped",
        "loading",
    ]
    assert coordinators[0]["worker_environment"] == {
        "RPENT_PREPARATION_PATH": str((tmp_path / "preparation.json").resolve()),
        "RPENT_EVALUATION_ID": manifest.evaluation_id,
        "RPENT_MANIFEST_ID": manifest.manifest_id,
    }
    assert [event.event for event in read_startup_events(tmp_path / "startup.jsonl")][
        :2
    ] == ["execution_started", "phase_started"]


def test_catalog_cli_publishes_durable_attachment_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "evaluation"
    args = _synthetic_args(root)
    calls = []

    class RecordingRepository(EvaluationRepository):
        def register_evaluation(
            self, selected_root, manifest, *, execution_state=None
        ):
            super().register_evaluation(
                selected_root,
                manifest,
                execution_state=execution_state,
            )
            if execution_state in {"terminal", "cancelled"}:
                calls.append(
                    ("list_state", manifest.evaluation_id, execution_state)
                )

        def update_evaluation_list_item(
            self, evaluation_id, *, execution_state=None
        ):
            item = super().update_evaluation_list_item(
                evaluation_id,
                execution_state=execution_state,
            )
            calls.append(("list_state", evaluation_id, execution_state))
            return item

    repository = RecordingRepository((catalog,))

    class Notifier:
        def publish(self, hint):
            if hint.kind == "evaluation_persisted":
                assert (root / "evaluation_manifest.json").is_file()
                assert (catalog / ".rpent-evaluations.json").is_file()
                calls.append(("persisted", hint.evaluation_id))
            elif hint.kind == "attempt_registered":
                locator = root / "cases" / hint.case_id / "current_attempt.json"
                assert locator.is_file()
                calls.append(
                    ("attempt", hint.evaluation_id, hint.case_id, hint.attempt_id)
                )
            else:
                assert hint.kind == "result_accepted"
                assert hint.attempt_id in (root / "results.jsonl").read_text()
                calls.append(
                    ("accepted", hint.evaluation_id, hint.case_id, hint.attempt_id)
                )
            return True

        def close(self):
            calls.append(("notifier_closed",))

    notifier = Notifier()

    class RecordingCoordinator:
        def __init__(self, **kwargs):
            self.arguments = kwargs
            assert kwargs["manifest_path"].is_file()

        def run(self, *, resume):
            assert resume is False
            manifest = self.arguments["manifest"]
            case = manifest.cases[0]
            registration = register_attempt(
                manifest,
                case,
                "attempt-1",
                "2026-08-14T03:04:05+00:00",
            )
            self.arguments["attempt_registered"](registration, manifest, case)
            result = _result(
                manifest,
                case,
                status="success",
                attempt_id="attempt-1",
            )
            acceptance = self.arguments["result_acceptance_factory"](
                manifest,
                ResultStore(self.arguments["results_path"], manifest),
            )
            acceptance.accept(
                FinalizedResultCandidate(result, ReapedWorker("attempt-1", 0))
            )
            return SimpleNamespace(exit_code=0, interrupted=False)

    lease = DashboardLease.acquire(catalog)
    try:
        descriptor = lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(
            front_controller,
            "get_repo_root",
            lambda: tmp_path,
            raising=False,
        )
        monkeypatch.setattr(
            front_controller,
            "EvaluationRepository",
            lambda _roots: repository,
            raising=False,
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: notifier,
            raising=False,
        )

        exit_code = run_evaluation_front_controller(
            args,
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            git_identity=lambda: ("a" * 40, False),
            coordinator_factory=RecordingCoordinator,
        )
    finally:
        lease.close()

    assert exit_code == 0
    persisted = (root / "evaluation_manifest.json").read_text()
    assert "worker-secret" not in persisted
    assert str(descriptor.path) not in persisted
    manifest = EvaluationManifest.load(root / "evaluation_manifest.json")
    case = manifest.cases[0]
    assert calls == [
        ("persisted", manifest.evaluation_id),
        ("attempt", manifest.evaluation_id, case.case_id, "attempt-1"),
        ("accepted", manifest.evaluation_id, case.case_id, "attempt-1"),
        ("list_state", manifest.evaluation_id, "terminal"),
        ("notifier_closed",),
    ]
    item = EvaluationRepository((catalog,)).list_evaluations()[0]
    assert item.execution_state == "terminal"
    assert item.completed == 1
    assert item.metrics.completed == 1
    assert item.metrics.successful == 1
    assert item.metrics.success_rate == 1.0
    detail = EvaluationRepository((catalog,)).get_evaluation(manifest.evaluation_id)
    assert detail.preparation is not None
    assert detail.preparation.current_phase == "simulator_starting"


def test_custom_output_outside_catalog_registers_globally(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = tmp_path / "custom"
    args = _synthetic_args(root)
    coordinator_calls = []

    class RecordingCoordinator:
        def __init__(self, **kwargs):
            coordinator_calls.append(kwargs)

        def run(self, *, resume):
            assert resume is False
            return SimpleNamespace(exit_code=0, interrupted=False)

    monkeypatch.setattr(
        front_controller,
        "get_repo_root",
        lambda: tmp_path,
        raising=False,
    )
    assert run_evaluation_front_controller(
        args,
        _synthetic_spec(root),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
        coordinator_factory=RecordingCoordinator,
    ) == 0

    assert coordinator_calls
    assert "attempt_registered" in coordinator_calls[0]
    assert "result_acceptance_factory" in coordinator_calls[0]
    payload = json.loads((catalog / ".rpent-evaluations.json").read_text())
    assert payload["evaluations"][0]["artifact_root"] == str(root.resolve())
    assert not (root / ".rpent-evaluations.json").exists()


def test_interrupted_catalog_cli_materializes_cancelled_list_state(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "interrupted"
    args = _synthetic_args(root)
    repository = EvaluationRepository((catalog,))

    class InterruptedCoordinator:
        def __init__(self, **_kwargs):
            pass

        def run(self, *, resume):
            assert resume is False
            return SimpleNamespace(exit_code=1, interrupted=True)

    lease = DashboardLease.acquire(catalog)
    try:
        lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(
            front_controller,
            "get_repo_root",
            lambda: tmp_path,
            raising=False,
        )
        monkeypatch.setattr(
            front_controller,
            "EvaluationRepository",
            lambda _roots: repository,
            raising=False,
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: SimpleNamespace(publish=lambda _hint: True),
            raising=False,
        )

        exit_code = run_evaluation_front_controller(
            args,
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            git_identity=lambda: ("a" * 40, False),
            coordinator_factory=InterruptedCoordinator,
        )
    finally:
        lease.close()

    assert exit_code == 1
    assert EvaluationRepository((catalog,)).list_evaluations()[0].execution_state == (
        "cancelled"
    )
    manifest = EvaluationManifest.load(root / "evaluation_manifest.json")
    preparation = load_preparation(
        root / "preparation.json",
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    assert preparation is not None
    assert preparation.status == "cancelled"
    assert preparation.current_phase == "simulator_starting"


def test_catalog_attachment_failures_warn_once_without_changing_exit(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "warning"
    args = _synthetic_args(root)
    repository = EvaluationRepository((catalog,))

    class RejectingNotifier:
        def publish(self, _hint):
            return False

    class RecordingCoordinator:
        def __init__(self, **kwargs):
            self.arguments = kwargs

        def run(self, *, resume):
            assert resume is False
            manifest = self.arguments["manifest"]
            case = manifest.cases[0]
            registration = register_attempt(
                manifest,
                case,
                "attempt-1",
                "2026-08-14T03:04:05+00:00",
            )
            self.arguments["attempt_registered"](registration, manifest, case)
            return SimpleNamespace(exit_code=23, interrupted=False)

    lease = DashboardLease.acquire(catalog)
    try:
        lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
        monkeypatch.setattr(
            front_controller,
            "EvaluationRepository",
            lambda _roots: repository,
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: RejectingNotifier(),
        )

        exit_code = run_evaluation_front_controller(
            args,
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            git_identity=lambda: ("a" * 40, False),
            coordinator_factory=RecordingCoordinator,
        )
    finally:
        lease.close()

    assert exit_code == 23
    assert capsys.readouterr().err.splitlines() == [
        "warning: dashboard attachment unavailable; evaluation continues"
    ]
    assert EvaluationRepository((catalog,)).list_evaluations()[0].execution_state == (
        "terminal"
    )


def test_catalog_coordinator_construction_failure_materializes_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "construction-failure"
    args = _synthetic_args(root)
    repository = EvaluationRepository((catalog,))

    def fail_construction(**_kwargs):
        raise RuntimeError("coordinator construction failed")

    lease = DashboardLease.acquire(catalog)
    try:
        lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
        monkeypatch.setattr(
            front_controller,
            "EvaluationRepository",
            lambda _roots: repository,
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: SimpleNamespace(publish=lambda _hint: True),
        )

        with pytest.raises(RuntimeError, match="construction failed"):
            run_evaluation_front_controller(
                args,
                _synthetic_spec(root),
                ["--env", "synthetic", "--task", "synthetic_task"],
                git_identity=lambda: ("a" * 40, False),
                coordinator_factory=fail_construction,
            )
    finally:
        lease.close()

    assert EvaluationRepository((catalog,)).list_evaluations()[0].execution_state == (
        "terminal"
    )
    manifest = EvaluationManifest.load(root / "evaluation_manifest.json")
    preparation = load_preparation(
        root / "preparation.json",
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    assert preparation is not None
    assert preparation.status == "failed"
    assert preparation.current_phase == "simulator_starting"


def test_catalog_resume_preserves_pre_feature_manifest_identity(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "legacy-run"
    args = _synthetic_args(root)
    args.resume = True
    manifest, run_config = build_evaluation_manifest(
        args,
        _synthetic_spec(root),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
    )
    manifest.write_new(root / "evaluation_manifest.json")
    before = (root / "evaluation_manifest.json").read_bytes()
    coordinator_calls = []

    class Coordinator:
        def __init__(self, **kwargs):
            coordinator_calls.append(kwargs)

        def run(self, *, resume):
            assert resume is True
            return SimpleNamespace(exit_code=0, interrupted=False)

    lease = DashboardLease.acquire(catalog)
    try:
        lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
        monkeypatch.setattr(
            front_controller,
            "build_evaluation_manifest",
            lambda *_args, **_kwargs: (manifest, run_config),
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: SimpleNamespace(
                publish=lambda _hint: True,
                close=lambda: None,
            ),
        )

        exit_code = run_evaluation_front_controller(
            args,
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            coordinator_factory=Coordinator,
        )
    finally:
        lease.close()

    assert exit_code == 0
    assert coordinator_calls[0]["manifest"].manifest_id == manifest.manifest_id
    assert (root / "evaluation_manifest.json").read_bytes() == before
    assert "evaluation_catalog_root" not in manifest.runtime
    preparation = load_preparation(
        root / "preparation.json",
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    assert preparation is not None
    assert preparation.current_phase == "simulator_starting"


def test_catalog_corrupt_index_fails_before_coordinator_launch(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    catalog.mkdir()
    (catalog / ".rpent-evaluations.json").write_text("{", encoding="utf-8")
    root = catalog / "evaluation"
    calls = []
    monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)

    with pytest.raises(ValueError):
        run_evaluation_front_controller(
            _synthetic_args(root),
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            git_identity=lambda: ("a" * 40, False),
            coordinator_factory=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []


def test_catalog_unwritable_initial_index_fails_before_coordinator_launch(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "evaluation"
    calls = []
    original = recording_media.atomic_write_json

    def reject_index(path, payload):
        if Path(path).name == ".rpent-evaluations.json":
            raise PermissionError("evaluation index is unwritable")
        return original(path, payload)

    monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(recording_media, "atomic_write_json", reject_index)

    with pytest.raises(PermissionError, match="unwritable"):
        run_evaluation_front_controller(
            _synthetic_args(root),
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            git_identity=lambda: ("a" * 40, False),
            coordinator_factory=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []
    assert (root / "evaluation_manifest.json").is_file()


def test_catalog_conflicting_identity_fails_before_coordinator_launch(
    tmp_path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "evaluation"
    args = _synthetic_args(root)
    manifest, run_config = build_evaluation_manifest(
        args,
        _synthetic_spec(root),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
    )
    conflicting = replace(
        manifest,
        runtime={**manifest.runtime, "identity_conflict": True},
    )
    conflicting_root = catalog / "conflicting"
    conflicting.write_new(conflicting_root / "evaluation_manifest.json")
    EvaluationRepository((catalog,)).register_evaluation(
        conflicting_root,
        conflicting,
    )
    calls = []
    monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        front_controller,
        "build_evaluation_manifest",
        lambda *_args, **_kwargs: (manifest, run_config),
    )

    with pytest.raises(ValueError, match="duplicate evaluation_id"):
        run_evaluation_front_controller(
            args,
            _synthetic_spec(root),
            ["--env", "synthetic", "--task", "synthetic_task"],
            coordinator_factory=lambda **kwargs: calls.append(kwargs),
        )

    assert calls == []


def test_catalog_malformed_descriptor_is_advisory_and_generic(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "evaluation"
    catalog.mkdir()
    descriptor = catalog / ".rpent-dashboard.json"
    descriptor.write_text('{"notification_token":"do-not-print"', encoding="utf-8")
    descriptor.chmod(0o600)
    monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def run(self, *, resume):
            assert resume is False
            return SimpleNamespace(exit_code=17, interrupted=False)

    assert run_evaluation_front_controller(
        _synthetic_args(root),
        _synthetic_spec(root),
        ["--env", "synthetic", "--task", "synthetic_task"],
        git_identity=lambda: ("a" * 40, False),
        coordinator_factory=Coordinator,
    ) == 17

    assert capsys.readouterr().err == (
        "warning: dashboard attachment unavailable; evaluation continues\n"
    )
    assert EvaluationRepository((catalog,)).list_evaluations()[0].execution_state == (
        "terminal"
    )


@pytest.mark.parametrize(
    "error",
    (RuntimeError("coordinator run failed"), KeyboardInterrupt()),
    ids=("exception", "keyboard_interrupt"),
)
def test_catalog_coordinator_run_failure_reraises_after_terminal_index(
    tmp_path,
    monkeypatch,
    capsys,
    error,
) -> None:
    catalog = tmp_path / "logs"
    root = catalog / "evaluation"
    repository = EvaluationRepository((catalog,))
    close_calls = []

    class Notifier:
        def publish(self, _hint):
            return True

        def close(self):
            close_calls.append("close")
            raise OSError("injected notifier close failure")

    class Coordinator:
        def __init__(self, **_kwargs):
            pass

        def run(self, *, resume):
            assert resume is False
            raise error

    lease = DashboardLease.acquire(catalog)
    try:
        lease.publish("http://127.0.0.1:43123", "worker-secret")
        monkeypatch.setattr(front_controller, "get_repo_root", lambda: tmp_path)
        monkeypatch.setattr(
            front_controller,
            "EvaluationRepository",
            lambda _roots: repository,
        )
        monkeypatch.setattr(
            front_controller,
            "LifecycleHintPublisher",
            lambda *_args, **_kwargs: Notifier(),
        )

        with pytest.raises(type(error), match=str(error) or None):
            run_evaluation_front_controller(
                _synthetic_args(root),
                _synthetic_spec(root),
                ["--env", "synthetic", "--task", "synthetic_task"],
                git_identity=lambda: ("a" * 40, False),
                coordinator_factory=Coordinator,
            )
    finally:
        lease.close()

    assert EvaluationRepository((catalog,)).list_evaluations()[0].execution_state == (
        "terminal"
    )
    assert close_calls == ["close"]
    assert capsys.readouterr().err == (
        "warning: dashboard attachment unavailable; evaluation continues\n"
    )


def test_public_main_routes_synthetic_env_without_shared_choice_edit(
    monkeypatch,
) -> None:
    calls = []
    env_lookups = []
    spec = SimpleNamespace(
        name="synthetic",
        add_cli_args=lambda parser, use_dashboard: parser.add_argument(
            "--task", required=not use_dashboard
        ),
        resolve_planning_profile=lambda args: args.planning_profile,
    )

    def get_spec(name):
        env_lookups.append(name)
        return spec

    monkeypatch.setattr(cli, "get_env_spec", get_spec)
    monkeypatch.setattr(
        "rpent.evaluation.front_controller.should_coordinate",
        lambda args, env_spec: True,
    )

    def route(args, env_spec, argv):
        calls.append((args.env_name, env_spec.name, argv))
        return 23

    monkeypatch.setattr(
        "rpent.evaluation.front_controller.run_evaluation_front_controller",
        route,
    )

    result = cli.main(
        [
            "--env",
            "synthetic",
            "--task",
            "synthetic_task",
            "--planning-profile",
            "libero-reactive-v1",
        ]
    )

    assert result == 23
    assert env_lookups == ["synthetic"]
    assert calls == [
        (
            "synthetic",
            "synthetic",
            [
                "--env",
                "synthetic",
                "--task",
                "synthetic_task",
                "--planning-profile",
                "libero-reactive-v1",
            ],
        )
    ]


@pytest.mark.parametrize("dry_run", [False, True])
def test_public_main_checks_launch_prerequisites_only_for_execution(
    monkeypatch,
    capsys,
    dry_run,
) -> None:
    calls = []
    preset = SimpleNamespace(
        cli_preflight=Mock(return_value=(
            LaunchPrerequisite(
                "planner",
                "Planner",
                "blocked",
                "Codex authentication is unavailable.",
            ),
        )),
    )
    spec = SimpleNamespace(
        name="synthetic",
        add_cli_args=lambda parser, use_dashboard: parser.add_argument(
            "--task", required=not use_dashboard
        ),
        resolve_planning_profile=lambda args: args.planning_profile,
        launch_preset=lambda: preset,
    )
    monkeypatch.setattr(cli, "get_env_spec", lambda _name: spec)
    monkeypatch.setattr(
        "rpent.evaluation.front_controller.should_coordinate",
        lambda _args, _env_spec: True,
    )
    monkeypatch.setattr(
        "rpent.evaluation.front_controller.run_evaluation_front_controller",
        lambda *_args: calls.append("coordinator") or 23,
    )

    result = cli.main([
        "--env",
        "synthetic",
        "--task",
        "synthetic_task",
        "--planning-profile",
        "libero-reactive-v1",
        *(["--dry-run"] if dry_run else []),
    ])

    if dry_run:
        assert result == 23
        assert calls == ["coordinator"]
        preset.cli_preflight.assert_not_called()
        assert capsys.readouterr().err == ""
        return

    assert result == 1
    assert calls == []
    preset.cli_preflight.assert_called_once()
    assert capsys.readouterr().err == (
        "rpent: error: Launch prerequisites are not ready: "
        "Codex authentication is unavailable.\n"
    )
















@pytest.mark.parametrize("environment", ["robocasa"])
def test_attached_formal_runtimes_record_endpoints_not_local_interpreters(
    monkeypatch,
    tmp_path,
    environment,
) -> None:
    monkeypatch.setenv("RPENT_SMOLVLA_PYTHON", "/opt/smolvla/bin/python")
    monkeypatch.setenv("RPENT_ROBOCASA_PYTHON", "/opt/robocasa/bin/python")
    task = "3" if environment == "robocasa" else "CloseFridge"
    argv = [
        "--env",
        environment,
        "--task",
        task,
        "--planning-profile",
        "authoritative-gated-v1",
        "--env-endpoint",
        "http://simulator:8001",
        "--policy-endpoint",
        "http://policy:8002",
        "--output-dir",
        str(tmp_path / environment),
    ]
    args = _parse(argv)
    spec = get_env_spec(environment)
    tasks = _robocasa_tasks(1)

    prepared = spec.prepare_evaluation(args, spec.parse_config(args), tasks)

    assert prepared.runtime["simulator_interpreter"] is None
    assert prepared.runtime["simulator_endpoint"] == "http://simulator:8001"
    assert prepared.runtime["policy_interpreter"] is None
    assert prepared.runtime["policy_endpoint"] == "http://policy:8002"




@pytest.mark.parametrize(
    ("selector", "task_count", "episodes", "expected_cases"),
    [
        ("CloseFridge", 1, 1, 1),
        ("CloseFridge,OpenDrawer", 2, 1, 2),
        ("atomic_seen", 18, 20, 360),
        ("pretrain50", 50, 1, 50),
    ],
)
def test_robocasa_public_schedule_and_provenance(
    tmp_path, selector, task_count, episodes, expected_cases
) -> None:
    argv = [
        "--env",
        "robocasa",
        "--task",
        selector,
        "--episodes",
        str(episodes),
        "--planning-profile",
        "authoritative-gated-v1",
        "--output-dir",
        str(tmp_path),
    ]
    args = _parse(argv)

    manifest, _config = build_evaluation_manifest(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=_robocasa_tasks(task_count),
        git_identity=lambda: ("a" * 40, False),
    )

    assert len(manifest.cases) == expected_cases
    assert manifest.environment["name"] == "robocasa"
    assert manifest.simulator["robocasa_commit"] == (
        "456174f62b89b8fca99eaaf33949c29fec9cfc2a"
    )
    assert manifest.simulator["robosuite_commit"] == (
        "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
    )
    assert len(manifest.simulator["tasks"]) == task_count
    assert manifest.simulator["tasks"][0]["family"] == "atomic"
    assert manifest.simulator["tasks"][0]["upstream_horizon"] == 500
    assert manifest.policy["checkpoint"]["repo_id"] == "lerobot/smolvla_robocasa"
    assert manifest.contracts["state"]["shape"] == (16,)
    assert manifest.contracts["images"] == {
        "count": 4,
        "dtype": "uint8",
        "shapes": {
            "robot0_agentview_center": (512, 768, 3),
            "robot0_agentview_left": (256, 256, 3),
            "robot0_agentview_right": (256, 256, 3),
            "robot0_eye_in_hand": (256, 256, 3),
        },
    }
    assert manifest.policy["input_camera_ids"] == (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    assert manifest.contracts["action"]["shape"] == (12,)
    assert all(case.task_index is None for case in manifest.cases)


def test_robocasa_manifest_preserves_upstream_horizon_without_ceiling(
    tmp_path,
) -> None:
    argv = [
        "--env",
        "robocasa",
        "--task",
        "CloseFridge",
        "--planning-profile",
        "authoritative-gated-v1",
        "--output-dir",
        str(tmp_path),
    ]
    args = _parse(argv)
    tasks = list(_robocasa_tasks(1))
    tasks[0].upstream_horizon = 7200

    manifest, config = build_evaluation_manifest(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=tasks,
        git_identity=lambda: ("a" * 40, False),
    )

    assert config.safety_ceiling is None
    assert manifest.cases[0].horizon == 7200
    assert manifest.runtime["requested_safety_ceiling"] is None
    assert manifest.contracts["horizon_rule"] == "upstream_horizon_or_explicit_min"


def test_manifest_materializes_one_canonical_planning_profile(tmp_path) -> None:
    argv = [
        "--env",
        "robocasa",
        "--task",
        "CloseFridge",
        "--planning-profile",
        "libero-authoritative-gated-v1",
        "--output-dir",
        str(tmp_path),
    ]
    args = _parse(argv)

    manifest, _config = build_evaluation_manifest(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=_robocasa_tasks(1),
        git_identity=lambda: ("a" * 40, False),
    )

    worker_argv = list(manifest.cli_arguments["argv"])
    assert manifest.policy["planning_profile"] == "authoritative-gated-v1"
    assert worker_argv.count("--planning-profile") == 1
    index = worker_argv.index("--planning-profile")
    assert worker_argv[index + 1] == "authoritative-gated-v1"


def test_xiaomi_profile_materializes_independent_policy_manifest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "RPENT_XIAOMI_MIBOT_PYTHON", "/workspace/envs/rpent-xiaomi-mibot-py312/bin/python"
    )
    monkeypatch.setenv(
        "RPENT_XIAOMI_MIBOT_SNAPSHOT",
        "/models/0d1aa76d0d82debc9b611e4d1e231096434d5be4",
    )
    argv = [
        "--env",
        "robocasa",
        "--task",
        "KettleBoiling",
        "--seed",
        "1001",
        "--episodes",
        "1",
        "--planning-profile",
        "robocasa-xiaomi-mibot-gated-v1",
        "--actions-per-policy-request",
        "16",
        "--worker-timeout-s",
        "7200",
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    ]
    args = _parse(argv)
    task = _robocasa_tasks(1)[0]
    task.name = "KettleBoiling"

    manifest, _config = build_evaluation_manifest(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=(task,),
        git_identity=lambda: ("a" * 40, False),
    )

    assert manifest.policy["policy_family"] == "xiaomi-mibot"
    assert manifest.policy["checkpoint"]["revision"] == (
        "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
    )
    assert manifest.policy["binding_name"] == "robocasa-xiaomi-mibot-v1"
    assert manifest.policy["observation_history"] == 4
    assert manifest.policy["observation_interval"] == 2
    assert manifest.policy["crop_ratio"] == 0.95
    assert manifest.schedule["actions_per_policy_request"] == 16
    assert manifest.runtime["policy_interpreter"].endswith("py312/bin/python")
    assert manifest.runtime["policy_snapshot_path"].endswith(
        "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
    )






def test_public_evaluation_argument_validation() -> None:
    with pytest.raises(SystemExit) as missing_env:
        cli.main([])
    assert missing_env.value.code == 2
    with pytest.raises(SystemExit):
        _parse(["--env", "robocasa", "--task", "3", "--episodes", "0"])
    with pytest.raises(SystemExit):
        _parse(["--env", "robocasa", "--task", "3", "--jobs", "0"])
    with pytest.raises(SystemExit):
        _parse(["--env", "robocasa", "--task", "3", "--profile", "x"])


def test_dry_run_writes_one_manifest_without_starting_services(tmp_path) -> None:
    argv = [
        "--env",
        "robocasa",
        "--task",
        "all",
        "--episodes",
        "10",
        "--planning-profile",
        "authoritative-gated-v1",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    ]
    args = _parse(argv)

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run must not construct policy or coordinator services")

    exit_code = run_evaluation_front_controller(
        args,
        get_env_spec("robocasa"),
        argv,
        resolved_tasks=_robocasa_tasks(10),
        git_identity=lambda: ("a" * 40, False),
        policy_daemon_factory=forbidden,
        coordinator_factory=forbidden,
    )

    assert exit_code == 0
    manifests = list(tmp_path.glob("evaluation_manifest.json"))
    assert len(manifests) == 1


def test_resume_requires_explicit_existing_output(tmp_path) -> None:
    argv = [
        "--env",
        "robocasa",
        "--task",
        "3",
        "--resume",
        "--output-dir",
        str(tmp_path),
    ]
    args = _parse(argv)

    with pytest.raises(ValueError, match="existing.*manifest"):
        run_evaluation_front_controller(
            args,
            get_env_spec("robocasa"),
            argv,
            resolved_tasks=_tasks(1),
            git_identity=lambda: ("a" * 40, False),
        )




def test_interactive_and_internal_worker_stay_foreground() -> None:
    base = _parse(["--env", "robocasa", "--task", "3"])
    spec = get_env_spec("robocasa")
    for name in ("interactive", "_episode_worker"):
        values = vars(base).copy()
        values[name] = True
        assert should_coordinate(SimpleNamespace(**values), spec) is False


def test_dashboard_server_command_is_not_an_evaluation_foreground_mode() -> None:
    args = _parse(["--env", "robocasa", "--task", "3", "--dashboard"])

    assert should_coordinate(args, get_env_spec("robocasa")) is True


def test_dashboard_server_command_options_are_stripped_from_worker_argv() -> None:
    assert front_controller._worker_base_argv(
        [
            "--dashboard",
            "--dashboard-host",
            "0.0.0.0",
            "--dashboard-port=43123",
            "--dashboard-language",
            "zh-cn",
            "--verbose",
        ]
    ) == ["--verbose"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--recording-mode", "selected"],
        ["--recording-mode", "none", "--recording-camera", "agent"],
        ["--recording-mode", "primary", "--recording-camera", "agent"],
        ["--recording-mode", "all", "--recording-camera", "agent"],
        ["--recording-camera", "agent"],
    ],
)
def test_invalid_recording_cli_combinations_are_rejected(tmp_path, argv) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--task",
            "3",
            "--output-dir",
            str(tmp_path),
            *argv,
        ]
    )

    with pytest.raises(ValueError, match="recording|camera"):
        build_evaluation_manifest(
            args,
            get_env_spec("robocasa"),
            argv,
            resolved_tasks=_tasks(1),
            git_identity=lambda: ("a" * 40, False),
        )


@pytest.mark.parametrize(
    "recording_option",
    [
        ["--recording-mode", "all"],
        ["--recording-camera", "agent"],
        ["--require-complete-artifacts"],
    ],
)
def test_foreground_rejects_explicit_recording_options(
    recording_option,
) -> None:
    args = _parse(
        [
            "--env",
            "robocasa",
            "--task",
            "3",
            "--interactive",
            *recording_option,
        ]
    )

    with pytest.raises(ValueError, match="foreground|recording"):
        should_coordinate(args, get_env_spec("robocasa"))


def test_no_public_benchmark_entrypoint_or_profile_alias() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "rpent-benchmark" not in pyproject
    assert "--profile" not in cli._build_argparser().format_help()


def test_public_main_routes_noninteractive_request_before_runtime_start(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setenv("RPENT_SMOLVLA_PYTHON", sys.executable)
    monkeypatch.setenv("RPENT_ROBOCASA_PYTHON", sys.executable)

    def route(args, env_spec, argv):
        calls.append((args.env_name, env_spec.name, argv))
        return 23

    monkeypatch.setattr(
        "rpent.evaluation.front_controller.run_evaluation_front_controller",
        route,
    )

    result = cli.main(
        [
            "--env",
            "robocasa",
            "--task",
            "3",
            "--planning-profile",
            "authoritative-gated-v1",
        ]
    )

    assert result == 23
    assert calls == [
        (
            "robocasa",
            "robocasa",
            [
                "--env",
                "robocasa",
                "--task",
                "3",
                "--planning-profile",
                "authoritative-gated-v1",
            ],
        )
    ]
