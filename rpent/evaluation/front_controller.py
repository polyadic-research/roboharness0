"""Public CLI to immutable manifest/coordinator assembly."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.dashboard.attachment import (
    DashboardDescriptor,
    LifecycleHint,
    LifecycleHintPublisher,
)
from rpent.evaluation.attempt_registration import AttemptRegistration
from rpent.evaluation.cases import EpisodeCase, expand_episode_cases
from rpent.evaluation.coordinator import EvaluationCoordinator
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.result_acceptance import ResultAcceptance
from rpent.evaluation.results import EpisodeResult, ResultStore
from rpent.evaluation.startup import EvaluationStartup, initialize_evaluation_startup
from rpent.evaluation.timeouts import worker_timeout_policy
from rpent.source_identity import deployed_source_commit
from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.repository import EvaluationRepository
from rpent.ui.schemas import CaseDetail
from rpent.utils.config import get_repo_root

GitIdentity = Callable[[], tuple[str, bool]]
_STORAGE_BYTES_PER_CAMERA_STEP_MIN = 8 * 1024
_STORAGE_BYTES_PER_CAMERA_STEP_MAX = 512 * 1024
_STORAGE_ESTIMATE_NOTE = "generic compressed-frame range; simulator output may vary"


@dataclass(slots=True)
class _CliAttachment:
    repository: EvaluationRepository
    notifier: LifecycleHintPublisher | None
    root: Path
    manifest: EvaluationManifest
    warning_emitted: bool = False

    def register(self, execution_state: str, *, authoritative: bool = False) -> bool:
        try:
            if execution_state == "running":
                self.repository.register_evaluation(
                    self.root,
                    self.manifest,
                    execution_state=execution_state,
                )
            else:
                self.repository.update_evaluation_list_item(
                    self.manifest.evaluation_id,
                    execution_state=execution_state,
                )
        except Exception:
            if authoritative:
                raise
            self.warn()
            return False
        return True

    def publish_persisted(self) -> None:
        self._publish(
            LifecycleHint(
                1,
                "evaluation_persisted",
                self.manifest.evaluation_id,
                self.manifest.manifest_id,
            )
        )

    def attempt_registered(
        self,
        registration: AttemptRegistration,
        _manifest: EvaluationManifest,
        _case: EpisodeCase,
    ) -> None:
        self._publish(
            LifecycleHint(
                1,
                "attempt_registered",
                registration.evaluation_id,
                registration.manifest_id,
                registration.case_id,
                registration.attempt_id,
            )
        )

    def accepted(self, result: EpisodeResult, detail: CaseDetail) -> None:
        try:
            self.repository.note_accepted_result(result, detail)
        except Exception:
            self.warn()
        self._publish(
            LifecycleHint(
                1,
                "result_accepted",
                result.evaluation_id,
                self.manifest.manifest_id,
                result.case_id,
                result.attempt_id,
            )
        )

    def warn(self) -> None:
        if self.warning_emitted:
            return
        self.warning_emitted = True
        print(
            "warning: dashboard attachment unavailable; evaluation continues",
            file=sys.stderr,
            flush=True,
        )

    def _publish(self, hint: LifecycleHint) -> None:
        if self.notifier is None:
            return
        try:
            queued = self.notifier.publish(hint)
        except Exception:
            queued = False
        if not queued:
            self.warn()


def _cli_attachment(
    root: Path,
    manifest: EvaluationManifest,
) -> tuple[EvaluationManifest, _CliAttachment | None, Mapping[str, str]]:
    catalog_root = (get_repo_root() / "logs").resolve()
    resolved_root = root.resolve()
    repository = EvaluationRepository((catalog_root,))
    from rpent.utils.config import get_dashboard_runtime_dir

    runtime_dir = get_dashboard_runtime_dir()
    descriptor_path = (
        runtime_dir if runtime_dir is not None else catalog_root
    ) / ".rpent-dashboard.json"
    attachment = _CliAttachment(
        repository,
        None,
        resolved_root,
        manifest,
    )
    worker_environment: Mapping[str, str] = {}
    try:
        DashboardDescriptor.load(
            descriptor_path,
            expected_runs_root=catalog_root,
            expected_descriptor_path=descriptor_path,
        )
        attachment.notifier = LifecycleHintPublisher(
            descriptor_path,
            expected_runs_root=catalog_root,
            on_failure=attachment.warn,
        )
    except Exception:
        pass
    else:
        worker_environment = {
            "RPENT_DASHBOARD_DESCRIPTOR": str(descriptor_path.resolve())
        }
    if attachment.notifier is None:
        attachment.warn()
    return manifest, attachment, worker_environment


def should_coordinate(args: argparse.Namespace, env_spec: Any) -> bool:
    """Select coordinated evaluation without changing foreground defaults."""
    explicit_recording = _has_explicit_recording_options(args)
    if bool(getattr(args, "interactive", False)):
        if explicit_recording:
            raise ValueError(
                "foreground interactive runs do not accept "
                "evaluation recording options"
            )
        return False
    if bool(getattr(args, "_episode_worker", False)):
        return False
    _validate_recording_options(args)
    return bool(
        getattr(env_spec, "coordinate_evaluation_by_default", False)
        or getattr(args, "episodes", None) is not None
        or getattr(args, "dry_run", False)
        or getattr(args, "resume", False)
        or getattr(args, "recording_mode", None) is not None
        or bool(getattr(args, "recording_camera", ()))
        or getattr(args, "require_complete_artifacts", False)
    )


def _has_explicit_recording_options(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "recording_mode", None) is not None
        or getattr(args, "recording_camera", ())
        or getattr(args, "require_complete_artifacts", False)
    )


def _validate_recording_options(args: argparse.Namespace) -> None:
    mode = getattr(args, "recording_mode", None)
    camera_ids = tuple(getattr(args, "recording_camera", ()) or ())
    if mode == "selected" and not camera_ids:
        raise ValueError("selected recording requires at least one camera")
    if camera_ids and mode != "selected":
        raise ValueError("recording cameras require --recording-mode selected")


def _effective_episode_count(args: argparse.Namespace) -> int:
    requested = getattr(args, "episodes", None)
    return 1 if requested is None else int(requested)


def build_evaluation_manifest(
    args: argparse.Namespace,
    env_spec: Any,
    argv: Iterable[str],
    *,
    resolved_tasks: Iterable[Any] | None = None,
    git_identity: GitIdentity | None = None,
    max_cases: int | None = None,
    launch: Mapping[str, Any] | None = None,
) -> tuple[EvaluationManifest, Any]:
    """Resolve all execution identities without starting a service."""
    argv_values = tuple(str(value) for value in argv)
    _validate_recording_options(args)
    run_config = env_spec.parse_config(args)
    prepared = env_spec.prepare_evaluation(args, run_config, resolved_tasks)
    generic_schedule = {
        "episodes_per_task": _effective_episode_count(args),
        "base_seed": args.seed,
        "seed_policy": "base_seed_plus_episode_index",
        "ordering": "task_registry_then_episode_index",
    }
    schedule = {
        **generic_schedule,
        **dict(getattr(prepared, "schedule", {})),
    }
    _validate_locked_inputs(
        args,
        argv_values,
        launch,
        schedule.get("locked_inputs", {}),
    )
    episodes = schedule["episodes_per_task"]
    if max_cases is not None:
        if (
            not isinstance(max_cases, int)
            or isinstance(max_cases, bool)
            or max_cases <= 0
        ):
            raise ValueError("max_cases must be a positive integer")
        requested_case_count = len(prepared.tasks) * episodes
        if requested_case_count > max_cases:
            raise ValueError(
                f"preview schedule supports at most {max_cases} Cases; "
                f"requested {requested_case_count}"
            )
    cases = expand_episode_cases(
        prepared.tasks,
        prepared.protocol,
        episodes=episodes,
        base_seed=schedule["base_seed"],
        seed_policy=schedule["seed_policy"],
    )
    schedule["worker_timeout"] = worker_timeout_policy(
        cases[0].environment, getattr(args, "worker_timeout_s", None)
    )
    commit, dirty = (git_identity or _git_identity)()
    submitted_mode = getattr(args, "recording_mode", None)
    contract_camera_count = _contract_camera_count(prepared.contracts)
    task_resolution = prepared.simulator.get("task_resolution", "environment_adapter")
    if not isinstance(task_resolution, str) or not task_resolution.strip():
        raise ValueError("simulator.task_resolution must be a non-empty string")
    default_recording_mode = (
        "none" if contract_camera_count == 0 or prepared.protocol.environment == "robocasa"
        else "all" if len(cases) == 1 else "primary"
    )
    recording_mode = submitted_mode or default_recording_mode
    camera_ids = tuple(getattr(args, "recording_camera", ()) or ())
    storage_estimate = estimate_recording_storage(
        cases,
        recording_mode,
        camera_ids,
        contract_camera_count=contract_camera_count,
        task_resolution=task_resolution,
    )
    worker_argv = _worker_base_argv(argv_values)
    planning_profile = prepared.policy.get("planning_profile")
    if isinstance(planning_profile, str) and planning_profile.strip():
        worker_argv.extend(["--planning-profile", planning_profile.strip()])
    manifest = EvaluationManifest(
        rpent_commit=commit,
        rpent_dirty=dirty,
        cli_arguments={
            "argv": worker_argv,
            "environment": args.env_name,
        },
        environment=prepared.environment,
        simulator=prepared.simulator,
        policy=prepared.policy,
        task_selector=str(prepared.environment.get("task_selector", args.task)),
        cases=cases,
        schedule=schedule,
        contracts=prepared.contracts,
        runtime=prepared.runtime,
        launch=_effective_launch(launch, prepared.environment, schedule),
        recording={
            "schema_version": 1,
            "mode": recording_mode,
            "camera_ids": camera_ids,
            "storage_estimate": storage_estimate,
            **({"llm_inputs_only": True,
                "llm_images": {"enabled": True, "storage": "exact_model_input_bytes",
                               "width": 256, "height": 256,
                               "cameras": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]}}
               if prepared.protocol.environment == "robocasa" and recording_mode == "none" else {}),
        },
        qualification={
            **{
                "require_complete_artifacts": bool(
                    getattr(args, "require_complete_artifacts", False)
                )
            },
            **dict(getattr(prepared, "qualification", {})),
        },
    )
    return manifest, run_config


def _effective_launch(
    launch: Mapping[str, Any] | None,
    environment: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected = environment.get("evaluation_profile")
    effective_inputs = schedule.get("effective_inputs")
    if effective_inputs is None:
        if selected is None:
            return {} if launch is None else launch
        effective_inputs = {"evaluation_profile": selected}
    if not isinstance(effective_inputs, Mapping):
        raise ValueError("schedule.effective_inputs must be a mapping")
    if launch:
        return {
            **dict(launch),
            "effective": {
                **dict(launch["effective"]),
                **dict(effective_inputs),
            },
        }
    return {
        "execution_mode": "cli",
        "preset_version": 1,
        "runner": "evaluation-front-controller",
        "submitted": {},
        "effective": dict(effective_inputs),
    }


def _validate_locked_inputs(
    args: argparse.Namespace,
    argv: tuple[str, ...],
    launch: Mapping[str, Any] | None,
    locked_inputs: Any,
) -> None:
    if not isinstance(locked_inputs, Mapping):
        raise ValueError("schedule.locked_inputs must be a mapping")
    if not locked_inputs:
        return
    explicit_options = {
        value.split("=", 1)[0] for value in argv if value.startswith("--")
    }
    for field, expected in locked_inputs.items():
        option = f"--{field.replace('_', '-')}"
        if option in explicit_options and getattr(args, field, None) != expected:
            raise ValueError(f"locked input {field!r} must equal {expected!r}")
    if not launch:
        return
    for section in ("submitted", "effective"):
        values = launch.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"launch.{section} must be a mapping")
        for field, expected in locked_inputs.items():
            if field in values and values[field] != expected:
                raise ValueError(
                    f"launch.{section} locked input {field!r} must equal {expected!r}"
                )


def estimate_recording_storage(
    cases: Iterable[Any],
    recording_mode: str,
    camera_ids: Iterable[str],
    *,
    contract_camera_count: int,
    task_resolution: str,
) -> dict[str, Any]:
    """Estimate retained image storage from the authoritative Case schedule."""
    scheduled = tuple(cases)
    total_horizon_steps = sum(int(case.horizon) for case in scheduled)
    selected_camera_count = len(tuple(camera_ids))
    if recording_mode == "none":
        camera_count_min = camera_count_max = 0
    elif recording_mode == "selected":
        if selected_camera_count > contract_camera_count:
            raise ValueError(
                "selected camera count cannot exceed contract camera count"
            )
        camera_count_min = camera_count_max = selected_camera_count
    elif recording_mode == "primary":
        camera_count_min = camera_count_max = min(1, contract_camera_count)
    elif recording_mode == "all":
        camera_count_min = camera_count_max = contract_camera_count
    else:
        raise ValueError(f"unknown recording mode: {recording_mode}")
    return {
        "minimum_bytes": (
            total_horizon_steps * camera_count_min * _STORAGE_BYTES_PER_CAMERA_STEP_MIN
        ),
        "maximum_bytes": (
            total_horizon_steps * camera_count_max * _STORAGE_BYTES_PER_CAMERA_STEP_MAX
        ),
        "basis": {
            "total_horizon_steps": total_horizon_steps,
            "retained_camera_count_min": camera_count_min,
            "retained_camera_count_max": camera_count_max,
            "bytes_per_camera_step_min": _STORAGE_BYTES_PER_CAMERA_STEP_MIN,
            "bytes_per_camera_step_max": _STORAGE_BYTES_PER_CAMERA_STEP_MAX,
            "contract_camera_count": contract_camera_count,
            "task_resolution": task_resolution,
            "note": _STORAGE_ESTIMATE_NOTE,
        },
    }


def _contract_camera_count(contracts: Mapping[str, Any]) -> int:
    images = contracts.get("images")
    if images is None:
        return 0
    if not isinstance(images, Mapping):
        raise ValueError("contracts.images must be a mapping")
    count = images.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("contracts.images.count must be a non-negative integer")
    return count


def run_evaluation_front_controller(
    args: argparse.Namespace,
    env_spec: Any,
    argv: Iterable[str],
    *,
    resolved_tasks: Iterable[Any] | None = None,
    git_identity: GitIdentity | None = None,
    policy_daemon_factory: Callable[..., Any] | None = None,
    coordinator_factory: Callable[..., Any] = EvaluationCoordinator,
) -> int:
    """Write/dry-run one manifest or execute it through the coordinator."""
    if args.resume:
        if args.output_dir is None:
            raise ValueError("--resume requires an explicit existing --output-dir")
        expected = Path(args.output_dir) / "evaluation_manifest.json"
        if not expected.exists():
            raise ValueError("--resume requires an existing evaluation manifest")
    manifest, run_config = build_evaluation_manifest(
        args,
        env_spec,
        argv,
        resolved_tasks=resolved_tasks,
        git_identity=git_identity,
    )
    root = Path(run_config.output_dir)
    manifest_path = root / "evaluation_manifest.json"
    if args.dry_run:
        manifest.write_new(manifest_path)
        print(
            f"dry-run manifest={manifest.manifest_id} cases={len(manifest.cases)} "
            f"path={manifest_path}",
            flush=True,
        )
        return 0

    manifest, attachment, worker_environment = _cli_attachment(root, manifest)
    runtimes: list[CaseRuntimeStore] = []
    attached = False
    outcome = None
    startup: EvaluationStartup | None = None
    try:
        def publish_persisted(
            persisted: EvaluationManifest, _selected_root: Path
        ) -> None:
            nonlocal attached
            if attachment is None:
                return
            attachment.manifest = persisted
            attached = attachment.register("running", authoritative=True)
            if attached:
                attachment.publish_persisted()

        startup = initialize_evaluation_startup(
            manifest,
            root,
            on_persisted=publish_persisted,
        )
        manifest = startup.manifest
        manifest_path = startup.root / "evaluation_manifest.json"
        startup.prepare_dependencies(None, lambda: False)
        if attachment is not None:
            attachment.manifest = manifest
        if policy_daemon_factory is not None:
            daemon = policy_daemon_factory(args=args, output_dir=root)
        else:
            daemon = env_spec.build_policy_daemon(args, root)
        startup.record(
            "simulator_starting",
            "policy_daemon_constructed",
        )
        effective_worker_environment = dict(worker_environment)
        effective_worker_environment.update(startup.worker_environment)
        coordinator_arguments = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "results_path": root / "results.jsonl",
            "summary_path": root / "summary.json",
            "policy_daemon": daemon,
            "worker_timeout_s": args.worker_timeout_s,
            "jobs": args.jobs,
            "worker_environment": effective_worker_environment,
        }
        if attached and attachment is not None:
            coordinator_arguments["attempt_registered"] = (
                attachment.attempt_registered
            )

            def result_acceptance_factory(
                selected_manifest: EvaluationManifest,
                store: ResultStore,
            ) -> ResultAcceptance:
                runtime = CaseRuntimeStore()
                runtimes.append(runtime)
                return ResultAcceptance(
                    manifest=selected_manifest,
                    store=store,
                    runtime_store=runtime,
                    assembler=lambda result: (
                        attachment.repository.reconstruct_case_detail(
                            result.evaluation_id,
                            result.case_id,
                            attempt_id=result.attempt_id,
                        )
                    ),
                    operation_reconstructor=lambda result: (
                        attachment.repository.reconstruct_operations(
                            result.evaluation_id,
                            result.case_id,
                            attempt_id=result.attempt_id,
                        )
                    ),
                    accepted_observer=attachment.accepted,
                )

            coordinator_arguments["result_acceptance_factory"] = (
                result_acceptance_factory
            )
        coordinator = coordinator_factory(**coordinator_arguments)
        startup.record(
            "simulator_starting",
            "coordinator_constructed",
        )
        outcome = coordinator.run(resume=args.resume)
        if outcome.interrupted:
            startup.cancel()
        elif getattr(outcome, "startup_error", False) or getattr(
            outcome, "infrastructure_error", False
        ):
            startup.fail(RuntimeError("coordinator infrastructure startup failed"))
        return outcome.exit_code
    except BaseException as exc:
        if startup is not None:
            startup.record(
                "simulator_starting",
                "coordinator_failed",
                error_type=_snake_error_type(type(exc).__name__),
            )
            if isinstance(exc, KeyboardInterrupt):
                startup.cancel()
            else:
                startup.fail(exc)
        raise
    finally:
        if attached and attachment is not None:
            state = (
                "cancelled"
                if outcome is not None and outcome.interrupted
                else "terminal"
            )
            attachment.register(state)
        for runtime in runtimes:
            try:
                runtime.close()
            except Exception:
                if attachment is not None:
                    attachment.warn()
        if attachment is not None and attachment.notifier is not None:
            close = getattr(attachment.notifier, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    try:
                        attachment.warn()
                    except BaseException:
                        pass


def _snake_error_type(name: str) -> str:
    return "".join(
        ("_" + char.lower()) if char.isupper() else char for char in name
    ).lstrip("_")


def _worker_base_argv(argv: Iterable[str]) -> list[str]:
    values = [str(value) for value in argv]
    value_options = {
        "--env",
        "--evaluation-profile",
        "--task",
        "--seed",
        "--max-episode-steps",
        "--output-dir",
        "--policy-endpoint",
        "--vla-endpoint",
        "--sam3-endpoint",
        "--env-endpoint",
        "--episodes",
        "--jobs",
        "--policy-devices",
        "--pi05-replica-memory-mib",
        "--xiaomi-replica-memory-mib",
        "--pi05-sessions-per-replica",
        "--gpu-reserve-mib",
        "--worker-timeout-s",
        "--recording-mode",
        "--recording-camera",
        "--submission-metadata-json",
        "--planning-profile",
        "--dashboard-host",
        "--dashboard-port",
        "--dashboard-language",
    }
    flag_options = {
        "--resume",
        "--dry-run",
        "--require-complete-artifacts",
        "--_episode-worker",
        "--dashboard",
    }
    retained: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        option = value.split("=", 1)[0]
        if option in flag_options:
            index += 1
            continue
        if option in value_options:
            index += 1 if "=" in value else 2
            continue
        retained.append(value)
        index += 1
    return retained


def _git_identity() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    deployed_commit = deployed_source_commit(root)
    if deployed_commit is not None:
        return deployed_commit, False
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        try:
            package_version = importlib.metadata.version("rpent")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        return f"installed-wheel:rpent@{package_version}", False
    return commit, dirty


__all__ = [
    "build_evaluation_manifest",
    "estimate_recording_storage",
    "run_evaluation_front_controller",
    "should_coordinate",
]
