"""Private process-isolated entrypoint for one evaluation case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from rpent.episode.recording import RecorderTransitionObserver
from rpent.evaluation.attempt_registration import (
    load_attempt_registration,
    load_case_registration,
    register_attempt,
)
from rpent.evaluation.batches import (
    validate_case_attempt_id,
    validate_task_batch_attempt_id,
)
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation_progress import (
    PreparationError,
    advance_preparation_from_environment,
    load_preparation,
)
from rpent.evaluation.results import EpisodeResult
from rpent.planning.manifest import canonical_json
from rpent.recording import CaseRecorder
from rpent.recording.contracts import (
    ArtifactHealth,
    QualificationOutcome,
    RecordingProfile,
)
from rpent.recording.media import atomic_write_json
from rpent.recording.narrated.capture import capture_initial
from rpent.recording.notifications import notification_sink

if __name__ == "__main__":
    sys.modules["rpent.evaluation.worker"] = sys.modules[__name__]


class PendingCaseRecorder:
    """Buffer startup events until reset supplies the authoritative episode ID."""

    def __init__(
        self,
        manifest: EvaluationManifest,
        case: EpisodeCase,
        attempt_id: str,
        root: Path,
        notify: Callable[[Any], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manifest = manifest
        self.case = case
        self.attempt_id = attempt_id
        self.root = root
        self.notify = notify
        self.monotonic = monotonic
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._recorder: CaseRecorder | None = None

    @property
    def recorder(self) -> CaseRecorder | None:
        return self._recorder

    def record(self, event: str, **payload: Any) -> dict[str, Any]:
        if self._recorder is None:
            self._pending.append((event, payload))
            return {"event": event, **payload}
        return self._recorder.record(event, **payload)

    def bind(self, bootstrap: Any, episode: Any) -> CaseRecorder:
        if self._recorder is not None:
            raise RuntimeError("Case recorder is already bound")
        profile = RecordingProfile.validate(
            str(self.manifest.recording["mode"]),
            tuple(self.manifest.recording.get("camera_ids", ())),
            tuple(bootstrap.display_adapter.descriptors),
        )
        recorder = CaseRecorder(
            root=self.root,
            evaluation_id=self.manifest.evaluation_id,
            case_id=self.case.case_id,
            attempt_id=self.attempt_id,
            episode_id=episode.episode_id,
            display=bootstrap.display_adapter,
            profile=profile,
            notify=self.notify,
            llm_inputs_only=bool(self.manifest.recording.get("llm_inputs_only", False)),
        )
        observer = RecorderTransitionObserver(
            recorder,
            monotonic=self.monotonic,
            reset_at=getattr(episode, "reset_monotonic", None),
        )
        episode.set_transition_observer(observer)
        self._recorder = recorder
        for event, payload in self._pending:
            recorder.record(event, **payload)
        self._pending.clear()
        metadata = dict(getattr(bootstrap, "runtime_metadata", None) or {})
        if metadata.get("policy_family") == "xiaomi-mibot":
            health = metadata.get("policy_health")
            if not isinstance(health, dict):
                raise ValueError("Xiaomi recording requires validated runtime policy health")
            policy = {
                **self.manifest.to_payload()["policy"],
                "artifact_identity": health["artifact_identity"],
                "remote_code_revision": health["remote_code_revision"],
                "server_version": health["server_version"],
                "health": health,
            }
            atomic_write_json(self.root / "policy.json", policy)
            recorder.record("policy_identity_verified", **metadata)
        initial_observation = getattr(episode, "observation", None)
        if initial_observation is not None and not recorder.llm_inputs_only:
            capture_initial(recorder, bootstrap.display_adapter, initial_observation)
        return recorder


@dataclass(frozen=True, slots=True)
class HarnessRunReport:
    """Structured outcome returned by the existing single-Harness entrypoint."""

    exit_code: int
    terminal_classification: Literal["SUCCESS", "FAILED", "INCOMPLETE"]
    harness_state: str | None
    simulator_success: bool
    contract_success: bool
    terminated: bool
    truncated: bool
    step_count: int
    action_count: int
    error_type: str | None
    error_message: str | None
    recorder: Any | None = None


HarnessRunner = Callable[
    [EpisodeCase, EvaluationManifest, Path],
    HarnessRunReport,
]


def load_case(manifest: EvaluationManifest, case_id: str) -> EpisodeCase:
    """Resolve exactly one already-verified case without launching resources."""
    matches = tuple(case for case in manifest.cases if case.case_id == case_id)
    if len(matches) != 1:
        raise ValueError(f"unknown case ID for manifest: {case_id}")
    return matches[0]


def run_episode_case(
    case: EpisodeCase,
    manifest: EvaluationManifest,
    *,
    harness_runner: HarnessRunner | None = None,
    worker_pid: int | None = None,
    clock: Callable[[], str] | None = None,
    attempt_id: str | None = None,
    attempt_id_factory: Callable[[], str] | None = None,
    task_batch_attempt_id: str | None = None,
) -> EpisodeResult:
    """Run one existing Harness episode and map it to a retained result."""
    selected = load_case(manifest, case.case_id)
    if selected != case:
        raise ValueError("case identity does not match immutable manifest")
    evaluation_root = Path(str(manifest.runtime.get("evaluation_root", ".")))
    if attempt_id is not None and attempt_id_factory is not None:
        raise ValueError("provide attempt_id or attempt_id_factory, not both")
    attempt_id = str(
        attempt_id if attempt_id is not None else (attempt_id_factory or uuid4)()
    )
    validate_case_attempt_id(attempt_id)
    if task_batch_attempt_id is not None:
        validate_task_batch_attempt_id(task_batch_attempt_id)
    now = clock or _utc_now
    proposed_started_at = _normalized_utc_timestamp(now(), "started_at")
    artifact_dir = evaluation_root / "cases" / case.case_id / "attempts" / attempt_id
    case_path = artifact_dir / "case.json"
    if case_path.exists():
        registration = load_case_registration(
            case_path,
            manifest=manifest,
            case=case,
            attempt_id=attempt_id,
        )
        load_attempt_registration(
            evaluation_root / "cases" / case.case_id / "current_attempt.json",
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            case_id=case.case_id,
            attempt_id=attempt_id,
        )
    else:
        registration = register_attempt(
            manifest,
            case,
            attempt_id,
            proposed_started_at,
        )
    started_at = registration.registered_at
    runner = harness_runner or _runner_for_manifest(case, manifest)
    interrupted: KeyboardInterrupt | None = None
    try:
        report = runner(case, manifest, artifact_dir)
        if not isinstance(report, HarnessRunReport):
            raise TypeError("Harness runner must return HarnessRunReport")
    except KeyboardInterrupt as exc:
        _write_error_artifact(artifact_dir, exc)
        report = _failure_report(exc, getattr(exc, "rpent_recorder", None), 130)
        interrupted = exc
    except Exception as exc:
        _write_error_artifact(artifact_dir, exc)
        report = _failure_report(exc, getattr(exc, "rpent_recorder", None), 1)
    finished_at = _normalized_utc_timestamp(now(), "finished_at")
    status = "cancelled" if interrupted is not None else _result_status(report)
    result = EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        task_batch_attempt_id=task_batch_attempt_id,
        status=status,
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=os.getpid() if worker_pid is None else worker_pid,
        worker_exit_code=report.exit_code,
        started_at=started_at,
        finished_at=finished_at,
        artifact_dir=str(artifact_dir),
        simulator_success=report.simulator_success,
        contract_success=report.contract_success,
        terminated=report.terminated,
        truncated=report.truncated,
        step_count=report.step_count,
        action_count=report.action_count,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
        error_type=report.error_type,
        error_message=report.error_message,
    )
    _advance_finalization(manifest, "loading")
    _finalize_recording(artifact_dir, manifest, result, report.recorder)
    _advance_finalization(manifest, "ready")
    if interrupted is not None:
        interrupted.rpent_result = result
        raise interrupted
    return result


def _runner_for_manifest(
    case: EpisodeCase, manifest: EvaluationManifest
) -> HarnessRunner:
    mode = str(manifest.launch.get("execution_mode") or "hybrid")
    if mode == "hybrid":
        return _run_cli_harness_episode
    from rpent.envs.base import get_env_spec

    runner = get_env_spec(case.environment).evaluation_runners.get(mode)
    if runner is None:
        raise ValueError(
            f"environment {case.environment!r} does not support execution mode {mode!r}"
        )
    return runner


def _failure_report(
    exc: BaseException,
    recorder: Any | None,
    exit_code: int,
) -> HarnessRunReport:
    transition_count = _recorded_transition_count(recorder)
    return HarnessRunReport(
        exit_code=exit_code,
        terminal_classification="FAILED",
        harness_state=None,
        simulator_success=False,
        contract_success=False,
        terminated=False,
        truncated=False,
        step_count=transition_count,
        action_count=transition_count,
        error_type=type(exc).__name__,
        error_message=str(exc),
        recorder=recorder,
    )


def _write_case_projection(
    artifact_dir: Path,
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
    started_at: str,
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "attempt_id": attempt_id,
        "started_at": started_at,
        "case": case.to_payload(),
        "recording": dict(manifest.recording),
        "qualification": dict(manifest.qualification),
    }
    (artifact_dir / "case.json").write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )


def _finalize_recording(
    artifact_dir: Path,
    manifest: EvaluationManifest,
    result: EpisodeResult,
    recorder: Any | None,
) -> None:
    if recorder is None:
        health = ArtifactHealth(
            "unavailable", ("recorder was unavailable before finalization",)
        )
    else:
        try:
            artifacts = recorder.finalize(result)
            health = ArtifactHealth.from_payload(artifacts["health"])
        except Exception:
            issues = ["recording finalization failed"]
            close = getattr(recorder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    issues.append("recording close failed")
            health = ArtifactHealth("unavailable", tuple(issues))
    require_complete = bool(
        manifest.qualification.get("require_complete_artifacts", False)
    )
    qualification = QualificationOutcome.evaluate(result, health, require_complete)
    try:
        encoded = canonical_json(qualification.to_payload()) + "\n"
        (artifact_dir / "qualification.json").write_text(encoded, encoding="utf-8")
    except Exception:
        unavailable = QualificationOutcome.evaluate(
            result,
            ArtifactHealth("unavailable", ("qualification write failed",)),
            require_complete,
        )
        try:
            (artifact_dir / "qualification_error.txt").write_text(
                canonical_json(unavailable.to_payload()) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass


def _result_status(report: HarnessRunReport) -> str:
    if report.truncated:
        return "truncated"
    if report.error_type == "TimeoutError":
        return "timeout"
    if report.terminal_classification == "SUCCESS":
        if report.simulator_success and report.contract_success:
            return "success"
        return "error"
    if report.harness_state == "terminal_failure":
        return "terminal_failure"
    # A planner that declares failure ends a normal failed episode, not a
    # worker error that marks the whole batch non-retryable (1.0.11 H2).
    if report.terminal_classification == "FAILED" and report.error_type is None:
        return "terminal_failure"
    return "error"


def _run_cli_harness_episode(
    case: EpisodeCase,
    manifest: EvaluationManifest,
    artifact_dir: Path,
) -> HarnessRunReport:
    """Invoke the existing public single-episode Harness path in this worker."""
    from rpent.cli.main import main as run_cli

    endpoint = os.environ.get("RPENT_SHARED_POLICY_ENDPOINT") or str(
        manifest.runtime.get("policy_endpoint", "")
    )
    shared_required = bool(manifest.runtime.get("shared_policy_required", True))
    if shared_required and not endpoint:
        raise RuntimeError("worker requires the coordinator shared policy endpoint")
    configured = manifest.cli_arguments.get("argv", ())
    if not isinstance(configured, (list, tuple)):
        raise ValueError("manifest cli_arguments.argv must be a list")
    selector = case.task_index if case.task_index is not None else case.task_name
    argv = [str(value) for value in configured]
    argv.extend(
        [
            "--env",
            case.environment,
            "--task",
            str(selector),
            "--seed",
            str(case.seed),
            "--max-episode-steps",
            str(case.horizon),
            "--output-dir",
            str(artifact_dir),
            "--_episode-worker",
        ]
    )
    if endpoint:
        argv.extend(
            [
                str(manifest.runtime.get("policy_endpoint_argument")),
                endpoint,
            ]
        )
    for endpoint_name, argument_name, fallback_argument in (
        ("simulator_endpoint", "simulator_endpoint_argument", "--env-endpoint"),
        ("sam3_endpoint", "sam3_endpoint_argument", "--sam3-endpoint"),
    ):
        resolved = str(manifest.runtime.get(endpoint_name) or "")
        if resolved:
            argv.extend(
                [
                    str(manifest.runtime.get(argument_name) or fallback_argument),
                    resolved,
                ]
            )
    notify = _notification_sink(manifest)
    pending = PendingCaseRecorder(
        manifest,
        case,
        artifact_dir.name,
        artifact_dir,
        notify=notify,
    )
    try:
        def run() -> HarnessRunReport:
            exit_code = run_cli(argv, worker_recording=pending)
            return _read_harness_report(exit_code, artifact_dir, pending.recorder)

        return _with_preparation_environment(manifest, run)
    except BaseException as exc:
        _fail_preparation(manifest, exc)
        recorder = pending.recorder
        if recorder is not None:
            try:
                recorder.record_event(
                    "worker_interrupted",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                pass
        try:
            exc.rpent_recorder = recorder
        except Exception:
            pass
        raise
    finally:
        close = getattr(notify, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                pass


def _with_preparation_environment(
    manifest: EvaluationManifest, function: Callable[[], Any]
) -> Any:
    names = {
        "RPENT_PREPARATION_PATH": str(
            Path(str(manifest.runtime["evaluation_root"])) / "preparation.json"
        ),
        "RPENT_EVALUATION_ID": manifest.evaluation_id,
        "RPENT_MANIFEST_ID": manifest.manifest_id,
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        return function()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _advance_finalization(
    manifest: EvaluationManifest, status: Literal["loading", "ready"]
) -> None:
    if str(manifest.launch.get("execution_mode") or "hybrid") != "hybrid":
        return
    snapshot = load_preparation(
        Path(str(manifest.runtime["evaluation_root"])) / "preparation.json",
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    # Startup failures retain their preparation status. Episodes that reached
    # running must finalize their records even when the runner was interrupted.
    if snapshot is None or snapshot.status != "ready":
        return
    _with_preparation_environment(
        manifest,
        lambda: advance_preparation_from_environment("finalizing", status),
    )


def _fail_preparation(manifest: EvaluationManifest, exc: BaseException) -> None:
    path = Path(str(manifest.runtime["evaluation_root"])) / "preparation.json"
    snapshot = load_preparation(
        path,
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
    )
    if snapshot is None or snapshot.status != "preparing":
        return
    phase = next(item for item in snapshot.phases if item.phase == snapshot.current_phase)
    if phase.status != "loading":
        return
    message = " ".join(str(exc).splitlines()).strip() or "worker preparation failed"
    _with_preparation_environment(
        manifest,
        lambda: advance_preparation_from_environment(
            snapshot.current_phase,
            "failed",
            error=PreparationError(type(exc).__name__[:4096], message[:4096]),
        ),
    )


def _notification_sink(manifest: EvaluationManifest):
    descriptor_path = os.environ.get("RPENT_DASHBOARD_DESCRIPTOR")
    if descriptor_path:
        descriptor = Path(descriptor_path)
        return notification_sink(
            descriptor_path=descriptor,
        )
    return notification_sink(
        manifest.runtime.get("notification_endpoint"),
        os.environ.get("RPENT_NOTIFICATION_TOKEN"),
    )


def _read_harness_report(
    exit_code: int,
    artifact_dir: Path,
    recorder: CaseRecorder | None,
) -> HarnessRunReport:
    transcripts = sorted(artifact_dir.glob("transcript_*.json"))
    if len(transcripts) != 1:
        raise RuntimeError(
            f"single Harness run produced {len(transcripts)} transcript artifacts"
        )
    try:
        payload = json.loads(transcripts[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read Harness transcript: {exc}") from exc
    harness = payload.get("harness") or {}
    metrics = payload.get("episode_metrics") or {}
    error = payload.get("error")
    error_type, error_message = _split_error(error)
    transition_count = _recorded_transition_count(recorder)
    return HarnessRunReport(
        exit_code=int(exit_code),
        terminal_classification=str(
            payload.get("terminal_classification", "INCOMPLETE")
        ),
        harness_state=harness.get("state", harness.get("status")),
        simulator_success=payload.get("simulator_success") is True,
        contract_success=payload.get("contract_success") is True,
        terminated=metrics.get("terminated") is True,
        truncated=metrics.get("truncated") is True,
        step_count=max(_nonnegative_count(metrics.get("step_count")), transition_count),
        action_count=max(
            _nonnegative_count(metrics.get("action_count")), transition_count
        ),
        error_type=error_type,
        error_message=error_message,
        recorder=recorder,
    )


def _recorded_transition_count(recorder: Any | None) -> int:
    if recorder is None:
        return 0
    try:
        return _nonnegative_count(recorder.snapshot().get("transition_count"))
    except Exception:
        return 0


def _nonnegative_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _split_error(error: Any) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    rendered = str(error)
    error_type, separator, message = rendered.partition(":")
    if not separator:
        return "RuntimeError", rendered
    return error_type.strip() or "RuntimeError", message.strip() or rendered


def _write_error_artifact(artifact_dir: Path, exc: BaseException) -> None:
    rendered = "".join(
        traceback.TracebackException.from_exception(
            exc,
            capture_locals=False,
        ).format()
    )
    (artifact_dir / "worker_error.txt").write_text(rendered, encoding="utf-8")


def _write_result(path: Path, result: EpisodeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(result.to_payload()) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ValueError(f"worker result path already exists: {path}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_utc_timestamp(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware UTC timestamp") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware timestamp")
    return parsed.astimezone(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--result", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate, run, and persist exactly one worker result."""
    args = _build_parser().parse_args(argv)
    manifest = EvaluationManifest.load(args.manifest)
    case = load_case(manifest, args.case_id)
    try:
        result = run_episode_case(case, manifest, attempt_id=args.attempt_id)
    except KeyboardInterrupt as exc:
        result = getattr(exc, "rpent_result", None)
        if isinstance(result, EpisodeResult):
            _write_result(Path(args.result), result)
        raise
    _write_result(Path(args.result), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HarnessRunReport",
    "PendingCaseRecorder",
    "load_case",
    "main",
    "run_episode_case",
]
