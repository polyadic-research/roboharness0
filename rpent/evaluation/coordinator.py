"""Bounded process coordinator for evaluation manifests."""

from __future__ import annotations

import importlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from rpent.evaluation.attempt_registration import (
    AttemptRegistration,
    register_attempt,
)
from rpent.evaluation.batches import (
    TaskBatchEvidence,
    TaskBatchProcessOutcome,
    append_task_batch_process_outcome,
    evaluate_task_batch_evidence,
    load_task_batch_attempts,
    load_task_batch_process_outcomes,
    select_complete_task_batch,
    validate_task_batch_attempt_id,
    validate_task_batch_result,
)
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.control import CancellationToken
from rpent.evaluation.diagnostics import aggregate_rpent_diagnostics
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import aggregate_official_metrics
from rpent.evaluation.result_acceptance import (
    FinalizedResultCandidate,
    NoWorker,
    ReapedWorker,
    ResultAcceptance,
)
from rpent.evaluation.results import (
    EpisodeResult,
    EvaluationSummary,
    ResultStore,
    aggregate_results,
    write_summary,
)
from rpent.evaluation.startup_log import record_startup_event
from rpent.evaluation.timeouts import (
    normalize_worker_timeout_s,
    resolve_group_worker_timeout_s,
    resolve_worker_timeout_s,
)
from rpent.utils.config import get_repo_root as get_repo_root


@dataclass(frozen=True, slots=True)
class CoordinatorOutcome:
    exit_code: int
    completed: int
    scheduled: int
    interrupted: bool
    infrastructure_error: bool
    summary: EvaluationSummary
    startup_error: bool = False


WorkerLauncher = Callable[..., Any]
ProgressReporter = Callable[[EpisodeCase, EpisodeResult, EvaluationSummary], None]
AttemptRegistered = Callable[
    [AttemptRegistration, EvaluationManifest, EpisodeCase], None
]
ResultAcceptanceFactory = Callable[[EvaluationManifest, ResultStore], ResultAcceptance]
_ALLOWED_WORKER_ENVIRONMENT = frozenset(
    {
        "RPENT_NOTIFICATION_TOKEN",
        "RPENT_DASHBOARD_DESCRIPTOR",
        "RPENT_PREPARATION_PATH",
        "RPENT_EVALUATION_ID",
        "RPENT_MANIFEST_ID",
    }
)


def _snake_error_type(name: str) -> str:
    return "".join(
        ("_" + char.lower()) if char.isupper() else char for char in name
    ).lstrip("_")


class EvaluationCoordinator:
    """Write one manifest and run each pending case in a fresh worker."""

    def __init__(
        self,
        *,
        manifest: EvaluationManifest,
        manifest_path: str | Path,
        results_path: str | Path,
        summary_path: str | Path,
        policy_daemon: Any | None,
        worker_launcher: WorkerLauncher | None = None,
        worker_timeout_s: float | None = None,
        jobs: int = 1,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
        worker_environment: Mapping[str, str] | None = None,
        attempt_registered: AttemptRegistered | None = None,
        result_acceptance_factory: ResultAcceptanceFactory | None = None,
    ) -> None:
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
            raise ValueError("jobs must be a positive integer")
        if jobs > 1 and not hasattr(policy_daemon, "lanes"):
            raise ValueError("jobs=1 is required without exclusive policy lanes")
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.results_path = Path(results_path)
        self.summary_path = Path(summary_path)
        self.policy_daemon = policy_daemon
        self.worker_launcher = worker_launcher or _default_worker_launcher
        self.worker_timeout_s = normalize_worker_timeout_s(worker_timeout_s)
        self.jobs = jobs
        self._result_lock = threading.RLock()
        self.progress = progress or _print_progress
        self.cancellation = cancellation or CancellationToken()
        selected_environment = {} if worker_environment is None else worker_environment
        if not isinstance(selected_environment, Mapping) or any(
            key not in _ALLOWED_WORKER_ENVIRONMENT
            or not isinstance(value, str)
            or not value
            or "\r" in value
            or "\n" in value
            for key, value in selected_environment.items()
        ):
            raise ValueError("invalid worker environment")
        self.worker_environment = MappingProxyType(dict(selected_environment))
        if attempt_registered is not None and not callable(attempt_registered):
            raise TypeError("attempt_registered must be callable")
        self.attempt_registered = attempt_registered
        if result_acceptance_factory is not None and not callable(
            result_acceptance_factory
        ):
            raise TypeError("result_acceptance_factory must be callable")
        self.result_acceptance_factory = result_acceptance_factory

    def cancel(self, reason: str) -> bool:
        """Request cancellation of this coordinator's active run."""
        return self.cancellation.cancel(reason)

    def run(self, *, resume: bool = False) -> CoordinatorOutcome:
        """Run all pending cases while retaining every terminal worker outcome."""
        raw_grouping = self.manifest.schedule.get("worker_grouping")
        grouping = "case" if raw_grouping is None else raw_grouping
        if not isinstance(grouping, str) or grouping not in {"case", "task"}:
            raise ValueError(f"unknown worker_grouping: {grouping!r}")
        if grouping == "task":
            if self.jobs != 1:
                raise ValueError("task-grouped evaluation currently requires jobs=1")
            self._validate_grouped_worker_module()
            if not resume and self._has_local_batch_evidence():
                raise ValueError(
                    "existing task batch evidence requires explicit resume"
                )
        if resume and not self.manifest_path.exists():
            raise ValueError("--resume requires an existing evaluation manifest")
        persisted_manifest = self.manifest.write_new(self.manifest_path)
        self.manifest = persisted_manifest
        store = ResultStore(self.results_path, self.manifest)
        acceptance = (
            _default_result_acceptance(self.manifest, store)
            if self.result_acceptance_factory is None
            else self.result_acceptance_factory(self.manifest, store)
        )
        retained = store.load()
        if retained and not resume:
            raise ValueError("existing results require explicit resume")
        if resume:
            for result in retained:
                acceptance.accept(
                    FinalizedResultCandidate(
                        result, NoWorker("already_accepted_repair")
                    )
                )
        pending_tasks: tuple[str, ...] = ()
        if grouping == "task":
            pending_tasks = (
                self._prepare_grouped_resume(store, acceptance)
                if resume
                else self._task_names()
            )
            if resume:
                retained = store.load()
            pending = tuple(
                case for case in self.manifest.cases if case.task_name in pending_tasks
            )
        else:
            latest = {result.case_id: result for result in retained}
            completed_ids = {
                case_id
                for case_id, result in latest.items()
                if result.status != "cancelled"
            }
            pending = tuple(
                case
                for case in self.manifest.cases
                if case.case_id not in completed_ids
            )
        summary = aggregate_results(self.manifest, retained)
        self._write_projections(summary, retained)
        if self.cancellation.is_cancelled():
            for case in pending:
                attempt_id = str(uuid4())
                result = _synthetic_result(
                    self.manifest,
                    case,
                    attempt_id=attempt_id,
                    status="cancelled",
                    worker_pid=None,
                    worker_exit_code=None,
                    error_type="EvaluationCancelled",
                    error_message=self.cancellation.reason or "cancelled before launch",
                )
                registration = register_attempt(
                    self.manifest, case, attempt_id, result.started_at
                )
                if self.attempt_registered is not None:
                    self.attempt_registered(registration, self.manifest, case)
                acceptance.accept(
                    FinalizedResultCandidate(
                        result, NoWorker("cancelled_before_launch")
                    )
                )
            retained = store.load()
            summary = aggregate_results(self.manifest, retained)
            self._write_projections(summary, retained)
            return CoordinatorOutcome(
                exit_code=1,
                completed=summary.completed,
                scheduled=len(self.manifest.cases),
                interrupted=True,
                infrastructure_error=False,
                summary=summary,
            )
        if not pending:
            return CoordinatorOutcome(
                exit_code=0,
                completed=summary.completed,
                scheduled=len(self.manifest.cases),
                interrupted=False,
                infrastructure_error=False,
                summary=summary,
            )

        interrupted = False
        infrastructure_error = False
        startup_error = False
        endpoint = str(self.manifest.runtime.get("policy_endpoint") or "")
        shared_required = bool(
            self.manifest.runtime.get("shared_policy_required", True)
        )
        phase = "policy daemon startup"
        try:
            if self.policy_daemon is not None:
                self.policy_daemon.start()
                for timing_phase, duration_ms in getattr(
                    self.policy_daemon, "startup_timings", {}
                ).items():
                    record_startup_event(
                        self.summary_path.parent,
                        self.manifest.evaluation_id,
                        timing_phase,
                        "completed",
                        duration_ms=int(duration_ms),
                    )
                endpoint = str(getattr(self.policy_daemon, "endpoint", ""))
            phase = "shared policy validation"
            if shared_required and not endpoint:
                raise RuntimeError("coordinator shared policy endpoint is unavailable")

            if grouping == "task":
                interrupted, grouped_error = self._run_grouped_workers(
                    store,
                    acceptance,
                    pending_tasks,
                    endpoint,
                )
                infrastructure_error = infrastructure_error or grouped_error
            else:
                phase = "case startup"
                runner = (
                    self._run_parallel_case_workers if self.jobs > 1
                    else self._run_case_workers
                )
                interrupted, case_error = runner(
                    store, acceptance, pending, endpoint
                )
                infrastructure_error = infrastructure_error or case_error
        except Exception as exc:
            infrastructure_error = True
            startup_error = phase in {
                "policy daemon startup",
                "shared policy validation",
            }
            if grouping == "case":
                retained = store.load()
                completed_ids = {
                    result.case_id
                    for result in retained
                    if result.status != "cancelled"
                }
                for case in pending:
                    if case.case_id in completed_ids:
                        continue
                    result = _synthetic_result(
                        self.manifest,
                        case,
                        attempt_id=str(uuid4()),
                        status="error",
                        worker_pid=None,
                        worker_exit_code=None,
                        error_type=type(exc).__name__,
                        error_message=(
                            f"coordinator {phase} failed ({type(exc).__name__})"
                        ),
                    )
                    registration = register_attempt(
                        self.manifest, case, result.attempt_id, result.started_at
                    )
                    if self.attempt_registered is not None:
                        self.attempt_registered(registration, self.manifest, case)
                    acceptance.accept(
                        FinalizedResultCandidate(result, NoWorker("startup_recovery"))
                    )
        finally:
            if self.policy_daemon is not None:
                try:
                    self.policy_daemon.stop()
                except (Exception, KeyboardInterrupt):
                    infrastructure_error = True

        retained = store.load()
        summary = aggregate_results(self.manifest, retained)
        self._write_projections(summary, retained)
        interrupted = interrupted or self.cancellation.is_cancelled()
        exit_code = 1 if interrupted or infrastructure_error else 0
        return CoordinatorOutcome(
            exit_code=exit_code,
            completed=summary.completed,
            scheduled=len(self.manifest.cases),
            interrupted=interrupted,
            infrastructure_error=infrastructure_error,
            summary=summary,
            startup_error=startup_error,
        )

    def _attempt_root(self, case: EpisodeCase, attempt_id: str) -> Path:
        return _attempt_root(self.manifest, case, attempt_id)

    def _worker_result_path(self, case: EpisodeCase, attempt_id: str) -> Path:
        root = self.results_path.parent / "worker_results"
        return root / f"{case.ordinal:05d}-{case.case_id}-{attempt_id}.json"

    def _write_projections(
        self,
        summary: EvaluationSummary,
        retained: tuple[EpisodeResult, ...],
    ) -> None:
        write_summary(self.summary_path, summary)
        _write_qualification_projection(
            self.summary_path.with_name("qualification.json"),
            self.manifest,
            retained,
        )
        from rpent.recording.media import atomic_write_json

        profile = self.manifest.schedule.get("evaluation_profile")
        evidences: tuple[TaskBatchEvidence, ...] = ()
        if isinstance(profile, str) and profile:
            evidences = tuple(
                evidence
                for task_name in self._task_names()
                for evidence in self._load_task_evidences(task_name)
            )
            official = aggregate_official_metrics(
                self.manifest, retained, evidences
            ).to_payload()
            diagnostics = aggregate_rpent_diagnostics(
                self.manifest, retained, evidences
            ).to_payload()
        else:
            unavailable = {
                "schema_version": 1,
                "available": False,
                "manifest_id": self.manifest.manifest_id,
                "reason": "official evaluation profile unavailable",
            }
            official = unavailable
            diagnostics = unavailable
        atomic_write_json(
            self.summary_path.with_name("official_metrics.json"), official
        )
        atomic_write_json(
            self.summary_path.with_name("rpent_diagnostics.json"), diagnostics
        )
        qualification_module = self.manifest.runtime.get("qualification_module")
        if qualification_module is not None:
            if (
                not isinstance(qualification_module, str)
                or re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
                    qualification_module,
                )
                is None
            ):
                raise ValueError(
                    "manifest runtime.qualification_module must be a dotted module"
                )
            projector = importlib.import_module(qualification_module)
            projector.write_qualification_bundle(
                self.manifest,
                retained,
                evidences,
                self.summary_path.parent,
            )

    def _worker_command(
        self,
        case: EpisodeCase,
        attempt_id: str,
        result_path: Path,
    ) -> list[str]:
        interpreter = str(self.manifest.runtime.get("main_interpreter", ""))
        if not interpreter:
            raise ValueError("manifest runtime.main_interpreter is required")
        return [
            interpreter,
            "-m",
            "rpent.evaluation.worker",
            "--manifest",
            str(self.manifest_path),
            "--case-id",
            case.case_id,
            "--attempt-id",
            attempt_id,
            "--result",
            str(result_path),
        ]

    def _run_case_workers(
        self,
        store: ResultStore,
        acceptance: ResultAcceptance,
        pending: tuple[EpisodeCase, ...],
        endpoint: str,
        environment: Mapping[str, str] | None = None,
    ) -> tuple[bool, bool]:
        interrupted = False
        infrastructure_error = False
        for case in pending:
            if self.cancellation.is_cancelled():
                interrupted = True
                break
            case_timeout_s = resolve_worker_timeout_s(
                self.manifest, case, override=self.worker_timeout_s
            )
            attempt_id = str(uuid4())
            registration = register_attempt(
                self.manifest,
                case,
                attempt_id,
                datetime.now(timezone.utc).isoformat(),
            )
            if self.attempt_registered is not None:
                self.attempt_registered(registration, self.manifest, case)
            attempt_root = self._attempt_root(case, attempt_id)
            result_path = self._worker_result_path(case, attempt_id)
            worker_env = self._worker_env(endpoint)
            worker_env.update(environment or {})
            worker_env["RPENT_WORKER_LOG_PATH"] = str(
                (attempt_root.resolve() / "console.log")
            )
            worker_env.update(
                {
                    "RPENT_STARTUP_LOG_PATH": str(
                        self.summary_path.parent / "startup.jsonl"
                    ),
                    "RPENT_STARTUP_CASE_ID": case.case_id,
                    "RPENT_STARTUP_ATTEMPT_ID": attempt_id,
                }
            )
            record_startup_event(
                self.summary_path.parent,
                self.manifest.evaluation_id,
                "simulator_starting",
                "worker_spawn_started",
                case_id=case.case_id,
                attempt_id=attempt_id,
            )
            try:
                process = self.worker_launcher(
                    command=self._worker_command(case, attempt_id, result_path),
                    case=case,
                    result_path=result_path,
                    env=worker_env,
                )
                record_startup_event(
                    self.summary_path.parent,
                    self.manifest.evaluation_id,
                    "simulator_starting",
                    "worker_spawned",
                    case_id=case.case_id,
                    attempt_id=attempt_id,
                    worker_pid=getattr(process, "pid", None),
                )
            except Exception as exc:
                record_startup_event(
                    self.summary_path.parent,
                    self.manifest.evaluation_id,
                    "simulator_starting",
                    "worker_spawn_failed",
                    case_id=case.case_id,
                    attempt_id=attempt_id,
                    error_type=_snake_error_type(type(exc).__name__),
                )
                infrastructure_error = True
                result = _synthetic_result(
                    self.manifest,
                    case,
                    attempt_id=attempt_id,
                    status="error",
                    worker_pid=None,
                    worker_exit_code=None,
                    error_type=type(exc).__name__,
                    error_message=f"worker launcher failed ({type(exc).__name__})",
                )
                self._accept_case_result(
                    store, acceptance, case, result, NoWorker("launch_failed")
                )
                continue
            try:
                wait_started = time.monotonic()
                return_code = _wait_for_worker(
                    process,
                    timeout=case_timeout_s,
                    cancellation=self.cancellation,
                )
                record_startup_event(
                    self.summary_path.parent,
                    self.manifest.evaluation_id,
                    "simulator_starting",
                    "worker_exited",
                    case_id=case.case_id,
                    attempt_id=attempt_id,
                    worker_pid=getattr(process, "pid", None),
                    duration_ms=round((time.monotonic() - wait_started) * 1000),
                )
                completion = ReapedWorker(attempt_id, return_code)
            except _WorkerCancelled:
                interrupted = True
                try:
                    completion = _finish_worker_reap(process, attempt_id)
                except RuntimeError:
                    infrastructure_error = True
                    break
                result, error = _try_load_valid_worker_result(
                    result_path, self.manifest, case, attempt_id
                )
                if result is None:
                    result = _synthetic_result(
                        self.manifest,
                        case,
                        attempt_id=attempt_id,
                        status="cancelled",
                        worker_pid=getattr(process, "pid", None),
                        worker_exit_code=None,
                        error_type="EvaluationCancelled",
                        error_message=self.cancellation.reason
                        or (str(error) if error is not None else "cancelled"),
                    )
            except subprocess.TimeoutExpired:
                record_startup_event(
                    self.summary_path.parent,
                    self.manifest.evaluation_id,
                    "simulator_starting",
                    "worker_timed_out",
                    case_id=case.case_id,
                    attempt_id=attempt_id,
                    worker_pid=getattr(process, "pid", None),
                    duration_ms=round((time.monotonic() - wait_started) * 1000),
                )
                try:
                    completion = _terminate_and_reap(process, attempt_id)
                except RuntimeError:
                    infrastructure_error = True
                    break
                result = _synthetic_result(
                    self.manifest,
                    case,
                    attempt_id=attempt_id,
                    status="timeout",
                    worker_pid=getattr(process, "pid", None),
                    worker_exit_code=None,
                    error_type="TimeoutError",
                    error_message=f"worker exceeded timeout {case_timeout_s:g}s",
                )
            except KeyboardInterrupt:
                interrupted = True
                try:
                    completion = _finish_worker_reap(process, attempt_id)
                except RuntimeError:
                    infrastructure_error = True
                    break
                result, error = _try_load_valid_worker_result(
                    result_path, self.manifest, case, attempt_id
                )
                if result is None:
                    result = _synthetic_result(
                        self.manifest,
                        case,
                        attempt_id=attempt_id,
                        status="cancelled",
                        worker_pid=getattr(process, "pid", None),
                        worker_exit_code=None,
                        error_type="KeyboardInterrupt",
                        error_message=(
                            str(error)
                            if error is not None
                            else "coordinator interrupted"
                        ),
                    )
            except Exception as exc:
                infrastructure_error = True
                try:
                    completion = _terminate_and_reap(process, attempt_id)
                except RuntimeError:
                    break
                result = _synthetic_result(
                    self.manifest,
                    case,
                    attempt_id=attempt_id,
                    status="error",
                    worker_pid=getattr(process, "pid", None),
                    worker_exit_code=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            else:
                try:
                    result = _load_worker_result(result_path)
                    _validate_worker_case(self.manifest, case, attempt_id, result)
                except Exception as exc:
                    infrastructure_error = True
                    result = _synthetic_result(
                        self.manifest,
                        case,
                        attempt_id=attempt_id,
                        status="error",
                        worker_pid=getattr(process, "pid", None),
                        worker_exit_code=int(return_code),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
            self._accept_case_result(store, acceptance, case, result, completion)
            if self.cancellation.is_cancelled():
                interrupted = True
            if interrupted:
                break
        return interrupted, infrastructure_error

    def _accept_case_result(self, store, acceptance, case, result, completion):
        # Acceptance, shared ledgers and their projections form one transaction.
        with self._result_lock:
            acceptance.accept(FinalizedResultCandidate(result, completion))
            retained = store.load()
            summary = aggregate_results(self.manifest, retained)
            self._write_projections(summary, retained)
            self.progress(case, result, summary)

    def _run_parallel_case_workers(self, store, acceptance, pending, endpoint):
        lanes = tuple(self.policy_daemon.lanes)
        if len(lanes) != self.jobs or any(not lane.endpoint for lane in lanes):
            raise ValueError("parallel workers require one policy lane each")
        for endpoint in {lane.endpoint for lane in lanes}:
            sharing = [lane for lane in lanes if lane.endpoint == endpoint]
            if len(sharing) == 1:
                continue
            slots = [lane.environment.get("RPENT_PI05_SESSION_SLOT") for lane in sharing]
            if (any(lane.session_capacity < len(sharing) for lane in sharing)
                    or len(set(slots)) != len(slots) or any(slot is None for slot in slots)):
                raise ValueError("parallel workers require distinct policy endpoints or isolated session slots")
        queue = Queue()
        for case in pending:
            queue.put(case)

        def run_lane(lane):
            interrupted = infrastructure_error = False
            try:
                while not self.cancellation.is_cancelled():
                    try:
                        case = queue.get_nowait()
                    except Empty:
                        break
                    stopped, error = self._run_case_workers(
                        store, acceptance, (case,), lane.endpoint, lane.environment
                    )
                    interrupted |= stopped
                    infrastructure_error |= error
                    if error:
                        # A failed reap may leave an owner alive. Never reuse its
                        # policy, and stop other dispatchers before cleanup.
                        self.cancellation.cancel("parallel worker infrastructure failure")
                    if stopped:
                        break
            except BaseException:
                self.cancellation.cancel("parallel worker lane failed")
                raise
            return interrupted, infrastructure_error

        executor = ThreadPoolExecutor(max_workers=self.jobs)
        futures = [executor.submit(run_lane, lane) for lane in lanes]
        try:
            outcomes = [future.result() for future in futures]
        except KeyboardInterrupt:
            self.cancellation.cancel("coordinator interrupted")
            outcomes = [future.result() for future in futures]
        finally:
            executor.shutdown(wait=True)
        return (
            self.cancellation.is_cancelled() or any(item[0] for item in outcomes),
            any(item[1] for item in outcomes),
        )

    def _worker_env(self, endpoint: str) -> dict[str, str]:
        env = os.environ.copy()
        for name in _ALLOWED_WORKER_ENVIRONMENT:
            env.pop(name, None)
        env.update(self.worker_environment)
        if endpoint:
            env["RPENT_SHARED_POLICY_ENDPOINT"] = endpoint
            env["RPENT_SHARED_POLICY_LOADED"] = "1"
        return env

    def _task_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(case.task_name for case in self.manifest.cases))

    def _validate_grouped_worker_module(self) -> str:
        module = self.manifest.runtime.get("grouped_worker_module")
        if (
            not isinstance(module, str)
            or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+",
                module,
            )
            is None
        ):
            raise ValueError(
                "manifest runtime.grouped_worker_module must be a dotted module"
            )
        return module

    def _has_local_batch_evidence(self) -> bool:
        evaluation_root = Path(str(self.manifest.runtime["evaluation_root"]))
        outcomes = evaluation_root / "coordinator_batch_outcomes.jsonl"
        if outcomes.exists() and outcomes.stat().st_size:
            return True
        root = evaluation_root / "task_batches"
        if not root.exists():
            return False
        return any(
            (attempt / name).exists()
            for task in root.iterdir()
            if task.is_dir()
            for attempt in task.iterdir()
            if attempt.is_dir()
            for name in ("batch.jsonl", "results.jsonl")
        )

    def _process_outcomes_path(self) -> Path:
        return (
            Path(str(self.manifest.runtime["evaluation_root"]))
            / "coordinator_batch_outcomes.jsonl"
        )

    def _task_batch_root(self, task_name: str, batch_attempt_id: str) -> Path:
        root = Path(str(self.manifest.runtime["evaluation_root"]))
        return root / "task_batches" / task_name / batch_attempt_id

    def _load_task_evidences(self, task_name: str) -> tuple[TaskBatchEvidence, ...]:
        task_root = self._task_batch_root(task_name, "unused").parent
        outcomes = load_task_batch_process_outcomes(
            self._process_outcomes_path(), self.manifest
        )
        if not task_root.exists():
            return ()
        canonical_task_root = (
            Path(str(self.manifest.runtime["evaluation_root"])).resolve()
            / "task_batches"
            / task_name
        )
        if task_root.resolve() != canonical_task_root:
            raise ValueError("task batch task directory escapes evaluation root")
        evidences = []
        for attempt_root in sorted(
            path for path in task_root.iterdir() if path.is_dir()
        ):
            validate_task_batch_attempt_id(attempt_root.name)
            if attempt_root.resolve() != canonical_task_root / attempt_root.name:
                raise ValueError("task batch attempt directory escapes evaluation root")
            ledger_path = attempt_root / "batch.jsonl"
            if not ledger_path.exists():
                continue
            attempts = load_task_batch_attempts(ledger_path, self.manifest)
            results = ResultStore(attempt_root / "results.jsonl", self.manifest).load()
            matching_outcomes = tuple(
                outcome
                for outcome in outcomes
                if outcome.task_name == task_name
                and outcome.batch_attempt_id == attempt_root.name
            )
            outcome = matching_outcomes[0] if matching_outcomes else None
            case_by_id = {case.case_id: case for case in self.manifest.cases}
            for result in results:
                validate_task_batch_result(
                    self.manifest,
                    case_by_id[result.case_id],
                    attempt_root.name,
                    result,
                )
            evidences.append(
                evaluate_task_batch_evidence(
                    self.manifest,
                    task_name,
                    attempt_root.name,
                    attempts,
                    results,
                    outcome,
                )
            )
        return tuple(evidences)

    def _import_batch_evidence(
        self,
        store: ResultStore,
        evidences: tuple[TaskBatchEvidence, ...],
        acceptance: ResultAcceptance,
        completion: ReapedWorker | None = None,
    ) -> tuple[EpisodeResult, ...]:
        retained = store.load()
        by_attempt = {result.attempt_id: result for result in retained}
        for evidence in evidences:
            for result in evidence.results:
                existing = by_attempt.get(result.attempt_id)
                if existing is not None and existing != result:
                    raise ValueError("central result attempt identity drift")
                by_attempt[result.attempt_id] = result
        imported = []
        existing_attempts = {result.attempt_id for result in retained}
        for evidence in evidences:
            for result in evidence.results:
                if result.attempt_id in existing_attempts:
                    continue
                proof = (
                    completion
                    if completion is not None
                    and completion.scope_id == evidence.batch_attempt_id
                    else NoWorker("startup_recovery")
                )
                acceptance.accept(FinalizedResultCandidate(result, proof))
                existing_attempts.add(result.attempt_id)
                imported.append(result)
        return tuple(imported)

    def _prepare_grouped_resume(
        self, store: ResultStore, acceptance: ResultAcceptance
    ) -> tuple[str, ...]:
        discovered = tuple(
            (task_name, self._load_task_evidences(task_name))
            for task_name in self._task_names()
        )
        selected = tuple(
            (task_name, evidences, select_complete_task_batch(evidences))
            for task_name, evidences in discovered
        )
        self._import_batch_evidence(
            store,
            tuple(
                evidence
                for _task_name, evidences, _complete in selected
                for evidence in evidences
            ),
            acceptance,
        )
        pending = []
        for task_name, _evidences, complete in selected:
            if complete is None:
                pending.append(task_name)
        return tuple(pending)

    def _grouped_worker_command(
        self,
        task_name: str,
        batch_attempt_id: str,
        results_path: Path,
        ledger_path: Path,
    ) -> list[str]:
        interpreter = str(self.manifest.runtime.get("main_interpreter", ""))
        module = self._validate_grouped_worker_module()
        if not interpreter:
            raise ValueError("manifest runtime.main_interpreter is required")
        return [
            interpreter,
            "-m",
            module,
            "--manifest",
            str(self.manifest_path),
            "--task",
            task_name,
            "--batch-attempt-id",
            batch_attempt_id,
            "--results",
            str(results_path),
            "--batch-ledger",
            str(ledger_path),
        ]

    def _run_grouped_workers(
        self,
        store: ResultStore,
        acceptance: ResultAcceptance,
        pending_tasks: tuple[str, ...],
        endpoint: str,
    ) -> tuple[bool, bool]:
        interrupted = False
        infrastructure_error = False
        first_case = {
            task_name: next(
                case for case in self.manifest.cases if case.task_name == task_name
            )
            for task_name in pending_tasks
        }
        for task_name in pending_tasks:
            if self.cancellation.is_cancelled():
                interrupted = True
                break
            task_cases = tuple(
                item for item in self.manifest.cases if item.task_name == task_name
            )
            task_timeout_s = resolve_group_worker_timeout_s(
                self.manifest, task_cases, override=self.worker_timeout_s
            )
            batch_attempt_id = str(uuid4())
            for case in task_cases:
                attempt_id = (
                    f"{batch_attempt_id}-{case.episode_index:03d}-{case.case_id[:12]}"
                )
                registration = register_attempt(
                    self.manifest,
                    case,
                    attempt_id,
                    datetime.now(timezone.utc).isoformat(),
                )
                if self.attempt_registered is not None:
                    self.attempt_registered(registration, self.manifest, case)
            attempt_root = self._task_batch_root(task_name, batch_attempt_id)
            attempt_root.mkdir(parents=True, exist_ok=True)
            results_path = attempt_root / "results.jsonl"
            ledger_path = attempt_root / "batch.jsonl"
            command = self._grouped_worker_command(
                task_name,
                batch_attempt_id,
                results_path,
                ledger_path,
            )
            process = None
            completion = None
            try:
                worker_env = self._worker_env(endpoint)
                worker_env["RPENT_WORKER_LOG_PATH"] = str(
                    attempt_root.resolve() / "console.log"
                )
                process = self.worker_launcher(
                    command=command,
                    case=first_case[task_name],
                    result_path=results_path,
                    env=worker_env,
                )
                return_code = _wait_for_worker(
                    process,
                    timeout=task_timeout_s,
                    cancellation=self.cancellation,
                )
                completion = ReapedWorker(batch_attempt_id, return_code)
            except _WorkerCancelled:
                interrupted = True
                if process is not None:
                    completion = _finish_worker_reap(process, batch_attempt_id)
            except subprocess.TimeoutExpired:
                infrastructure_error = True
                if process is not None:
                    completion = _terminate_and_reap(process, batch_attempt_id)
            except KeyboardInterrupt:
                interrupted = True
                if process is not None:
                    completion = _finish_worker_reap(process, batch_attempt_id)
            except Exception:
                infrastructure_error = True
                if process is not None:
                    try:
                        completion = _terminate_and_reap(process, batch_attempt_id)
                    except RuntimeError:
                        completion = None

            evidence = None
            try:
                if process is None or completion is None:
                    raise RuntimeError("grouped worker was not reaped")
                return_code = completion.exit_status
                append_task_batch_process_outcome(
                    self._process_outcomes_path(),
                    self.manifest,
                    TaskBatchProcessOutcome(
                        manifest_id=self.manifest.manifest_id,
                        task_name=task_name,
                        batch_attempt_id=batch_attempt_id,
                        worker_exit_code=return_code,
                    ),
                )
                evidences = self._load_task_evidences(task_name)
                matches = tuple(
                    item
                    for item in evidences
                    if item.batch_attempt_id == batch_attempt_id
                )
                if len(matches) == 1:
                    evidence = matches[0]
                    imported = self._import_batch_evidence(
                        store, matches, acceptance, completion
                    )
                    retained = store.load()
                    summary = aggregate_results(self.manifest, retained)
                    self._write_projections(summary, retained)
                    case_by_id = {case.case_id: case for case in self.manifest.cases}
                    for result in imported:
                        self.progress(case_by_id[result.case_id], result, summary)
            except Exception:
                infrastructure_error = True
            if (
                evidence is None
                or not evidence.complete
                or completion is None
                or completion.exit_status != 0
            ):
                infrastructure_error = True
            if interrupted:
                break
        return interrupted, infrastructure_error


class _OwnedWorkerProcess:
    """Own the worker's process group, including isolated simulator children."""

    terminate_grace_s = 15.0

    def __init__(self, *args, **kwargs):
        self._process = subprocess.Popen(*args, **kwargs)
        self._group_reaped = False

    @property
    def pid(self):
        return self._process.pid

    def poll(self):
        return self._process.poll()

    def _signal_group(self, selected_signal):
        if getattr(self, "_group_reaped", False):
            return
        try:
            os.killpg(self.pid, selected_signal)
        except ProcessLookupError:
            pass

    def terminate(self):
        if os.name == "posix":
            # The worker already finalizes recordings and releases native API
            # sessions on KeyboardInterrupt. Keep its simulator alive during
            # that cleanup; wait()/kill() then remove the remaining group.
            self._process.send_signal(signal.SIGINT)
        else:
            self._process.terminate()

    def kill(self):
        if os.name == "posix":
            self._signal_group(signal.SIGKILL)
        else:
            self._process.kill()

    def wait(self, timeout=None):
        code = self._process.wait(timeout=timeout)
        if os.name == "posix":
            # A worker crash or timeout must not strand its GPU simulator.
            # Only this freshly created group is owned; shared policies live
            # in the coordinator's group and are never signalled here.
            self._signal_group(signal.SIGKILL)
            self._group_reaped = True
        return code


def _default_worker_launcher(
    *,
    command: list[str],
    case: EpisodeCase,
    result_path: Path,
    env: Mapping[str, str],
):
    del case, result_path
    child_env = dict(env)
    log_path = child_env.pop("RPENT_WORKER_LOG_PATH", None)
    if not isinstance(log_path, str) or not log_path:
        raise ValueError("worker Console path is required")
    with Path(log_path).open("ab") as stream:
        return _OwnedWorkerProcess(
            command,
            env=child_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )


class _WorkerCancelled(Exception):
    pass


def _wait_for_worker(
    process: Any,
    *,
    timeout: float,
    cancellation: CancellationToken,
) -> int:
    deadline = time.monotonic() + timeout
    while True:
        if cancellation.is_cancelled():
            raise _WorkerCancelled
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("worker", timeout)
        try:
            return int(process.wait(timeout=min(0.25, remaining)))
        except subprocess.TimeoutExpired:
            continue


def _write_qualification_projection(
    path: Path,
    manifest: EvaluationManifest,
    retained: tuple[EpisodeResult, ...],
) -> None:
    from rpent.recording.contracts import ArtifactHealth, QualificationOutcome
    from rpent.recording.media import atomic_write_json

    latest: dict[str, EpisodeResult] = {}
    for result in retained:
        latest[result.case_id] = result
    require_complete = bool(
        manifest.qualification.get("require_complete_artifacts", False)
    )
    outcomes = []
    cases = {case.case_id: case for case in manifest.cases}
    for result in latest.values():
        try:
            case = cases[result.case_id]
            artifact_root = _validated_artifact_root(
                manifest, case, result.attempt_id, result.artifact_dir
            )
            payload = json.loads(
                (artifact_root / "qualification.json").read_text(encoding="utf-8")
            )
            outcome = QualificationOutcome.from_payload(payload)
            if outcome.episode_result != result:
                raise ValueError("qualification result mismatch")
        except Exception:
            outcome = QualificationOutcome.evaluate(
                result,
                ArtifactHealth(
                    "unavailable", ("qualification projection unavailable",)
                ),
                require_complete,
            )
        outcomes.append(outcome.to_payload())
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "evaluation_id": manifest.evaluation_id,
            "manifest_id": manifest.manifest_id,
            "require_complete_artifacts": require_complete,
            "accepted": len(latest) == len(manifest.cases)
            and all(item["accepted"] for item in outcomes),
            "cases": outcomes,
        },
    )


def _load_worker_result(path: Path) -> EpisodeResult:
    if not path.exists():
        raise FileNotFoundError(f"worker produced no result at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"worker result is corrupt: {exc}") from exc
    return EpisodeResult.from_payload(payload)


def _validate_worker_case(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
    result: EpisodeResult,
) -> None:
    if (
        result.evaluation_id != manifest.evaluation_id
        or result.manifest_id != manifest.manifest_id
        or result.case_id != case.case_id
        or result.attempt_id != attempt_id
    ):
        raise ValueError("worker result evaluation/manifest/case/attempt mismatch")
    _validated_artifact_root(manifest, case, attempt_id, result.artifact_dir)


def _try_load_valid_worker_result(
    path: Path,
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
) -> tuple[EpisodeResult | None, Exception | None]:
    try:
        result = _load_worker_result(path)
        _validate_worker_case(manifest, case, attempt_id, result)
    except Exception as exc:
        return None, exc
    return result, None


def _attempt_root(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
) -> Path:
    evaluation_root = Path(str(manifest.runtime.get("evaluation_root", ".")))
    return evaluation_root / "cases" / case.case_id / "attempts" / attempt_id


def _validated_artifact_root(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    attempt_id: str,
    artifact_dir: str,
) -> Path:
    expected = _attempt_root(manifest, case, attempt_id).resolve()
    actual = Path(artifact_dir).resolve()
    if actual != expected:
        raise ValueError(
            f"worker result artifact_dir mismatch: expected {expected}, actual {actual}"
        )
    return actual


def _synthetic_result(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    *,
    attempt_id: str,
    status: str,
    worker_pid: int | None,
    worker_exit_code: int | None,
    error_type: str,
    error_message: str,
) -> EpisodeResult:
    now = datetime.now(timezone.utc).isoformat()
    evaluation_root = Path(str(manifest.runtime.get("evaluation_root", ".")))
    artifact_dir = evaluation_root / "cases" / case.case_id / "attempts" / attempt_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
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
        worker_pid=worker_pid,
        worker_exit_code=worker_exit_code,
        started_at=now,
        finished_at=now,
        artifact_dir=str(artifact_dir),
        simulator_success=False,
        contract_success=False,
        terminated=False,
        truncated=False,
        step_count=0,
        action_count=0,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
        error_type=error_type,
        error_message=error_message,
    )


def _finish_worker_reap(process: Any, scope_id: str) -> ReapedWorker:
    """Return proof only after a final successful wait, otherwise escalate."""
    try:
        status = int(process.wait(timeout=0.25))
    except (Exception, KeyboardInterrupt):
        return _terminate_and_reap(process, scope_id)
    return ReapedWorker(scope_id, status)


def _terminate_and_reap(process: Any, scope_id: str) -> ReapedWorker:
    """Own terminate/wait/kill/wait and fail closed when reap is unprovable."""
    timeout = getattr(process, "terminate_grace_s", 5.0)
    try:
        process.terminate()
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
        pass
    try:
        status = int(process.wait(timeout=timeout))
        return ReapedWorker(scope_id, status)
    except (Exception, KeyboardInterrupt):
        pass
    try:
        process.kill()
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt):
        pass
    try:
        status = int(process.wait(timeout=timeout))
    except (Exception, KeyboardInterrupt) as exc:
        raise RuntimeError("worker lifecycle could not be reaped") from exc
    return ReapedWorker(scope_id, status)


def _default_result_acceptance(
    manifest: EvaluationManifest, store: ResultStore
) -> ResultAcceptance:
    """Build the same snapshot-producing acceptance boundary for CLI runs."""
    from rpent.ui.case_runtime import CaseRuntimeStore
    from rpent.ui.repository import EvaluationRepository

    evaluation_root = Path(str(manifest.runtime["evaluation_root"])).resolve()
    manifest.write_new(evaluation_root / "evaluation_manifest.json")
    repository = EvaluationRepository(((evaluation_root.parent).resolve(),))
    try:
        repository.evaluation_metadata(manifest.evaluation_id)
    except KeyError:
        repository.register_evaluation(evaluation_root, manifest)
    runtime_store = CaseRuntimeStore()
    return ResultAcceptance(
        manifest=manifest,
        store=store,
        runtime_store=runtime_store,
        assembler=lambda result: repository.reconstruct_case_detail(
            result.evaluation_id,
            result.case_id,
            attempt_id=result.attempt_id,
        ),
        operation_reconstructor=lambda result: repository.reconstruct_operations(
            result.evaluation_id,
            result.case_id,
            attempt_id=result.attempt_id,
        ),
        accepted_observer=repository.note_accepted_result,
    )


def _print_progress(
    case: EpisodeCase,
    result: EpisodeResult,
    summary: EvaluationSummary,
) -> None:
    print(
        f"{summary.completed}/{summary.scheduled} "
        f"task={case.task_name} seed={case.seed} status={result.status} "
        f"success_rate={summary.success_rate:.3f}",
        flush=True,
    )


__all__ = ["CoordinatorOutcome", "EvaluationCoordinator"]
