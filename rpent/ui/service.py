"""Background ownership for evaluation coordinator runs."""

from __future__ import annotations

import copy
import json
import os
import stat
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from rpent.evaluation.control import CancellationToken
from rpent.evaluation.coordinator import CoordinatorOutcome, EvaluationCoordinator
from rpent.evaluation.front_controller import build_evaluation_manifest
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import OfficialMetrics
from rpent.evaluation.preparation_progress import load_preparation
from rpent.evaluation.startup import (
    EvaluationStartup,
    initialize_evaluation_startup,
    persist_evaluation_manifest,
)
from rpent.utils.logging import get_logger

logger = get_logger("ui.service")


def _snake_error_type(name: str) -> str:
    return "".join(
        ("_" + char.lower()) if char.isupper() else char for char in name
    ).lstrip("_")


class ConflictError(RuntimeError):
    """The requested lifecycle transition conflicts with current state."""


class EvaluationNotFoundError(LookupError):
    """No evaluation with the requested identity is known to this service."""


class ManifestConfirmationError(ValueError):
    """The final authoritative manifest does not match a confirmed Preview."""


@dataclass(frozen=True, slots=True)
class StartEvaluation:
    """Inputs already understood by the existing evaluation front controller."""

    args: Any
    env_spec: Any
    argv: tuple[str, ...]
    resolved_tasks: Iterable[Any] | None = None
    git_identity: Callable[[], tuple[str, bool]] | None = None
    policy_daemon_factory: Callable[..., Any] | None = None
    worker_environment: Mapping[str, str] = field(default_factory=dict)
    launch: Mapping[str, Any] = field(default_factory=dict)
    prerequisites: tuple[Any, ...] = ()
    manifest_validator: Callable[[EvaluationManifest], None] | None = None

    def __post_init__(self) -> None:
        allowed = {"RPENT_NOTIFICATION_TOKEN", "RPENT_DASHBOARD_DESCRIPTOR"}
        if not isinstance(self.worker_environment, Mapping) or any(
            key not in allowed
            or not isinstance(value, str)
            or not value
            or "\r" in value
            or "\n" in value
            for key, value in self.worker_environment.items()
        ):
            raise ValueError("invalid worker environment")
        object.__setattr__(
            self,
            "worker_environment",
            MappingProxyType(dict(self.worker_environment)),
        )
        object.__setattr__(self, "launch", MappingProxyType(dict(self.launch)))
        if self.manifest_validator is not None and not callable(
            self.manifest_validator
        ):
            raise ValueError("manifest_validator must be callable")


@dataclass(frozen=True, slots=True)
class PreparedEvaluation:
    """A manifest plus the snapshotted runtime inputs needed to execute it."""

    request: StartEvaluation
    manifest: EvaluationManifest
    root: Path


class EvaluationRun:
    """Observable state for one background lifecycle operation."""

    def __init__(
        self,
        evaluation_id: str,
        operation: Literal["start", "resume", "rerun"],
        cancellation: CancellationToken,
    ) -> None:
        self.evaluation_id = evaluation_id
        self.operation = operation
        self._cancellation = cancellation
        self._future: Future[CoordinatorOutcome] | None = None
        self._state: Literal[
            "queued", "running", "completed", "cancelled", "failed"
        ] = "queued"
        self._lock = threading.Lock()

    @property
    def future(self) -> Future[CoordinatorOutcome]:
        with self._lock:
            if self._future is None:
                raise RuntimeError("evaluation run has not been submitted")
            return self._future

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def cancellation_reason(self) -> str | None:
        return self._cancellation.reason

    @property
    def outcome(self) -> CoordinatorOutcome | None:
        future = self._bound_future()
        if future is None or not future.done() or future.cancelled():
            return None
        try:
            return future.result()
        except BaseException:
            return None

    @property
    def exception(self) -> BaseException | None:
        future = self._bound_future()
        if future is None or not future.done():
            return None
        try:
            return future.exception()
        except CancelledError as exc:
            return exc

    def _bound_future(self) -> Future[CoordinatorOutcome] | None:
        with self._lock:
            return self._future

    def _is_done(self) -> bool:
        future = self._bound_future()
        return future is not None and future.done()

    def _bind(self, future: Future[CoordinatorOutcome]) -> None:
        with self._lock:
            self._future = future

    def _set_state(
        self,
        state: Literal["queued", "running", "completed", "cancelled", "failed"],
    ) -> None:
        with self._lock:
            self._state = state


class EvaluationService:
    """Own at most one active coordinator run for each Evaluation."""

    def __init__(
        self,
        *,
        manifest_builder: Callable[..., tuple[EvaluationManifest, Any]] = (
            build_evaluation_manifest
        ),
        coordinator_factory: Callable[..., Any] = EvaluationCoordinator,
        executor: Executor | None = None,
        rerun_builder: Callable[[PreparedEvaluation, str], PreparedEvaluation]
        | None = None,
        context_loader: Callable[[str], PreparedEvaluation] | None = None,
        inference_pool: Any | None = None,
        robocasa_policy_pool: Any | None = None,
        robocasa_xiaomi_policy_pool: Any | None = None,
        on_persisted: Callable[[PreparedEvaluation], None] | None = None,
        on_finished: Callable[[PreparedEvaluation], None] | None = None,
        attempt_registered: Callable[..., None] | None = None,
        result_acceptance_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._manifest_builder = manifest_builder
        self._coordinator_factory = coordinator_factory
        self._executor = executor or ThreadPoolExecutor(thread_name_prefix="rpent-eval")
        self._owns_executor = executor is None
        self._rerun_builder = rerun_builder
        self._context_loader = context_loader
        self._inference_pool = inference_pool
        self._owns_inference_pool = inference_pool is None
        self._robocasa_policy_pool = robocasa_policy_pool
        self._owns_robocasa_policy_pool = robocasa_policy_pool is None
        self._robocasa_xiaomi_policy_pool = robocasa_xiaomi_policy_pool
        self._owns_robocasa_xiaomi_policy_pool = (
            robocasa_xiaomi_policy_pool is None
        )
        if on_persisted is not None and not callable(on_persisted):
            raise TypeError("on_persisted must be callable")
        self._on_persisted = on_persisted
        if on_finished is not None and not callable(on_finished):
            raise TypeError("on_finished must be callable")
        self._on_finished = on_finished
        if attempt_registered is not None and not callable(attempt_registered):
            raise TypeError("attempt_registered must be callable")
        self._attempt_registered = attempt_registered
        if result_acceptance_factory is not None and not callable(
            result_acceptance_factory
        ):
            raise TypeError("result_acceptance_factory must be callable")
        self._result_acceptance_factory = result_acceptance_factory
        self._lock = threading.RLock()
        self._active: dict[str, EvaluationRun] = {}
        self._contexts: dict[str, PreparedEvaluation] = {}
        self._root_owners: dict[Path, str] = {}
        self._runs: dict[str, EvaluationRun] = {}
        self._closed = False

    def start(self, request: StartEvaluation) -> EvaluationRun:
        snapshot = _snapshot_request(request)
        if (
            snapshot.manifest_validator is not None
            and getattr(snapshot.env_spec, "name", None) == "libero"
            and snapshot.launch
        ):
            raise ManifestConfirmationError(
                "confirmed workloads cannot acquire mutable inference endpoints"
            )
        inference_lease = None
        if snapshot.manifest_validator is None or getattr(
            snapshot.env_spec, "name", None
        ) == "robocasa":
            inference_lease, snapshot = self._acquire_inference(snapshot)
        try:
            manifest, run_config = self._manifest_builder(
                snapshot.args,
                snapshot.env_spec,
                snapshot.argv,
                resolved_tasks=snapshot.resolved_tasks,
                git_identity=snapshot.git_identity,
                launch=snapshot.launch,
            )
            if snapshot.manifest_validator is not None:
                snapshot.manifest_validator(manifest)
        except BaseException:
            if inference_lease is not None:
                inference_lease.release()
            raise
        root = Path(run_config.output_dir).resolve()
        startup: EvaluationStartup | None = None
        with self._lock:
            if self._closed:
                raise ConflictError("evaluation service is closed")
            owner = self._root_owners.get(root)
            if owner is not None:
                raise ConflictError(f"evaluation root is already owned by {owner}")
            try:
                startup = initialize_evaluation_startup(
                    manifest,
                    root,
                    on_persisted=lambda persisted, selected_root: (
                        self._publish_persisted(
                            PreparedEvaluation(snapshot, persisted, selected_root)
                        )
                    ),
                )
                context = PreparedEvaluation(
                    snapshot,
                    startup.manifest,
                    startup.root,
                )
                if inference_lease is None:
                    startup.prepare_dependencies(None, lambda: False)
                run = self._launch(
                    context,
                    operation="start",
                    resume=False,
                    reject_owned_root=True,
                    inference_lease=inference_lease,
                    startup=startup,
                )
            except BaseException:
                if inference_lease is not None:
                    inference_lease.release()
                raise
        return run

    def _acquire_inference(self, request: StartEvaluation):
        simulator = getattr(request.env_spec, "name", None)
        if not request.launch:
            return None, request
        if simulator == "robocasa":
            return self._acquire_robocasa_policy(request)
        if simulator != "libero":
            return None, request
        effective = request.launch.get("effective")
        mode = request.launch.get("execution_mode")
        if not isinstance(effective, Mapping) or not isinstance(mode, str):
            raise ValueError("LIBERO launch identity is incomplete")
        if self._inference_pool is None:
            from robots.libero.inference_pool import LiberoInferencePool

            self._inference_pool = LiberoInferencePool()
        lease = self._inference_pool.acquire(
            effective,
            mode=mode,
            reuse=bool(effective.get("reuse_inference", True)),
        )
        request.args.vla_endpoint = lease.endpoints.vla
        argv = _set_option(request.argv, "--vla-endpoint", lease.endpoints.vla)
        if mode == "hybrid" and lease.endpoints.sam3 is not None:
            request.args.sam3_endpoint = lease.endpoints.sam3
            argv = _set_option(argv, "--sam3-endpoint", lease.endpoints.sam3)
        return lease, replace(request, argv=argv)

    def _acquire_robocasa_policy(self, request: StartEvaluation):
        effective = request.launch.get("effective")
        if not isinstance(effective, Mapping):
            raise ValueError("RoboCasa launch identity is incomplete")
        if effective.get("vla") == "pi05":
            # The coordinator shares one Pi0.5 process across this Evaluation;
            # the SmolVLA idle pool must never serve a different VLA identity.
            return None, request
        if not bool(effective.get("reuse_policy_service", False)):
            return None, request
        if getattr(request.args, "policy_endpoint", None):
            return None, request
        if effective.get("planning_profile") == (
            "robocasa-xiaomi-mibot-gated-v1"
        ):
            if self._robocasa_xiaomi_policy_pool is None:
                from robots.robocasa.xiaomi_policy_pool import (
                    RoboCasaXiaomiPolicyPool,
                )

                self._robocasa_xiaomi_policy_pool = RoboCasaXiaomiPolicyPool()
            pool = self._robocasa_xiaomi_policy_pool
        else:
            if self._robocasa_policy_pool is None:
                from robots.robocasa.policy_pool import RoboCasaPolicyPool

                self._robocasa_policy_pool = RoboCasaPolicyPool()
            pool = self._robocasa_policy_pool
        lease = pool.acquire(effective, reuse=True)
        request.args.policy_endpoint = lease.endpoint
        argv = _set_option(
            request.argv,
            "--policy-endpoint",
            lease.endpoint,
        )
        return lease, replace(
            request,
            argv=argv,
            policy_daemon_factory=_preloaded_policy_daemon,
        )

    def cancel(self, evaluation_id: str, reason: str) -> EvaluationRun:
        with self._lock:
            run = self._active.get(evaluation_id)
            if run is None:
                if evaluation_id in self._runs:
                    raise ConflictError(f"evaluation {evaluation_id} is not active")
                raise EvaluationNotFoundError(evaluation_id)
            if run._is_done():
                raise ConflictError(f"evaluation {evaluation_id} is not active")
            run._cancellation.cancel(reason)
            return run

    def get_run(self, evaluation_id: str) -> EvaluationRun:
        with self._lock:
            try:
                return self._runs[evaluation_id]
            except KeyError:
                raise EvaluationNotFoundError(evaluation_id) from None

    def available_commands(
        self, evaluation_id: str, case_id: str | None = None
    ) -> tuple[str, ...]:
        """Project commands the service can currently revalidate and execute."""
        loaded_context: PreparedEvaluation | None = None
        if self._context_loader is not None:
            with self._lock:
                needs_load = evaluation_id not in self._contexts
            if needs_load:
                try:
                    loaded_context = self._get_context(evaluation_id)
                except (EvaluationNotFoundError, OSError, ValueError):
                    return ()
        with self._lock:
            run = self._active.get(evaluation_id)
            active = run is not None and not run._is_done()
            context = self._contexts.get(evaluation_id) or loaded_context
            known = context is not None or evaluation_id in self._runs
            reloadable = self._context_loader is not None
            if case_id is None:
                if active:
                    return ("cancel",)
                if context is not None:
                    if _official_run_is_terminal(context):
                        return ()
                    try:
                        preparation = load_preparation(
                            context.root / "preparation.json",
                            evaluation_id=context.manifest.evaluation_id,
                            manifest_id=context.manifest.manifest_id,
                        )
                    except (OSError, ValueError):
                        return ()
                    if preparation is not None and (
                        preparation.status == "failed"
                        or (
                            preparation.status == "cancelled"
                            and preparation.current_phase
                            in {"evaluation_created", "sam3_loading", "vla_loading"}
                        )
                    ):
                        return ("retry",)
                if known or reloadable:
                    latest = self._runs.get(evaluation_id)
                    if latest is None or latest.state in {"cancelled", "failed"}:
                        return ("resume",)
                return ()
            if active or self._rerun_builder is None or not (known or reloadable):
                return ()
            if context is not None and not _allows_selective_case_rerun(
                context.manifest
            ):
                return ()
            return ("rerun",)

    def resume(self, evaluation_id: str) -> EvaluationRun:
        context = self._get_context(evaluation_id, publish_existing=True)
        startup = initialize_evaluation_startup(context.manifest, context.root)
        context = PreparedEvaluation(context.request, startup.manifest, startup.root)
        return self._launch(
            context,
            operation="resume",
            resume=True,
            startup=startup,
        )

    def retry(self, evaluation_id: str, request: StartEvaluation) -> EvaluationRun:
        """Retry failed preparation after revalidating the retained source."""
        source = self._get_context(evaluation_id)
        with self._lock:
            active = self._active.get(evaluation_id)
            if active is not None and not active._is_done():
                raise ConflictError(f"evaluation {evaluation_id} is already active")
            if _official_run_is_terminal(source):
                raise ConflictError("completed official evaluation cannot be retried")
            try:
                preparation = load_preparation(
                    source.root / "preparation.json",
                    evaluation_id=source.manifest.evaluation_id,
                    manifest_id=source.manifest.manifest_id,
                )
            except (OSError, ValueError) as exc:
                raise ConflictError("evaluation preparation is not retryable") from exc
            if preparation is None or preparation.status not in {
                "failed",
                "cancelled",
            }:
                raise ConflictError("evaluation preparation is not retryable")
            return self.start(request)

    def rerun(self, evaluation_id: str, case_id: str) -> EvaluationRun:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        source = self._get_context(evaluation_id)
        if not _allows_selective_case_rerun(source.manifest):
            raise ConflictError("selective Case rerun is disabled for this profile")
        if case_id not in {case.case_id for case in source.manifest.cases}:
            raise EvaluationNotFoundError(case_id)
        if self._rerun_builder is None:
            raise ConflictError("rerun capability is unavailable")
        prepared = self._rerun_builder(source, case_id)
        if not isinstance(prepared, PreparedEvaluation):
            raise TypeError("rerun_builder must return PreparedEvaluation")
        if prepared.manifest.evaluation_id == source.manifest.evaluation_id:
            raise ValueError("rerun must create a new evaluation_id")
        if prepared.root.resolve() == source.root:
            raise ValueError("rerun must use a new evaluation root")
        context = self._persist(
            replace(prepared, request=_snapshot_request(prepared.request)),
            require_existing=False,
        )
        if context.manifest.evaluation_id == source.manifest.evaluation_id:
            raise ValueError("rerun must create a new evaluation_id")
        if context.root == source.root:
            raise ValueError("rerun must use a new evaluation root")
        startup = initialize_evaluation_startup(context.manifest, context.root)
        context = PreparedEvaluation(context.request, startup.manifest, startup.root)
        return self._launch(
            context,
            operation="rerun",
            resume=False,
            startup=startup,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(self._active.values())
            for run in active:
                if not run._is_done():
                    run._cancellation.cancel("service_closed")
        if self._owns_executor:
            self._executor.shutdown(wait=True)
        if self._owns_inference_pool and self._inference_pool is not None:
            self._inference_pool.close()
        if (
            self._owns_robocasa_policy_pool
            and self._robocasa_policy_pool is not None
        ):
            self._robocasa_policy_pool.close()
        if (
            self._owns_robocasa_xiaomi_policy_pool
            and self._robocasa_xiaomi_policy_pool is not None
        ):
            self._robocasa_xiaomi_policy_pool.close()

    def _get_context(
        self,
        evaluation_id: str,
        *,
        publish_existing: bool = False,
    ) -> PreparedEvaluation:
        with self._lock:
            context = self._contexts.get(evaluation_id)
        if context is not None:
            if publish_existing:
                self._publish_persisted(context)
            return context
        if self._context_loader is None:
            raise EvaluationNotFoundError(evaluation_id)
        loaded = self._context_loader(evaluation_id)
        if not isinstance(loaded, PreparedEvaluation):
            raise TypeError("context_loader must return PreparedEvaluation")
        if loaded.manifest.evaluation_id != evaluation_id:
            raise ValueError("loaded context evaluation_id mismatch")
        context = self._persist(
            replace(loaded, request=_snapshot_request(loaded.request)),
            require_existing=True,
        )
        return context

    def _persist(
        self,
        context: PreparedEvaluation,
        *,
        require_existing: bool,
    ) -> PreparedEvaluation:
        root = context.root.resolve()
        manifest_path = root / "evaluation_manifest.json"
        if require_existing and not manifest_path.is_file():
            raise EvaluationNotFoundError(context.manifest.evaluation_id)
        persisted, root = persist_evaluation_manifest(context.manifest, root)
        retained = PreparedEvaluation(context.request, persisted, root)
        self._publish_persisted(retained)
        return retained

    def _publish_persisted(self, context: PreparedEvaluation) -> None:
        if self._on_persisted is not None:
            self._on_persisted(context)

    def _launch(
        self,
        context: PreparedEvaluation,
        *,
        operation: Literal["start", "resume", "rerun"],
        resume: bool,
        startup: EvaluationStartup,
        reject_owned_root: bool = False,
        inference_lease: Any | None = None,
    ) -> EvaluationRun:
        evaluation_id = context.manifest.evaluation_id
        token = CancellationToken()
        run = EvaluationRun(evaluation_id, operation, token)
        previous_context: PreparedEvaluation | None
        previous_owner: str | None
        previous_run: EvaluationRun | None
        with self._lock:
            if self._closed:
                raise ConflictError("evaluation service is closed")
            if operation == "resume" and _official_run_is_terminal(context):
                raise ConflictError("completed official evaluation cannot be resumed")
            if evaluation_id in self._active:
                raise ConflictError(f"evaluation {evaluation_id} is already active")
            owner = self._root_owners.get(context.root)
            if owner is not None and (owner != evaluation_id or reject_owned_root):
                raise ConflictError(f"evaluation root is already owned by {owner}")
            previous_context = self._contexts.get(evaluation_id)
            previous_owner = owner
            previous_run = self._runs.get(evaluation_id)
            self._active[evaluation_id] = run
            self._contexts[evaluation_id] = context
            self._root_owners[context.root] = evaluation_id
            self._runs[evaluation_id] = run

        def execute() -> CoordinatorOutcome:
            run._set_state("running")
            request = context.request

            try:
                startup.prepare_dependencies(inference_lease, token.is_cancelled)

                daemon = None
                if not token.is_cancelled():
                    validator = getattr(
                        request.env_spec, "validate_evaluation_launch", None
                    )
                    if validator is not None:
                        try:
                            validator(request.args)
                        except BaseException as exc:
                            startup.fail(exc)
                            raise
                    daemon = (
                        request.policy_daemon_factory(
                            args=request.args,
                            output_dir=context.root,
                        )
                        if request.policy_daemon_factory is not None
                        else request.env_spec.build_policy_daemon(
                            request.args, context.root
                        )
                    )
                    startup.record(
                        "simulator_starting",
                        "policy_daemon_constructed",
                    )
                effective_worker_environment = dict(request.worker_environment)
                effective_worker_environment.update(startup.worker_environment)
                coordinator_arguments = {
                    "manifest": context.manifest,
                    "manifest_path": context.root / "evaluation_manifest.json",
                    "results_path": context.root / "results.jsonl",
                    "summary_path": context.root / "summary.json",
                    "policy_daemon": daemon,
                    "worker_timeout_s": request.args.worker_timeout_s,
                    "jobs": request.args.jobs,
                    "cancellation": token,
                    "worker_environment": effective_worker_environment,
                }
                if self._attempt_registered is not None:
                    coordinator_arguments["attempt_registered"] = (
                        self._attempt_registered
                    )
                if self._result_acceptance_factory is not None:
                    coordinator_arguments["result_acceptance_factory"] = (
                        self._result_acceptance_factory
                    )
                try:
                    coordinator = self._coordinator_factory(**coordinator_arguments)
                    startup.record(
                        "simulator_starting",
                        "coordinator_constructed",
                    )
                    outcome = coordinator.run(resume=resume)
                except BaseException as exc:
                    startup.record(
                        "simulator_starting",
                        "coordinator_failed",
                        error_type=_snake_error_type(type(exc).__name__),
                    )
                    startup.fail(exc)
                    logger.exception(
                        "evaluation %s coordinator failed",
                        evaluation_id,
                    )
                    raise
                if outcome.interrupted:
                    startup.cancel()
                else:
                    preparation_failure = _incomplete_preparation_failure(
                        outcome,
                        request.args.worker_timeout_s,
                    )
                    if preparation_failure is not None:
                        startup.fail(preparation_failure)
                if getattr(outcome, "startup_error", False):
                    startup.fail(
                        RuntimeError("coordinator infrastructure startup failed"),
                    )
                return outcome
            finally:
                if inference_lease is not None:
                    inference_lease.release()

        try:
            future = self._executor.submit(execute)
        except BaseException:
            with self._lock:
                self._active.pop(evaluation_id, None)
                if previous_context is None:
                    self._contexts.pop(evaluation_id, None)
                else:
                    self._contexts[evaluation_id] = previous_context
                if previous_run is None:
                    self._runs.pop(evaluation_id, None)
                else:
                    self._runs[evaluation_id] = previous_run
                if previous_owner is None:
                    self._root_owners.pop(context.root)
                else:
                    self._root_owners[context.root] = previous_owner
            if startup.preparation_created:
                startup.preparation.path.unlink(missing_ok=True)
            raise
        run._bind(future)

        def finished(completed: Future[CoordinatorOutcome]) -> None:
            try:
                outcome = completed.result()
            except BaseException:
                state = "failed"
            else:
                state = (
                    "cancelled"
                    if outcome.interrupted
                    else "failed"
                    if getattr(outcome, "infrastructure_error", False)
                    else "completed"
                )
            with self._lock:
                if self._active.get(evaluation_id) is run:
                    self._active.pop(evaluation_id)
                run._set_state(state)
            if self._on_finished is not None:
                try:
                    self._on_finished(context)
                except BaseException:
                    logger.exception(
                        "evaluation %s completion observer failed",
                        evaluation_id,
                    )

        future.add_done_callback(finished)
        return run


def _snapshot_request(request: StartEvaluation) -> StartEvaluation:
    if not isinstance(request, StartEvaluation):
        raise TypeError("request must be StartEvaluation")
    try:
        args = copy.deepcopy(request.args)
        resolved_tasks = (
            None if request.resolved_tasks is None else tuple(request.resolved_tasks)
        )
    except Exception as exc:
        raise ValueError(f"cannot snapshot evaluation request: {exc}") from exc
    return replace(
        request,
        args=args,
        argv=tuple(request.argv),
        resolved_tasks=resolved_tasks,
        worker_environment=dict(request.worker_environment),
    )


def _allows_selective_case_rerun(manifest: EvaluationManifest) -> bool:
    return manifest.qualification.get("allow_selective_case_rerun") is not False


def _official_run_is_terminal(context: PreparedEvaluation) -> bool:
    profile = context.manifest.qualification.get("official_profile")
    if not isinstance(profile, str) or not profile:
        return False
    try:
        qualification = _direct_json(context.root, "qualification.json")
        metrics = _direct_json(context.root, "official_metrics.json")
    except (ConflictError, OSError):
        return True
    if qualification is not None and (
        qualification.get("profile_id") == profile
        and qualification.get("accepted") is True
        and qualification.get("status") == "qualified"
    ):
        return True
    if metrics is None:
        return False
    try:
        return OfficialMetrics.from_payload(context.manifest, metrics).is_complete
    except ValueError:
        return True


def _direct_json(root: Path, name: str) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root / name, flags)
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8 * 1024 * 1024:
            raise ConflictError(f"invalid official projection: {name}")
        raw = os.read(descriptor, metadata.st_size + 1)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw)
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConflictError(f"invalid official projection: {name}") from exc
    if not isinstance(payload, dict):
        raise ConflictError(f"invalid official projection: {name}")
    return payload


def _set_option(argv: Iterable[str], option: str, value: str) -> tuple[str, ...]:
    values = [str(item) for item in argv]
    for index, current in enumerate(values):
        if current == option:
            if index + 1 >= len(values):
                raise ValueError(f"{option} requires a value")
            values[index + 1] = value
            return tuple(values)
        if current.startswith(f"{option}="):
            values[index] = f"{option}={value}"
            return tuple(values)
    return (*values, option, value)


def _preloaded_policy_daemon(*, args: Any, output_dir: Any) -> None:
    del args, output_dir
    return None


def _incomplete_preparation_failure(
    outcome: Any,
    worker_timeout_s: float | None,
) -> Exception | None:
    summary = getattr(outcome, "summary", None)
    if getattr(summary, "timeouts", 0):
        deadline = (
            "its task-specific timeout"
            if worker_timeout_s is None
            else f"timeout {worker_timeout_s:g}s"
        )
        return TimeoutError(
            f"worker exceeded {deadline} before preparation completed"
        )
    if getattr(summary, "errors", 0):
        return RuntimeError("coordinator failed before preparation completed")
    return None


__all__ = [
    "ConflictError",
    "EvaluationNotFoundError",
    "EvaluationRun",
    "EvaluationService",
    "ManifestConfirmationError",
    "PreparedEvaluation",
    "StartEvaluation",
]
