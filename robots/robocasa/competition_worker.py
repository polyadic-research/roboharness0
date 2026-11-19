"""Run one official RoboCasa task as a sequential-reset batch."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.robocasa.competition import (
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.daemon import RoboCasaDaemon
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.tasks import packaged_registry_snapshot
from rpent.evaluation.attempt_registration import register_attempt
from rpent.evaluation.batches import (
    TaskBatchAttempt,
    append_task_batch_attempt,
    validate_case_attempt_id,
    validate_task_batch_attempt_id,
    validate_task_batch_result,
)
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult, ResultStore
from rpent.evaluation.worker import run_episode_case

_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def run_task_batch(
    manifest: EvaluationManifest,
    task_name: str,
    batch_attempt_id: str,
    output_jsonl: Path,
    *,
    batch_ledger: Path,
    daemon_factory: Callable[..., Any] = RoboCasaDaemon,
    case_runner: Callable[..., EpisodeResult] = run_episode_case,
    clock: Callable[[], str] | None = None,
) -> int:
    """Run the manifest's exact ordered Cases for one task-owned environment."""
    validate_task_batch_attempt_id(batch_attempt_id)
    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or any(character in task_name for character in ("/", "\\", "\0"))
    ):
        raise ValueError("task_name must be a safe non-empty string")
    results_path = Path(output_jsonl)
    ledger_path = Path(batch_ledger)
    batch_root = _validate_output_paths(
        manifest,
        task_name,
        batch_attempt_id,
        results_path,
        ledger_path,
    )
    _validate_official_manifest(manifest)
    cases, task_fact, config = _batch_inputs(manifest, task_name, batch_attempt_id)
    policy_endpoint = _policy_endpoint(manifest)
    if manifest.runtime.get("shared_policy_required") is True and not policy_endpoint:
        raise ValueError("task batch requires a shared policy endpoint")
    result_store = ResultStore(results_path, manifest)
    if result_store.load():
        raise ValueError("task batch results path must start empty")
    now = clock or _utc_now
    started_at = now()
    expected = tuple(case.case_id for case in cases)
    append_task_batch_attempt(
        ledger_path,
        manifest,
        TaskBatchAttempt(
            manifest_id=manifest.manifest_id,
            task_name=task_name,
            batch_attempt_id=batch_attempt_id,
            status="running",
            expected_case_ids=expected,
            completed_case_ids=(),
            started_at=started_at,
            finished_at=None,
        ),
    )
    for case in cases:
        register_attempt(
            manifest,
            case,
            _case_attempt_id(batch_attempt_id, case),
            started_at,
        )
    endpoint = manifest.runtime.get("simulator_endpoint")
    daemon = None
    completed: list[str] = []
    terminal_status = "complete"
    exit_code = 0
    try:
        daemon = daemon_factory(
            config=config,
            endpoint=endpoint,
            interpreter=manifest.runtime.get("simulator_interpreter"),
            log_path=batch_root / "robocasa_env_server.log",
            config_path=batch_root / "robocasa_env_config.json",
            formal=True,
            close_remote=endpoint is None,
        )
        daemon.start()
        if not isinstance(daemon.endpoint, str) or not daemon.endpoint.strip():
            raise RuntimeError("task-owned RoboCasa daemon did not publish an endpoint")
        with _temporary_environment(
            RPENT_SHARED_ENV_ENDPOINT=daemon.endpoint,
            RPENT_SHARED_ENV_SCENE_SPLIT=str(task_fact["split"]),
            RPENT_SHARED_POLICY_ENDPOINT=policy_endpoint,
        ):
            for case in cases:
                attempt_id = _case_attempt_id(batch_attempt_id, case)
                result = case_runner(
                    case,
                    manifest,
                    attempt_id=attempt_id,
                    task_batch_attempt_id=batch_attempt_id,
                )
                _validate_batch_result(
                    result,
                    manifest=manifest,
                    case=case,
                    attempt_id=attempt_id,
                    batch_attempt_id=batch_attempt_id,
                )
                result_store.append(result)
                if not _is_completed_rollout(result):
                    terminal_status = "failed"
                    exit_code = 1
                    break
                completed.append(case.case_id)
    except KeyboardInterrupt as exc:
        _retain_interrupted_result(
            exc,
            result_store,
            completed,
            cases,
            batch_attempt_id,
        )
        terminal_status = "cancelled"
        exit_code = 130
    except Exception:
        terminal_status = "failed"
        exit_code = 1
    finally:
        if daemon is not None:
            try:
                daemon.stop()
            except Exception:
                terminal_status = "failed"
                exit_code = 1
    append_task_batch_attempt(
        ledger_path,
        manifest,
        TaskBatchAttempt(
            manifest_id=manifest.manifest_id,
            task_name=task_name,
            batch_attempt_id=batch_attempt_id,
            status=terminal_status,
            expected_case_ids=expected,
            completed_case_ids=tuple(completed),
            started_at=started_at,
            finished_at=now(),
        ),
    )
    return exit_code


def _validate_official_manifest(manifest: EvaluationManifest) -> None:
    snapshot = packaged_registry_snapshot()
    profile = resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)
    failures: list[str] = []
    if manifest.environment.get("evaluation_profile") != profile.profile_id:
        failures.append("environment evaluation_profile")
    if manifest.simulator.get("evaluation_profile") != profile.profile_id:
        failures.append("simulator evaluation_profile")
    expected_schedule = {
        "evaluation_profile": profile.profile_id,
        "worker_grouping": profile.worker_grouping,
        "episodes_per_task": profile.rollouts_per_task,
        "base_seed": profile.scenario_seed,
        "seed_policy": "shared_environment_sequence",
        "scene_split": profile.scene_split,
        "scenario_sampling": profile.scenario_sampling,
        "success_source": profile.success_source,
        "horizon_source": profile.horizon_source,
        "task_count": len(profile.task_names),
        "task_sets": profile.task_sets,
        "task_registry_digest": snapshot["registry_digest"],
    }
    for name, expected in expected_schedule.items():
        if manifest.schedule.get(name) != expected:
            failures.append(f"schedule.{name}")
    expected_membership = {
        name: tuple(snapshot["groups"][name]["tasks"]) for name in profile.task_sets
    }
    actual_membership = manifest.schedule.get("task_set_membership")
    if (
        not isinstance(actual_membership, Mapping)
        or {name: tuple(actual_membership.get(name, ())) for name in profile.task_sets}
        != expected_membership
    ):
        failures.append("schedule.task_set_membership")
    if (
        manifest.simulator.get("task_registry_digest") != snapshot["registry_digest"]
        or manifest.simulator.get("robocasa_commit") != ROBOCASA_COMMIT
        or manifest.simulator.get("robosuite_commit") != ROBOSUITE_COMMIT
        or manifest.simulator.get("scene_split") != profile.scene_split
        or manifest.simulator.get("split") != profile.scene_split
    ):
        failures.append("simulator registry/split identity")
    task_facts = manifest.simulator.get("tasks")
    fact_names = (
        tuple(fact.get("name") for fact in task_facts)
        if isinstance(task_facts, tuple)
        and all(isinstance(fact, Mapping) for fact in task_facts)
        else ()
    )
    if fact_names != profile.task_names:
        failures.append("simulator exact 50-task membership")
    expected_case_count = len(profile.task_names) * int(profile.rollouts_per_task or 0)
    if len(manifest.cases) != expected_case_count:
        failures.append("case count")
    else:
        for task_index, task_name in enumerate(profile.task_names):
            start = task_index * int(profile.rollouts_per_task or 0)
            task_cases = manifest.cases[
                start : start + int(profile.rollouts_per_task or 0)
            ]
            fact = task_facts[task_index] if isinstance(task_facts, tuple) else {}
            if (
                tuple(case.task_name for case in task_cases)
                != (task_name,) * int(profile.rollouts_per_task or 0)
                or tuple(case.episode_index for case in task_cases)
                != tuple(range(int(profile.rollouts_per_task or 0)))
                or any(case.seed != profile.scenario_seed for case in task_cases)
                or not isinstance(fact, Mapping)
                or fact.get("split") != profile.scene_split
                or any(
                    case.horizon != fact.get("upstream_horizon") for case in task_cases
                )
            ):
                failures.append(f"Cases for {task_name}")
                break
    if manifest.task_selector != ",".join(profile.task_names):
        failures.append("task_selector")
    if manifest.runtime.get("shared_policy_required") is not True:
        failures.append("runtime.shared_policy_required")
    if failures:
        raise ValueError("invalid official RoboCasa manifest: " + ", ".join(failures))


def _validate_output_paths(
    manifest: EvaluationManifest,
    task_name: str,
    batch_attempt_id: str,
    results_path: Path,
    ledger_path: Path,
) -> Path:
    configured_root = str(manifest.runtime.get("evaluation_root", ""))
    if not configured_root.strip() or not Path(configured_root).is_absolute():
        raise ValueError("evaluation_root must be an absolute canonical path")
    root = Path(configured_root).resolve()
    batch_root = root / "task_batches" / task_name / batch_attempt_id
    expected_results = batch_root / "results.jsonl"
    expected_ledger = batch_root / "batch.jsonl"
    if (
        results_path.resolve() != expected_results
        or ledger_path.resolve() != expected_ledger
    ):
        raise ValueError("batch outputs must use canonical attempt-local paths")
    return batch_root


def _batch_inputs(
    manifest: EvaluationManifest,
    task_name: str,
    batch_attempt_id: str,
) -> tuple[tuple[Any, ...], Mapping[str, Any], dict[str, Any]]:
    if not isinstance(manifest, EvaluationManifest):
        raise ValueError("manifest must be an EvaluationManifest")
    if manifest.schedule.get("worker_grouping") != "task":
        raise ValueError("task batch worker requires worker_grouping='task'")
    for name, value in (
        ("task_name", task_name),
        ("batch_attempt_id", batch_attempt_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if any(character in value for character in ("/", "\\", "\0")):
            raise ValueError(f"{name} contains an unsafe path character")
    cases = tuple(case for case in manifest.cases if case.task_name == task_name)
    expected_count = manifest.schedule.get("episodes_per_task")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(cases) != expected_count
    ):
        raise ValueError("task batch Case count does not match immutable schedule")
    if tuple(case.episode_index for case in cases) != tuple(range(len(cases))):
        raise ValueError("task batch Cases are not in exact episode order")
    seed = manifest.schedule.get("base_seed")
    if any(case.seed != seed for case in cases):
        raise ValueError("task batch Case seeds do not match immutable schedule")
    raw_tasks = manifest.simulator.get("tasks")
    if not isinstance(raw_tasks, tuple):
        raise ValueError("simulator.tasks must be immutable task facts")
    facts = tuple(
        fact
        for fact in raw_tasks
        if isinstance(fact, Mapping) and fact.get("name") == task_name
    )
    if len(facts) != 1:
        raise ValueError("manifest must contain exactly one immutable task fact")
    fact = facts[0]
    upstream_horizon = fact.get("upstream_horizon")
    if (
        isinstance(upstream_horizon, bool)
        or not isinstance(upstream_horizon, int)
        or upstream_horizon <= 0
        or any(case.horizon != upstream_horizon for case in cases)
    ):
        raise ValueError("task batch horizon does not match upstream task fact")
    split = fact.get("split")
    if split != manifest.schedule.get("scene_split"):
        raise ValueError("task split does not match immutable schedule")
    config = {
        "task": task_name,
        "split": split,
        "seed": seed,
        "robot": manifest.simulator.get("robot"),
        "camera_names": list(_CAMERAS),
        "camera_size": 256,
        "object_registries": list(manifest.simulator.get("object_registries", ())),
        "rendering_mode": "headless",
        "upstream_horizon": upstream_horizon,
        "safety_ceiling": upstream_horizon,
        "task_registry_commit": manifest.simulator.get("robocasa_commit"),
        "task_registry_digest": manifest.simulator.get("task_registry_digest"),
    }
    return cases, fact, config


def _validate_batch_result(
    result: EpisodeResult,
    *,
    manifest: EvaluationManifest,
    case: Any,
    attempt_id: str,
    batch_attempt_id: str,
) -> None:
    if not isinstance(result, EpisodeResult):
        raise TypeError("case runner must return EpisodeResult")
    if attempt_id != _case_attempt_id(batch_attempt_id, case):
        raise ValueError("case runner attempt identity is not canonical")
    validate_task_batch_result(manifest, case, batch_attempt_id, result)
    expected = {
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "environment": case.environment,
        "preset_name": case.preset_name,
        "task_name": case.task_name,
        "task_index": case.task_index,
        "seed": case.seed,
        "episode_index": case.episode_index,
        "checkpoint_repo_id": case.checkpoint_repo_id,
        "checkpoint_revision": case.checkpoint_revision,
        "binding_name": case.binding_name,
    }
    drift = {
        name: (value, getattr(result, name))
        for name, value in expected.items()
        if getattr(result, name) != value
    }
    if drift:
        raise ValueError(f"case runner returned drifting result identity: {drift}")


def _is_completed_rollout(result: EpisodeResult) -> bool:
    if result.status in {"success", "terminal_failure", "truncated"}:
        return True
    return (
        result.status == "error"
        and result.error_type is None
        and any(
            (
                result.simulator_success,
                result.contract_success,
                result.terminated,
                result.truncated,
            )
        )
    )


def _retain_interrupted_result(
    exc: BaseException,
    store: ResultStore,
    completed: list[str],
    cases: tuple[Any, ...],
    batch_attempt_id: str,
) -> None:
    result = getattr(exc, "rpent_result", None)
    if not isinstance(result, EpisodeResult):
        return
    next_case = cases[len(completed)]
    _validate_batch_result(
        result,
        manifest=store.manifest,
        case=next_case,
        attempt_id=_case_attempt_id(batch_attempt_id, next_case),
        batch_attempt_id=batch_attempt_id,
    )
    store.append(result)


def _case_attempt_id(batch_attempt_id: str, case: Any) -> str:
    return validate_case_attempt_id(
        f"{batch_attempt_id}-{case.episode_index:03d}-{case.case_id[:12]}"
    )


def _policy_endpoint(manifest: EvaluationManifest) -> str | None:
    inherited = os.environ.get("RPENT_SHARED_POLICY_ENDPOINT")
    configured = manifest.runtime.get("policy_endpoint")
    if configured is not None and (
        not isinstance(configured, str) or not configured.strip()
    ):
        raise ValueError("manifest shared policy endpoint must be a non-empty string")
    if inherited and configured and inherited != configured:
        raise ValueError("shared policy endpoint conflicts with manifest runtime")
    endpoint = inherited or configured
    return endpoint if endpoint else None


@contextmanager
def _temporary_environment(**updates: str | None):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--batch-attempt-id", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--batch-ledger", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_task_batch(
        EvaluationManifest.load(args.manifest),
        args.task,
        args.batch_attempt_id,
        Path(args.results),
        batch_ledger=Path(args.batch_ledger),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_task_batch"]
