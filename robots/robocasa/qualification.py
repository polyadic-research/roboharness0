"""Fail-closed RoboCasa365 leaderboard qualification and bundle projection."""

from __future__ import annotations

import importlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.robocasa.binding import make_robocasa_binding
from robots.robocasa.competition import (
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    RoboCasaDependencyReport,
    inspect_robocasa_dependencies,
)
from robots.robocasa.submission import SubmissionMetadata
from robots.robocasa.tasks import packaged_registry_snapshot
from rpent.evaluation.batches import TaskBatchEvidence, select_complete_task_batch
from rpent.evaluation.diagnostics import aggregate_rpent_diagnostics
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import (
    OfficialMetrics,
    aggregate_official_metrics,
)
from rpent.evaluation.results import EpisodeResult, ResultStore, aggregate_results
from rpent.planning import (
    ROBOCASA_HORIZON_PLANNING_PROFILE,
    get_planning_profile,
)
from rpent.policies.lerobot_smolvla_artifacts import binding_identity
from rpent.policies.smolvla_catalog import (
    SMOLVLA_CHECKPOINTS,
    SMOLVLA_LEROBOT_REVISION,
    SMOLVLA_PROCESSOR_DIGESTS,
)
from rpent.recording.contracts import QualificationOutcome
from rpent.recording.media import atomic_write_json

QUALIFICATION_CHECK_IDS = (
    "profile_identity",
    "rpent_identity",
    "official_launch_runtime",
    "task_membership_digest",
    "scene_split",
    "schedule_completion",
    "seed_reset_protocol",
    "installed_dependencies",
    "object_registries",
    "upstream_horizons",
    "policy_binding_planner_identity",
    "native_success_authority",
    "accepted_task_batches",
    "selective_rerun_disabled",
    "complete_artifacts",
    "submission_metadata",
)


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    id: str
    accepted: bool
    evidence: Any
    remediation: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "accepted": self.accepted,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class RoboCasaQualification:
    profile_id: str
    status: str
    accepted: bool
    checks: tuple[QualificationCheck, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "status": self.status,
            "accepted": self.accepted,
            "checks": [check.to_payload() for check in self.checks],
        }


def _policy_binding_identity_matches(policy: Mapping) -> bool:
    if policy.get("policy_family") == "xiaomi-mibot":
        from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
        from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
        from rpent.policies.xiaomi_mibot_server import SERVER_VERSION

        health = policy.get("health")
        return (
            policy.get("protocol_version") == 1
            and policy.get("checkpoint") == XIAOMI_MIBOT_CHECKPOINT.to_payload()
            and policy.get("binding_digest") == binding_identity(make_xiaomi_policy_binding())
            and policy.get("server_version") == SERVER_VERSION
            and policy.get("remote_code_revision") == XIAOMI_MIBOT_CHECKPOINT.revision
            and isinstance(policy.get("artifact_identity"), str)
            and re.fullmatch(r"[0-9a-f]{64}", policy["artifact_identity"]) is not None
            and isinstance(health, Mapping)
            and all(health.get(key) == policy.get(key) for key in (
                "protocol_version", "checkpoint", "artifact_identity",
                "remote_code_revision", "server_version",
            ))
            and health.get("binding_identity") == policy["binding_digest"]
        )
    if policy.get("policy_family") == "pi05":
        from robots.robocasa.pi05_binding import make_pi05_binding
        from rpent.policies.robocasa_pi05_catalog import (
            OPENPI_COMMIT,
            PI05_CHECKPOINT,
            PI05_CONFIG,
        )

        return (
            policy.get("protocol_version") == 1
            and policy.get("checkpoint") == PI05_CHECKPOINT.to_payload()
            and policy.get("openpi_commit") == OPENPI_COMMIT
            and policy.get("config_name") == PI05_CONFIG
            and policy.get("binding_digest") == binding_identity(make_pi05_binding())
        )
    return (
        policy.get("policy_family", "smolvla") == "smolvla"
        and policy.get("checkpoint") == SMOLVLA_CHECKPOINTS["robocasa"].to_payload()
        and policy.get("lerobot_revision") == SMOLVLA_LEROBOT_REVISION
        and policy.get("binding_digest") == binding_identity(make_robocasa_binding())
        and policy.get("processor_digests") == SMOLVLA_PROCESSOR_DIGESTS["robocasa"]
    )


def _runtime_policy_identities_match(policy: Mapping, results: Iterable[EpisodeResult]) -> bool:
    if policy.get("policy_family") != "xiaomi-mibot":
        return _policy_binding_identity_matches(policy)
    retained = tuple(results)
    if not retained:
        return False
    identities = set()
    for result in retained:
        try:
            runtime_policy = json.loads((Path(result.artifact_dir) / "policy.json").read_text())
            if not _policy_binding_identity_matches(runtime_policy):
                return False
            if any(runtime_policy.get(key) != policy.get(key) for key in (
                "policy_family", "protocol_version", "checkpoint", "binding_digest",
            )):
                return False
            identities.add(runtime_policy["artifact_identity"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
    return len(identities) == 1


def qualify_robocasa_run(
    manifest: EvaluationManifest,
    results: Iterable[EpisodeResult],
    batches: Iterable[TaskBatchEvidence],
    *,
    dependencies: RoboCasaDependencyReport | None,
) -> tuple[RoboCasaQualification, OfficialMetrics]:
    """Evaluate every official invariant without inventing missing evidence."""
    retained = tuple(results)
    evidences = tuple(batches)
    metrics = aggregate_official_metrics(manifest, retained, evidences)
    snapshot = packaged_registry_snapshot()
    profile = resolve_evaluation_profile(LEADERBOARD_PROFILE_ID, snapshot)
    task_names = tuple(dict.fromkeys(case.task_name for case in manifest.cases))
    facts = manifest.simulator.get("tasks")
    fact_map = (
        {
            str(fact.get("name")): fact
            for fact in facts
            if isinstance(facts, tuple) and isinstance(fact, Mapping)
        }
        if isinstance(facts, tuple)
        else {}
    )
    expected_membership = {
        name: tuple(snapshot["groups"][name]["tasks"]) for name in profile.task_sets
    }
    selected = tuple(
        batch
        for task in task_names
        if (
            batch := select_complete_task_batch(
                item for item in evidences if item.task_name == task
            )
        )
        is not None
    )
    selected_results = tuple(result for batch in selected for result in batch.results)
    exact_attempts = (
        len(evidences) == 50
        and all(
            sum(item.task_name == task for item in evidences) == 1
            for task in task_names
        )
        and len(retained) == 2_500
        and retained == selected_results
    )
    try:
        SubmissionMetadata.from_payload(
            manifest.qualification.get("submission_metadata")
        )
    except ValueError:
        metadata_valid = False
    else:
        metadata_valid = True
    planning_profile = manifest.policy.get("planning_profile")
    submitted = manifest.launch.get("submitted")
    effective = manifest.launch.get("effective")
    try:
        canonical_planning_profile = (
            get_planning_profile(planning_profile).id
            if isinstance(planning_profile, str) and planning_profile.strip()
            else None
        )
    except ValueError:
        canonical_planning_profile = None
    submitted_planning_profile = (
        submitted.get(
            "planning_profile", ROBOCASA_HORIZON_PLANNING_PROFILE
        )
        if isinstance(submitted, Mapping)
        else None
    )
    try:
        canonical_submitted_planning_profile = (
            get_planning_profile(submitted_planning_profile).id
            if isinstance(submitted_planning_profile, str)
            and submitted_planning_profile.strip()
            else None
        )
    except ValueError:
        canonical_submitted_planning_profile = None
    canonical_worker_planning_profile = _canonical_worker_planning_profile(
        manifest.cli_arguments.get("argv")
    )
    planning_profile_consistent = (
        canonical_planning_profile == planning_profile
        and isinstance(submitted, Mapping)
        and isinstance(effective, Mapping)
        and effective.get("planning_profile") == planning_profile
        and (
            manifest.launch.get("execution_mode") == "cli"
            and not submitted
            or canonical_submitted_planning_profile == planning_profile
        )
        and canonical_worker_planning_profile == planning_profile
    )

    checks = (
        _check(
            "profile_identity",
            manifest.environment.get("evaluation_profile") == profile.profile_id
            and manifest.simulator.get("evaluation_profile") == profile.profile_id
            and manifest.schedule.get("evaluation_profile") == profile.profile_id
            and manifest.qualification.get("official_profile") == profile.profile_id,
            manifest.schedule.get("evaluation_profile"),
            "recreate the run with the locked leaderboard profile",
        ),
        _check(
            "rpent_identity",
            manifest.rpent_dirty is False
            and re.fullmatch(r"[0-9a-f]{40}", manifest.rpent_commit) is not None,
            {"commit": manifest.rpent_commit, "dirty": manifest.rpent_dirty},
            "run from a clean content-addressed RPent commit",
        ),
        _official_launch_runtime_check(manifest, profile.profile_id),
        _check(
            "task_membership_digest",
            task_names == profile.task_names
            and tuple(manifest.schedule.get("task_sets", ())) == profile.task_sets
            and {
                name: tuple(members)
                for name, members in manifest.schedule.get(
                    "task_set_membership", {}
                ).items()
            }
            == expected_membership
            and manifest.schedule.get("task_set_sizes")
            == {name: len(members) for name, members in expected_membership.items()}
            and manifest.schedule.get("task_registry_digest")
            == snapshot["registry_digest"]
            and manifest.simulator.get("task_registry_digest")
            == snapshot["registry_digest"],
            {
                "task_count": len(task_names),
                "digest": manifest.schedule.get("task_registry_digest"),
            },
            "use the packaged immutable 50-task registry",
        ),
        _check(
            "scene_split",
            manifest.schedule.get("scene_split") == "pretrain"
            and manifest.simulator.get("scene_split") == "pretrain"
            and manifest.simulator.get("split") == "pretrain"
            and all(fact.get("split") == "pretrain" for fact in fact_map.values()),
            manifest.schedule.get("scene_split"),
            "run every task on the locked pretrain scene split",
        ),
        _check(
            "schedule_completion",
            len(task_names) == metrics.scheduled_tasks == metrics.completed_tasks == 50
            and metrics.scheduled_rollouts == metrics.completed_rollouts == 2_500
            and manifest.schedule.get("episodes_per_task") == 50
            and metrics.is_complete,
            {"tasks": metrics.completed_tasks, "rollouts": metrics.completed_rollouts},
            "resume every incomplete task batch until all 2,500 rollouts complete",
        ),
        _check(
            "seed_reset_protocol",
            manifest.schedule.get("base_seed") == 7
            and manifest.schedule.get("seed_policy") == "shared_environment_sequence"
            and manifest.schedule.get("scenario_sampling")
            == "seeded_task_environment_sequential_resets"
            and all(case.seed == 7 for case in manifest.cases),
            {
                "seed": manifest.schedule.get("base_seed"),
                "seed_policy": manifest.schedule.get("seed_policy"),
            },
            "use seed 7 and the shared environment sequential-reset protocol",
        ),
        _dependency_check(manifest, dependencies),
        _check(
            "object_registries",
            tuple(manifest.simulator.get("object_registries", ())) == ("lightwheel",)
            and dependencies is not None
            and "lightwheel" in dependencies.available_registries,
            manifest.simulator.get("object_registries"),
            "install and select the locked lightwheel object registry",
        ),
        _check(
            "upstream_horizons",
            manifest.contracts.get("horizon_rule") == "upstream_horizon"
            and manifest.runtime.get("requested_safety_ceiling") is None
            and len(fact_map) == 50
            and all(
                case.horizon == fact_map.get(case.task_name, {}).get("upstream_horizon")
                for case in manifest.cases
            ),
            manifest.contracts.get("horizon_rule"),
            "remove lower safety ceilings and use every upstream horizon",
        ),
        _check(
            "policy_binding_planner_identity",
            _runtime_policy_identities_match(manifest.policy, selected_results)
            and planning_profile_consistent,
            {
                "checkpoint": dict(manifest.policy.get("checkpoint", {})),
                "planning_profile": manifest.policy.get("planning_profile"),
            },
            "use one registered planning profile consistently in policy and launch identity",
        ),
        _check(
            "native_success_authority",
            manifest.schedule.get("success_source") == "robocasa_native_task_predicate"
            and manifest.contracts.get("success_source")
            == "robocasa_native_task_predicate",
            manifest.schedule.get("success_source"),
            "use the native RoboCasa task predicate as official authority",
        ),
        _check(
            "accepted_task_batches",
            len(selected) == 50 and metrics.is_complete and exact_attempts,
            {
                "accepted_batches": len(selected),
                "attempts": len(evidences),
                "central_results": len(retained),
            },
            "retain exactly one attempt and exit-zero outcome per task with no extra results",
        ),
        _check(
            "selective_rerun_disabled",
            manifest.qualification.get("allow_selective_case_rerun") is False
            and all(
                tuple(result.episode_index for result in batch.results)
                == tuple(range(50))
                for batch in selected
            ),
            manifest.qualification.get("allow_selective_case_rerun"),
            "resume incomplete tasks from episode zero; never rerun selected Cases",
        ),
        _artifact_check(selected),
        _check(
            "submission_metadata",
            metadata_valid,
            "present" if metadata_valid else "missing_or_invalid",
            "provide schema-valid bounded submission metadata",
        ),
    )
    accepted = all(check.accepted for check in checks)
    status = (
        "qualified"
        if accepted
        else "partial"
        if not metrics.is_complete
        else "rejected"
    )
    return RoboCasaQualification(profile.profile_id, status, accepted, checks), metrics


def _check(
    id: str, accepted: bool, evidence: Any, remediation: str
) -> QualificationCheck:
    return QualificationCheck(id, bool(accepted), evidence, remediation)


def _canonical_worker_planning_profile(argv: Any) -> str | None:
    if not isinstance(argv, (list, tuple)):
        return None
    requested = []
    index = 0
    while index < len(argv):
        value = str(argv[index])
        if value == "--planning-profile":
            index += 1
            if index == len(argv) or str(argv[index]).startswith("--"):
                return None
            requested.append(str(argv[index]))
        elif value.startswith("--planning-profile="):
            requested.append(value.split("=", 1)[1])
        index += 1
    if not requested:
        requested.append(ROBOCASA_HORIZON_PLANNING_PROFILE)
    try:
        canonical = tuple(get_planning_profile(value).id for value in requested)
    except ValueError:
        return None
    return canonical[0] if len(set(canonical)) == 1 else None


def _dependency_check(
    manifest: EvaluationManifest,
    report: RoboCasaDependencyReport | None,
) -> QualificationCheck:
    accepted = (
        report is not None
        and report.robocasa_commit == ROBOCASA_COMMIT
        and report.robosuite_commit == ROBOSUITE_COMMIT
        and report.robocasa_version == "1.0.1"
        and bool(report.robosuite_version)
        and bool(report.mujoco_version)
        and report.mujoco_gl.lower() == "egl"
        and report.default_assets_ready is True
        and manifest.simulator.get("robocasa_commit") == ROBOCASA_COMMIT
        and manifest.simulator.get("robosuite_commit") == ROBOSUITE_COMMIT
    )
    return _check(
        "installed_dependencies",
        accepted,
        None if report is None else report.to_payload(),
        "inspect the installed simulator and match both pinned commits",
    )


def _official_launch_runtime_check(
    manifest: EvaluationManifest, profile_id: str
) -> QualificationCheck:
    launch = manifest.launch
    submitted = launch.get("submitted")
    effective = launch.get("effective")
    locked = manifest.schedule.get("locked_inputs")
    scheduled = manifest.schedule.get("effective_inputs")
    mappings = all(
        isinstance(value, Mapping)
        for value in (submitted, effective, locked, scheduled)
    )
    accepted = mappings
    if mappings:
        launch_identity = (
            launch.get("execution_mode") == "hybrid"
            and launch.get("preset_version") in {6, 13}
            and launch.get("runner") == manifest.environment.get("preset_name")
            and submitted.get("evaluation_profile") == profile_id
        ) or (
            launch.get("execution_mode") == "cli"
            and launch.get("preset_version") == 1
            and launch.get("runner") == "evaluation-front-controller"
            and not submitted
        )
        accepted = (
            launch_identity
            and all(effective.get(name) == value for name, value in locked.items())
            and all(effective.get(name) == value for name, value in scheduled.items())
            and all(
                name not in submitted or submitted.get(name) == value
                for name, value in locked.items()
            )
            and manifest.schedule.get("worker_grouping") == "task"
            and manifest.runtime.get("grouped_worker_module")
            == "robots.robocasa.competition_worker"
            and manifest.runtime.get("qualification_module")
            == "robots.robocasa.qualification"
            and manifest.runtime.get("shared_policy_required") is True
            and manifest.runtime.get("policy_endpoint_argument") == "--policy-endpoint"
            and manifest.qualification.get("require_workload_confirmation") is True
        )
    return _check(
        "official_launch_runtime",
        accepted,
        {
            "execution_mode": launch.get("execution_mode"),
            "runner": launch.get("runner"),
            "grouped_worker_module": manifest.runtime.get("grouped_worker_module"),
            "qualification_module": manifest.runtime.get("qualification_module"),
        },
        "recreate the run through the locked official launch preset and modules",
    )


def _artifact_check(selected: tuple[TaskBatchEvidence, ...]) -> QualificationCheck:
    complete = 0
    expected = sum(len(batch.results) for batch in selected)
    for batch in selected:
        for result in batch.results:
            artifact = Path(result.artifact_dir).resolve()
            path = artifact / "qualification.json"
            try:
                if path.resolve() != path or not path.is_file():
                    raise ValueError
                outcome = QualificationOutcome.from_payload(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if (
                    outcome.episode_result != result
                    or outcome.artifact_health.status != "complete"
                ):
                    raise ValueError
            except (OSError, RecursionError, ValueError, json.JSONDecodeError):
                continue
            complete += 1
    return _check(
        "complete_artifacts",
        expected == 2_500 and complete == expected,
        {"complete": complete, "expected": expected},
        "repair every Case recording until artifact health is complete",
    )


def _validate_required_projections(
    root: Path,
    manifest: EvaluationManifest,
    results: tuple[EpisodeResult, ...],
    batches: tuple[TaskBatchEvidence, ...],
    metrics: OfficialMetrics,
) -> None:
    canonical_root = root.resolve()
    if (
        Path(str(manifest.runtime.get("evaluation_root", ""))).resolve()
        != canonical_root
    ):
        raise ValueError("qualification root does not match manifest evaluation_root")
    manifest_path = canonical_root / "evaluation_manifest.json"
    if (
        manifest_path.resolve() != manifest_path
        or EvaluationManifest.load(manifest_path) != manifest
    ):
        raise ValueError("evaluation_manifest.json identity drift")
    results_path = canonical_root / "results.jsonl"
    if (
        results_path.resolve() != results_path
        or ResultStore(results_path, manifest).load() != results
    ):
        raise ValueError("results.jsonl identity or payload drift")
    expected = {
        "summary.json": aggregate_results(manifest, results).to_payload(),
        "official_metrics.json": metrics.to_payload(),
        "rpent_diagnostics.json": aggregate_rpent_diagnostics(
            manifest, results, batches
        ).to_payload(),
    }
    for name, payload in expected.items():
        path = canonical_root / name
        try:
            if path.resolve() != path or json.loads(path.read_bytes()) != payload:
                raise ValueError
        except (OSError, RecursionError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{name} identity or payload drift") from exc


def _revoke_submission_exports(root: Path) -> None:
    removed = False
    for name in ("leaderboard_submission.json", "reproduction.md"):
        path = root / name
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                raise ValueError(f"cannot revoke non-file export target: {name}")
            path.unlink()
            removed = True
    if removed:
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_qualification_bundle(
    manifest: EvaluationManifest,
    results: tuple[EpisodeResult, ...],
    batches: tuple[TaskBatchEvidence, ...],
    root: Path,
) -> RoboCasaQualification:
    """Inspect installed dependencies and atomically materialize terminal artifacts."""
    qualification, metrics = qualify_robocasa_run(
        manifest, results, batches, dependencies=None
    )
    if not metrics.is_complete:
        _revoke_submission_exports(root)
        atomic_write_json(root / "qualification.json", qualification.to_payload())
        return qualification
    registries = tuple(manifest.simulator.get("object_registries", ()))
    try:
        dependencies = inspect_robocasa_dependencies(
            manifest.runtime.get("simulator_interpreter"),
            requested_registries=registries,
            formal=True,
        )
    except (OSError, RuntimeError, ValueError):
        dependencies = None
    qualification, metrics = qualify_robocasa_run(
        manifest, results, batches, dependencies=dependencies
    )
    if qualification.accepted:
        _validate_required_projections(root, manifest, results, batches, metrics)
        submission = importlib.import_module("robots.robocasa.submission")
        submission.write_submission_bundle(
            root, manifest, metrics, qualification, dependencies
        )
    else:
        _revoke_submission_exports(root)
    atomic_write_json(root / "qualification.json", qualification.to_payload())
    return qualification


__all__ = [
    "QUALIFICATION_CHECK_IDS",
    "QualificationCheck",
    "RoboCasaQualification",
    "qualify_robocasa_run",
    "write_qualification_bundle",
]
