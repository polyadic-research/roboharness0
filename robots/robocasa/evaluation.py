"""RoboCasa365-owned preparation for coordinated RPent evaluation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from robots.robocasa.binding import (
    CHECKPOINT_CHUNK_SIZE,
    CHECKPOINT_N_ACTION_STEPS,
    DEFAULT_ACTIONS_PER_POLICY_REQUEST,
    make_robocasa_binding,
)
from robots.robocasa.competition import resolve_evaluation_profile
from robots.robocasa.data_schema import robocasa_data_schema
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.display import RoboCasaDisplayAdapter
from robots.robocasa.observation import CAMERA_SHAPES
from robots.robocasa.policy_selection import (
    policy_binding,
    policy_checkpoint,
    resolve_policy_family,
)
from robots.robocasa.submission import SubmissionMetadata
from robots.robocasa.tasks import (
    packaged_registry_snapshot,
    resolve_robocasa_tasks,
    resolve_robocasa_tasks_from_contract,
)
from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
from rpent.evaluation.cases import EpisodeProtocol
from rpent.evaluation.preparation import PreparedEvaluation
from rpent.planning import (
    ROBOCASA_HORIZON_PLANNING_PROFILE,
    get_planning_profile,
    resolve_required_planning_profile,
)
from rpent.policies.lerobot_smolvla_artifacts import binding_identity
from rpent.policies.robocasa_pi05_catalog import OPENPI_COMMIT, PI05_CONFIG
from rpent.policies.smolvla_catalog import (
    SMOLVLA_CHECKPOINTS,
    SMOLVLA_LEROBOT_REVISION,
    SMOLVLA_PROCESSOR_DIGESTS,
)
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_MIBOT_POLICY_SPEC,
)


def prepare_evaluation(
    args,
    run_config,
    resolved_tasks=None,
) -> PreparedEvaluation:
    registry_snapshot = packaged_registry_snapshot()
    planning_profile = resolve_planning_profile(args)
    selected_planning_profile = get_planning_profile(planning_profile)
    family = resolve_policy_family(args)
    use_xiaomi = family == "xiaomi-mibot"
    use_pi05 = family == "pi05"
    if getattr(args, "jobs", 1) > 1:
        if family not in {"pi05", "xiaomi-mibot"} or args.policy_endpoint or args.env_endpoint:
            raise ValueError("parallel evaluation requires local Pi0.5 or Xiaomi replicas and simulators")
        if not getattr(args, "policy_devices", None):
            raise ValueError("parallel evaluation requires explicit --policy-devices")
    profile = resolve_evaluation_profile(
        getattr(args, "evaluation_profile", None),
        registry_snapshot,
    )
    if profile.official:
        if getattr(args, "jobs", 1) > 1:
            raise ValueError("task-grouped official evaluation currently requires jobs=1")
        _validate_official_inputs(args, profile)
    if resolved_tasks is None or profile.official:
        resolver = (
            resolve_robocasa_tasks_from_contract
            if args.dry_run
            else resolve_robocasa_tasks
        )
        selector = ",".join(profile.task_names) if profile.official else args.task
        tasks = tuple(resolver(selector, scene_split=profile.scene_split))
    else:
        tasks = tuple(resolved_tasks)
    commits = {str(getattr(task, "registry_commit", "")) for task in tasks}
    digests = {str(getattr(task, "registry_digest", "")) for task in tasks}
    splits = {str(getattr(task, "split", "")) for task in tasks}
    if commits != {ROBOCASA_COMMIT} or len(digests) != 1 or "" in digests:
        raise ValueError("resolved RoboCasa tasks have inconsistent provenance")
    if len(splits) != 1 or not splits.issubset({"target", "pretrain"}):
        raise ValueError("resolved RoboCasa tasks have inconsistent split")
    checkpoint = policy_checkpoint(family)
    binding = policy_binding(family)
    metadata_path = getattr(args, "submission_metadata_json", None)
    submission_metadata = (
        SubmissionMetadata.from_json_path(metadata_path)
        if profile.official and metadata_path is not None
        else None
    )
    horizon_ceiling = None if profile.official else args.max_episode_steps
    if selected_planning_profile.environment_horizon_only and horizon_ceiling is not None:
        from robots.robocasa.runtime import _effective_horizon

        for task in tasks:
            _effective_horizon(task.upstream_horizon, horizon_ceiling, horizon_only=True)
    protocol = EpisodeProtocol(
        environment="robocasa",
        preset_name=(
            "robocasa365-pandaomron-pi05-v1" if use_pi05 else
            XIAOMI_MIBOT_POLICY_SPEC.preset_name
            if use_xiaomi
            else "robocasa365-pandaomron-smolvla-v1"
        ),
        horizon_ceiling=horizon_ceiling,
        checkpoint_repo_id=checkpoint.repo_id,
        checkpoint_revision=checkpoint.revision,
        binding_name=checkpoint.binding_name,
    )
    return PreparedEvaluation(
        tasks=tasks,
        protocol=protocol,
        environment={
            "name": "robocasa",
            "preset_name": protocol.preset_name,
            "protocol": "rpent-episode-v1",
            **(
                {
                    "evaluation_profile": profile.profile_id,
                    "task_selector": ",".join(profile.task_names),
                }
                if profile.official
                else {}
            ),
        },
        simulator={
            "name": "robocasa365",
            **({"evaluation_profile": profile.profile_id} if profile.official else {}),
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "task_registry_digest": next(iter(digests)),
            "split": next(iter(splits)),
            "scene_split": next(iter(splits)),
            "task_sets": list(profile.task_sets),
            "robot": "PandaOmron",
            "object_registries": list(run_config.object_registries),
            "tasks": [
                {
                    "name": task.name,
                    "instruction": task.instruction,
                    "family": task.family,
                    "category": task.category,
                    "split": task.split,
                    "upstream_horizon": task.upstream_horizon,
                    "registry_id": getattr(task, "registry_id", None),
                }
                for task in tasks
            ],
            "task_resolution": (
                "packaged_contract" if args.dry_run else "upstream_registry"
            ),
        },
        policy=(
            {
                "protocol_version": 1,
                "policy_family": "xiaomi-mibot",
                "checkpoint": checkpoint.to_payload(),
                "binding_name": checkpoint.binding_name,
                "binding_digest": binding_identity(
                    make_xiaomi_policy_binding()
                ),
                "planning_profile": planning_profile,
                "canonical_task_input": True,
                "instruction_source": "environment_language",
                "forward_step_guidance": not getattr(args, "no_forward_step_guidance", False),
                "observation_history": XIAOMI_MIBOT_POLICY_SPEC.observation_history,
                "observation_interval": XIAOMI_MIBOT_POLICY_SPEC.observation_interval,
                "crop_ratio": XIAOMI_MIBOT_POLICY_SPEC.crop_ratio,
                "processor_robot_type": checkpoint.robot_type,
                "generated_chunk_size": XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size,
                "executed_actions_per_request": int(
                    getattr(args, "actions_per_policy_request", None) or XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request
                ),
                "native_action_shape": list(XIAOMI_MIBOT_POLICY_SPEC.native_action_shape),
                "decode_order": XIAOMI_MIBOT_POLICY_SPEC.decode_order,
                "input_camera_ids": list(XIAOMI_MIBOT_POLICY_SPEC.input_camera_ids),
            }
            if use_xiaomi
            else {
                "protocol_version": 1, "policy_family": "pi05",
                "checkpoint": checkpoint.to_payload(), "openpi_commit": OPENPI_COMMIT,
                "config_name": PI05_CONFIG, "binding_digest": binding_identity(binding),
                "planning_profile": planning_profile,
                "noise_seed_policy": "case_seed", "policy_method": "infer",
                "checkpoint_chunk_size": 50,
                "executed_actions_per_request": args.actions_per_policy_request,
                "instruction_source": "environment_language",
                "input_camera_ids": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            } if use_pi05 else {
            "protocol_version": 1,
            "lerobot_revision": SMOLVLA_LEROBOT_REVISION,
            "checkpoint": checkpoint.to_payload(),
            "processor_digests": dict(SMOLVLA_PROCESSOR_DIGESTS["robocasa"]),
            "binding_digest": binding_identity(binding),
            "planning_profile": planning_profile,
            "noise_mode": "episode_generator",
            "noise_seed_policy": "case_seed",
            "policy_method": "predict_action_chunk",
            "checkpoint_chunk_size": CHECKPOINT_CHUNK_SIZE,
            "checkpoint_n_action_steps": CHECKPOINT_N_ACTION_STEPS,
            "executed_actions_per_request": int(
                getattr(
                    args,
                    "actions_per_policy_request",
                    DEFAULT_ACTIONS_PER_POLICY_REQUEST,
                )
            ),
            "input_camera_ids": [
                feature.native_key
                for feature in binding.inputs
                if feature.policy_key.startswith("observation.images.")
            ],
            }
        ),
        contracts={
            "data_schema": robocasa_data_schema(
                RoboCasaDisplayAdapter().descriptors
            ).to_payload(),
            "control_hz": 20,
            "images": {
                "count": len(CAMERA_SHAPES),
                "dtype": "uint8",
                "shapes": {
                    camera_id: list(shape)
                    for camera_id, shape in CAMERA_SHAPES.items()
                },
            },
            "state": {"shape": [16], "dtype": "float32"},
            "action": {
                "shape": [12],
                "dtype": "float32",
                "range": [-1, 1],
            },
            "horizon_rule": (
                "upstream_horizon"
                if profile.official
                else "upstream_horizon_or_explicit_min"
            ),
            "success_source": (
                profile.success_source
                if profile.official
                else "robocasa_task_predicate"
            ),
        },
        runtime={
            "main_interpreter": sys.executable,
            "policy_interpreter": (
                None
                if args.policy_endpoint is not None
                else (
                    os.environ.get("RPENT_XIAOMI_MIBOT_PYTHON")
                    if use_xiaomi
                    else os.environ.get("RPENT_OPENPI_PYTHON") if use_pi05
                    else os.environ.get("RPENT_SMOLVLA_PYTHON") or sys.executable
                )
            ),
            **(
                {
                    "policy_snapshot_path": os.environ.get(
                        "RPENT_XIAOMI_MIBOT_SNAPSHOT"
                    ),
                    "policy_dtype": "bfloat16",
                }
                if use_xiaomi
                else {"policy_checkpoint_path": os.environ.get("RPENT_ROBOCASA_PI05_CHECKPOINT"),
                      "policy_dtype": "bfloat16"} if use_pi05 else {}
            ),
            "simulator_interpreter": (
                None
                if args.env_endpoint is not None
                else os.environ.get("RPENT_ROBOCASA_PYTHON")
            ),
            "simulator_endpoint": args.env_endpoint,
            "rendering_backend": os.environ.get("MUJOCO_GL", "egl"),
            "evaluation_root": str(run_config.output_dir),
            "requested_safety_ceiling": run_config.safety_ceiling,
            "policy_endpoint": args.policy_endpoint,
            "policy_endpoint_argument": "--policy-endpoint",
            "shared_policy_required": True,
            **(
                {
                    "grouped_worker_module": "robots.robocasa.competition_worker",
                    "qualification_module": "robots.robocasa.qualification",
                }
                if profile.official
                else {}
            ),
        },
        schedule=(
            _official_schedule(
                profile,
                next(iter(digests)),
                registry_snapshot,
                planning_profile,
                family,
            )
            if profile.official
            else (
                {"actions_per_policy_request": XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request}
                if use_xiaomi
                else {}
            )
        ),
        qualification=(
            _official_qualification(profile, submission_metadata)
            if profile.official
            else {}
        ),
    )


def _validate_official_inputs(args, profile) -> None:
    canonical_selector = ",".join(profile.task_names)
    task = getattr(args, "task", None)
    if task is not None and str(task).strip() not in {"", canonical_selector}:
        raise ValueError("official evaluation profile owns task selector")
    if getattr(args, "seed", None) not in {None, 0, profile.scenario_seed}:
        raise ValueError("official evaluation profile owns seed")
    if getattr(args, "episodes", None) not in {None, profile.rollouts_per_task}:
        raise ValueError("official evaluation profile owns episodes")
    if getattr(args, "scene_split", None) not in {None, profile.scene_split}:
        raise ValueError("official evaluation profile owns scene_split")
    if getattr(args, "max_episode_steps", None) is not None:
        raise ValueError("official evaluation profile owns max_episode_steps")


def _official_schedule(
    profile,
    registry_digest: str,
    registry_snapshot,
    planning_profile: str,
    family: str,
) -> dict:
    task_set_sizes = {
        name: len(registry_snapshot["groups"][name]["tasks"])
        for name in profile.task_sets
    }
    task_set_membership = {
        name: tuple(registry_snapshot["groups"][name]["tasks"])
        for name in profile.task_sets
    }
    canonical_selector = ",".join(profile.task_names)
    locked_inputs = {
        "evaluation_profile": profile.profile_id,
        "task": canonical_selector,
        "seed": profile.scenario_seed,
        "episodes": profile.rollouts_per_task,
        "scene_split": profile.scene_split,
        "max_episode_steps": None,
    }
    return {
        "evaluation_profile": profile.profile_id,
        "episodes_per_task": profile.rollouts_per_task,
        "base_seed": profile.scenario_seed,
        "seed_policy": "shared_environment_sequence",
        "ordering": "task_registry_then_episode_index",
        "task_count": len(profile.task_names),
        "task_sets": profile.task_sets,
        "task_set_sizes": task_set_sizes,
        "task_set_membership": task_set_membership,
        "task_registry_digest": registry_digest,
        "scene_split": profile.scene_split,
        "scenario_sampling": profile.scenario_sampling,
        "worker_grouping": profile.worker_grouping,
        "success_source": profile.success_source,
        "horizon_source": profile.horizon_source,
        "locked_inputs": locked_inputs,
        "effective_inputs": {
            **locked_inputs,
            "planning_profile": planning_profile,
            "task_count": len(profile.task_names),
            "case_count": len(profile.task_names) * profile.rollouts_per_task,
            "batch_count": len(profile.task_names),
            "worker_grouping": profile.worker_grouping,
            "scenario_sampling": profile.scenario_sampling,
            "success_source": profile.success_source,
            "horizon_source": profile.horizon_source,
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
        },
        **(
            {"actions_per_policy_request": XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request}
            if family == "xiaomi-mibot"
            else {}
        ),
    }


def _official_qualification(profile, metadata: SubmissionMetadata | None) -> dict:
    return {
        "official_profile": profile.profile_id,
        "require_exact_task_membership": True,
        "require_complete_task_batches": True,
        "require_single_task_batch_attempt": True,
        "require_workload_confirmation": True,
        "require_pinned_dependencies": True,
        "require_fixed_policy_identity": True,
        "require_native_success": True,
        "require_upstream_horizons": True,
        "allow_selective_case_rerun": False,
        "require_complete_artifacts": True,
        **(
            {"submission_metadata": metadata.to_payload()}
            if metadata is not None
            else {}
        ),
    }


def build_policy_daemon(*, args, output_dir):
    if resolve_policy_family(args) == "pi05":
        from rpent.policies.robocasa_pi05_daemon import OpenPiRoboCasaDaemon

        if getattr(args, "jobs", 1) > 1:
            from rpent.evaluation.parallel import Pi05ReplicaPool

            devices = tuple(int(item.strip()) for item in args.policy_devices.split(","))
            return Pi05ReplicaPool(
                jobs=args.jobs, devices=devices, output_dir=output_dir,
                seed=int(args.seed),
                replica_memory_mib=args.pi05_replica_memory_mib,
                reserve_mib=args.gpu_reserve_mib,
                sessions_per_replica=args.pi05_sessions_per_replica,
            )
        return OpenPiRoboCasaDaemon(
            endpoint=args.policy_endpoint, device=args.policy_device, seed=int(args.seed),
            log_path=Path(output_dir) / "pi05_policy_server.log",
        )
    if resolve_policy_family(args) == "xiaomi-mibot":
        from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon

        if getattr(args, "jobs", 1) > 1:
            from rpent.evaluation.parallel import XiaomiReplicaPool

            return XiaomiReplicaPool(
                jobs=args.jobs,
                devices=tuple(int(item.strip()) for item in args.policy_devices.split(",")),
                output_dir=output_dir,
                dtype=args.policy_dtype or XIAOMI_MIBOT_POLICY_SPEC.default_dtype,
                replica_memory_mib=args.xiaomi_replica_memory_mib,
                reserve_mib=args.gpu_reserve_mib,
            )

        snapshot = os.environ.get("RPENT_XIAOMI_MIBOT_SNAPSHOT")
        if not snapshot:
            raise RuntimeError(
                "RPENT_XIAOMI_MIBOT_SNAPSHOT must point to the pinned snapshot"
            )
        attach_loaded = os.environ.get("RPENT_SHARED_POLICY_LOADED") == "1"
        expected_artifact_identity = None
        if attach_loaded:
            from rpent.policies.xiaomi_mibot_artifacts import inspect_xiaomi_artifacts

            expected_artifact_identity = inspect_xiaomi_artifacts(
                snapshot, XIAOMI_MIBOT_CHECKPOINT
            ).identity_digest
        return XiaomiMiBoTDaemon(
            checkpoint=XIAOMI_MIBOT_CHECKPOINT,
            snapshot_path=snapshot,
            binding_identity=binding_identity(make_xiaomi_policy_binding()),
            device=args.policy_device,
            dtype=getattr(args, "policy_dtype", None) or XIAOMI_MIBOT_POLICY_SPEC.default_dtype,
            endpoint=args.policy_endpoint,
            log_path=Path(output_dir) / "xiaomi_policy_server.log",
            attach_loaded=attach_loaded,
            expected_artifact_identity=expected_artifact_identity,
        )

    from rpent.policies.smolvla_daemon import SmolVLADaemon

    return SmolVLADaemon(
        binding=make_robocasa_binding(),
        checkpoint=SMOLVLA_CHECKPOINTS["robocasa"],
        device=args.policy_device,
        dtype=args.policy_dtype,
        noise_seed=args.seed,
        endpoint=args.policy_endpoint,
        log_path=Path(output_dir) / "smolvla_policy_server.log",
    )


def resolve_planning_profile(args) -> str:
    from robots.robocasa.policy_selection import resolve_policy_options

    resolve_policy_options(args)
    profile_id = resolve_required_planning_profile(
        getattr(args, "planning_profile", None) or ROBOCASA_HORIZON_PLANNING_PROFILE
    )
    if get_planning_profile(profile_id).environment_horizon_only:
        from rpent.planning import resolve_planning_stack
        from rpent.planning.stack import resolve_planner_turn_limit

        resolve_planning_stack(
            profile_id,
            harness_mode=getattr(args, "harness_mode", None),
            max_replans=getattr(args, "max_replans", None),
        )
        resolve_planner_turn_limit(profile_id, getattr(args, "max_turns", None))
    return profile_id


__all__ = [
    "build_policy_daemon",
    "prepare_evaluation",
    "resolve_planning_profile",
]
