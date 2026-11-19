"""Compose one RoboCasa365 case from RPent-owned runtime components."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from robots.robocasa.dependencies import ROBOSUITE_COMMIT
from robots.robocasa.display import RoboCasaDisplayAdapter
from robots.robocasa.observation import OBSERVATION_SCHEMA
from robots.robocasa.policy_selection import (
    policy_binding,
    policy_checkpoint,
    resolve_policy_family,
)
from robots.robocasa.tasks import resolve_robocasa_tasks
from rpent.episode.runtime import EpisodeBootstrap
from rpent.planning import (
    ROBOCASA_HORIZON_PLANNING_PROFILE,
    get_planning_profile,
)
from rpent.policies.lerobot_smolvla_artifacts import binding_identity
from rpent.policies.smolvla_catalog import (
    SMOLVLA_CHECKPOINTS,
    SMOLVLA_PROCESSOR_DIGESTS,
)
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_MIBOT_POLICY_SPEC,
)


def _shared_environment_endpoint() -> str | None:
    endpoint = os.environ.get("RPENT_SHARED_ENV_ENDPOINT")
    if endpoint is None:
        return None
    match = re.fullmatch(
        r"(?:(?:http://(?:127\.0\.0\.1|localhost|\[::1\]))|"
        r"(?:socket://(?:127\.0\.0\.1|localhost))):"
        r"([1-9][0-9]{0,4})",
        endpoint,
    )
    if match is None or int(match.group(1)) > 65535:
        raise ValueError(
            "RPENT_SHARED_ENV_ENDPOINT must be an exact lowercase loopback "
            "http:// or socket:// origin with a canonical explicit port"
        )
    return endpoint


def _shared_environment_scene_split() -> str | None:
    split = os.environ.get("RPENT_SHARED_ENV_SCENE_SPLIT")
    if split is None:
        return None
    if split not in {"target", "pretrain"}:
        raise ValueError(
            "RPENT_SHARED_ENV_SCENE_SPLIT must be exactly 'target' or 'pretrain'"
        )
    if _shared_environment_endpoint() is None:
        raise ValueError(
            "RPENT_SHARED_ENV_SCENE_SPLIT requires RPENT_SHARED_ENV_ENDPOINT"
        )
    return split


def _default_environment_builder(
    *,
    args,
    task,
    horizon: int,
    output_dir: Path,
    daemon_factory: Callable[..., Any] | None = None,
):
    if daemon_factory is None:
        from robots.robocasa.daemon import RoboCasaDaemon

        daemon_factory = RoboCasaDaemon
    shared_endpoint = _shared_environment_endpoint()
    configured_endpoint = args.env_endpoint
    if (
        shared_endpoint is not None
        and configured_endpoint is not None
        and configured_endpoint != shared_endpoint
    ):
        raise ValueError(
            "RPENT_SHARED_ENV_ENDPOINT conflicts with the configured environment "
            "endpoint"
        )
    endpoint = shared_endpoint or configured_endpoint
    registries = tuple(args.object_registries)
    config = {
        "task": task.name,
        "split": task.split,
        "seed": int(args.seed),
        "robot": "PandaOmron",
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_size": 256,
        "object_registries": list(registries),
        "rendering_mode": "headless",
        "upstream_horizon": int(task.upstream_horizon),
        "safety_ceiling": int(horizon),
        "task_registry_commit": task.registry_commit,
        "task_registry_digest": task.registry_digest,
    }
    daemon = daemon_factory(
        config=config,
        endpoint=endpoint,
        log_path=output_dir / "robocasa_env_server.log",
        config_path=output_dir / "robocasa_env_config.json",
        formal=True,
        close_remote=shared_endpoint is None,
    )
    return (
        daemon,
        daemon.start,
        {
            "endpoint": endpoint,
            "effective_horizon": horizon,
            "object_registries": list(registries),
        },
    )


def _default_policy_daemon_builder(*, args, binding, output_dir: Path):
    if resolve_policy_family(args) == "pi05":
        from rpent.policies.robocasa_pi05_daemon import OpenPiRoboCasaDaemon

        return OpenPiRoboCasaDaemon(
            endpoint=args.policy_endpoint, device=args.policy_device, seed=int(args.seed),
            log_path=output_dir / "pi05_policy_server.log",
        )
    if resolve_policy_family(args) == "xiaomi-mibot":
        from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon

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
            binding_identity=binding_identity(binding),
            device=args.policy_device,
            dtype=getattr(args, "policy_dtype", None) or XIAOMI_MIBOT_POLICY_SPEC.default_dtype,
            endpoint=args.policy_endpoint,
            log_path=output_dir / "xiaomi_policy_server.log",
            attach_loaded=attach_loaded,
            expected_artifact_identity=expected_artifact_identity,
        )

    from rpent.policies.smolvla_daemon import SmolVLADaemon

    return SmolVLADaemon(
        binding=binding,
        checkpoint=SMOLVLA_CHECKPOINTS["robocasa"],
        device=args.policy_device,
        dtype=args.policy_dtype,
        noise_seed=int(args.seed),
        endpoint=args.policy_endpoint,
        log_path=output_dir / "smolvla_policy_server.log",
        attach_loaded=(os.environ.get("RPENT_SHARED_POLICY_LOADED") == "1"),
    )


def _default_toolkit_builder(
    controller,
    observation,
    *,
    instruction,
    policy_instruction,
    forward_step_guidance,
    actions_per_policy_request,
    agent_artifact_root,
    dashboard,
    perception_source=None,
    episode_horizon=None,
):
    from robots.robocasa.toolkit import RoboCasaToolkit

    return RoboCasaToolkit(
        controller,
        initial_observation=dict(observation),
        instruction=instruction,
        policy_instruction=policy_instruction,
        forward_step_guidance=forward_step_guidance,
        actions_per_policy_request=actions_per_policy_request,
        perception_source=perception_source,
        episode_horizon=episode_horizon,
        agent_artifact_root=agent_artifact_root,
        dashboard=dashboard,
    )


def _effective_horizon(
    upstream_horizon: int, ceiling: int | None, *, horizon_only: bool = False
) -> int:
    upstream = int(upstream_horizon)
    if horizon_only and ceiling is not None and int(ceiling) < upstream:
        raise ValueError(
            "environment-horizon profile requires the full upstream horizon; "
            "remove --max-episode-steps or choose a bounded planning profile"
        )
    return upstream if ceiling is None else min(upstream, int(ceiling))


def build_episode_bootstrap(
    args,
    output_dir: Path,
    dashboard: Any,
    *,
    task_resolver: Callable[..., Any] = resolve_robocasa_tasks,
    environment_builder: Callable[..., Any] = _default_environment_builder,
    policy_daemon_builder: Callable[..., Any] = _default_policy_daemon_builder,
    toolkit_builder: Callable[..., Any] = _default_toolkit_builder,
) -> EpisodeBootstrap:
    shared_scene_split = _shared_environment_scene_split()
    tasks = tuple(
        task_resolver(str(args.task), scene_split=shared_scene_split)
        if shared_scene_split is not None
        else task_resolver(str(args.task))
    )
    if len(tasks) != 1:
        raise ValueError(
            "one RoboCasa worker requires exactly one canonical task; "
            "use the evaluation coordinator for lists or groups"
        )
    task = tasks[0]
    planning_profile = get_planning_profile(
        getattr(args, "planning_profile", None) or ROBOCASA_HORIZON_PLANNING_PROFILE
    )
    horizon = _effective_horizon(
        int(task.upstream_horizon),
        args.max_episode_steps,
        horizon_only=planning_profile.environment_horizon_only,
    )
    if horizon <= 0:
        raise ValueError("effective RoboCasa horizon must be positive")
    family = resolve_policy_family(args)
    use_xiaomi = family == "xiaomi-mibot"
    use_pi05 = family == "pi05"
    from robots.robocasa.control_modes import LANGUAGE_POLICY_FAMILIES

    language_policy = family in LANGUAGE_POLICY_FAMILIES
    forward_guidance = language_policy and not getattr(args, "no_forward_step_guidance", False)
    binding = policy_binding(family)
    env_dependency, simulator_factory, environment_meta = environment_builder(
        args=args,
        task=task,
        horizon=horizon,
        output_dir=Path(output_dir),
    )
    policy_daemon = policy_daemon_builder(
        args=args,
        binding=binding,
        output_dir=Path(output_dir),
    )

    def policy_factory():
        session = policy_daemon.start()
        if use_xiaomi:
            health = dict(policy_daemon.check_health())
            bootstrap.runtime_metadata.update(
                policy_health=health,
                artifact_identity=health["artifact_identity"],
                remote_code_revision=health["remote_code_revision"],
                server_version=health["server_version"],
            )
        return session

    def make_toolkit(controller, observation):
        simulator = simulator_factory()
        instruction = getattr(simulator, "instruction", task.instruction)
        if language_policy:
            bootstrap.runtime_metadata["policy_instruction"] = instruction
        return toolkit_builder(
            controller,
            observation,
            instruction=instruction,
            policy_instruction=instruction if language_policy else task.name,
            forward_step_guidance=forward_guidance,
            perception_source=simulator,
            episode_horizon=simulator.effective_horizon,
            actions_per_policy_request=args.actions_per_policy_request,
            agent_artifact_root=Path(output_dir) / "agent_artifacts",
            dashboard=dashboard,
        )

    checkpoint = policy_checkpoint(family)
    dependencies = tuple(
        dependency
        for dependency in (env_dependency, policy_daemon)
        if dependency is not None
    )
    bootstrap = EpisodeBootstrap(
        preset_name=(
            "robocasa365-pandaomron-pi05-v1" if use_pi05 else
            XIAOMI_MIBOT_POLICY_SPEC.preset_name
            if use_xiaomi
            else "robocasa365-pandaomron-smolvla-v1"
        ),
        task=task.name,
        seed=int(args.seed),
        instruction=task.instruction,
        simulator_factory=simulator_factory,
        policy_factory=policy_factory,
        binding=binding,
        toolkit_factory=make_toolkit,
        display_adapter=RoboCasaDisplayAdapter(),
        dependencies=dependencies,
        runtime_metadata={
            "environment": dict(environment_meta),
            "execution_mode": "rpent_harness",
            "task": task.name,
            "policy_instruction": task.instruction if language_policy else task.name,
            "policy_instruction_source": "environment_language" if language_policy else "training_task_id",
            "forward_step_guidance": forward_guidance,
            "family": task.family,
            "category": task.category,
            "split": task.split,
            "upstream_horizon": int(task.upstream_horizon),
            "requested_safety_ceiling": args.max_episode_steps,
            "effective_horizon": horizon,
            "seed": int(args.seed),
            "object_registries": list(args.object_registries),
            "simulator_commit": task.registry_commit,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "task_registry_digest": task.registry_digest,
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": "robocasa-pandaomron-action-v1",
            "checkpoint": checkpoint.to_payload(),
            **(
                {
                    "policy_family": "xiaomi-mibot",
                    "preset_name": XIAOMI_MIBOT_POLICY_SPEC.preset_name,
                    "generated_chunk_size": XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size,
                    "native_action_shape": list(XIAOMI_MIBOT_POLICY_SPEC.native_action_shape),
                    "input_camera_ids": list(XIAOMI_MIBOT_POLICY_SPEC.input_camera_ids),
                    "observation_history": XIAOMI_MIBOT_POLICY_SPEC.observation_history,
                    "observation_interval": XIAOMI_MIBOT_POLICY_SPEC.observation_interval,
                    "crop_ratio": XIAOMI_MIBOT_POLICY_SPEC.crop_ratio,
                }
                if use_xiaomi
                else {"policy_family": "pi05"} if use_pi05 else {
                    "processor_digests": dict(
                        SMOLVLA_PROCESSOR_DIGESTS["robocasa"]
                    )
                }
            ),
            "binding_identity": binding_identity(binding),
        },
    )

    return bootstrap


__all__ = ["build_episode_bootstrap"]
