"""RoboCasa365 RPent preset with an isolated direct-upstream simulator."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robots.robocasa.config import RoboCasaConfig
    from rpent.envs.env_spec import EnvSpec


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def get_env_spec() -> EnvSpec:
    from robots.robocasa.prompts.system import system_prompt, user_prompt
    from rpent.envs.env_spec import EnvSpec
    from rpent.envs.prompt_bundle import PromptBundle

    return EnvSpec(
        name="robocasa",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        build_episode_bootstrap=_build_episode_bootstrap,
        prepare_evaluation=_prepare_evaluation,
        build_policy_daemon=_build_policy_daemon,
        resolve_planning_profile=_resolve_planning_profile,
        data_schema=_data_schema,
        coordinate_evaluation_by_default=True,
        launch_preset=_launch_preset,
        validate_evaluation_launch=_validate_evaluation_launch,
        evaluation_runners={"vla-only": _run_vla_only_case},
    )


def _run_vla_only_case(case, manifest, artifact_dir):
    from robots.robocasa.vla_runner import run_vla_only_case

    return run_vla_only_case(case, manifest, artifact_dir)


def _data_schema(cameras):
    from robots.robocasa.data_schema import robocasa_data_schema

    return robocasa_data_schema(cameras)


def _launch_preset():
    from robots.robocasa.launch import build_launch_preset

    return build_launch_preset()


def _validate_evaluation_launch(args: argparse.Namespace) -> None:
    from robots.robocasa.config import parse_object_registries
    from robots.robocasa.object_registry_preflight import (
        inspect_object_registry_coverage,
    )
    from robots.robocasa.tasks import resolve_robocasa_tasks
    from rpent.envs.launch import LaunchValidationError

    resolve_robocasa_tasks(str(args.task))
    coverage = inspect_object_registry_coverage(
        None,
        selector=str(args.task),
        selected_registries=parse_object_registries(args.object_registries),
        seed=int(args.seed),
    )
    if not coverage.gaps:
        return
    missing = " ".join(
        f"{gap.task} is missing object groups: {','.join(gap.object_groups)}."
        for gap in coverage.gaps
    )
    recommended = ",".join(coverage.recommended_registries)
    raise LaunchValidationError(
        {
            "object_registries": (
                f"{missing} Recommended object registries: {recommended}."
            )
        }
    )


def _prepare_evaluation(args, run_config, resolved_tasks=None):
    from robots.robocasa.evaluation import prepare_evaluation

    return prepare_evaluation(args, run_config, resolved_tasks)


def _build_policy_daemon(args, output_dir):
    from robots.robocasa.evaluation import build_policy_daemon

    return build_policy_daemon(args=args, output_dir=output_dir)


def _resolve_planning_profile(args):
    from robots.robocasa.evaluation import resolve_planning_profile

    return resolve_planning_profile(args)


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    from robots.robocasa.competition import (
        DEVELOPMENT_PROFILE_ID,
        LEADERBOARD_PROFILE_ID,
    )
    from rpent.policies.robocasa_pi05_catalog import PI05_CHECKPOINT
    from rpent.policies.smolvla_catalog import SMOLVLA_CHECKPOINTS
    from rpent.policies.xiaomi_mibot_catalog import (
        XIAOMI_MIBOT_CHECKPOINT,
        XIAOMI_REPLICA_MEMORY_MIB,
    )

    del use_dashboard
    parser.add_argument(
        "--evaluation-profile",
        choices=(DEVELOPMENT_PROFILE_ID, LEADERBOARD_PROFILE_ID),
        default=DEVELOPMENT_PROFILE_ID,
    )
    parser.add_argument("--task")
    parser.add_argument("--vla", choices=("smolvla", "pi05", "xiaomi-mibot"), default=None,
                        help="VLA policy, independent of planning profile (default: smolvla).")
    parser.add_argument("--no-forward-step-guidance", action="store_true",
                        help="Disable step guidance for pi05 and xiaomi-mibot.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=_positive_integer, default=None)
    parser.add_argument(
        "--actions-per-policy-request",
        type=_positive_integer,
        default=None,
    )
    parser.add_argument(
        "--object-registries",
        "--env.obj_registries",
        dest="object_registries",
        default="lightwheel,objaverse",
        help=(
            "Comma list or [objaverse,lightwheel]; "
            "default: lightwheel,objaverse."
        ),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--policy-endpoint", default=None)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--policy-devices", default=None,
                        help="Physical GPU indices for concurrent Pi0.5 or Xiaomi, e.g. 0,1.")
    parser.add_argument("--pi05-replica-memory-mib", type=int, default=11264,
                        help="Admission budget per Pi0.5 replica plus simulator.")
    parser.add_argument("--xiaomi-replica-memory-mib", type=int, default=XIAOMI_REPLICA_MEMORY_MIB,
                        help="Admission budget per Xiaomi replica plus one simulator.")
    parser.add_argument("--pi05-sessions-per-replica", type=int, default=2,
                        help="Independent episode sessions sharing one Pi0.5 model (default: 2).")
    parser.add_argument("--gpu-reserve-mib", type=int, default=1536,
                        help="Free GPU memory reserved outside replica budgets.")
    parser.add_argument("--policy-dtype", default=None)
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        default=None,
        choices=[
            SMOLVLA_CHECKPOINTS["robocasa"].repo_id,
            XIAOMI_MIBOT_CHECKPOINT.repo_id,
            PI05_CHECKPOINT.repo_id,
        ],
    )
    parser.add_argument("--submission-metadata-json", default=None)
    parser.add_argument("--no-transition-takeover", action="store_true",
                        help="Disable transition steps and their prompt/skill sections.")
    parser.add_argument("--no-op-takeover", action="store_true",
                        help="Disable takeover steps and their prompt/skill sections.")


def _parse_config(args: argparse.Namespace) -> RoboCasaConfig:
    from robots.robocasa.acceptance_skills import load_acceptance_skill
    from robots.robocasa.competition import resolve_evaluation_profile
    from robots.robocasa.config import RoboCasaConfig, parse_object_registries
    from robots.robocasa.control_ablation import (
        control_metadata,
        control_options,
        prompt_ablation,
    )
    from robots.robocasa.policy_selection import (
        policy_checkpoint,
        resolve_policy_options,
    )
    from robots.robocasa.policy_skills import load_policy_skill, strip_sections
    from robots.robocasa.prompts.system import task_rules
    from robots.robocasa.tasks import packaged_registry_snapshot
    from rpent.utils.config import get_repo_root

    profile = resolve_evaluation_profile(
        getattr(args, "evaluation_profile", None), packaged_registry_snapshot()
    )
    metadata_path = getattr(args, "submission_metadata_json", None)
    if metadata_path is not None:
        from robots.robocasa.submission import SubmissionMetadata

        SubmissionMetadata.from_json_path(metadata_path)
    submitted_task = str(args.task).strip() if args.task is not None else ""
    if profile.official:
        if any(
            bool(getattr(args, name, False)) for name in ("interactive", "dashboard")
        ):
            raise ValueError(
                "official evaluation profile requires coordinated evaluation; "
                "interactive and dashboard foreground modes are unsupported"
            )
        canonical_task = ",".join(profile.task_names)
        if submitted_task and submitted_task != canonical_task:
            raise ValueError("official evaluation profile owns task selector")
        if args.seed not in {None, profile.scenario_seed}:
            raise ValueError("official evaluation profile owns seed")
        if getattr(args, "episodes", None) not in {
            None,
            profile.rollouts_per_task,
        }:
            raise ValueError("official evaluation profile owns episodes")
        if args.max_episode_steps is not None:
            raise ValueError("official evaluation profile owns max_episode_steps")
        task = canonical_task
        args.task = task
        args.seed = profile.scenario_seed
        args.episodes = profile.rollouts_per_task
    else:
        task = submitted_task
        if not task:
            raise ValueError("--task is required")
        if args.seed is None:
            args.seed = 0
    registries = parse_object_registries(args.object_registries)
    family = resolve_policy_options(args)
    checkpoint = policy_checkpoint(family)
    recipe_name = profile.profile_id if profile.official else task.replace(",", "_")
    recipe_tag = f"robocasa_{recipe_name}_s{args.seed}"
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{recipe_tag}"
    controls = control_options(args)
    headings = [section.title() for section in sorted(prompt_ablation(controls))]
    skill = load_policy_skill(family)
    if headings:
        skill = strip_sections(skill, headings)
    acceptance = load_acceptance_skill(task)
    prompt_vars = {
        **controls,
        "task": task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
        "task_rules": task_rules(task, policy_family=family) + (
            "\n\n" + acceptance["text"] if acceptance["text"] else ""
        ),
        "policy_skill": skill,
    }
    return RoboCasaConfig(
        task=task,
        vla=family,
        seed=int(args.seed),
        checkpoint=checkpoint,
        object_registries=registries,
        env_endpoint=args.env_endpoint,
        policy_endpoint=args.policy_endpoint,
        policy_device=args.policy_device,
        policy_dtype=args.policy_dtype,
        safety_ceiling=(
            None if args.max_episode_steps is None else int(args.max_episode_steps)
        ),
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars=prompt_vars,
        dashboard_state=None,
        task_desc={
            "environment": "robocasa",
            "task": task,
            "seed": int(args.seed),
            "object_registries": list(registries),
            "execution_mode": "rpent_harness",
            "control_modes": control_metadata(family, skill, controls),
            "acceptance_skill": acceptance["metadata"],
        },
    )


def _init_runtime(
    args: argparse.Namespace, output_dir: Path
) -> tuple[list[Any], dict[str, Any]]:
    del args, output_dir
    return [], {}


def _build_episode_bootstrap(
    args: argparse.Namespace, output_dir: Path, dashboard: Any
):
    from robots.robocasa.config import parse_object_registries
    from robots.robocasa.runtime import build_episode_bootstrap

    args.object_registries = parse_object_registries(args.object_registries)
    return build_episode_bootstrap(args, output_dir, dashboard)


__all__ = ["get_env_spec"]
