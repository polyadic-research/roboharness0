"""Product launch preset for the built-in RoboCasa365 adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from robots.robocasa.binding import DEFAULT_ACTIONS_PER_POLICY_REQUEST
from robots.robocasa.competition import (
    DEVELOPMENT_PROFILE_ID,
    LEADERBOARD_PROFILE_ID,
    resolve_evaluation_profile,
)
from robots.robocasa.config import OBJECT_REGISTRIES
from robots.robocasa.control_modes import policy_display_name
from robots.robocasa.tasks import packaged_registry_snapshot, task_selector_suggestions
from rpent.envs.launch import (
    BUILTIN_PLANNER_MODEL_SUGGESTIONS,
    LaunchField,
    LaunchGroup,
    LaunchOwnership,
    LaunchPrerequisite,
    LaunchPreset,
    LaunchValidationError,
    validate_api_planner_model,
)
from rpent.envs.prerequisites import planner_prerequisite, runtime_prerequisite
from rpent.planning import (
    ROBOCASA_HORIZON_PLANNING_PROFILE,
    ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE,
    get_planning_profile,
    list_planning_profile_ids,
)
from rpent.policies.robocasa_pi05_catalog import PI05_CHECKPOINT
from rpent.policies.smolvla_catalog import SMOLVLA_CHECKPOINTS
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_MIBOT_POLICY_SPEC,
)

_HYBRID = ("hybrid",)

_HELP = {
    "vla": "Selects the action model independently of the planning profile.",
    "evaluation_profile": "Selects a flexible development run or the locked RoboCasa365 leaderboard protocol.",
    "task": "Accepts one packaged group shortcut or comma-separated canonical RoboCasa task names.",
    "seed": "Sets the deterministic environment reset seed.",
    "episodes": "Creates this many Cases for each resolved task with derived seeds.",
    "max_episode_steps": "Optionally overrides each Case's upstream low-level environment step ceiling.",
    "actions_per_policy_request": "Caps the low-level actions executed before taking a fresh observation.",
    "planner": "Selects the Hybrid planning implementation.",
    "model": "Selects the planning model; API requires provider:model while Claude Code and Codex may use their backend default.",
    "planning_profile": "Selects the plan, verification, and replanning behavior profile.",
    "max_turns": "Caps planner decision and tool-call rounds independently of other budgets.",
    "max_tokens": "Caps tokens available to one planner request; 0 explicitly omits the API token cap.",
    "worker_timeout_s": (
        "Optional worker deadline in seconds. Leave blank for 30 minutes per "
        "750 official task steps, rounded up."
    ),
    "env_endpoint": "Uses an existing RoboCasa environment service URL instead of starting one locally.",
    "policy_endpoint": "Uses an existing policy service URL instead of starting one locally.",
    "policy_device": "Selects the accelerator device used for local policy inference.",
    "policy_dtype": "Selects numeric precision for local policy inference and memory use.",
    "policy_path": "Selects the immutable policy checkpoint used for the run.",
    "reuse_policy_service": "Keeps one compatible SmolVLA model loaded for later RoboCasa Evaluations.",
    "object_registries": "Selects comma-separated RoboCasa object registry plugins loaded by the environment.",
    "instruction": "Overrides the opening planner instruction without changing the selected scene or success predicate; one override applies to every selected task.",
    "base_url": "Optionally overrides the selected planner backend's API base URL.",
    "harness_mode": "Optionally overrides the planning profile's execution harness.",
    "max_replans": "Caps accepted structured replans during one run.",
    "use_chines": "Requests Simplified Chinese planning, verification, and replan text.",
    "no_images": "Runs the planner without image attachments for text-only models.",
    "planner_timeout_s": "Caps the complete planner subprocess wall-clock time; displayed in minutes.",
    "plan_generation_timeout_s": "Caps one plan or replan model request in seconds.",
    "claude_code_max_budget_usd": "Optionally caps Claude Code planner spend in US dollars.",
    "recording_mode": "By default, retains only the 256x256 images supplied to the LLM. This setting controls additional simulator camera streams.",
    "recording_camera": "Lists camera identifiers retained when the recording profile is selected.",
    "require_complete_artifacts": "Fails completion when required recordings or artifacts are missing.",
    "submission_metadata_json": "Loads bounded official leaderboard metadata from a JSON file.",
}


def _field(
    name: str, label: str, field_type: str, group: str, order: int, **kwargs: object
) -> LaunchField:
    return LaunchField(
        name, label, field_type, group, order, description=_HELP[name], **kwargs
    )


def build_launch_preset() -> LaunchPreset:
    checkpoint = SMOLVLA_CHECKPOINTS["robocasa"].repo_id
    official_owned = LaunchOwnership("evaluation_profile", (LEADERBOARD_PROFILE_ID,))
    return LaunchPreset(
        version=14,
        default_mode="hybrid",
        modes=_HYBRID,
        groups=(
            LaunchGroup("basic", "Basic configuration", 10),
            LaunchGroup("tuning", "Common tuning", 20),
            LaunchGroup("advanced", "Advanced", 30, collapsed=True),
        ),
        fields=(
            _field("vla", "VLA model", "string", "basic", 7,
                   default="smolvla", choices=("smolvla", "pi05", "xiaomi-mibot"),
                   choice_labels=("SmolVLA", "π0.5 (OpenPI)", policy_display_name("xiaomi-mibot")),
                   visible_when=LaunchOwnership("planning_profile", tuple(
                       p for p in list_planning_profile_ids(include_aliases=True)
                       if p != ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE))),
            _field(
                "evaluation_profile",
                "Workload profile",
                "string",
                "basic",
                5,
                default=DEVELOPMENT_PROFILE_ID,
                choices=(DEVELOPMENT_PROFILE_ID, LEADERBOARD_PROFILE_ID),
            ),
            _field(
                "task",
                "Task or task selector",
                "string",
                "basic",
                10,
                required=True,
                suggestions=task_selector_suggestions(),
                selection_mode="multiple",
                exclusive_choices=task_selector_suggestions()[:7],
                owned_when=official_owned,
            ),
            _field(
                "seed",
                "Seed",
                "integer",
                "basic",
                20,
                default=0,
                minimum=0,
                owned_when=official_owned,
            ),
            _field(
                "episodes",
                "Cases per task",
                "integer",
                "basic",
                10,
                default=1,
                minimum=1,
                owned_when=official_owned,
            ),
            _field(
                "max_episode_steps",
                "Environment step ceiling",
                "integer",
                "tuning",
                20,
                minimum=1,
                owned_when=official_owned,
            ),
            _field(
                "actions_per_policy_request",
                "Actions per policy request",
                "integer",
                "tuning",
                25,
                default=DEFAULT_ACTIONS_PER_POLICY_REQUEST,
                minimum=1,
                maximum=50,
            ),
            _field(
                "planner",
                "Planner",
                "string",
                "tuning",
                10,
                default="codex",
                choices=("api", "claude_code", "codex"),
            ),
            _field(
                "model", "Model", "string", "tuning", 20,
                suggestions_when=BUILTIN_PLANNER_MODEL_SUGGESTIONS,
                selection_mode="single", allow_custom_value=True,
            ),
            _field(
                "planning_profile",
                "Planning profile",
                "string",
                "tuning",
                30,
                default=ROBOCASA_HORIZON_PLANNING_PROFILE,
                choices=tuple(list_planning_profile_ids(include_aliases=True)),
            ),
            _field(
                "max_turns",
                "Planner turn budget",
                "integer",
                "tuning",
                40,
                default=100,
                minimum=1,
                visible_when=LaunchOwnership(
                    "planning_profile",
                    tuple(p for p in list_planning_profile_ids(include_aliases=True)
                          if p != ROBOCASA_HORIZON_PLANNING_PROFILE),
                ),
            ),
            _field(
                "max_tokens",
                "Planner token budget",
                "integer",
                "tuning",
                50,
                default=8192,
                minimum=0,
                visible_when=LaunchOwnership("planner", ("api",)),
            ),
            _field(
                "worker_timeout_s",
                "Case worker timeout",
                "integer",
                "advanced",
                10,
                minimum=1,
            ),
            _field(
                "env_endpoint", "Environment endpoint", "string", "advanced", 20
            ),
            _field(
                "policy_endpoint", "Policy endpoint", "string", "advanced", 30
            ),
            _field(
                "policy_device",
                "Policy device",
                "string",
                "advanced",
                10,
                default="cuda:0",
                suggestions=("cuda:0", "cuda:1", "cpu", "mps"),
                selection_mode="single",
                allow_custom_value=True,
            ),
            _field(
                "policy_dtype",
                "Policy dtype",
                "string",
                "advanced",
                20,
                default="float32",
                choices=("float32", "float16", "bfloat16"),
            ),
            _field(
                "policy_path",
                "Policy checkpoint",
                "string",
                "advanced",
                30,
                default=checkpoint,
                choices=(checkpoint, XIAOMI_MIBOT_CHECKPOINT.repo_id, PI05_CHECKPOINT.repo_id),
            ),
            _field(
                "reuse_policy_service",
                "Reuse loaded SmolVLA",
                "boolean",
                "advanced",
                40,
                default=False,
            ),
            _field(
                "object_registries",
                "Object registries",
                "string",
                "advanced",
                30,
                default="lightwheel,objaverse",
                suggestions=OBJECT_REGISTRIES,
                selection_mode="multiple",
            ),
            _field("instruction", "Instruction override", "string", "basic", 40),
            _field(
                "max_replans",
                "Maximum replans",
                "integer",
                "tuning",
                60,
                default=8,
                minimum=0,
                maximum=50,
                visible_when=LaunchOwnership(
                    "planning_profile",
                    tuple(p for p in list_planning_profile_ids(include_aliases=True)
                          if p != ROBOCASA_HORIZON_PLANNING_PROFILE),
                ),
            ),
            _field(
                "use_chines",
                "Chinese planning",
                "boolean",
                "tuning",
                70,
                default=False,
            ),
            _field(
                "no_images",
                "Text-only planner",
                "boolean",
                "advanced",
                80,
                default=False,
                visible_when=LaunchOwnership("planner", ("api",)),
            ),
            _field(
                "planner_timeout_s",
                "Planner wall-clock budget (minutes)",
                "integer",
                "tuning",
                90,
                default=3600,
                minimum=600,
                maximum=7200,
                step=10,
                display_scale=60,
                visible_when=LaunchOwnership(
                    "planner", ("claude_code", "codex")
                ),
            ),
            _field(
                "plan_generation_timeout_s",
                "Single plan/replan timeout (seconds)",
                "integer",
                "tuning",
                100,
                default=180,
                minimum=60,
                maximum=1800,
                step=60,
            ),
            _field(
                "base_url", "Planner API base URL", "string", "advanced", 110,
                visible_when=LaunchOwnership("planner", ("api",)),
            ),
            _field(
                "harness_mode",
                "Harness mode",
                "string",
                "tuning",
                10,
                choices=("reactive", "trace", "plan-gated"),
            ),
            _field(
                "claude_code_max_budget_usd",
                "Claude Code budget (USD)",
                "number",
                "advanced",
                20,
                minimum=0.01,
                visible_when=LaunchOwnership("planner", ("claude_code",)),
            ),
            _field(
                "recording_mode",
                "Recording profile",
                "string",
                "tuning",
                10,
                default="none",
                choices=("none", "primary", "selected", "all"),
            ),
            _field(
                "recording_camera",
                "Selected cameras",
                "array",
                "tuning",
                20,
                default=[],
            ),
            _field(
                "require_complete_artifacts",
                "Require complete artifacts",
                "boolean",
                "tuning",
                30,
                default=False,
            ),
            _field(
                "submission_metadata_json",
                "Submission metadata JSON",
                "string",
                "advanced",
                40,
            ),
        ),
        canonicalize=_canonicalize,
        derive=validate_api_planner_model,
        check_prerequisites=_check_prerequisites,
    )


def _canonicalize(mode: str, values: dict[str, object]) -> dict[str, object]:
    del mode
    planning_profile = values.get("planning_profile")
    if planning_profile is not None:
        try:
            values["planning_profile"] = get_planning_profile(
                str(planning_profile)
            ).id
        except ValueError as exc:
            raise LaunchValidationError(
                {"planning_profile": "Must use an allowed value."}
            ) from exc
    if values.get("planning_profile") == ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE:
        values.update(
            {
                "policy_path": XIAOMI_MIBOT_CHECKPOINT.repo_id,
                "policy_dtype": "bfloat16",
                "actions_per_policy_request": 16,
            }
        )
    elif values.get("vla") == "pi05":
        values.update(policy_path=PI05_CHECKPOINT.repo_id, policy_dtype="bfloat16",
                      reuse_policy_service=False)
        values.setdefault("actions_per_policy_request", 5)
    elif values.get("vla") == "xiaomi-mibot":
        values.update(policy_path=XIAOMI_MIBOT_CHECKPOINT.repo_id,
                      policy_dtype=XIAOMI_MIBOT_POLICY_SPEC.default_dtype,
                      reuse_policy_service=False)
        values.setdefault("actions_per_policy_request", XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request)
    elif values.get("vla") == "smolvla":
        values["policy_path"] = SMOLVLA_CHECKPOINTS["robocasa"].repo_id
    profile_id = values.get("evaluation_profile")
    try:
        profile = resolve_evaluation_profile(
            None if profile_id is None else str(profile_id),
            packaged_registry_snapshot(),
        )
    except ValueError as exc:
        raise LaunchValidationError(
            {"evaluation_profile": "Must use an allowed value."}
        ) from exc
    metadata_path = values.get("submission_metadata_json")
    if metadata_path is not None:
        from robots.robocasa.submission import SubmissionMetadata

        try:
            SubmissionMetadata.from_json_path(str(metadata_path))
        except (OSError, ValueError) as exc:
            raise LaunchValidationError(
                {"submission_metadata_json": "Must be a valid metadata JSON file."}
            ) from exc
    if not profile.official:
        return values

    locked = {
        "task": ",".join(profile.task_names),
        "seed": profile.scenario_seed,
        "episodes": profile.rollouts_per_task,
    }
    errors = {
        name: "Owned by the selected evaluation profile."
        for name, expected in locked.items()
        if name in values and values[name] != expected
    }
    if "max_episode_steps" in values:
        errors["max_episode_steps"] = "Owned by the selected evaluation profile."
    if errors:
        raise LaunchValidationError(errors)
    values.update(locked)
    return values


def _check_prerequisites(
    mode: str, values: Mapping[str, object]
) -> tuple[LaunchPrerequisite, ...]:
    if values.get("vla") == "pi05":
        endpoint = bool(values.get("policy_endpoint"))
        path = os.environ.get("RPENT_ROBOCASA_PI05_CHECKPOINT")
        ready = endpoint or bool(path and (Path(path) / "params").is_dir() and (Path(path) / "assets/norm_stats.json").is_file())
        return (
            runtime_prerequisite("environment-runtime", "RoboCasa runtime", values,
                                 endpoint_field="env_endpoint", env_var="RPENT_ROBOCASA_PYTHON"),
            runtime_prerequisite("pi05-policy", "OpenPI π0.5 runtime", values,
                                 endpoint_field="policy_endpoint", env_var="RPENT_OPENPI_PYTHON"),
            LaunchPrerequisite("pi05-checkpoint", "π0.5 checkpoint", "ready" if ready else "blocked",
                               "Pi0.5 checkpoint is configured." if ready else "Configure RPENT_ROBOCASA_PI05_CHECKPOINT or provide policy_endpoint."),
            planner_prerequisite(mode, values),
        )
    if values.get("vla") == "xiaomi-mibot" or values.get("planning_profile") == ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE:
        endpoint = str(values.get("policy_endpoint") or "").strip()
        snapshot_value = os.environ.get("RPENT_XIAOMI_MIBOT_SNAPSHOT", "").strip()
        snapshot = Path(snapshot_value).expanduser() if snapshot_value else None
        checkpoint = LaunchPrerequisite(
            "xiaomi-mibot-checkpoint",
            "Xiaomi MiBoT checkpoint",
            "ready" if endpoint or (snapshot and snapshot.is_dir()) else "blocked",
            (
                "Configured through policy_endpoint."
                if endpoint
                else "RPENT_XIAOMI_MIBOT_SNAPSHOT names a checkpoint directory."
                if snapshot and snapshot.is_dir()
                else "Configure RPENT_XIAOMI_MIBOT_SNAPSHOT or provide policy_endpoint."
            ),
        )
        return (
            runtime_prerequisite(
                "environment-runtime",
                "RoboCasa runtime",
                values,
                endpoint_field="env_endpoint",
                env_var="RPENT_ROBOCASA_PYTHON",
            ),
            runtime_prerequisite(
                "xiaomi-mibot-policy",
                "Xiaomi MiBoT policy",
                values,
                endpoint_field="policy_endpoint",
                env_var="RPENT_XIAOMI_MIBOT_PYTHON",
            ),
            checkpoint,
            planner_prerequisite(mode, values),
        )
    return (
        runtime_prerequisite(
            "environment-runtime",
            "RoboCasa runtime",
            values,
            endpoint_field="env_endpoint",
            env_var="RPENT_ROBOCASA_PYTHON",
        ),
        runtime_prerequisite(
            "smolvla-policy",
            "SmolVLA policy",
            values,
            endpoint_field="policy_endpoint",
            env_var="RPENT_SMOLVLA_PYTHON",
        ),
        planner_prerequisite(mode, values),
    )


__all__ = ["build_launch_preset"]
