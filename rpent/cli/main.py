"""Physical agent main CLI entrypoint."""

# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --env libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.runtime`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import queue
import shlex
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

# isort: off
# Importing config loads .env before other RPent modules inspect os.environ.
from rpent.utils.config import (
    get_dashboard_runtime_dir,
    get_repo_root,
    validate_runtime_requirements,
)
# isort: on

from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.context.prompt_utils import format_prompt
from rpent.envs import get_env_spec
from rpent.episode import open_episode
from rpent.evaluation.preparation_progress import (
    advance_preparation_from_environment,
)
from rpent.evaluation.startup_log import record_worker_startup_event
from rpent.harness.verifiers import EnvironmentPredicateVerifier
from rpent.planning import (
    JsonlEventRecorder,
    PlanningOrchestrator,
    build_experiment_manifest,
    list_planning_profile_ids,
    resolve_planning_stack,
)
from rpent.planning.manifest import content_hash
from rpent.planning.stack import resolve_planner_turn_limit
from rpent.runtime import (
    RuntimePlanGenerator,
    RuntimePredicateVerifier,
    authoritative_runtime_options,
    build_runtime,
    classify_termination,
)
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")
_PREDICATE_MAX_TURNS = 4
# Executor system-prompt sections shared with the nested plan generator.
_PLANNER_SHARED_SECTIONS = (
    "EVIDENCE", "BUDGET", "ACTION", "TASK SCOPE AND DECOMPOSITION", "CONTROL MODES",
    "TRANSITIONS", "TAKEOVER", "TASK RULES", "POLICY SKILL",
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


@dataclass(frozen=True)
class PlanningRuntimeBundle:
    """Isolated model runtimes and their reproducibility metadata."""

    executor: Any
    plan_generator: Any | None
    predicate_verifier: Any | None
    metadata: dict[str, Any]


def _runtime_kwargs(
    *,
    args: argparse.Namespace,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    timeout_s: int | None,
    dashboard: Any,
    recorder: Any,
    no_images: bool,
) -> dict[str, Any]:
    return {
        "output_dir": output_dir,
        "recipe_tag": recipe_tag,
        "env_name": env_name,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "planner_timeout_s": timeout_s,
        "claude_code_max_budget_usd": args.claude_code_max_budget_usd,
        "dashboard": dashboard,
        "recorder": recorder,
        "no_images": no_images,
    }


def _planning_control_options(args: argparse.Namespace) -> dict[str, bool]:
    if args.env_name != "robocasa":
        return {}
    from robots.robocasa.control_ablation import control_options

    return control_options(args)


def _configured_prompt_bundle(env_name: str, prompt_vars: Mapping[str, object]):
    bundle = get_env_spec(env_name).prompts
    if env_name != "robocasa":
        return bundle
    from robots.robocasa.control_ablation import prompt_ablation
    from robots.robocasa.prompts.system import system_prompt
    from rpent.envs.prompt_bundle import PromptBundle

    return PromptBundle(
        system=partial(system_prompt, ablate=prompt_ablation(prompt_vars)),
        user=bundle.user,
    )


def _planner_action_guidance(env_name: str, prompt_vars: Mapping[str, object]) -> str:
    """Render the executor sections that nested plan generation must respect."""
    sections = _configured_prompt_bundle(env_name, prompt_vars).system()
    shared = {
        name: sections[name] for name in _PLANNER_SHARED_SECTIONS if name in sections
    }
    return format_prompt(
        shared, variables={"policy_skill": "", **prompt_vars}
    ).strip()


def _build_planning_runtime_bundle(
    *,
    stack: Any,
    args: argparse.Namespace,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    dashboard: Any,
    recorder: Any,
    prompt_vars: Mapping[str, object] | None = None,
) -> PlanningRuntimeBundle:
    """Build isolated executor, planner, and strict predicate runtimes."""
    planner_output_dir = Path(output_dir) / "agent_artifacts"
    runtime_options = (
        authoritative_runtime_options(args.planner)
        if stack.uses_authoritative_session
        else {}
    )
    executor = build_runtime(
        args.planner,
        **_runtime_kwargs(
            args=args,
            output_dir=planner_output_dir,
            recipe_tag=recipe_tag,
            env_name=env_name,
            timeout_s=args.planner_timeout_s,
            dashboard=dashboard,
            recorder=recorder,
            no_images=args.no_images,
        ),
        **runtime_options,
    )
    plan_generator = None
    predicate_verifier = None
    recipe_tags = {"executor": recipe_tag}
    no_images = {"executor": bool(args.no_images)}
    action_guidance = (
        _planner_action_guidance(env_name, prompt_vars or {})
        if env_name == "robocasa" and stack.harness_mode != "reactive"
        else ""
    )

    if stack.harness_mode != "reactive":
        plan_tag = f"{recipe_tag}_plan"
        plan_no_images = (
            bool(args.no_images) if stack.uses_authoritative_session else True
        )
        plan_runtime = build_runtime(
            args.planner,
            **_runtime_kwargs(
                args=args,
                output_dir=planner_output_dir,
                recipe_tag=plan_tag,
                env_name=env_name,
                timeout_s=args.plan_generation_timeout_s,
                dashboard=None,
                recorder=recorder,
                no_images=plan_no_images,
            ),
            **runtime_options,
        )
        plan_generator = RuntimePlanGenerator(
            plan_runtime,
            use_chinese=stack.use_chinese,
            require_outcome_program=stack.profile.require_outcome_program,
            action_guidance=action_guidance,
            allow_information_requests=(
                env_name == "robocasa" and not stack.uses_authoritative_session
            ),
        )
        recipe_tags["plan"] = plan_tag
        no_images["plan"] = plan_no_images

    if stack.uses_authoritative_session:
        predicate_tag = f"{recipe_tag}_predicate"
        predicate_runtime = build_runtime(
            args.planner,
            **_runtime_kwargs(
                args=args,
                output_dir=planner_output_dir,
                recipe_tag=predicate_tag,
                env_name=env_name,
                timeout_s=args.plan_generation_timeout_s,
                dashboard=None,
                recorder=recorder,
                no_images=bool(args.no_images),
            ),
            **runtime_options,
        )
        predicate_verifier = RuntimePredicateVerifier(
            predicate_runtime,
            evidence_root=Path(output_dir),
            model_id=args.model or "configured-default",
            max_turns=_PREDICATE_MAX_TURNS,
            recorder=recorder,
        )
        recipe_tags["predicate"] = predicate_tag
        no_images["predicate"] = bool(args.no_images)

    if stack.uses_authoritative_session:
        interface = "authoritative-harness-session-v1"
    elif stack.harness_mode == "reactive":
        interface = "reactive-tool-loop-v1"
    else:
        interface = "legacy-harness-controller-v1"
    metadata: dict[str, Any] = {
        "interface": interface,
        "recipe_tags": recipe_tags,
        "no_images": no_images,
    }
    if action_guidance:
        metadata["plan_action_context"] = {
            "version": "executor-action-guidance-v2",
            "sha256": content_hash(action_guidance),
        }
    if stack.uses_authoritative_session:
        metadata["isolation"] = {
            "backend": args.planner,
            "native_sandbox": runtime_options.get("codex_sandbox"),
            "native_approval": runtime_options.get("codex_approval_mode"),
        }
    if predicate_verifier is not None:
        metadata["predicate_verifier"] = {
            "name": getattr(predicate_verifier, "name", "runtime-predicate"),
            "version": getattr(predicate_verifier, "version", "1"),
            "model": args.model or "configured-default",
            "recipe_tag": recipe_tags["predicate"],
            "no_images": bool(args.no_images),
            "max_turns": _PREDICATE_MAX_TURNS,
        }
    if stack.uses_authoritative_session:
        recorder.record("planning_runtimes_configured", runtime_config=metadata)
    return PlanningRuntimeBundle(
        executor=executor,
        plan_generator=plan_generator,
        predicate_verifier=predicate_verifier,
        metadata=metadata,
    )


def _attach_planning_harness(
    *,
    stack: Any,
    toolkit: Any,
    output_dir: str | Path,
    recorder: Any,
    plan_generator: Any | None,
    predicate_verifier: Any | None,
    environment_name: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any | None, Any | None]:
    """Attach exactly one profile-selected Harness interface."""
    trace_path = Path(output_dir) / "harness_trace.jsonl"
    if stack.uses_authoritative_session:
        if predicate_verifier is None:
            raise ValueError("authoritative session requires a predicate verifier")
        if environment_name == "robocasa":
            # RoboCasa declares environment.success as its sole official
            # simulator predicate. Camera evidence remains in the audit trail,
            # but an occluded or visually unchanged view must not veto the
            # execution-bound native result.
            predicate_verifiers = (
                EnvironmentPredicateVerifier(bound_outcome_ids=None),
            )
        else:
            predicate_verifiers = (predicate_verifier,)
        session = stack.build_harness_session(
            trace_path=trace_path,
            event_recorder=recorder,
            plan_generator=plan_generator,
            predicate_verifiers=predicate_verifiers,
        )
        if environment_name == "robocasa" and runtime_metadata is not None:
            session.controller.configure_control_modes(
                policy_family=runtime_metadata.get("policy_family", "smolvla"),
                task_name=runtime_metadata.get("task"),
            )
        toolkit.attach_harness_session(session, Path(output_dir))
        return None, session

    harness = stack.build_harness(
        trace_path=trace_path,
        event_recorder=recorder,
        plan_generator=plan_generator,
    )
    if harness is not None:
        if environment_name == "robocasa":
            harness.require_object_states = True
            if runtime_metadata is not None:
                harness.configure_control_modes(
                    policy_family=runtime_metadata.get("policy_family", "smolvla"),
                    task_name=runtime_metadata.get("task"),
                )
        toolkit.attach_harness(harness)
    return harness, None


@contextmanager
def _open_episode_toolkit_with_harness(
    *,
    env_spec: Any,
    args: argparse.Namespace,
    output_dir: str | Path,
    dashboard: Any,
    recorder: Any,
    context_policy: Any,
    stack: Any,
    plan_generator: Any | None,
    predicate_verifier: Any | None,
    worker_recording: Any | None = None,
) -> Iterator[tuple[Any, Any | None, Any | None]]:
    """Open the canonical episode, then attach its planning interface."""
    bootstrap = env_spec.build_episode_bootstrap(args, Path(output_dir), dashboard)
    with open_episode(bootstrap) as episode:
        toolkit = episode.toolkit
        active_recorder = (
            worker_recording.bind(bootstrap, episode)
            if worker_recording is not None
            else recorder
        )
        toolkit.attach_event_recorder(active_recorder)
        toolkit.set_context_policy(context_policy)
        history_source = getattr(toolkit, "action_history_snapshot", None)
        bind_history = getattr(plan_generator, "set_action_history_source", None)
        if callable(history_source) and callable(bind_history):
            bind_history(history_source)
        if getattr(args, "skill_library_read", False):
            from rpent.skill_library.reader import attach_reader

            metadata = dict(bootstrap.runtime_metadata or {})
            checkpoint = metadata.get("checkpoint", {})
            reader_meta = attach_reader(
                toolkit,
                enabled=True,
                snapshot=args.skill_library_snapshot,
                policy={
                    "family": metadata.get("policy_family", "smolvla"),
                    "checkpoint": checkpoint.get("repo_id"),
                    "revision": checkpoint.get("revision"),
                    "action_schema": metadata.get("action_schema"),
                    "observation_schema": metadata.get("observation_schema"),
                    "binding_identity": metadata.get("binding_identity"),
                },
                llm=args.model,
            )
            active_recorder.record("skill_library_reader_configured", **reader_meta)
        harness, session = _attach_planning_harness(
            stack=stack,
            toolkit=toolkit,
            output_dir=output_dir,
            recorder=active_recorder,
            plan_generator=plan_generator,
            predicate_verifier=predicate_verifier,
            environment_name=str(getattr(env_spec, "name", "")),
            runtime_metadata=getattr(bootstrap, "runtime_metadata", None),
        )
        yield toolkit, harness, session


def _planning_snapshot(*, session: Any | None, harness: Any | None) -> dict[str, Any]:
    """Return the authoritative audit snapshot for terminal classification."""
    if session is not None:
        return session.snapshot()
    if harness is not None:
        return harness.snapshot()
    return {"mode": "reactive"}


def _resolve_cli_planning_profile(
    args: argparse.Namespace,
    env_spec=None,
) -> str:
    """Delegate environment-safe planning selection to the active Adapter."""
    spec = env_spec or get_env_spec(args.env_name)
    profile_id = spec.resolve_planning_profile(args)
    args.max_turns = resolve_planner_turn_limit(
        profile_id, getattr(args, "max_turns", None)
    )
    return profile_id


def _requires_environment_success(run_config: Any, planning_stack: Any) -> bool:
    return bool(
        getattr(
            run_config,
            "require_environment_success",
            planning_stack.require_environment_success,
        )
    )


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {
            **{k: v for k, v in m.items() if k != "content"},
            "content": _strip_images(m.get("content")),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="RPent multi-simulator embodied-agent evaluation runtime",
    )

    ap.add_argument(
        "--env",
        dest="env_name",
        help=(
            "Environment Adapter name under robots.<name>; built-ins include "
            "libero, robocerebra, and robocasa."
        ),
    )
    ap.add_argument(
        "--_episode-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--episodes",
        type=_positive_integer,
        default=None,
        help="Episodes per resolved task (default: 1 when evaluation is selected).",
    )
    ap.add_argument(
        "--jobs",
        type=_positive_integer,
        default=1,
        help="Concurrent episode workers (RoboCasa Pi0.5 supports GPU replicas).",
    )
    ap.add_argument(
        "--worker-timeout-s",
        type=_positive_integer,
        default=None,
        help=(
            "Override each worker's wall-clock timeout in seconds. By default, "
            "RoboCasa allows 30 minutes per 750 official task steps, rounded up; "
            "other simulators allow 1800 seconds."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume valid retained cases from an existing evaluation output.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve cases and write the immutable manifest without execution.",
    )
    ap.add_argument(
        "--recording-mode",
        choices=["none", "primary", "selected", "all"],
        default=None,
        help=(
            "Control-step camera retention (default: none for RoboCasa, which "
            "archives actual LLM input images; otherwise none without cameras, "
            "all for one Case, primary otherwise)."
        ),
    )
    ap.add_argument(
        "--recording-camera",
        action="append",
        default=[],
        help="Camera ID retained by --recording-mode selected; repeat as needed.",
    )
    ap.add_argument(
        "--require-complete-artifacts",
        action="store_true",
        help="Reject formal qualification when recording artifacts are degraded.",
    )

    # models
    ap.add_argument(
        "--planner",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2, codexgateway:gpt-5.6-luna). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--instruction",
        default=None,
        help=(
            "Override the opening task instruction sent to the planner. "
            "The selected environment task and success predicate are unchanged."
        ),
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument(
        "--max-turns",
        type=_positive_integer,
        default=None,
        help="Outer planner turn cap (default: 100; uncapped for the RoboCasa horizon profile).",
    )
    ap.add_argument(
        "--max-tokens",
        type=_nonnegative_integer,
        default=8192,
        help="API output token cap; 0 explicitly omits the cap for compatible gateways.",
    )
    ap.add_argument(
        "--planning-profile",
        choices=list_planning_profile_ids(include_aliases=True),
        default=None,
        help=(
            "Versioned planning policy configuration. RoboCasa defaults to "
            "robocasa-horizon-gated-v1; other direct evaluations require an "
            "explicit value."
        ),
    )
    ap.add_argument(
        "--harness-mode",
        choices=["reactive", "trace", "plan-gated"],
        default=None,
        help=(
            "Override the planning profile's execution harness: reactive "
            "preserves the existing loop; trace logs plan protocol events; "
            "plan-gated blocks unplanned/unchecked actions."
        ),
    )
    ap.add_argument(
        "--max-replans",
        type=int,
        default=None,
        help=(
            "Override the planning profile's maximum structured replans in "
            "trace/plan-gated modes."
        ),
    )
    ap.add_argument(
        "--use_chines",
        "--use-chinese",
        dest="use_chines",
        action="store_true",
        help=(
            "Require Simplified Chinese for planning, verification, replan, "
            "and other user/audit-facing intermediate text. Tool names and "
            "machine-readable keys remain unchanged."
        ),
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Never send image bytes to the model (api planner only). "
        "Use for text-only models that reject image input "
        "(e.g. 400 \"message type 'image_url' is not supported\"); "
        "read_image then returns the file path with a notice.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for the claude_code/codex planner "
        "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--plan-generation-timeout-s",
        type=int,
        default=180,
        help=(
            "Wall-clock cap for each nested create_plan/replan model call. "
            "This is separate from --planner-timeout-s so one slow proposal "
            "cannot consume the whole embodied-agent run."
        ),
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # other config
    ap.add_argument(
        "--skill-library-read",
        action="store_true",
        default=False,
        help="Expose read-only skill lookup from an explicitly pinned snapshot (experimental; default off).",
    )
    ap.add_argument(
        "--skill-library-snapshot",
        type=Path,
        default=None,
        help="Immutable skill snapshot JSON used only with --skill-library-read.",
    )
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the singleton local Dashboard and wait.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "variant (index.zh-cn.html); defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )

    return ap


def main(
    argv: list[str] | None = None,
    *,
    worker_recording: Any | None = None,
) -> int:
    requested_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_argparser()
    # Two-phase argparse: route the standalone Dashboard before resolving an
    # environment, otherwise add the selected environment's flags.
    early, _ = parser.parse_known_args(argv)

    if early.dashboard:
        from rpent.dashboard import DashboardAlreadyRunning, DashboardServer

        args = parser.parse_args(argv)
        server: DashboardServer | None = None
        dashboard_runtime_dir = get_dashboard_runtime_dir()
        try:
            server = DashboardServer(
                host=args.dashboard_host,
                port=args.dashboard_port,
                language=args.dashboard_language,
                runtime_dir=(
                    str(dashboard_runtime_dir)
                    if dashboard_runtime_dir is not None
                    else None
                ),
            )
            print(f"Dashboard: {server.start()}", flush=True)
            server.wait()
        except (DashboardAlreadyRunning, ValueError) as exc:
            print(f"rpent: error: {exc}", file=sys.stderr, flush=True)
            return 1
        except KeyboardInterrupt:
            pass
        finally:
            if server is not None:
                server.stop()
        return 0

    if early.env_name is None:
        parser.error("the following arguments are required: --env")
    env_spec = get_env_spec(early.env_name)
    env_spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(argv)

    if getattr(args, "skill_library_read", False) and (
        args.env_name != "robocasa"
        or args.skill_library_snapshot is None
        or not args.model
    ):
        parser.error(
            "--skill-library-read requires RoboCasa, an explicit --model "
            "and --skill-library-snapshot"
        )

    from rpent.evaluation.front_controller import (
        run_evaluation_front_controller,
        should_coordinate,
    )

    args.planning_profile = _resolve_cli_planning_profile(args, env_spec)

    if should_coordinate(args, env_spec):
        preset_factory = getattr(env_spec, "launch_preset", None)
        if preset_factory is not None and not args.dry_run:
            blocked = [
                item
                for item in preset_factory().cli_preflight(args)
                if item.status == "blocked"
            ]
            if blocked:
                messages = " ".join(item.message for item in blocked)
                print(
                    "rpent: error: Launch prerequisites are not ready: " + messages,
                    file=sys.stderr,
                    flush=True,
                )
                return 1
        return run_evaluation_front_controller(
            args,
            env_spec,
            requested_argv,
        )

    validate_runtime_requirements(
        require_pi05=args.env_name == "libero"
        and getattr(args, "vla_endpoint", None) is None,
        require_sam3=args.env_name == "libero"
        and getattr(args, "sam3_endpoint", None) is None,
        require_libero=args.env_name == "libero",
        planner=args.planner,
        model=args.model,
    )

    ignore_environment_termination = bool(
        getattr(args, "ignore_environment_termination", False)
    )
    planning_stack = resolve_planning_stack(
        args.planning_profile,
        harness_mode=args.harness_mode,
        max_replans=args.max_replans,
        use_chinese=args.use_chines,
        require_environment_success=(False if ignore_environment_termination else None),
        **_planning_control_options(args),
    )
    # Downstream code consumes the effective values; the selected profile and
    # any overrides are preserved separately in planning-stack metadata.
    args.harness_mode = planning_stack.harness_mode
    args.max_replans = planning_stack.max_replans

    run_config = env_spec.parse_config(args)
    require_environment_success = _requires_environment_success(
        run_config,
        planning_stack,
    )
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    dashboard_state = run_config.dashboard_state
    task_desc = run_config.task_desc

    env_name = args.env_name

    # mkdir + logging wiring (env-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("planning config: %s", planning_stack.metadata())
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))
    planning_config_path = Path(output_dir) / "planning_config.json"
    planning_config_path.write_text(
        json.dumps(planning_stack.metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    if args.env_name == "robocasa":
        (Path(output_dir) / "control_mode_config.json").write_text(
            json.dumps(run_config.task_desc["control_modes"], indent=2) + "\n",
            encoding="utf-8",
        )
    event_recorder = worker_recording or JsonlEventRecorder(
        Path(output_dir) / "events.jsonl",
        episode_id=recipe_tag,
        context={
            "planning_profile": planning_stack.profile.id,
            "planner_backend": args.planner,
            "model": args.model,
        },
    )
    event_recorder.record(
        "run_configured",
        task=task_desc,
        planning_config=planning_stack.metadata(),
    )

    ensure_resources(env_name)

    runtime_bundle = _build_planning_runtime_bundle(
        stack=planning_stack,
        args=args,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        dashboard=dashboard_state,
        recorder=event_recorder,
        prompt_vars=prompt_vars,
    )
    runtime = runtime_bundle.executor
    plan_generator = runtime_bundle.plan_generator
    prompt_bundle = _configured_prompt_bundle(env_name, prompt_vars)

    prompt_vars = {
        **prompt_vars,
        "output_dir": output_dir,
        "harness_protocol": planning_stack.prompt_context(),
        "planning_startup": planning_stack.startup_instruction(),
    }
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )
    instruction_override = (args.instruction or "").strip()
    if instruction_override:
        system_prompt += (
            "\n\nINSTRUCTION OVERRIDE\n"
            "The launcher supplied an explicit instruction override. Treat the "
            "override in the opening user message as the authoritative wording "
            "for planning and execution, instead of states.json task_language. "
            "The selected simulator task, scene, and success predicate are still "
            "unchanged; use task_language only as environment metadata and do "
            "not silently replace the override with it."
        )
        user_msg += (
            "\n\nINSTRUCTION OVERRIDE (authoritative for planning):\n"
            f"{instruction_override}"
        )
    if ignore_environment_termination and not planning_stack.uses_authoritative_session:
        system_prompt += (
            "\n\nCUSTOM SEQUENCE TERMINATION POLICY (overrides earlier "
            "environment-success instructions)\n"
            "The original LIBERO success predicate is audit-only for this run. "
            "Reaching it does not complete the custom instruction and must not "
            "cause you to stop physical actions. `base_task_success=true` means "
            "only that the original benchmark goal was observed. Continue until "
            "every registered custom-plan step has grounded verification. "
            "`libero_terminated` intentionally remains false unless an effective "
            "non-suppressed termination occurs. You may call "
            "finish(status='success') after the registered plan is fully verified; "
            "do not require the original environment predicate. Time-limit "
            "truncation and controller errors remain effective stop conditions.\n"
            "TERMINATION ACTIONS:\n"
            "1. Fully verified registered plan: save the required audit, then "
            "call finish(status='success', summary=...).\n"
            "2. Unrecoverable execution, episode_truncated=true, or exhausted "
            "replan budget: save the required audit, then call "
            "finish(status='failure', summary=...).\n"
            "3. base_task_success=true: do not call finish; continue the remaining "
            "registered plan.\n"
            "4. Never call finish with status='stuck'; all unsuccessful terminal "
            "outcomes must use status='failure'."
        )
        user_msg += (
            "\n\nThe original LIBERO environment termination is disabled for "
            "this custom sequence. Do not stop when base_task_success becomes "
            "true; finish the entire instruction."
        )

    input_queue: queue.Queue[str | None] | None = None
    await_first_prompt: Callable[[], str | None] | None = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # --- agent loop --------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    first_user_msg: str | None = user_msg
    manifest: dict = {}
    episode_metrics: dict[str, Any] = {}
    with _open_episode_toolkit_with_harness(
        env_spec=env_spec,
        args=args,
        output_dir=output_dir,
        dashboard=dashboard_state,
        recorder=event_recorder,
        context_policy=planning_stack.context_policy,
        stack=planning_stack,
        plan_generator=plan_generator,
        predicate_verifier=runtime_bundle.predicate_verifier,
        worker_recording=worker_recording,
    ) as (toolkit, harness, harness_session):
        record_worker_startup_event("simulator_starting", "simulator_ready")
        advance_preparation_from_environment("simulator_starting", "ready")
        advance_preparation_from_environment("planner_starting", "loading")
        try:
            if await_first_prompt is not None:
                # Block until the opening prompt typed during startup is ready.
                first_user_msg = await_first_prompt()
                if first_user_msg is None:
                    logger.info("no task entered; ending session before start.")
            if first_user_msg is not None:
                event_recorder.record(
                    "instruction_received",
                    instruction=first_user_msg,
                    explicit_override=bool(instruction_override),
                    interactive=args.interactive,
                )
            manifest = build_experiment_manifest(
                config=planning_stack,
                planner_backend=args.planner,
                model=args.model,
                max_tokens=args.max_tokens,
                no_images=args.no_images,
                system_prompt=system_prompt,
                user_message=first_user_msg or user_msg,
                tool_schemas=toolkit.get_tools_spec(),
                repo_root=get_repo_root(),
                skill_outcome_adapter_id=toolkit.skill_outcome_adapter_id,
                runtime_metadata=(
                    runtime_bundle.metadata
                    if planning_stack.uses_authoritative_session
                    or "plan_action_context" in runtime_bundle.metadata
                    else None
                ),
                volatile_values={str(output_dir): "{{output_dir}}"},
            )
            manifest_path = Path(output_dir) / "experiment_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            event_recorder.record("experiment_manifest_created", manifest=manifest)
            orchestrator = PlanningOrchestrator(runtime, recorder=event_recorder)
            advance_preparation_from_environment("planner_starting", "ready")
            if first_user_msg is not None:
                advance_preparation_from_environment("running", "loading")
                advance_preparation_from_environment("running", "ready")
                result = orchestrator.run(
                    system_prompt=system_prompt,
                    user_message=first_user_msg,
                    toolkit=toolkit,
                    max_turns=args.max_turns,
                    input_queue=input_queue,
                )
                finish_result = result.finish_result
                messages = result.messages
                stats = result.stats
                agent_error = result.error
        except Exception as e:
            agent_error = f"{type(e).__name__}: {e}"
            logger.error("EXCEPTION in agent loop: %s", e)
        finally:
            # Write the recipe while its toolkit and episode are still open.
            try:
                recipe_path = toolkit.write_recipe(recipe_tag)
                logger.info("recipe: %s", recipe_path)
            except Exception as e:
                logger.warning("failed to write recipe: %s", e)
            metrics = getattr(toolkit, "evaluation_metrics", None)
            if isinstance(metrics, dict):
                episode_metrics = dict(metrics)

    elapsed = time.time() - t0

    harness_snapshot = _planning_snapshot(
        session=harness_session,
        harness=harness,
    )
    termination = classify_termination(
        finish_result=finish_result,
        error=agent_error,
        harness_snapshot=harness_snapshot,
        stop_reason=stats.get("stop_reason"),
        require_environment_success=require_environment_success,
    )
    if termination.error is not None and agent_error is None:
        agent_error = termination.error

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "instruction_override": instruction_override or None,
        "planning_profile": planning_stack.profile.id,
        "planning_config": planning_stack.metadata(),
        "runtime_budgets": {
            "max_turns": args.max_turns,
            "planner_timeout_s": args.planner_timeout_s,
            "plan_generation_timeout_s": args.plan_generation_timeout_s,
            "max_episode_steps": args.max_episode_steps,
        },
        "termination_policy": {
            "ignore_environment_termination": (ignore_environment_termination),
            "require_environment_success": (require_environment_success),
        },
        "experiment_manifest": manifest,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "terminal_classification": termination.classification,
        "simulator_success": harness_snapshot.get("environment_success"),
        "contract_success": harness_snapshot.get("contract_success"),
        "episode_metrics": episode_metrics,
        "finish": finish_result,
        "stats": stats,
        "error": agent_error,
        "harness": harness_snapshot,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)
    event_recorder.record(
        "episode_finished",
        terminal_classification=termination.classification,
        finish=finish_result,
        error=agent_error,
        elapsed_s=round(elapsed, 1),
        stats=stats,
        transcript_path=transcript_path,
    )

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)

    return termination.exit_code


if __name__ == "__main__":
    sys.exit(main())
