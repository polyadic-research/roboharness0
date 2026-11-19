"""CLI assembly tests for the authoritative HarnessSession profile."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import rpent.cli.main as cli
from rpent.planning import resolve_planning_stack
from rpent.runtime import classify_termination
from rpent.tools.toolkit import Toolkit


class Recorder:
    def __init__(self) -> None:
        self.events = []

    def record(self, event, **payload) -> None:
        self.events.append((event, payload))


def runtime_args(*, no_images=False, planner="api"):
    return SimpleNamespace(
        planner=planner,
        base_url=None,
        model="test-model",
        max_tokens=2048,
        planner_timeout_s=90,
        plan_generation_timeout_s=30,
        claude_code_max_budget_usd=None,
        no_images=no_images,
    )


ROBOCASA_PROMPT_VARS = {"task_rules": "TASK-RULES-SENTINEL"}


def test_formal_runtime_uses_one_effective_environment_success_requirement() -> None:
    stack = resolve_planning_stack("libero-authoritative-gated-v1")
    run_config = SimpleNamespace(require_environment_success=True)

    effective = cli._requires_environment_success(run_config, stack)

    assert effective is True
    assert {
        "ignore_environment_termination": False,
        "require_environment_success": effective,
    }["require_environment_success"] is effective


@pytest.mark.parametrize("no_images", [False, True])
def test_authoritative_runtime_bundle_builds_three_isolated_runtimes(
    tmp_path, no_images
):
    stack = resolve_planning_stack("libero-authoritative-gated-v1")
    args = runtime_args(no_images=no_images)
    built = [object(), object(), object()]
    recorder = Recorder()

    with (
        patch.object(cli, "build_runtime", side_effect=built) as build,
        patch.object(cli, "RuntimePlanGenerator", return_value="generator"),
        patch.object(
            cli, "RuntimePredicateVerifier", return_value="verifier"
        ) as verify,
    ):
        bundle = cli._build_planning_runtime_bundle(
            stack=stack,
            args=args,
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="libero",
            dashboard=object(),
            recorder=recorder,
        )

    calls = build.call_args_list
    assert [call.kwargs["recipe_tag"] for call in calls] == [
        "episode",
        "episode_plan",
        "episode_predicate",
    ]
    assert len({id(runtime) for runtime in built}) == 3
    assert calls[0].kwargs["dashboard"] is not None
    assert calls[1].kwargs["dashboard"] is None
    assert calls[2].kwargs["dashboard"] is None
    assert [call.kwargs["no_images"] for call in calls] == [
        no_images,
        no_images,
        no_images,
    ]
    assert bundle.executor is built[0]
    assert bundle.plan_generator == "generator"
    assert bundle.predicate_verifier == "verifier"
    verify.assert_called_once_with(
        built[2],
        evidence_root=Path(tmp_path),
        model_id="test-model",
        max_turns=4,
        recorder=recorder,
    )
    assert bundle.metadata["interface"] == "authoritative-harness-session-v1"
    assert bundle.metadata["predicate_verifier"] == {
        "name": "runtime-predicate",
        "version": "1",
        "model": "test-model",
        "recipe_tag": "episode_predicate",
        "no_images": no_images,
        "max_turns": 4,
    }
    assert bundle.metadata["isolation"] == {
        "backend": "api",
        "native_sandbox": None,
        "native_approval": None,
    }
    assert recorder.events[-1] == (
        "planning_runtimes_configured",
        {"runtime_config": bundle.metadata},
    )


def test_legacy_runtime_bundle_preserves_two_runtime_configuration(tmp_path):
    stack = resolve_planning_stack("libero-hooked-gated-v3")
    args = runtime_args(no_images=False, planner="codex")

    with (
        patch.object(cli, "build_runtime", side_effect=[object(), object()]) as build,
        patch.object(cli, "RuntimePlanGenerator", return_value="legacy-generator"),
    ):
        bundle = cli._build_planning_runtime_bundle(
            stack=stack,
            args=args,
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="libero",
            dashboard=None,
            recorder=Recorder(),
        )

    assert build.call_count == 2
    assert build.call_args_list[1].kwargs["no_images"] is True
    assert bundle.plan_generator == "legacy-generator"
    assert bundle.predicate_verifier is None
    assert bundle.metadata["interface"] == "legacy-harness-controller-v1"


@pytest.mark.parametrize(
    ("profile_id", "require_outcome_program"),
    [
        ("episode-hooked-gated-v1", False),
        ("authoritative-gated-v1", True),
    ],
)
def test_runtime_bundle_binds_outcome_program_to_profile(
    tmp_path, profile_id, require_outcome_program
):
    stack = resolve_planning_stack(profile_id)
    runtime_count = 3 if stack.uses_authoritative_session else 2

    with (
        patch.object(cli, "build_runtime", side_effect=[object()] * runtime_count),
        patch.object(cli, "RuntimePlanGenerator", return_value="generator") as build,
        patch.object(cli, "RuntimePredicateVerifier", return_value="verifier"),
    ):
        cli._build_planning_runtime_bundle(
            stack=stack,
            args=runtime_args(),
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="robocasa",
            dashboard=None,
            recorder=Recorder(),
            prompt_vars=ROBOCASA_PROMPT_VARS,
        )

    assert build.call_args.kwargs["require_outcome_program"] is require_outcome_program


def test_authoritative_codex_builds_three_read_only_runtimes(tmp_path):
    stack = resolve_planning_stack("libero-authoritative-gated-v1")

    with (
        patch.object(
            cli, "build_runtime", side_effect=[object(), object(), object()]
        ) as build,
        patch.object(cli, "RuntimePlanGenerator", return_value="generator"),
        patch.object(cli, "RuntimePredicateVerifier", return_value="verifier"),
    ):
        bundle = cli._build_planning_runtime_bundle(
            stack=stack,
            args=runtime_args(planner="codex"),
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="libero",
            dashboard=None,
            recorder=Recorder(),
        )

    assert build.call_count == 3
    assert [call.kwargs["codex_sandbox"] for call in build.call_args_list] == [
        "read-only",
        "read-only",
        "read-only",
    ]
    assert [call.kwargs["codex_approval_mode"] for call in build.call_args_list] == [
        "auto_review",
        "auto_review",
        "auto_review",
    ]
    assert bundle.metadata["isolation"] == {
        "backend": "codex",
        "native_sandbox": "read-only",
        "native_approval": "auto_review",
    }


def test_planner_runtime_outputs_are_confined_to_agent_artifacts(tmp_path):
    stack = resolve_planning_stack("robocasa-hooked-gated-v1")

    with (
        patch.object(cli, "build_runtime", side_effect=[object(), object()]) as build,
        patch.object(cli, "RuntimePlanGenerator", return_value="generator"),
    ):
        cli._build_planning_runtime_bundle(
            stack=stack,
            args=runtime_args(planner="codex"),
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="robocasa",
            dashboard=None,
            recorder=Recorder(),
            prompt_vars=ROBOCASA_PROMPT_VARS,
        )

    assert [call.kwargs["output_dir"] for call in build.call_args_list] == [
        tmp_path / "agent_artifacts",
        tmp_path / "agent_artifacts",
    ]


def test_authoritative_claude_code_still_fails_closed(tmp_path):
    stack = resolve_planning_stack("libero-authoritative-gated-v1")

    with pytest.raises(
        ValueError, match="does not provide tested authoritative isolation"
    ):
        cli._build_planning_runtime_bundle(
            stack=stack,
            args=runtime_args(planner="claude_code"),
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="libero",
            dashboard=None,
            recorder=Recorder(),
        )


def test_cli_attaches_only_session_and_exposes_no_legacy_control_tools(tmp_path):
    stack = resolve_planning_stack("libero-authoritative-gated-v1")
    toolkit = Toolkit()

    harness, session = cli._attach_planning_harness(
        stack=stack,
        toolkit=toolkit,
        output_dir=tmp_path,
        recorder=Recorder(),
        plan_generator=object(),
        predicate_verifier=object(),
    )

    assert harness is None
    assert session is toolkit._harness_session
    assert toolkit._harness is None
    names = {schema["name"] for schema in toolkit.get_tools_spec()}
    assert not names & {
        "create_plan",
        "register_plan",
        "replan",
        "request_replan",
        "select_outcome",
        "advance_stage",
        "view_plan",
    }


def test_robocasa_session_uses_only_official_environment_verifier(tmp_path):
    session = object()
    stack = Mock(uses_authoritative_session=True)
    stack.build_harness_session.return_value = session
    toolkit = Mock()
    runtime_verifier = object()

    harness, attached = cli._attach_planning_harness(
        stack=stack,
        toolkit=toolkit,
        output_dir=tmp_path,
        recorder=Recorder(),
        plan_generator=object(),
        predicate_verifier=runtime_verifier,
        environment_name="robocasa",
    )

    verifiers = stack.build_harness_session.call_args.kwargs["predicate_verifiers"]
    assert harness is None
    assert attached is session
    assert len(verifiers) == 1
    assert isinstance(verifiers[0], cli.EnvironmentPredicateVerifier)


def test_legacy_and_reactive_attachment_paths_remain_separate(tmp_path):
    toolkit = Mock()
    controller = object()
    legacy = SimpleNamespace(
        uses_authoritative_session=False,
        harness_mode="plan-gated",
        build_harness=Mock(return_value=controller),
    )

    harness, session = cli._attach_planning_harness(
        stack=legacy,
        toolkit=toolkit,
        output_dir=tmp_path,
        recorder=Recorder(),
        plan_generator=object(),
        predicate_verifier=None,
    )

    assert (harness, session) == (controller, None)
    toolkit.attach_harness.assert_called_once_with(controller)
    toolkit.attach_harness_session.assert_not_called()

    reactive = SimpleNamespace(
        uses_authoritative_session=False,
        harness_mode="reactive",
        build_harness=Mock(return_value=None),
    )
    toolkit.reset_mock()
    assert cli._attach_planning_harness(
        stack=reactive,
        toolkit=toolkit,
        output_dir=tmp_path,
        recorder=Recorder(),
        plan_generator=None,
        predicate_verifier=None,
    ) == (None, None)
    toolkit.attach_harness.assert_not_called()
    toolkit.attach_harness_session.assert_not_called()


def test_session_snapshot_is_authoritative_for_termination():
    session = Mock()
    session.snapshot.return_value = {
        "state": "complete",
        "contract_success": True,
    }
    legacy = Mock()

    snapshot = cli._planning_snapshot(session=session, harness=legacy)
    result = classify_termination(
        finish_result={"status": "success"},
        error=None,
        harness_snapshot=snapshot,
    )

    assert result.classification == "SUCCESS"
    session.snapshot.assert_called_once_with()
    legacy.snapshot.assert_not_called()
    assert cli._planning_snapshot(session=None, harness=legacy) is legacy.snapshot()
    assert cli._planning_snapshot(session=None, harness=None) == {"mode": "reactive"}


def test_canonical_toolkit_assembly_uses_open_episode_cleanup_on_failure(tmp_path):
    env_spec = Mock()
    bootstrap = object()
    env_spec.build_episode_bootstrap.return_value = bootstrap
    toolkit = Mock()
    toolkit.attach_harness_session.side_effect = RuntimeError("attach failed")
    stack = Mock(uses_authoritative_session=True, harness_mode="plan-gated")
    stack.build_harness_session.return_value = object()
    events: list[str] = []
    args = object()

    @contextmanager
    def fake_open_episode(selected):
        assert selected is bootstrap
        events.append("enter")
        try:
            yield SimpleNamespace(toolkit=toolkit)
        finally:
            events.append("exit")

    with (
        patch.object(cli, "open_episode", fake_open_episode),
        pytest.raises(RuntimeError, match="attach failed"),
    ):
        with cli._open_episode_toolkit_with_harness(
            env_spec=env_spec,
            args=args,
            output_dir=tmp_path,
            dashboard=None,
            recorder=Recorder(),
            context_policy=object(),
            stack=stack,
            plan_generator=object(),
            predicate_verifier=object(),
        ):
            pytest.fail("runtime must not yield")

    env_spec.build_episode_bootstrap.assert_called_once_with(args, tmp_path, None)
    env_spec.init_runtime.assert_not_called()
    assert events == ["enter", "exit"]


def test_robocasa_nested_planner_gets_the_existing_executor_action_rules(tmp_path):
    from robots.robocasa.prompts.system import system_prompt

    with (
        patch.object(cli, "build_runtime", side_effect=[object(), object()]),
        patch.object(cli, "RuntimePlanGenerator", return_value="generator") as build,
    ):
        bundle = cli._build_planning_runtime_bundle(
            stack=resolve_planning_stack("robocasa-horizon-gated-v1"),
            args=runtime_args(), output_dir=tmp_path, recipe_tag="episode",
            env_name="robocasa", dashboard=None, recorder=Recorder(),
            prompt_vars=ROBOCASA_PROMPT_VARS,
        )
    guidance = build.call_args.kwargs["action_guidance"]
    sections = system_prompt()
    assert sections["BUDGET"] in guidance
    assert sections["ACTION"] in guidance
    assert "TASK-RULES-SENTINEL" in guidance
    assert "{{task_rules}}" not in guidance
    from rpent.planning.manifest import content_hash
    assert bundle.metadata["plan_action_context"] == {
        "version": "executor-action-guidance-v2",
        "sha256": content_hash(guidance),
    }
