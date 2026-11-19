"""RoboCasa prompts must not let local recovery limits end a horizon episode."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

import rpent.cli.main as cli
from robots.robocasa.control_modes import LANGUAGE_POLICY_FAMILIES
from robots.robocasa.prompts.system import (
    burner_control_available,
    system_prompt,
    task_rules,
)
from rpent.envs import get_env_spec
from rpent.harness.hooks import ToolHookContext
from rpent.planning import resolve_planning_stack
from rpent.planning.schemas import ActionFeedback
from rpent.runtime import RuntimePlanGenerator, RuntimeResult

HORIZON = "robocasa-horizon-gated-v1"
# The executor ACTION text that, with the horizon protocol, ended five
# 2026-09-30 episodes early with 200-3494 environment steps still unused.
OLD_EXIT_RULE = "finish failure without another physical attempt"


def _render(task: str, family: str, profile: str = HORIZON) -> str:
    stack = resolve_planning_stack(profile)
    return get_env_spec("robocasa").prompts.render(
        "system",
        variables={
            "harness_protocol": stack.prompt_context(),
            "task_rules": task_rules(task, policy_family=family),
            "policy_skill": "",
        },
    )


def test_executor_rules_grant_no_episode_termination_authority():
    sections = system_prompt()
    local = sections["BUDGET"] + sections["ACTION"]

    assert OLD_EXIT_RULE not in local
    assert "finish failure" not in local
    assert "finish(status='failure'" not in local
    assert "none of them authorizes ending the episode" in sections["BUDGET"]
    assert "Only the HARNESS section defines when finish is allowed" in (
        sections["BUDGET"]
    )
    assert "failure_type='recovery_exhausted'" in sections["ACTION"]
    assert "do not repeat the same pose recovery" in sections["ACTION"]


def test_budget_section_separates_call_step_and_episode_budgets():
    budget = system_prompt()["BUDGET"]

    for name in (
        "max_chunks",
        "transition_budget",
        "max_attempts",
        "episode_budget",
        "steps_remaining",
    ):
        assert name in budget
    assert "exhausting one never implies another" in budget
    assert "do not resend it" in budget
    assert "reply without a tool call" in budget


def test_retry_refusal_blocks_only_the_grasp_tool():
    action = system_prompt()["ACTION"]

    refusal = action.index("When smolvla_pick returns retry_allowed=false")
    assert "never call smolvla_pick again" in action[refusal:]
    assert "continue the original goal with continue_vla" in action[refusal:]
    assert "never issue another physical attempt" not in action
    assert 'continue_vla with step_intent="task_progress" is always a legal' in action


def test_action_states_which_instruction_text_reaches_the_policy():
    action = system_prompt()["ACTION"]

    assert "instruction description" in action
    assert "audit-only" in action
    assert "forwarded" in action
    assert "without coordinates, negations, or names of other objects" in action
    assert "A language-instructed VLA still permits intervene=false" in action
    assert "When the VLA takes a training task ID, the recovery forwards its instruction" in action


@pytest.mark.parametrize(
    "task", ["SlideDishwasherRack", "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter"]
)
def test_stall_intervention_requires_goal_impact_in_executor_and_planner(task):
    expected = (
        "A suspected stall warrants intervention only when fresh evidence shows "
        "that it obstructs progress toward the task goal."
    )
    preclose_rule = (
        "Pre-close wrist evidence of grasp misalignment also justifies adjust_eef "
        "before any failed grasp or stall; correct the pose while the fingers are open."
    )
    action = system_prompt()["ACTION"]
    assert expected in action
    assert action.index(expected) < action.index(preclose_rule)
    variables = {
        "task_rules": task_rules(task, policy_family="pi05"),
        "policy_skill": "",
    }
    executor = _render(task, "pi05")
    planner = cli._planner_action_guidance("robocasa", variables)
    for prompt in (executor, planner):
        assert expected in prompt
        assert preclose_rule in prompt


def test_contact_alone_cannot_shorten_policy_window_or_authorize_stall_correction():
    action = system_prompt()["ACTION"]
    assert "shorter policy windows only when fresh observations show actual regression" in action
    assert "Mere contact or potentially fragile contact is not a trigger" in action
    assert "task-directed progress across all axes" in action
    assert "pressing against the intended target" in action
    assert "blocked axis can be contact doing useful work" in action
    assert "policy has stopped moving" in action
    assert "obstruction is not the intended target" in action
    assert "If uncertain, run one more configured default policy window" in action
    assert "These same stall and timing criteria apply to adjust_eef, move_base and adjust_torso" in action
    assert "use adjust_eef for a grounded physical correction" not in action
    assert "current images show fragile contact or regression" not in action


@pytest.mark.parametrize(
    "task", ["TurnOffStove", "PickPlaceToasterToCounter", "OpenStandMixerHead"]
)
@pytest.mark.parametrize("family", ["pi05", "smolvla"])
def test_non_kettle_tasks_receive_no_kettle_or_burner_procedure(task, family):
    rendered = _render(task, family)

    assert "kettle" not in rendered.lower()
    assert "burner_control" not in rendered
    assert "recommended_standoff_xyz" not in rendered
    assert "No task-specific procedure is configured" in rendered


def test_kettle_rules_follow_the_toolkit_burner_capability():
    with_burner = task_rules("KettleBoiling", policy_family="smolvla")
    without_burner = task_rules("KettleBoiling", policy_family="pi05")

    for rules in (with_burner, without_burner):
        assert "Recovery grasp: acquire and lift the small gray kettle" in rules
        assert "one combined physical plan step" in rules
    assert 'step_intent="burner_control"' in with_burner
    assert "recommended_standoff_xyz" in with_burner
    assert 'step_intent="burner_control"' not in without_burner
    assert "recommended_standoff_xyz" not in without_burner
    assert "offers no burner_control step intent" in without_burner


class _Controller:
    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        raise AssertionError("schema inspection must not execute the policy")


@pytest.mark.parametrize(
    ("task", "family"),
    [
        ("KettleBoiling", "pi05"),
        ("KettleBoiling", "smolvla"),
        ("KettleBoiling", "xiaomi-mibot"),
        ("TurnOffStove", "smolvla"),
    ],
)
def test_burner_capability_mirrors_the_toolkit_schema(task, family):
    from robots.robocasa.toolkit import RoboCasaToolkit

    language = "Boil water in the kettle."
    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation={
            "state": np.zeros(16, dtype=np.float32),
            "robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        instruction=language,
        # The RoboCasa runtime's policy-instruction choice per family.
        policy_instruction=language if family in LANGUAGE_POLICY_FAMILIES else task,
    )
    specs = {spec["name"]: spec for spec in toolkit.get_tools_spec()}
    intents = specs["continue_vla"]["input_schema"]["properties"]["step_intent"]

    assert ("burner_control" in intents["enum"]) is burner_control_available(
        task, policy_family=family
    )


def _parse(argv: list[str]):
    with patch.object(sys, "argv", ["rpent", *argv]):
        parser = cli._build_argparser()
        early, _ = parser.parse_known_args()
        spec = get_env_spec(early.env_name)
        spec.add_cli_args(parser, use_dashboard=False)
        return parser.parse_args()


@pytest.mark.parametrize("family", ["pi05", "smolvla"])
def test_parse_config_selects_task_rules_for_the_policy_family(family):
    args = _parse(["--env", "robocasa", "--task", "KettleBoiling", "--vla", family])
    variables = get_env_spec("robocasa").parse_config(args).prompt_vars

    assert variables["task_rules"] == task_rules(
        "KettleBoiling", policy_family=family
    )


def test_system_render_requires_task_rules():
    with pytest.raises(KeyError, match="task_rules"):
        get_env_spec("robocasa").prompts.render(
            "system", variables={"harness_protocol": "protocol"}
        )


def test_horizon_protocol_defines_termination_and_rejected_finish_recovery():
    rendered = _render("PickPlaceDrawerToCounter", "pi05")

    assert OLD_EXIT_RULE not in rendered
    assert "TERMINATION: finish is accepted only in three cases" in rendered
    assert "Harness rejects finish(status='failure')" in rendered
    assert "never reply without a tool call" in rendered
    assert "INCOMPLETE protocol error" in rendered
    assert "replan then register_plan in plan mode" in rendered
    assert "Do not give up merely because a grasp, subgoal, or recovery failed" in (
        rendered
    )


def test_legacy_hooked_protocol_keeps_its_own_failure_termination():
    rendered = _render("PickPlaceDrawerToCounter", "pi05", "robocasa-hooked-gated-v1")

    assert "TERMINATION PROTOCOL" in rendered
    assert "replan budget is exhausted" in rendered
    assert OLD_EXIT_RULE not in rendered


def test_nested_planner_receives_rendered_budget_action_and_task_rules(tmp_path):
    class Recorder:
        def record(self, *_args, **_kwargs):
            return None

    args = type(
        "Args",
        (),
        {
            "planner": "api",
            "planner_timeout_s": None,
            "plan_generation_timeout_s": 180,
            "no_images": False,
            "base_url": None,
            "model": "test",
            "max_tokens": 0,
            "claude_code_max_budget_usd": None,
        },
    )()
    variables = {
        "task_rules": task_rules("TurnOffStove", policy_family="pi05"),
        "policy_skill": "",
    }
    with (
        patch.object(cli, "build_runtime", side_effect=[object(), object()]),
        patch.object(cli, "RuntimePlanGenerator", return_value="generator") as build,
    ):
        cli._build_planning_runtime_bundle(
            stack=resolve_planning_stack(HORIZON),
            args=args,
            output_dir=tmp_path,
            recipe_tag="episode",
            env_name="robocasa",
            dashboard=None,
            recorder=Recorder(),
            prompt_vars=variables,
        )

    guidance = build.call_args.kwargs["action_guidance"]
    assert "BUDGET" in guidance and "ACTION" in guidance and "TASK RULES" in guidance
    assert "No task-specific procedure is configured" in guidance
    assert "kettle" not in guidance.lower()
    assert "{{" not in guidance


class _PlanRuntime:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def solve(self, *, system_prompt, user_message, toolkit, max_turns, input_queue=None):
        del user_message, toolkit, max_turns, input_queue
        self.system_prompts.append(system_prompt)
        return RuntimeResult(
            finish_result={
                "_finish": True,
                "strategy": "Keep the policy acquiring the grounded whisk.",
                "steps": [
                    {
                        "id": "acquire",
                        "intent": "Acquire the whisk from the open drawer.",
                        "allowed_tools": ["continue_vla"],
                        "expected_evidence": ["Fresh views show the whisk lifted."],
                    }
                ],
            }
        )


def test_nested_planner_finish_returns_a_plan_and_never_plans_an_exit():
    runtime = _PlanRuntime()
    generator = RuntimePlanGenerator(runtime, require_outcome_program=False)

    generator.generate(
        goal="Pick the whisk from the drawer.",
        observations=["The whisk is in the open drawer."],
        constraints=[],
        physical_tools={"continue_vla"},
    )
    generator.generate_replan(
        goal="Pick the whisk from the drawer.",
        reason="The grasp missed; the whisk remains in the drawer.",
        observations=["The whisk is still supported by the drawer bottom."],
        constraints=["episode_budget steps_remaining=550"],
        current_plan={"strategy": "grasp", "unfinished_steps": []},
        physical_tools={"continue_vla"},
    )

    initial, replan = runtime.system_prompts
    for prompt in (initial, replan):
        assert "neither executes the plan nor ends the episode" in prompt
        assert "acknowledge, record, or declare failure" in prompt
        assert "never as a request to stop the episode" in prompt
        assert "by itself it never ends the episode" in prompt
    assert "Renaming a step as a final attempt is not a change" in replan
    assert "executable condition the replacement changes" in replan


def _feedback(index: int, *, truncated: bool = False) -> ActionFeedback:
    return ActionFeedback(
        execution_id=f"action-{index}",
        tool_name="move",
        physical=True,
        execution_status="completed",
        environment_truncated=truncated,
    )


def _harness(tmp_path, profile: str = HORIZON):
    harness = resolve_planning_stack(profile).build_harness(
        trace_path=tmp_path / "trace.jsonl",
    )
    harness.set_physical_tools({"move"})
    step = [{"id": "move", "intent": "complete task", "allowed_tools": ["move"]}]
    assert harness.state.register_plan("complete task", step)["ok"]
    return harness


def test_harness_reminder_names_the_next_legal_operation(tmp_path):
    harness = _harness(tmp_path)
    state = harness.state

    reminder = harness.protocol_continuation()
    assert "no finish has been accepted" in reminder
    assert "['move']" in reminder

    state.record_action_feedback(_feedback(1), input_dict={})
    assert "verify_step with execution_id='action-1'" in (
        harness.protocol_continuation()
    )

    assert state.verify_step(
        "fail", ["missed target"], "reposition", execution_id="action-1"
    )["ok"]
    assert "call replan" in harness.protocol_continuation()


def test_harness_reminder_requests_failure_finish_after_truncation(tmp_path):
    harness = _harness(tmp_path)
    feedback = _feedback(1, truncated=True)
    verdicts = harness.hooks.after_tool(
        ToolHookContext("action-1", "move", {}, True), feedback, harness.state
    )
    for verdict in verdicts:
        harness.state.apply_verdict(verdict, require_execution_id=True)

    reminder = harness.protocol_continuation()
    assert "environment_ended=True" in reminder
    assert "finish(status='failure'" in reminder
    assert harness.hooks.before_tool(
        ToolHookContext("finish", "finish", {"status": "failure"}, False),
        harness.state,
    ).allowed


def test_only_the_horizon_harness_requests_a_continuation(tmp_path):
    assert _harness(tmp_path, "robocasa-hooked-gated-v1").protocol_continuation() is None

    from rpent.tools.toolkit import Toolkit

    assert Toolkit().protocol_continuation() is None


def _api_solve(respond, toolkit):
    from pydantic_ai.models.function import FunctionModel

    from rpent.runtime.api import ApiAgentRuntime

    return ApiAgentRuntime(FunctionModel(respond)).solve(
        system_prompt="",
        user_message="run",
        toolkit=toolkit,
        max_turns=None,
    )


def _reminder_toolkit():
    from rpent.tools.toolkit import Toolkit

    class ReminderToolkit(Toolkit):
        def __init__(self) -> None:
            super().__init__()
            self.add_tool(
                "tick",
                {
                    "name": "tick",
                    "description": "Continue the test.",
                    "input_schema": {"type": "object", "properties": {}},
                },
                lambda: {"ok": True},
            )

        def protocol_continuation(self) -> str:
            return "HARNESS PROTOCOL: the episode is still open."

    return ReminderToolkit()


def test_api_runtime_resumes_after_a_reply_without_a_tool_call():
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart("Task failed: cannot grasp.")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "finish",
                    {"status": "failure", "summary": "horizon ended"},
                    tool_call_id="finish-1",
                )
            ]
        )

    result = _api_solve(respond, _reminder_toolkit())

    assert calls == 2
    assert result.stop_reason == "finish"
    assert result.stats["protocol_continuations"] == 1
    assert {
        "role": "user",
        "content": "HARNESS PROTOCOL: the episode is still open.",
    } in result.messages


def test_api_runtime_stops_after_two_idle_reminders():
    from pydantic_ai.messages import ModelResponse, TextPart

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("Task failed.")])

    result = _api_solve(respond, _reminder_toolkit())

    assert calls == 3
    assert result.stop_reason == "natural_end"
    assert result.finish_result is None
    assert result.stats["protocol_continuations"] == 2


def test_api_runtime_caps_total_reminders_per_solve():
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls % 2:
            return ModelResponse(parts=[TextPart("Task failed.")])
        return ModelResponse(parts=[ToolCallPart("tick", {}, tool_call_id=str(calls))])

    result = _api_solve(respond, _reminder_toolkit())

    assert calls == 7
    assert result.stop_reason == "natural_end"
    assert result.stats["protocol_continuations"] == 3


def test_api_runtime_without_harness_still_stops_on_prose():
    from pydantic_ai.messages import ModelResponse, TextPart

    from rpent.tools.toolkit import Toolkit

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("done")])

    result = _api_solve(respond, Toolkit())

    assert calls == 1
    assert result.stop_reason == "natural_end"
    assert "protocol_continuations" not in result.stats
