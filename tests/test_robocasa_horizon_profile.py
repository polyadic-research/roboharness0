"""RoboCasa recovery must not spend a second, shorter episode budget."""

from types import SimpleNamespace

import pytest

from rpent.envs import get_env_spec
from rpent.harness.hooks import ToolHookContext
from rpent.planning import resolve_planning_stack
from rpent.planning.schemas import ActionFeedback

PROFILE = "robocasa-horizon-gated-v1"


def _feedback(index, *, success=False, truncated=False):
    return ActionFeedback(
        execution_id=f"action-{index}",
        tool_name="move",
        physical=True,
        execution_status="completed",
        environment_success=success,
        environment_truncated=truncated,
    )


def _step():
    return [{"id": "move", "intent": "complete task", "allowed_tools": ["move"]}]


def _harness(tmp_path):
    harness = resolve_planning_stack(PROFILE).build_harness(
        trace_path=tmp_path / "trace.jsonl",
    )
    harness.set_physical_tools({"move"})
    assert harness.state.register_plan("complete task", _step())["ok"]
    return harness


def test_horizon_profile_is_the_robocasa_cli_and_dashboard_default():
    spec = get_env_spec("robocasa")
    assert (
        spec.resolve_planning_profile(SimpleNamespace(planning_profile=None)) == PROFILE
    )
    preset = spec.launch_preset()
    effective = preset.normalize("hybrid", {"task": "CloseDrawer"})
    assert effective["planning_profile"] == PROFILE
    assert "max_turns" not in effective
    assert "max_replans" not in effective


def test_horizon_profile_preserves_explicit_legacy_selection():
    spec = get_env_spec("robocasa")
    old = "robocasa-hooked-gated-v1"
    assert spec.resolve_planning_profile(SimpleNamespace(planning_profile=old)) == old
    effective = spec.launch_preset().normalize(
        "hybrid", {"task": "CloseDrawer", "planning_profile": old}
    )
    assert effective["max_replans"] == 8
    assert effective["max_turns"] == 100


def test_horizon_profile_has_no_independent_replan_or_tool_call_cap():
    stack = resolve_planning_stack(PROFILE)
    assert stack.max_replans is None
    assert not stack.uses_authoritative_session  # Retains the default hooked loop.
    assert stack.require_environment_success
    assert stack.metadata()["effective"]["environment_horizon_only"] is True
    assert "no independent" in stack.prompt_context().lower()
    with pytest.raises(ValueError, match="horizon"):
        resolve_planning_stack(PROFILE, max_replans=8)


def test_horizon_profile_continues_after_old_recovery_caps(tmp_path):
    harness = _harness(tmp_path)
    state = harness.state
    for index in range(110):
        state.record_action_feedback(_feedback(index), input_dict={})
        assert state.verify_step(
            "fail", ["missed target"], "reposition", execution_id=f"action-{index}"
        )["ok"]
        denied = harness.hooks.before_tool(
            ToolHookContext("finish", "finish", {"status": "failure"}, False), state
        )
        assert not denied.allowed
        assert state.replan("try again", _step())["ok"]
    assert state.replans_used == 110
    assert state.replans_remaining is None


def test_completed_plan_without_native_success_remains_recoverable(tmp_path):
    harness = _harness(tmp_path)
    state = harness.state
    state.record_action_feedback(_feedback(1), input_dict={})
    assert state.verify_step("pass", ["local goal met"], "done locally")["ok"]
    assert state.needs_replan
    assert not harness.hooks.before_tool(
        ToolHookContext("finish", "finish", {"status": "success"}, False), state
    ).allowed
    assert state.replan("native task remains unfinished", _step())["ok"]


@pytest.mark.parametrize(
    "success,truncated,status", [(True, False, "success"), (False, True, "failure")]
)
def test_environment_terminal_evidence_allows_finish(
    tmp_path, success, truncated, status
):
    harness = _harness(tmp_path)
    feedback = _feedback(1, success=success, truncated=truncated)
    verdicts = harness.hooks.after_tool(
        ToolHookContext("action-1", "move", {}, True), feedback, harness.state
    )
    for verdict in verdicts:
        harness.state.apply_verdict(verdict, require_execution_id=True)
    assert harness.hooks.before_tool(
        ToolHookContext("finish", "finish", {"status": status}, False), harness.state
    ).allowed


def test_planner_turn_defaults_follow_profile():
    from rpent.planning.stack import resolve_planner_turn_limit

    assert resolve_planner_turn_limit(PROFILE, None) is None
    assert resolve_planner_turn_limit("robocasa-hooked-gated-v1", None) == 100
    assert resolve_planner_turn_limit("robocasa-hooked-gated-v1", 25) == 25
    with pytest.raises(ValueError, match="horizon"):
        resolve_planner_turn_limit(PROFILE, 100)


def test_api_runtime_without_turn_cap_exceeds_old_default():
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from rpent.runtime.api import ApiAgentRuntime
    from rpent.tools.toolkit import Toolkit

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        name = "tick" if calls <= 105 else "finish"
        args = (
            {} if name == "tick" else {"status": "failure", "summary": "test complete"}
        )
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=str(calls))])

    toolkit = Toolkit()
    toolkit.add_tool(
        "tick",
        {
            "name": "tick",
            "description": "Continue the test.",
            "input_schema": {"type": "object", "properties": {}},
        },
        lambda: {"ok": True},
    )
    result = ApiAgentRuntime(FunctionModel(respond)).solve(
        system_prompt="",
        user_message="run",
        toolkit=toolkit,
        max_turns=None,
    )
    assert calls == 106
    assert result.stop_reason == "finish"


def test_horizon_profile_rejects_a_shorter_environment_ceiling():
    from robots.robocasa.runtime import _effective_horizon

    assert _effective_horizon(750, None, horizon_only=True) == 750
    assert _effective_horizon(750, 750, horizon_only=True) == 750
    assert _effective_horizon(750, 10000, horizon_only=True) == 750
    with pytest.raises(ValueError, match="upstream horizon"):
        _effective_horizon(750, 450, horizon_only=True)
    assert _effective_horizon(750, 450) == 450
