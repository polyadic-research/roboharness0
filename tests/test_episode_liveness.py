"""Bound model-only loops independently of the physical episode horizon."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from rpent.episode.liveness import EpisodeBudget, EpisodeBudgetExceeded, current_budget
from rpent.runtime.api import ApiAgentRuntime
from rpent.runtime.plan_generation import RuntimePlanGenerator
from rpent.runtime.termination import classify_termination
from rpent.tools.toolkit import Toolkit
from tests.test_plan_generation import FakePlanningRuntime


def looping_toolkit(budget):
    toolkit = Toolkit()
    toolkit.episode_budget = budget
    toolkit.add_tool("tick", {"name": "tick", "description": "Read only.",
                             "input_schema": {"type": "object", "properties": {}}},
                     lambda: {"ok": True})
    return toolkit


def test_zero_action_model_loop_stops_before_request_501():
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("tick", {}, tool_call_id=str(calls))])

    budget = EpisodeBudget(max_requests=500, max_seconds=5400)
    result = ApiAgentRuntime(FunctionModel(respond)).solve(
        system_prompt="", user_message="run", toolkit=looping_toolkit(budget), max_turns=None,
    )
    assert calls == 500
    assert result.stats["tool_calls"] == 500
    assert result.stop_reason == "episode_budget_exhausted"
    assert result.stats["episode_budget"]["exhausted_by"] == "llm_requests"
    assert classify_termination(finish_result=result.finish_result, error=result.error,
                                harness_snapshot={}, stop_reason=result.stop_reason).classification == "FAILED"


def test_deadline_cancels_an_outstanding_model_request():
    async def respond(messages, info):
        await asyncio.sleep(60)
        return ModelResponse(parts=[])

    result = ApiAgentRuntime(FunctionModel(respond)).solve(
        system_prompt="", user_message="run", max_turns=None,
        toolkit=looping_toolkit(EpisodeBudget(max_requests=500, max_seconds=0.03)),
    )
    assert result.stop_reason == "episode_budget_exhausted"
    assert result.stats["episode_budget"]["exhausted_by"] == "wall_time"
    assert result.error


def test_budget_counts_nested_planner_requests_across_its_worker_thread():
    seen = []

    class NestedRuntime(FakePlanningRuntime):
        def solve(self, **kwargs):
            budget = current_budget(kwargs["toolkit"])
            seen.append(budget)
            budget.reserve_request()
            return super().solve(**kwargs)

    budget = EpisodeBudget(max_requests=2, max_seconds=5400)
    toolkit = looping_toolkit(budget)
    generator = RuntimePlanGenerator(NestedRuntime(tool="move"), require_outcome_program=False)
    toolkit.add_tool("plan", {"name": "plan", "description": "Nested planning.",
                             "input_schema": {"type": "object", "properties": {}}},
                     lambda: generator.generate(goal="Move.", observations=["Visible."],
                                                constraints=[], physical_tools={"move"}))
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart("plan", {}, tool_call_id=str(calls))])

    result = ApiAgentRuntime(FunctionModel(respond)).solve(
        system_prompt="", user_message="run", toolkit=toolkit, max_turns=None,
    )
    assert seen == [budget]
    assert calls == 1
    assert result.stop_reason == "episode_budget_exhausted"
    assert result.stats["episode_budget"]["llm_requests"] == 2


def test_expired_episode_never_dispatches_another_tool():
    clock = [0.0]
    budget = EpisodeBudget(max_requests=500, max_seconds=5400, clock=lambda: clock[0])
    toolkit = looping_toolkit(budget)
    budget.start()
    clock[0] = 5400
    rejected = toolkit.execute_tool("tick", {}).result
    assert rejected["stop_reason"] == "episode_budget_exhausted"
    assert rejected["error"]


def test_request_budget_is_atomic_and_does_not_reset_on_start():
    budget = EpisodeBudget(max_requests=20, max_seconds=5400)
    budget.start()

    def request(_):
        try:
            budget.reserve_request()
            return True
        except EpisodeBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(request, range(50))) == 20
    budget.start()
    assert budget.snapshot()["llm_requests"] == 20


@pytest.mark.parametrize("status", [None, "success"])
def test_budget_stop_is_failed_without_fabricating_a_finish(status):
    result = classify_termination(
        finish_result=None if status is None else {"status": status}, error=None,
        harness_snapshot={}, stop_reason="episode_budget_exhausted",
    )
    assert result.classification == "FAILED"
    assert result.error_code == "episode_budget_exhausted"


def test_last_allowed_request_can_finish_without_being_marked_exhausted():
    toolkit = looping_toolkit(EpisodeBudget(max_requests=1, max_seconds=5400))
    toolkit.add_tool("finish", {"name": "finish", "description": "End the test.",
                               "input_schema": {"type": "object", "properties": {}}},
                     lambda: {"_finish": True, "status": "failure"})
    result = ApiAgentRuntime(FunctionModel(lambda messages, info: ModelResponse(
        parts=[ToolCallPart("finish", {}, tool_call_id="done")]))).solve(
        system_prompt="", user_message="run", toolkit=toolkit, max_turns=None,
    )
    assert result.stop_reason == "finish"
    assert result.stats["episode_budget"]["exhausted_by"] is None






def test_shorter_provider_cancellation_does_not_exhaust_the_episode():
    async def respond(messages, info):
        await asyncio.sleep(60)
        return ModelResponse(parts=[])

    budget = EpisodeBudget(max_requests=500, max_seconds=5400)
    runtime = ApiAgentRuntime(FunctionModel(respond))

    async def cancel():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime._solve(
                system_prompt="", user_message="run", toolkit=looping_toolkit(budget),
                max_turns=None), timeout=0.03)

    asyncio.run(cancel())
    assert budget.snapshot()["exhausted_by"] is None
    assert budget.snapshot()["llm_requests"] == 1
