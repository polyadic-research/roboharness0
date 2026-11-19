"""Grounded scene handoff and read-only planner clarification regressions."""

import json
from copy import deepcopy

import pytest

from rpent.harness import Harness
from rpent.harness.lifecycle import PLANNING_INFORMATION_SCOPE
from rpent.planning.information import PLANNING_INFORMATION_GUIDANCE
from rpent.runtime import RuntimePlanGenerator, RuntimeResult
from rpent.tools.toolkit import Toolkit


def objects(step=0, door="unknown"):
    return [
        {
            "id": "fridge",
            "identity": "gray refrigerator",
            "location": "behind counter",
            "states": {"door": door, "interior_support": "unknown"},
            "relations": [],
            "evidence": [f"step {step} left camera"],
            "observation_step": step,
            "uncertainty": "interior support not visible",
        }
    ]


QUESTION = [
    {
        "object_id": "fridge",
        "fields": ["door", "interior_support"],
        "question": "Is the door open and is a bowl support visible?",
        "reason": "Choose access and placement prerequisites.",
    }
]
PLAN = {
    "strategy": "transport to the already open refrigerator",
    "steps": [
        {
            "id": "store",
            "intent": "place bowl on the fridge shelf",
            "allowed_tools": ["move"],
            "expected_evidence": ["bowl released on shelf"],
        }
    ],
}


class Planner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return deepcopy(self.outputs.pop(0))

    generate_replan = generate


def harness(tmp_path, planner):
    return Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=planner,
        require_object_states=True,
    )


def test_scene_handoff_is_required_and_validated_before_model_call(tmp_path):
    planner = Planner([PLAN])
    h = harness(tmp_path, planner)
    assert (
        "object_states" in h.create_plan("store bowl", ["there is a fridge"])["error"]
    )
    invalid = objects()
    invalid[0]["states"] = {}
    assert (
        "states"
        in h.create_plan("store bowl", ["fridge visible"], object_states=invalid)[
            "error"
        ]
    )
    assert not planner.calls
    result = h.create_plan(
        "store bowl", ["door is open"], object_states=objects(door="open")
    )
    assert result["proposal_id"]
    assert planner.calls[0]["object_states"][0]["states"]["door"] == "open"


def test_question_and_answer_keep_goal_and_do_not_activate_or_move(tmp_path):
    planner = Planner([{"information_request": QUESTION}, PLAN])
    h = harness(tmp_path, planner)
    request = h.create_plan("store bowl", ["fridge visible"], object_states=objects())
    assert request["status"] == "information_required"
    assert "proposal_id" not in request
    assert not h.state.has_plan
    assert not h.before_tool("move", {})["allowed"]
    wrong = h.provide_planner_information("wrong", ["door open"], objects(door="open"))
    assert "error" in wrong
    assert len(planner.calls) == 1
    assert "error" in h.create_plan(
        "change goal", ["ignore question"], object_states=objects()
    )
    answer = h.provide_planner_information(
        request["request_id"], ["door open; shelf unknown"], objects(door="open")
    )
    assert answer["proposal_id"]
    assert not h.state.has_plan
    assert planner.calls[1]["goal"] == "store bowl"
    assert planner.calls[1]["information_history"][0]["questions"] == QUESTION
    assert (
        planner.calls[1]["object_states"][0]["states"]["interior_support"] == "unknown"
    )
    assert h.register_plan(answer["proposal_id"])["ok"]


def test_stale_or_incomplete_answer_does_not_consume_round(tmp_path):
    planner = Planner([{"information_request": QUESTION}, PLAN])
    h = harness(tmp_path, planner)
    request = h.create_plan("store bowl", ["uncertain"], object_states=objects(step=10))
    stale = h.provide_planner_information(
        request["request_id"], ["old view"], objects(step=0)
    )
    assert "stale" in stale["error"]
    missing = objects(step=10)
    del missing[0]["states"]["door"]
    assert (
        "door"
        in h.provide_planner_information(request["request_id"], ["view"], missing)[
            "error"
        ]
    )
    assert len(planner.calls) == 1


def test_clarification_budget_fails_closed(tmp_path):
    planner = Planner([{"information_request": QUESTION}] * 3)
    h = harness(tmp_path, planner)
    result = h.create_plan("store bowl", ["uncertain"], object_states=objects())
    for _ in range(2):
        result = h.provide_planner_information(
            result["request_id"], ["still unknown"], objects()
        )
    assert result["status"] == "information_budget_exhausted"
    assert not h.state.has_plan
    assert not h.before_tool("move", {})["allowed"]
    assert "error" in h.provide_planner_information(
        result["request_id"], ["unknown"], objects()
    )
    assert len(planner.calls) == 3
    h.state.environment_horizon_only = True
    assert h.protocol_continuation() is None
    assert (
        h.snapshot()["planner_information"]["status"] == "information_budget_exhausted"
    )


def test_toolkit_blocks_motion_and_keeps_pending_questions_visible(tmp_path):
    planner = Planner([{"information_request": QUESTION}, PLAN])
    h = harness(tmp_path, planner)
    toolkit = Toolkit()
    moved = []
    toolkit.add_tool(
        "move",
        {"name": "move", "input_schema": {"type": "object"}},
        lambda: moved.append(True) or {"ok": True},
    )
    toolkit.configure_harness_physical_tools({"move"})
    toolkit.attach_harness(h)
    request = toolkit.execute_tool(
        "create_plan",
        {
            "goal": "store bowl",
            "observations": ["fridge visible"],
            "object_states": objects(),
        },
    ).result
    assert request["harness"]["planner_information"]["questions"] == QUESTION
    blocked = toolkit.execute_tool("move", {}).result
    assert "pending planner information" in blocked["error"]
    assert not moved
    answer = toolkit.execute_tool(
        "provide_planner_information",
        {
            "request_id": request["request_id"],
            "observations": ["door open"],
            "object_states": objects(door="open"),
        },
    ).result
    assert answer["proposal_id"]
    assert "planner_information" not in answer["harness"]


def test_disabled_controller_cannot_enter_unanswerable_information_protocol(tmp_path):
    planner = Planner([{"information_request": QUESTION}])
    h = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=planner,
    )
    result = h.create_plan("store bowl", ["fridge visible"])
    assert "not enabled" in result["error"]
    assert h._pending_information is None


def test_replan_clarification_preserves_recovery_budget_and_completed_context(tmp_path):
    planner = Planner([PLAN, {"information_request": QUESTION}, PLAN])
    h = harness(tmp_path, planner)
    initial = h.create_plan("store bowl", ["visible"], object_states=objects())
    h.register_plan(initial["proposal_id"])
    h.after_tool("move", {}, {"ok": True})
    h.verify_step("fail", ["transport stalled"], "stalled")
    budget = h.state.replans_remaining
    request = h.replan(
        "transport stalled", ["support unknown"], object_states=objects()
    )
    assert request["status"] == "information_required"
    assert h.state.replans_remaining == budget
    answer = h.provide_planner_information(
        request["request_id"], ["door open"], objects(door="open")
    )
    assert answer["proposal_type"] == "replan"
    assert h.state.replans_remaining == budget
    assert planner.calls[-1]["current_plan"] == planner.calls[-2]["current_plan"]
    h.register_plan(answer["proposal_id"])
    assert h.state.replans_remaining == budget - 1


class RequestRuntime:
    def solve(self, *, toolkit, user_message, system_prompt, **kwargs):
        self.payload = json.loads(user_message)
        self.prompt = system_prompt
        self.spec = next(x for x in toolkit.get_tools_spec() if x["name"] == "finish")
        result = toolkit.execute_tool("finish", {"information_request": QUESTION})
        return RuntimeResult(finish_result=result.result)


def test_runtime_returns_questions_without_parsing_or_fabricating_plan():
    runtime = RequestRuntime()
    generator = RuntimePlanGenerator(runtime, allow_information_requests=True)
    result = generator.generate(
        goal="store bowl",
        observations=["fridge visible"],
        constraints=[],
        physical_tools={"move"},
        object_states=objects(),
    )
    assert result == {"information_request": QUESTION}
    assert runtime.payload["object_states"] == objects()
    assert "information_request" in runtime.spec["input_schema"]["properties"]
    assert "already satisfied" in runtime.prompt


def test_next_action_information_scope_is_shared_by_executor_and_planner():
    # The real D07 preflight asked for an exact destination before grounded
    # food acquisition. Both sides must receive the same prerequisite scope.
    runtime = RequestRuntime()
    RuntimePlanGenerator(runtime, allow_information_requests=True).generate(
        goal="Put both foods in the bowl, then put the bowl in the fridge.",
        observations=["Foods and empty bowl visible; fridge door already open."],
        constraints=[],
        physical_tools={"move"},
        object_states=objects(),
    )
    assert PLANNING_INFORMATION_SCOPE in runtime.prompt
    assert PLANNING_INFORMATION_SCOPE in PLANNING_INFORMATION_GUIDANCE


@pytest.mark.parametrize("bad", [[], [{"object_id": "fridge", "question": "door?"}]])
def test_runtime_rejects_malformed_questions(bad):
    class BadRuntime(RequestRuntime):
        def solve(self, **kwargs):
            return RuntimeResult(
                finish_result={"_finish": True, "information_request": bad}
            )

    with pytest.raises(ValueError):
        RuntimePlanGenerator(BadRuntime(), allow_information_requests=True).generate(
            goal="store",
            observations=["unknown"],
            constraints=[],
            physical_tools={"move"},
        )


def test_chinese_planning_requires_chinese_question_and_reason(tmp_path):
    h = harness(tmp_path, Planner([{"information_request": QUESTION}]))
    h.use_chinese = True
    result = h.create_plan("把碗放入冰箱", ["冰箱门状态未知"], object_states=objects())
    assert "Chinese" in result["error"]
    assert h._pending_information is None
    with pytest.raises(ValueError, match="Chinese"):
        RuntimePlanGenerator(
            RequestRuntime(), allow_information_requests=True, use_chinese=True
        ).generate(
            goal="把碗放入冰箱",
            observations=["冰箱门状态未知"],
            constraints=[],
            physical_tools={"move"},
            object_states=objects(),
        )


def test_handoff_retains_original_evidence_in_question_history(tmp_path):
    planner = Planner([{"information_request": QUESTION}, PLAN])
    h = harness(tmp_path, planner)
    original = objects()
    request = h.create_plan(
        "store bowl", ["original source identity"], object_states=original
    )
    h.provide_planner_information(
        request["request_id"], ["door open"], objects(door="open")
    )
    history = planner.calls[1]["information_history"][0]
    assert history["request_observations"] == ["original source identity"]
    assert history["request_object_states"] == original


def test_failed_planner_calls_cannot_bypass_answer_exchange_budget(tmp_path):
    class FailingPlanner(Planner):
        def generate(self, **kwargs):
            if self.calls:
                self.calls.append(deepcopy(kwargs))
                raise RuntimeError("provider unavailable")
            return super().generate(**kwargs)

    planner = FailingPlanner([{"information_request": QUESTION}])
    h = harness(tmp_path, planner)
    request = h.create_plan("store bowl", ["unknown"], object_states=objects())
    assert request["remaining_rounds"] == 2
    first = h.provide_planner_information(
        request["request_id"], ["door open"], objects(door="open")
    )
    assert first["harness"]["planner_information"]["remaining_rounds"] == 1
    second = h.provide_planner_information(
        request["request_id"], ["door open"], objects(door="open")
    )
    assert (
        second["harness"]["planner_information"]["status"]
        == "information_budget_exhausted"
    )
    assert "error" in h.provide_planner_information(
        request["request_id"], ["door open"], objects(door="open")
    )
    assert len(planner.calls) == 3
    assert not h.before_tool("move", {})["allowed"]
