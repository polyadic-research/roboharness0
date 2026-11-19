"""Completed local steps must be revisited when the simulator still says no."""

from copy import deepcopy

import pytest

from rpent.harness import Harness


class ReviewPlanner:
    def __init__(self):
        self.contexts = []

    def generate(self, **kwargs):
        return {
            "goal": kwargs["goal"],
            "strategy": "place the cup, then close the door",
            "steps": [self.step("place"), self.step("close")],
        }

    @staticmethod
    def step(name):
        return {
            "id": name,
            "intent": name,
            "operation": name,
            "allowed_tools": ["move"],
            "expected_evidence": [f"fresh images confirm {name}"],
            "max_attempts": 3,
        }

    def generate_replan(self, **kwargs):
        self.contexts.append(deepcopy(kwargs["current_plan"]))
        return {
            "goal": kwargs["goal"],
            "strategy": "repair the cup placement while preserving the closed door",
            "steps": [self.step("place")],
        }


def controller(tmp_path, *, horizon=True):
    planner = ReviewPlanner()
    harness = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=planner,
        require_environment_success=True,
        require_execution_id=True,
        environment_horizon_only=horizon,
        max_replans=None if horizon else 2,
    )
    proposal = harness.create_plan("put cup on tray and close door", ["cup visible"])
    assert harness.register_plan(proposal["proposal_id"])["ok"]
    return harness, planner


def pass_step(harness, execution_id, *, success=False, truncated=False):
    harness.after_tool(
        "move", {},
        {"environment": {"success": success, "terminated": False, "truncated": truncated}},
        execution_id=execution_id,
    )
    return harness.verify_step(
        "pass", ["looks complete in the current image"], "local step accepted",
        execution_id=execution_id,
    )


def complete_plan(harness):
    pass_step(harness, "place-1")
    pass_step(harness, "close-1")


def observe(harness, observation_id="review-view"):
    return harness.after_tool("observe", {}, {"step": 40}, execution_id=observation_id)


def reviews(harness, *, verdict="needs_repair"):
    return [
        {
            "fact_id": fact["fact_id"],
            "verdict": verdict if fact["step_id"] == "place" else "confirmed",
            "evidence": ["cup remains too far from the tray centre" if fact["step_id"] == "place" else "door is closed"],
        }
        for fact in harness.snapshot()["completion_review"]["facts"]
    ]


def repair(harness, assessments=None, observation_id="review-view"):
    return harness.replan(
        "all local steps passed but the environment is not successful",
        ["fresh view shows the cup near the rim and the door closed"],
        completed_step_reviews=reviews(harness) if assessments is None else assessments,
        review_observation_id=observation_id,
    )


def test_completed_plan_requires_review_before_planning_or_physical_action(tmp_path):
    harness, planner = controller(tmp_path)
    complete_plan(harness)
    pending = harness.snapshot()["completion_review"]
    assert pending["required"] is True
    assert [fact["step_id"] for fact in pending["facts"]] == ["place", "close"]
    assert pending["facts"][0]["expected_evidence"] == ["fresh images confirm place"]
    assert harness.snapshot()["contract_success"] is False
    assert not harness.before_tool("move", {})["allowed"]
    result = harness.replan("continue", ["door looks closed"])
    assert "review" in result["error"]
    assert planner.contexts == []
    assert "review" in harness.protocol_continuation()


def test_final_pass_tool_response_advertises_the_review(tmp_path):
    harness, _ = controller(tmp_path)
    pass_step(harness, "place-1")
    result = pass_step(harness, "close-1")
    assert result["harness"]["completion_review"]["required"]


def test_cached_proposal_cannot_skip_review_after_later_completion(tmp_path):
    harness, _ = controller(tmp_path)
    pass_step(harness, "place-1")
    harness.after_tool("move", {}, {"step": 20}, execution_id="close-miss")
    harness.verify_step("retry", ["door partly open"], "try again", execution_id="close-miss")
    stale_proposal = harness.replan("retry closing", ["door still open"])
    assert stale_proposal["ok"]
    pass_step(harness, "close-1")
    assert "review" in harness.register_plan(stale_proposal["proposal_id"])["error"]
    observe(harness)
    new_proposal = repair(harness)
    assert new_proposal["ok"]
    assert "review" in harness.register_plan(stale_proposal["proposal_id"])["error"]
    assert harness.register_plan(new_proposal["proposal_id"])["ok"]


def test_new_observation_invalidates_a_previously_reviewed_proposal(tmp_path):
    harness, _ = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    proposal = repair(harness)
    observe(harness, "newer-view")
    assert "review" in harness.register_plan(proposal["proposal_id"])["error"]


def test_failed_observation_cannot_supply_review_evidence(tmp_path):
    harness, _ = controller(tmp_path)
    complete_plan(harness)
    harness.after_tool("observe", {}, {"error": "camera failed", "step": 40}, execution_id="failed-view")
    assert "observe" in repair(harness, observation_id="failed-view")["error"]


def test_runtime_replanner_may_revisit_passed_work_from_review(tmp_path):
    from rpent.runtime import RuntimePlanGenerator, RuntimeResult

    harness, planner = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    assert repair(harness)["ok"]

    class Runtime:
        def solve(self, *, toolkit, system_prompt, **kwargs):
            self.prompt = system_prompt
            result = {
                "strategy": "repair cup position; keep the closed door",
                "steps": [ReviewPlanner.step("place")],
            }
            toolkit.execute_tool("finish", result)
            return RuntimeResult(finish_result={"_finish": True, **result})

    runtime = Runtime()
    result = RuntimePlanGenerator(runtime, require_outcome_program=False).generate_replan(
        goal="put cup on tray and close door", reason="native success is false",
        observations=["cup at tray rim"], constraints=[],
        current_plan=planner.contexts[-1], physical_tools={"move"},
    )
    assert result["steps"][0]["id"] == "place"
    assert "Earlier passed steps may need repair" in runtime.prompt
    assert "sticky facts" not in runtime.prompt
    assert "cannot authorize finish(success)" in runtime.prompt


def test_review_requires_an_observation_after_final_local_acceptance(tmp_path):
    harness, planner = controller(tmp_path)
    pass_step(harness, "place-1")
    observe(harness, "too-early")
    pass_step(harness, "close-1")
    result = repair(harness, observation_id="too-early")
    assert "observe" in result["error"]
    assert planner.contexts == []
    observe(harness)
    assert repair(harness, observation_id="invented-view").get("error")
    assert repair(harness)["ok"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "empty_evidence", "invalid_verdict"])
def test_review_rejects_incomplete_or_unbound_assessments_atomically(tmp_path, problem):
    harness, planner = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    assessments = reviews(harness)
    if problem == "missing":
        assessments.pop()
    elif problem == "duplicate":
        assessments.append(deepcopy(assessments[0]))
    elif problem == "unknown":
        assessments[0]["fact_id"] = "not-a-passed-step"
    elif problem == "empty_evidence":
        assessments[0]["evidence"] = [" "]
    else:
        assessments[0]["verdict"] = "success"
    before = deepcopy(harness.snapshot())
    assert repair(harness, assessments).get("error")
    assert harness.snapshot() == before
    assert planner.contexts == []


@pytest.mark.parametrize("verdict", ["needs_repair", "uncertain"])
def test_review_reopens_earlier_work_without_erasing_history_or_budgets(tmp_path, verdict):
    harness, planner = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    accounting_before = harness.state.control_planning_context()["control_subtasks"]
    proposal = repair(harness, reviews(harness, verdict=verdict))
    assert proposal["ok"]
    context = planner.contexts[-1]
    assert [fact["step_id"] for fact in context["verified_facts"]] == ["close"]
    assert len(context["completed_step_history"]) == 2
    assert context["completed_step_reviews"][0]["verdict"] == verdict
    assert context["completion_review"]["environment_success"] is False
    assert harness.snapshot()["replans_used"] == 0
    accounting_after = harness.state.control_planning_context()["control_subtasks"]
    assert [dict(row, passed=None) for row in accounting_after] == [
        dict(row, passed=None) for row in accounting_before
    ]
    assert not harness.before_tool("move", {})["allowed"]
    assert harness.register_plan(proposal["proposal_id"])["ok"]
    subtasks = harness.state.control_planning_context()["control_subtasks"]
    assert next(row for row in subtasks if row["operation_key"] == ["operation", "place"])["passed"] is False
    assert next(row for row in subtasks if row["operation_key"] == ["operation", "close"])["passed"] is True
    assert harness.snapshot()["replans_used"] == 1
    assert harness.before_tool("move", {})["allowed"]
    pass_step(harness, "repair-1")
    assert harness.snapshot()["completion_review"]["required"] is True
    assert len(harness.snapshot()["completion_review"]["facts"]) == 3
    assert harness.snapshot()["contract_success"] is False


def test_all_confirmed_reviews_cannot_authorize_global_success(tmp_path):
    harness, planner = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    assessments = reviews(harness, verdict="confirmed")
    assert repair(harness, assessments)["ok"]
    assert harness.snapshot()["contract_success"] is False
    assert not harness.before_tool("finish", {"status": "success"})["allowed"]
    assert planner.contexts[-1]["completion_review"]["environment_success"] is False


@pytest.mark.parametrize("latest_verdict", ["confirmed", "needs_repair"])
def test_latest_review_controls_repeated_operation_credit(tmp_path, latest_verdict):
    harness, planner = controller(tmp_path)
    complete_plan(harness)
    observe(harness)
    proposal = repair(harness)
    harness.register_plan(proposal["proposal_id"])
    pass_step(harness, "repair-1")
    observe(harness, "second-review")
    assessments = reviews(harness)
    assessments[0]["verdict"] = "needs_repair" if latest_verdict == "confirmed" else "confirmed"
    assessments[-1]["verdict"] = latest_verdict
    assert repair(harness, assessments, observation_id="second-review")["ok"]
    confirmed = planner.contexts[-1]["verified_facts"]
    placement = [fact for fact in confirmed if fact["step_id"] == "place"]
    assert len(placement) == (1 if latest_verdict == "confirmed" else 0)
    if placement:
        assert placement[0]["execution_id"] == "repair-1"
    subtasks = harness.state.control_planning_context()["control_subtasks"]
    assert next(row for row in subtasks if row["operation_key"] == ["operation", "place"])["passed"] is (latest_verdict == "confirmed")


@pytest.mark.parametrize("ending", ["success", "truncated"])
def test_native_terminal_signals_do_not_require_local_review(tmp_path, ending):
    harness, _ = controller(tmp_path)
    pass_step(harness, "place-1")
    pass_step(harness, "close-1", success=ending == "success", truncated=ending == "truncated")
    assert not harness.snapshot().get("completion_review", {}).get("required", False)
    assert harness.before_tool("finish", {"status": "success" if ending == "success" else "failure"})["allowed"]


def test_ordinary_step_failure_does_not_trigger_whole_plan_review(tmp_path):
    harness, planner = controller(tmp_path)
    pass_step(harness, "place-1")
    harness.after_tool("move", {}, {"step": 30}, execution_id="close-1")
    harness.verify_step("fail", ["door is still open"], "closing failed", execution_id="close-1")
    assert harness.replan("repair door", ["door is open"])["ok"]
    assert len(planner.contexts[-1]["verified_facts"]) == 1


def test_review_schema_is_exposed_only_in_environment_horizon_loop(tmp_path):
    harness, _ = controller(tmp_path)
    schema = next(spec for name, spec, _ in harness.tool_definitions() if name == "replan")
    assert "completed_step_reviews" in schema["input_schema"]["properties"]
    ordinary, _ = controller(tmp_path / "ordinary", horizon=False)
    schema = next(spec for name, spec, _ in ordinary.tool_definitions() if name == "replan")
    assert "completed_step_reviews" not in schema["input_schema"]["properties"]
