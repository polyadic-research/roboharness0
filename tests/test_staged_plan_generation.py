"""Contract tests for staged, outcome-aware runtime plan generation."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from rpent.planning.schemas import Plan, PlanStep, Stage
from rpent.planning.stack import DEFAULT_PLANNING_PROFILE, resolve_planning_stack
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


def _outcome(
    outcome_id: str,
    index: int,
    description: str,
    *,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": outcome_id,
        "instruction_index": index,
        "description": description,
        "success_predicate": f"visibly verify: {description}",
        "depends_on": list(depends_on or ()),
        "lifecycle": "terminal",
    }


def _staged_output() -> dict[str, Any]:
    return {
        "strategy": "open, then place",
        "outcomes": [
            _outcome("outcome-1", 1, "open drawer"),
            _outcome(
                "outcome-2",
                2,
                "place cup inside",
                depends_on=["outcome-1"],
            ),
        ],
        "stages": [
            {
                "id": "stage-1",
                "intent": "open drawer",
                "outcome_ids": ["outcome-1"],
            },
            {
                "id": "stage-2",
                "intent": "place cup",
                "outcome_ids": ["outcome-2"],
            },
        ],
        "steps": [
            {
                "id": "step-1",
                "intent": "open the drawer",
                "allowed_tools": ["move"],
                "expected_evidence": ["drawer is visibly open"],
                "stage_id": "stage-1",
                "covers_outcomes": ["outcome-1"],
            }
        ],
    }


class _OutputRuntime:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.user_message = ""
        self.system_prompt = ""
        self.tool_specs: list[dict[str, Any]] = []

    def solve(self, *, system_prompt, user_message, toolkit, max_turns, input_queue):
        self.user_message = user_message
        self.system_prompt = system_prompt
        self.tool_specs = toolkit.get_tools_spec()
        toolkit.execute_tool("finish", deepcopy(self.output))
        return RuntimeResult(finish_result={"_finish": True, **deepcopy(self.output)})


def _initial_plan() -> tuple[dict[str, Any], _OutputRuntime]:
    runtime = _OutputRuntime(_staged_output())
    result = RuntimePlanGenerator(runtime).generate(
        goal="1. open drawer; 2. place cup inside",
        observations=["drawer and cup visible"],
        constraints=[],
        physical_tools={"move"},
    )
    return result, runtime


def test_initial_plan_compiles_immutable_outcome_program_and_first_stage() -> None:
    result, runtime = _initial_plan()

    assert result["outcome_program"]["schema_version"] == "outcome-program/v1"
    assert result["outcome_program"]["contract_id"].startswith("contract-")
    assert result["contract_id"] == result["outcome_program"]["contract_id"]
    assert result["outcome_program_schema_version"] == "outcome-program/v1"
    assert result["outcomes"] == result["outcome_program"]["outcomes"]
    assert result["required_outcome_ids"] == ["outcome-1", "outcome-2"]
    assert result["stages"][1]["outcome_ids"] == ["outcome-2"]
    assert result["steps"][0]["stage_id"] == "stage-1"
    assert result["steps"][0]["covers_outcomes"] == ["outcome-1"]

    finish_schema = runtime.tool_specs[0]["input_schema"]
    assert {"outcomes", "stages"} <= set(finish_schema["required"])


def test_initial_plan_rejects_step_covering_multiple_stage_outcomes() -> None:
    output = _staged_output()
    output["stages"] = [
        {
            "id": "stage-1",
            "intent": "open, then place",
            "outcome_ids": ["outcome-1", "outcome-2"],
        }
    ]
    output["steps"][0]["covers_outcomes"] = ["outcome-1", "outcome-2"]
    runtime = _OutputRuntime(output)

    with pytest.raises(
        ValueError,
        match="each staged plan step must cover exactly one outcome",
    ):
        RuntimePlanGenerator(runtime).generate(
            goal="1. open drawer; 2. place cup inside",
            observations=["drawer and cup visible"],
            constraints=[],
            physical_tools={"move"},
        )

    step_schema = runtime.tool_specs[0]["input_schema"]["properties"]["steps"]["items"][
        "properties"
    ]["covers_outcomes"]
    assert step_schema["maxItems"] == 1
    assert "exactly one" in step_schema["description"]


def test_natural_language_outcome_ids_are_canonicalized_before_compilation() -> None:
    output = _staged_output()
    output["outcomes"][0]["id"] = "drawer-open"
    output["outcomes"][1]["id"] = "cup-placed"
    output["outcomes"][1]["depends_on"] = ["drawer-open"]
    output["stages"][0]["outcome_ids"] = ["drawer-open"]
    output["stages"][1]["outcome_ids"] = ["cup-placed"]
    output["steps"][0]["covers_outcomes"] = ["drawer-open"]
    runtime = _OutputRuntime(output)

    result = RuntimePlanGenerator(runtime).generate(
        goal="open the drawer and place the cup inside",
        observations=["drawer and cup visible"],
        constraints=[],
        physical_tools={"move"},
    )

    assert [item["id"] for item in result["outcome_program"]["outcomes"]] == [
        "outcome-1",
        "outcome-2",
    ]
    assert result["outcome_program"]["outcomes"][1]["depends_on"] == ["outcome-1"]


def test_replan_preserves_contract_and_exactly_partitions_unresolved_ids() -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "place carefully",
        "stages": [
            {
                "id": "repair-stage-1",
                "intent": "place cup",
                "outcome_ids": ["outcome-2"],
            }
        ],
        "steps": [
            {
                "id": "repair-step-1",
                "intent": "place cup inside",
                "allowed_tools": ["move"],
                "expected_evidence": ["cup visibly inside"],
                "stage_id": "repair-stage-1",
                "covers_outcomes": ["outcome-2"],
            }
        ],
    }
    runtime = _OutputRuntime(replacement)

    result = RuntimePlanGenerator(runtime).generate_replan(
        goal=initial["goal"],
        reason="cup slipped",
        observations=["drawer remains open"],
        constraints=[],
        current_plan={
            **initial,
            "completed_outcome_ids": ["outcome-1"],
            "required_outcome_ids": ["outcome-2"],
        },
        physical_tools={"move"},
    )

    assert result["outcome_program"] == initial["outcome_program"]
    assert result["outcomes"] == initial["outcomes"]
    assert result["required_outcome_ids"] == ["outcome-2"]
    assert result["stages"][0]["outcome_ids"] == ["outcome-2"]
    request = json.loads(runtime.user_message)
    assert request["immutable_outcome_program"] == json.loads(
        json.dumps(initial["outcome_program"])
    )


def test_contract_replan_scopes_current_outcome_when_later_outcome_is_deferred(
) -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "retry opening before the later placement stage",
        "stages": [
            {
                "id": "stage-1",
                "intent": "open drawer",
                "outcome_ids": ["outcome-1"],
            }
        ],
        "steps": [
            {
                "id": "repair-step-1",
                "intent": "open the drawer",
                "allowed_tools": ["move"],
                "expected_evidence": ["drawer visibly open"],
                "stage_id": "stage-1",
                "covers_outcomes": ["outcome-1"],
            }
        ],
    }
    runtime = _OutputRuntime(replacement)

    result = RuntimePlanGenerator(runtime).generate_replan(
        goal=initial["goal"],
        reason="drawer did not open",
        observations=["drawer remains closed"],
        constraints=[],
        current_plan={
            **initial,
            "completed_outcome_ids": [],
            "required_outcome_ids": ["outcome-1"],
            "deferred_outcome_ids": ["outcome-2"],
        },
        physical_tools={"move"},
    )

    assert result["required_outcome_ids"] == ["outcome-1"]
    assert result["stages"][0]["outcome_ids"] == ["outcome-1"]
    request = json.loads(runtime.user_message)
    assert request["required_outcome_ids"] == ["outcome-1"]
    assert request["deferred_outcome_ids"] == ["outcome-2"]
    assert "Do not cover deferred_outcome_ids" in runtime.system_prompt


@pytest.mark.parametrize(
    ("required", "deferred"),
    (
        (["outcome-1"], []),
        (["outcome-2"], ["outcome-1"]),
        (["outcome-1"], ["outcome-1"]),
    ),
)
def test_scoped_contract_replan_rejects_invalid_deferred_partition(
    required: list[str],
    deferred: list[str],
) -> None:
    initial, _ = _initial_plan()

    with pytest.raises(ValueError, match="partition the immutable outcome contract"):
        RuntimePlanGenerator(_OutputRuntime({})).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=[],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": [],
                "required_outcome_ids": required,
                "deferred_outcome_ids": deferred,
            },
            physical_tools={"move"},
        )


def test_contract_replan_rejects_step_covering_multiple_stage_outcomes() -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "open, then place",
        "stages": [
            {
                "id": "repair-stage",
                "intent": "complete both remaining outcomes",
                "outcome_ids": ["outcome-1", "outcome-2"],
            }
        ],
        "steps": [
            {
                "id": "repair-step",
                "intent": "open drawer and place cup",
                "allowed_tools": ["move"],
                "expected_evidence": ["drawer open and cup inside"],
                "stage_id": "repair-stage",
                "covers_outcomes": ["outcome-1", "outcome-2"],
            }
        ],
    }
    runtime = _OutputRuntime(replacement)

    with pytest.raises(
        ValueError,
        match="each staged plan step must cover exactly one outcome",
    ):
        RuntimePlanGenerator(runtime).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=[],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": [],
                "required_outcome_ids": ["outcome-1", "outcome-2"],
            },
            physical_tools={"move"},
        )

    assert "exactly one outcome" in runtime.system_prompt
    step_schema = runtime.tool_specs[0]["input_schema"]["properties"]["steps"]["items"][
        "properties"
    ]["covers_outcomes"]
    assert step_schema["maxItems"] == 1


def test_replan_derives_unresolved_ids_from_contract_when_not_explicit() -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "place carefully",
        "stages": [
            {
                "id": "repair-stage",
                "intent": "place cup",
                "outcome_ids": ["outcome-2"],
            }
        ],
        "steps": [
            {
                "id": "repair-step",
                "intent": "place cup",
                "allowed_tools": ["move"],
                "expected_evidence": ["cup placed"],
                "stage_id": "repair-stage",
                "covers_outcomes": ["outcome-2"],
            }
        ],
    }

    result = RuntimePlanGenerator(_OutputRuntime(replacement)).generate_replan(
        goal=initial["goal"],
        reason="repair",
        observations=[],
        constraints=[],
        current_plan={
            key: value
            for key, value in {
                **initial,
                "completed_outcome_ids": ["outcome-1"],
            }.items()
            if key != "required_outcome_ids"
        },
        physical_tools={"move"},
    )

    assert result["required_outcome_ids"] == ["outcome-2"]
    assert result["outcome_program"] == initial["outcome_program"]


def test_contract_replan_cannot_use_empty_required_ids_to_hide_unresolved() -> None:
    initial, _ = _initial_plan()

    with pytest.raises(ValueError, match="partition the immutable outcome contract"):
        RuntimePlanGenerator(_OutputRuntime({})).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=[],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": ["outcome-1"],
                "required_outcome_ids": [],
            },
            physical_tools={"move"},
        )


def test_contract_replan_rejects_when_all_outcomes_are_completed() -> None:
    initial, _ = _initial_plan()

    with pytest.raises(ValueError, match="no unresolved outcomes remain"):
        RuntimePlanGenerator(_OutputRuntime({})).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=[],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": ["outcome-1", "outcome-2"],
                "required_outcome_ids": [],
            },
            physical_tools={"move"},
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda output: output.update(
                {"outcomes": [_outcome("outcome-2", 2, "rewritten")]}
            ),
            "cannot replace outcomes",
        ),
        (
            lambda output: output["stages"][0].update(
                {"outcome_ids": ["outcome-1", "outcome-2"]}
            ),
            "completed|unknown",
        ),
        (
            lambda output: output["stages"][0].update({"outcome_ids": ["outcome-99"]}),
            "unknown",
        ),
        (
            lambda output: output.update({"stages": []}),
            "exactly once|non-empty",
        ),
    ],
)
def test_replan_cannot_replace_or_change_outcome_contract(mutate, message: str) -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "repair",
        "stages": [
            {
                "id": "repair-stage",
                "intent": "place cup",
                "outcome_ids": ["outcome-2"],
            }
        ],
        "steps": [
            {
                "id": "repair-step",
                "intent": "place cup",
                "allowed_tools": ["move"],
                "expected_evidence": ["cup placed"],
                "stage_id": "repair-stage",
                "covers_outcomes": ["outcome-2"],
            }
        ],
    }
    mutate(replacement)

    with pytest.raises(ValueError, match=message):
        RuntimePlanGenerator(_OutputRuntime(replacement)).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=["drawer open"],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": ["outcome-1"],
                "required_outcome_ids": ["outcome-2"],
            },
            physical_tools={"move"},
        )


def test_replan_rejects_current_plan_that_silently_deletes_unresolved_outcome() -> None:
    initial, _ = _initial_plan()
    replacement = {
        "strategy": "nothing left",
        "stages": [{"id": "stage-x", "intent": "open", "outcome_ids": ["outcome-1"]}],
        "steps": [
            {
                "id": "step-x",
                "intent": "open",
                "allowed_tools": ["move"],
                "expected_evidence": ["open"],
                "stage_id": "stage-x",
                "covers_outcomes": ["outcome-1"],
            }
        ],
    }

    with pytest.raises(ValueError, match="partition the immutable outcome contract"):
        RuntimePlanGenerator(_OutputRuntime(replacement)).generate_replan(
            goal=initial["goal"],
            reason="repair",
            observations=[],
            constraints=[],
            current_plan={
                **initial,
                "completed_outcome_ids": [],
                "required_outcome_ids": ["outcome-1"],
            },
            physical_tools={"move"},
        )


def test_schema_and_stack_expose_staged_contract_metadata_compatibly() -> None:
    stage = Stage(id="stage-1", intent="move", outcome_ids=("outcome-1",))
    step = PlanStep(
        id="step-1",
        intent="move",
        allowed_tools=("move",),
        stage_id="stage-1",
        covers_outcomes=("outcome-1",),
    )
    plan = Plan(
        goal="move",
        steps=(step,),
        stages=(stage,),
        contract_id="contract-test",
        outcome_program_schema_version="outcome-program/v1",
        required_outcome_ids=("outcome-1",),
    )

    assert plan.steps[0].covers_outcomes == ("outcome-1",)
    metadata = resolve_planning_stack(DEFAULT_PLANNING_PROFILE).metadata()
    assert metadata["effective"]["outcome_program_schema_version"] == (
        "outcome-program/v1"
    )
