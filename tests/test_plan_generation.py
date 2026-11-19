"""Tests for the runtime-backed candidate plan generator."""

from __future__ import annotations

import json
import unittest

from rpent.harness.state import MAX_STEP_ATTEMPTS
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


class FakePlanningRuntime:
    def __init__(
        self,
        *,
        tool: str = "move",
        covers_outcomes: list[str] | None = None,
        omit_finish_attempts: int = 0,
    ) -> None:
        self.tool = tool
        self.covers_outcomes = covers_outcomes
        self.omit_finish_attempts = omit_finish_attempts
        self.solve_calls = 0
        self.tool_names: list[str] = []
        self.tool_specs: list[dict] = []
        self.system_prompt = ""
        self.system_prompts: list[str] = []
        self.user_message = ""

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit,
        max_turns: int,
        input_queue=None,
    ) -> RuntimeResult:
        self.solve_calls += 1
        self.tool_specs = toolkit.get_tools_spec()
        self.tool_names = [spec["name"] for spec in self.tool_specs]
        self.system_prompt = system_prompt
        self.system_prompts.append(system_prompt)
        self.user_message = user_message
        chinese = "Simplified Chinese" in system_prompt
        request = json.loads(user_message)
        payload = {
            "strategy": (
                "移动一次并检查相机结果"
                if chinese
                else "move once and verify the camera result"
            ),
            "steps": [
                {
                    "id": "step-1",
                    "intent": (
                        "移动到已经定位的目标"
                        if chinese
                        else "move to the grounded target"
                    ),
                    "allowed_tools": [self.tool],
                    "expected_evidence": [
                        "相机画面确认已经到达" if chinese else "camera confirms arrival"
                    ],
                }
            ],
        }
        if request["request_type"] == "initial_plan":
            contract = request.get("outcome_contract") or [
                {"id": "draft-outcome", "number": 1, "text": "move safely"}
            ]
            outcome_ids = [item["id"] for item in contract]
            payload["outcomes"] = [
                {
                    "id": item["id"],
                    "instruction_index": item["number"],
                    "description": item["text"],
                    "success_predicate": f"visibly verify {item['text']}",
                    "depends_on": [] if index == 0 else [outcome_ids[index - 1]],
                    "lifecycle": "terminal",
                }
                for index, item in enumerate(contract)
            ]
            payload["stages"] = [
                {
                    "id": "stage-1",
                    "intent": "complete requested outcomes",
                    "outcome_ids": outcome_ids,
                }
            ]
            payload["steps"][0]["stage_id"] = "stage-1"
            if self.covers_outcomes is not None:
                payload["steps"][0]["covers_outcomes"] = self.covers_outcomes
            elif not request.get("outcome_contract"):
                payload["steps"][0]["covers_outcomes"] = outcome_ids
        elif self.covers_outcomes is not None:
            payload["steps"][0]["covers_outcomes"] = self.covers_outcomes
        if self.solve_calls <= self.omit_finish_attempts:
            return RuntimeResult(messages=[{"role": "assistant", "content": payload}])
        toolkit.execute_tool("finish", payload)
        return RuntimeResult(finish_result={"_finish": True, **payload})


class RuntimePlanGeneratorTest(unittest.TestCase):
    def test_retries_once_when_planner_returns_json_without_finish_tool(self) -> None:
        runtime = FakePlanningRuntime(omit_finish_attempts=1)

        result = RuntimePlanGenerator(runtime).generate(
            goal="move safely",
            observations=["target is visible"],
            constraints=[],
            physical_tools={"move"},
        )

        self.assertEqual(result["strategy"], "move once and verify the camera result")
        self.assertEqual(runtime.solve_calls, 2)
        self.assertIn("assistant text is invalid", runtime.system_prompts[1])

    def test_missing_finish_retry_is_strictly_bounded(self) -> None:
        runtime = FakePlanningRuntime(omit_finish_attempts=2)

        with self.assertRaisesRegex(RuntimeError, "did not call finish"):
            RuntimePlanGenerator(runtime).generate(
                goal="move safely",
                observations=["target is visible"],
                constraints=[],
                physical_tools={"move"},
            )

        self.assertEqual(runtime.solve_calls, 2)

    def test_legacy_initial_plan_does_not_request_outcome_program(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime, require_outcome_program=False)

        result = generator.generate(
            goal="move safely",
            observations=["target is visible"],
            constraints=[],
            physical_tools={"move"},
        )

        request = json.loads(runtime.user_message)
        self.assertNotIn("outcome_contract", request)
        self.assertNotIn("outcome_program", result)
        self.assertNotIn("outcomes", result)
        self.assertNotIn("stages", result)
        self.assertNotIn("outcome contract", runtime.system_prompt)

    def test_generates_validated_plan_with_minimal_toolkit(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime)

        result = generator.generate(
            goal="move safely",
            observations=["target xyz is [0.1, 0.2, 0.3]"],
            constraints=["avoid the table"],
            physical_tools={"move"},
        )

        self.assertEqual(runtime.tool_names, ["finish"])
        self.assertIn("target xyz", runtime.user_message)
        self.assertEqual(result["goal"], "move safely")
        self.assertEqual(result["steps"][0]["allowed_tools"], ["move"])
        self.assertIn("never exceed 16", runtime.system_prompt)
        self.assertIn("remain on that step", runtime.system_prompt)
        self.assertIn(
            "total physical tool-call budget",
            runtime.system_prompt,
        )
        self.assertNotIn("pi0_pick", runtime.system_prompt)
        self.assertNotIn("set_gripper", runtime.system_prompt)
        self.assertIn("every physical call", runtime.system_prompt)
        finish_spec = runtime.tool_specs[0]
        self.assertEqual(
            finish_spec["input_schema"]["properties"]["steps"]["maxItems"],
            16,
        )
        self.assertEqual(
            finish_spec["input_schema"]["properties"]["steps"]["items"]["properties"][
                "max_attempts"
            ]["maximum"],
            MAX_STEP_ATTEMPTS,
        )

    def test_libero_tool_set_keeps_bounded_pick_example(self) -> None:
        runtime = FakePlanningRuntime(tool="pi0_pick")

        RuntimePlanGenerator(runtime).generate(
            goal="pick safely",
            observations=["target is visible"],
            constraints=[],
            physical_tools={"pi0_pick", "set_gripper"},
        )

        self.assertIn("pi0_pick followed by set_gripper", runtime.system_prompt)
        self.assertIn(">=3 when preserving one safe", runtime.system_prompt)

    def test_vla_replan_prompt_has_no_libero_primitive_example(self) -> None:
        runtime = FakePlanningRuntime(tool="execute_vla")

        RuntimePlanGenerator(runtime).generate_replan(
            goal="close the fridge",
            reason="the first chunk did not finish",
            observations=["the door remains open"],
            constraints=[],
            current_plan={"strategy": "close", "unfinished_steps": []},
            physical_tools={"execute_vla"},
        )

        self.assertNotIn("pi0_pick", runtime.system_prompt)
        self.assertNotIn("set_gripper", runtime.system_prompt)
        self.assertIn("every physical call", runtime.system_prompt)

    def test_libero_replan_preserves_operation_switches_but_rejects_paraphrase_repairs(self):
        runtime = FakePlanningRuntime(tool="pi0_pick")
        generator = RuntimePlanGenerator(runtime, require_outcome_program=False)
        plan = generator.generate(goal="Place the cup.", observations=[], constraints=[],
                                  physical_tools={"pi0_pick", "pi0_place", "set_gripper"})
        generator.generate_replan(goal=plan["goal"], reason="Grasp missed.", observations=[],
                                  constraints=[], current_plan=plan,
                                  physical_tools={"pi0_pick", "pi0_place", "set_gripper"})
        self.assertIn("rewording the same tool call is not a change.", runtime.system_prompt)
        self.assertNotIn("unless the action constraints say the policy receives", runtime.system_prompt)
        self.assertIn("a different task phase supported by fresh observations", runtime.system_prompt)
        self.assertIn("a different tool or stop condition", runtime.system_prompt)

    def test_rejects_unknown_physical_tool_from_model(self) -> None:
        generator = RuntimePlanGenerator(FakePlanningRuntime(tool="delete_world"))

        with self.assertRaisesRegex(ValueError, "unknown tools"):
            generator.generate(
                goal="move",
                observations=["target is visible"],
                constraints=[],
                physical_tools={"move"},
            )

    def test_generates_replan_with_failure_and_current_plan_context(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime)

        result = generator.generate_replan(
            goal="move safely",
            reason="the first move stalled",
            observations=["eef remains 8 cm from the target"],
            constraints=["retreat before retrying"],
            current_plan={
                "proposal_id": "plan-123",
                "strategy": "move directly",
                "verified_facts": [
                    {
                        "step_id": "step-0",
                        "summary": "The handled vessel is the target.",
                        "evidence": [
                            "Fresh multi-view observation confirmed identity."
                        ],
                    }
                ],
                "completed_steps": [],
                "unfinished_steps": [
                    {
                        "id": "step-1",
                        "intent": "move",
                        "allowed_tools": ["move"],
                    }
                ],
            },
            physical_tools={"move"},
        )

        self.assertIn('"request_type": "replan"', runtime.user_message)
        self.assertIn("the first move stalled", runtime.user_message)
        self.assertIn("unfinished_steps", runtime.user_message)
        self.assertIn("verified_facts", runtime.user_message)
        self.assertIn("sticky", runtime.system_prompt)
        self.assertIn("must not rename, replace, or negate", runtime.system_prompt)
        self.assertEqual(result["steps"][0]["allowed_tools"], ["move"])

        generator.generate_replan(
            goal="move safely",
            reason="ordinary failure",
            observations=["target remains visible"],
            constraints=[],
            current_plan={
                "proposal_id": "plan-plain",
                "strategy": "move directly",
                "completed_steps": [],
                "unfinished_steps": [
                    {
                        "id": "step-1",
                        "intent": "move",
                        "allowed_tools": ["move"],
                    }
                ],
            },
            physical_tools={"move"},
        )
        self.assertNotIn("sticky facts", runtime.system_prompt)

    def test_chinese_requirement_applies_to_initial_plan_and_replan(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime, use_chinese=True)

        generator.generate(
            goal="移动物体",
            observations=["目标可见"],
            constraints=[],
            physical_tools={"move"},
        )
        self.assertIn("Simplified Chinese", runtime.system_prompt)
        self.assertIn("expected_evidence", runtime.system_prompt)
        self.assertIn("environment_success=false", runtime.system_prompt)
        self.assertIn("环境成功条件尚未满足", runtime.system_prompt)

        generator.generate_replan(
            goal="移动物体",
            reason="第一次移动失败",
            observations=["末端未到达目标"],
            constraints=[],
            current_plan={
                "proposal_id": "plan-1",
                "strategy": "移动一次",
                "completed_steps": [],
                "unfinished_steps": [],
            },
            physical_tools={"move"},
        )
        self.assertIn("Simplified Chinese", runtime.system_prompt)

    def test_numbered_goal_rejects_grouped_staged_outcome_coverage(self) -> None:
        runtime = FakePlanningRuntime(
            covers_outcomes=["outcome-1", "outcome-2"],
        )
        generator = RuntimePlanGenerator(runtime)

        with self.assertRaisesRegex(
            ValueError,
            "each staged plan step must cover exactly one outcome",
        ):
            generator.generate(
                goal="1. open the drawer; 2. place the cup inside",
                observations=["drawer and cup are visible"],
                constraints=[],
                physical_tools={"move"},
            )

        request = json.loads(runtime.user_message)
        self.assertEqual(
            request["required_outcome_ids"],
            ["outcome-1", "outcome-2"],
        )
        self.assertEqual(
            request["outcome_contract"],
            [
                {
                    "id": "outcome-1",
                    "number": 1,
                    "text": "open the drawer",
                },
                {
                    "id": "outcome-2",
                    "number": 2,
                    "text": "place the cup inside",
                },
            ],
        )
        step_schema = runtime.tool_specs[0]["input_schema"]["properties"]["steps"][
            "items"
        ]
        self.assertIn("covers_outcomes", step_schema["properties"])
        self.assertNotIn("covers_outcomes", step_schema["required"])

    def test_numbered_goal_rejects_missing_outcome_coverage(self) -> None:
        generator = RuntimePlanGenerator(FakePlanningRuntime())

        with self.assertRaisesRegex(ValueError, "missing"):
            generator.generate(
                goal="1. open the drawer; 2. place the cup inside",
                observations=["drawer and cup are visible"],
                constraints=[],
                physical_tools={"move"},
            )

    def test_replan_enforces_only_explicit_remaining_outcome_ids(self) -> None:
        runtime = FakePlanningRuntime(covers_outcomes=["outcome-2"])
        generator = RuntimePlanGenerator(runtime)

        result = generator.generate_replan(
            goal="1. open the drawer; 2. place the cup inside",
            reason="the cup slipped",
            observations=["the drawer is already open"],
            constraints=[],
            current_plan={
                "required_outcome_ids": ["outcome-2"],
                "completed_outcome_ids": ["outcome-1"],
                "unfinished_steps": [],
            },
            physical_tools={"move"},
        )

        request = json.loads(runtime.user_message)
        self.assertEqual(request["required_outcome_ids"], ["outcome-2"])
        self.assertEqual(result["steps"][0]["covers_outcomes"], ["outcome-2"])

    def test_legacy_replan_allows_grouped_outcome_coverage(self) -> None:
        runtime = FakePlanningRuntime(
            covers_outcomes=["outcome-1", "outcome-2"],
        )
        generator = RuntimePlanGenerator(runtime)

        result = generator.generate_replan(
            goal="1. open the drawer; 2. place the cup inside",
            reason="retry both unfinished outcomes",
            observations=["drawer remains closed"],
            constraints=[],
            current_plan={
                "required_outcome_ids": ["outcome-1", "outcome-2"],
                "unfinished_steps": [],
            },
            physical_tools={"move"},
        )

        self.assertEqual(
            result["steps"][0]["covers_outcomes"],
            ["outcome-1", "outcome-2"],
        )
        step_schema = runtime.tool_specs[0]["input_schema"]["properties"]["steps"][
            "items"
        ]["properties"]["covers_outcomes"]
        self.assertNotIn("maxItems", step_schema)

    def test_replan_rejects_reintroduced_completed_outcome(self) -> None:
        generator = RuntimePlanGenerator(
            FakePlanningRuntime(covers_outcomes=["outcome-1", "outcome-2"])
        )

        with self.assertRaisesRegex(ValueError, "unknown"):
            generator.generate_replan(
                goal="1. open the drawer; 2. place the cup inside",
                reason="the cup slipped",
                observations=["the drawer is already open"],
                constraints=[],
                current_plan={
                    "required_outcome_ids": ["outcome-2"],
                    "completed_outcome_ids": ["outcome-1"],
                    "unfinished_steps": [],
                },
                physical_tools={"move"},
            )

    def test_ordinary_goal_and_legacy_replan_preserve_step_shape(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime)

        initial = generator.generate(
            goal="move safely",
            observations=["target is visible"],
            constraints=[],
            physical_tools={"move"},
        )
        replan = generator.generate_replan(
            goal="move safely",
            reason="the move stalled",
            observations=["target is visible"],
            constraints=[],
            current_plan={"unfinished_steps": []},
            physical_tools={"move"},
        )

        self.assertEqual(initial["steps"][0]["covers_outcomes"], ["outcome-1"])
        self.assertIn("outcome_program", initial)
        self.assertNotIn("covers_outcomes", replan["steps"][0])
        self.assertNotIn("required_outcome_ids", json.loads(runtime.user_message))

    def test_empty_required_outcome_ids_preserve_legacy_replan_shape(self) -> None:
        runtime = FakePlanningRuntime()
        generator = RuntimePlanGenerator(runtime)

        result = generator.generate_replan(
            goal="move safely",
            reason="the move stalled",
            observations=["target is visible"],
            constraints=[],
            current_plan={
                "required_outcome_ids": [],
                "unfinished_steps": [],
            },
            physical_tools={"move"},
        )

        self.assertNotIn("covers_outcomes", result["steps"][0])
        self.assertNotIn("required_outcome_ids", json.loads(runtime.user_message))


if __name__ == "__main__":
    unittest.main()


def test_planning_and_replanning_receive_executor_action_constraints():
    guidance = "Only use servo_eef with an already measured target pose."
    runtime = FakePlanningRuntime(tool="continue_vla")
    generator = RuntimePlanGenerator(
        runtime, require_outcome_program=False, action_guidance=guidance,
    )
    generator.generate(
        goal="close drawer", observations=["drawer is open"], constraints=[],
        physical_tools={"continue_vla", "servo_eef"},
    )
    assert guidance in runtime.system_prompt
    generator.generate_replan(
        goal="close drawer", reason="drawer remains open", observations=["still open"],
        constraints=[], current_plan={"strategy": "close", "unfinished_steps": []},
        physical_tools={"continue_vla", "servo_eef"},
    )
    assert guidance in runtime.system_prompt
    assert runtime.tool_names == ["finish"]
