"""Fast, dependency-free tests for plan-gated execution semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rpent.harness import Harness, PlanGate, PlanState
from rpent.planning.events import JsonlEventRecorder
from rpent.tools.toolkit import Toolkit


def _steps(*tools: str) -> list[dict]:
    return [
        {
            "id": "step-1",
            "intent": "perform the controlled action",
            "allowed_tools": list(tools),
            "expected_evidence": ["camera confirms the expected change"],
        }
    ]


class FakePlanGenerator:
    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
    ) -> dict:
        tool = next(
            (name for name in sorted(physical_tools) if name in goal),
            sorted(physical_tools)[0],
        )
        return {
            "goal": goal,
            "strategy": f"act from {observations[0]}",
            "steps": _steps(tool),
        }

    def generate_replan(
        self,
        *,
        goal: str,
        reason: str,
        observations: list[str],
        constraints: list[str],
        current_plan: dict,
        physical_tools: set[str],
    ) -> dict:
        tool = "move" if "move" in physical_tools else sorted(physical_tools)[0]
        return {
            "goal": goal,
            "strategy": f"recover from {reason} using {observations[0]}",
            "steps": _steps(tool),
        }


class CapturingTwoStepPlanGenerator(FakePlanGenerator):
    def __init__(self) -> None:
        self.replan_context = None

    def generate(self, **kwargs) -> dict:
        del kwargs
        return {
            "goal": "manipulate the target",
            "strategy": "preserve verified progress",
            "steps": [
                {**_steps("move")[0], "id": "identify-and-place"},
                {**_steps("move")[0], "id": "operate-control"},
            ],
        }

    def generate_replan(self, **kwargs) -> dict:
        self.replan_context = kwargs["current_plan"]
        return {
            "goal": kwargs["goal"],
            "strategy": "repair only the unfinished control",
            "steps": [{**_steps("move")[0], "id": "repair-control"}],
        }


class ChinesePlanGenerator(FakePlanGenerator):
    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
    ) -> dict:
        tool = sorted(physical_tools)[0]
        return {
            "goal": goal,
            "strategy": "根据当前观测执行一次动作",
            "steps": [
                {
                    "id": "step-1",
                    "intent": "执行受控动作",
                    "allowed_tools": [tool],
                    "expected_evidence": ["相机确认预期变化"],
                }
            ],
        }

    def generate_replan(
        self,
        *,
        goal: str,
        reason: str,
        observations: list[str],
        constraints: list[str],
        current_plan: dict,
        physical_tools: set[str],
    ) -> dict:
        return {
            "goal": goal,
            "strategy": "根据失败证据执行恢复动作",
            "steps": [
                {
                    "id": "repair-1",
                    "intent": "执行恢复动作",
                    "allowed_tools": [sorted(physical_tools)[0]],
                    "expected_evidence": ["相机确认恢复成功"],
                }
            ],
        }


class PickLockPlanGenerator(FakePlanGenerator):
    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
    ) -> dict:
        return {
            "goal": goal,
            "strategy": "grasp, recover once if needed, then lock",
            "steps": [
                {
                    "id": "pick-lock",
                    "intent": "grasp the bowl and lock the gripper",
                    "allowed_tools": ["pi0_pick", "set_gripper"],
                    "expected_evidence": [
                        "the bowl is lifted and the gripper is locked"
                    ],
                    "max_attempts": 1,
                }
            ],
        }


class TwoStepPlanGenerator(FakePlanGenerator):
    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
    ) -> dict:
        return {
            "goal": goal,
            "strategy": "move, then grasp",
            "steps": [
                {
                    "id": "move-step",
                    "intent": "move to the target",
                    "allowed_tools": ["move"],
                    "expected_evidence": ["the target is reached"],
                },
                {
                    "id": "grasp-step",
                    "intent": "grasp the target",
                    "allowed_tools": ["grasp"],
                    "expected_evidence": ["the target is held"],
                },
            ],
        }


class HarnessControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.trace = Path(self.tmp.name) / "harness_trace.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _controller(self, *, max_replans: int = 1) -> Harness:
        return Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=max_replans,
            physical_tools={"move", "grasp"},
            plan_generator=FakePlanGenerator(),
        )

    def _v3_controller(self, *, max_replans: int = 1) -> Harness:
        return Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=max_replans,
            physical_tools={"move", "grasp", "pi0_pick"},
            plan_generator=FakePlanGenerator(),
            require_execution_id=True,
            auto_verify_tool_signals=True,
            require_environment_success=True,
        )

    def test_control_schema_separates_generation_from_registration(self) -> None:
        definitions = {
            name: spec for name, spec, _handler in self._controller().tool_definitions()
        }
        create_schema = definitions["create_plan"]["input_schema"]
        self.assertEqual(create_schema["required"], ["goal", "observations"])
        self.assertNotIn("steps", create_schema["properties"])
        replan_schema = definitions["replan"]["input_schema"]
        self.assertEqual(replan_schema["required"], ["reason", "observations"])
        self.assertNotIn("steps", replan_schema["properties"])
        self.assertIn("register_plan", definitions)

    def test_blocks_unplanned_and_requires_verification(self) -> None:
        controller = self._controller()
        self.assertFalse(controller.before_tool("move", {})["allowed"])

        created = controller.create_plan(
            "move safely",
            ["camera shows a clear target"],
        )
        self.assertTrue(created["ok"])
        self.assertFalse(created["registered"])
        self.assertFalse(controller.before_tool("move", {})["allowed"])
        registered = controller.register_plan(created["proposal_id"])
        self.assertTrue(registered["ok"])
        self.assertTrue(controller.before_tool("move", {})["allowed"])
        controller.after_tool("move", {}, {"ok": True})
        self.assertEqual(controller.snapshot()["status"], "awaiting_verification")
        self.assertFalse(controller.before_tool("move", {})["allowed"])

        verified = controller.verify_step(
            "pass", ["camera shows arrival"], "reached the target"
        )
        self.assertTrue(verified["ok"])
        self.assertEqual(controller.snapshot()["status"], "completed")
        self.assertTrue(self.trace.exists())
        events = [
            json.loads(line)["event"] for line in self.trace.read_text().splitlines()
        ]
        self.assertIn("plan_violation", events)
        self.assertIn("plan_proposed", events)
        self.assertIn("plan_registered", events)
        self.assertIn("proposal_registered", events)
        self.assertIn("step_verified", events)

    def test_snapshot_exposes_legacy_contract_completion(self) -> None:
        controller = self._controller()
        self.assertFalse(controller.snapshot()["contract_success"])

        created = controller.create_plan(
            "move safely",
            ["camera shows a clear target"],
        )
        controller.register_plan(created["proposal_id"])
        controller.after_tool("move", {}, {"ok": True})
        self.assertFalse(controller.snapshot()["contract_success"])

        controller.verify_step(
            "pass", ["camera shows arrival"], "reached the target"
        )
        self.assertTrue(controller.snapshot()["contract_success"])

    def test_failed_verification_requires_bounded_replan(self) -> None:
        controller = self._controller(max_replans=1)
        proposal = controller.create_plan("grasp", ["grasp target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("grasp", {}, {"ok": True})
        controller.verify_step("fail", ["gripper is fully closed"], "grasp missed")
        self.assertEqual(controller.snapshot()["status"], "needs_replan")
        self.assertFalse(controller.before_tool("grasp", {})["allowed"])

        replanned = controller.replan(
            "grasp missed",
            ["gripper is closed but the target remains on the table"],
        )
        self.assertTrue(replanned["ok"])
        self.assertFalse(replanned["registered"])
        self.assertEqual(controller.snapshot()["status"], "needs_replan")
        self.assertEqual(controller.snapshot()["replans_remaining"], 1)
        registered = controller.register_plan(replanned["proposal_id"])
        self.assertTrue(registered["ok"])
        self.assertEqual(registered["proposal_type"], "replan")
        self.assertEqual(controller.snapshot()["replans_remaining"], 0)
        controller.after_tool("move", {}, {"ok": True})
        controller.verify_step("fail", ["camera uncertain"], "still uncertain")
        exhausted = controller.replan(
            "second recovery",
            ["camera remains uncertain"],
        )
        self.assertIn("budget exhausted", exhausted["error"])

    def test_replan_context_preserves_fresh_pass_evidence_as_verified_facts(
        self,
    ) -> None:
        generator = CapturingTwoStepPlanGenerator()
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=1,
            physical_tools={"move"},
            plan_generator=generator,
        )
        proposal = controller.create_plan("manipulate", ["target is visible"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("move", {}, {"ok": True})
        controller.verify_step(
            "pass",
            [
                "The handled metal vessel is the target.",
                "The target is stable on the selected support.",
            ],
            "Target identity and placement are visually verified.",
        )
        controller.after_tool("move", {}, {"ok": True})
        controller.verify_step(
            "fail",
            ["The control did not move."],
            "Control operation failed.",
        )

        repair = controller.replan(
            "repair the control",
            ["The target placement is slightly off-center."],
        )

        assert generator.replan_context is not None
        self.assertEqual(
            generator.replan_context["verified_facts"],
            [
                {
                    "step_id": "identify-and-place",
                    "intent": "perform the controlled action",
                    "summary": "Target identity and placement are visually verified.",
                    "evidence": [
                        "The handled metal vessel is the target.",
                        "The target is stable on the selected support.",
                    ],
                    "execution_id": None,
                }
            ],
        )
        self.assertTrue(repair["ok"])
        self.assertEqual(repair["goal"], "manipulate")
        self.assertEqual(repair["harness"]["replans_remaining"], 1)
        registered = controller.register_plan(repair["proposal_id"])
        self.assertTrue(registered["ok"])
        self.assertEqual(registered["harness"]["replans_remaining"], 0)

    def test_automatic_pass_is_preserved_for_later_replan_context(self) -> None:
        generator = CapturingTwoStepPlanGenerator()
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=1,
            physical_tools={"move"},
            plan_generator=generator,
            require_execution_id=True,
            auto_verify_tool_signals=True,
        )
        proposal = controller.create_plan("manipulate", ["target is visible"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool(
            "move",
            {},
            {
                "environment": {
                    "success": True,
                    "terminated": False,
                    "truncated": False,
                }
            },
            execution_id="placed-target",
        )
        controller.after_tool(
            "move",
            {},
            {"success": False},
            execution_id="control-failed",
        )
        controller.verify_step(
            "fail",
            ["The control did not move."],
            "Control operation failed.",
            execution_id="control-failed",
        )

        controller.replan("repair control", ["control did not move"])

        assert generator.replan_context is not None
        fact = generator.replan_context["verified_facts"][0]
        self.assertEqual(fact["step_id"], "identify-and-place")
        self.assertEqual(fact["execution_id"], "placed-target")

    def test_replan_updates_active_proposal_metadata(self) -> None:
        controller = self._controller()
        proposal = controller.create_plan("grasp", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("grasp", {}, {"ok": True})
        controller.verify_step(
            "fail",
            ["fresh view confirms the grasp missed"],
            "grasp failed",
        )

        repair = controller.replan(
            "target is occluded",
            ["gripper and target overlap in the camera"],
        )
        before_registration = controller.view_plan()
        self.assertEqual(before_registration["proposal_id"], proposal["proposal_id"])
        self.assertEqual(before_registration["harness"]["status"], "needs_replan")

        controller.register_plan(repair["proposal_id"])
        active = controller.view_plan()
        self.assertEqual(active["proposal_id"], repair["proposal_id"])
        self.assertEqual(active["strategy"], repair["strategy"])
        self.assertEqual(active["harness"]["status"], "ready")

    def test_chinese_mode_rejects_non_chinese_planning_and_verification(self) -> None:
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=1,
            physical_tools={"move"},
            plan_generator=ChinesePlanGenerator(),
            use_chinese=True,
        )
        rejected_plan = controller.create_plan("move", ["target is visible"])
        self.assertIn("--use_chines", rejected_plan["error"])

        proposal = controller.create_plan("move", ["目标清晰可见"])
        self.assertTrue(proposal["ok"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("move", {}, {"ok": True})

        rejected_verification = controller.verify_step(
            "fail",
            ["camera shows no movement"],
            "move failed",
        )
        self.assertIn("--use_chines", rejected_verification["error"])
        self.assertEqual(controller.snapshot()["status"], "awaiting_verification")

        verified = controller.verify_step(
            "fail",
            ["相机显示机械臂没有移动"],
            "本次移动失败",
        )
        self.assertTrue(verified["ok"])
        rejected_replan = controller.replan(
            "move failed",
            ["目标仍然可见"],
        )
        self.assertIn("--use_chines", rejected_replan["error"])
        repair = controller.replan(
            "移动没有到达目标",
            ["目标仍然清晰可见"],
        )
        self.assertTrue(repair["ok"])

    def test_v3_binds_verdict_to_latest_execution_and_tracks_modes(self) -> None:
        controller = self._v3_controller()
        proposal = controller.create_plan("move", ["target is localized"])
        self.assertEqual(controller.snapshot()["execution_mode"], "plan")
        controller.register_plan(proposal["proposal_id"])
        self.assertEqual(controller.snapshot()["execution_mode"], "act")

        controller.after_tool(
            "move",
            {},
            {"moved": True},
            execution_id="exec-1",
        )
        pending = controller.snapshot()
        self.assertEqual(pending["execution_mode"], "verify")
        self.assertEqual(pending["pending_execution_id"], "exec-1")

        stale = controller.verify_step(
            "progress",
            ["camera shows useful motion"],
            "still approaching",
            execution_id="exec-old",
        )
        self.assertIn("does not match", stale["error"])
        advanced = controller.verify_step(
            "progress",
            ["camera shows useful motion"],
            "still approaching",
            execution_id="exec-1",
            confidence=0.8,
        )
        self.assertTrue(advanced["ok"])
        self.assertEqual(controller.snapshot()["execution_mode"], "act")
        self.assertEqual(controller.snapshot()["active_step_index"], 0)

    def test_v3_progress_is_bounded_by_step_attempts(self) -> None:
        controller = self._v3_controller()
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])

        controller.after_tool(
            "move",
            {},
            {"moved": True},
            execution_id="move-progress-1",
        )
        first = controller.verify_step(
            "progress",
            ["机械臂正在接近目标"],
            "本次动作取得有效进展",
            execution_id="move-progress-1",
        )
        self.assertFalse(first["retry_escalated"])
        self.assertEqual(controller.snapshot()["execution_mode"], "act")

        controller.after_tool(
            "move",
            {},
            {"moved": True},
            execution_id="move-progress-2",
        )
        second = controller.verify_step(
            "progress",
            ["机械臂仍未到达最终目标"],
            "尝试次数已用完",
            execution_id="move-progress-2",
        )
        self.assertTrue(second["retry_escalated"])
        self.assertEqual(controller.snapshot()["execution_mode"], "plan")
        self.assertEqual(controller.snapshot()["status"], "needs_replan")

    def test_v3_auto_verifier_uses_bounded_pick_retry(self) -> None:
        controller = self._v3_controller()
        proposal = controller.create_plan("pi0_pick", ["bowl is localized"])
        controller.register_plan(proposal["proposal_id"])

        first = controller.after_tool(
            "pi0_pick",
            {"prompt": "pick the bowl"},
            {"success": False},
            execution_id="pick-1",
        )
        self.assertEqual(first["execution_mode"], "recover")
        self.assertEqual(first["last_verdict"]["verdict"], "retry")
        self.assertEqual(first["active_step_attempts"], 1)

        second = controller.after_tool(
            "pi0_pick",
            {"prompt": "pick the bowl"},
            {"success": False},
            execution_id="pick-2",
        )
        self.assertEqual(second["execution_mode"], "plan")
        self.assertEqual(second["status"], "needs_replan")
        self.assertEqual(second["last_verdict"]["verdict"], "retry")

    def test_v3_auto_verifier_respects_chinese_mode(self) -> None:
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=1,
            physical_tools={"pi0_pick"},
            plan_generator=ChinesePlanGenerator(),
            use_chinese=True,
            require_execution_id=True,
            auto_verify_tool_signals=True,
        )
        proposal = controller.create_plan("抓取碗", ["碗的位置清晰可见"])
        controller.register_plan(proposal["proposal_id"])
        state = controller.after_tool(
            "pi0_pick",
            {"prompt": "抓取碗"},
            {"success": False, "lift_height": 0.0},
            execution_id="pick-zh-1",
        )

        verdict = state["last_verdict"]
        self.assertEqual(verdict["verdict"], "retry")
        self.assertTrue(
            all(
                any("\u4e00" <= char <= "\u9fff" for char in item)
                for item in verdict["evidence"]
            )
        )
        self.assertIn("抓取", verdict["summary"])
        self.assertIn("重新观察", verdict["recommended_recovery"])

    def test_v3_verdict_source_is_mirrored_without_event_source_collision(
        self,
    ) -> None:
        events_path = Path(self.tmp.name) / "events.jsonl"
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            physical_tools={"pi0_pick"},
            plan_generator=FakePlanGenerator(),
            event_recorder=JsonlEventRecorder(
                events_path,
                episode_id="v3-source",
            ),
            require_execution_id=True,
            auto_verify_tool_signals=True,
        )
        proposal = controller.create_plan("pi0_pick", ["bowl is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool(
            "pi0_pick",
            {"prompt": "pick bowl"},
            {"success": False},
            execution_id="pick-source-1",
        )

        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        verified = next(
            event for event in events if event["event"] == "step_verified"
        )
        self.assertEqual(verified["source"], "harness")
        self.assertEqual(verified["verdict_source"], "tool-signal")

    def test_v3_finish_success_requires_environment_predicate(self) -> None:
        controller = self._v3_controller()
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool(
            "move",
            {},
            {"moved": True},
            execution_id="move-1",
        )
        controller.verify_step(
            "pass",
            ["camera shows the planned motion"],
            "step complete",
            execution_id="move-1",
        )
        self.assertFalse(controller.snapshot()["contract_success"])
        blocked = controller.before_tool(
            "finish",
            {"status": "success", "summary": "done"},
            execution_id="finish-1",
        )
        self.assertFalse(blocked["allowed"])
        self.assertIn("environment success", blocked["reason"])

    def test_v3_environment_signal_completes_plan_and_allows_finish(self) -> None:
        controller = self._v3_controller()
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        state = controller.after_tool(
            "move",
            {},
            {"libero_terminated": True},
            execution_id="move-success",
        )
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["environment_success"])
        self.assertTrue(state["contract_success"])
        allowed = controller.before_tool(
            "finish",
            {"status": "success", "summary": "done"},
            execution_id="finish-1",
        )
        self.assertTrue(allowed["allowed"])

    def test_v3_environment_success_closes_remaining_procedural_steps(self) -> None:
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            physical_tools={"move", "grasp"},
            plan_generator=TwoStepPlanGenerator(),
            require_execution_id=True,
            auto_verify_tool_signals=True,
            require_environment_success=True,
        )
        proposal = controller.create_plan("complete the task", ["target visible"])
        controller.register_plan(proposal["proposal_id"])

        state = controller.after_tool(
            "move",
            {},
            {"libero_terminated": True},
            execution_id="move-success",
        )

        self.assertEqual(state["active_step"]["id"], "grasp-step")
        self.assertTrue(state["environment_success"])
        self.assertTrue(state["contract_success"])
        allowed = controller.before_tool(
            "finish",
            {"status": "success", "summary": "done"},
            execution_id="finish-1",
        )
        self.assertTrue(allowed["allowed"])

    def test_finish_failure_requires_remaining_replan_for_recoverable_error(
        self,
    ) -> None:
        controller = self._controller(max_replans=1)
        proposal = controller.create_plan("grasp", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("grasp", {}, {"ok": True})
        controller.verify_step(
            "fail",
            ["target remains upright and reachable"],
            "grasp missed",
            failure_type="grasp_missed",
        )

        blocked = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "grasp missed"},
        )

        self.assertFalse(blocked["allowed"])
        self.assertIn("replans_remaining=1", blocked["reason"])

        repair = controller.replan(
            "grasp missed",
            ["target remains upright and reachable"],
        )
        controller.register_plan(repair["proposal_id"])
        controller.after_tool("move", {}, {"ok": True})
        controller.verify_step(
            "fail",
            ["target still cannot be reached"],
            "recovery exhausted",
            failure_type="motion_blocked",
        )
        allowed = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "replan budget exhausted"},
        )
        self.assertTrue(allowed["allowed"])

    def test_finish_failure_allows_terminal_controller_error(self) -> None:
        controller = self._controller(max_replans=1)
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("move", {}, {"error": "Broken pipe"})
        controller.verify_step(
            "fail",
            ["environment RPC connection is closed"],
            "controller unavailable",
            failure_type="controller_error",
        )

        allowed = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "environment unavailable"},
        )

        self.assertTrue(allowed["allowed"])

    def test_finish_failure_refused_call_is_not_terminal(self) -> None:
        # 1.0.11 H1: D06 gave up at step 679 of 2100 because a guard refused a
        # carry (ValueError, nothing moved) and that counted as controller_error.
        controller = self._controller(max_replans=1)
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool(
            "move",
            {},
            {"error": "closing an open gripper needs a review", "error_type": "ValueError", "refused": True},
        )
        controller.verify_step(
            "fail",
            ["the tool refused the call before moving"],
            "guard refused the call",
            failure_type="controller_error",
        )

        blocked = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "guard refused the carry"},
        )

        self.assertFalse(blocked["allowed"])
        self.assertIn("replans_remaining=1", blocked["reason"])

    def test_finish_failure_allowed_after_the_episode_ended(self) -> None:
        # End-of-episode errors are terminal, never refusals, so the planner
        # can always finish once nothing can move any more.
        controller = self._controller(max_replans=1)
        proposal = controller.create_plan("move", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool(
            "move",
            {},
            {"error": "motion correction/handoff cannot execute after the episode ended",
             "error_type": "RuntimeError", "refused": False},
        )
        controller.verify_step(
            "fail", ["the episode ended"], "episode ended", failure_type="controller_error",
        )
        allowed = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "episode ended"},
        )
        self.assertTrue(allowed["allowed"])

    def test_finish_failure_allows_exhausted_recovery(self) -> None:
        controller = self._controller(max_replans=2)
        proposal = controller.create_plan("grasp", ["target is localized"])
        controller.register_plan(proposal["proposal_id"])
        controller.after_tool("grasp", {}, {"ok": True})
        controller.verify_step(
            "fail",
            ["the bounded recovery did not retain the target"],
            "recovery exhausted",
            failure_type="recovery_exhausted",
        )

        allowed = controller.before_tool(
            "finish",
            {"status": "failure", "summary": "recovery exhausted"},
        )

        self.assertTrue(allowed["allowed"])

    def test_pick_lock_step_preserves_retry_and_followup_budget(self) -> None:
        controller = Harness(
            mode="plan-gated",
            trace_path=self.trace,
            max_replans=1,
            physical_tools={"pi0_pick", "set_gripper"},
            plan_generator=PickLockPlanGenerator(),
            require_execution_id=True,
            auto_verify_tool_signals=True,
        )
        proposal = controller.create_plan("pick and lock", ["bowl is localized"])
        controller.register_plan(proposal["proposal_id"])
        self.assertEqual(controller.snapshot()["active_step_max_attempts"], 3)

        missed = controller.after_tool(
            "pi0_pick",
            {},
            {"success": False},
            execution_id="pick-missed",
        )
        self.assertEqual(missed["execution_mode"], "recover")

        lifted = controller.after_tool(
            "pi0_pick",
            {},
            {"success": True},
            execution_id="pick-lifted",
        )
        self.assertEqual(lifted["execution_mode"], "act")
        self.assertEqual(lifted["status"], "ready")
        self.assertTrue(controller.before_tool("set_gripper", {})["allowed"])


class ToolkitIntegrationTest(unittest.TestCase):
    def test_toolkit_enforces_gate_and_returns_harness_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Harness(
                mode="plan-gated",
                trace_path=Path(tmp) / "trace.jsonl",
                plan_generator=FakePlanGenerator(),
            )
            toolkit = Toolkit()
            toolkit.configure_harness_physical_tools({"move"})
            toolkit.attach_harness(controller)
            toolkit.add_tool(
                "move", {"name": "move", "input_schema": {}}, lambda: {"moved": True}
            )

            blocked = toolkit.execute_tool("move", {})
            self.assertIn("harness blocked", blocked.result["error"])
            proposed = toolkit.execute_tool(
                "create_plan",
                {
                    "goal": "move",
                    "observations": ["camera shows a clear path"],
                },
            )
            self.assertTrue(proposed.result["ok"])
            still_blocked = toolkit.execute_tool("move", {})
            self.assertIn("register_plan", still_blocked.result["error"])
            registered = toolkit.execute_tool(
                "register_plan",
                {"proposal_id": proposed.result["proposal_id"]},
            )
            self.assertTrue(registered.result["ok"])
            moved = toolkit.execute_tool("move", {})
            self.assertTrue(moved.result["moved"])
            self.assertEqual(moved.result["harness"]["status"], "awaiting_verification")

    def test_v3_toolkit_joins_execution_feedback_and_auto_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Harness(
                mode="plan-gated",
                trace_path=Path(tmp) / "trace.jsonl",
                plan_generator=FakePlanGenerator(),
                require_execution_id=True,
                auto_verify_tool_signals=True,
                require_environment_success=True,
            )
            toolkit = Toolkit()
            toolkit.configure_harness_physical_tools({"pi0_pick"})
            toolkit.attach_harness(controller)
            toolkit.add_tool(
                "pi0_pick",
                {"name": "pi0_pick", "input_schema": {}},
                lambda: {"success": False},
            )
            proposal = toolkit.execute_tool(
                "create_plan",
                {
                    "goal": "pi0_pick",
                    "observations": ["the bowl is localized"],
                },
            )
            toolkit.execute_tool(
                "register_plan",
                {"proposal_id": proposal.result["proposal_id"]},
            )

            attempted = toolkit.execute_tool("pi0_pick", {})

            self.assertEqual(
                attempted.result["harness"]["execution_mode"],
                "recover",
            )
            self.assertEqual(
                attempted.result["harness"]["last_verdict"]["execution_id"],
                attempted.call_id,
            )
            self.assertFalse(attempted.is_finish)


class HarnessComponentTest(unittest.TestCase):
    def test_multi_tool_step_gets_minimum_physical_call_budget(self) -> None:
        state = PlanState(
            max_replans=1,
            physical_tools={"pi0_pick", "set_gripper"},
        )
        registered = state.register_plan(
            "grasp and lock",
            [
                {
                    "id": "pick-lock",
                    "intent": "grasp the bowl and lock the gripper",
                    "allowed_tools": ["pi0_pick", "set_gripper"],
                    "max_attempts": 1,
                }
            ],
        )

        self.assertTrue(registered["ok"])
        self.assertEqual(state.snapshot()["active_step_max_attempts"], 3)

    def test_pure_gate_can_be_tested_without_toolkit_or_trace_io(self) -> None:
        state = PlanState(max_replans=1, physical_tools={"move"})
        gate = PlanGate(mode="plan-gated")
        blocked = gate.evaluate(
            name="move",
            input_dict={},
            state=state,
            control_tools={"create_plan", "register_plan"},
        )
        self.assertFalse(blocked.allowed)
        self.assertIn("register_plan", blocked.reason or "")

        state.register_plan("move", _steps("move"))
        allowed = gate.evaluate(
            name="move",
            input_dict={},
            state=state,
            control_tools={"create_plan", "register_plan"},
        )
        self.assertTrue(allowed.allowed)

    def test_unknown_proposal_cannot_be_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Harness(
                mode="plan-gated",
                trace_path=Path(tmp) / "trace.jsonl",
                physical_tools={"move"},
                plan_generator=FakePlanGenerator(),
            )
            result = controller.register_plan("plan-does-not-exist")
            self.assertIn("unknown plan proposal", result["error"])

    def test_returned_proposal_cannot_rewrite_cached_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Harness(
                mode="plan-gated",
                trace_path=Path(tmp) / "trace.jsonl",
                physical_tools={"move", "grasp"},
                plan_generator=FakePlanGenerator(),
            )
            proposal = controller.create_plan("move", ["target is localized"])
            proposal["steps"][0]["allowed_tools"] = ["grasp"]

            registered = controller.register_plan(proposal["proposal_id"])

            self.assertTrue(registered["ok"])
            self.assertTrue(controller.before_tool("move", {})["allowed"])
            self.assertFalse(controller.before_tool("grasp", {})["allowed"])

    def test_returned_replan_cannot_rewrite_cached_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Harness(
                mode="plan-gated",
                trace_path=Path(tmp) / "trace.jsonl",
                max_replans=1,
                physical_tools={"move", "grasp"},
                plan_generator=FakePlanGenerator(),
            )
            initial = controller.create_plan("grasp", ["target is localized"])
            controller.register_plan(initial["proposal_id"])
            controller.after_tool("grasp", {}, {"ok": True})
            controller.verify_step("fail", ["grasp missed"], "retry")
            repair = controller.replan("grasp missed", ["target remains visible"])
            repair["steps"][0]["allowed_tools"] = ["grasp"]

            registered = controller.register_plan(repair["proposal_id"])

            self.assertTrue(registered["ok"])
            self.assertTrue(controller.before_tool("move", {})["allowed"])
            self.assertFalse(controller.before_tool("grasp", {})["allowed"])


if __name__ == "__main__":
    unittest.main()
