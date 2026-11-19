"""Tests for normalized planning contracts, events, and runtime boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rpent.planning import (
    JsonlEventRecorder,
    PlanningOrchestrator,
    action_feedback_from_result,
    build_experiment_manifest,
    load_events,
    observation_from_tool_result,
    recorded_decisions,
    resolve_planning_stack,
)
from rpent.runtime import PlannerResult, RuntimeResult
from rpent.runtime.finish import contains_finish_sentinel
from rpent.tools.common import finish
from rpent.tools.toolkit import Toolkit


def _state_result(step: int = 1) -> dict:
    return {
        "step": step,
        "task_language": "move the bowl to the plate",
        "state": {
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_gripper_qpos": [0.04, -0.04],
            "obj_of_interest": "bowl",
            "object_names": ["bowl", "plate"],
        },
        "image_cam_hi_path": f"images_cam_hi/image_cam_hi_{step:02d}.png",
        "libero_terminated": False,
        "log": {
            "command": {"action": "move"},
            "result": {"final_dist_m": 0.01},
        },
    }


class ObservationAndFeedbackTest(unittest.TestCase):
    def test_manifest_prefers_deployment_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            deployed_commit = "2" * 40
            (repo_root / ".rpent-release").write_text(
                f"SOURCE_COMMIT={deployed_commit}\n",
                encoding="utf-8",
            )

            manifest = build_experiment_manifest(
                config=resolve_planning_stack("libero-reactive-v1"),
                planner_backend="api",
                model=None,
                max_tokens=100,
                no_images=False,
                system_prompt="system",
                user_message="move the bowl",
                tool_schemas=[],
                repo_root=repo_root,
            )

        self.assertEqual(manifest["code"]["git_commit"], deployed_commit)

    def test_normalizes_visible_robot_state(self) -> None:
        observation = observation_from_tool_result(_state_result())
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.step, 1)
        self.assertEqual(
            observation.task_instruction,
            "move the bowl to the plate",
        )
        self.assertEqual(observation.eef_position, (0.1, 0.2, 0.3))
        self.assertAlmostEqual(observation.gripper_opening or 0, 0.08)
        self.assertEqual(observation.holding_object, "bowl")
        self.assertEqual(
            observation.image_paths,
            ("images_cam_hi/image_cam_hi_01.png",),
        )

    def test_feedback_distinguishes_rejection_and_controller_error(self) -> None:
        rejected = action_feedback_from_result(
            execution_id="a",
            tool_name="move",
            physical=True,
            result={"error": "blocked"},
            rejected=True,
        )
        failed = action_feedback_from_result(
            execution_id="b",
            tool_name="move",
            physical=True,
            result={"error": "controller failed"},
        )
        self.assertEqual(rejected.execution_status, "rejected")
        self.assertEqual(failed.execution_status, "controller_error")

    def test_manifest_records_selected_outcome_adapter(self) -> None:
        manifest = build_experiment_manifest(
            config=resolve_planning_stack("libero-reactive-v1"),
            planner_backend="api",
            model=None,
            max_tokens=100,
            no_images=False,
            system_prompt="system",
            user_message="move the bowl",
            tool_schemas=[],
            repo_root=Path.cwd(),
            skill_outcome_adapter_id="libero-skill-outcome-v1",
        )

        self.assertEqual(
            manifest["execution"]["skill_outcome_adapter_id"],
            "libero-skill-outcome-v1",
        )

    def test_manifest_outcome_adapter_argument_is_backward_compatible(self) -> None:
        manifest = build_experiment_manifest(
            config=resolve_planning_stack("libero-reactive-v1"),
            planner_backend="api",
            model=None,
            max_tokens=100,
            no_images=False,
            system_prompt="system",
            user_message="move the bowl",
            tool_schemas=[],
            repo_root=Path.cwd(),
        )

        self.assertEqual(
            manifest["execution"]["skill_outcome_adapter_id"],
            "unknown-physical-outcome-v1",
        )

    def test_manifest_records_authoritative_runtime_interface(self) -> None:
        stack = resolve_planning_stack("libero-authoritative-gated-v1")
        runtime_metadata = {
            "interface": "authoritative-harness-session-v1",
            "predicate_verifier": {
                "name": "runtime-predicate",
                "version": "1",
            },
        }

        manifest = build_experiment_manifest(
            config=stack,
            planner_backend="codex",
            model="test-model",
            max_tokens=100,
            no_images=False,
            system_prompt="system",
            user_message="user",
            tool_schemas=[],
            repo_root=".",
            runtime_metadata=runtime_metadata,
        )

        self.assertEqual(manifest["execution"]["runtime"], runtime_metadata)


class EventRecorderTest(unittest.TestCase):
    def test_records_ordered_json_without_binary_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(path, episode_id="episode-1")
            recorder.record("first", image=b"\x89PNG")
            recorder.record("second", value=2)
            events = load_events(path)
            self.assertEqual([event["sequence"] for event in events], [1, 2])
            self.assertEqual(events[0]["image"]["bytes"], 4)
            self.assertNotIn("PNG", path.read_text())

    def test_toolkit_trace_joins_decision_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(path, episode_id="episode-1")
            toolkit = Toolkit()
            toolkit.configure_harness_physical_tools({"move"})
            toolkit.attach_event_recorder(recorder)
            toolkit.add_tool(
                "move",
                {"name": "move", "input_schema": {}},
                lambda: _state_result(),
            )

            result = toolkit.execute_tool("move", {})
            self.assertIsNotNone(result.call_id)
            decisions = recorded_decisions(load_events(path))
            self.assertEqual(len(decisions), 1)
            self.assertTrue(decisions[0].physical)
            assert decisions[0].feedback is not None
            self.assertEqual(
                decisions[0].feedback["execution_status"],
                "completed",
            )

    def test_manifest_has_stable_policy_hash_across_output_directories(self) -> None:
        common = {
            "config": resolve_planning_stack("libero-reactive-v1"),
            "planner_backend": "api",
            "model": "test:model",
            "max_tokens": 100,
            "no_images": False,
            "user_message": "move the bowl",
            "repo_root": Path.cwd(),
        }
        first = build_experiment_manifest(
            **common,
            system_prompt="write /tmp/run-a/result.json",
            tool_schemas=[{"name": "write", "description": "/tmp/run-a"}],
            volatile_values={"/tmp/run-a": "{{output_dir}}"},
        )
        second = build_experiment_manifest(
            **common,
            system_prompt="write /tmp/run-b/result.json",
            tool_schemas=[{"name": "write", "description": "/tmp/run-b"}],
            volatile_values={"/tmp/run-b": "{{output_dir}}"},
        )
        self.assertNotEqual(
            first["artifacts"]["system_prompt_sha256"],
            second["artifacts"]["system_prompt_sha256"],
        )
        self.assertEqual(
            first["artifacts"]["policy_prompt_sha256"],
            second["artifacts"]["policy_prompt_sha256"],
        )
        self.assertEqual(
            first["artifacts"]["tool_schema_sha256"],
            second["artifacts"]["tool_schema_sha256"],
        )


class _FakeRuntime:
    def solve(self, **kwargs) -> PlannerResult:
        return PlannerResult(
            finish_result={"status": "success"},
            stats={"tool_calls": 0},
        )


class _ContractRuntime:
    def solve(self, **kwargs) -> PlannerResult:
        return PlannerResult(
            finish_result={
                "status": "success",
                "outcome_program": {
                    "schema_version": "outcome-program/v1",
                    "contract_id": "contract-test",
                },
                "required_outcome_ids": ["outcome-2"],
                "stages": [
                    {
                        "id": "stage-2",
                        "intent": "place",
                        "outcome_ids": ["outcome-2"],
                    }
                ],
            }
        )


class RuntimeResultTest(unittest.TestCase):
    def test_preserves_stop_reason_in_public_result_and_stats(self) -> None:
        result = RuntimeResult(stop_reason="turn_limit", stats={"tool_calls": 2})

        self.assertEqual(result.stop_reason, "turn_limit")
        self.assertEqual(result.stats["stop_reason"], "turn_limit")

    def test_reads_legacy_stop_reason_from_stats(self) -> None:
        result = RuntimeResult(stats={"stop_reason": "natural_end"})

        self.assertEqual(result.stop_reason, "natural_end")


class PlanningOrchestratorTest(unittest.TestCase):
    def test_records_backend_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(path, episode_id="episode-1")
            orchestrator = PlanningOrchestrator(_FakeRuntime(), recorder=recorder)
            result = orchestrator.run(
                system_prompt="system",
                user_message="move the bowl",
                toolkit=Toolkit(),
                max_turns=3,
            )
            self.assertEqual(result.finish_result, {"status": "success"})
            names = [event["event"] for event in load_events(path)]
            self.assertEqual(names, ["planning_started", "planning_finished"])

    def test_records_staged_contract_identity_without_mutating_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(path, episode_id="episode-1")
            result = PlanningOrchestrator(_ContractRuntime(), recorder=recorder).run(
                system_prompt="system",
                user_message="place the cup",
                toolkit=Toolkit(),
                max_turns=3,
            )

            event = load_events(path)[-1]
            self.assertEqual(event["contract_id"], "contract-test")
            self.assertEqual(
                event["outcome_program_schema_version"], "outcome-program/v1"
            )
            self.assertEqual(event["required_outcome_ids"], ["outcome-2"])
            self.assertEqual(event["stage_ids"], ["stage-2"])
            self.assertEqual(
                result.finish_result["outcome_program"]["contract_id"],
                "contract-test",
            )


class FinishSentinelTest(unittest.TestCase):
    def test_backends_only_stop_after_successful_finish_tool_result(self) -> None:
        blocked = {
            "content": [
                {
                    "type": "text",
                    "text": '{"error":"harness blocked finish"}',
                }
            ]
        }
        completed = {
            "content": [
                {
                    "type": "text",
                    "text": '{"_finish":true,"status":"success"}',
                }
            ]
        }
        self.assertFalse(contains_finish_sentinel(blocked))
        self.assertTrue(contains_finish_sentinel(completed))

    def test_common_finish_rejects_stuck_and_normalizes_valid_status(self) -> None:
        rejected = finish("stuck", "cannot continue")
        completed = finish(" SUCCESS ", "all verified")
        self.assertNotIn("_finish", rejected)
        self.assertIn("must be 'success' or 'failure'", rejected["error"])
        self.assertEqual(completed["status"], "success")
        self.assertTrue(completed["_finish"])


if __name__ == "__main__":
    unittest.main()
