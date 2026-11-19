"""Tests for provider-independent physical skill outcomes."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from rpent.planning import (
    ActionFeedback,
    JsonlEventRecorder,
    ObservationSummary,
    SkillOutcome,
    action_feedback_from_result,
    load_events,
    recorded_decisions,
)
from rpent.tools.toolkit import Toolkit


class _ExplodingOutcomeAdapter:
    id = "exploding-test-adapter"

    def classify(self, *, tool_name, result, observation):
        del tool_name, result, observation
        raise RuntimeError("adapter exploded")


class _AlwaysProgressAdapter:
    id = "always-progress-test-v1"

    def classify(self, *, tool_name, result, observation):
        del result, observation
        return SkillOutcome(
            status="progress",
            progress=0.5,
            evidence=(f"{tool_name} made test progress",),
            source="adapter",
        )


class SkillOutcomeContractTest(unittest.TestCase):
    def test_rejects_unknown_status_and_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid outcome status"):
            SkillOutcome(status="bogus", evidence=("unsupported status",))
        with self.assertRaisesRegex(ValueError, "invalid outcome source"):
            SkillOutcome(status="uncertain", source="bogus")

    def test_validates_status_semantics_and_finite_metrics(self) -> None:
        outcome = SkillOutcome(
            status="success",
            metrics={"final_dist_m": 0.004},
            evidence=("final distance is within tolerance",),
            source="adapter",
        )
        self.assertEqual(outcome.status, "success")

        with self.assertRaisesRegex(ValueError, "retryable_failure"):
            SkillOutcome(
                status="retryable_failure",
                retryable=False,
                evidence=("target was not reached",),
            )
        with self.assertRaisesRegex(ValueError, "progress"):
            SkillOutcome(status="progress", progress=1.1, evidence=("moving",))
        with self.assertRaisesRegex(ValueError, "failure_type"):
            SkillOutcome(
                status="success",
                failure_type="motion_blocked",
                evidence=("contradictory",),
            )
        for value in (math.inf, math.nan, True, "0.2"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                SkillOutcome(status="uncertain", metrics={"bad": value})

    def test_requires_evidence_for_decisive_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence"):
            SkillOutcome(status="failure", retryable=False)
        self.assertEqual(SkillOutcome(status="uncertain").evidence, ())
        self.assertEqual(SkillOutcome(status="not_applicable").evidence, ())

    def test_legacy_action_feedback_defaults_to_unknown_outcome(self) -> None:
        observation = ObservationSummary(step=0, image_paths=())
        feedback = ActionFeedback(
            execution_id="legacy",
            tool_name="move",
            physical=True,
            execution_status="completed",
            environment_success=False,
            environment_terminated=False,
            environment_truncated=False,
        )

        self.assertEqual(feedback.outcome.status, "uncertain")
        self.assertEqual(feedback.outcome.source, "unknown")

        positional = ActionFeedback("legacy", "move", True, "completed", observation)
        self.assertIs(positional.observation, observation)
        self.assertEqual(positional.outcome.status, "uncertain")


class FeedbackNormalizationTest(unittest.TestCase):
    def test_separates_execution_status_from_physical_outcome(self) -> None:
        physical = action_feedback_from_result(
            execution_id="physical",
            tool_name="move",
            physical=True,
            result={"ok": True},
        )
        read_only = action_feedback_from_result(
            execution_id="read-only",
            tool_name="view_driver_state",
            physical=False,
            result={"ok": True},
        )

        self.assertEqual(physical.execution_status, "completed")
        self.assertEqual(physical.outcome.status, "uncertain")
        self.assertEqual(read_only.execution_status, "completed")
        self.assertEqual(read_only.outcome.status, "not_applicable")

    def test_execution_failures_produce_tool_signal_outcomes(self) -> None:
        cases = (
            ({"error": "blocked"}, True, "rejected", "policy_rejected"),
            ({"error": "controller timeout"}, False, "timeout", "timeout"),
            (
                {"error": "broken pipe"},
                False,
                "controller_error",
                "controller_error",
            ),
        )
        for index, (result, rejected, status, failure_type) in enumerate(cases):
            with self.subTest(status=status):
                feedback = action_feedback_from_result(
                    execution_id=str(index),
                    tool_name="move",
                    physical=True,
                    result=result,
                    rejected=rejected,
                )
                self.assertEqual(feedback.execution_status, status)
                self.assertEqual(feedback.outcome.status, "failure")
                self.assertEqual(feedback.outcome.failure_type, failure_type)
                self.assertEqual(feedback.outcome.source, "tool-signal")

    def test_adapter_exception_degrades_to_uncertain(self) -> None:
        feedback = action_feedback_from_result(
            execution_id="a",
            tool_name="move",
            physical=True,
            result={"ok": True},
            outcome_adapter=_ExplodingOutcomeAdapter(),
        )

        self.assertEqual(feedback.execution_status, "completed")
        self.assertEqual(feedback.outcome.status, "uncertain")
        self.assertIn("adapter exploded", feedback.outcome_adapter_error or "")
        self.assertEqual(feedback.raw_result, {"ok": True})


class ToolkitOutcomeAdapterTest(unittest.TestCase):
    def test_injects_adapter_and_exposes_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(path, episode_id="episode-1")
            toolkit = Toolkit()
            toolkit.configure_harness_physical_tools({"move"})
            toolkit.set_skill_outcome_adapter(_AlwaysProgressAdapter())
            toolkit.attach_event_recorder(recorder)
            toolkit.add_tool(
                "move", {"name": "move", "input_schema": {}}, lambda: {"ok": True}
            )

            toolkit.execute_tool("move", {})

            feedback = recorded_decisions(load_events(path))[0].feedback
            assert feedback is not None
            self.assertEqual(feedback["outcome"]["status"], "progress")
            self.assertEqual(
                toolkit.skill_outcome_adapter_id, "always-progress-test-v1"
            )


if __name__ == "__main__":
    unittest.main()
