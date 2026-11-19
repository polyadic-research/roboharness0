from __future__ import annotations

from robots.robocasa.outcomes import RoboCasaSkillOutcomeAdapter


def test_unconfirmed_pick_is_retryable_failure() -> None:
    outcome = RoboCasaSkillOutcomeAdapter().classify(
        tool_name="smolvla_pick",
        result={"success": False, "failure_reason": "budget_exhausted"},
        observation=None,
    )

    assert outcome.status == "retryable_failure"
    assert outcome.retryable is True
    assert outcome.failure_type == "grasp_missed"


def test_closed_empty_pick_is_retryable_failure_not_progress() -> None:
    outcome = RoboCasaSkillOutcomeAdapter().classify(
        tool_name="smolvla_pick",
        result={"success": False, "failure_reason": "closed_empty"},
        observation=None,
    )

    assert outcome.status == "retryable_failure"
    assert "closed empty" in " ".join(outcome.evidence)


def test_confirmed_pick_motion_is_progress_not_task_success() -> None:
    outcome = RoboCasaSkillOutcomeAdapter().classify(
        tool_name="smolvla_pick",
        result={
            "success": True,
            "diagnostics": {
                "post_descent_ascent_m": 0.07,
                "final_gripper_opening": 0.012,
            },
        },
        observation=None,
    )

    assert outcome.status == "progress"
    assert outcome.progress == 0.5


def test_continue_vla_is_success_only_for_official_environment_success() -> None:
    adapter = RoboCasaSkillOutcomeAdapter()

    unfinished = adapter.classify(
        tool_name="continue_vla",
        result={"success": False, "failure_reason": "budget_exhausted"},
        observation=None,
    )
    finished = adapter.classify(
        tool_name="continue_vla",
        result={"environment": {"success": True}},
        observation=None,
    )

    assert unfinished.status == "uncertain"
    assert finished.status == "success"


def test_unknown_physical_tool_remains_uncertain() -> None:
    outcome = RoboCasaSkillOutcomeAdapter().classify(
        tool_name="execute_vla",
        result={"requested_actions": 25, "executed_actions": 25},
        observation=None,
    )

    assert outcome.status == "uncertain"
