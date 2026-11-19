from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rpent.dashboard.state import State
from rpent.harness.hooks import StructuredFeedbackVerifierHook, ToolHookContext
from rpent.harness.state import PlanStateMachine
from rpent.harness.verification import (
    PredicateVerificationRequest,
    VerificationCoordinator,
    VerificationPolicy,
)
from rpent.harness.verifiers import EnvironmentPredicateVerifier
from rpent.planning.feedback import action_feedback_from_result


def _state_result(*, environment: dict | None = None, **extra) -> dict:
    result = {
        "step": 1,
        "state": {
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_gripper_qpos": [0.04, -0.04],
        },
        "log": {"command": {"action": "move"}, "result": {}},
        **extra,
    }
    if environment is not None:
        result["environment"] = environment
    return result


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {"success": True, "terminated": False, "truncated": False},
            (True, False, False),
        ),
        (
            {"success": False, "terminated": True, "truncated": False},
            (False, True, False),
        ),
        (
            {"success": False, "terminated": False, "truncated": True},
            (False, False, True),
        ),
    ],
)
def test_nested_environment_status_is_normalized_without_conflating_signals(
    environment, expected
) -> None:
    feedback = action_feedback_from_result(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        result=_state_result(environment=environment),
    )

    assert (
        feedback.environment_success,
        feedback.environment_terminated,
        feedback.environment_truncated,
    ) == expected
    assert feedback.observation is not None
    assert (
        feedback.observation.environment_success,
        feedback.observation.environment_terminated,
        feedback.observation.environment_truncated,
    ) == expected


def test_legacy_libero_success_is_translated_only_when_generic_block_is_absent() -> None:
    legacy = action_feedback_from_result(
        execution_id="legacy",
        tool_name="move",
        physical=True,
        result=_state_result(libero_terminated=True),
    )
    generic = action_feedback_from_result(
        execution_id="generic",
        tool_name="move",
        physical=True,
        result=_state_result(
            environment={"success": False, "terminated": True, "truncated": False},
            libero_terminated=True,
        ),
    )

    assert legacy.environment_success is True
    assert legacy.environment_terminated is False
    assert legacy.environment_truncated is False
    assert generic.environment_success is False
    assert generic.environment_terminated is True


def test_legacy_libero_state_artifact_preserves_truncation() -> None:
    feedback = action_feedback_from_result(
        execution_id="legacy-truncated",
        tool_name="view_driver_state",
        physical=False,
        result=_state_result(
            libero_terminated=False,
            episode_done=True,
            episode_truncated=True,
        ),
    )

    assert feedback.environment_success is False
    assert feedback.environment_terminated is False
    assert feedback.environment_truncated is True


def test_malformed_truthy_environment_values_fail_closed() -> None:
    feedback = action_feedback_from_result(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        result=_state_result(
            environment={
                "success": "false",
                "terminated": 1,
                "truncated": [True],
            },
            libero_terminated="true",
        ),
    )

    assert feedback.environment_success is False
    assert feedback.environment_terminated is False
    assert feedback.environment_truncated is False


def test_stale_environment_success_cannot_authorize_predicate() -> None:
    feedback = action_feedback_from_result(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        result=_state_result(
            environment={"success": True, "terminated": False, "truncated": False}
        ),
    )
    request = PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="task success",
        execution_id="exec-1",
        observation_refs=("obs-1",),
        observed_at=("2026-08-04T00:00:00+00:00",),
        feedback=feedback,
    )
    coordinator = VerificationCoordinator(
        (EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-1"}),),
        policy=VerificationPolicy(
            max_observation_age_s=60,
            clock=lambda: datetime(2026, 8, 5, tzinfo=timezone.utc),
        ),
    )

    result = coordinator.verify(request)

    assert result.authorized is False
    assert result.verdict == "uncertain"
    assert "stale" in result.reason


def test_dashboard_payload_exposes_all_generic_environment_fields(tmp_path) -> None:
    state = State(
        run_id="run-1",
        name="case",
        suite="suite",
        task=1,
        seed=2,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )
    state.on_tool_result(
        "move",
        _state_result(
            environment={"success": False, "terminated": True, "truncated": True}
        ),
    )

    detail = state.run_detail()

    assert detail["environment_success"] is False
    assert detail["environment_terminated"] is True
    assert detail["environment_truncated"] is True
    assert detail["timeline"][0]["environment_success"] is False
    assert detail["timeline"][0]["environment_terminated"] is True
    assert detail["timeline"][0]["environment_truncated"] is True


def test_structured_hook_passes_only_environment_success() -> None:
    state = PlanStateMachine(max_replans=0, physical_tools={"move"})
    state.register_plan(
        "move",
        [{"id": "move", "intent": "move", "allowed_tools": ["move"]}],
    )
    context = ToolHookContext(
        execution_id="exec-1",
        name="move",
        input_dict={},
        physical=True,
    )
    hook = StructuredFeedbackVerifierHook()

    terminated = action_feedback_from_result(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        result=_state_result(
            environment={"success": False, "terminated": True, "truncated": False}
        ),
    )
    succeeded = action_feedback_from_result(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        result=_state_result(
            environment={"success": True, "terminated": False, "truncated": False}
        ),
    )

    assert hook.after_tool(context, terminated, state) is None
    verdict = hook.after_tool(context, succeeded, state)
    assert verdict is not None
    assert verdict.verdict == "pass"
    assert verdict.evidence == ("environment_success=true",)
