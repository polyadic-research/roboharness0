"""Occlusion must preserve grounded progress without authorizing final success."""

from robots.robocasa.prompts.system import system_prompt
from rpent.cli.main import _planner_action_guidance
from rpent.harness.hooks import ToolHookContext
from rpent.planning import resolve_planning_stack
from rpent.planning.schemas import ActionFeedback
from rpent.runtime import RuntimePlanGenerator, RuntimeResult

PROFILE = "robocasa-horizon-gated-v1"


def test_occluded_target_can_use_textual_feedback_and_confirmed_history():
    sections = system_prompt()
    evidence = sections["EVIDENCE"]
    assert (
        "images and returned textual information are valid evidence sources" in evidence
    )
    assert "occluded, unavailable, or inconclusive" in evidence
    assert "fresh measured robot feedback" in evidence
    assert "read_action_history" in evidence
    assert "confirmed execution history" in evidence
    assert "mark inferences explicitly" in evidence
    assert "Occlusion alone does not invalidate" in evidence
    assert "commands alone" in evidence
    assert "unexecuted" in evidence
    assert (
        "Before concluding a grasp, transport, or placement succeeded, use fresh camera views"
        not in sections["OBSERVATION"]
    )


def test_horizon_verification_resolves_occlusion_without_an_identical_view_loop():
    stack = resolve_planning_stack(PROFILE)
    protocol = stack.prompt_context()
    assert "returned textual feedback" in protocol
    assert "read-only history inspection" in protocol
    assert "Repeating an unchanged occluded view is not new evidence" in protocol
    assert (
        "Uncertain evidence requires another observation, never an unchecked action"
        not in protocol
    )
    assert "verify_step in verify mode" in protocol
    assert "finish(status='success') requires that signal" in protocol
    assert stack.profile.verifier_prompt_version == "multimodal-feedback-verifier-v3"


class _RecordingPlanner:
    def __init__(self):
        self.prompts = []

    def solve(self, *, system_prompt, **_kwargs):
        self.prompts.append(system_prompt)
        return RuntimeResult(
            finish_result={
                "_finish": True,
                "strategy": "Continue the grounded cabinet placement and clearance.",
                "steps": [
                    {
                        "id": "clear",
                        "intent": "Move the open gripper clear of the placed sugar cube.",
                        "allowed_tools": ["continue_vla"],
                        "expected_evidence": ["Fresh feedback confirms clearance."],
                    }
                ],
            }
        )


def test_initial_and_repair_planners_receive_the_same_evidence_fallback():
    runtime = _RecordingPlanner()
    guidance = _planner_action_guidance(
        "robocasa",
        {
            "task_rules": "No task-specific procedure is configured for this task.",
            "policy_skill": "",
        },
    )
    generator = RuntimePlanGenerator(
        runtime,
        require_outcome_program=False,
        action_guidance=guidance,
    )
    observations = [
        "The previously identified sugar cube was held over the cabinet shelf.",
        "Measured gripper width is now 79 mm; current images occlude the cube.",
    ]
    generator.generate(
        goal="Place the sugar cube in the cabinet.",
        observations=observations,
        constraints=["Native environment success remains false."],
        physical_tools={"continue_vla"},
    )
    generator.generate_replan(
        goal="Place the sugar cube in the cabinet.",
        reason="Release progressed but clearance remains unfinished.",
        observations=observations,
        constraints=["Use remaining episode transitions."],
        current_plan={"strategy": "placement", "unfinished_steps": []},
        physical_tools={"continue_vla"},
    )
    assert len(runtime.prompts) == 2
    for prompt in runtime.prompts:
        assert system_prompt()["EVIDENCE"] in prompt
        assert "unexecuted" in prompt
        assert "final success" in prompt


def test_log_grounded_progress_still_obeys_verification_and_success_gates(tmp_path):
    harness = resolve_planning_stack(PROFILE).build_harness(
        trace_path=tmp_path / "trace.jsonl",
    )
    harness.set_physical_tools({"continue_vla"})
    assert harness.state.register_plan(
        "Place sugar cube in cabinet.",
        [
            {
                "id": "place_and_clear",
                "intent": "Place and clear the sugar cube.",
                "allowed_tools": ["continue_vla"],
            }
        ],
    )["ok"]
    harness.state.record_action_feedback(
        ActionFeedback(
            execution_id="release-1",
            tool_name="continue_vla",
            physical=True,
            execution_status="completed",
            environment_success=False,
        ),
        input_dict={},
    )
    next_action = ToolHookContext("clear-1", "continue_vla", {}, True)
    assert not harness.hooks.before_tool(next_action, harness.state).allowed
    assert harness.state.verify_step(
        "progress",
        [
            "step1375 target held over shelf; release-1 measured open gripper; placement inferred, clearance unfinished"
        ],
        "Preserve placement progress despite occlusion.",
        execution_id="release-1",
    )["ok"]
    assert harness.hooks.before_tool(next_action, harness.state).allowed
    assert not harness.hooks.before_tool(
        ToolHookContext(
            "finish-1",
            "finish",
            {"status": "success"},
            False,
        ),
        harness.state,
    ).allowed
