"""Policy scope and decomposition guidance must reach every planning surface."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import pytest

import robots.robocasa.policy_skills as skills
import rpent.cli.main as cli
from robots.robocasa.prompts.system import system_prompt
from rpent.envs import get_env_spec
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


def _guidance(family="xiaomi-mibot"):
    return cli._planner_action_guidance(
        "robocasa", {"task_rules": "", "policy_skill": skills.load_policy_skill(family)}
    )


def test_policy_scope_and_success_only_typical_steps_load():
    text = skills.load_policy_skill("xiaomi-mibot")
    assert "任务范围判定" in text
    assert "Typical steps" in text
    assert "StoreLeftoversInBowl" in text
    assert "65 atomic tasks × 15 episodes" in text
    assert "single operation" in text
    assert "success-only" in text


def test_generic_scope_is_policy_neutral_and_shared_with_executor():
    sections = system_prompt()
    scope = sections["TASK SCOPE AND DECOMPOSITION"]
    assert "two or more ordered operations" in scope
    assert "intervene=true" in scope and "intervene=false" in scope
    assert "preconditions" in scope and "expected_evidence" in scope
    assert "max_attempts" in scope and "retry" in scope
    assert "steps_remaining" in scope and "1.5" in scope
    assert "object_states" in scope and "on_failure" in scope
    assert "strategy" in scope
    assert "budget" in scope and "stall" in scope
    assert "Pi0.5" not in scope and "xiaomi-mibot" not in scope and "D06" not in scope
    assert "PreSoakPan" not in scope and "30%" not in scope
    variables = {"task_rules": "", "policy_skill": skills.load_policy_skill("xiaomi-mibot")}
    executor = get_env_spec("robocasa").prompts.render(
        "system", variables={**variables, "harness_protocol": "existing protocol"}
    )
    assert scope in executor and scope in _guidance()
    assert variables["policy_skill"] in executor
    for name in ("CONTROL MODES", "TRANSITIONS", "TAKEOVER"):
        assert sections[name] in executor and sections[name] in _guidance()
        assert "Pi0.5" not in sections[name] and "xiaomi-mibot" not in sections[name]
    assert "transition between fixtures is its own step" in scope
    assert "pick-and-place whose destination is beyond arm reach becomes pick, carry and place" in scope
    assert "within reach it stays one operation even between different fixtures" in scope
    assert "Reserve one evidence-grounded safe retry" not in scope
    assert "reserve horizon for its monitor fallback" in scope


def test_source_sentence_selection_and_freeze_reach_all_planning_surfaces():
    sections = system_prompt()
    scope = sections["TASK SCOPE AND DECOMPOSITION"]
    action = sections["ACTION"]
    assert "quote it verbatim" in scope
    assert "write one instruction once and freeze it" in scope
    assert "after a stall, failure, retry, recovery or replan" in scope
    assert "fresh current-goal evidence within that same registered operation" in scope
    assert "same original sentence for every covered operation" in scope
    assert "let the policy choose the internal order" in scope
    assert "clear failure evidence" not in scope
    assert "revised instruction retry" not in scope
    assert "intervene=true when entering a new registered long-task operation" in action
    assert "within the same registered operation" in action
    assert "Do not reword the language goal" in action
    assert "Only when no original sentence covers the operation" in action
    runtime = _Runtime()
    generator = RuntimePlanGenerator(runtime, require_outcome_program=False,
                                     action_guidance=_guidance())
    plan = generator.generate(goal="Open the stand mixer head.", observations=[],
                              constraints=[], physical_tools={"continue_vla"})
    generator.generate_replan(goal=plan["goal"], reason="Stall", observations=[],
                              constraints=[], current_plan=plan,
                              physical_tools={"continue_vla"})
    assert all(scope in prompt and action in prompt for prompt in runtime.prompts)


@pytest.mark.parametrize("family", ["xiaomi-mibot", "unknown-policy"])
def test_stall_timing_guard_is_generic_and_shared_with_planners(family):
    scope = system_prompt()["TASK SCOPE AND DECOMPOSITION"]
    guard = (
        "Before the operation reaches the matched skill row's typical steps, "
        "do not make stall-driven physical corrections unless fresh evidence "
        "shows a drop, collision or wrong target."
    )
    assert guard in scope
    assert "elapsed confirmed control transitions since operation entry" in scope
    assert "preserve this count across replanning of the same operation" in scope
    assert "A budget overrun alone does not prove a stall or authorize correction" in scope
    assert "For short and long operations alike" in scope
    assert "193" not in scope and "D04" not in scope
    assert guard in _guidance(family)
    assert "distinct pre-close wrist-misalignment review" in scope
    assert "Productive target contact does not qualify as a collision exception" in scope
    # 1.0.10: the protection is scoped to policy-driven steps only.
    assert "The protection binds only delegate and monitor steps" in scope
    assert "takeover and transition steps are driven by you" in scope


@pytest.mark.parametrize("sentence", [
    "Pick the chicken drumstick and eggplant from their plates and place them in the bowl.",
    "Close the microwave door and press the start button.",
])
def test_one_original_sentence_can_survive_as_two_separately_verified_operations(sentence):
    class SharedSentenceRuntime:
        def solve(self, **kwargs):
            return RuntimeResult(finish_result={"strategy": "Two separately verified operations.",
                "steps": [{"id": f"operation-{index}", "intent": f'VLA: "{sentence}"',
                           "allowed_tools": ["continue_vla"],
                           "expected_evidence": [f"Operation {index} visibly complete."]}
                          for index in (1, 2)]})

    generator = RuntimePlanGenerator(SharedSentenceRuntime(), require_outcome_program=False,
                                     action_guidance=_guidance())
    plan = generator.generate(goal=sentence, observations=[], constraints=[],
                              physical_tools={"continue_vla"})
    replacement = generator.generate_replan(goal=sentence, reason="Physical correction needed.",
        observations=[], constraints=[], current_plan=plan, physical_tools={"continue_vla"})
    assert plan["steps"] == replacement["steps"]
    assert len(plan["steps"]) == 2
    assert all(step["intent"] == f'VLA: "{sentence}"' for step in plan["steps"])


class _Runtime:
    def __init__(self):
        self.prompts = []

    def solve(self, *, system_prompt, **kwargs):
        self.prompts.append(system_prompt)
        return RuntimeResult(
            finish_result={
                "strategy": "Scope: short; one complete operation.",
                "steps": [
                    {
                        "id": "operate",
                        "intent": "Open the stand mixer head.",
                        "allowed_tools": ["continue_vla"],
                        "preconditions": ["Mixer head closed."],
                        "expected_evidence": ["Mixer head open."],
                        "on_failure": "Review evidence and replan within the remaining horizon.",
                        "max_attempts": 8,
                    }
                ],
            }
        )


@pytest.mark.parametrize("family", ["xiaomi-mibot", "unknown-policy"])
def test_initial_and_replacement_plans_receive_scope_without_schema_changes(family):
    runtime = _Runtime()
    generator = RuntimePlanGenerator(
        runtime, require_outcome_program=False, action_guidance=_guidance(family)
    )
    plan = generator.generate(
        goal="Open the stand mixer head.",
        observations=[],
        constraints=[],
        physical_tools={"continue_vla"},
    )
    replacement = generator.generate_replan(
        goal=plan["goal"],
        reason="Observed stall.",
        observations=[],
        constraints=[],
        current_plan=plan,
        physical_tools={"continue_vla"},
    )
    assert len(plan["steps"]) == len(replacement["steps"]) == 1
    assert plan["strategy"].startswith("Scope: short")
    assert all("TASK SCOPE AND DECOMPOSITION" in p for p in runtime.prompts)
    if family == "unknown-policy":
        assert all("two or more ordered operations" in p for p in runtime.prompts)
        assert all("POLICY SKILL" not in p for p in runtime.prompts)


def test_missing_skill_keeps_default_classification_and_planner_running(tmp_path):
    with patch.object(skills, "SKILL_DIRECTORY", tmp_path):
        assert skills.load_policy_skill("xiaomi-mibot") == ""
        test_initial_and_replacement_plans_receive_scope_without_schema_changes("xiaomi-mibot")
        assert "two or more ordered operations" in _guidance()
        assert "POLICY SKILL" not in _guidance()


def test_long_operation_metadata_survives_candidate_validation():
    # Existing text fields must survive the actual candidate parser. Otherwise
    # a prompt could ask for budgets/instructions that disappear before execution.
    operation = {
        "id": "activate",
        "intent": 'VLA: "Turn on the microwave."; row=TurnOnMicrowave; tier=weak.',
        "allowed_tools": ["continue_vla"],
        "preconditions": ["Food supported inside; door closed; not held."],
        "expected_evidence": ["Microwave visibly activated."],
        "on_failure": "Attempt budget=450; retry count=1; reserve=300; total=750; review stall.",
        "max_attempts": 12,
    }

    class LongRuntime:
        def solve(self, **kwargs):
            return RuntimeResult(
                finish_result={
                    "strategy": "Scope: long; placement then activation. Total <= steps_remaining.",
                    "steps": [deepcopy(operation)],
                }
            )

    plan = RuntimePlanGenerator(
        LongRuntime(), require_outcome_program=False, action_guidance=_guidance()
    ).generate(
        goal="Place the food, then start the microwave.",
        observations=[],
        constraints=["steps_remaining=1000"],
        physical_tools={"continue_vla"},
    )
    assert plan["steps"] == [operation]
    assert plan["strategy"].startswith("Scope: long")
