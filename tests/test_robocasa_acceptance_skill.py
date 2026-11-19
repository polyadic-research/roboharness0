"""Static acceptance knowledge reaches both LLMs without live simulator reads."""

from __future__ import annotations

import hashlib
import sys
from unittest.mock import patch

import pytest

from robots.robocasa.acceptance_skills import load_acceptance_skill
from rpent.cli import main as cli
from rpent.envs import get_env_spec
from rpent.runtime import RuntimePlanGenerator, RuntimeResult


def config(task="PortionHotDogs", policy="xiaomi-mibot"):
    with patch.object(sys, "argv", ["rpent", "--env", "robocasa", "--task", task, "--vla", policy]):
        parser = cli._build_argparser()
        early, _ = parser.parse_known_args()
        spec = get_env_spec(early.env_name)
        spec.add_cli_args(parser, use_dashboard=False)
        return spec.parse_config(parser.parse_args())


def test_exact_task_loads_only_its_acceptance_reference():
    skill = load_acceptance_skill("PortionHotDogs")
    assert skill["metadata"]["matched_tasks"] == ["PortionHotDogs"]
    assert skill["metadata"]["advisory_only"] is True
    assert skill["metadata"]["content_sha256"] == hashlib.sha256(skill["text"].encode()).hexdigest()
    assert "sausage" in skill["text"] and "direct contact" in skill["text"]
    assert "either bowl may be the upper bowl" not in skill["text"]
    assert "both tools" not in skill["text"]


@pytest.mark.parametrize("task", ["CloseFridge", "../PortionHotDogs", "UnknownTask"])
def test_uncovered_task_has_no_invented_task_predicate(task):
    skill = load_acceptance_skill(task)
    assert skill["metadata"]["matched_tasks"] == []
    assert len(skill["metadata"]["files_sha256"]) == 1  # manifest only; no advice loaded
    assert skill["text"] == ""


@pytest.mark.parametrize("policy", ["xiaomi-mibot"])
def test_executor_and_nested_planner_receive_reference_for_each_policy(policy):
    run = config(policy=policy)
    skill = load_acceptance_skill("PortionHotDogs")
    assert run.task_desc["acceptance_skill"] == skill["metadata"]
    executor = get_env_spec("robocasa").prompts.render(
        "system", variables={**run.prompt_vars, "harness_protocol": "existing protocol"},
    )
    planner = cli._planner_action_guidance("robocasa", run.prompt_vars)
    assert skill["text"] in executor and skill["text"] in planner
    assert run.prompt_vars["policy_skill"] not in ("", skill["text"]) or policy == "smolvla"


def test_multiple_exact_task_selector_does_not_load_unrelated_references():
    skill = load_acceptance_skill("WeighIngredients,PortionHotDogs")
    assert skill["metadata"]["matched_tasks"] == ["PortionHotDogs", "WeighIngredients"]
    assert "packaged ingredient" in skill["text"]
    assert "both tools" not in skill["text"]


@pytest.mark.parametrize("task,target", [
    ("WeighIngredients", "packaged ingredient"),
    ("RestockSinkSupplies", "soap dispenser"),
    ("OrganizeMeasuringCups", "all three measuring cups"),
    ("StrainerSetup", "tea strainer"),
])
def test_upright_advice_is_bound_to_verified_task_and_object(task, target):
    skill = load_acceptance_skill(task)
    assert skill["metadata"]["matched_tasks"] == [task]
    assert skill["metadata"]["conditions"] == ["upright_placement"]
    assert target in skill["text"]
    assert "sausage" not in skill["text"]


@pytest.mark.parametrize("task", ["CuttingToolSelection", "StackBowlsCabinet"])
def test_unrequested_condition_families_are_not_loaded(task):
    skill = load_acceptance_skill(task)
    assert skill["metadata"]["matched_tasks"] == []
    assert skill["metadata"]["conditions"] == []


def test_reference_flows_into_actual_initial_and_repair_planning_requests():
    prompts = []

    class Runtime:
        def solve(self, *, system_prompt, **kwargs):
            prompts.append(system_prompt)
            return RuntimeResult(finish_result={
                "_finish": True, "strategy": "Inspect the placement support.",
                "steps": [{"id": "place", "intent": "Place a sausage beside the bun.",
                           "allowed_tools": ["continue_vla"],
                           "expected_evidence": ["Sausage supported directly by plate."]}],
            })

    run = config()
    generator = RuntimePlanGenerator(Runtime(), require_outcome_program=False,
        action_guidance=cli._planner_action_guidance("robocasa", run.prompt_vars))
    plan = generator.generate(goal="Place one bun and one sausage on each plate.",
        observations=[], constraints=[], physical_tools={"continue_vla"})
    generator.generate_replan(goal="Place one bun and one sausage on each plate.", reason="Official success remains false.",
        observations=[], constraints=[], current_plan=plan, physical_tools={"continue_vla"})
    text = load_acceptance_skill("PortionHotDogs")["text"]
    assert len(prompts) == 2 and all(text in prompt for prompt in prompts)
