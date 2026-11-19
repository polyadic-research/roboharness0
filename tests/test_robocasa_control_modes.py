"""1.0.10 block A: per-step control modes, transitions and takeover are prompt/skill rules.

They add no lifecycle state, tool or control-loop branch. These tests pin the
shared vocabulary, the plumbing to every planning surface, the ablation seams
and the consistency between tool names in the prompt and the tool schemas.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import robots.robocasa.policy_skills as skills
import rpent.cli.main as cli
from robots.robocasa import control_modes as vocab
from robots.robocasa.prompts import system as prompts
from robots.robocasa.prompts.system import system_prompt, task_rules
from rpent.envs import get_env_spec
from rpent.planning import resolve_planning_stack

NEW_SECTIONS = ("CONTROL MODES", "TRANSITIONS", "TAKEOVER")
SKILL_HEADINGS = ("## Control modes", "## Transitions", "## Takeover")


def _skill() -> str:
    return skills.load_policy_skill("xiaomi-mibot")


def _planner_guidance() -> str:
    return cli._planner_action_guidance(
        "robocasa", {"task_rules": "", "policy_skill": _skill()}
    )


def _executor(policy_skill: str | None = None) -> str:
    return get_env_spec("robocasa").prompts.render(
        "system",
        variables={
            "harness_protocol": resolve_planning_stack(
                "robocasa-horizon-gated-v1"
            ).prompt_context(),
            "task_rules": task_rules("StoreLeftoversInBowl", policy_family="xiaomi-mibot"),
            "policy_skill": _skill() if policy_skill is None else policy_skill,
        },
    )


class _Controller:
    def execute_policy_chunk(self, observation, instruction, *, max_actions=None):
        raise AssertionError("schema inspection must not execute the policy")


def _toolkit_tool_names() -> set[str]:
    from robots.robocasa.toolkit import RoboCasaToolkit

    toolkit = RoboCasaToolkit(
        _Controller(),
        initial_observation={
            "state": np.zeros(16, dtype=np.float32),
            "robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        instruction="Open the stand mixer head.",
        policy_instruction="Open the stand mixer head.",
        forward_step_guidance=True,
    )
    return {spec["name"] for spec in toolkit.get_tools_spec()}


# Harness-session and nested-planner tools that are not RoboCasa toolkit tools.
_SESSION_TOOLS = {
    "create_plan", "register_plan", "replan", "verify_step", "finish",
    "provide_planner_information", "read_action_history", "view_plan",
}

# Underscore identifiers in the prompts that are fields, parameters, result
# keys, enum values or evidence tags rather than tool names.
_NON_TOOL_IDENTIFIERS = {
    "progress_evidence", "target_id", "phase_complete", "observed_change", "observation_step",
    "conflict_type", "expected_target", "observed_target",
    "completed_step_reviews", "completion_review", "review_observation_id",
    "return_upright",
    "action_history", "action_order", "allowed_tools", "attempted_action_index",
    "interrupt_reason", "settled_under_load",
    "burner_control", "closed_empty", "closure_confirmed", "compute_typical_steps",
    "contract_success", "control_mode", "control_mode_escalations", "control_source",
    "covers_outcomes", "delta_position",
    "descent_confirmed", "episode_budget", "evidence_scope", "executed_actions",
    "execution_mode", "expected_evidence", "expected_step", "failure_type",
    "first_success_step", "grasp_lost", "handoff_pose_summary", "harness_protocol",
    "information_budget_exhausted", "information_required", "max_actions",
    "max_attempts", "max_chunks", "max_chunks_exhausted", "multitask_learning",
    "object_dropped", "object_states", "on_failure", "outcome_program",
    "pending_execution_id", "per_task", "pi05_pretrain_human300", "planner_prompt",
    "policy_close_allowed", "policy_instruction", "policy_skill",
    "preclose_review_required", "recommended_standoff_quaternion",
    "recommended_standoff_xyz", "recovery_close_persisted", "recovery_exhausted",
    "recovery_pose", "relative_base_xyz", "release_commanded", "request_id",
    "retry_allowed", "robocasa365_checkpoints", "robot0_eye_in_hand",
    "rotation_degrees", "skill_row", "step_ids", "step_intent", "steps_remaining",
    "steps_used", "stop_reason", "target_reached", "task_language", "task_progress",
    "task_rules", "transition_budget", "transitions_used", "verifier_prompt",
    "visible_objects",
    # 1.0.11 fields, result keys, events and tool modules named in prompts/skill.
    "back_projected", "base_stall", "base_stalled", "bowl_to_fridge", "build_handback",
    "control_mode_failures", "control_mode_fallback", "control_mode_fallbacks",
    "control_segments", "control_subtasks", "gripper_settle", "repeat_warning",
    "skill_table", "standoff_m", "target_distance_m", "travel_m", "unsafe_trajectory",
    # 1.2.1 review_grasp argument and value.
    "closure_target", "fixed_control",
    # 1.2.2 review_grasp argument.
    "target_visible",
}


def test_vocabulary_module_is_internally_consistent():
    assert set(vocab.TIER_CONTROL_MODE) == set(vocab.TIERS)
    assert set(vocab.TIER_CONTROL_MODE.values()) <= set(vocab.CONTROL_MODES)
    assert vocab.TIER_CONTROL_MODE["weak"] == vocab.TIER_CONTROL_MODE["none"] == "takeover"
    assert vocab.TIER_CONTROL_MODE["reliable"] == "delegate"
    assert vocab.TIER_CONTROL_MODE["unstable"] == "monitor"
    assert set(vocab.TIER_LABELS) == {"可靠", "不稳定", "弱", "无记录"}
    assert set(vocab.TIER_LABELS.values()) == set(vocab.TIERS)
    assert vocab.POLICY_DRIVEN_MODES | vocab.LLM_DRIVEN_MODES == set(vocab.CONTROL_MODES)
    assert not vocab.POLICY_DRIVEN_MODES & vocab.LLM_DRIVEN_MODES
    assert set(vocab.PRIVILEGED_TOOLS_BY_MODE) == vocab.LLM_DRIVEN_MODES
    for tools in vocab.PRIVILEGED_TOOLS_BY_MODE.values():
        assert set(tools) == set(vocab.PRIVILEGED_TOOLS)
    assert not set(vocab.PRIVILEGED_TOOLS) & set(vocab.POLICY_TOOLS)
    assert not set(vocab.READ_ONLY_GROUNDING_TOOLS) & set(vocab.PRIVILEGED_TOOLS)
    assert vocab.FALLBACK_FAILURES == 2
    assert vocab.ABLATABLE_SECTIONS == NEW_SECTIONS


def test_prompt_text_uses_the_shared_vocabulary_verbatim():
    sections = system_prompt()
    modes = sections["CONTROL MODES"]
    for field in vocab.STEP_FIELDS:
        assert field in modes
    for mode in vocab.CONTROL_MODES:
        assert mode in modes
    for tier in vocab.TIERS:
        assert tier in modes
    for tool in vocab.PRIVILEGED_TOOLS:
        assert tool in modes and tool in sections["TRANSITIONS"] and tool in sections["TAKEOVER"]
    for tool in vocab.READ_ONLY_GROUNDING_TOOLS:
        assert tool in modes
    assert "row=<name>; tier=<tier>; control_mode=<mode>" in modes
    assert "reliable gives delegate" in modes
    assert "unstable gives monitor" in modes
    assert "weak or none gives takeover" in modes
    assert "never match a whole-composite row to one operation" in modes
    assert "Never estimate, transfer or soften a tier" in modes
    assert "entry the skill explicitly declares (a proxy of one measured row or a demonstrated operation" in modes
    assert "the control modes it lists override the mode-by-tier rule" in modes
    assert "transition steps list only the privileged motion tools" in modes
    assert "no policy tool, because the policy does not run and receives no instruction during a carry" in modes
    # Adam 2026-10-09: takeover steps may list the policy tool, for staged takeover only.
    assert "add policy tools only for a staged takeover" in modes
    assert "are never listed in allowed_tools" in modes


def test_new_sections_are_independent_keys_in_a_fixed_order():
    sections = system_prompt()
    names = list(sections)
    assert names.index("TASK SCOPE AND DECOMPOSITION") + 1 == names.index("CONTROL MODES")
    assert names.index("ACTION") + 1 == names.index("TRANSITIONS")
    assert names.index("TRANSITIONS") + 1 == names.index("TAKEOVER")
    assert sections["CONTROL MODES"] == prompts.CONTROL_MODES_GUIDANCE
    assert sections["TRANSITIONS"] == prompts.TRANSITIONS_GUIDANCE
    assert sections["TAKEOVER"] == prompts.TAKEOVER_GUIDANCE
    assert len({sections[name] for name in NEW_SECTIONS}) == 3


def test_new_sections_reach_executor_and_nested_planners():
    sections = system_prompt()
    executor = _executor()
    planner = _planner_guidance()
    for name in NEW_SECTIONS:
        assert sections[name] in executor
        assert sections[name] in planner
    for heading in SKILL_HEADINGS:
        assert heading in executor and heading in planner
    assert "{{" not in planner


@pytest.mark.parametrize("name", NEW_SECTIONS)
def test_new_sections_are_policy_neutral_and_name_no_evaluation_item(name):
    text = system_prompt()[name]
    assert "Pi0.5" not in text and "xiaomi-mibot" not in text and "π0.5" not in text
    assert re.search(r"\bD\d\d\b", text) is None
    assert re.search(r"\bseed\s*[=:]?\s*\d", text) is None  # no evaluation seed
    assert "kettle" not in text.lower()
    assert "max_chunks=" not in text


def test_skill_control_sections_name_no_evaluation_item():
    text = _skill()
    for heading in SKILL_HEADINGS:
        start = text.index(heading)
        end = text.find("\n## ", start + 1)
        body = text[start: end if end > 0 else len(text)]
        assert re.search(r"\bD\d\d\b", body) is None, heading
        assert "seed 7" not in body


def test_timing_protection_is_scoped_to_delegate_and_monitor_everywhere():
    sections = system_prompt()
    scope, action, modes = (
        sections["TASK SCOPE AND DECOMPOSITION"], sections["ACTION"], sections["CONTROL MODES"],
    )
    assert "The protection binds only delegate and monitor steps" in scope
    assert "They bind delegate and monitor steps only" in action
    assert "bind delegate and monitor steps only" in modes
    assert "The stall timing protection does not apply to a transition" in sections["TRANSITIONS"]
    assert "The stall timing protection does not bind a takeover" in sections["TAKEOVER"]
    # The 1.0.9 guard sentences themselves are unchanged.
    assert "If uncertain, run one more configured default policy window" in action
    assert "These same stall and timing criteria apply to adjust_eef, move_base and adjust_torso" in action
    skill = " ".join(_skill().split())
    assert "none of this timing protection binds takeover or transition steps" in skill


def test_role_and_action_state_the_division_of_motion_authority():
    sections = system_prompt()
    role, action = sections["ROLE"], sections["ACTION"]
    assert "in takeover and transition steps you drive the robot" in role
    assert "Privilege means more motion authority, never more information" in role
    assert ("Transition steps, and takeover steps except the contact phase of a staged takeover, "
            "run no policy tool and send no instruction") in action
    assert "use the sentence already frozen for that operation" in action
    assert "Both are small corrections inside delegate and monitor steps" in action
    assert "Every delegate or monitor grasp plan step must allow smolvla_pick" in " ".join(action.split())
    # Unchanged 1.0.9 authority rules.
    assert "never use servo_eef to approach a target the policy has not reached" in action
    assert "Never use simulator internals, hidden joints, fixture coordinates" in action


def test_transitions_are_long_task_only_and_split_cross_fixture_pick_place():
    text = system_prompt()["TRANSITIONS"]
    assert "used only in long tasks" in text
    assert "a short task never contains one" in text
    assert "more than about 0.8 m away in base x-y by fresh back_project (provisional threshold)" in text
    assert "treat all earlier base-relative points as stale" in text
    assert "lift the object to a carry height clear of every surface before the base moves" in text
    assert "the base should stop facing the fixture" in text
    assert "pick (delegate or monitor, original sentence), carry (transition) and place" in text
    assert "its carry belongs to the takeover step and no separate transition is planned" in text
    assert "allowed_tools: navigate_to, adjust_torso_to and move_eef_path only" in text
    assert "handoff pose reached" in text and "object still held" in text
    assert "zoom" in text and "back_project" in text
    assert "intervene=true and its frozen sentence" in text
    assert "continue_vla with intervene=false when the next operation shares" in text
    # 1.0.11: a quota-ended transition folds into the next operation (B4) and
    # hands over at the training start pose (B7) after holonomic travel (C2).
    assert "When the harness ends the transition segment" in text
    assert "replan the carry into the next operation as monitor" in text
    assert "Set travel_m to the planar base travel" in text
    assert "training start pose of the next operation's matched atomic row" in text
    assert "first back straight out the way you came" in text
    assert "base_stall (blocked phase and direction)" in text
    scope = system_prompt()["TASK SCOPE AND DECOMPOSITION"]
    assert "pick-and-place whose destination is beyond arm reach becomes pick, carry and place" in scope
    assert "within reach it stays one operation even between different fixtures" in scope


def test_takeover_covers_weak_and_none_by_operation_type_with_existing_reviews():
    text = system_prompt()["TAKEOVER"]
    assert "whose tier is weak or none, or one listed under control_mode_escalations, in short and long tasks alike" in text
    assert "you drive the whole operation, including contact" in text
    assert "one to eight waypoints" in text
    assert "A direct close must first pass review_grasp" in text
    assert "retreat to a clear pose before verifying expected_evidence" in text
    for op_type in ("a grasp", "a placement", "an articulation", "a pour or tilt"):
        assert op_type in text
    assert "Never use a task- or seed-specific script, simulator internals" in text
    assert "max_attempts counts every privileged call" in text
    assert "the segment quota in CONTROL MODES bounds the takeover" in text
    assert "Then replan the subtask as monitor" in " ".join(text.split())
    assert "in a takeover it opens its own review" in text
    assert "Keeping a closed gripper, as when an object is already held, needs no review" in " ".join(text.split())
    assert "Split takeover:" in text and "skill_row none" in text
    assert "episode liveness budget" in text
    assert "approaches about 10 cm above the grasp point" in text
    assert "If the policy skill marks an operation type as staged takeover" in text
    assert "no row is marked that way until the fork experiments decide" in text


def test_fallback_is_quota_based_bidirectional_and_counted_across_replans():
    modes = " ".join(system_prompt()["CONTROL MODES"].split())
    # 1.0.11 A5: stable operation and subtask names carry the counts.
    assert "names its operation, a short stable name for the complete operation" in modes
    assert "subtask names that part (pick, carry, place, grasp, pour)" in modes
    # A4: only physically executed failures count.
    assert "Only a verified fail or retry of a call that executed actions is a failure" in modes
    assert "a retry with failure_type timeout (your own allocation ran out) only extends the no-progress window" in modes
    # B2/B3: progress-gated escalation; the numbers mirror control_modes.
    assert f"{vocab.ESCALATION_TYPICAL_FACTOR:g} times its typical steps" in modes
    assert f"a further {vocab.NO_PROGRESS_WINDOW_STEPS} steps pass without a progress signal" in modes
    assert "has no step threshold and escalates once" in modes
    assert "Two counted physical VLA failures permit, but never require, bounded LLM takeover" in modes
    assert f"escalation-available after {vocab.FALLBACK_FAILURES} verified physical failure" in modes
    assert "end the attempt with a fail verdict that cites the escalation" in modes
    assert "across replanning" in modes
    # B4/B5: quotas replace the two-failure fallback, which is not permanent.
    assert f"{vocab.TRANSITION_BASE_STEPS} steps plus {vocab.TRANSITION_STEPS_PER_M} per metre of travel_m" in modes
    assert f"or {vocab.TAKEOVER_NONE_STEPS} steps without them" in modes
    assert f"and {vocab.SEGMENT_CALL_QUOTA} physical calls" in modes
    assert f"when {vocab.SEGMENT_NO_PROGRESS_STEPS} of its steps pass without a progress signal" in modes
    assert "A fallback is not permanent" in modes
    assert "until three LLM-driven segments of that subtask have ended in a fallback" in modes
    assert "segments that pass do not count" in modes
    assert vocab.MAX_LLM_FALLBACKS == 3
    assert f"one call per {vocab.SEGMENT_STEPS_PER_CALL} quota steps" in modes
    assert "never escalates again" not in modes
    skill = " ".join(_skill().split())
    assert "A takeover runs on its step and call quota, not on a failure count" in skill
    assert "1.5× its typical steps and then 120 more steps without progress" in skill


def test_ablating_a_section_drops_only_that_key_and_the_text_declares_the_fallback():
    modes = system_prompt()["CONTROL MODES"]
    assert "When this prompt has no TRANSITIONS section, transition is unavailable" in modes
    assert "when it has no TAKEOVER section, weak and none operations use monitor" in modes
    without_transitions = system_prompt(ablate={"TRANSITIONS"})
    assert "TRANSITIONS" not in without_transitions
    assert without_transitions["TAKEOVER"] == prompts.TAKEOVER_GUIDANCE
    assert without_transitions["CONTROL MODES"] == modes
    full = system_prompt()
    for name in full:
        if name != "TRANSITIONS":
            assert without_transitions[name] == full[name]
    assert set(system_prompt(ablate=set(NEW_SECTIONS))) == set(full) - set(NEW_SECTIONS)
    with pytest.raises(ValueError, match="can be ablated"):
        system_prompt(ablate={"ACTION"})
    skill = _skill()
    assert "(monitor when the TAKEOVER section is absent)" in skill
    stripped = skills.strip_sections(skill, ["Takeover"])
    assert "## Takeover" not in stripped and "## Transitions" in stripped


def test_handback_variant_exists_for_experiments_but_not_in_the_default_prompt():
    variant = prompts.TAKEOVER_HANDBACK_VARIANT
    assert "Position-then-handback variant" in variant
    assert "intervene=true" in variant and "control_mode=takeover" in variant
    rendered = _executor()
    assert variant not in rendered and "Position-then-handback" not in rendered
    skill = " ".join(_skill().split())
    assert "No weak or none operation type is marked \"staged takeover\" yet" in skill
    # 1.0.11 H4: escalated takeovers of measured rows are staged.
    # 1.0.11 H5: restore the policy's start pose before the monitor fallback.


def test_instruction_rules_of_1_0_9_survive_with_the_no_instruction_rule():
    skill = " ".join(_skill().split())
    assert "quote the corresponding original task sentence verbatim" in skill
    assert "Retries are physical actions, never paraphrases" in skill
    assert ("In transition steps, and in takeover steps except the contact phase of a staged takeover, "
            "the policy does not run and receives no instruction") in skill
    assert "use the sentence already frozen for that operation" in skill
    action = system_prompt()["ACTION"]
    assert "intervene=true when entering a new registered long-task operation" in action
    assert "within the same registered operation" in action
    assert "Do not reword the language goal" in action


def test_every_tool_name_in_the_prompts_exists_in_a_tool_schema():
    """Every underscore identifier is a known tool, a privileged tool pending block B,
    or a listed non-tool identifier; anything else is a probable typo."""
    text = "\n".join([
        *system_prompt().values(),
        _skill(),
        task_rules("KettleBoiling", policy_family="smolvla"),
        task_rules("KettleBoiling", policy_family="xiaomi-mibot"),
        resolve_planning_stack("robocasa-horizon-gated-v1").prompt_context(),
    ])
    identifiers = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text))
    known_tools = _toolkit_tool_names() | _SESSION_TOOLS
    pending = set(vocab.PRIVILEGED_TOOLS) | set(vocab.READ_ONLY_GROUNDING_TOOLS)
    unexplained = identifiers - known_tools - pending - _NON_TOOL_IDENTIFIERS
    assert not unexplained, sorted(unexplained)
    for tool in vocab.PRIVILEGED_TOOLS:
        assert tool in identifiers
    for tool in ("move_base", "adjust_torso", "adjust_eef", "servo_eef", "review_grasp"):
        assert tool in identifiers and tool in known_tools


def test_privileged_tools_are_registered_physical_tools():
    names = _toolkit_tool_names()
    assert set(vocab.PRIVILEGED_TOOLS) <= names
    assert "zoom" in names


def test_step_metadata_survives_composite_candidate_validation():
    # A composite task authorizes the carry step; generated structured fields survive.
    from copy import deepcopy

    from rpent.runtime import RuntimePlanGenerator, RuntimeResult

    steps = [
        {"id": "pick", "intent": 'VLA: "Then put the bowl in the fridge."; row=none; tier=none; control_mode=monitor',
         "allowed_tools": ["smolvla_pick", "adjust_eef", "resume_vla"],
         "expected_evidence": ["Bowl lifted and retained."], "max_attempts": 6,
         "skill_row": "none", "tier": "none", "control_mode": "monitor"},
        {"id": "carry", "intent": "Carry the bowl to the fridge; row=none; tier=none; control_mode=transition",
         "allowed_tools": ["navigate_to", "adjust_torso_to", "move_eef_path"],
         "expected_evidence": ["Handoff pose reached; fridge visible; bowl still held."],
         "max_attempts": 4, "skill_row": "none", "tier": "none", "control_mode": "transition"},
    ]

    class Runtime:
        def solve(self, **kwargs):
            return RuntimeResult(finish_result={
                "strategy": "Scope: long. carry: transition failures=0.",
                "steps": deepcopy(steps),
            })

    generator = RuntimePlanGenerator(Runtime(), require_outcome_program=False)
    generator.configure_control_modes(policy_family="xiaomi-mibot", task_name="StoreLeftoversInBowl")
    plan = generator.generate(
        goal="Then put the bowl in the fridge.", observations=[], constraints=[],
        physical_tools={"smolvla_pick", "adjust_eef", "resume_vla",
                        "navigate_to", "adjust_torso_to", "move_eef_path"},
    )
    assert [step["id"] for step in plan["steps"]] == ["pick", "carry"]
    assert "control_mode=transition" in plan["steps"][1]["intent"]
    assert plan["steps"][1]["allowed_tools"] == ["navigate_to", "adjust_torso_to", "move_eef_path"]
    assert plan["steps"][1]["control_mode"] == "transition"
    assert plan["steps"][1]["skill_row"] == "none"
    assert plan["steps"][1]["tier"] == "none"


def test_skill_rows_quote_official_instructions_and_rule_matches_on_them():
    # D06 replay (2026-10-09): "Sink → counter/plate" made the planner treat a
    # bowl on the counter as unmatched and take over a reliable operation.
    import json
    from pathlib import Path

    tasks = json.loads((Path(vocab.__file__).parent / "task_groups.json").read_text())["tasks"]
    rows = re.findall(r"^\| ([A-Za-z0-9]+): (.*?) \| (?:Reliable|Unstable|Weak)", _skill(), re.M)
    assert len(rows) == 65
    for name, label in rows:
        assert label == f'"{tasks[name]["instruction"]}"', name
    modes = system_prompt()["CONTROL MODES"]
    assert "exactly to one row's official task instruction" in modes
    assert "'the container located on the counter' includes a bowl on the counter" in modes


def test_split_pick_and_place_steps_share_the_whole_operation_row():
    # Replay 2026-10-09: an entry read as "placement only" left the counter pick
    # with no row, so the planner took over the whole bowl-to-fridge transfer.
    text = system_prompt()["TRANSITIONS"]
    assert ("the pick and place steps both carry the whole operation's skill_row, tier "
            "and control_mode") in text
