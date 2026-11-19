"""1.0.11 control accounting: stable identities, physical failures, progress
windows, LLM quotas and a non-permanent fallback (backlog A4, A5, B1-B5, D3)."""

import json
import math

import pytest

from robots.robocasa import control_modes as vocab
from robots.robocasa.capabilities import (
    PRIVILEGED_PHYSICAL_TOOLS,
    capability_summary,
    load_capabilities,
)
from rpent.harness.controller import HarnessController
from rpent.harness.state import (
    PlanStateMachine,
    operation_key,
    parse_steps,
    subtask_key,
)
from rpent.planning.schemas import ActionFeedback

pytestmark = pytest.mark.usefixtures("synthetic_control_catalog")

TOOLS = {"move", "smolvla_pick", "continue_vla", "execute_vla", "resume_vla", *PRIVILEGED_PHYSICAL_TOOLS}
NONE_TAKEOVER = {"skill_row": "none", "tier": "none", "control_mode": "takeover",
                 "allowed_tools": ["move_eef_path", "navigate_to"]}
SINK = {"skill_row": "PickPlaceCounterToSink", "tier": "reliable", "control_mode": "delegate"}  # typical 316
FRIDGE = {"skill_row": "PickPlaceCounterToOpenFridge", "tier": "reliable", "control_mode": "monitor"}


def step(**overrides):
    return {"id": "op", "intent": "Place it. row=x", "allowed_tools": ["move"], "max_attempts": 30,
            "operation": "dump_vegetables", **overrides}


def machine(events=None, *, horizon=None, task="PanTransfer", **kwargs):
    state = PlanStateMachine(
        max_replans=40, physical_tools=TOOLS, policy_family="synthetic", task_name=task,
        record=(lambda event, **payload: events.append({"event": event, **payload})) if events is not None else None,
        **kwargs,
    )
    state.set_episode_horizon(horizon)
    return state


def act(state, executed, *, verdict="progress", tool="move", execution_id=None, failure_type=None,
        error=None, metrics=None, raw=None, step_id=None):
    execution_id = execution_id or f"e{len(getattr(state, '_seen', []))}-{executed}-{verdict}-{id(raw)}"
    state._seen = [*getattr(state, "_seen", []), execution_id]
    result = {"executed_actions": executed, **(raw or {})}
    if step_id is not None:
        result["environment"] = {"step_id": step_id}
    state.record_action_feedback(
        ActionFeedback(execution_id=execution_id, tool_name=tool, physical=True,
                       execution_status="controller_error" if error else "completed",
                       error=error, raw_result=result),
        input_dict={},
    )
    return state.verify_step(verdict, ["observed"], "observed", execution_id=execution_id,
                             failure_type=failure_type, metrics=metrics)


def named(events, name):
    return [event for event in events if event["event"] == name]


# --- A5: stable operation identity --------------------------------------------------------


def test_operation_field_survives_rewritten_intent_evidence_and_ids():
    first = parse_steps([step(intent="Dump it", preconditions=["pan on stove"], expected_evidence=["a"])],
                        TOOLS, policy_family="synthetic")[0]
    second = parse_steps([step(id="new", intent="Dump the vegetables again",
                               preconditions=["pan now held"], expected_evidence=["b", "c"])],
                         TOOLS, policy_family="synthetic")[0]
    assert operation_key(first) == operation_key(second) == ("operation", "dump_vegetables")


def test_split_pick_carry_place_share_the_operation_but_not_the_subtask():
    pick, carry, place = parse_steps([
        step(id="p", **FRIDGE, subtask="pick"),
        step(id="c", skill_row="none", tier="none", control_mode="transition",
             allowed_tools=["navigate_to"], travel_m=2.5),
        step(id="l", **FRIDGE, subtask="place"),
    ], TOOLS, policy_family="synthetic", task_name="StoreLeftoversInBowl")
    assert operation_key(pick) == operation_key(carry) == operation_key(place)
    assert [subtask_key(s)[1] for s in (pick, carry, place)] == ["pick", "carry", "place"]
    assert carry.travel_m == 2.5 and pick.travel_m is None


def test_legacy_identity_is_row_plus_quoted_instruction_not_rewritten_grounding():
    base = {"skill_row": "none", "tier": "none", "control_mode": "monitor", "operation": None}
    old = parse_steps([step(**base, intent="Use 'Pick up the pan and dump the vegetables in it onto the plate.'",
                            preconditions=["pan on stove"], expected_evidence=["pan lifted"])], TOOLS)[0]
    new = parse_steps([step(**base, id="later", intent="Retry: 'Pick up the pan and dump the vegetables in it onto the plate.'",
                            preconditions=["pan held above plate"], expected_evidence=["vegetables on plate"])], TOOLS)[0]
    other = parse_steps([step(**base, intent="Use 'Then return the pan to the stove.'")], TOOLS)[0]
    assert operation_key(old) == operation_key(new) != operation_key(other)


def test_identity_fields_are_validated():
    with pytest.raises(ValueError, match="operation"):
        parse_steps([step(operation=7)], TOOLS)
    with pytest.raises(ValueError, match="travel_m"):
        parse_steps([step(skill_row="none", tier="none", control_mode="transition", travel_m=-1,
                          allowed_tools=["navigate_to"])], TOOLS, task_name="StoreLeftoversInBowl")
    parsed = parse_steps([step(travel_m=2.0)], TOOLS)[0]
    assert parsed.travel_m is None  # only a transition's carry distance is kept


# --- A4: only physically executed failures count -----------------------------------------


@pytest.mark.parametrize("kind", ["validation_error", "guard_pause", "rejected"])
def test_non_physical_failures_never_count(kind):
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    for index in range(4):
        if kind == "validation_error":
            act(state, 0, verdict="fail", error="each path segment must be <=0.30 metres")
        elif kind == "guard_pause":
            act(state, 0, verdict="fail", raw={"stop_reason": "preclose_review_required"})
        else:
            state.record_action_feedback(
                ActionFeedback(execution_id=f"r{index}", tool_name="move", physical=True,
                               execution_status="rejected", error="harness blocked"),
                input_dict={})
            state.verify_step("fail", ["blocked"], "blocked", execution_id=f"r{index}")
        assert state.replan("retry", [step(**NONE_TAKEOVER)])["ok"]
    context = state.control_planning_context(include_capabilities=False)
    assert context["control_mode_failures"] == [] and context["control_mode_fallbacks"] == []
    assert all(e["physical_failure"] is False for e in named(events, "step_verified"))


def test_physical_failures_count_per_subtask_and_mode():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, 40, verdict="fail")
    [verified] = named(events, "step_verified")
    assert verified["physical_failure"] is True and verified["subtask"] == "whole"
    [failure] = state.control_planning_context(include_capabilities=False)["control_mode_failures"]
    assert failure == {"operation_key": ["operation", "dump_vegetables"], "subtask": "whole",
                       "failed_verifications": {"takeover": 1}}


# --- B4: LLM segments run on quotas, not on two failures ----------------------------------


def test_takeover_survives_failures_inside_its_quota():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    for _ in range(3):
        act(state, 30, verdict="fail")
        assert state.replan("change approach", [step(**NONE_TAKEOVER)])["ok"]
    assert not named(events, "control_mode_fallback")
    [segment] = state.control_planning_context(include_capabilities=False)["control_segments"]
    assert segment["step_quota"] == vocab.TAKEOVER_NONE_STEPS and segment["steps_used"] == 90
    assert segment["calls_used"] == 3 and segment["call_quota"] == vocab.SEGMENT_CALL_QUOTA


def test_step_quota_ends_the_segment_at_its_next_verdict():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, vocab.TAKEOVER_NONE_STEPS - 10)
    assert not named(events, "control_mode_fallback")
    act(state, 20)  # 10 over the quota, verdict progress still falls back
    [fallback] = named(events, "control_mode_fallback")
    assert fallback["reason"] == "step_quota" and fallback["to_mode"] == "monitor"
    assert state.needs_replan
    assert "monitor" in state.replan("again", [step(**NONE_TAKEOVER)])["error"]
    assert state.replan("monitor", [step(skill_row="none", tier="none", control_mode="monitor")])["ok"]


def test_call_quota_and_pass_inside_quota():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER), step(id="next", operation="return_pan",
                                                                      **NONE_TAKEOVER)])["ok"]
    for _ in range(vocab.SEGMENT_CALL_QUOTA - 1):
        act(state, 1)
    act(state, 1, verdict="pass")  # quota reached on a passing call: no fallback
    assert not named(events, "control_mode_fallback")
    [closed] = named(events, "control_segment_closed")
    assert closed["reason"] == "passed" and closed["quota_exhausted"] == "call_quota"


def test_no_progress_window_and_destructive_failure_end_the_segment_early():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, 100, verdict="retry", failure_type="grasp_missed")
    act(state, 60, verdict="retry", failure_type="grasp_missed")  # 160 steps without progress
    assert named(events, "control_mode_fallback")[0]["reason"] == "no_progress"

    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, 10, verdict="fail", failure_type="object_dropped")
    assert named(events, "control_mode_fallback")[0]["reason"] == "destructive_failure"


def test_hold_and_shrinking_target_distance_are_progress():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    held = {"grasp_status": {"retention_monitor_active": True}}
    act(state, 100, verdict="retry", failure_type="timeout", raw=held)
    act(state, 100, verdict="retry", failure_type="timeout", metrics={"target_distance_m": 0.50})
    act(state, 40, verdict="retry", failure_type="timeout", metrics={"target_distance_m": 0.40})
    [segment] = state.control_planning_context(include_capabilities=False)["control_segments"]
    assert segment["steps_since_progress"] == 0
    assert not named(events, "control_mode_fallback")


def test_transition_quota_uses_travel_and_horizon_keeps_the_fallback_reserve():
    events = []
    state = machine(events, horizon=2550, task="StoreLeftoversInBowl")
    carry = step(id="carry", operation="bowl_to_fridge", skill_row="none", tier="none",
                 control_mode="transition", allowed_tools=["navigate_to"], travel_m=2.0)
    place = step(id="place", operation="bowl_to_fridge", subtask="place", **FRIDGE)
    assert state.register_plan("goal", [carry, place])["ok"]
    act(state, 10, step_id=1200)
    [opened] = named(events, "control_segment_opened")
    assert opened["step_quota"] == 150 + 200 * 2 and opened["quota_basis"]["travel_declared"]

    events = []
    state = machine(events, horizon=2550, task="StoreLeftoversInBowl")
    assert state.register_plan("goal", [carry, place])["ok"]
    act(state, 10, step_id=2000)  # 560 left before the call, 346.5 reserved for the place
    assert named(events, "control_segment_opened")[0]["step_quota"] == 560 - 347

    events = []
    state = machine(events, horizon=2550, task="StoreLeftoversInBowl")
    assert state.register_plan("goal", [carry, place])["ok"]
    act(state, 10, step_id=2500)
    assert named(events, "control_segment_opened")[0]["step_quota"] == vocab.MIN_SEGMENT_STEPS


def test_takeover_quota_uses_typical_steps_of_its_row():
    events = []
    state = machine(events)
    weak = step(skill_row="OpenMicrowave", tier="weak", control_mode="takeover", allowed_tools=["move_eef_path"])
    assert state.register_plan("goal", [weak])["ok"]
    act(state, 5)
    assert named(events, "control_segment_opened")[0]["step_quota"] == 639  # ceil(1.5 x 426)


def test_transition_fallback_folds_the_carry_but_allows_the_place_monitor():
    events = []
    state = machine(events, task="StoreLeftoversInBowl")
    carry = step(id="carry", operation="bowl_to_fridge", skill_row="none", tier="none",
                 control_mode="transition", allowed_tools=["navigate_to"], travel_m=0.5)
    place = step(id="place", operation="bowl_to_fridge", subtask="place", **FRIDGE)
    assert state.register_plan("goal", [carry, place])["ok"]
    act(state, 260)  # quota 250
    assert named(events, "control_mode_fallback")[0]["from_mode"] == "transition"
    assert "fold the carry" in state.replan("again", [carry, place])["error"]
    assert state.replan("fold", [place])["ok"]


def test_takeover_fallback_does_not_reject_the_same_operations_transition():
    events = []
    state = machine(events, task="StoreLeftoversInBowl")
    pick = step(id="pick", operation="pan_dump", subtask="grasp", **{**NONE_TAKEOVER})
    assert state.register_plan("goal", [pick])["ok"]
    act(state, vocab.TAKEOVER_NONE_STEPS + 50)
    assert named(events, "control_mode_fallback")
    carry = step(id="carry", operation="pan_dump", skill_row="none", tier="none",
                 control_mode="transition", allowed_tools=["navigate_to"])
    monitor = step(id="pick", operation="pan_dump", subtask="grasp", skill_row="none", tier="none",
                   control_mode="monitor")
    assert state.replan("monitor and carry", [monitor, carry])["ok"]


# --- B5: a fallback is not permanent ------------------------------------------------------


def test_policy_failure_after_fallback_reopens_takeover_up_to_the_fallback_cap():
    events = []
    state = machine(events)
    takeover = step(**NONE_TAKEOVER)
    monitor = step(skill_row="none", tier="none", control_mode="monitor")
    assert state.register_plan("goal", [takeover])["ok"]
    for segment in range(vocab.MAX_LLM_FALLBACKS):
        act(state, vocab.TAKEOVER_NONE_STEPS + 10)  # quota exhausted -> fallback
        assert state.replan("fallback", [monitor])["ok"]
        act(state, 50, verdict="retry", failure_type="object_dropped")  # explicit public evidence: policy dropped the pan
        if segment < vocab.MAX_LLM_FALLBACKS - 1:
            assert named(events, "control_mode_fallback_cleared")[-1]["reason"] == "destructive_failure"
            assert state.replan("take over again", [takeover])["ok"]
    assert len(named(events, "control_segment_opened")) == vocab.MAX_LLM_FALLBACKS
    assert "ended in fallback" in state.replan("again", [takeover])["error"]


def test_restore_pose_transition_is_legal_after_a_takeover_fallback():
    # 1.0.11 H5: before the monitor fallback, return arm and torso to the
    # policy's start pose in a short transition of the same operation.
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, vocab.TAKEOVER_NONE_STEPS + 10)  # quota exhausted -> fallback
    assert named(events, "control_mode_fallback")
    restore = step(id="restore", subtask="restore", skill_row="none", tier="none", control_mode="transition",
                   allowed_tools=["move_eef_path", "adjust_torso_to"], travel_m=0.0)
    monitor = step(id="resume", skill_row="none", tier="none", control_mode="monitor")
    assert state.replan("restore the start pose, then hand back", [restore, monitor])["ok"]
    from robots.robocasa.prompts.system import system_prompt

    takeover_rules = " ".join(system_prompt()["TAKEOVER"].split())
    assert "subtask restore" in takeover_rules and "handoff table's arm start position" in takeover_rules


def test_passing_carry_legs_do_not_consume_the_fallback_cap():
    # 1.0.11 canary D07: three short passing transition legs of one carry made
    # the fourth leg illegal ("used all 3 LLM-driven segments").
    events = []
    state = machine(events, task="StoreLeftoversInBowl")
    legs = [step(id=f"leg{i}", operation="counter_to_fridge_access", subtask="carry", skill_row="none",
                 tier="none", control_mode="transition", allowed_tools=["navigate_to", "move_eef_path"],
                 travel_m=0.0) for i in range(5)]
    assert state.register_plan("goal", legs)["ok"]
    for _ in legs:
        act(state, 30, verdict="pass")
    assert len(named(events, "control_segment_opened")) == 5
    assert not named(events, "control_mode_fallback")
    assert state.control_constraints()["llm_exhausted"] == set()
    assert state.parse_steps([{**legs[0], "id": "leg5"}])


def test_declared_row_on_a_transition_is_not_held_to_its_mode_limit():
    carry = step(operation="bowl_to_fridge", subtask="carry", control_mode="transition",
                 skill_row="PickPlaceCounterToOpenFridge", tier="reliable", allowed_tools=["navigate_to"])
    [parsed] = parse_steps([carry], TOOLS, policy_family="synthetic", task_name="StoreLeftoversInBowl")
    assert parsed.control_mode == "transition"
    with pytest.raises(ValueError, match="allows control_mode"):
        parse_steps([{**carry, "control_mode": "delegate", "allowed_tools": ["move"]}], TOOLS, policy_family="synthetic",
                    task_name="StoreLeftoversInBowl")


def test_call_quota_grows_with_long_step_quotas():
    events = []
    state = machine(events)
    weak = step(skill_row="PickPlaceCounterToMicrowave", tier="weak", control_mode="takeover",
                allowed_tools=["move_eef_path"])  # typical 741.5 -> 1113 quota steps
    assert state.register_plan("goal", [weak])["ok"]
    act(state, 5)
    [opened] = named(events, "control_segment_opened")
    assert opened["step_quota"] == 1113
    assert opened["call_quota"] == math.ceil(1113 / vocab.SEGMENT_STEPS_PER_CALL) == 28


# --- B1-B3: per-subtask counting and progress-gated escalation ----------------------------


def test_exact_row_escalates_only_after_threshold_and_a_no_progress_window():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**SINK)])["ok"]
    takeover = step(skill_row="PickPlaceCounterToSink", tier="reliable", control_mode="takeover",
                    allowed_tools=["move_eef_path", "continue_vla"])  # H4: staged
    for _ in range(10):  # 500 steps > 1.5 x 316 while still progressing
        act(state, 50)
    assert not named(events, "control_mode_escalation_available")
    act(state, 50, verdict="retry", failure_type="timeout")
    act(state, 50, verdict="retry", failure_type="timeout")  # 100 < 120 without progress
    assert not named(events, "control_mode_escalation_available")
    act(state, 30, verdict="retry", failure_type="timeout")
    [escalation] = named(events, "control_mode_escalation_available")
    assert escalation["reason"] == "steps_no_progress" and escalation["subtask"] == "whole"
    assert escalation["steps"] == 630 and escalation["step_escalation"] is True
    assert state.replan("escalate", [takeover])["ok"]


def test_physical_policy_failure_escalates_after_first_executed_failure():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**SINK)])["ok"]
    act(state, 0, verdict="fail", error="bad argument")  # not physical
    assert not named(events, "control_mode_escalation_available")
    act(state, 20, verdict="retry", failure_type="grasp_missed")
    assert not named(events, "control_mode_escalation_available")
    act(state, 20, verdict="retry", failure_type="grasp_missed")
    [escalation] = named(events, "control_mode_escalation_available")
    assert escalation["reason"] == "failures" and escalation["failed_verifications"] == 2


def test_split_subtasks_count_separately_and_escalate_by_window_only():
    events = []
    state = machine(events, task="StoreLeftoversInBowl")
    pick = step(id="pick", operation="bowl_to_fridge", subtask="pick", **FRIDGE)
    place = step(id="place", operation="bowl_to_fridge", subtask="place", **FRIDGE)
    assert state.register_plan("goal", [pick, place])["ok"]
    act(state, 180, verdict="pass")
    for _ in range(19):  # 950 policy steps of a slow but progressing carry
        act(state, 50)
    assert not named(events, "control_mode_escalation_available")
    subtasks = {row["subtask"]: row for row in state.control_planning_context()["control_subtasks"]}
    assert "control_subtasks" not in state.snapshot()
    assert subtasks["pick"]["policy_steps"] == 180 and subtasks["place"]["policy_steps"] == 950
    act(state, 60, verdict="retry", failure_type="timeout")
    act(state, 60, verdict="retry", failure_type="timeout")
    [escalation] = named(events, "control_mode_escalation_available")
    assert escalation["reason"] == "no_progress" and escalation["subtask"] == "place"
    takeover = step(id="place", operation="bowl_to_fridge", subtask="place", skill_row="PickPlaceCounterToOpenFridge",
                    tier="reliable", control_mode="takeover", allowed_tools=["move_eef_path"])
    # 1.0.11 H4: an escalated takeover is staged, so it must list the policy tool.
    with pytest.raises(ValueError, match="escalated takeover is staged"):
        state.parse_steps([takeover])
    takeover["allowed_tools"] = ["move_eef_path", "continue_vla"]
    assert state.replan("take over the placement", [takeover])["ok"]
    with pytest.raises(ValueError, match="escalation-available"):
        state.parse_steps([{**takeover, "subtask": "pick"}])


def test_escalation_reopens_a_folded_carry_for_rehandoff():
    events = []
    state = machine(events, task="StoreLeftoversInBowl")
    carry = step(id="carry", operation="bowl_to_fridge", skill_row="none", tier="none",
                 control_mode="transition", allowed_tools=["navigate_to"], travel_m=0.5)
    place = step(id="place", operation="bowl_to_fridge", subtask="place", **FRIDGE)
    assert state.register_plan("goal", [carry, place])["ok"]
    act(state, 260)
    assert state.replan("fold", [place])["ok"]
    act(state, 130, verdict="retry", failure_type="timeout")  # handed-off policy makes no progress
    assert named(events, "control_mode_fallback_cleared")[0]["subtask"] == "carry"
    assert state.replan("take back and re-hand", [carry, place])["ok"]


def test_ablation_disables_escalation_but_progress_rules_still_run():
    events = []
    state = machine(events, op_takeover_enabled=False)
    assert state.register_plan("goal", [step(**SINK)])["ok"]
    act(state, 600, verdict="retry", failure_type="timeout")
    act(state, 20, verdict="retry", failure_type="timeout")
    assert not named(events, "control_mode_escalation_available")


# --- Controller plumbing and D3 contract filter --------------------------------------------


def test_contract_carries_atomic_declared_and_only_the_current_composite_row():
    catalog = load_capabilities("synthetic")
    composites = {name for name, row in catalog["rows"].items() if row["family"] == "composite"}
    rows = {row["skill_row"]: row for row in capability_summary(catalog, task_name="PanTransfer")}
    assert set(rows) & composites == {"PanTransfer"} and rows["PanTransfer"]["family"] == "composite"
    assert {"PickPlaceCounterToOpenFridge", "PlateToBowlSameCounter"} <= set(rows)
    atomic = {name for name, row in catalog["rows"].items() if row["family"] == "atomic"}
    assert atomic <= set(rows)
    assert len(json.dumps(capability_summary(catalog, task_name="PanTransfer"))) < 0.6 * len(
        json.dumps([{"skill_row": name} | row for name, row in catalog["rows"].items()]))


def test_controller_contract_and_generator_see_segments_and_constraints(tmp_path):
    class Generator:
        def configure_control_modes(self, **kwargs):
            self.config = kwargs

    generator = Generator()
    controller = HarnessController(mode="plan-gated", plan_generator=generator,
                                   trace_path=tmp_path / "trace.jsonl", physical_tools=TOOLS)
    controller.configure_control_modes(policy_family="synthetic", task_name="PanTransfer")
    assert controller.state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(controller.state, vocab.TAKEOVER_NONE_STEPS + 10)
    constraints = controller.planning_constraints([])
    contract = json.loads(constraints[-1].split(": ", 1)[1])
    assert contract["control_mode_fallbacks"][0]["reason"] == "step_quota"
    assert "PanTransfer" in {row["skill_row"] for row in contract["capabilities"]}
    key = (("operation", "dump_vegetables"), "whole")
    assert generator.config["control_constraints"]["fallbacks"] == {key: "takeover"}


def test_snapshot_step_payload_keeps_identity_fields():
    from rpent.harness.controller import _step_payload

    state = machine()
    assert state.register_plan("goal", [step(subtask="grasp", skill_row="none", tier="none")])["ok"]
    payload = _step_payload(state.active_step)
    assert payload["operation"] == "dump_vegetables" and payload["subtask"] == "grasp"
    assert "travel_m" not in payload


def test_policy_fallback_starts_a_fresh_no_progress_window():
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**NONE_TAKEOVER)])["ok"]
    act(state, 160, verdict="retry", failure_type="timeout")
    assert named(events, "control_mode_fallback")[0]["reason"] == "no_progress"
    assert state.replan("monitor", [step(skill_row="none", tier="none", control_mode="monitor")])["ok"]
    act(state, 50, verdict="retry", failure_type="timeout")
    assert not named(events, "control_mode_fallback_cleared")
    act(state, 80, verdict="retry", failure_type="timeout")  # 130 >= 120 of its own
    assert named(events, "control_mode_fallback_cleared")[0]["reason"] == "no_progress"


def test_environment_success_survives_a_later_refused_call():
    # Review of 1.0.11: after native success the toolkit refuses further motion;
    # that refusal must not erase the observed success.
    state = machine()
    assert state.register_plan("goal", [step(**SINK, max_attempts=20)])["ok"]
    state.record_action_feedback(
        ActionFeedback(execution_id="win", tool_name="move", physical=True, execution_status="completed",
                       environment_success=True, raw_result={"executed_actions": 5}),
        input_dict={},
    )
    assert state.environment_success
    state.verify_step("pass", ["observed"], "observed", execution_id="win")
    act(state, 0, error="cannot execute after the episode ended", verdict="fail", failure_type="controller_error")
    assert state.environment_success


def test_only_tool_layer_value_errors_are_refusals(tmp_path):
    from rpent.tools.toolkit import _refusal

    def raise_from(filename, exc):
        code = compile(f"def f():\n    raise {exc}('x')\n", str(tmp_path / filename), "exec")
        scope = {}
        exec(code, scope)
        try:
            scope["f"]()
        except Exception as caught:  # noqa: BLE001
            return caught

    assert _refusal(raise_from("toolkit.py", "ValueError"))
    assert _refusal(raise_from("privileged_tools.py", "ValueError"))
    assert not _refusal(raise_from("pi05_binding.py", "ValueError"))
    assert not _refusal(raise_from("toolkit.py", "RuntimeError"))


def test_policy_progress_after_a_fallback_does_not_reopen_takeover():
    # Review of 1.0.11: the two failures that escalated stayed counted, so any
    # later policy verdict cleared the fallback and escalated again.
    events = []
    state = machine(events)
    assert state.register_plan("goal", [step(**SINK, max_attempts=40)])["ok"]
    act(state, 20, verdict="retry", failure_type="grasp_missed")
    assert not named(events, "control_mode_escalation_available")
    act(state, 20, verdict="fail", failure_type="grasp_missed")
    assert named(events, "control_mode_escalation_available")
    takeover = step(**{**SINK, "control_mode": "takeover"}, allowed_tools=["move_eef_path", "continue_vla"])
    assert state.replan("escalate", [takeover])["ok"]
    act(state, 480)  # quota 1.5 x 316 = 474 -> fallback
    assert named(events, "control_mode_fallback")
    assert state.replan("monitor", [step(**{**SINK, "control_mode": "monitor"})])["ok"]
    act(state, 10, verdict="progress")
    assert not named(events, "control_mode_fallback_cleared")
    act(state, 10, verdict="fail", failure_type="grasp_missed")  # explicit physical failure after fallback (B5)
    assert named(events, "control_mode_fallback_cleared")


def test_handing_back_to_the_policy_keeps_the_takeover_quota():
    events = []
    state = machine(events)
    takeover = step(**NONE_TAKEOVER)
    assert state.register_plan("goal", [takeover])["ok"]
    act(state, vocab.TAKEOVER_NONE_STEPS - 10, verdict="retry", failure_type="predicate_not_satisfied",
        raw={"grasp_status": {"retention_monitor_active": True}})
    assert state.replan("hand back", [step(skill_row="none", tier="none", control_mode="monitor")])["ok"]
    act(state, 5, verdict="retry", failure_type="grasp_missed")
    act(state, 5, verdict="retry", failure_type="grasp_missed")
    assert state.replan("take over again", [takeover])["ok"]
    act(state, 15)  # (quota - 10) + 5 + 15 > quota: the quota is used up, not renewed
    assert named(events, "control_mode_fallback")


def test_horizon_caps_a_segment_whose_first_call_moved_nothing():
    events = []
    state = machine(events, horizon=1000)
    monitor = step(skill_row="none", tier="none", control_mode="monitor")
    assert state.register_plan("goal", [monitor])["ok"]
    act(state, 10, step_id=900, verdict="retry", failure_type="predicate_not_satisfied")
    assert state.replan("take over", [step(**NONE_TAKEOVER)])["ok"]
    act(state, 0, error="segment exceeds 0.30 m", verdict="retry", failure_type="predicate_not_satisfied")
    [opened] = named(events, "control_segment_opened")
    assert opened["step_quota"] == 50  # 100 steps left minus the 200-step reserve, floored
