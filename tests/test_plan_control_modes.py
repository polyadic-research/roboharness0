"""Control-mode contracts, legacy compatibility and operation-local fallback."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from robots.robocasa.capabilities import (
    PRIVILEGED_PHYSICAL_TOOLS,
    capability_summary,
    load_capabilities,
    validate_capabilities,
)
from rpent.harness.controller import HarnessController
from rpent.harness.gate import PlanSafetyGate
from rpent.harness.state import PlanStateMachine, parse_steps
from rpent.planning.schemas import ActionFeedback
from rpent.planning.stack import resolve_planning_stack
from rpent.runtime.plan_generation import RuntimePlanGenerator
from rpent.tools.toolkit import Toolkit

pytestmark = pytest.mark.usefixtures("synthetic_control_catalog")

TOOLS = {
    "move",
    "smolvla_pick",
    "continue_vla",
    "execute_vla",
    "resume_vla",
    *PRIVILEGED_PHYSICAL_TOOLS,
}


def step(**overrides):
    return {
        "id": "operation",
        "intent": "place the object",
        "allowed_tools": ["move"],
        "covers_outcomes": ["outcome-b", "outcome-a"],
        "max_attempts": 8,
        "skill_row": "TurnOnMicrowave",
        "tier": "weak",
        "control_mode": "monitor",
        **overrides,
    }


def parse(raw, **kwargs):
    return parse_steps([raw], TOOLS, policy_family="synthetic", **kwargs)[0]


def state(**kwargs):
    return PlanStateMachine(
        max_replans=8, physical_tools=TOOLS, policy_family="synthetic", **kwargs
    )


def fail(machine, *, verdict="fail", execution_id="physical", executed=5):
    # 1.0.11 A4: only a call that executed actions can fail physically.
    machine.record_action_feedback(
        ActionFeedback(
            execution_id=execution_id,
            tool_name="move",
            physical=True,
            execution_status="completed",
            raw_result={"executed_actions": executed},
        ),
        input_dict={},
    )
    return machine.verify_step(
        verdict, ["observed physical miss"], "failed placement", execution_id=execution_id,
        failure_type="grasp_missed"
    )


def test_three_fields_and_defaulted():
    parsed = parse(step(control_mode="takeover"))
    assert (parsed.skill_row, parsed.tier, parsed.control_mode, parsed.defaulted) == (
        "TurnOnMicrowave",
        "weak",
        "takeover",
        False,
    )
    assert asdict(parsed)["control_mode"] == "takeover"


@pytest.mark.parametrize(
    "field,value",
    [
        ("skill_row", 3),
        ("skill_row", ""),
        ("tier", "good"),
        ("control_mode", "manual"),
        ("tier", None),
    ],
)
def test_illegal_fields_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        parse(step(**{field: value}))


def test_old_plans_and_partial_defaults():
    legacy = {"id": "old", "intent": "act", "allowed_tools": ["move"]}
    parsed = parse(legacy)
    assert (parsed.skill_row, parsed.tier, parsed.control_mode, parsed.defaulted) == (
        "none",
        "none",
        "monitor",
        True,
    )
    assert parse({**legacy, "control_mode": "takeover"}).defaulted
    observed = parse({"id": "look", "intent": "look", "step_type": "observation"})
    assert observed.step_type == "observation"
    assert not observed.defaulted


def test_catalog_tier_must_match_and_unknown_row_requires_none():
    with pytest.raises(ValueError, match="tier"):
        parse(step(skill_row="PickPlaceSinkToCounter", tier="weak"))
    with pytest.raises(ValueError, match="tier"):
        parse(step(skill_row="NeverMeasured", tier="weak"))
    assert parse(step(skill_row="NeverMeasured", tier="none")).tier == "none"


@pytest.mark.parametrize(
    "tier,row", [("reliable", "PickPlaceSinkToCounter"), ("unstable", "OpenCabinet")]
)
def test_takeover_rejects_nonweak_tiers(tier, row):
    with pytest.raises(ValueError, match="takeover"):
        parse(step(skill_row=row, tier=tier, control_mode="takeover"))


@pytest.mark.parametrize("task", [None, "OpenDrawer", "UnlistedTask"])
def test_transition_requires_registry_composite(task):
    with pytest.raises(ValueError, match="transition"):
        parse(step(control_mode="transition"), task_name=task)


def test_transition_allowed_on_composite():
    assert (
        parse(
            step(control_mode="transition"), task_name="SteamInMicrowave"
        ).control_mode
        == "transition"
    )


@pytest.mark.parametrize(
    "tool", ["smolvla_pick", "continue_vla", "execute_vla", "resume_vla"]
)
def test_transition_cannot_allow_policy_tools(tool):
    with pytest.raises(ValueError, match="transition"):
        parse(
            step(control_mode="transition", allowed_tools=[tool]),
            task_name="SteamInMicrowave",
        )


@pytest.mark.parametrize("mode", ["delegate", "monitor"])
@pytest.mark.parametrize("tool", sorted(PRIVILEGED_PHYSICAL_TOOLS))
def test_privileged_tools_rejected_by_parser(mode, tool):
    with pytest.raises(ValueError, match="privileged"):
        parse(step(control_mode=mode, allowed_tools=[tool]))


def test_existing_allowlist_gate_and_read_only_zoom():
    machine = state()
    assert machine.register_plan("goal", [step()])["ok"]
    gate = PlanSafetyGate(mode="plan-gated")
    assert not gate.evaluate(
        name="navigate_to", input_dict={}, state=machine, control_tools=set()
    ).allowed
    assert gate.evaluate(
        name="zoom", input_dict={}, state=machine, control_tools=set()
    ).allowed
    assert parse(step(allowed_tools=["move", "zoom"])).allowed_tools == ("move", "zoom")


@pytest.mark.parametrize(
    "mode,flag",
    [
        ("transition", "transition_takeover_enabled"),
        ("takeover", "op_takeover_enabled"),
    ],
)
def test_ablation_switches(mode, flag):
    with pytest.raises(ValueError, match=flag):
        parse(step(control_mode=mode), task_name="SteamInMicrowave", **{flag: False})


def test_fallback_persists_across_replan_and_rejects_repeat_takeover():
    events = []
    machine = state(
        record=lambda event, **payload: events.append({"event": event, **payload})
    )
    assert machine.register_plan("goal", [step(control_mode="takeover")])["ok"]
    assert fail(machine)["ok"]
    assert machine.replan(
        "retry",
        [
            step(
                id="renamed",
                control_mode="takeover",
                covers_outcomes=["outcome-a", "outcome-b"],
            )
        ],
    )["ok"]
    assert fail(machine, execution_id="second")["ok"]
    # 1.0.11 B4: failures inside the quota no longer end the takeover ...
    assert not [e for e in events if e["event"] == "control_mode_fallback"]
    assert machine.replan("retry", [step(id="renamed", control_mode="takeover")])["ok"]
    # ... its step quota does (TurnOnMicrowave: 1.5 x 284 = 426 steps).
    assert fail(machine, execution_id="third", executed=420)["ok"]
    fallback = [e for e in events if e["event"] == "control_mode_fallback"]
    assert len(fallback) == 1
    assert {k: fallback[0][k] for k in ("from_mode", "to_mode", "operation_key", "subtask",
                                         "failed_verifications", "step", "reason")} == {
        "from_mode": "takeover",
        "to_mode": "monitor",
        "operation_key": ("outcome-a", "outcome-b"),
        "subtask": "whole",
        "failed_verifications": 3,
        "step": "renamed",
        "reason": "step_quota",
    }
    assert (
        "monitor" in machine.replan("again", [step(control_mode="takeover")])["error"]
    )
    assert machine.replan("delegate", [step(control_mode="monitor")])["ok"]
    assert machine.snapshot()["control_mode_fallbacks"]
    assert all(
        e["control_mode"] == "takeover" for e in events if e["event"] == "step_verified"
    )
    assert all(
        e["operation_key"] == ["outcome-a", "outcome-b"]
        for e in events if e["event"] == "step_verified"
    )
    for event in events:
        if event["event"] in {"plan_registered", "plan_replanned"}:
            for item in event.get("steps", event.get("replacement", [])):
                assert all(
                    key in item
                    for key in ("skill_row", "tier", "control_mode", "defaulted")
                )


def test_weak_monitor_twice_allows_but_does_not_force_takeover():
    machine = state()
    machine.register_plan("goal", [step()])
    fail(machine)
    machine.replan("retry", [step(id="new")])
    fail(machine, execution_id="second")
    assert machine.replan("stay", [step()])["ok"]
    fail(machine, execution_id="third")
    assert machine.replan("take over", [step(control_mode="takeover")])["ok"]


def test_uncertain_and_progress_do_not_count_as_failure():
    machine = state()
    machine.register_plan("goal", [step(control_mode="takeover")])
    fail(machine, verdict="uncertain")
    machine.verify_step(
        "progress", ["some motion"], "progress", execution_id="physical"
    )
    assert machine.snapshot()["control_mode_failures"] == []


def test_retry_verifications_count_and_empty_coverage_key_is_stable():
    machine = state()
    machine.register_plan("goal", [step(control_mode="takeover", covers_outcomes=[])])
    fail(machine, verdict="retry")
    machine.replan(
        "retry",
        [
            step(
                id="new-id",
                intent="different wording",
                control_mode="takeover",
                covers_outcomes=[],
            )
        ],
    )
    fail(machine, verdict="retry", execution_id="second")
    # Without coverage or an operation name the row keeps the identity.
    [failure] = machine.control_planning_context(include_capabilities=False)["control_mode_failures"]
    assert failure["failed_verifications"] == {"takeover": 2}
    assert failure["operation_key"] == ["legacy-operation", "", "TurnOnMicrowave"]
    assert machine.replan("again", [step(control_mode="takeover", covers_outcomes=[])])["ok"]


def test_missing_policy_catalog_normalizes_tier_and_warns(tmp_path):
    harness = HarnessController(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools=TOOLS,
        policy_family="xiaomi",
        task_name="SteamInMicrowave",
    )
    assert harness.state.register_plan("goal", [step(control_mode="takeover")])["ok"]
    assert harness.state.active_step.tier == "none"
    warnings = [
        json.loads(line)
        for line in harness.trace_path.read_text().splitlines()
        if "capability_table_missing" in line
    ]
    assert len(warnings) == 1
    assert warnings[0]["policy"] == "xiaomi"


def test_planner_gets_summary_and_fallback_without_coordinates(tmp_path):
    class Generator:
        def generate(self, **kwargs):
            self.initial = kwargs
            return {"strategy": "act", "steps": [step(control_mode="takeover")]}

        def generate_replan(self, **kwargs):
            self.replacement = kwargs
            return {"strategy": "monitor", "steps": [step()]}

    generator = Generator()
    harness = HarnessController(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools=TOOLS,
        policy_family="synthetic",
        plan_generator=generator,
    )
    proposal = harness.create_plan("goal", ["camera"])
    assert harness.register_plan(proposal["proposal_id"])["ok"]
    fail(harness.state, verdict="retry")
    fail(harness.state, verdict="retry", execution_id="second", executed=430)
    assert harness.replan("failure", ["camera"])["ok"]
    assert "PickPlaceSinkToCounter" in " ".join(generator.initial["constraints"])
    assert "distance_m" not in " ".join(generator.initial["constraints"])
    [fallback] = generator.replacement["current_plan"]["control_mode_fallbacks"]
    assert fallback["failed_verifications"] == 2 and fallback["reason"] == "step_quota"






@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "reliable"),
        ("episodes", 0),
        ("wilson95", [0.9, 0.1]),
        ("typical_steps", -1),
        ("crosses_threshold", True),
    ],
)
def test_loader_rejects_malformed_row(field, value):
    catalog = deepcopy(load_capabilities("synthetic"))
    catalog["rows"]["TurnOnMicrowave"][field] = value
    with pytest.raises(ValueError):
        validate_capabilities(catalog)


def test_planning_config_switches_reach_controller(tmp_path):
    stack = resolve_planning_stack(
        "robocasa-hooked-gated-v1",
        transition_takeover_enabled=False,
        op_takeover_enabled=False,
    )
    harness = stack.build_harness(trace_path=tmp_path / "trace")
    harness.set_physical_tools(TOOLS)
    assert stack.metadata()["effective"]["transition_takeover_enabled"] is False
    assert (
        "op_takeover_enabled"
        in harness.state.register_plan("goal", [step(control_mode="takeover")])["error"]
    )


def test_tool_call_event_and_result_control_mode(tmp_path):
    from rpent.planning.events import JsonlEventRecorder

    recorder = JsonlEventRecorder(tmp_path / "events.jsonl", episode_id="test")
    toolkit = Toolkit()
    toolkit.add_tool(
        "move",
        {"name": "move", "input_schema": {"type": "object", "properties": {}}},
        lambda: {"ok": True},
    )
    toolkit.configure_harness_physical_tools(TOOLS)
    toolkit.attach_event_recorder(recorder)
    harness = HarnessController(
        mode="plan-gated",
        trace_path=tmp_path / "trace",
        physical_tools=TOOLS,
        policy_family="synthetic",
    )
    toolkit.attach_harness(harness)
    harness.state.register_plan("goal", [step(control_mode="takeover")])
    result = toolkit.execute_tool("move", {})
    assert result.result["control_mode"] == "takeover"
    event = next(
        json.loads(line)
        for line in recorder.path.read_text().splitlines()
        if json.loads(line)["event"] == "tool_call_proposed"
    )
    assert event["control_mode"] == "takeover"


def test_runtime_generator_preserves_fields_and_schema():
    from rpent.runtime.base import RuntimeResult

    class Runtime:
        def solve(self, **kwargs):
            self.schema = kwargs["toolkit"].get_tools_spec()[0]["input_schema"]
            self.message = kwargs["user_message"]
            return RuntimeResult(
                finish_result={
                    "strategy": "act",
                    "steps": [step(control_mode="takeover")],
                }
            )

    runtime = Runtime()
    generator = RuntimePlanGenerator(runtime, require_outcome_program=False)
    generator.configure_control_modes(
        policy_family="synthetic", task_name="SteamInMicrowave"
    )
    candidate = generator.generate(
        goal="goal", observations=["camera"], constraints=[], physical_tools=TOOLS
    )
    assert candidate["steps"][0]["control_mode"] == "takeover"
    assert candidate["steps"][0]["skill_row"] == "TurnOnMicrowave"
    assert not candidate["steps"][0]["defaulted"]
    fields = runtime.schema["properties"]["steps"]["items"]["properties"]
    assert all(key in fields for key in ("skill_row", "tier", "control_mode"))




def test_declared_proxy_inherits_tier_but_is_capped_at_monitor():
    parsed = parse(step(skill_row="PickPlaceCounterToOpenFridge", tier="reliable", control_mode="monitor"))
    assert (parsed.skill_row, parsed.tier, parsed.control_mode) == ("PickPlaceCounterToOpenFridge", "reliable", "monitor")
    with pytest.raises(ValueError, match="allows control_mode"):
        parse(step(skill_row="PickPlaceCounterToOpenFridge", tier="reliable", control_mode="delegate"))
    with pytest.raises(ValueError, match="requires tier=reliable"):
        parse(step(skill_row="PickPlaceCounterToOpenFridge", tier="unstable", control_mode="monitor"))


def test_declared_demonstrated_operation_is_unstable_monitor():
    parsed = parse(step(skill_row="PlateToBowlSameCounter", tier="unstable", control_mode="monitor"))
    assert parsed.tier == "unstable"
    with pytest.raises(ValueError):
        parse(step(skill_row="PlateToBowlSameCounter", tier="none", control_mode="takeover"))
    with pytest.raises(ValueError, match="requires tier=none"):
        parse(step(skill_row="none", tier="unstable", control_mode="monitor"))


def test_planner_summary_lists_declared_entries_with_mode_limits():
    summary = {row["skill_row"]: row for row in capability_summary(load_capabilities("synthetic"))}
    proxy = summary["PickPlaceCounterToOpenFridge"]
    assert proxy["declared"] == "proxy" and proxy["proxy_row"] == "PickPlaceFridgeDrawerToShelf"
    assert proxy["tier"] == "reliable" and proxy["allowed_control_modes"] == ["monitor"]
    assert summary["PlateToBowlSameCounter"]["typical_steps"] is None
    assert summary["TurnOnMicrowave"].get("allowed_control_modes") is None


@pytest.mark.parametrize(
    "name,entry",
    [
        ("TurnOnMicrowave", {"kind": "demonstrated", "tier": "weak"}),
        ("Proxy", {"kind": "proxy", "proxy_row": "NotMeasured"}),
        ("Proxy", {"kind": "proxy", "proxy_row": "TurnOnMicrowave", "tier": "weak"}),
        ("Demo", {"kind": "demonstrated", "tier": "none"}),
        ("Demo", {"kind": "demonstrated", "tier": "weak", "allowed_control_modes": ["transition"]}),
        ("Demo", {"kind": "unknown", "tier": "weak"}),
    ],
)
def test_loader_rejects_malformed_declared_entries(name, entry):
    catalog = deepcopy(load_capabilities("synthetic"))
    catalog["declared"] = {name: {"operation": "op", "evidence": "ev", "allowed_control_modes": ["monitor"], **entry}}
    with pytest.raises(ValueError):
        validate_capabilities(catalog)






def act(machine, executed, *, execution_id="physical", verdict="progress", failure_type=None):
    machine.record_action_feedback(
        ActionFeedback(
            execution_id=execution_id,
            tool_name="move",
            physical=True,
            execution_status="completed",
            raw_result={"executed_actions": executed},
        ),
        input_dict={},
    )
    return machine.verify_step(
        verdict, ["policy still working"], "observed", execution_id=execution_id,
        failure_type=failure_type,
    )


def escalation_events(events):
    return [e for e in events if e["event"] == "control_mode_escalation_available"]


SINK = {"skill_row": "PickPlaceCounterToSink", "tier": "reliable", "control_mode": "delegate"}


def test_steps_beyond_factor_times_typical_make_reliable_takeover_legal():
    events = []
    machine = state(record=lambda event, **payload: events.append({"event": event, **payload}))
    assert machine.register_plan("goal", [step(**SINK, max_attempts=20)])["ok"]
    # 1.0.11 H4: an escalated takeover is staged and lists the policy tool.
    takeover = step(skill_row="PickPlaceCounterToSink", tier="reliable", control_mode="takeover",
                    allowed_tools=["move", "continue_vla"])
    with pytest.raises(ValueError, match="requires weak or none tier"):
        machine.parse_steps([takeover])
    for index in range(5):  # 550 > 1.5 x 316 = 474, but every window progressed
        act(machine, 110, execution_id=f"e{index}")
    assert not escalation_events(events)
    # 1.0.11 B2: past the threshold, a further no-progress window is needed.
    act(machine, 80, execution_id="e5", verdict="retry", failure_type="timeout")
    assert not escalation_events(events)
    act(machine, 40, execution_id="e6", verdict="retry", failure_type="timeout")  # 120 without progress
    [event] = escalation_events(events)
    assert event["reason"] == "steps_no_progress" and event["steps"] == 670
    assert event["typical_steps"] == 316 and event["factor"] == 1.5
    assert event["from_mode"] == "delegate" and event["subtask"] == "whole"
    assert event["operation_key"] == ("outcome-a", "outcome-b")
    context = machine.control_planning_context(include_capabilities=False)
    assert context["control_mode_escalations"][0]["operation_key"] == ["outcome-a", "outcome-b"]
    assert machine.replan("escalate", [takeover])["ok"]
    act(machine, 50, execution_id="e7")
    assert len(escalation_events(events)) == 1


def test_verified_pass_stops_step_escalation():
    events = []
    machine = state(record=lambda event, **payload: events.append({"event": event, **payload}))
    assert machine.register_plan("goal", [step(**SINK), step(id="next", **SINK, covers_outcomes=["c"])])["ok"]
    act(machine, 400, execution_id="pass", verdict="pass")
    act(machine, 10, execution_id="after")
    assert not [e for e in escalation_events(events) if e["operation_key"] == ("outcome-a", "outcome-b")]


def test_first_verified_failure_escalates_any_tier_without_typical_steps():
    events = []
    machine = state(record=lambda event, **payload: events.append({"event": event, **payload}))
    demo = {"skill_row": "PlateToBowlSameCounter", "tier": "unstable", "control_mode": "monitor"}
    assert machine.register_plan("goal", [step(**demo)])["ok"]
    act(machine, 5000, execution_id="long")  # no typical steps: steps alone never escalate
    assert not escalation_events(events)
    assert fail(machine, execution_id="f1")["ok"]
    assert not escalation_events(events)
    assert fail(machine, execution_id="f2")["ok"]
    [event] = escalation_events(events)
    assert event["reason"] == "failures" and event["failed_verifications"] == 2
    assert event["typical_steps"] is None
    assert machine.replan("escalate", [step(skill_row="PlateToBowlSameCounter", tier="unstable",
                                            control_mode="takeover", allowed_tools=["move", "continue_vla"])])["ok"]


def test_monitor_only_proxy_may_take_over_only_after_escalation():
    proxy = {"skill_row": "PickPlaceCounterToOpenFridge", "tier": "reliable"}
    machine = state()
    assert machine.register_plan("goal", [step(**proxy, control_mode="monitor", max_attempts=20)])["ok"]
    takeover = step(**proxy, control_mode="takeover", allowed_tools=["move", "continue_vla"])
    with pytest.raises(ValueError, match="allows control_mode"):
        machine.parse_steps([takeover])
    with pytest.raises(ValueError, match="allows control_mode"):
        parse(step(**proxy, control_mode="delegate"))
    # A declared proxy has no step threshold (1.0.11 B3): one stalled window escalates.
    act(machine, 520, execution_id="slow", verdict="fail")
    assert machine.escalated_operations() == {(("outcome-a", "outcome-b"), "whole")}
    assert machine.replan("escalate", [takeover])["ok"]


def test_escalated_takeover_falls_back_on_quota_and_policy_failure_reopens_it():
    events = []
    machine = state(record=lambda event, **payload: events.append({"event": event, **payload}))
    assert machine.register_plan("goal", [step(**SINK, max_attempts=20)])["ok"]
    act(machine, 500, execution_id="slow", verdict="fail")
    takeover = step(skill_row="PickPlaceCounterToSink", tier="reliable", control_mode="takeover",
                    allowed_tools=["move", "continue_vla"])
    assert machine.replan("escalate", [takeover])["ok"]
    assert fail(machine, execution_id="t1")["ok"]
    assert machine.replan("retry", [takeover])["ok"]
    assert fail(machine, execution_id="t2", executed=470)["ok"]  # quota 1.5 x 316 = 474
    assert [e for e in events if e["event"] == "control_mode_fallback"]
    assert machine.escalated_operations() == set()
    assert not machine.control_planning_context(include_capabilities=False)["control_mode_escalations"]
    assert "monitor" in machine.replan("again", [takeover])["error"]
    monitor = step(skill_row="PickPlaceCounterToSink", tier="reliable", control_mode="monitor")
    assert machine.replan("monitor", [monitor])["ok"]
    # 1.0.11 B5: a physical policy failure after the fallback reopens takeover.
    assert fail(machine, execution_id="m1")["ok"]
    assert len(escalation_events(events)) == 2
    assert escalation_events(events)[-1]["reason"] == "policy_failed_after_fallback"
    assert machine.replan("take over again", [takeover])["ok"]


def test_op_takeover_ablation_disables_escalation():
    machine = state(op_takeover_enabled=False)
    assert machine.register_plan("goal", [step(**SINK, max_attempts=20)])["ok"]
    act(machine, 1000, execution_id="slow")
    assert machine.escalated_operations() == set()


def test_legacy_operation_key_ignores_repeated_mode_text_in_intent():
    from rpent.harness.state import operation_key

    base = {"skill_row": "none", "tier": "none", "covers_outcomes": []}
    monitor = parse(step(**base, control_mode="monitor", intent="Put the bowl in the fridge; row=none; tier=none; control_mode=monitor"))
    takeover = parse(step(**base, control_mode="takeover", intent="Put the bowl in the fridge; row=none; tier=none; control_mode=takeover"))
    assert operation_key(monitor) == operation_key(takeover)


def test_generator_candidates_see_current_escalations(tmp_path):
    class Generator:
        def configure_control_modes(self, **kwargs):
            self.config = kwargs

    generator = Generator()
    controller = HarnessController(mode="plan-gated", plan_generator=generator,
                                   trace_path=tmp_path / "trace.jsonl", physical_tools=TOOLS)
    controller.configure_control_modes(policy_family="synthetic", task_name="StoreLeftoversInBowl")
    assert controller.state.register_plan("goal", [step(**SINK, max_attempts=20)])["ok"]
    controller.state.record_action_feedback(
        ActionFeedback(execution_id="slow", tool_name="move", physical=True,
                       execution_status="completed", raw_result={"executed_actions": 500}),
        input_dict={},
    )
    # 1.0.11: escalation is decided with the verdict (progress or not).
    controller.state.verify_step("retry", ["stalled"], "stalled", execution_id="slow",
                                 failure_type="timeout")
    constraints = controller.planning_constraints([])
    assert generator.config["control_constraints"]["escalated"] == {(("outcome-a", "outcome-b"), "whole")}
    contract = json.loads(constraints[-1].split(": ", 1)[1])
    assert contract["control_mode_escalations"][0]["reason"] == "steps_no_progress"


def test_planner_summary_gives_atomic_rows_their_official_instruction():
    summary = {row["skill_row"]: row for row in capability_summary(load_capabilities("synthetic"))}
    assert summary["PickPlaceSinkToCounter"]["instruction"] == (
        "Pick the item from the sink and place it on the container located on the counter."
    )
    assert summary["OpenFridge"]["instruction"] == "Open the fridge door(s)."
    # 1.0.11 D3: composite rows only for the current task, never with an instruction.
    assert "TiltPan" not in summary
    current = {row["skill_row"]: row for row in capability_summary(load_capabilities("synthetic"), task_name="TiltPan")}
    assert "instruction" not in current["TiltPan"]
    assert all("instruction" in row for row in summary.values()
               if row.get("declared") is None and row["skill_row"] in {
                   name for name, r in load_capabilities("synthetic")["rows"].items() if r["family"] == "atomic"})
