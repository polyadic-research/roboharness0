"""Contract tests for the authoritative Harness session boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from rpent.harness import Harness
from rpent.harness.lifecycle import ALLOWED_OPERATIONS, HarnessLifecycleState
from rpent.harness.recovery import RecoveryPolicy
from rpent.harness.session import HarnessSession
from rpent.harness.verification import (
    PredicateVerificationResult,
    VerificationCoordinator,
    VerificationPolicy,
)
from rpent.planning.outcome_program import compile_outcome_program
from rpent.planning.schemas import ActionFeedback, SkillOutcome

NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)
OBSERVED_AT = NOW.isoformat()


class Generator:
    def __init__(self, *, count: int = 2, durable_first: bool = False) -> None:
        self.count = count
        self.durable_first = durable_first
        self.fail_recovery = False
        self.all_durable = False
        self.last_replan_context = None

    def generate(self, *, goal, observations, constraints, physical_tools):
        del observations, constraints, physical_tools
        outcomes = [
            {
                "id": f"outcome-{index}",
                "instruction_index": index,
                "description": f"complete outcome {index}",
                "success_predicate": f"outcome {index} is visibly complete",
                "depends_on": [] if index == 1 else [f"outcome-{index - 1}"],
                "lifecycle": (
                    "persistent"
                    if self.all_durable or (self.durable_first and index == 1)
                    else "event"
                ),
            }
            for index in range(1, self.count + 1)
        ]
        return {
            "goal": goal,
            "strategy": "complete outcomes in order",
            "outcomes": outcomes,
            "stages": [
                {
                    "id": "stage-1",
                    "intent": "complete outcomes",
                    "outcome_ids": [item["id"] for item in outcomes],
                }
            ],
            "steps": [
                {
                    "id": f"step-{index}",
                    "intent": f"complete outcome {index}",
                    "allowed_tools": ["move"],
                    "expected_evidence": [f"outcome {index} complete"],
                    "covers_outcomes": [f"outcome-{index}"],
                    "stage_id": "stage-1",
                    "max_attempts": 2,
                }
                for index in range(1, self.count + 1)
            ],
        }

    def generate_replan(self, **kwargs):
        if self.fail_recovery:
            raise RuntimeError("planner unavailable")
        current = kwargs["current_plan"]
        self.last_replan_context = deepcopy(current)
        active = current["active_outcome_id"]
        stage_id = current["active_stage_id"]
        return {
            "goal": kwargs["goal"],
            "strategy": "recover locally",
            "stages": [
                {
                    "id": stage_id,
                    "intent": "recover",
                    "outcome_ids": current["required_outcome_ids"],
                }
            ],
            "steps": [
                {
                    "id": "repair-step",
                    "intent": "retry outcome",
                    "allowed_tools": ["move"],
                    "expected_evidence": ["recovery visible"],
                    "covers_outcomes": [active],
                    "stage_id": stage_id,
                    "max_attempts": 2,
                }
            ],
        }


class Verifier:
    name = "test-verifier"
    version = "1"

    def __init__(self) -> None:
        self.verdict = "satisfied"
        self.authoritative = True
        self.by_outcome = {}
        self.requests = []

    def verify(self, request):
        self.requests.append(request)
        verdict = self.by_outcome.get(request.outcome_id, self.verdict)
        return PredicateVerificationResult(
            verdict=verdict,
            evidence_refs=request.observation_refs,
            verifier=self.name,
            verifier_version=self.version,
            confidence=1.0,
            authoritative=self.authoritative,
            reason="test decision",
        )


class TwoStageGenerator(Generator):
    def __init__(self) -> None:
        super().__init__(count=2)

    def generate(self, **kwargs):
        generated = super().generate(**kwargs)
        generated["stages"] = [
            {"id": "stage-1", "intent": "first", "outcome_ids": ["outcome-1"]},
            {"id": "stage-2", "intent": "second", "outcome_ids": ["outcome-2"]},
        ]
        generated["steps"] = generated["steps"][:1]
        return generated

    def generate_replan(self, **kwargs):
        return {
            "goal": kwargs["goal"],
            "strategy": "execute next stage",
            "stages": [
                {"id": "stage-2", "intent": "second", "outcome_ids": ["outcome-2"]}
            ],
            "steps": [
                {
                    "id": "step-2",
                    "intent": "complete outcome 2",
                    "allowed_tools": ["move"],
                    "expected_evidence": ["outcome 2 complete"],
                    "covers_outcomes": ["outcome-2"],
                    "stage_id": "stage-2",
                }
            ],
        }


class CrossStageRecoveryGenerator(TwoStageGenerator):
    def generate(self, **kwargs):
        generated = super().generate(**kwargs)
        generated["outcomes"][0]["lifecycle"] = "persistent"
        generated["outcomes"][1]["depends_on"] = []
        return generated

    def generate_replan(self, **kwargs):
        current = kwargs["current_plan"]
        if current["active_stage_id"] == "stage-2":
            return super().generate_replan(**kwargs)
        return {
            "goal": kwargs["goal"],
            "strategy": "restore earlier durable outcome",
            "stages": [
                {
                    "id": "stage-1",
                    "intent": "first",
                    "outcome_ids": current["required_outcome_ids"],
                }
            ],
            "steps": [
                {
                    "id": "restore-step-1",
                    "intent": "restore outcome one",
                    "allowed_tools": ["move"],
                    "covers_outcomes": ["outcome-1"],
                    "stage_id": "stage-1",
                }
            ],
        }


class ThreeStageGenerator(Generator):
    def __init__(self) -> None:
        super().__init__(count=3)
        self.received_required_ids = None
        self.received_active_stage_id = None

    def generate(self, **kwargs):
        generated = super().generate(**kwargs)
        generated["stages"] = [
            {
                "id": f"stage-{index}",
                "intent": f"stage {index}",
                "outcome_ids": [f"outcome-{index}"],
            }
            for index in range(1, 4)
        ]
        generated["steps"] = generated["steps"][:1]
        return generated

    def generate_replan(self, **kwargs):
        self.received_required_ids = kwargs["current_plan"]["required_outcome_ids"]
        self.received_active_stage_id = kwargs["current_plan"]["active_stage_id"]
        return {
            "goal": kwargs["goal"],
            "strategy": "execute all remaining stages",
            "stages": [
                {"id": "stage-2", "intent": "stage 2", "outcome_ids": ["outcome-2"]},
                {"id": "stage-3", "intent": "stage 3", "outcome_ids": ["outcome-3"]},
            ],
            "steps": [
                {
                    "id": "step-2",
                    "intent": "complete outcome 2",
                    "allowed_tools": ["move"],
                    "covers_outcomes": ["outcome-2"],
                    "stage_id": "stage-2",
                }
            ],
        }


class SerializedProgramGenerator(Generator):
    def __init__(self, *, mismatch_top_level: bool = False) -> None:
        super().__init__(count=1)
        self.mismatch_top_level = mismatch_top_level

    def generate(self, **kwargs):
        generated = super().generate(**kwargs)
        program = compile_outcome_program(
            kwargs["goal"],
            generated["outcomes"],
            generated["stages"],
            explicit_contract=[
                {"id": "outcome-1", "number": 1, "text": "complete outcome 1"}
            ],
        )
        payload = program.to_payload()
        generated.update(
            outcome_program=payload,
            contract_id=(
                "contract-wrong" if self.mismatch_top_level else program.contract_id
            ),
            outcome_program_schema_version=program.schema_version,
            outcomes=list(payload["outcomes"]),
            stages=list(payload["stages"]),
        )
        return generated


def feedback(index: int, *, failed: bool = False, affected=()) -> ActionFeedback:
    status = "retryable_failure" if failed else "success"
    return ActionFeedback(
        execution_id=f"execution-{index}",
        tool_name="move",
        physical=True,
        execution_status="completed",
        environment_success=False,
        environment_terminated=False,
        environment_truncated=False,
        outcome=SkillOutcome(
            status=status,
            failure_type="motion_blocked" if failed else None,
            retryable=True if failed else None,
            evidence=("controller evidence",),
            source="adapter",
        ),
        raw_result={"affected_outcome_ids": list(affected)},
    )


def make_session(tmp_path, *, count=2, durable_first=False, recovery_policy=None):
    verifier = Verifier()
    generator = Generator(count=count, durable_first=durable_first)
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=generator,
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier],
            policy=VerificationPolicy(clock=lambda: NOW),
        ),
        recovery_policy=recovery_policy,
    )
    return session, generator, verifier


def test_start_is_atomic_one_shot_and_assigns_action(tmp_path):
    session, _, _ = make_session(tmp_path)
    assert session.lifecycle_state is None

    result = session.start_instruction("do two things", ["scene visible"])

    assert result["ok"] is True
    assert session.lifecycle_state is HarnessLifecycleState.ACTION_REQUIRED
    before = deepcopy(session.execution_status())
    rejected = session.start_instruction("replace it", ["different scene"])
    assert "already started" in rejected["error"]
    assert session.execution_status() == before


def test_every_advertised_non_physical_operation_has_session_entrypoint(tmp_path):
    session, _, _ = make_session(tmp_path)
    operations = {
        operation
        for declared in ALLOWED_OPERATIONS.values()
        for operation in declared
        if operation != "configured_physical_tool"
    }

    assert operations == {
        "apply_recovery",
        "execution_status",
        "finish",
        "request_exception",
        "submit_observation",
        "verify_observation",
        "verify_outcomes",
        "verify_step",
    }
    assert all(callable(getattr(session, operation)) for operation in operations)


def test_failed_bootstrap_leaves_session_unstarted(tmp_path):
    session, generator, _ = make_session(tmp_path)
    generator.count = 0

    result = session.start_instruction("bad", ["scene"])

    assert "error" in result
    assert session.lifecycle_state is None
    assert session.controller.outcome_graph is None


def test_illegal_operation_has_no_mutation(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    before = deepcopy(session.controller.strict_snapshot())

    result = session.verify_outcomes(["outcome-1"], ["obs-1"], [OBSERVED_AT])

    assert "not allowed" in result["error"]
    assert session.controller.strict_snapshot() == before


def test_uncertain_verification_keeps_exact_execution_and_blocks_second_action(
    tmp_path,
):
    session, _, verifier = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    session.record_action_feedback(feedback(1))
    verifier.verdict = "uncertain"

    checked = session.submit_observation(["obs-1"], [OBSERVED_AT])
    before = deepcopy(session.controller.strict_snapshot())
    second = session.record_action_feedback(feedback(2))

    assert checked["verification"]["verdict"] == "uncertain"
    assert session.execution_status()["execution_id"] == "execution-1"
    assert "not allowed" in second["error"]
    assert session.controller.strict_snapshot() == before


def test_external_state_changes_only_through_transition_helper(tmp_path, monkeypatch):
    import rpent.harness.session as session_module

    session, _, _ = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    calls = []
    real_transition = session_module.transition

    def recording_transition(state, event):
        calls.append((state, event))
        return real_transition(state, event)

    monkeypatch.setattr(session_module, "transition", recording_transition)
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])

    assert [event.value for _, event in calls] == [
        "action_progressed",
        "outcome_satisfied",
    ]


def test_authoritative_success_advances_outcome_and_contract(tmp_path):
    session, _, _ = make_session(tmp_path, count=2)
    session.start_instruction("do two things", ["scene visible"])

    session.record_action_feedback(feedback(1))
    assert session.submit_observation(["obs-1"], [OBSERVED_AT])["ok"]
    assert session.execution_status()["outcome_id"] == "outcome-2"
    session.record_action_feedback(feedback(2))
    assert session.submit_observation(["obs-2"], [OBSERVED_AT])["ok"]

    status = session.execution_status()
    assert status["state"] == "complete"
    assert status["contract_success"] is True
    assert session.finish("success", "done")["ok"] is True


def test_snapshot_combines_strict_domains_lifecycle_and_finish(tmp_path):
    session, _, _ = make_session(tmp_path, count=1)
    session.start_instruction("do one thing", ["scene visible"])
    session.record_action_feedback(feedback(1))
    assert session.submit_observation(["obs-1"], [OBSERVED_AT])["ok"]
    assert session.finish("success", "done")["ok"] is True

    snapshot = session.snapshot()

    assert snapshot["state"] == "complete"
    assert snapshot["contract_success"] is True
    assert snapshot["finished"] == {"status": "success", "summary": "done"}
    assert snapshot["outcome_program"]["contract_id"]
    assert snapshot["outcome_graph"]["active_outcome_id"] is None
    assert snapshot["stage"]["active"] == "stage-1"
    assert snapshot["recovery"]["outstanding_decisions"] == []

    snapshot["finished"]["summary"] = "mutated"
    assert session.snapshot()["finished"]["summary"] == "done"


def test_authoritative_not_satisfied_enters_recovery(tmp_path):
    session, _, verifier = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    session.record_action_feedback(feedback(1))
    verifier.verdict = "not_satisfied"

    result = session.submit_observation(["obs-1"], [OBSERVED_AT])

    assert result["verification"]["verdict"] == "not_satisfied"
    assert session.execution_status()["state"] == "recovery_required"


def test_recovery_failure_is_atomic_and_tracker_owns_budget(tmp_path):
    policy = RecoveryPolicy(
        local_replans_per_outcome=1,
        stage_replans_per_stage=0,
        remaining_stage_repartitions_per_run=0,
    )
    session, generator, _ = make_session(tmp_path, recovery_policy=policy)
    session.start_instruction("do two things", ["scene visible"])
    session.request_exception("path blocked", ["obs-blocked"])
    generator.fail_recovery = True
    before = deepcopy(session.controller.strict_snapshot())

    failed = session.apply_recovery("recover", ["path still blocked"])

    assert "planner unavailable" in failed["error"]
    assert session.controller.strict_snapshot() == before
    assert session.execution_status()["state"] == "recovery_required"
    generator.fail_recovery = False
    recovered = session.apply_recovery("recover", ["path visible"])
    assert recovered["scope"] == "replan_outcome"
    assert recovered["ok"] is True
    assert session.controller.recovery_tracker.snapshot()[
        "outcome_replans_consumed"
    ] == {"outcome-1": 1}
    assert generator.last_replan_context["required_outcome_ids"] == ["outcome-1"]
    assert generator.last_replan_context["deferred_outcome_ids"] == ["outcome-2"]


def test_applied_recovery_requires_physical_action_before_another_exception(
    tmp_path,
):
    policy = RecoveryPolicy(
        local_replans_per_outcome=1,
        stage_replans_per_stage=1,
        remaining_stage_repartitions_per_run=1,
        max_total_physical_actions_per_run=9,
    )
    session, _, verifier = make_session(tmp_path, recovery_policy=policy)
    session.start_instruction("do two things", ["scene visible"])
    session.request_exception("path blocked", ["obs-blocked"])
    recovered = session.apply_recovery("recover", ["path visible"])
    assert recovered["scope"] == "replan_outcome"
    before = deepcopy(session.controller.strict_snapshot())

    blocked = session.request_exception(
        "camera views are unchanged",
        ["obs-still-unchanged"],
    )

    assert "physical action" in blocked["error"]
    assert blocked["state"] == "action_required"
    assert session.controller.strict_snapshot() == before

    allowed = session.physical_action_allowed("move", "execution-1")
    assert allowed["allowed"] is True
    assert (
        session.controller.recovery_tracker.snapshot()["pending_recovery_action"]
        is None
    )
    assert session.record_action_feedback(feedback(1))["ok"] is True
    verifier.verdict = "not_satisfied"

    verified = session.submit_observation(["obs-after-action"], [OBSERVED_AT])

    assert verified["verification"]["verdict"] == "not_satisfied"
    assert verified["state"] == "recovery_required"


def test_retryable_feedback_uses_tracker_issued_step_retry(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    session.record_action_feedback(feedback(1, failed=True))

    recovered = session.apply_recovery("retry with clearance", ["path clear"])

    assert recovered["ok"] is True
    assert recovered["scope"] == "retry_step"
    assert session.execution_status()["state"] == "action_required"
    assert (
        session.controller.recovery_tracker.snapshot()["outcome_replans_consumed"] == {}
    )


def test_retryable_primitive_failure_cannot_be_overridden_by_manual_pass(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])
    active_step_id = session.controller.state.active_step.id

    session.record_action_feedback(feedback(1, failed=True))
    attempted_pass = session.verify_step(
        "pass",
        ["image claim without physical confirmation"],
        "claim grasp succeeded",
        execution_id="execution-1",
    )

    assert "error" in attempted_pass
    assert attempted_pass["state"] == "recovery_required"
    assert session.controller.state.active_step.id == active_step_id


def test_exhausted_recovery_fails_closed(tmp_path):
    policy = RecoveryPolicy(
        local_replans_per_outcome=0,
        stage_replans_per_stage=0,
        remaining_stage_repartitions_per_run=0,
    )
    session, _, _ = make_session(tmp_path, recovery_policy=policy)
    session.start_instruction("do two things", ["scene visible"])
    session.request_exception("unsafe", ["blocked"])

    result = session.apply_recovery("recover", ["blocked"])

    assert result["scope"] == "terminal_failure"
    assert session.execution_status()["state"] == "terminal_failure"
    assert session.finish("failure", "budget exhausted")["ok"] is True


def test_durable_outcome_requires_fresh_final_evidence_and_is_invalidated(tmp_path):
    session, _, _ = make_session(tmp_path, count=2, durable_first=True)
    session.start_instruction("do two things", ["scene visible"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    status = session.execution_status()
    assert status["required_operation"] == "verify_outcomes"
    assert status["requires_fresh_observation"] is True

    reused = session.verify_outcomes(["outcome-1"], ["obs-1"], [OBSERVED_AT])
    assert "fresh" in reused["error"]
    assert session.verify_outcomes(["outcome-1"], ["obs-final-1"], [OBSERVED_AT])["ok"]
    session.record_action_feedback(feedback(2, affected=("outcome-1",)))

    graph = session.controller.outcome_graph.snapshot()
    first = graph["outcomes"][0]
    assert first["status"] == "invalidated"
    assert first["final_verified"] is False
    assert session.execution_status()["state"] == "recovery_required"


def test_unchanged_content_ref_is_fresh_when_recaptured_later(tmp_path):
    session, _, _ = make_session(tmp_path, count=1, durable_first=True)
    session.start_instruction("close it", ["scene visible"])
    session.record_action_feedback(feedback(1))
    first_capture = (NOW - timedelta(seconds=1)).isoformat()
    session.submit_observation(["same-content-ref"], [first_capture])

    result = session.verify_outcomes(["outcome-1"], ["same-content-ref"], [OBSERVED_AT])

    assert result["ok"] is True
    assert result["contract_success"] is True


def test_pending_final_verification_projects_strict_contract_failure(tmp_path):
    session, _, _ = make_session(tmp_path, count=1, durable_first=True)
    session.start_instruction("do one durable thing", ["scene visible"])
    session.record_action_feedback(feedback(1))

    session.submit_observation(["initial"], [OBSERVED_AT])

    assert session.execution_status()["state"] == "verification_required"
    assert session.execution_status()["contract_success"] is False
    assert session.snapshot()["contract_success"] is False


def test_final_verification_reuses_satisfying_execution_feedback(tmp_path):
    session, _, verifier = make_session(tmp_path, count=1, durable_first=True)
    session.start_instruction("do one durable thing", ["scene visible"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["initial"], [OBSERVED_AT])

    session.verify_outcomes(["outcome-1"], ["final"], [OBSERVED_AT])

    request = verifier.requests[-1]
    assert request.execution_id == "execution-1"
    assert request.feedback is not None
    assert request.feedback.execution_id == "execution-1"


def test_uncertain_final_verification_exhaustion_fails_closed(tmp_path):
    policy = RecoveryPolicy(
        local_replans_per_outcome=0,
        stage_replans_per_stage=0,
        remaining_stage_repartitions_per_run=0,
    )
    session, _, verifier = make_session(
        tmp_path,
        count=1,
        durable_first=True,
        recovery_policy=policy,
    )
    session.start_instruction("do one durable thing", ["scene visible"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["initial"], [OBSERVED_AT])
    verifier.verdict = "uncertain"

    result = session.verify_outcomes(["outcome-1"], ["final"], [OBSERVED_AT])

    assert result["verification"]["verdict"] == "uncertain"
    assert result["state"] == "terminal_failure"
    assert result["contract_success"] is False
    assert session.finish("failure", "final verification budget exhausted")["ok"]


def test_uncertain_final_verification_retries_while_recovery_budget_remains(tmp_path):
    session, _, verifier = make_session(tmp_path, count=1, durable_first=True)
    session.start_instruction("do one durable thing", ["scene visible"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["initial"], [OBSERVED_AT])
    verifier.verdict = "uncertain"

    result = session.verify_outcomes(["outcome-1"], ["final"], [OBSERVED_AT])

    assert result["verification"]["verdict"] == "uncertain"
    assert result["state"] == "verification_required"
    assert result["required_operation"] == "verify_outcomes"


def test_verifier_uncertainty_cannot_request_physical_recovery(tmp_path):
    session, _, verifier = make_session(tmp_path, count=1)
    session.start_instruction("do one thing", ["scene visible"])
    session.record_action_feedback(feedback(1))
    verifier.verdict = "uncertain"
    uncertain = session.submit_observation(["obs-1"], [OBSERVED_AT])
    before = deepcopy(session.controller.recovery_tracker.snapshot())

    rejected = session.request_exception("verifier whitelist issue", ["obs-1"])

    assert uncertain["state"] == "verification_required"
    assert "verification is pending" in rejected["error"]
    assert session.execution_status()["state"] == "verification_required"
    assert session.controller.recovery_tracker.snapshot() == before


def test_stage_runtime_alone_advances_after_observation(tmp_path):
    verifier = Verifier()
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=TwoStageGenerator(),
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier], policy=VerificationPolicy(clock=lambda: NOW)
        ),
    )
    session.start_instruction("two stages", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    assert session.execution_status()["state"] == "observation_required"

    advanced = session.submit_observation(["obs-between-stages"])

    assert advanced["ok"] is True
    assert advanced["state"] == "action_required"
    assert advanced["outcome_id"] == "outcome-2"
    assert controller.stage_runtime.active.id == "stage-2"


def test_stage_boundary_exception_cannot_consume_next_outcome_early(tmp_path):
    verifier = Verifier()
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=TwoStageGenerator(),
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier], policy=VerificationPolicy(clock=lambda: NOW)
        ),
    )
    session.start_instruction("two stages", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    assert session.execution_status()["state"] == "observation_required"
    before = deepcopy(controller.strict_snapshot())

    rejected = session.request_exception("boundary unclear", ["camera"])

    assert "not allowed" in rejected["error"]
    assert controller.strict_snapshot() == before
    assert session.execution_status()["state"] == "observation_required"
    assert "not allowed" in session.apply_recovery("recover", ["camera"])["error"]
    advanced = session.submit_observation(["clear boundary view"])
    assert advanced["ok"] is True
    assert advanced["outcome_id"] == "outcome-2"
    assert controller.stage_runtime.active.id == "stage-2"


def test_exception_cannot_bypass_pending_durable_final_verification(tmp_path):
    session, _, _ = make_session(tmp_path, durable_first=True)
    session.start_instruction("two", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["initial-1"], [OBSERVED_AT])
    assert session.execution_status()["required_operation"] == "verify_outcomes"
    before = deepcopy(session.controller.strict_snapshot())

    rejected = session.request_exception("unclear", ["camera"])

    assert "final verification" in rejected["error"]
    assert session.controller.strict_snapshot() == before
    assert session.verify_outcomes(["outcome-1"], ["final-1"], [OBSERVED_AT])["ok"]
    assert session.execution_status()["outcome_id"] == "outcome-2"


def test_finish_success_guard_uses_final_audit(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("do two things", ["scene visible"])

    rejected = session.finish("success", "premature")

    assert "not allowed" in rejected["error"]


def test_serialized_program_preserves_identity_and_rejects_top_level_drift(tmp_path):
    good = HarnessSession(
        Harness(
            mode="plan-gated",
            trace_path=tmp_path / "good.jsonl",
            physical_tools={"move"},
            plan_generator=SerializedProgramGenerator(),
        )
    )
    started = good.start_instruction("1. complete outcome 1", ["scene"])
    assert started["contract_id"] == good.controller.outcome_program.contract_id

    bad_controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "bad.jsonl",
        physical_tools={"move"},
        plan_generator=SerializedProgramGenerator(mismatch_top_level=True),
    )
    bad = HarnessSession(bad_controller)
    rejected = bad.start_instruction("1. complete outcome 1", ["scene"])
    assert "contract identity" in rejected["error"]
    assert bad.lifecycle_state is None
    assert bad_controller.outcome_graph is None


def test_strict_bootstrap_rejects_multi_outcome_step(tmp_path):
    generator = Generator(count=2)
    original = generator.generate

    def combined(**kwargs):
        generated = original(**kwargs)
        generated["steps"] = [
            {
                "id": "combined",
                "intent": "complete both",
                "allowed_tools": ["move"],
                "covers_outcomes": ["outcome-1", "outcome-2"],
                "stage_id": "stage-1",
            }
        ]
        return generated

    generator.generate = combined
    session = HarnessSession(
        Harness(
            mode="plan-gated",
            trace_path=tmp_path / "trace.jsonl",
            physical_tools={"move"},
            plan_generator=generator,
        )
    )
    rejected = session.start_instruction("two", ["scene"])
    assert "exactly one outcome" in rejected["error"]
    assert session.lifecycle_state is None


def test_three_stage_generation_receives_all_remaining_contract_ids(tmp_path):
    generator = ThreeStageGenerator()
    verifier = Verifier()
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=generator,
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier], policy=VerificationPolicy(clock=lambda: NOW)
        ),
    )
    session.start_instruction("three", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    advanced = session.submit_observation(["between"])

    assert advanced["ok"] is True
    assert generator.received_required_ids == ["outcome-2", "outcome-3"]
    assert generator.received_active_stage_id == "stage-2"
    assert [stage.id for stage in controller.stage_runtime.stages] == [
        "stage-1",
        "stage-2",
        "stage-3",
    ]


def test_local_outcome_replan_preserves_later_step_tail(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    session.request_exception("blocked", ["blocked"])
    assert session.apply_recovery("recover", ["clear"])["ok"]
    assert [step["id"] for step in session.controller.state.view_plan()["steps"]] == [
        "repair-step",
        "step-2",
    ]


def test_recovery_post_generation_failure_rolls_back_all_domains(tmp_path, monkeypatch):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    session.request_exception("blocked", ["blocked"])
    before = deepcopy(session.controller.strict_snapshot())

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("graph update failed")

    monkeypatch.setattr(
        session.controller.outcome_graph, "prepare_recovery", fail_prepare
    )
    result = session.apply_recovery("recover", ["clear"])
    assert "graph update failed" in result["error"]
    assert session.execution_status()["state"] == "recovery_required"
    assert session.controller.strict_snapshot() == before


def test_finish_normalizes_failed_status(tmp_path):
    policy = RecoveryPolicy(
        local_replans_per_outcome=0,
        stage_replans_per_stage=0,
        remaining_stage_repartitions_per_run=0,
    )
    session, _, _ = make_session(tmp_path, recovery_policy=policy)
    session.start_instruction("two", ["scene"])
    session.request_exception("blocked", ["blocked"])
    session.apply_recovery("recover", ["blocked"])
    assert session.finish("failed", "done")["status"] == "failure"


def test_physical_action_preflight_fails_closed_when_budget_is_zero(tmp_path):
    session, _, _ = make_session(
        tmp_path,
        recovery_policy=RecoveryPolicy(max_total_physical_actions_per_run=0),
    )
    session.start_instruction("two", ["scene"])
    result = session.physical_action_allowed("move")
    assert result["allowed"] is False
    assert session.execution_status()["state"] == "terminal_failure"


def test_physical_action_lease_reserves_last_budget_and_matches_feedback(tmp_path):
    session, _, _ = make_session(
        tmp_path,
        recovery_policy=RecoveryPolicy(max_total_physical_actions_per_run=1),
    )
    session.start_instruction("two", ["scene"])
    allowed = session.physical_action_allowed("move", "execution-1")
    assert allowed["allowed"] is True
    assert allowed["plan_step_id"] == "step-1"
    assert session.physical_action_allowed("move", "execution-2")["allowed"] is False
    mismatch = session.record_action_feedback(feedback(2))
    assert "lease" in mismatch["error"]
    assert session.record_action_feedback(feedback(1))["ok"] is True
    assert (
        session.controller.recovery_tracker.snapshot()["physical_actions_consumed"] == 1
    )


def test_invalid_feedback_clears_lease_but_keeps_reserved_budget(tmp_path):
    session, _, _ = make_session(
        tmp_path,
        recovery_policy=RecoveryPolicy(max_total_physical_actions_per_run=1),
    )
    session.start_instruction("two", ["scene"])
    session.physical_action_allowed("move", "execution-1")
    invalid = session.record_action_feedback(feedback(1, affected=("outcome-missing",)))
    assert "unknown affected" in invalid["error"]
    assert session.execution_status()["state"] == "recovery_required"
    blocked = session.physical_action_allowed("move", "execution-2")
    assert blocked["allowed"] is False


def test_pending_lease_blocks_exception_until_runtime_aborts_fail_closed(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    assert session.physical_action_allowed("move", "execution-1")["allowed"] is True
    before = deepcopy(session.controller.strict_snapshot())

    blocked = session.request_exception("feedback lost", ["transport timeout"])

    assert "lease" in blocked["error"]
    assert session.execution_status()["state"] == "action_required"
    assert session.controller.strict_snapshot() == before
    assert "not allowed" in session.apply_recovery("recover", ["scene"])["error"]
    wrong = session.abort_physical_action(
        "execution-other", "feedback lost", ["transport timeout"]
    )
    assert "lease" in wrong["error"]
    aborted = session.abort_physical_action(
        "execution-1", "feedback lost", ["transport timeout"]
    )
    assert aborted["ok"] is True
    assert aborted["state"] == "recovery_required"
    assert session.apply_recovery("recover", ["scene"])["ok"] is True
    assert session.physical_action_allowed("move", "execution-2")["allowed"] is True


def test_direct_legacy_mutator_is_blocked_after_strict_start(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    before = deepcopy(session.controller.strict_snapshot())
    result = session.controller.verify_step("pass", ["invented"], "bypass")
    assert "strict session" in result["error"]
    assert session.controller.strict_snapshot() == before


def test_batch_final_verification_is_atomic_when_second_is_uncertain(tmp_path):
    session, generator, verifier = make_session(tmp_path)
    generator.all_durable = True
    session.start_instruction("two durable", ["scene"])
    for index in (1, 2):
        session.record_action_feedback(feedback(index))
        session.submit_observation([f"obs-{index}"], [OBSERVED_AT])
        session.verify_outcomes(
            [f"outcome-{index}"], [f"initial-final-{index}"], [OBSERVED_AT]
        )
    session.controller.outcome_graph.invalidate("outcome-1", reason="later change")
    session.controller.outcome_graph.invalidate("outcome-2", reason="later change")
    session._pending_final_outcome_ids = ("outcome-1", "outcome-2")
    session._lifecycle_state = HarnessLifecycleState.VERIFICATION_REQUIRED
    verifier.by_outcome["outcome-2"] = "uncertain"
    before = deepcopy(session.controller.outcome_graph.snapshot())

    result = session.verify_outcomes(
        ["outcome-1", "outcome-2"],
        ["final-1", "final-2"],
        [OBSERVED_AT, OBSERVED_AT],
    )
    assert result["verification"]["verdict"] == "uncertain"
    assert session.controller.outcome_graph.snapshot() == before


def test_durable_negative_final_evidence_enters_recovery_with_provenance(tmp_path):
    session, _, verifier = make_session(tmp_path, durable_first=True)
    session.start_instruction("two", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["initial"], [OBSERVED_AT])
    verifier.verdict = "not_satisfied"

    result = session.verify_outcomes(["outcome-1"], ["negative"], [OBSERVED_AT])

    assert result["verification"]["verdict"] == "not_satisfied"
    assert session.execution_status()["state"] == "recovery_required"
    outcome = session.controller.outcome_graph.snapshot()["outcomes"][0]
    assert outcome["status"] == "recovery_ready"
    assert outcome["evidence"][-1]["observation_id"] == "negative"
    assert outcome["evidence"][-1]["captured_at"] == OBSERVED_AT

    recovered = session.apply_recovery("restore outcome one", ["target visible"])
    assert recovered["ok"] is True
    assert recovered["outcome_id"] == "outcome-1"
    assert session.execution_status()["outcome_id"] == "outcome-1"
    assert [step["id"] for step in session.controller.state.view_plan()["steps"]] == [
        "step-1",
        "repair-step",
        "step-2",
    ]


def test_malformed_verification_request_is_conservative_and_non_mutating(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    session.record_action_feedback(feedback(1))
    before = deepcopy(session.controller.strict_snapshot())
    result = session.submit_observation(["obs"], ["not-a-time"])
    assert "error" in result
    assert session.controller.strict_snapshot() == before


def test_empty_recovery_reason_is_non_mutating(tmp_path):
    session, _, _ = make_session(tmp_path)
    session.start_instruction("two", ["scene"])
    session.request_exception("blocked", ["blocked"])
    before = deepcopy(session.controller.strict_snapshot())
    result = session.apply_recovery("", ["clear"])
    assert "reason" in result["error"]
    assert session.controller.strict_snapshot() == before


def test_forward_dependency_bootstrap_failure_is_atomic(tmp_path):
    generator = Generator(count=2)
    original = generator.generate

    def forward(**kwargs):
        generated = original(**kwargs)
        generated["outcomes"][0]["depends_on"] = ["outcome-2"]
        generated["outcomes"][1]["depends_on"] = []
        return generated

    generator.generate = forward
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=generator,
    )
    session = HarnessSession(controller)
    result = session.start_instruction("two", ["scene"])
    assert "ready outcome" in result["error"]
    assert session.lifecycle_state is None
    assert controller.outcome_graph is None
    assert controller.state.has_plan is False


def test_independent_durable_invalidation_enters_recovery_without_empty_target(
    tmp_path,
):
    session, generator, _ = make_session(tmp_path, durable_first=True)
    original = generator.generate

    def independent(**kwargs):
        generated = original(**kwargs)
        generated["outcomes"][1]["depends_on"] = []
        return generated

    generator.generate = independent
    session.start_instruction("two", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-1"], ["final-1"], [OBSERVED_AT])
    session.record_action_feedback(feedback(2, affected=("outcome-1",)))
    status = session.execution_status()
    assert status["state"] == "recovery_required"
    assert status.get("outcome_id") is not None or "outcome_id" not in status

    recovered = session.apply_recovery("restore outcome one", ["target visible"])
    assert recovered["ok"] is True
    assert recovered["outcome_id"] == "outcome-1"
    session.record_action_feedback(feedback(3))
    session.submit_observation(["restore-1"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-1"], ["restore-final-1"], [OBSERVED_AT])
    assert session.execution_status()["outcome_id"] == "outcome-2"


def test_later_stage_forward_dependency_rejects_advance_without_mutation(tmp_path):
    generator = ThreeStageGenerator()
    original = generator.generate

    def forward(**kwargs):
        generated = original(**kwargs)
        generated["outcomes"][1]["depends_on"] = ["outcome-3"]
        generated["outcomes"][2]["depends_on"] = []
        return generated

    generator.generate = forward
    verifier = Verifier()
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=generator,
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier], policy=VerificationPolicy(clock=lambda: NOW)
        ),
    )
    session.start_instruction("three", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    before = deepcopy(controller.strict_snapshot())

    rejected = session.submit_observation(["between"])

    assert "ready" in rejected["error"]
    assert controller.strict_snapshot() == before
    assert session.execution_status()["state"] == "observation_required"


def test_outcome_program_object_contract_id_is_recomputed(tmp_path):
    generator = SerializedProgramGenerator()
    original = generator.generate

    def forged(**kwargs):
        generated = original(**kwargs)
        valid = compile_outcome_program(
            kwargs["goal"],
            list(generated["outcome_program"]["outcomes"]),
            list(generated["outcome_program"]["stages"]),
            explicit_contract=[
                {"id": "outcome-1", "number": 1, "text": "complete outcome 1"}
            ],
        )
        return {
            "goal": kwargs["goal"],
            "strategy": generated["strategy"],
            "outcome_program": replace(valid, contract_id="contract-forged"),
            "steps": generated["steps"],
        }

    generator.generate = forged
    session = HarnessSession(
        Harness(
            mode="plan-gated",
            trace_path=tmp_path / "trace.jsonl",
            physical_tools={"move"},
            plan_generator=generator,
        )
    )
    rejected = session.start_instruction("1. complete outcome 1", ["scene"])
    assert "contract identity" in rejected["error"]
    assert session.lifecycle_state is None


def test_bootstrap_restores_state_when_registration_raises_after_mutation(
    tmp_path, monkeypatch
):
    session, _, _ = make_session(tmp_path)
    real_register = session.controller.state.register_plan

    def mutate_then_raise(*args, **kwargs):
        result = real_register(*args, **kwargs)
        assert result["ok"]
        raise OSError("trace write failed")

    monkeypatch.setattr(session.controller.state, "register_plan", mutate_then_raise)
    rejected = session.start_instruction("two", ["scene"])
    assert "trace write failed" in rejected["error"]
    assert session.lifecycle_state is None
    assert session.controller.state.has_plan is False
    assert session.controller.outcome_graph is None


def test_multiple_invalidated_durable_outcomes_recover_before_suspended_action(
    tmp_path,
):
    session, generator, _ = make_session(tmp_path, count=3)
    original = generator.generate

    def durable_independent(**kwargs):
        generated = original(**kwargs)
        for outcome in generated["outcomes"]:
            outcome["depends_on"] = []
        generated["outcomes"][0]["lifecycle"] = "persistent"
        generated["outcomes"][1]["lifecycle"] = "persistent"
        return generated

    generator.generate = durable_independent
    session.start_instruction("three", ["scene"])
    for index in (1, 2):
        session.record_action_feedback(feedback(index))
        session.submit_observation([f"obs-{index}"], [OBSERVED_AT])
        session.verify_outcomes([f"outcome-{index}"], [f"final-{index}"], [OBSERVED_AT])
    session.record_action_feedback(feedback(3, affected=("outcome-1", "outcome-2")))

    first = session.apply_recovery("restore first", ["visible"])
    assert first["outcome_id"] == "outcome-1"
    session.record_action_feedback(feedback(4))
    session.submit_observation(["restore-1"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-1"], ["restore-final-1"], [OBSERVED_AT])
    assert session.execution_status()["state"] == "recovery_required"

    second = session.apply_recovery("restore second", ["visible"])
    assert second["outcome_id"] == "outcome-2"
    session.record_action_feedback(feedback(5))
    session.submit_observation(["restore-2"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-2"], ["restore-final-2"], [OBSERVED_AT])
    assert session.execution_status()["outcome_id"] == "outcome-3"


@pytest.mark.parametrize(
    ("policy", "expected_scope"),
    [
        (
            RecoveryPolicy(
                local_replans_per_outcome=0,
                stage_replans_per_stage=1,
                remaining_stage_repartitions_per_run=0,
            ),
            "replan_stage",
        ),
        (
            RecoveryPolicy(
                local_replans_per_outcome=0,
                stage_replans_per_stage=0,
                remaining_stage_repartitions_per_run=1,
            ),
            "repartition_remaining_stages",
        ),
    ],
)
def test_cross_stage_recovery_escalation_preserves_current_stage_tail(
    tmp_path, policy, expected_scope
):
    generator = CrossStageRecoveryGenerator()
    verifier = Verifier()
    controller = Harness(
        mode="plan-gated",
        trace_path=tmp_path / "trace.jsonl",
        physical_tools={"move"},
        plan_generator=generator,
        require_execution_id=True,
    )
    session = HarnessSession(
        controller,
        verification_coordinator=VerificationCoordinator(
            [verifier], policy=VerificationPolicy(clock=lambda: NOW)
        ),
        recovery_policy=policy,
    )
    session.start_instruction("two stages", ["scene"])
    session.record_action_feedback(feedback(1))
    session.submit_observation(["obs-1"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-1"], ["final-1"], [OBSERVED_AT])
    session.submit_observation(["between"])
    session.record_action_feedback(feedback(2, affected=("outcome-1",)))

    recovered = session.apply_recovery("restore old stage", ["visible"])

    assert recovered["ok"] is True
    assert recovered["scope"] == expected_scope
    assert recovered["outcome_id"] == "outcome-1"
    assert controller.stage_runtime.active.id == "stage-2"
    assert [step["id"] for step in controller.state.view_plan()["steps"]] == [
        "restore-step-1",
        "step-2",
    ]
    session.record_action_feedback(feedback(3))
    session.submit_observation(["restored"], [OBSERVED_AT])
    session.verify_outcomes(["outcome-1"], ["restored-final"], [OBSERVED_AT])
    assert session.execution_status()["outcome_id"] == "outcome-2"
    assert controller.stage_runtime.active.id == "stage-2"


def test_six_outcome_authoritative_loop(tmp_path):
    session, _, _ = make_session(tmp_path, count=6)
    session.start_instruction("complete six outcomes", ["scene visible"])

    for index in range(1, 7):
        assert session.execution_status()["state"] == "action_required"
        assert session.record_action_feedback(feedback(index))["ok"]
        assert session.submit_observation([f"obs-{index}"], [OBSERVED_AT])["ok"]

    assert session.execution_status()["contract_success"] is True
