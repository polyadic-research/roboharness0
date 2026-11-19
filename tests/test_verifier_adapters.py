"""Contract tests for built-in predicate verifier adapters."""

from rpent.harness.verification import PredicateVerificationRequest
from rpent.harness.verifiers import (
    AdvisoryVerifier,
    EnvironmentPredicateVerifier,
    StructuredControllerVerifier,
)
from rpent.planning.schemas import ActionFeedback, SkillOutcome


def _request(
    *,
    outcome_id="outcome-1",
    success=False,
    terminated=False,
    truncated=False,
    status="success",
    execution_status="completed",
    request_execution_id="exec-1",
):
    feedback = ActionFeedback(
        execution_id="exec-1",
        tool_name="move",
        physical=True,
        outcome=SkillOutcome(
            status=status,
            evidence=("structured predicate passed",),
            source="adapter",
        ),
        execution_status=execution_status,
        environment_success=success,
        environment_terminated=terminated,
        environment_truncated=truncated,
    )
    return PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id=outcome_id,
        success_predicate="target predicate",
        execution_id=request_execution_id,
        observation_refs=("obs-1",),
        observed_at=("2026-08-04T00:00:00+00:00",),
        feedback=feedback,
    )


def test_environment_predicate_is_authoritative_only_for_bound_outcome():
    verifier = EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-2"})

    unbound = verifier.verify(_request(success=True))
    bound = verifier.verify(_request(outcome_id="outcome-2", success=True))

    assert unbound.verdict == "not_applicable"
    assert unbound.authoritative is False
    assert bound.verdict == "satisfied"
    assert bound.authoritative is True


def test_environment_predicate_can_bind_all_outcomes_explicitly():
    result = EnvironmentPredicateVerifier(bound_outcome_ids=None).verify(
        _request(outcome_id="generated-outcome", success=True)
    )

    assert result.verdict == "satisfied"
    assert result.authoritative is True


def test_generic_termination_is_not_authoritative_success():
    result = EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-1"}).verify(
        _request(success=False, terminated=True)
    )

    assert result.verdict != "satisfied"
    assert result.authoritative is False


def test_generic_truncation_is_not_authoritative_success():
    result = EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-1"}).verify(
        _request(success=False, truncated=True)
    )

    assert result.verdict != "satisfied"
    assert result.authoritative is False


def test_bound_live_environment_failure_is_authoritative_not_satisfied():
    result = EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-1"}).verify(
        _request(success=False)
    )

    assert result.verdict == "not_satisfied"
    assert result.authoritative is True
    assert result.confidence == 1.0
    assert result.evidence_refs == ("obs-1",)


def test_environment_success_requires_current_execution_binding():
    result = EnvironmentPredicateVerifier(bound_outcome_ids={"outcome-1"}).verify(
        _request(success=True, request_execution_id=None)
    )

    assert result.verdict == "not_applicable"
    assert result.authoritative is False


def test_structured_controller_requires_explicit_outcome_binding():
    verifier = StructuredControllerVerifier(bound_outcome_ids={"outcome-2"})

    unbound = verifier.verify(_request())
    bound = verifier.verify(_request(outcome_id="outcome-2"))

    assert unbound.authoritative is False
    assert bound.verdict == "satisfied"
    assert bound.authoritative is True


def test_structured_controller_does_not_promote_uncertain_skill_outcome():
    result = StructuredControllerVerifier(bound_outcome_ids={"outcome-1"}).verify(
        _request(status="uncertain")
    )

    assert result.verdict == "uncertain"
    assert result.authoritative is False


def test_controller_feedback_must_be_completed_and_execution_bound():
    verifier = StructuredControllerVerifier(bound_outcome_ids={"outcome-1"})

    incomplete = verifier.verify(_request(execution_status="controller_error"))
    mismatched = verifier.verify(_request(request_execution_id="exec-other"))

    assert incomplete.authoritative is False
    assert mismatched.authoritative is False


def test_advisory_verifier_never_returns_authoritative_result():
    verifier = AdvisoryVerifier(lambda request: ("satisfied", 0.95, "looks done"))

    result = verifier.verify(_request())

    assert result.verdict == "satisfied"
    assert result.authoritative is False
