"""Tests for conservative, provider-independent predicate verification."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rpent.harness.verification import (
    PredicateVerificationRequest,
    PredicateVerificationResult,
    VerificationCoordinator,
    VerificationPolicy,
)

NOW = datetime(2026, 8, 4, 0, 1, tzinfo=timezone.utc)


def _request(*, observed_at: tuple[str, ...] = ("2026-08-04T00:00:55+00:00",)):
    return PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="cup is in drawer",
        execution_id="move-1",
        observation_refs=("obs-1",),
        observed_at=observed_at,
        feedback=None,
    )


class FixedVerifier:
    name = "fixed"
    version = "1"

    def __init__(self, verdict: str = "satisfied", *, authoritative: bool = True):
        self.verdict = verdict
        self.authoritative = authoritative

    def verify(self, request):
        return PredicateVerificationResult(
            verdict=self.verdict,
            evidence_refs=request.observation_refs,
            verifier=self.name,
            verifier_version=self.version,
            confidence=0.99,
            authoritative=self.authoritative,
            reason="predicate evaluated",
        )


class ForeignEvidenceVerifier(FixedVerifier):
    def verify(self, request):
        result = super().verify(request)
        return PredicateVerificationResult(
            **{**result.__dict__, "evidence_refs": ("not-whitelisted",)}
        )


class FirstEvidenceOnlyVerifier(FixedVerifier):
    def verify(self, request):
        result = super().verify(request)
        return PredicateVerificationResult(
            **{**result.__dict__, "evidence_refs": request.observation_refs[:1]}
        )


class FinalEvidenceOnlyVerifier(FixedVerifier):
    def verify(self, request):
        result = super().verify(request)
        return PredicateVerificationResult(
            **{**result.__dict__, "evidence_refs": request.observation_refs[-1:]}
        )


class BrokenVerifier:
    name = "broken"
    version = "1"

    def verify(self, request):
        raise RuntimeError("adapter unavailable")


class SelfAuthorizingVerifier(FixedVerifier):
    def verify(self, request):
        result = super().verify(request)
        return PredicateVerificationResult(**{**result.__dict__, "authorized": True})


def _policy() -> VerificationPolicy:
    return VerificationPolicy(
        max_observation_age_s=10,
        min_confidence=0.8,
        clock=lambda: NOW,
    )


def test_coordinator_authorizes_only_fresh_authoritative_consensus():
    result = VerificationCoordinator([FixedVerifier()], policy=_policy()).verify(
        _request()
    )

    assert result.verdict == "satisfied"
    assert result.authoritative is True
    assert result.authorized is True


def test_slow_verifier_cannot_make_timely_observation_stale():
    current = NOW

    class SlowVerifier(FixedVerifier):
        def verify(self, request):
            nonlocal current
            current = datetime(2026, 8, 4, 0, 2, tzinfo=timezone.utc)
            return super().verify(request)

    policy = VerificationPolicy(
        max_observation_age_s=10,
        min_confidence=0.8,
        clock=lambda: current,
    )

    result = VerificationCoordinator([SlowVerifier()], policy=policy).verify(_request())

    assert result.authorized is True


@pytest.mark.parametrize(
    ("verifiers", "verification_request"),
    [
        ([FixedVerifier(authoritative=False)], _request()),
        ([FixedVerifier("uncertain")], _request()),
        (
            [FixedVerifier("satisfied"), FixedVerifier("not_satisfied")],
            _request(),
        ),
        (
            [FixedVerifier()],
            _request(observed_at=("2026-08-04T00:00:00+00:00",)),
        ),
    ],
    ids=("advisory", "uncertain", "conflicting", "stale"),
)
def test_non_authorizing_evidence_fails_closed(verifiers, verification_request):
    result = VerificationCoordinator(verifiers, policy=_policy()).verify(
        verification_request
    )

    assert result.authorized is False
    assert result.authoritative is False


def test_coordinator_without_verifiers_is_not_applicable_and_unauthorized():
    result = VerificationCoordinator([], policy=_policy()).verify(_request())

    assert result.verdict == "not_applicable"
    assert result.authorized is False


def test_coordinator_never_trusts_verifier_supplied_authorization():
    result = VerificationCoordinator([SelfAuthorizingVerifier()]).verify(_request())

    assert result.verdict == "satisfied"
    assert result.authoritative is True
    assert result.authorized is False


def test_verifier_exception_fails_closed_as_uncertain():
    result = VerificationCoordinator([BrokenVerifier()], policy=_policy()).verify(
        _request()
    )

    assert result.verdict == "uncertain"
    assert result.authorized is False
    assert "failed" in result.reason


def test_advisory_contradiction_prevents_authoritative_success():
    result = VerificationCoordinator(
        [
            FixedVerifier("satisfied"),
            FixedVerifier("not_satisfied", authoritative=False),
        ],
        policy=_policy(),
    ).verify(_request())

    assert result.verdict == "uncertain"
    assert result.authorized is False
    assert "conflicting" in result.reason


def test_result_may_only_cite_request_whitelisted_evidence():
    result = VerificationCoordinator(
        [ForeignEvidenceVerifier()], policy=_policy()
    ).verify(_request())

    assert result.verdict == "uncertain"
    assert result.authorized is False
    assert "whitelist" in result.reason


def test_uncertain_adapter_prevents_success_authorization():
    result = VerificationCoordinator(
        [FixedVerifier("satisfied"), FixedVerifier("uncertain")], policy=_policy()
    ).verify(_request())

    assert result.verdict == "uncertain"
    assert result.authorized is False


def test_authoritative_failure_overrides_uncertain_advisory_without_authorizing():
    result = VerificationCoordinator(
        [FixedVerifier("not_satisfied"), FixedVerifier("uncertain")], policy=_policy()
    ).verify(_request())

    assert result.verdict == "not_satisfied"
    assert result.authoritative is True
    assert result.authorized is False


def test_fresh_unreferenced_context_cannot_refresh_stale_cited_evidence():
    request = PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="cup is in drawer",
        execution_id="move-1",
        observation_refs=("stale.png", "fresh.png"),
        observed_at=(
            "2026-08-04T00:00:00+00:00",
            "2026-08-04T00:00:55+00:00",
        ),
        feedback=None,
    )

    result = VerificationCoordinator(
        [FirstEvidenceOnlyVerifier()], policy=_policy()
    ).verify(request)

    assert result.verdict == "uncertain"
    assert result.authorized is False
    assert "stale" in result.reason


def test_fresh_cited_evidence_cannot_authorize_a_result_that_also_cites_stale_evidence():
    request = PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="cup is in drawer",
        execution_id="move-1",
        observation_refs=("stale.png", "fresh.png"),
        observed_at=(
            "2026-08-04T00:00:00+00:00",
            "2026-08-04T00:00:55+00:00",
        ),
        feedback=None,
    )

    result = VerificationCoordinator([FixedVerifier()], policy=_policy()).verify(
        request
    )

    assert result.verdict == "uncertain"
    assert result.authorized is False
    assert "stale" in result.reason


def test_policy_accepts_historical_context_when_final_observation_is_fresh():
    request = PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="cup is in drawer",
        execution_id="move-1",
        observation_refs=("initial.png", "final.png"),
        observed_at=(
            "2026-08-04T00:00:00+00:00",
            "2026-08-04T00:00:55+00:00",
        ),
        feedback=None,
    )

    validated = _policy().validate(request, FinalEvidenceOnlyVerifier().verify(request))

    assert validated.authorized is True


def test_request_rejects_mismatched_observation_timestamps():
    with pytest.raises(ValueError, match="one timestamp"):
        _request(observed_at=())


@pytest.mark.parametrize(
    "timestamp",
    ("2026-08-04T00:00:55", "not-a-timestamp"),
)
def test_request_rejects_naive_or_invalid_observation_timestamps(timestamp):
    with pytest.raises(ValueError, match="timezone-aware"):
        _request(observed_at=(timestamp,))


def test_policy_rejects_naive_clock():
    policy = VerificationPolicy(clock=lambda: datetime(2026, 8, 4, 0, 1))

    with pytest.raises(ValueError, match="timezone-aware"):
        policy.validate(_request(), FixedVerifier().verify(_request()))


def test_result_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        PredicateVerificationResult(
            verdict="satisfied",
            evidence_refs=("obs-1",),
            verifier="fixed",
            verifier_version="1",
            confidence=1.1,
            authoritative=True,
            reason="invalid",
        )
