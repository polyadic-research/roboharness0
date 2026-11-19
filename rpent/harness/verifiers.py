"""Built-in predicate verifier adapters."""

from __future__ import annotations

from collections.abc import Callable

from rpent.harness.verification import (
    PredicateVerdict,
    PredicateVerificationRequest,
    PredicateVerificationResult,
)


def _not_applicable(
    name: str, version: str, reason: str
) -> PredicateVerificationResult:
    return PredicateVerificationResult(
        verdict="not_applicable",
        evidence_refs=(),
        verifier=name,
        verifier_version=version,
        confidence=None,
        authoritative=False,
        reason=reason,
    )


class EnvironmentPredicateVerifier:
    """Trust environment success only for explicitly bound outcomes."""

    name = "environment-predicate"
    version = "1"

    def __init__(self, *, bound_outcome_ids: set[str] | None) -> None:
        self._bound_outcome_ids = (
            None if bound_outcome_ids is None else frozenset(bound_outcome_ids)
        )

    def verify(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult:
        feedback = request.feedback
        if (
            self._bound_outcome_ids is not None
            and request.outcome_id not in self._bound_outcome_ids
        ):
            return _not_applicable(
                self.name, self.version, "outcome is not bound to environment predicate"
            )
        if (
            feedback is None
            or request.execution_id is None
            or feedback.execution_status != "completed"
            or feedback.execution_id != request.execution_id
        ):
            return _not_applicable(
                self.name, self.version, "environment feedback is not execution-bound"
            )
        if not feedback.environment_success:
            if (
                not feedback.environment_terminated
                and not feedback.environment_truncated
            ):
                return PredicateVerificationResult(
                    verdict="not_satisfied",
                    evidence_refs=request.observation_refs,
                    verifier=self.name,
                    verifier_version=self.version,
                    confidence=1.0,
                    authoritative=True,
                    reason="bound environment success predicate is not satisfied",
                )
            return _not_applicable(
                self.name,
                self.version,
                "environment success predicate is not satisfied",
            )
        return PredicateVerificationResult(
            verdict="satisfied",
            evidence_refs=request.observation_refs,
            verifier=self.name,
            verifier_version=self.version,
            confidence=1.0,
            authoritative=True,
            reason="bound environment success predicate is satisfied",
        )


class StructuredControllerVerifier:
    """Use outcome-specific structured controller feedback when configured."""

    name = "structured-controller"
    version = "1"

    def __init__(self, *, bound_outcome_ids: set[str]) -> None:
        self._bound_outcome_ids = frozenset(bound_outcome_ids)

    def verify(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult:
        feedback = request.feedback
        if request.outcome_id not in self._bound_outcome_ids:
            return _not_applicable(
                self.name, self.version, "outcome is not bound to controller predicate"
            )
        if feedback is not None and (
            feedback.execution_status != "completed"
            or request.execution_id is None
            or feedback.execution_id != request.execution_id
        ):
            return _not_applicable(
                self.name, self.version, "controller feedback is not execution-bound"
            )
        if feedback is None or feedback.outcome.source != "adapter":
            return _not_applicable(
                self.name, self.version, "structured adapter feedback is unavailable"
            )
        mapping: dict[str, PredicateVerdict] = {
            "success": "satisfied",
            "failure": "not_satisfied",
            "retryable_failure": "not_satisfied",
            "progress": "uncertain",
            "uncertain": "uncertain",
            "not_applicable": "not_applicable",
        }
        verdict = mapping[feedback.outcome.status]
        decisive = verdict in {"satisfied", "not_satisfied"}
        return PredicateVerificationResult(
            verdict=verdict,
            evidence_refs=request.observation_refs,
            verifier=self.name,
            verifier_version=self.version,
            confidence=1.0 if decisive else None,
            authoritative=decisive,
            reason="; ".join(feedback.outcome.evidence)
            or "structured controller result",
        )


class AdvisoryVerifier:
    """Wrap an untrusted reviewer without allowing it to claim authority."""

    name = "advisory-verifier"
    version = "1"

    def __init__(
        self,
        review: Callable[
            [PredicateVerificationRequest], tuple[PredicateVerdict, float | None, str]
        ],
    ) -> None:
        self._review = review

    def verify(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult:
        verdict, confidence, reason = self._review(request)
        return PredicateVerificationResult(
            verdict=verdict,
            evidence_refs=request.observation_refs,
            verifier=self.name,
            verifier_version=self.version,
            confidence=confidence,
            authoritative=False,
            reason=reason,
        )


__all__ = [
    "AdvisoryVerifier",
    "EnvironmentPredicateVerifier",
    "StructuredControllerVerifier",
]
