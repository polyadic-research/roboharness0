"""Provider-independent predicate verification contracts and coordination."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Literal, Protocol, Sequence

from rpent.planning.schemas import ActionFeedback

PredicateVerdict = Literal[
    "satisfied",
    "not_satisfied",
    "uncertain",
    "not_applicable",
]
_DECISIVE_VERDICTS = {"satisfied", "not_satisfied"}
_VALID_VERDICTS = _DECISIVE_VERDICTS | {"uncertain", "not_applicable"}


def _parse_observation_time(raw: str) -> datetime:
    try:
        captured = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "observation timestamp must be timezone-aware ISO-8601"
        ) from exc
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError("observation timestamp must be timezone-aware")
    return captured


@dataclass(frozen=True)
class PredicateVerificationRequest:
    """Everything a verifier may use to evaluate one frozen predicate."""

    run_id: str
    contract_id: str
    outcome_id: str
    success_predicate: str
    execution_id: str | None
    observation_refs: tuple[str, ...]
    observed_at: tuple[str, ...]
    feedback: ActionFeedback | None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "contract_id", "outcome_id", "success_predicate"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if len(self.observation_refs) != len(self.observed_at):
            raise ValueError("each observation requires exactly one timestamp")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.observation_refs
        ):
            raise ValueError("observation refs must be non-empty strings")
        if len(self.observation_refs) != len(set(self.observation_refs)):
            raise ValueError("observation refs must be unique")
        for timestamp in self.observed_at:
            _parse_observation_time(timestamp)


@dataclass(frozen=True)
class PredicateVerificationResult:
    """One verifier's evidence-backed predicate decision."""

    verdict: PredicateVerdict
    evidence_refs: tuple[str, ...]
    verifier: str
    verifier_version: str
    confidence: float | None
    authoritative: bool
    reason: str
    authorized: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(f"invalid predicate verdict: {self.verdict!r}")
        for field_name in ("verifier", "verifier_version", "reason"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be a finite number between 0 and 1")
        if self.authorized and not (self.authoritative and self.verdict == "satisfied"):
            raise ValueError("only authoritative satisfaction can authorize success")


class PredicateVerifier(Protocol):
    name: str
    version: str

    def verify(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult: ...


@dataclass(frozen=True)
class VerificationPolicy:
    """Freshness and confidence requirements for predicate evidence."""

    max_observation_age_s: float = 120.0
    min_confidence: float = 0.8
    clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc), compare=False, repr=False
    )
    future_tolerance_s: float = 5.0

    def __post_init__(self) -> None:
        numeric = (
            self.max_observation_age_s,
            self.min_confidence,
            self.future_tolerance_s,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("verification policy values must be finite numbers")
        if self.max_observation_age_s < 0 or self.future_tolerance_s < 0:
            raise ValueError("verification freshness limits must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    def validate(
        self,
        request: PredicateVerificationRequest,
        result: PredicateVerificationResult,
        *,
        as_of: datetime | None = None,
    ) -> PredicateVerificationResult:
        """Return a result marked authorized only after all checks pass."""
        now = self.clock() if as_of is None else as_of
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("verification clock must be timezone-aware")
        timestamps = tuple(
            _parse_observation_time(value) for value in request.observed_at
        )
        if not timestamps:
            raise ValueError("verification requires fresh observation evidence")
        ages_by_ref = {
            ref: (now - captured).total_seconds()
            for ref, captured in zip(request.observation_refs, timestamps, strict=True)
        }
        ages = tuple(ages_by_ref.values())
        if any(age < -self.future_tolerance_s for age in ages):
            raise ValueError("observation timestamp is in the future")
        if not any(age <= self.max_observation_age_s for age in ages):
            raise ValueError("observation evidence is stale")
        if result.authoritative and result.verdict in _DECISIVE_VERDICTS:
            if not result.evidence_refs:
                raise ValueError("authoritative result requires cited evidence")
            if not set(result.evidence_refs).issubset(request.observation_refs):
                raise ValueError("result evidence is outside the request whitelist")
            if any(
                ages_by_ref[ref] > self.max_observation_age_s
                for ref in result.evidence_refs
            ):
                raise ValueError("authoritative result cites stale evidence")
            if result.confidence is None or result.confidence < self.min_confidence:
                raise ValueError("authoritative result is below confidence threshold")
        authorized = result.authoritative and result.verdict == "satisfied"
        return replace(result, authorized=authorized)


class VerificationCoordinator:
    """Combine verifier results deterministically and fail closed."""

    def __init__(
        self,
        verifiers: Sequence[PredicateVerifier],
        *,
        policy: VerificationPolicy | None = None,
    ) -> None:
        self._verifiers = tuple(verifiers)
        self._policy = policy

    def verify(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult:
        as_of = self._policy.clock() if self._policy is not None else None
        combined = self._combine(request)
        if self._policy is None:
            return combined
        try:
            return self._policy.validate(request, combined, as_of=as_of)
        except ValueError as exc:
            return _coordinator_result(
                "uncertain",
                f"verification policy rejected evidence: {exc}",
                evidence_refs=combined.evidence_refs,
            )

    def _combine(
        self, request: PredicateVerificationRequest
    ) -> PredicateVerificationResult:
        if not self._verifiers:
            return _coordinator_result(
                "not_applicable", "no predicate verifier is configured"
            )

        collected: list[PredicateVerificationResult] = []
        for verifier in self._verifiers:
            try:
                result = verifier.verify(request)
                if not isinstance(result, PredicateVerificationResult):
                    raise TypeError("verifier returned an invalid result")
                result = replace(result, authorized=False)
            except Exception as exc:
                name = str(getattr(verifier, "name", type(verifier).__name__))
                result = _coordinator_result(
                    "uncertain", f"predicate verifier {name!r} failed: {exc}"
                )
            collected.append(result)
        results = tuple(collected)
        all_decisive = tuple(
            result for result in results if result.verdict in _DECISIVE_VERDICTS
        )
        if len({result.verdict for result in all_decisive}) > 1:
            refs = tuple(
                dict.fromkeys(
                    ref for result in all_decisive for ref in result.evidence_refs
                )
            )
            return _coordinator_result(
                "uncertain",
                "conflicting predicate results",
                evidence_refs=refs,
            )
        authoritative_failure = next(
            (
                result
                for result in results
                if result.authoritative and result.verdict == "not_satisfied"
            ),
            None,
        )
        if authoritative_failure is not None:
            # A trusted negative cannot authorize success; it may safely move the
            # lifecycle into bounded recovery even when an advisory verifier is
            # uncertain. Conflicting decisive evidence was rejected above.
            return authoritative_failure
        uncertain_results = tuple(
            result for result in results if result.verdict == "uncertain"
        )
        if uncertain_results:
            refs = tuple(
                dict.fromkeys(
                    ref for result in uncertain_results for ref in result.evidence_refs
                )
            )
            return _coordinator_result(
                "uncertain",
                "one or more predicate results are uncertain: "
                + "; ".join(result.reason for result in uncertain_results),
                evidence_refs=refs,
            )
        authoritative = tuple(result for result in results if result.authoritative)
        decisive = tuple(
            result for result in authoritative if result.verdict in _DECISIVE_VERDICTS
        )
        if decisive:
            return decisive[0]
        if authoritative:
            result = authoritative[0]
            return replace(result, authoritative=False, authorized=False)

        advisory_decisive = next(
            (result for result in results if result.verdict in _DECISIVE_VERDICTS),
            None,
        )
        if advisory_decisive is not None:
            return replace(advisory_decisive, authoritative=False, authorized=False)
        uncertain = next(
            (result for result in results if result.verdict == "uncertain"), None
        )
        return replace(uncertain or results[0], authoritative=False, authorized=False)


def _coordinator_result(
    verdict: PredicateVerdict,
    reason: str,
    *,
    evidence_refs: tuple[str, ...] = (),
) -> PredicateVerificationResult:
    return PredicateVerificationResult(
        verdict=verdict,
        evidence_refs=evidence_refs,
        verifier="verification-coordinator",
        verifier_version="1",
        confidence=None,
        authoritative=False,
        reason=reason,
    )


__all__ = [
    "PredicateVerificationRequest",
    "PredicateVerificationResult",
    "PredicateVerifier",
    "PredicateVerdict",
    "VerificationCoordinator",
    "VerificationPolicy",
]
