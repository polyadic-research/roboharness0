"""Independent runtime-backed verification for frozen outcome predicates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any
from uuid import uuid4

from rpent.context.policy import ContextProjection, ProjectedImage
from rpent.harness.evidence import EvidenceResolver, ResolvedEvidence
from rpent.harness.verification import (
    PredicateVerificationRequest,
    PredicateVerificationResult,
)
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("predicate_verification")

_SYSTEM_PROMPT = """You are an independent robot outcome verifier.
Evaluate only the supplied frozen success predicate from the whitelisted raw
artifacts. You did not execute the task and must not trust an execution agent's
claims. The payload's execution_feedback is execution-bound simulator evidence
when present; use only its declared status and environment booleans.
Inspect the artifacts with inspect_evidence. Return exactly one
structured finish verdict. Use satisfied or not_satisfied only when the raw
evidence decisively proves it; otherwise use uncertain. Never infer missing
state and never ask for or perform a physical action."""

_DECISIVE_VERDICTS = {"satisfied", "not_satisfied"}
_STRUCTURED_VERDICTS = _DECISIVE_VERDICTS | {"uncertain"}


class RuntimePredicateVerifier:
    """Evaluate one frozen predicate in an isolated read-only model episode."""

    name = "runtime-predicate"
    version = "1"

    def __init__(
        self,
        runtime: Any,
        *,
        evidence_root: Path,
        model_id: str,
        max_turns: int = 4,
        recorder: Any = None,
    ) -> None:
        if isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._runtime = runtime
        self._resolver = EvidenceResolver(evidence_root)
        self._model_id = str(model_id).strip() or "configured-default"
        self._max_turns = int(max_turns)
        self._recorder = recorder

    def verify(
        self,
        request: PredicateVerificationRequest,
    ) -> PredicateVerificationResult:
        """Run an isolated evidence review and return a fail-closed decision."""
        started = time.monotonic()
        runtime_result = None
        resolved: tuple[ResolvedEvidence, ...] = ()
        preserved: dict[str, dict[str, str]] = {}
        try:
            resolved = self._resolver.resolve_many(list(request.observation_refs))
            toolkit = _PredicateVerificationToolkit(resolved)
            payload = {
                "contract_id": request.contract_id,
                "outcome_id": request.outcome_id,
                "success_predicate": request.success_predicate,
                "evidence": [_evidence_metadata(item) for item in resolved],
            }
            feedback = request.feedback
            if (
                feedback is not None
                and request.execution_id is not None
                and feedback.execution_id == request.execution_id
            ):
                payload["execution_feedback"] = {
                    "execution_id": feedback.execution_id,
                    "execution_status": feedback.execution_status,
                    "environment_success": feedback.environment_success,
                    "environment_terminated": feedback.environment_terminated,
                    "environment_truncated": feedback.environment_truncated,
                }
            user_message = json.dumps(payload, ensure_ascii=False, indent=2)
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="predicate-verifier",
            ) as pool:
                prompts = (
                    _SYSTEM_PROMPT,
                    _SYSTEM_PROMPT
                    + " Your previous response did not invoke the finish tool. A JSON "
                    "object in assistant text is invalid. Retry the identical request "
                    "and invoke the available finish tool exactly once.",
                )
                for candidate_prompt in prompts:
                    runtime_result = pool.submit(
                        copy_context().run,
                        self._runtime.solve,
                        system_prompt=candidate_prompt,
                        user_message=user_message,
                        toolkit=toolkit,
                        max_turns=self._max_turns,
                        input_queue=None,
                    ).result()
                    if (
                        getattr(runtime_result, "error", None)
                        or toolkit.inspection_failed
                        or isinstance(
                            getattr(runtime_result, "finish_result", None), dict
                        )
                    ):
                        break

            runtime_error = getattr(runtime_result, "error", None)
            if runtime_error:
                decision = self._uncertain(str(runtime_error))
            elif toolkit.inspection_failed:
                decision = self._uncertain(
                    "verifier attempted evidence inspection outside the whitelist"
                )
            else:
                finish = getattr(runtime_result, "finish_result", None)
                decision = self._parse_result(
                    finish,
                    resolved,
                    inspected_hashes=toolkit.inspected_hashes,
                )

            stats = getattr(runtime_result, "stats", {})
            if not isinstance(stats, dict):
                stats = {}
            preserved = self._preserve_runtime_artifacts(stats)
        except Exception as exc:
            decision = self._uncertain(
                f"verifier failed closed: {type(exc).__name__}: {exc}"
            )

        stats = getattr(runtime_result, "stats", {})
        if not isinstance(stats, dict):
            stats = {}
        self._record(
            request=request,
            resolved=resolved,
            decision=decision,
            elapsed_s=round(time.monotonic() - started, 3),
            runtime_error=getattr(runtime_result, "error", None),
            runtime_stats=stats,
            preserved_artifacts=preserved,
        )
        return decision

    def _parse_result(
        self,
        payload: object,
        whitelist: tuple[ResolvedEvidence, ...],
        *,
        inspected_hashes: dict[str, str],
    ) -> PredicateVerificationResult:
        if not isinstance(payload, dict) or payload.get("_finish") is not True:
            return self._uncertain("verifier did not submit a structured finish")

        verdict = payload.get("verdict")
        refs = payload.get("evidence_refs")
        confidence = payload.get("confidence")
        reason = payload.get("reason")
        if verdict not in _STRUCTURED_VERDICTS:
            return self._uncertain("verifier returned an invalid verdict")
        if not isinstance(reason, str) or not reason.strip():
            return self._uncertain("verifier returned an empty reason")
        if (
            not isinstance(refs, list)
            or not refs
            or not all(isinstance(ref, str) and ref.strip() for ref in refs)
            or len(refs) != len(set(refs))
        ):
            return self._uncertain("verifier returned invalid evidence refs")
        allowed = {item.ref for item in whitelist}
        if not set(refs).issubset(allowed):
            return self._uncertain(
                "verifier referenced evidence outside the request whitelist"
            )
        if not set(refs).issubset(inspected_hashes):
            return self._uncertain(
                "verifier cited evidence that was not successfully inspected"
            )
        by_ref = {item.ref: item for item in whitelist}
        for ref in refs:
            item = by_ref[ref]
            try:
                current_hash = hashlib.sha256(item.path.read_bytes()).hexdigest()
            except OSError:
                return self._uncertain(
                    f"verifier evidence changed after inspection: {ref}"
                )
            if current_hash != item.sha256 or inspected_hashes[ref] != item.sha256:
                return self._uncertain(
                    f"verifier evidence changed after inspection: {ref}"
                )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return self._uncertain("verifier returned invalid confidence")
        return PredicateVerificationResult(
            verdict=verdict,
            evidence_refs=tuple(refs),
            verifier=f"{self.name}:{self._model_id}",
            verifier_version=self.version,
            confidence=float(confidence),
            authoritative=verdict in _DECISIVE_VERDICTS,
            reason=reason.strip(),
        )

    def _uncertain(self, reason: str) -> PredicateVerificationResult:
        return PredicateVerificationResult(
            verdict="uncertain",
            evidence_refs=(),
            verifier=f"{self.name}:{self._model_id}",
            verifier_version=self.version,
            confidence=None,
            authoritative=False,
            reason=str(reason).strip() or "predicate verification was uncertain",
        )

    def _preserve_runtime_artifacts(
        self,
        stats: dict[str, Any],
    ) -> dict[str, dict[str, str]]:
        candidates = {
            key: stats.get(key)
            for key in ("output_path", "raw_stream_path", "last_message_path")
        }
        if not any(
            isinstance(value, str) and value.strip() for value in candidates.values()
        ):
            return {}

        audit_root = self._resolver.root / "verifier_audit"
        resolved_audit_root = audit_root.resolve(strict=False)
        _require_contained(resolved_audit_root, self._resolver.root)
        audit_root.mkdir(parents=True, exist_ok=True)
        resolved_audit_root = audit_root.resolve(strict=True)
        _require_contained(resolved_audit_root, self._resolver.root)

        preserved: dict[str, dict[str, str]] = {}
        for key, raw_path in candidates.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            try:
                source = Path(raw_path).resolve(strict=True)
            except FileNotFoundError:
                continue
            if not source.is_file():
                continue
            suffix = source.suffix.lower()
            if len(suffix) > 16 or not suffix.removeprefix(".").isalnum():
                suffix = ".artifact"
            target = resolved_audit_root / f"{uuid4().hex}{suffix}"
            _require_contained(target.resolve(strict=False), resolved_audit_root)
            shutil.copy2(source, target)
            preserved[key] = {
                "path": str(target),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        return preserved

    def _record(
        self,
        *,
        request: PredicateVerificationRequest,
        resolved: tuple[ResolvedEvidence, ...],
        decision: PredicateVerificationResult,
        elapsed_s: float,
        runtime_error: str | None,
        runtime_stats: dict[str, Any],
        preserved_artifacts: dict[str, dict[str, str]],
    ) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.record(
                "predicate_verifier_finished",
                contract_id=request.contract_id,
                outcome_id=request.outcome_id,
                verifier=decision.verifier,
                verifier_version=decision.verifier_version,
                model=self._model_id,
                artifacts=[_evidence_metadata(item) for item in resolved],
                verdict=decision.verdict,
                authoritative=decision.authoritative,
                confidence=decision.confidence,
                reason=decision.reason,
                elapsed_s=elapsed_s,
                runtime_error=runtime_error,
                runtime_stats=runtime_stats,
                preserved_artifacts=preserved_artifacts,
            )
        except Exception as exc:
            logger.warning("failed to record predicate verifier event: %s", exc)


def _evidence_metadata(item: ResolvedEvidence) -> dict[str, str]:
    return {
        "ref": item.ref,
        "sha256": item.sha256,
        "captured_at": item.captured_at,
        "media_type": item.media_type,
    }


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("verifier audit path escapes the evidence root") from exc


class _EvidenceContextPolicy:
    """Project verifier evidence without losing its declared image media type."""

    id = "predicate-evidence-v1"

    def project_tool_result(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
        max_text_chars: int,
    ) -> ContextProjection:
        del tool_name
        payload = dict(result)
        raw_image = payload.pop("_evidence_image_bytes", None)
        media_type = str(payload.get("media_type") or "image/png")
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "\n[truncated]"
        images: tuple[ProjectedImage, ...] = ()
        if isinstance(raw_image, bytes) and raw_image:
            images = (
                ProjectedImage(
                    data=raw_image,
                    media_type=media_type,
                    role="predicate-evidence",
                ),
            )
        return ContextProjection(text=text, images=images)


class _PredicateVerificationToolkit(Toolkit):
    """Expose only immutable evidence inspection and structured completion."""

    def __init__(self, whitelist: tuple[ResolvedEvidence, ...]) -> None:
        self._whitelist = {item.ref: item for item in whitelist}
        self.inspection_failed = False
        self.inspected_hashes: dict[str, str] = {}
        super().__init__()
        self.set_context_policy(_EvidenceContextPolicy())
        self.add_tool(
            "inspect_evidence",
            self._inspect_spec(),
            self._inspect_evidence,
        )
        self.add_tool("finish", self._finish_spec(), self._finish)

    def _register_common_tools(self) -> None:
        """Do not expose common filesystem or environment tools."""

    @staticmethod
    def _inspect_spec() -> dict[str, Any]:
        return {
            "name": "inspect_evidence",
            "description": "Inspect one artifact from the request whitelist.",
            "input_schema": {
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _finish_spec() -> dict[str, Any]:
        return {
            "name": "finish",
            "description": "Submit one evidence-grounded predicate verdict.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": sorted(_STRUCTURED_VERDICTS),
                    },
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": [
                    "verdict",
                    "evidence_refs",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }

    def _inspect_evidence(self, ref: str) -> dict[str, Any]:
        item = self._whitelist.get(str(ref))
        if item is None:
            self.inspection_failed = True
            raise ValueError(f"evidence is not in the verification whitelist: {ref}")
        try:
            raw = item.path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item.sha256:
                raise ValueError(
                    f"evidence changed after verification began: {item.ref}"
                )
            result: dict[str, Any] = _evidence_metadata(item)
            if item.media_type.startswith("image/"):
                result["_evidence_image_bytes"] = raw
            else:
                result["content"] = raw.decode("utf-8")
            self.inspected_hashes[item.ref] = item.sha256
            return result
        except Exception:
            self.inspection_failed = True
            raise

    @staticmethod
    def _finish(
        verdict: str,
        evidence_refs: list[str],
        confidence: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "_finish": True,
            "verdict": verdict,
            "evidence_refs": evidence_refs,
            "confidence": confidence,
            "reason": reason,
        }


__all__ = ["RuntimePredicateVerifier"]
