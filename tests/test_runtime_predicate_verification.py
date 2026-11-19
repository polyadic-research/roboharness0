"""Tests for the isolated runtime-backed predicate verifier."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from rpent.harness.evidence import EvidenceResolver
from rpent.harness.verification import PredicateVerificationRequest
from rpent.planning.events import JsonlEventRecorder
from rpent.planning.schemas import ActionFeedback
from rpent.runtime import RuntimeResult
from rpent.runtime.predicate_verification import RuntimePredicateVerifier


def _artifact(tmp_path: Path, name: str = "final.json") -> Path:
    artifact = tmp_path / name
    if artifact.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        artifact.write_bytes(b"image-payload")
    else:
        artifact.write_text('{"supported": true}\n', encoding="utf-8")
    return artifact


def _request(path: Path) -> PredicateVerificationRequest:
    item = EvidenceResolver(path.parent).resolve_many([path.name])[0]
    return PredicateVerificationRequest(
        run_id="run-1",
        contract_id="contract-1",
        outcome_id="outcome-1",
        success_predicate="the object is supported",
        execution_id=None,
        observation_refs=(path.name,),
        observed_at=(item.captured_at,),
        feedback=None,
    )


def _finish(
    *,
    verdict: str = "satisfied",
    evidence_refs: object = None,
    confidence: object = 0.97,
    reason: object = "support relation is visible",
) -> dict:
    return {
        "_finish": True,
        "verdict": verdict,
        "evidence_refs": ["final.json"] if evidence_refs is None else evidence_refs,
        "confidence": confidence,
        "reason": reason,
    }


class InspectingRuntime:
    def __init__(
        self,
        finish: dict | None,
        *,
        error: str | None = None,
        inspected_ref: str = "final.json",
        output_path: Path | None = None,
    ) -> None:
        self.finish = finish
        self.error = error
        self.inspected_ref = inspected_ref
        self.output_path = output_path
        self.tool_names: set[str] = set()
        self.tool_specs: dict[str, dict] = {}
        self.inspected = None
        self.system_prompt = ""
        self.user_message = ""

    def solve(
        self,
        *,
        system_prompt,
        user_message,
        toolkit,
        max_turns,
        input_queue=None,
    ) -> RuntimeResult:
        del max_turns, input_queue
        self.system_prompt = system_prompt
        self.user_message = user_message
        specs = toolkit.get_tools_spec()
        self.tool_names = {spec["name"] for spec in specs}
        self.tool_specs = {spec["name"]: spec for spec in specs}
        self.inspected = toolkit.execute_tool(
            "inspect_evidence",
            {"ref": self.inspected_ref},
        )
        stats = {}
        if self.output_path is not None:
            stats["output_path"] = str(self.output_path)
        return RuntimeResult(
            finish_result=self.finish,
            stats=stats,
            error=self.error,
        )


class StructuredUncertainRuntime(InspectingRuntime):
    def __init__(self) -> None:
        super().__init__(None)
        self.finish_is_finish = False

    def solve(self, **kwargs) -> RuntimeResult:
        toolkit = kwargs["toolkit"]
        super().solve(**kwargs)
        finished = toolkit.execute_tool(
            "finish",
            {
                "verdict": "uncertain",
                "evidence_refs": ["final.json"],
                "confidence": 0.4,
                "reason": "the target is occluded",
            },
        )
        self.finish_is_finish = finished.is_finish
        return RuntimeResult(finish_result=finished.result)


class FinishComplianceRuntime(InspectingRuntime):
    def __init__(self, *, finish_on_retry: bool) -> None:
        super().__init__(None)
        self.finish_on_retry = finish_on_retry
        self.system_prompts: list[str] = []

    def solve(self, **kwargs) -> RuntimeResult:
        self.system_prompts.append(kwargs["system_prompt"])
        result = super().solve(**kwargs)
        if self.finish_on_retry and len(self.system_prompts) == 2:
            finished = kwargs["toolkit"].execute_tool(
                "finish",
                {
                    "verdict": "satisfied",
                    "evidence_refs": ["final.json"],
                    "confidence": 0.97,
                    "reason": "support relation is visible",
                },
            )
            return RuntimeResult(finish_result=finished.result)
        return result


def test_authoritative_decision_requires_structured_finish_and_whitelisted_evidence(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    events_path = tmp_path / "events.jsonl"
    runtime = InspectingRuntime(_finish())
    verifier = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
        recorder=JsonlEventRecorder(events_path, episode_id="episode-1"),
    )

    result = verifier.verify(_request(artifact))

    assert runtime.tool_names == {"inspect_evidence", "finish"}
    assert set(runtime.tool_specs["finish"]["input_schema"]["required"]) == {
        "verdict",
        "evidence_refs",
        "confidence",
        "reason",
    }
    assert runtime.inspected.result["content"] == '{"supported": true}\n'
    assert result.verdict == "satisfied"
    assert result.authoritative is True
    assert result.authorized is False
    assert result.evidence_refs == ("final.json",)
    assert result.verifier == "runtime-predicate:test-model"
    assert "classify_termination" not in runtime.system_prompt
    request_payload = json.loads(runtime.user_message)
    assert request_payload["success_predicate"] == "the object is supported"
    assert len(request_payload["evidence"][0]["sha256"]) == 64

    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["event"] == "predicate_verifier_finished"
    assert event["outcome_id"] == "outcome-1"
    assert event["artifacts"][0]["ref"] == "final.json"
    assert event["artifacts"][0]["sha256"] == request_payload["evidence"][0]["sha256"]


def test_execution_bound_simulator_feedback_is_available_as_structured_evidence(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    runtime = InspectingRuntime(_finish())
    request = replace(
        _request(artifact),
        execution_id="exec-1",
        feedback=ActionFeedback(
            execution_id="exec-1",
            tool_name="execute_vla",
            physical=True,
            execution_status="completed",
            environment_success=True,
            environment_terminated=False,
            environment_truncated=False,
        ),
    )

    RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(request)

    payload = json.loads(runtime.user_message)
    assert payload["execution_feedback"] == {
        "execution_id": "exec-1",
        "execution_status": "completed",
        "environment_success": True,
        "environment_terminated": False,
        "environment_truncated": False,
    }
    assert "execution-bound simulator evidence" in runtime.system_prompt


def test_structured_uncertain_finish_is_available_and_remains_advisory(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    runtime = StructuredUncertainRuntime()

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    verdict_schema = runtime.tool_specs["finish"]["input_schema"]["properties"][
        "verdict"
    ]
    assert verdict_schema["enum"] == ["not_satisfied", "satisfied", "uncertain"]
    assert "otherwise use uncertain" in runtime.system_prompt
    assert runtime.finish_is_finish is True
    assert result.verdict == "uncertain"
    assert result.reason == "the target is occluded"
    assert result.authoritative is False
    assert result.authorized is False


def test_missing_finish_gets_one_bounded_tool_compliance_retry(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    runtime = FinishComplianceRuntime(finish_on_retry=True)

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert len(runtime.system_prompts) == 2
    assert (
        "previous response did not invoke the finish tool" in runtime.system_prompts[1]
    )
    assert "JSON object in assistant text is invalid" in runtime.system_prompts[1]
    assert result.verdict == "satisfied"
    assert result.authoritative is True


def test_missing_finish_retry_is_bounded_and_still_fails_closed(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    runtime = FinishComplianceRuntime(finish_on_retry=False)

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert len(runtime.system_prompts) == 2
    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert result.reason == "verifier did not submit a structured finish"


@pytest.mark.parametrize(
    "suffix, media_type", [(".jpg", "image/jpeg"), (".png", "image/png")]
)
def test_inspection_preserves_image_media_type(
    tmp_path: Path,
    suffix: str,
    media_type: str,
) -> None:
    artifact = _artifact(tmp_path, f"final{suffix}")
    runtime = InspectingRuntime(
        _finish(evidence_refs=[artifact.name]),
        inspected_ref=artifact.name,
    )

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.authoritative is True
    assert len(runtime.inspected.projection.images) == 1
    assert runtime.inspected.projection.images[0].media_type == media_type


@pytest.mark.parametrize(
    "finish",
    [
        None,
        {
            "verdict": "satisfied",
            "evidence_refs": ["final.json"],
            "confidence": 0.9,
            "reason": "visible",
        },
        {"_finish": True, "verdict": "satisfied"},
        _finish(evidence_refs=[]),
        _finish(evidence_refs=["final.json", "final.json"]),
        _finish(evidence_refs=["other.json"]),
        _finish(confidence=True),
        _finish(confidence=math.nan),
        _finish(confidence=math.inf),
        _finish(confidence=1.1),
        _finish(reason=""),
    ],
    ids=[
        "missing",
        "missing-sentinel",
        "missing-fields",
        "no-evidence",
        "duplicate-evidence",
        "foreign-evidence",
        "boolean-confidence",
        "nan-confidence",
        "infinite-confidence",
        "out-of-range-confidence",
        "empty-reason",
    ],
)
def test_missing_or_malformed_finish_fails_closed(
    tmp_path: Path,
    finish: dict | None,
) -> None:
    artifact = _artifact(tmp_path)

    result = RuntimePredicateVerifier(
        InspectingRuntime(finish),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert result.authorized is False


def test_runtime_error_overrides_otherwise_valid_finish(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)

    result = RuntimePredicateVerifier(
        InspectingRuntime(_finish(), error="provider unavailable"),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert "provider unavailable" in result.reason


@pytest.mark.parametrize("ref", ["secret.txt", "../outside.txt", "/tmp/outside.txt"])
def test_toolkit_rejects_non_whitelisted_or_escaping_inspection(
    tmp_path: Path,
    ref: str,
) -> None:
    artifact = _artifact(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    runtime = InspectingRuntime(_finish(), inspected_ref=ref)

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert "not in the verification whitelist" in runtime.inspected.result["error"]
    assert result.verdict == "uncertain"
    assert result.authoritative is False


def test_missing_request_evidence_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    request = _request(artifact)
    artifact.unlink()

    result = RuntimePredicateVerifier(
        InspectingRuntime(_finish()),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(request)

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert "does not exist" in result.reason


def test_unreadable_text_inspection_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "final.txt"
    artifact.write_bytes(b"\xff\xfe")
    runtime = InspectingRuntime(
        _finish(evidence_refs=[artifact.name]),
        inspected_ref=artifact.name,
    )

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert "error" in runtime.inspected.result
    assert result.verdict == "uncertain"
    assert result.authoritative is False


def test_evidence_changed_after_resolution_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)

    class MutatingRuntime(InspectingRuntime):
        def solve(self, **kwargs):
            artifact.write_text('{"supported": false}\n', encoding="utf-8")
            return super().solve(**kwargs)

    runtime = MutatingRuntime(_finish())

    result = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert "changed after verification began" in runtime.inspected.result["error"]
    assert result.verdict == "uncertain"
    assert result.authoritative is False


def test_cited_evidence_that_was_never_inspected_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)

    class NonInspectingRuntime:
        def solve(self, **kwargs):
            del kwargs
            return RuntimeResult(finish_result=_finish())

    result = RuntimePredicateVerifier(
        NonInspectingRuntime(),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert "not successfully inspected" in result.reason


def test_evidence_changed_after_inspection_before_finish_fails_closed(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)

    class MutatingAfterInspectionRuntime:
        def solve(self, *, toolkit, **kwargs):
            del kwargs
            inspected = toolkit.execute_tool(
                "inspect_evidence",
                {"ref": artifact.name},
            )
            assert "error" not in inspected.result
            artifact.write_text('{"supported": false}\n', encoding="utf-8")
            return RuntimeResult(finish_result=_finish())

    result = RuntimePredicateVerifier(
        MutatingAfterInspectionRuntime(),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert "changed after inspection" in result.reason


def test_runtime_output_is_copied_to_safe_unique_audit_artifacts(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    external = tmp_path.parent / "shared-verifier.out"
    external.write_text("verifier transcript", encoding="utf-8")
    runtime = InspectingRuntime(_finish(), output_path=external)
    verifier = RuntimePredicateVerifier(
        runtime,
        evidence_root=tmp_path,
        model_id="test-model",
    )

    verifier.verify(_request(artifact))
    verifier.verify(_request(artifact))

    audit_root = (tmp_path / "verifier_audit").resolve()
    preserved = list(audit_root.glob("*.out"))
    assert len(preserved) == 2
    assert all(path.resolve().parent == audit_root for path in preserved)
    assert all(path.name != external.name for path in preserved)
    assert all(
        path.read_text(encoding="utf-8") == "verifier transcript" for path in preserved
    )


def test_symlinked_audit_directory_fails_closed_without_writing_outside_root(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    external_output = tmp_path.parent / "runtime-output.txt"
    external_output.write_text("transcript", encoding="utf-8")
    escaped_root = tmp_path.parent / "escaped-audit"
    escaped_root.mkdir()
    (tmp_path / "verifier_audit").symlink_to(escaped_root, target_is_directory=True)

    result = RuntimePredicateVerifier(
        InspectingRuntime(_finish(), output_path=external_output),
        evidence_root=tmp_path,
        model_id="test-model",
    ).verify(_request(artifact))

    assert result.verdict == "uncertain"
    assert result.authoritative is False
    assert list(escaped_root.iterdir()) == []
