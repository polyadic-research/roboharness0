"""RPent/system diagnostics kept separate from official native-success scores."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rpent.evaluation.batches import TaskBatchEvidence, select_complete_task_batch
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import aggregate_official_metrics
from rpent.evaluation.results import EpisodeResult


@dataclass(frozen=True, slots=True)
class RPentDiagnostics:
    manifest_id: str
    selected_cohort: Mapping[str, Any]
    all_retained_audit: Mapping[str, Any]
    structured_evidence: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "selected_cohort": _plain(self.selected_cohort),
            "all_retained_audit": _plain(self.all_retained_audit),
            "structured_evidence": _plain(self.structured_evidence),
        }


def aggregate_rpent_diagnostics(
    manifest: EvaluationManifest,
    results: Iterable[EpisodeResult],
    batches: Iterable[TaskBatchEvidence],
) -> RPentDiagnostics:
    """Project accepted-cohort diagnostics and a separate all-attempt audit."""
    retained = tuple(results)
    evidences = tuple(batches)
    aggregate_official_metrics(manifest, retained, evidences)
    task_names = tuple(dict.fromkeys(case.task_name for case in manifest.cases))
    selected_evidence = tuple(
        evidence
        for task in task_names
        if (
            evidence := select_complete_task_batch(
                item for item in evidences if item.task_name == task
            )
        )
        is not None
    )
    selected = tuple(
        result for evidence in selected_evidence for result in evidence.results
    )
    selected_cohort = _result_projection(
        selected, denominator=len(manifest.cases), scheduled=len(manifest.cases)
    )
    audit = _result_projection(
        retained, denominator=len(retained), scheduled=len(manifest.cases)
    )
    audit = {
        **audit,
        "retained_attempts": len(retained),
        "rate_denominator": "retained_attempts",
    }
    structured = _structured_projection(manifest, selected)
    return RPentDiagnostics(
        manifest.manifest_id,
        _freeze(selected_cohort),
        _freeze(audit),
        _freeze(structured),
    )


def _result_projection(
    results: tuple[EpisodeResult, ...], *, denominator: int, scheduled: int
) -> dict[str, Any]:
    statuses = Counter(result.status for result in results)
    faults = Counter(
        result.error_type or result.status
        for result in results
        if result.error_type is not None
        or result.status in {"error", "timeout", "cancelled"}
    )
    contract_successes = sum(result.contract_success for result in results)
    system_successes = sum(result.status == "success" for result in results)
    return {
        "rollouts": len(results),
        "scheduled_rollouts": scheduled,
        "contract_successes": contract_successes,
        "contract_success_rate": (
            contract_successes / denominator if denominator else 0.0
        ),
        "system_successes": system_successes,
        "system_success_rate": (system_successes / denominator if denominator else 0.0),
        "simulator_successes": sum(result.simulator_success for result in results),
        "harness_classifications": dict(statuses),
        "worker_faults": dict(faults),
        "step_count": sum(result.step_count for result in results),
        "action_count": sum(result.action_count for result in results),
    }


def _structured_projection(
    manifest: EvaluationManifest,
    results: tuple[EpisodeResult, ...],
) -> dict[str, Any]:
    from rpent.recording.contracts import QualificationOutcome

    events: list[Any] = []
    health = Counter()
    issues: list[str] = []
    root_value = manifest.runtime.get("evaluation_root")
    root = Path(str(root_value)).resolve() if root_value else None
    for result in results:
        artifact = Path(result.artifact_dir).resolve()
        if root is None or root not in artifact.parents:
            issues.append("artifact root unavailable")
            continue
        event_path = artifact / "events.jsonl"
        if event_path.is_file():
            try:
                if event_path.resolve() != event_path:
                    raise ValueError("event evidence escapes artifact root")
                events.extend(_read_events(event_path, result))
            except (OSError, RecursionError, ValueError):
                issues.append("event evidence unavailable")
        else:
            issues.append("event evidence unavailable")
        qualification = artifact / "qualification.json"
        if qualification.is_file():
            try:
                if qualification.resolve() != qualification:
                    raise ValueError("artifact health evidence escapes artifact root")
                outcome = QualificationOutcome.from_payload(
                    json.loads(qualification.read_text(encoding="utf-8"))
                )
                if outcome.episode_result != result:
                    raise ValueError("qualification result identity drift")
                health[outcome.artifact_health.status] += 1
            except (OSError, RecursionError, ValueError, json.JSONDecodeError):
                issues.append("artifact health evidence unavailable")
        else:
            issues.append("artifact health evidence unavailable")
    event_evidence = (
        bool(events)
        or bool(results)
        and not any(issue == "event evidence unavailable" for issue in issues)
    )
    configured_planner = _planner_identity(manifest)
    issues.extend(
        (
            "recovery evidence unavailable",
            "planner usage evidence unavailable",
            "policy chunk evidence unavailable",
        )
    )
    return {
        "available": event_evidence or bool(health) or configured_planner is not None,
        "issues": tuple(dict.fromkeys(issues)),
        "verification_events": (
            sum(event.event_type == "automatic_verification" for event in events)
            if event_evidence
            else None
        ),
        "recovery_events": None,
        "planner_usage": None,
        "configured_planner": configured_planner,
        "policy_chunks": None,
        "artifact_health": dict(health) if health else None,
    }


def _read_events(path: Path, result: EpisodeResult) -> tuple[Any, ...]:
    from rpent.recording.contracts import EventEnvelope

    data = path.read_bytes()
    if data and not data.endswith((b"\n", b"\r")):
        raise ValueError("incomplete event evidence")
    records = []
    for line in data.splitlines():
        event = EventEnvelope.from_payload(json.loads(line))
        if (
            event.evaluation_id != result.evaluation_id
            or event.case_id != result.case_id
            or event.attempt_id != result.attempt_id
        ):
            raise ValueError("event evidence identity drift")
        records.append(event)
    if tuple(event.sequence for event in records) != tuple(range(1, len(records) + 1)):
        raise ValueError("event evidence sequence is not contiguous")
    return tuple(records)


def _planner_identity(manifest: EvaluationManifest) -> str | None:
    planner = manifest.cli_arguments.get("planner")
    if isinstance(planner, str) and planner:
        return planner
    argv = manifest.cli_arguments.get("argv")
    if isinstance(argv, (tuple, list)) and "--planner" in argv:
        index = argv.index("--planner") + 1
        if index < len(argv) and isinstance(argv[index], str) and argv[index]:
            return argv[index]
    return None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = ["RPentDiagnostics", "aggregate_rpent_diagnostics"]
