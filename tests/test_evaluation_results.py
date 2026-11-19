from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rpent.evaluation.batches import TaskBatchEvidence
from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.diagnostics import aggregate_rpent_diagnostics
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import (
    EpisodeResult,
    ResultStore,
    aggregate_results,
    write_summary,
)
from rpent.recording.contracts import (
    ArtifactHealth,
    EventEnvelope,
    QualificationOutcome,
)
from tests.synthetic_tasks import SyntheticTask


def _manifest(task_count: int = 5) -> EvaluationManifest:
    tasks = tuple(
        SyntheticTask(
            suite="libero_10",
            index=index,
            name=f"task_{index}",
            language=f"instruction {index}",
            bddl_identifier=f"task_{index}.bddl",
            upstream_horizon=500,
        )
        for index in range(task_count)
    )
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="robocerebra-libero10-smolvla-v1",
        horizon_ceiling=480,
        checkpoint_repo_id="lerobot/smolvla_robocerebra",
        checkpoint_revision="7" * 40,
        binding_name="robocerebra-smolvla-v1",
    )
    cases = expand_episode_cases(tasks, protocol, base_seed=0)
    return EvaluationManifest(
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={"env": "synthetic", "task": "all"},
        environment={"name": "synthetic"},
        simulator={"revision": "b" * 40},
        policy={"lerobot_revision": "c" * 40},
        task_selector="all",
        cases=cases,
        schedule={"episodes_per_task": 1, "base_seed": 0},
        contracts={"action": {"shape": [7]}},
        runtime={"main_interpreter": "/opt/python"},
        created_at="2026-08-05T12:00:00+10:00",
    )


def _result(manifest, ordinal: int, status: str, **overrides) -> EpisodeResult:
    case = manifest.cases[ordinal]
    values = {
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "attempt_id": f"attempt-{ordinal}-{status}",
        "status": status,
        "environment": case.environment,
        "preset_name": case.preset_name,
        "task_name": case.task_name,
        "task_index": case.task_index,
        "seed": case.seed,
        "episode_index": case.episode_index,
        "worker_pid": 1000 + ordinal,
        "worker_exit_code": 0 if status in {"success", "terminal_failure"} else 1,
        "started_at": f"2026-08-05T12:00:0{ordinal}+10:00",
        "finished_at": f"2026-08-05T12:01:0{ordinal}+10:00",
        "artifact_dir": f"cases/{case.case_id}",
        "simulator_success": status == "success",
        "contract_success": status == "success",
        "terminated": status in {"success", "terminal_failure"},
        "truncated": status == "truncated",
        "step_count": ordinal + 1,
        "action_count": ordinal + 1,
        "checkpoint_repo_id": case.checkpoint_repo_id,
        "checkpoint_revision": case.checkpoint_revision,
        "binding_name": case.binding_name,
        "error_type": "RuntimeError" if status == "error" else None,
        "error_message": "worker failed" if status == "error" else None,
    }
    values.update(overrides)
    return EpisodeResult(**values)


def test_aggregate_keeps_every_scheduled_case_in_denominator() -> None:
    manifest = _manifest()
    results = tuple(
        _result(manifest, ordinal, status)
        for ordinal, status in enumerate(
            ("success", "terminal_failure", "truncated", "timeout", "error")
        )
    )

    summary = aggregate_results(manifest, results)

    assert summary.scheduled == 5
    assert summary.completed == 5
    assert summary.successful == 1
    assert summary.success_rate == pytest.approx(0.2)
    assert summary.errors == 1
    assert summary.timeouts == 1
    assert summary.status_counts == {
        "success": 1,
        "terminal_failure": 1,
        "truncated": 1,
        "timeout": 1,
        "error": 1,
        "cancelled": 0,
    }
    assert summary.per_task["task_0"]["success_rate"] == pytest.approx(1.0)
    assert summary.per_task["task_4"]["success_rate"] == pytest.approx(0.0)


def test_partial_aggregate_still_uses_all_scheduled_cases() -> None:
    manifest = _manifest()
    summary = aggregate_results(manifest, [_result(manifest, 0, "success")])

    assert summary.scheduled == 5
    assert summary.completed == 1
    assert summary.success_rate == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("simulator_success", "contract_success"),
    [(False, True), (True, False), (False, False)],
)
def test_success_requires_both_authorities(simulator_success, contract_success) -> None:
    manifest = _manifest(1)

    with pytest.raises(ValueError, match="success.*simulator.*contract"):
        _result(
            manifest,
            0,
            "success",
            simulator_success=simulator_success,
            contract_success=contract_success,
        )


def test_aggregate_rejects_duplicate_foreign_and_case_drift() -> None:
    manifest = _manifest(2)
    result = _result(manifest, 0, "terminal_failure")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(manifest, [result, result])
    with pytest.raises(ValueError, match="foreign manifest"):
        aggregate_results(
            manifest,
            [replace(result, manifest_id="f" * 64)],
        )
    with pytest.raises(ValueError, match="case identity"):
        aggregate_results(
            manifest,
            [replace(result, task_name="different")],
        )


def test_result_store_appends_canonical_records_and_rejects_duplicates(
    tmp_path,
) -> None:
    manifest = _manifest(2)
    path = tmp_path / "results.jsonl"
    store = ResultStore(path, manifest)
    first = _result(manifest, 0, "terminal_failure")
    second = _result(manifest, 1, "error")

    store.append(first)
    store.append(second)

    assert store.load() == (first, second)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(
        json.loads(line)["manifest_id"] == manifest.manifest_id for line in lines
    )
    with pytest.raises(ValueError, match="duplicate"):
        store.append(first)


def test_append_or_confirm_is_idempotent_and_rejects_conflicting_attempt(
    tmp_path,
) -> None:
    manifest = _manifest(1)
    store = ResultStore(tmp_path / "results.jsonl", manifest)
    result = _result(manifest, 0, "terminal_failure")

    assert store.append_or_confirm(result) == "appended"
    assert store.append_or_confirm(result) == "identical"
    before = store.path.read_bytes()

    with pytest.raises(ValueError, match="attempt payload conflicts"):
        store.append_or_confirm(replace(result, step_count=result.step_count + 1))

    assert store.path.read_bytes() == before
    assert store.load() == (result,)


def test_schema_v2_rejects_legacy_evaluation_id_fallback(tmp_path) -> None:
    manifest = _manifest(1)
    result = _result(manifest, 0, "terminal_failure")

    with pytest.raises(ValueError, match="foreign evaluation"):
        ResultStore(tmp_path / "results.jsonl", manifest).append(
            replace(result, evaluation_id=manifest.manifest_id)
        )


def test_loaded_schema_v1_accepts_verified_legacy_evaluation_id(tmp_path) -> None:
    legacy_path = tmp_path / "legacy-manifest.json"
    replace(_manifest(1), schema_version=1).write_new(legacy_path)
    legacy = EvaluationManifest.load(legacy_path)
    result = _result(legacy, 0, "terminal_failure")

    ResultStore(tmp_path / "legacy-results.jsonl", legacy).append(result)

    assert legacy.legacy_identity is True


def test_result_store_recovers_only_one_incomplete_trailing_line(tmp_path) -> None:
    manifest = _manifest(2)
    path = tmp_path / "results.jsonl"
    store = ResultStore(path, manifest)
    first = _result(manifest, 0, "terminal_failure")
    second = _result(manifest, 1, "error")
    store.append(first)
    with path.open("ab") as stream:
        stream.write(b'{"manifest_id":"interrupted"')

    assert store.load() == (first,)
    store.append(second)
    assert store.load() == (first, second)

    path.write_bytes(b'{"broken":\n' + path.read_bytes())
    with pytest.raises(ValueError, match="corrupt"):
        store.load()


def test_summary_is_materialized_atomically(tmp_path) -> None:
    manifest = _manifest(1)
    summary = aggregate_results(
        manifest,
        [_result(manifest, 0, "terminal_failure")],
    )
    path = tmp_path / "summary.json"

    write_summary(path, summary)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest_id"] == manifest.manifest_id
    assert payload["scheduled"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_rpent_diagnostics_separate_selected_contract_from_all_attempt_audit(
    tmp_path,
) -> None:
    manifest = replace(
        _manifest(1),
        cli_arguments={"planner": "api"},
        runtime={"evaluation_root": str(tmp_path)},
        schedule={
            "episodes_per_task": 1,
            "base_seed": 0,
            "worker_grouping": "task",
        },
    )
    case = manifest.cases[0]
    batch_id = "accepted-batch"
    attempt_id = f"{batch_id}-000-{case.case_id[:12]}"
    artifact = tmp_path / "cases" / case.case_id / "attempts" / attempt_id
    artifact.mkdir(parents=True)
    selected = _result(
        manifest,
        0,
        "error",
        attempt_id=attempt_id,
        task_batch_attempt_id=batch_id,
        artifact_dir=str(artifact),
        worker_exit_code=0,
        simulator_success=True,
        contract_success=False,
        terminated=True,
        error_type=None,
        error_message=None,
    )
    old = _result(manifest, 0, "terminal_failure", attempt_id="old-attempt")
    event = EventEnvelope(
        schema_version=1,
        evaluation_id=selected.evaluation_id,
        case_id=selected.case_id,
        attempt_id=selected.attempt_id,
        sequence=1,
        recorded_at="2026-08-09T00:00:00+00:00",
        event_type="automatic_verification",
        step_id=None,
        operation_id=None,
        execution_id="execution-1",
        payload={"source": "harness", "applied": True},
    )
    (artifact / "events.jsonl").write_text(
        json.dumps(event.to_payload()) + "\n", encoding="utf-8"
    )
    qualification = QualificationOutcome.evaluate(
        selected, ArtifactHealth("complete"), True
    )
    (artifact / "qualification.json").write_text(
        json.dumps(qualification.to_payload()) + "\n",
        encoding="utf-8",
    )
    evidence = TaskBatchEvidence(case.task_name, batch_id, "complete", (selected,), 0)

    diagnostics = aggregate_rpent_diagnostics(manifest, (old, selected), (evidence,))

    assert diagnostics.selected_cohort["simulator_successes"] == 1
    assert diagnostics.selected_cohort["contract_successes"] == 0
    assert diagnostics.selected_cohort["contract_success_rate"] == 0.0
    assert diagnostics.selected_cohort["system_successes"] == 0
    assert diagnostics.selected_cohort["system_success_rate"] == 0.0
    assert diagnostics.all_retained_audit["retained_attempts"] == 2
    assert diagnostics.structured_evidence["verification_events"] == 1
    assert diagnostics.structured_evidence["recovery_events"] is None
    assert diagnostics.structured_evidence["policy_chunks"] is None
    assert diagnostics.structured_evidence["planner_usage"] is None
    assert diagnostics.structured_evidence["configured_planner"] == "api"
    assert diagnostics.structured_evidence["artifact_health"] == {"complete": 1}
    with pytest.raises(TypeError):
        diagnostics.selected_cohort["worker_faults"]["drift"] = 1


def test_rpent_diagnostics_mark_missing_structured_evidence_unavailable(
    tmp_path,
) -> None:
    manifest = replace(
        _manifest(1),
        runtime={"evaluation_root": str(tmp_path)},
        schedule={
            "episodes_per_task": 1,
            "base_seed": 0,
            "worker_grouping": "task",
        },
    )
    case = manifest.cases[0]
    batch_id = "accepted-batch"
    attempt_id = f"{batch_id}-000-{case.case_id[:12]}"
    selected = _result(
        manifest,
        0,
        "success",
        attempt_id=attempt_id,
        task_batch_attempt_id=batch_id,
        artifact_dir=str(tmp_path / "cases" / case.case_id / "attempts" / attempt_id),
    )
    evidence = TaskBatchEvidence(case.task_name, batch_id, "complete", (selected,), 0)

    diagnostics = aggregate_rpent_diagnostics(manifest, (selected,), (evidence,))

    structured = diagnostics.structured_evidence
    assert structured["available"] is False
    assert structured["verification_events"] is None
    assert structured["recovery_events"] is None
    assert structured["policy_chunks"] is None
    assert structured["planner_usage"] is None
    assert structured["artifact_health"] is None
    assert structured["issues"]


def test_rpent_diagnostics_keep_native_contract_and_system_four_quadrants(
    tmp_path,
) -> None:
    manifest = replace(
        _manifest(4),
        runtime={"evaluation_root": str(tmp_path)},
        schedule={"episodes_per_task": 1, "base_seed": 0, "worker_grouping": "task"},
    )
    signals = ((True, True), (True, False), (False, True), (False, False))
    retained = []
    evidences = []
    for case, (simulator, contract) in zip(manifest.cases, signals, strict=True):
        batch_id = f"batch-{case.task_name}"
        attempt_id = f"{batch_id}-000-{case.case_id[:12]}"
        status = "success" if simulator and contract else "terminal_failure"
        result = _result(
            manifest,
            case.ordinal,
            status,
            attempt_id=attempt_id,
            task_batch_attempt_id=batch_id,
            artifact_dir=str(
                tmp_path / "cases" / case.case_id / "attempts" / attempt_id
            ),
            simulator_success=simulator,
            contract_success=contract,
        )
        retained.append(result)
        evidences.append(
            TaskBatchEvidence(case.task_name, batch_id, "complete", (result,), 0)
        )

    diagnostics = aggregate_rpent_diagnostics(manifest, retained, evidences)
    selected = diagnostics.selected_cohort

    assert selected["simulator_successes"] == 2
    assert selected["contract_successes"] == 2
    assert selected["contract_success_rate"] == 0.5
    assert selected["system_successes"] == 1
    assert selected["system_success_rate"] == 0.25
    assert diagnostics.all_retained_audit["system_success_rate"] == 0.25
    assert diagnostics.all_retained_audit["rate_denominator"] == "retained_attempts"


def test_rpent_diagnostics_reject_swapped_and_symlinked_child_evidence(
    tmp_path,
) -> None:
    manifest = replace(
        _manifest(1),
        runtime={"evaluation_root": str(tmp_path)},
        schedule={"episodes_per_task": 1, "base_seed": 0, "worker_grouping": "task"},
    )
    case = manifest.cases[0]
    batch_id = "accepted-batch"
    attempt_id = f"{batch_id}-000-{case.case_id[:12]}"
    artifact = tmp_path / "cases" / case.case_id / "attempts" / attempt_id
    artifact.mkdir(parents=True)
    result = _result(
        manifest,
        0,
        "success",
        attempt_id=attempt_id,
        task_batch_attempt_id=batch_id,
        artifact_dir=str(artifact),
    )
    evidence = TaskBatchEvidence(case.task_name, batch_id, "complete", (result,), 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    event = EventEnvelope(
        1,
        result.evaluation_id,
        result.case_id,
        result.attempt_id,
        1,
        "2026-08-09T00:00:00+00:00",
        "automatic_verification",
        None,
        None,
        "execution-1",
        {"applied": True},
    )
    (outside / "events.jsonl").write_text(
        json.dumps(event.to_payload()) + "\n", encoding="utf-8"
    )
    qualification = QualificationOutcome.evaluate(
        result, ArtifactHealth("complete"), True
    )
    (outside / "qualification.json").write_text(
        json.dumps(qualification.to_payload()) + "\n", encoding="utf-8"
    )
    (artifact / "events.jsonl").symlink_to(outside / "events.jsonl")
    (artifact / "qualification.json").symlink_to(outside / "qualification.json")

    escaped = aggregate_rpent_diagnostics(manifest, (result,), (evidence,))
    assert escaped.structured_evidence["verification_events"] is None
    assert escaped.structured_evidence["artifact_health"] is None

    (artifact / "events.jsonl").unlink()
    (artifact / "qualification.json").unlink()
    alternate_events = artifact / "alternate-events.jsonl"
    alternate_qualification = artifact / "alternate-qualification.json"
    alternate_events.write_text(json.dumps(event.to_payload()) + "\n", encoding="utf-8")
    alternate_qualification.write_text(
        json.dumps(qualification.to_payload()) + "\n", encoding="utf-8"
    )
    (artifact / "events.jsonl").symlink_to(alternate_events)
    (artifact / "qualification.json").symlink_to(alternate_qualification)

    same_directory = aggregate_rpent_diagnostics(manifest, (result,), (evidence,))
    assert same_directory.structured_evidence["verification_events"] is None
    assert same_directory.structured_evidence["artifact_health"] is None

    (artifact / "events.jsonl").unlink()
    (artifact / "qualification.json").unlink()
    swapped_event = replace(event, case_id="foreign-case")
    (artifact / "events.jsonl").write_text(
        json.dumps(swapped_event.to_payload()) + "\n", encoding="utf-8"
    )
    swapped_qualification = QualificationOutcome.evaluate(
        replace(result, action_count=result.action_count + 1),
        ArtifactHealth("complete"),
        True,
    )
    (artifact / "qualification.json").write_text(
        json.dumps(swapped_qualification.to_payload()) + "\n", encoding="utf-8"
    )

    swapped = aggregate_rpent_diagnostics(manifest, (result,), (evidence,))
    assert swapped.structured_evidence["verification_events"] is None
    assert swapped.structured_evidence["artifact_health"] is None
    assert "event evidence unavailable" in swapped.structured_evidence["issues"]
    assert (
        "artifact health evidence unavailable" in swapped.structured_evidence["issues"]
    )
