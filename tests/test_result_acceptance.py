from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rpent.evaluation.attempt_registration import AttemptRegistration
from rpent.evaluation.result_acceptance import (
    FinalizedResultCandidate,
    NoWorker,
    ReapedWorker,
    ResultAcceptance,
)
from rpent.evaluation.results import ResultStore
from rpent.ui.app import create_evaluation_app
from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.case_snapshot import SnapshotIdentity
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.operation_snapshot import decode_operation_snapshot
from rpent.ui.repository import EvaluationRepository
from tests.test_case_runtime import _operation_events, _registration
from tests.test_episode_coordinator import _manifest, _result


def _runtime_and_detail(manifest, result):
    case = manifest.cases[0]
    runtime = CaseRuntimeStore()
    runtime.start_attempt(
        AttemptRegistration(
            1,
            manifest.evaluation_id,
            manifest.manifest_id,
            case.case_id,
            result.attempt_id,
            result.started_at,
        ),
        manifest,
        case,
    )
    entry = runtime.get_entry(manifest.evaluation_id, case.case_id)
    assert entry is not None
    detail = entry.value.model_copy(
        update={
            "execution_state": "terminal",
            "episode_result": result.to_payload(),
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "last_sequence": entry.last_sequence,
        }
    )
    return runtime, detail


@pytest.mark.parametrize(
    "reason",
    [
        "launch_failed",
        "cancelled_before_launch",
        "startup_recovery",
        "already_accepted_repair",
    ],
)
def test_no_worker_reason_is_a_closed_validated_set(reason: str) -> None:
    assert NoWorker(reason).reason == reason


def test_no_worker_reason_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="NoWorker reason"):
        NoWorker("worker_maybe_running")


@pytest.mark.parametrize(
    "completion",
    [None, ReapedWorker("foreign-attempt", 0), object()],
)
def test_acceptance_rejects_missing_wrong_scope_or_open_worker_proof(
    tmp_path: Path, completion: object
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
    )

    with pytest.raises((TypeError, ValueError), match="candidate|completion|scope"):
        acceptance.accept(FinalizedResultCandidate(result, completion))  # type: ignore[arg-type]

    assert not (tmp_path / "results.jsonl").exists()


def test_result_append_precedes_terminal_pending_assembly_write_and_publish(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    trace: list[str] = []

    class TrackingStore(ResultStore):
        def append_or_confirm(self, selected, *, require_existing=False):
            trace.append("append")
            return super().append_or_confirm(
                selected, require_existing=require_existing
            )

    original_pending = runtime.mark_terminal_pending
    original_publish = runtime.publish_terminal

    def pending(selected):
        trace.append("terminal_pending")
        return original_pending(selected)

    def publish(selected_detail, selected_result, operation_page=None):
        trace.append("publish")
        return original_publish(selected_detail, selected_result, operation_page)

    runtime.mark_terminal_pending = pending  # type: ignore[method-assign]
    runtime.publish_terminal = publish  # type: ignore[method-assign]

    def assemble(_result):
        trace.append("assemble")
        return detail

    def write(path, payload):
        trace.append(f"write:{path.name}")
        assert path.parent == Path(result.artifact_dir)
        assert payload["schema_version"] == 1

    acceptance = ResultAcceptance(
        manifest=manifest,
        store=TrackingStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=assemble,
        snapshot_writer=write,
        accepted_observer=lambda *_args: trace.append("summary"),
    )

    assert acceptance.accept(
        FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
    ) == result
    assert trace == [
        "append",
        "terminal_pending",
        "assemble",
        "write:case_detail.json",
        "write:operations.json",
        "publish",
        "summary",
    ]


def test_acceptance_writes_terminal_operation_snapshot(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
    )

    acceptance.accept(
        FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
    )

    payload = json.loads(
        (Path(result.artifact_dir) / "operations.json").read_text()
    )
    assert payload["schema_version"] == 1
    assert payload["page"]["attempt_id"] == result.attempt_id
    assert payload["page"]["last_sequence"] == runtime.get_case(
        result.evaluation_id, result.case_id
    ).last_sequence


def test_equal_reacceptance_repairs_missing_snapshot_without_duplicate_ledger_row(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
    )
    candidate = FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))

    acceptance.accept(candidate)
    snapshot = Path(result.artifact_dir) / "case_detail.json"
    snapshot.unlink()
    acceptance.accept(
        FinalizedResultCandidate(result, NoWorker("already_accepted_repair"))
    )

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1
    assert snapshot.is_file()


def test_fresh_runtime_repair_reconstructs_nonzero_terminal_operations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    manifest = _manifest(root, task_count=1)
    manifest.write_new(root / "evaluation_manifest.json")
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-1")
    store = ResultStore(root / "results.jsonl", manifest)
    store.append_or_confirm(result)
    events = _operation_events(_registration(manifest, case))
    attempt = Path(result.artifact_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "events.jsonl").write_text(
        "".join(json.dumps(event.to_payload()) + "\n" for event in events),
        encoding="utf-8",
    )
    repository = EvaluationRepository((root.parent,))
    repository.register_evaluation(root, manifest)
    runtime = CaseRuntimeStore()
    reconstructions = 0

    def reconstruct(selected):
        nonlocal reconstructions
        reconstructions += 1
        return repository.reconstruct_operations(
            selected.evaluation_id,
            selected.case_id,
            attempt_id=selected.attempt_id,
        )

    acceptance = ResultAcceptance(
        manifest=manifest,
        store=store,
        runtime_store=runtime,
        assembler=lambda selected: repository.reconstruct_case_detail(
            selected.evaluation_id,
            selected.case_id,
            attempt_id=selected.attempt_id,
        ),
        operation_reconstructor=reconstruct,
    )

    acceptance.accept(
        FinalizedResultCandidate(result, NoWorker("already_accepted_repair"))
    )

    payload = json.loads((attempt / "operations.json").read_text())
    page = decode_operation_snapshot(
        payload,
        SnapshotIdentity(
            result.evaluation_id,
            result.manifest_id,
            result.case_id,
            result.attempt_id,
        ),
        len(events),
    )
    assert reconstructions == 1
    assert page.last_sequence == len(events)
    assert page.incomplete_tail is False
    assert [operation.operation_id for operation in page.operations] == [
        "operation-1"
    ]
    assert runtime.get_operations(
        result.evaluation_id,
        result.case_id,
        result.attempt_id,
    ) == page
    app = create_evaluation_app(
        repository=repository,
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"repair-read-secret"),
        case_runtime_store=runtime,
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/evaluations/{result.evaluation_id}/cases/{result.case_id}"
            f"/operations?attempt_id={result.attempt_id}"
        )
    assert response.status_code == 200
    assert response.json() == page.model_dump(mode="json")
    assert len((root / "results.jsonl").read_text().splitlines()) == 1


def test_fresh_runtime_acceptance_reconstructs_nonzero_terminal_operations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluation"
    manifest = _manifest(root, task_count=1)
    manifest.write_new(root / "evaluation_manifest.json")
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-1")
    events = _operation_events(_registration(manifest, case))
    attempt = Path(result.artifact_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "events.jsonl").write_text(
        "".join(json.dumps(event.to_payload()) + "\n" for event in events),
        encoding="utf-8",
    )
    repository = EvaluationRepository((root.parent,))
    repository.register_evaluation(root, manifest)
    runtime = CaseRuntimeStore()
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(root / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda selected: repository.reconstruct_case_detail(
            selected.evaluation_id,
            selected.case_id,
            attempt_id=selected.attempt_id,
        ),
        operation_reconstructor=lambda selected: repository.reconstruct_operations(
            selected.evaluation_id,
            selected.case_id,
            attempt_id=selected.attempt_id,
        ),
    )

    acceptance.accept(
        FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
    )

    assert runtime.get_operations(
        result.evaluation_id,
        result.case_id,
        result.attempt_id,
    ).last_sequence == len(events)
    assert (attempt / "case_detail.json").is_file()


def test_already_accepted_repair_proof_cannot_create_a_new_ledger_row(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    result_path = tmp_path / "results.jsonl"
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(result_path, manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
    )

    with pytest.raises(
        ValueError, match="already_accepted_repair requires an accepted result"
    ):
        acceptance.accept(
            FinalizedResultCandidate(
                result, NoWorker("already_accepted_repair")
            )
        )

    assert not result_path.exists()


def test_conflicting_attempt_payload_is_rejected_without_projection_change(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
    )
    acceptance.accept(
        FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
    )
    ledger = (tmp_path / "results.jsonl").read_bytes()
    snapshot_path = Path(result.artifact_dir) / "case_detail.json"
    snapshot = snapshot_path.read_bytes()
    entry = runtime.get_entry(manifest.evaluation_id, result.case_id)

    with pytest.raises(ValueError, match="attempt payload conflicts"):
        acceptance.accept(
            FinalizedResultCandidate(
                replace(result, step_count=2),
                NoWorker("already_accepted_repair"),
            )
        )

    assert (tmp_path / "results.jsonl").read_bytes() == ledger
    assert snapshot_path.read_bytes() == snapshot
    assert runtime.get_entry(manifest.evaluation_id, result.case_id) == entry


def test_assembly_failure_marks_unavailable_but_preserves_accepted_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, _detail = _runtime_and_detail(manifest, result)
    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: (_ for _ in ()).throw(ValueError("broken")),
    )

    with pytest.raises(ValueError, match="broken"):
        acceptance.accept(
            FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
        )

    assert ResultStore(tmp_path / "results.jsonl", manifest).load() == (result,)
    entry = runtime.get_entry(manifest.evaluation_id, result.case_id)
    assert entry is not None and entry.phase == "unavailable"


def test_operation_write_failure_publishes_memory_value_and_preserves_accepted_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    result = _result(manifest, manifest.cases[0], attempt_id="attempt-1")
    runtime, detail = _runtime_and_detail(manifest, result)
    def write(path: Path, _payload: object) -> None:
        if path.name == "operations.json":
            raise OSError("disk full")

    acceptance = ResultAcceptance(
        manifest=manifest,
        store=ResultStore(tmp_path / "results.jsonl", manifest),
        runtime_store=runtime,
        assembler=lambda _result: detail,
        snapshot_writer=write,
    )

    acceptance.accept(
        FinalizedResultCandidate(result, ReapedWorker(result.attempt_id, 0))
    )

    assert ResultStore(tmp_path / "results.jsonl", manifest).load() == (result,)
    entry = runtime.get_entry(manifest.evaluation_id, result.case_id)
    assert entry is not None and entry.phase == "sealed"
    assert acceptance.snapshot()["snapshot_write_failures"] == 1
    assert runtime.snapshot()["snapshot_write_failures"] == 1
