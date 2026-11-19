from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rpent.ui.app import create_evaluation_app
from rpent.ui.case_runtime import CaseRuntimeStore
from rpent.ui.case_snapshot import (
    SnapshotRebuilding,
    SnapshotUnavailable,
    encode_case_snapshot,
)
from rpent.ui.event_hub import EvaluationEventHub
from rpent.ui.operation_snapshot import encode_operation_snapshot
from rpent.ui.schemas import CaseDetail, OperationPage
from tests.test_case_runtime import _registration
from tests.test_case_snapshot import _terminal
from tests.test_episode_coordinator import _manifest, _result
from tests.test_ui_repository import (
    _append_jsonl,
    _write_evaluation,
)
from tests.test_ui_repository import (
    _result as _repository_result,
)


def test_cold_terminal_snapshot_is_cached_after_one_selected_load(tmp_path) -> None:
    detail, _result, _identity = _terminal(tmp_path)
    calls = 0

    def load(evaluation_id, case_id, attempt_id):
        nonlocal calls
        calls += 1
        assert (evaluation_id, case_id, attempt_id) == (
            detail.evaluation_id,
            detail.case_id,
            None,
        )
        return detail

    store = CaseRuntimeStore(snapshot_loader=load)

    assert store.get_case(detail.evaluation_id, detail.case_id) == detail
    assert store.get_case(detail.evaluation_id, detail.case_id) == detail
    assert calls == 1


def test_missing_snapshot_returns_rebuilding_then_memory_value(tmp_path) -> None:
    detail, _result, _identity = _terminal(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def reconstruct(evaluation_id, case_id, *, attempt_id=None):
        entered.set()
        assert release.wait(timeout=2)
        return detail

    store = CaseRuntimeStore(
        snapshot_loader=lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        snapshot_reconstructor=reconstruct,
    )

    with pytest.raises(SnapshotRebuilding):
        store.get_case(detail.evaluation_id, detail.case_id)
    assert entered.wait(timeout=1)
    release.set()
    assert store.wait_for_snapshots(timeout=2)
    assert store.get_case(detail.evaluation_id, detail.case_id) == detail


def test_failed_rebuild_returns_unavailable_until_authoritative_change(tmp_path) -> None:
    detail, _result, _identity = _terminal(tmp_path)
    store = CaseRuntimeStore(
        snapshot_loader=lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        snapshot_reconstructor=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("damaged")
        ),
    )

    with pytest.raises(SnapshotRebuilding):
        store.get_case(detail.evaluation_id, detail.case_id)
    assert store.wait_for_snapshots(timeout=2)
    with pytest.raises(SnapshotUnavailable):
        store.get_case(detail.evaluation_id, detail.case_id)


def test_attempt_switch_during_rebuild_cannot_publish_obsolete_value(tmp_path) -> None:
    manifest = _manifest(tmp_path, task_count=1)
    case = manifest.cases[0]
    result = _result(manifest, case, attempt_id="attempt-1")
    source = CaseRuntimeStore()
    source.start_attempt(_registration(manifest, case), manifest, case)
    source.mark_terminal_pending(result)
    base = source.get_entry(manifest.evaluation_id, case.case_id).value  # type: ignore[union-attr]
    source.publish_terminal(base, result)
    old = source.get_entry(manifest.evaluation_id, case.case_id).value  # type: ignore[union-attr]
    entered = threading.Event()
    release = threading.Event()

    def reconstruct(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return old

    store = CaseRuntimeStore(
        snapshot_loader=lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        snapshot_reconstructor=reconstruct,
    )
    with pytest.raises(SnapshotRebuilding):
        store.get_case(old.evaluation_id, old.case_id)
    assert entered.wait(timeout=1)

    store.start_attempt(_registration(manifest, case, "attempt-2"), manifest, case)
    release.set()
    assert store.wait_for_snapshots(timeout=2)
    current = store.get_case(manifest.evaluation_id, case.case_id)
    assert current.attempt_id == "attempt-2"
    assert current.episode_result is None


def test_get_uses_exact_rebuilding_and_unavailable_503_contract(tmp_path) -> None:
    detail, _result, _identity = _terminal(tmp_path)
    release = threading.Event()
    fail = False

    def reconstruct(*_args, **_kwargs):
        assert release.wait(timeout=2)
        if fail:
            raise ValueError("damaged")
        return detail

    store = CaseRuntimeStore(
        snapshot_loader=lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        snapshot_reconstructor=reconstruct,
    )
    app = create_evaluation_app(
        repository=SimpleNamespace(),
        service=SimpleNamespace(),
        event_hub=EvaluationEventHub(secret=b"snapshot-route-secret"),
        case_runtime_store=store,
    )
    client = TestClient(app, raise_server_exceptions=False)
    path = f"/api/evaluations/{detail.evaluation_id}/cases/{detail.case_id}"

    rebuilding = client.get(path)
    assert rebuilding.status_code == 503
    assert rebuilding.headers["Retry-After"] == "1"
    assert rebuilding.json() == {"detail": "case_snapshot_rebuilding"}
    stream_rebuilding = client.get(f"{path}/stream?timeout=0")
    assert stream_rebuilding.status_code == 503
    assert stream_rebuilding.headers["Retry-After"] == "1"
    assert stream_rebuilding.json() == {"detail": "case_snapshot_rebuilding"}

    fail = True
    release.set()
    assert store.wait_for_snapshots(timeout=2)
    unavailable = client.get(path)
    assert unavailable.status_code == 503
    assert "Retry-After" not in unavailable.headers
    assert unavailable.json() == {"detail": "case_snapshot_unavailable"}


def test_257_attempts_return_256_newest_total_and_truncated(tmp_path) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    results = tuple(
        _repository_result(
            manifest, case, f"attempt-{index:03d}", status="error"
        )
        for index in range(257)
    )
    _append_jsonl(root / "results.jsonl", *(item.to_payload() for item in results))
    from rpent.ui.repository import EvaluationRepository

    detail = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert detail.attempts_total == 257
    assert detail.attempts_truncated is True
    assert len(detail.attempts) == 256
    assert detail.attempts[0].attempt_id == "attempt-001"
    assert detail.attempts[-1].attempt_id == "attempt-256"


def test_loading_an_old_attempt_does_not_reorder_the_newest_summary_window(
    tmp_path,
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    results = tuple(
        _repository_result(
            manifest, case, f"attempt-{index:03d}", status="error"
        )
        for index in range(257)
    )
    _append_jsonl(root / "results.jsonl", *(item.to_payload() for item in results))
    from rpent.ui.repository import EvaluationRepository

    repository = EvaluationRepository((root.parent,))
    before = repository._attempt_summary_index[
        (manifest.evaluation_id, case.case_id)
    ]
    assert before[0].attempt_id == "attempt-001"
    old = repository.get_case(
        manifest.evaluation_id, case.case_id, attempt_id="attempt-000"
    )
    assert len(repository._result_index[(manifest.evaluation_id, case.case_id)]) == 257
    assert repository._attempt_summary_index[
        (manifest.evaluation_id, case.case_id)
    ][0].attempt_id == "attempt-001"
    latest = repository.decorate_case_detail(old)

    assert len(latest.attempts) == 256
    assert latest.attempts[0].attempt_id == "attempt-001"
    assert latest.attempts[-1].attempt_id == "attempt-256"


def test_fresh_process_loads_old_operation_snapshot_beyond_summary_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    results = tuple(
        _repository_result(
            manifest, case, f"attempt-{index:03d}", status="error"
        )
        for index in range(257)
    )
    _append_jsonl(root / "results.jsonl", *(item.to_payload() for item in results))
    from rpent.ui.repository import EvaluationRepository

    for result in results:
        detail = CaseDetail(
            case_id=case.case_id,
            ordinal=case.ordinal,
            task_name=case.task_name,
            task_index=case.task_index,
            episode_index=case.episode_index,
            seed=case.seed,
            execution_state="terminal",
            attempt_id=result.attempt_id,
            episode_result=result.to_payload(),
            artifact_health=None,
            current_step=result.step_count,
            started_at=result.started_at,
            finished_at=result.finished_at,
            run_summary=None,
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            last_sequence=0,
            case=case.to_payload(),
        )
        attempt = (
            root
            / "cases"
            / case.case_id
            / "attempts"
            / result.attempt_id
        )
        attempt.mkdir(parents=True, exist_ok=True)
        (attempt / "case_detail.json").write_text(
            json.dumps(encode_case_snapshot(detail)), encoding="utf-8"
        )
        (attempt / "operations.json").write_text(
            json.dumps(
                encode_operation_snapshot(
                    OperationPage(
                        evaluation_id=manifest.evaluation_id,
                        case_id=case.case_id,
                        attempt_id=result.attempt_id,
                        last_sequence=detail.last_sequence,
                        incomplete_tail=False,
                        first_step_id=None,
                        latest_step_id=None,
                        operations=(),
                    ),
                    manifest.manifest_id,
                )
            ),
            encoding="utf-8",
        )

    restarted = EvaluationRepository((root.parent,))
    monkeypatch.setattr(
        restarted,
        "events_since",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("operation snapshot load replayed the journal")
        ),
    )

    page = restarted.load_terminal_operation_snapshot(
        manifest.evaluation_id, case.case_id, "attempt-000"
    )

    assert page.attempt_id == "attempt-000"
    assert page.last_sequence == 0


def test_restart_attempt_summary_preserves_extended_snapshot_metrics(tmp_path) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    results = tuple(
        _repository_result(manifest, case, f"attempt-{index}", status="error")
        for index in range(2)
    )
    _append_jsonl(root / "results.jsonl", *(item.to_payload() for item in results))
    from rpent.ui.repository import EvaluationRepository

    builder = EvaluationRepository((root.parent,))
    for index, result in enumerate(results):
        detail = builder.get_case(
            manifest.evaluation_id, case.case_id, attempt_id=result.attempt_id
        )
        assert detail.run_summary is not None
        detail = detail.model_copy(
            update={
                "run_summary": detail.run_summary.model_copy(
                    update={
                        "planner_elapsed_seconds": 10.5 + index,
                        "vla_chunks": 20 + index,
                        "retained_frame_count": 30 + index,
                    }
                )
            }
        )
        target = (
            root
            / "cases"
            / case.case_id
            / "attempts"
            / result.attempt_id
            / "case_detail.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(encode_case_snapshot(detail)), encoding="utf-8")

    restarted = EvaluationRepository((root.parent,))
    loaded = restarted.load_terminal_case_snapshot(
        manifest.evaluation_id, case.case_id
    )

    assert [item.attempt_id for item in loaded.attempts] == [
        "attempt-0",
        "attempt-1",
    ]
    assert [item.summary.planner_elapsed_seconds for item in loaded.attempts] == [
        10.5,
        11.5,
    ]
    assert [item.summary.vla_chunks for item in loaded.attempts] == [20, 21]
    assert [item.summary.retained_frame_count for item in loaded.attempts] == [30, 31]


def test_cold_terminal_get_opens_only_selected_case_detail_json(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    result = _repository_result(manifest, case, "attempt-1", status="error")
    _append_jsonl(root / "results.jsonl", result.to_payload())
    attempt = root / "cases" / case.case_id / "attempts" / "attempt-1"
    attempt.mkdir(parents=True)
    from rpent.ui.repository import EvaluationRepository

    builder = EvaluationRepository((root.parent,))
    detail = builder.get_case(manifest.evaluation_id, case.case_id)
    (attempt / "case_detail.json").write_text(
        json.dumps(encode_case_snapshot(detail)), encoding="utf-8"
    )
    repository = EvaluationRepository((root.parent,))
    import rpent.ui.repository as repository_module

    original = repository_module._open_beneath
    opened = []

    def record(boundary, path):
        opened.append(path)
        return original(boundary, path)

    monkeypatch.setattr(repository_module, "_open_beneath", record)

    loaded = repository.load_terminal_case_snapshot(
        manifest.evaluation_id, case.case_id
    )

    assert loaded.attempt_id == "attempt-1"
    assert opened == [attempt / "case_detail.json"]
