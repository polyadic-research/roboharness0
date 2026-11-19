from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import rpent.ui.repository as repository_module
from robots.robocasa.qualification import QUALIFICATION_CHECK_IDS
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation_progress import (
    PreparationError,
    PreparationProgress,
)
from rpent.evaluation.results import EpisodeResult, aggregate_results
from rpent.recording import media as recording_media
from rpent.recording.contracts import JSON_SAFE_INTEGER_MAX
from rpent.ui.repository import (
    CurrentAttemptUnavailable,
    EvaluationRepository,
    _open_beneath,
)
from rpent.ui.schemas import (
    ArtifactHealthView,
    EvaluationListItem,
    EvaluationMetricsView,
    EvaluationSummaryView,
)

FIXTURE = Path(__file__).parent / "fixtures/ui/completed-evaluation"


def _case(*, ordinal: int = 0, task_name: str = "pick mug") -> EpisodeCase:
    return EpisodeCase(
        ordinal=ordinal,
        environment="robocasa",
        preset_name="robocasa-smoke",
        task_name=task_name,
        task_index=ordinal,
        instruction="pick up the mug",
        seed=ordinal,
        episode_index=0,
        horizon=100,
        checkpoint_repo_id="org/policy",
        checkpoint_revision="a" * 40,
        binding_name="robocasa-v1",
    )


def _write_evaluation(
    root: Path,
    *,
    evaluation_id: str = "evaluation-route",
    case: EpisodeCase | None = None,
    index_root: Path | None = None,
    persist_index: bool = True,
):
    case = case or _case()
    manifest = EvaluationManifest(
        evaluation_id=evaluation_id,
        rpent_commit="b" * 40,
        rpent_dirty=False,
        cli_arguments={"task": "0"},
        environment={"name": "robocasa", "preset_name": "robocasa-smoke"},
        simulator={"name": "robocasa"},
        policy={"name": "smolvla"},
        task_selector="0",
        cases=(case,),
        schedule={"episodes_per_task": 1},
        contracts={},
        runtime={"evaluation_root": str(root)},
        recording={"schema_version": 1, "mode": "all", "camera_ids": []},
        qualification={"require_complete_artifacts": False},
        created_at="2026-08-06T00:00:00+00:00",
    )
    manifest.write_new(root / "evaluation_manifest.json")
    if persist_index:
        boundary = root.parent if index_root is None else index_root
        EvaluationRepository((boundary,)).register_evaluation(root, manifest)
    return manifest, case


def _register_after_barrier(
    boundary: Path, root: Path, gate: multiprocessing.synchronize.Barrier
) -> None:
    repository = EvaluationRepository((boundary,))
    manifest = EvaluationManifest.load(root / "evaluation_manifest.json")
    gate.wait()
    repository.register_evaluation(root, manifest)


def _update_after_barrier(
    boundary: Path,
    evaluation_id: str,
    gate: multiprocessing.synchronize.Barrier,
) -> None:
    repository = EvaluationRepository((boundary,))
    gate.wait()
    repository.update_evaluation_list_item(
        evaluation_id,
        execution_state="terminal",
    )


def _result(manifest, case, attempt_id: str, *, status: str = "success"):
    success = status == "success"
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        status=status,
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=123,
        worker_exit_code=0 if success else 1,
        started_at="2026-08-06T00:00:00+00:00",
        finished_at="2026-08-06T00:00:02+00:00",
        artifact_dir=str(
            Path(manifest.runtime["evaluation_root"])
            / "cases"
            / case.case_id
            / "attempts"
            / attempt_id
        ),
        simulator_success=success,
        contract_success=success,
        terminated=success,
        truncated=False,
        step_count=2,
        action_count=2,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )


def _append_jsonl(path: Path, *values: dict, tail: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join((json.dumps(value) + "\n").encode() for value in values) + tail
    )


def _qualification_checks(accepted: bool) -> list[dict]:
    return [
        {
            "id": check_id,
            "accepted": accepted,
            "evidence": (
                {"robocasa_version": "1.0.1"}
                if check_id == "installed_dependencies"
                else {}
            ),
            "remediation": "none" if accepted else "resume the evaluation",
        }
        for check_id in QUALIFICATION_CHECK_IDS
    ]


def _event(manifest, case, attempt_id: str, sequence: int, step_id: int):
    return {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": attempt_id,
        "sequence": sequence,
        "recorded_at": "2026-08-06T00:00:01+00:00",
        "event_type": "transition_recorded",
        "step_id": step_id,
        "operation_id": None,
        "execution_id": None,
        "payload": {},
    }


def _tool_result_event(
    manifest,
    case,
    attempt_id: str,
    sequence: int,
    *references: str,
):
    event = _event(manifest, case, attempt_id, sequence, sequence)
    event.update(
        event_type="tool_result",
        payload={
            "tool": "segment",
            "result": {"observation_refs": list(references)},
        },
    )
    return event


def _write_result(root: Path, result: EpisodeResult) -> None:
    _append_jsonl(root / "results.jsonl", result.to_payload())


def _attempt_root(root: Path, case_id: str, attempt_id: str) -> Path:
    attempt = root / "cases" / case_id / "attempts" / attempt_id
    attempt.mkdir(parents=True)
    return attempt


def _write_case_projection(
    attempt: Path, manifest, case, *, started_at: str | None = None
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "attempt_id": attempt.name,
        "case": case.to_payload(),
        "recording": dict(manifest.recording),
        "qualification": dict(manifest.qualification),
    }
    if started_at is not None:
        payload["started_at"] = started_at
    (attempt / "case.json").write_text(json.dumps(payload))


def _write_artifacts(
    attempt: Path,
    manifest,
    case,
    *,
    status: str = "complete",
    issues: list[str] | None = None,
    artifacts: list[dict] | None = None,
    identity: dict | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": attempt.name,
        "health": {"status": status, "issues": issues or []},
        "artifacts": artifacts or [],
    }
    payload.update(identity or {})
    (attempt / "artifacts.json").write_text(json.dumps(payload))


def _camera_discovery_payload(camera_id: str = "main") -> dict:
    return {
        "episode_id": "episode-live",
        "profile": {"mode": "all", "camera_ids": []},
        "cameras": [
            {
                "camera_id": camera_id,
                "label": "Main",
                "role": "primary",
                "width": 64,
                "height": 48,
                "channels": 3,
                "fps": 20.0,
                "media_type": "rgb",
            }
        ],
        "retained_camera_ids": [camera_id],
    }


def _write_recording_discovery(
    attempt: Path, manifest, case, *, camera_id: str = "main", **updates
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": attempt.name,
        **_camera_discovery_payload(camera_id),
        "health": {"status": "complete", "issues": []},
        "artifacts": [{"kind": "untrusted", "relative_path": "outside"}],
    }
    payload.update(updates)
    (attempt / "recording.json").write_text(json.dumps(payload))


def test_missing_index_does_not_discover_existing_manifest(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluation(
        evaluations / "old",
        evaluation_id="old",
        persist_index=False,
    )

    assert EvaluationRepository((evaluations,)).list_evaluations() == ()


def test_registration_persists_index_and_restart_loads_only_named_manifest(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    repository = EvaluationRepository((evaluations,))
    manifest, _ = _write_evaluation(
        evaluations / "new",
        evaluation_id="new",
        persist_index=False,
    )

    repository.register_evaluation(evaluations / "new", manifest)

    index_path = evaluations / ".rpent-evaluations.json"
    assert index_path.is_file()

    restarted = EvaluationRepository((evaluations,))
    assert [item.evaluation_id for item in restarted.list_evaluations()] == ["new"]
    with pytest.raises(KeyError):
        restarted.evaluation_metadata("missing")


def test_external_artifact_root_registers_in_single_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "logs"
    artifact_root = tmp_path / "outputs" / "run"
    manifest, _ = _write_evaluation(
        artifact_root,
        evaluation_id="external",
        persist_index=False,
    )

    EvaluationRepository((catalog,)).register_evaluation(artifact_root, manifest)

    payload = json.loads((catalog / ".rpent-evaluations.json").read_text())
    assert payload["schema_version"] == 3
    assert payload["evaluations"][0]["artifact_root"] == str(
        artifact_root.resolve()
    )
    assert not (artifact_root / ".rpent-evaluations.json").exists()
    restarted = EvaluationRepository((catalog,))
    assert restarted.evaluation_metadata("external").manifest_id == (
        manifest.manifest_id
    )


@pytest.mark.parametrize(
        "payload",
    [
        {},
        {"schema_version": 4, "evaluations": []},
        {"schema_version": 1, "evaluations": {}},
        {
            "schema_version": 1,
            "evaluations": [
                {
                    "evaluation_id": "escape",
                    "manifest_id": "sha256:bad",
                    "relative_root": "../outside",
                }
            ],
        },
    ],
)
def test_invalid_persistent_index_is_rejected(tmp_path: Path, payload: dict) -> None:
    root = tmp_path / "evaluations"
    root.mkdir()
    (root / ".rpent-evaluations.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="index|schema|artifact"):
        EvaluationRepository((root,))


def test_persistent_index_schema_version_must_be_an_integer(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    root.mkdir()
    (root / ".rpent-evaluations.json").write_text(
        json.dumps({"schema_version": 2.0, "evaluations": []})
    )

    with pytest.raises(ValueError, match="schema version"):
        EvaluationRepository((root,))


def test_persistent_index_manifest_identity_must_match(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    manifest, _ = _write_evaluation(
        evaluations / "run",
        persist_index=False,
    )
    item = EvaluationListItem(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        created_at=manifest.created_at,
        environment="robocasa",
        task_selector=manifest.task_selector,
        execution_state="queued",
        scheduled=1,
        completed=0,
        metrics=EvaluationMetricsView(scheduled=1),
    )
    (evaluations / ".rpent-evaluations.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "evaluations": [
                    {
                        "evaluation_id": manifest.evaluation_id,
                        "manifest_id": "wrong-manifest-id",
                        "artifact_root": str((evaluations / "run").resolve()),
                        "list_item": item.model_dump(mode="json"),
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="manifest_id"):
        EvaluationRepository((evaluations,))


def test_materialized_list_does_not_project_retained_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))

    def fail_projection(*_args, **_kwargs):
        raise AssertionError("list path projected retained files")

    monkeypatch.setattr(repository, "_evaluation_item", fail_projection)

    assert [item.evaluation_id for item in repository.list_evaluations()] == [
        manifest.evaluation_id
    ]


def test_schema_v3_rejects_mismatched_list_item_identity(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    item = EvaluationListItem(
        evaluation_id="other-evaluation",
        manifest_id=manifest.manifest_id,
        created_at=manifest.created_at,
        environment="robocasa",
        task_selector=manifest.task_selector,
        execution_state="queued",
        scheduled=1,
        completed=0,
        metrics=EvaluationMetricsView(scheduled=1),
    )
    (root.parent / ".rpent-evaluations.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "evaluations": [
                    {
                        "evaluation_id": manifest.evaluation_id,
                        "manifest_id": manifest.manifest_id,
                        "artifact_root": str(root.resolve()),
                        "list_item": item.model_dump(mode="json"),
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="list item evaluation_id"):
        EvaluationRepository((root.parent,))


@pytest.mark.parametrize(
    "variant",
    ("created_at", "environment", "task_selector", "scheduled"),
)
def test_schema_v2_rejects_manifest_metadata_corruption(
    tmp_path: Path,
    variant: str,
) -> None:
    root = tmp_path / "evaluations/run"
    _write_evaluation(root)
    index_path = root.parent / ".rpent-evaluations.json"
    payload = json.loads(index_path.read_text())
    item = payload["evaluations"][0]["list_item"]
    item["execution_state"] = "terminal"
    item[variant] = 2 if variant == "scheduled" else "corrupt"
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"list item {variant}"):
        EvaluationRepository((root.parent,))


@pytest.mark.parametrize(
    "variant",
    (
        "negative_completed",
        "excess_completed",
        "metrics_scheduled",
        "metrics_completed",
    ),
)
def test_schema_v2_rejects_inconsistent_list_item_counts(
    tmp_path: Path,
    variant: str,
) -> None:
    root = tmp_path / "evaluations/run"
    _write_evaluation(root)
    index_path = root.parent / ".rpent-evaluations.json"
    payload = json.loads(index_path.read_text())
    item = payload["evaluations"][0]["list_item"]
    item["execution_state"] = "terminal"
    if variant == "negative_completed":
        item["completed"] = -1
    elif variant == "excess_completed":
        item["completed"] = item["scheduled"] + 1
    elif variant == "metrics_scheduled":
        item["metrics"]["scheduled"] = 0
    else:
        item["metrics"]["completed"] = 1
    index_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="list item counts"):
        EvaluationRepository((root.parent,))


def test_schema_v1_is_rejected_without_migration(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    index_path = root.parent / ".rpent-evaluations.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluations": [
                    {
                        "evaluation_id": manifest.evaluation_id,
                        "manifest_id": manifest.manifest_id,
                        "relative_root": "run",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="schema version"):
        EvaluationRepository((root.parent,))


def test_empty_schema_v1_index_is_rejected_without_migration(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    root.mkdir()
    index_path = root / ".rpent-evaluations.json"
    index_path.write_text(json.dumps({"schema_version": 1, "evaluations": []}))

    with pytest.raises(ValueError, match="schema version"):
        EvaluationRepository((root,))


def test_schema_v2_startup_reconciles_active_rows(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    _write_result(root, _result(manifest, case, "attempt-1"))

    repository = EvaluationRepository((root.parent,))

    item = repository.list_evaluations()[0]
    assert item.execution_state == "terminal"
    assert item.completed == 1
    persisted = json.loads((root.parent / ".rpent-evaluations.json").read_text())
    assert persisted["evaluations"][0]["list_item"] == item.model_dump(mode="json")


def test_schema_v2_startup_trusts_terminal_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    terminal = repository.update_evaluation_list_item(
        manifest.evaluation_id,
        execution_state="terminal",
    )

    def fail_projection(*_args, **_kwargs):
        raise AssertionError("terminal row was reconciled")

    monkeypatch.setattr(EvaluationRepository, "_evaluation_item", fail_projection)

    assert EvaluationRepository((root.parent,)).list_evaluations() == (terminal,)


def test_list_changes_only_after_explicit_materialization(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))

    assert repository.list_evaluations()[0].execution_state == "queued"
    _write_result(root, _result(manifest, case, "attempt-1"))

    assert repository.list_evaluations()[0].execution_state == "queued"

    projected = repository.update_evaluation_list_item(
        manifest.evaluation_id,
        execution_state="terminal",
    )

    assert projected.execution_state == "terminal"
    assert projected.completed == 1
    assert EvaluationRepository((root.parent,)).list_evaluations()[0] == projected


def test_failed_list_item_write_preserves_disk_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    index_path = root.parent / ".rpent-evaluations.json"
    before = index_path.read_bytes()
    _write_result(root, _result(manifest, case, "attempt-1"))

    def fail_write(path: Path, payload: object) -> None:
        raise OSError("volume unavailable")

    monkeypatch.setattr(recording_media, "atomic_write_json", fail_write)

    with pytest.raises(OSError, match="volume unavailable"):
        repository.update_evaluation_list_item(
            manifest.evaluation_id,
            execution_state="terminal",
        )

    assert index_path.read_bytes() == before
    assert repository.list_evaluations()[0].execution_state == "queued"


def test_invalid_list_item_state_preserves_disk_and_memory(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    index_path = root.parent / ".rpent-evaluations.json"
    before = index_path.read_bytes()

    with pytest.raises(ValidationError, match="execution_state"):
        repository.update_evaluation_list_item(
            manifest.evaluation_id,
            execution_state="bogus",  # type: ignore[arg-type]
        )

    assert index_path.read_bytes() == before
    assert repository.list_evaluations()[0].execution_state == "queued"


def test_identical_stateful_registration_skips_disk_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    repository = EvaluationRepository((root.parent,))
    repository.register_evaluation(root, manifest, execution_state="running")
    before = (root.parent / ".rpent-evaluations.json").read_bytes()

    def fail_write(path: Path, payload: object) -> None:
        raise AssertionError("identical registration wrote the sidecar")

    monkeypatch.setattr(recording_media, "atomic_write_json", fail_write)

    repository.register_evaluation(root, manifest, execution_state="running")

    assert (root.parent / ".rpent-evaluations.json").read_bytes() == before
    assert repository.list_evaluations()[0].execution_state == "running"


def test_idempotent_registration_publishes_fresh_boundary_rows(
    tmp_path: Path,
) -> None:
    boundary = tmp_path / "evaluations"
    first, _ = _write_evaluation(
        boundary / "first",
        evaluation_id="evaluation-first",
        persist_index=False,
    )
    second, _ = _write_evaluation(
        boundary / "second",
        evaluation_id="evaluation-second",
        persist_index=False,
    )
    EvaluationRepository((boundary,)).register_evaluation(boundary / "first", first)
    repository_a = EvaluationRepository((boundary,))
    repository_b = EvaluationRepository((boundary,))
    repository_b.register_evaluation(boundary / "second", second)

    repository_a.register_evaluation(boundary / "first", first)

    assert {item.evaluation_id for item in repository_a.list_evaluations()} == {
        "evaluation-first",
        "evaluation-second",
    }


def test_failed_index_write_does_not_publish_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluations = tmp_path / "evaluations"
    repository = EvaluationRepository((evaluations,))
    manifest, _ = _write_evaluation(
        evaluations / "run",
        persist_index=False,
    )

    def fail_write(path: Path, payload: object) -> None:
        raise OSError("volume unavailable")

    monkeypatch.setattr(recording_media, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="volume unavailable"):
        repository.register_evaluation(evaluations / "run", manifest)
    with pytest.raises(KeyError):
        repository.evaluation_metadata(manifest.evaluation_id)


def test_register_evaluation_indexes_preexisting_accepted_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root, persist_index=False)
    result = _result(manifest, case, "accepted-before-registration")
    _append_jsonl(root / "results.jsonl", result.to_payload())
    repository = EvaluationRepository((root.parent,))

    repository.register_evaluation(root, manifest)

    assert repository.authorize_attempt(
        manifest.evaluation_id,
        case.case_id,
        result.attempt_id,
    ).attempt_id == result.attempt_id


def test_cross_process_registration_keeps_every_index_row(tmp_path: Path) -> None:
    boundary = tmp_path / "logs"
    first, _ = _write_evaluation(
        boundary / "one",
        evaluation_id="evaluation-one",
        persist_index=False,
    )
    second, _ = _write_evaluation(
        boundary / "two",
        evaluation_id="evaluation-two",
        persist_index=False,
    )
    context = multiprocessing.get_context("spawn")
    gate = context.Barrier(2)
    processes = [
        context.Process(
            target=_register_after_barrier,
            args=(boundary, root, gate),
        )
        for root in (
            Path(first.runtime["evaluation_root"]),
            Path(second.runtime["evaluation_root"]),
        )
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    assert {
        item.evaluation_id
        for item in EvaluationRepository((boundary,)).list_evaluations()
    } == {"evaluation-one", "evaluation-two"}


def test_cross_process_terminal_updates_keep_every_list_item(tmp_path: Path) -> None:
    boundary = tmp_path / "logs"
    first, _ = _write_evaluation(
        boundary / "one",
        evaluation_id="evaluation-one",
        persist_index=False,
    )
    second, _ = _write_evaluation(
        boundary / "two",
        evaluation_id="evaluation-two",
        persist_index=False,
    )
    repository = EvaluationRepository((boundary,))
    repository.register_evaluation(boundary / "one", first)
    repository.register_evaluation(boundary / "two", second)
    context = multiprocessing.get_context("spawn")
    gate = context.Barrier(2)
    processes = [
        context.Process(
            target=_update_after_barrier,
            args=(boundary, evaluation_id, gate),
        )
        for evaluation_id in ("evaluation-one", "evaluation-two")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    assert {
        item.evaluation_id: item.execution_state
        for item in EvaluationRepository((boundary,)).list_evaluations()
    } == {"evaluation-one": "terminal", "evaluation-two": "terminal"}


def test_registration_persists_only_the_most_specific_configured_root(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "evaluations"
    child = parent / "nested"
    root = child / "run"
    manifest, _ = _write_evaluation(root, persist_index=False)

    repository = EvaluationRepository((parent, child))
    repository.register_evaluation(root, manifest)

    assert not (parent / ".rpent-evaluations.json").exists()
    persisted = json.loads((child / ".rpent-evaluations.json").read_text())
    assert persisted["schema_version"] == 3
    assert persisted["evaluations"][0]["evaluation_id"] == manifest.evaluation_id
    assert persisted["evaluations"][0]["list_item"]["evaluation_id"] == (
        manifest.evaluation_id
    )
    restarted = EvaluationRepository((parent, child))
    assert restarted.evaluation_metadata(manifest.evaluation_id).manifest_id == (
        manifest.manifest_id
    )


def test_existing_registration_writes_its_recorded_owner_with_overlapping_roots(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "evaluations"
    child = parent / "nested"
    root = child / "run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    EvaluationRepository((parent,)).register_evaluation(root, manifest)

    repository = EvaluationRepository((parent, child))
    repository.register_evaluation(root, manifest, execution_state="cancelled")

    parent_payload = json.loads((parent / ".rpent-evaluations.json").read_text())
    assert parent_payload["evaluations"][0]["list_item"]["execution_state"] == (
        "cancelled"
    )
    assert not (child / ".rpent-evaluations.json").exists()
    restarted = EvaluationRepository((parent, child))
    assert restarted.list_evaluations()[0].execution_state == "cancelled"


def test_independent_roots_retain_sidecars_and_reject_cross_root_conflicts(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, _ = _write_evaluation(
        first_root / "run",
        evaluation_id="first",
        persist_index=False,
    )
    second, _ = _write_evaluation(
        second_root / "run",
        evaluation_id="second",
        persist_index=False,
    )
    repository = EvaluationRepository((first_root, second_root))

    repository.register_evaluation(first_root / "run", first)
    first_sidecar = (first_root / ".rpent-evaluations.json").read_bytes()
    repository.register_evaluation(second_root / "run", second)

    assert (first_root / ".rpent-evaluations.json").read_bytes() == first_sidecar
    assert (
        json.loads((second_root / ".rpent-evaluations.json").read_text())[
            "evaluations"
        ][0]["evaluation_id"]
        == "second"
    )

    duplicate_id, _ = _write_evaluation(
        second_root / "duplicate-id",
        evaluation_id="first",
        persist_index=False,
    )
    with pytest.raises(ValueError, match="duplicate evaluation_id"):
        repository.register_evaluation(second_root / "duplicate-id", duplicate_id)

    duplicate_root = replace(second, evaluation_id="other-id")
    with pytest.raises(ValueError, match="root is already registered"):
        repository.register_evaluation(second_root / "run", duplicate_root)

    repository.register_evaluation(first_root / "run", first)
    restarted = EvaluationRepository((first_root, second_root))
    assert {item.evaluation_id for item in restarted.list_evaluations()} == {
        "first",
        "second",
    }


def test_registration_cannot_be_overwritten_by_index_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluations = tmp_path / "evaluations"
    original, _ = _write_evaluation(
        evaluations / "original",
        evaluation_id="original",
    )
    repository = EvaluationRepository((evaluations,))
    added, _ = _write_evaluation(
        evaluations / "added",
        evaluation_id="added",
        persist_index=False,
    )
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    registration_started = threading.Event()
    registration_finished = threading.Event()
    real_discover = repository._discover

    def blocking_discover():
        generation = real_discover()
        refresh_started.set()
        assert release_refresh.wait(timeout=2)
        return generation

    monkeypatch.setattr(repository, "_discover", blocking_discover)
    with ThreadPoolExecutor(max_workers=2) as executor:
        refresh = executor.submit(repository.refresh_index, force=True)
        assert refresh_started.wait(timeout=2)

        def register() -> None:
            registration_started.set()
            repository.register_evaluation(evaluations / "added", added)
            registration_finished.set()

        registration = executor.submit(register)
        assert registration_started.wait(timeout=2)
        assert not registration_finished.wait(timeout=0.1)
        release_refresh.set()
        refresh.result(timeout=2)
        registration.result(timeout=2)

    assert repository.evaluation_metadata(original.evaluation_id).manifest_id == (
        original.manifest_id
    )
    assert repository.evaluation_metadata(added.evaluation_id).manifest_id == (
        added.manifest_id
    )
    restarted = EvaluationRepository((evaluations,))
    assert {item.evaluation_id for item in restarted.list_evaluations()} == {
        "original",
        "added",
    }


def test_list_projects_outcomes_and_artifact_health(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)

    repository = EvaluationRepository((root.parent,))
    item = repository.update_evaluation_list_item(
        manifest.evaluation_id,
        execution_state="terminal",
    )

    assert item.metrics is not None
    assert item.metrics.completed == 1
    assert item.metrics.successful == 1
    assert item.metrics.success_rate == 1.0
    assert item.metrics.transitions == result.step_count


def test_artifact_projection_includes_retained_byte_size(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    retained = attempt / "console.log"
    retained.write_bytes(b"retained log\n")
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"kind": "console_log", "relative_path": "console.log"}],
    )

    artifact = (
        EvaluationRepository((root.parent,))
        .get_case(manifest.evaluation_id, case.case_id)
        .artifacts[0]
    )

    assert artifact.available is True
    assert artifact.byte_size == len(b"retained log\n")


def test_projects_durable_preparation_progress_and_starting_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    progress = PreparationProgress.create(
        root / "preparation.json", manifest.evaluation_id, manifest.manifest_id
    )
    progress.advance("sam3_loading", "loading", detail="loading SAM3")

    detail = repository.get_evaluation(manifest.evaluation_id)

    assert detail.execution_state == "starting"
    assert detail.preparation is not None
    assert detail.preparation.status == "preparing"
    assert detail.preparation.current_phase == "sam3_loading"
    assert detail.preparation.phases[1].status == "loading"
    assert detail.preparation.phases[1].detail == "loading SAM3"
    assert repository.list_evaluations()[0].execution_state == "queued"


@pytest.mark.parametrize(
    ("phase_status", "expected_state"),
    (("failed", "terminal"), ("cancelled", "cancelled")),
)
def test_preparation_terminal_status_overrides_case_state(
    tmp_path: Path, phase_status: str, expected_state: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    progress = PreparationProgress.create(
        root / "preparation.json", manifest.evaluation_id, manifest.manifest_id
    )
    progress.advance("sam3_loading", "loading")
    progress.advance(
        "sam3_loading",
        phase_status,
        error=(
            PreparationError("TimeoutError", "SAM3 did not become ready")
            if phase_status == "failed"
            else None
        ),
    )

    detail = EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)

    assert detail.execution_state == expected_state
    assert detail.preparation is not None
    assert detail.preparation.status == phase_status


def test_evaluation_without_preparation_keeps_legacy_projection(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)

    detail = EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)

    assert detail.preparation is None
    assert detail.execution_state == "queued"


def test_overlapping_configured_roots_do_not_duplicate_an_evaluation(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    root = evaluations / "run"
    _write_evaluation(root)

    items = EvaluationRepository((evaluations, root)).list_evaluations()

    assert len(items) == 1


def test_gets_case_only_through_its_evaluation(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    first, case = _write_evaluation(evaluations / "one", evaluation_id="first")
    second, _ = _write_evaluation(
        evaluations / "two", evaluation_id="second", case=_case(task_name="other")
    )
    repository = EvaluationRepository((evaluations,))

    assert (
        repository.get_evaluation(first.evaluation_id).cases[0].case_id == case.case_id
    )
    with pytest.raises(KeyError):
        repository.get_case(second.evaluation_id, case.case_id)
    with pytest.raises(KeyError):
        repository.get_evaluation(first.manifest_id)


def test_keeps_execution_result_and_artifact_health_separate(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    (attempt / "video.mp4").write_bytes(b"video")
    _write_artifacts(
        attempt,
        manifest,
        case,
        status="degraded",
        issues=["encoder failed"],
        artifacts=[{"kind": "primary_video", "relative_path": "video.mp4"}],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.execution_state == "terminal"
    assert projected.episode_result is not None
    assert projected.episode_result["status"] == "success"
    assert projected.artifact_health is not None
    assert projected.artifact_health.status == "degraded"
    assert projected.artifacts[0].available is True


def test_projects_structured_attempt_and_evaluation_metrics(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = replace(
        _result(manifest, case, "attempt-1"),
        step_count=4,
        action_count=0,
        finished_at="2026-08-06T00:00:03+00:00",
    )
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(attempt, manifest, case, status="degraded", issues=["encoder"])
    usage = _event(manifest, case, result.attempt_id, 1, 0)
    usage["event_type"] = "usage_updated"
    usage["step_id"] = None
    usage["payload"] = {
        "input_tokens": 7,
        "cached_input_tokens": 2,
        "output_tokens": 3,
        "reasoning_tokens": 1,
        "tool_calls": 0,
    }
    _append_jsonl(attempt / "events.jsonl", usage)

    repository = EvaluationRepository((root.parent,))
    projected = repository.get_case(manifest.evaluation_id, case.case_id)
    evaluation = repository.get_evaluation(manifest.evaluation_id)

    assert projected.run_summary is not None
    assert projected.run_summary.status == "success"
    assert projected.run_summary.success is True
    assert projected.run_summary.step_count == 4
    assert projected.run_summary.action_count == 0
    assert projected.run_summary.duration_seconds == 3.0
    assert projected.run_summary.usage.input_tokens == 7
    assert projected.run_summary.usage.output_tokens == 3
    assert projected.run_summary.usage.tool_calls == 0
    assert projected.run_summary.artifact_health is not None
    assert projected.run_summary.artifact_health.status == "degraded"
    assert evaluation.metrics.completed == 1
    assert evaluation.metrics.successful == 1
    assert evaluation.metrics.success_rate == 1.0
    assert evaluation.metrics.transitions == 4
    assert evaluation.metrics.input_tokens == 7
    assert evaluation.metrics.output_tokens == 3
    assert evaluation.metrics.tool_calls == 0


def test_projects_retained_runtime_metrics_from_terminal_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    planning = _event(manifest, case, result.attempt_id, 1, 0)
    planning.update(
        event_type="planning_finished",
        step_id=None,
        payload={"stats": {"elapsed_s": 12.5}},
    )
    rollout = _event(manifest, case, result.attempt_id, 2, 0)
    rollout.update(
        event_type="vla_rollout_completed",
        step_id=None,
        payload={"chunks_used": 7},
    )
    _append_jsonl(attempt / "events.jsonl", planning, rollout)
    frame = attempt / "frame.png"
    frame.write_bytes(b"frame")
    index = attempt / "index.json"
    index.write_text(
        json.dumps(
            {
                "camera": _camera_discovery_payload()["cameras"][0],
                "frames": [
                    {
                        "camera_id": "main",
                        "frame_index": 0,
                        "step_id": 0,
                        "timestamp_s": 0.0,
                        "relative_path": "frame.png",
                    }
                ],
            }
        )
    )
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"kind": "camera_main_index", "relative_path": "index.json"}],
        identity=_camera_discovery_payload(),
    )

    summary = (
        EvaluationRepository((root.parent,))
        .get_case(manifest.evaluation_id, case.case_id)
        .run_summary
    )

    assert summary is not None
    assert summary.planner_elapsed_seconds == 12.5
    assert summary.vla_chunks == 7
    assert summary.retained_frame_count == 1


def test_runtime_event_metrics_reuses_unchanged_journal_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-1")
    planning = _event(manifest, case, attempt.name, 1, 0)
    planning.update(
        event_type="planning_finished",
        step_id=None,
        payload={"stats": {"elapsed_s": 12.5}},
    )
    rollout = _event(manifest, case, attempt.name, 2, 0)
    rollout.update(
        event_type="vla_rollout_completed",
        step_id=None,
        payload={"chunks_used": 7},
    )
    _append_jsonl(attempt / "events.jsonl", planning, rollout)
    repository = EvaluationRepository((root.parent,))
    decode_event = repository_module._decode_event
    decode_calls = 0

    def counting_decode_event(line: bytes, path: Path):
        nonlocal decode_calls
        decode_calls += 1
        return decode_event(line, path)

    monkeypatch.setattr(repository_module, "_decode_event", counting_decode_event)

    first = repository._runtime_event_metrics(
        root,
        manifest.evaluation_id,
        case.case_id,
        attempt.name,
        finalized=True,
    )
    second = repository._runtime_event_metrics(
        root,
        manifest.evaluation_id,
        case.case_id,
        attempt.name,
        finalized=True,
    )

    assert first == second == (12.5, 7)
    assert decode_calls == 2


def test_active_attempt_uses_chunk_and_transition_frame_fallbacks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-live")
    _write_case_projection(attempt, manifest, case)
    first = _event(manifest, case, attempt.name, 1, 0)
    first.update(event_type="vla_chunk_completed", payload={})
    second = _event(manifest, case, attempt.name, 2, 1)
    second.update(event_type="vla_chunk_completed", payload={})
    _append_jsonl(attempt / "events.jsonl", first, second)
    transition = {
        "case_id": case.case_id,
        "attempt_id": attempt.name,
        "episode_id": "episode-live",
        "step_id": 0,
        "frame_index": 0,
        "timestamp_s": 0.0,
        "state": {},
        "action": [],
        "reward": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "operation_id": None,
        "execution_id": None,
        "frames": [
            {
                "camera_id": "main",
                "frame_index": 0,
                "step_id": 0,
                "timestamp_s": 0.0,
                "relative_path": "trajectory/cameras/main/segments/00000000.png",
            }
        ],
    }
    _append_jsonl(attempt / "trajectory/transitions.jsonl", transition)

    summary = (
        EvaluationRepository((root.parent,))
        .get_case(manifest.evaluation_id, case.case_id)
        .run_summary
    )

    assert summary is not None
    assert summary.vla_chunks == 2
    assert summary.retained_frame_count == 1


def test_active_transition_frame_count_over_limit_is_optional(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-live")
    _write_case_projection(attempt, manifest, case)
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, attempt.name, 1, 0),
    )
    frames = [
        {
            "camera_id": "main",
            "frame_index": frame_index,
            "step_id": 0,
            "timestamp_s": 0.0,
            "relative_path": f"frame-{frame_index}.png",
        }
        for frame_index in range(2)
    ]
    _append_jsonl(
        attempt / "trajectory/transitions.jsonl",
        {
            "case_id": case.case_id,
            "attempt_id": attempt.name,
            "episode_id": "episode-live",
            "step_id": 0,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "state": {},
            "action": [],
            "reward": 0.0,
            "success": False,
            "terminated": False,
            "truncated": False,
            "operation_id": None,
            "execution_id": None,
            "frames": frames,
        },
    )

    summary = (
        EvaluationRepository((root.parent,), max_jsonl_records=1)
        .get_case(manifest.evaluation_id, case.case_id)
        .run_summary
    )

    assert summary is not None
    assert summary.retained_frame_count is None


@pytest.mark.parametrize(
    ("started_at", "finished_at"),
    [
        ("not-a-timestamp", "2026-08-06T00:00:03+00:00"),
        ("2026-08-06T00:00:00", "2026-08-06T00:00:03+00:00"),
        ("2026-08-06T00:00:03+00:00", "2026-08-06T00:00:00+00:00"),
    ],
)
def test_attempt_summary_degrades_invalid_duration_to_unavailable(
    tmp_path: Path,
    started_at: str,
    finished_at: str,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    _write_result(
        root,
        replace(
            _result(manifest, case, "attempt-1"),
            started_at=started_at,
            finished_at=finished_at,
        ),
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.run_summary is not None
    assert projected.run_summary.duration_seconds is None


def test_projects_official_scores_diagnostics_qualification_and_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    cases = (
        _case(ordinal=0, task_name="atomic task"),
        _case(ordinal=1, task_name="composite seen task"),
        _case(ordinal=2, task_name="composite unseen task"),
    )
    metadata = {
        "model_name": "policy",
        "policy_family": "VLA",
        "date": "2026-08-09",
        "submission_source": "external",
        "code_url": "N/A",
        "commit_hash": "N/A",
        "paper_link": "N/A",
        "open_source": "no",
        "training_config": {"batch_size": 1, "num_training_steps": 1},
        "notes": "test",
    }
    official_manifest = replace(
        manifest,
        cases=cases,
        policy={
            **manifest.policy,
            "checkpoint": {
                "repo_id": "org/policy",
                "revision": "a" * 40,
            },
        },
        schedule={
            **manifest.schedule,
            "worker_grouping": "task",
            "task_sets": (
                "atomic_seen",
                "composite_seen",
                "composite_unseen",
            ),
            "task_set_membership": {
                "atomic_seen": ("atomic task",),
                "composite_seen": ("composite seen task",),
                "composite_unseen": ("composite unseen task",),
            },
        },
        qualification={
            **manifest.qualification,
            "official_profile": "robocasa-leaderboard-v1",
            "submission_metadata": metadata,
        },
    )
    (root / "evaluation_manifest.json").unlink()
    official_manifest.write_new(root / "evaluation_manifest.json")
    EvaluationRepository((root.parent,)).register_evaluation(root, official_manifest)
    (root / "official_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "available": True,
                "manifest_id": official_manifest.manifest_id,
                "task_success_rates": {case.task_name: 0.0 for case in cases},
                "split_success_rates": {
                    "atomic_seen": 0.0,
                    "composite_seen": 0.0,
                    "composite_unseen": 0.0,
                },
                "overall_success_rate": 0.0,
                "completed_rollouts": 0,
                "scheduled_rollouts": 3,
                "completed_tasks": 0,
                "scheduled_tasks": 3,
                "is_complete": False,
            }
        )
    )
    (root / "rpent_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": official_manifest.manifest_id,
                "selected_cohort": {
                    "rollouts": 0,
                    "scheduled_rollouts": 3,
                    "contract_successes": 0,
                    "contract_success_rate": 0.0,
                    "system_successes": 0,
                    "system_success_rate": 0.0,
                },
                "all_retained_audit": {"retained_attempts": 26},
                "structured_evidence": {"available": False, "issues": []},
            }
        )
    )
    (root / "qualification.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "robocasa-leaderboard-v1",
                "status": "partial",
                "accepted": False,
                "checks": _qualification_checks(False),
            }
        )
    )

    projected = EvaluationRepository((root.parent,)).get_evaluation(
        official_manifest.evaluation_id
    )

    assert projected.official_metrics.available is True
    assert projected.official_metrics.overall_success_rate == 0.0
    assert projected.official_metrics.is_complete is False
    assert projected.rpent_diagnostics.available is True
    assert projected.rpent_diagnostics.system_success_rate == 0.0
    assert projected.qualification.available is True
    assert projected.qualification.status == "partial"
    assert projected.qualification.checks[0].id == "profile_identity"
    assert projected.submission.available is False

    official_path = root / "official_metrics.json"
    valid_official = json.loads(official_path.read_text())
    official_path.write_text(json.dumps({**valid_official, "scheduled_rollouts": 99}))
    drifted = EvaluationRepository((root.parent,)).get_evaluation(
        official_manifest.evaluation_id
    )
    assert drifted.official_metrics.available is False
    assert drifted.rpent_diagnostics.available is True
    official_path.write_text(json.dumps(valid_official))
    qualification_path = root / "qualification.json"
    valid_qualification = json.loads(qualification_path.read_text())
    qualification_path.write_text(
        json.dumps(
            {
                **valid_qualification,
                "status": "qualified",
                "accepted": True,
                "checks": [],
            }
        )
    )
    assert (
        EvaluationRepository((root.parent,))
        .get_evaluation(official_manifest.evaluation_id)
        .qualification.available
        is False
    )
    qualification_path.write_text(json.dumps(valid_qualification))

    (root / "leaderboard_submission.json").write_text(
        json.dumps({"model_name": "policy"})
    )
    projected = EvaluationRepository((root.parent,)).get_evaluation(
        official_manifest.evaluation_id
    )
    assert projected.submission.available is False

    (root / "official_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "available": True,
                "manifest_id": official_manifest.manifest_id,
                "task_success_rates": {case.task_name: 1.0 for case in cases},
                "split_success_rates": {
                    "atomic_seen": 1.0,
                    "composite_seen": 1.0,
                    "composite_unseen": 1.0,
                },
                "overall_success_rate": 1.0,
                "completed_rollouts": 3,
                "scheduled_rollouts": 3,
                "completed_tasks": 3,
                "scheduled_tasks": 3,
                "is_complete": True,
            }
        )
    )
    (root / "qualification.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "robocasa-leaderboard-v1",
                "status": "qualified",
                "accepted": True,
                "checks": _qualification_checks(True),
            }
        )
    )
    submission = {
        **metadata,
        "robocasa_version": "1.0.1",
        "atomic_seen_success": 100.0,
        "composite_seen_success": 100.0,
        "composite_unseen_success": 100.0,
        "checkpoint_url": f"https://huggingface.co/org/policy/tree/{'a' * 40}",
    }
    (root / "leaderboard_submission.json").write_text(json.dumps(submission))
    (root / "reproduction.md").write_text(
        f"# Reproduction\n\nManifest: `{official_manifest.manifest_id}`\n"
    )
    retained = tuple(
        _result(official_manifest, case, f"attempt-{case.ordinal}") for case in cases
    )
    _append_jsonl(root / "results.jsonl", *(result.to_payload() for result in retained))
    (root / "summary.json").write_text(
        json.dumps(aggregate_results(official_manifest, retained).to_payload())
    )
    repository = EvaluationRepository((root.parent,))
    projected = repository.get_evaluation(official_manifest.evaluation_id)
    assert projected.submission.available is True
    assert (
        json.loads(repository.read_submission(official_manifest.evaluation_id))
        == submission
    )
    submission_path = root / "leaderboard_submission.json"
    valid_submission = submission_path.read_bytes()
    metadata_stat = submission_path.stat()
    original_views = repository._official_views

    def replace_after_validation(*args):
        projected = original_views(*args)
        drift = valid_submission.replace(b'"policy"', b'"drift!"', 1)
        assert len(drift) == len(valid_submission)
        replacement = root / "replacement-submission.json"
        replacement.write_bytes(drift)
        os.utime(
            replacement,
            ns=(metadata_stat.st_atime_ns, metadata_stat.st_mtime_ns),
        )
        os.replace(replacement, submission_path)
        return projected

    monkeypatch.setattr(repository, "_official_views", replace_after_validation)
    with pytest.raises(ValueError, match="metadata drift"):
        repository.read_submission(official_manifest.evaluation_id)
    monkeypatch.setattr(repository, "_official_views", original_views)
    submission_path.write_bytes(valid_submission)
    alternate = root / "alternate-submission.json"
    (root / "leaderboard_submission.json").replace(alternate)
    (root / "leaderboard_submission.json").symlink_to(alternate.name)
    assert (
        EvaluationRepository((root.parent,))
        .get_evaluation(official_manifest.evaluation_id)
        .submission.available
        is False
    )


@pytest.mark.parametrize(
    "name",
    ("official_metrics.json", "rpent_diagnostics.json", "qualification.json"),
)
def test_malformed_official_projection_is_unavailable_without_summary_fallback(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root, persist_index=False)
    official_manifest = replace(
        manifest,
        schedule={**manifest.schedule, "worker_grouping": "task"},
        qualification={**manifest.qualification, "official_profile": "official-v1"},
    )
    (root / "evaluation_manifest.json").unlink()
    official_manifest.write_new(root / "evaluation_manifest.json")
    EvaluationRepository((root.parent,)).register_evaluation(root, official_manifest)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "manifest_id": official_manifest.manifest_id,
                "evaluation_id": official_manifest.evaluation_id,
                "scheduled": 1,
                "completed": 1,
                "successful": 1,
                "success_rate": 1.0,
                "errors": 0,
                "timeouts": 0,
                "status_counts": {"success": 1},
                "per_task": {
                    "pick mug": {
                        "scheduled": 1,
                        "completed": 1,
                        "successful": 1,
                        "success_rate": 1.0,
                    }
                },
            }
        )
    )
    (root / name).write_text("{broken")

    projected = EvaluationRepository((root.parent,)).get_evaluation(
        official_manifest.evaluation_id
    )

    assert projected.summary is not None
    assert projected.summary.success_rate == 1.0
    assert projected.official_metrics.available is False
    assert projected.rpent_diagnostics.available is False
    assert projected.qualification.available is False


def test_video_never_implies_episode_success(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1", status="error")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    (attempt / "video.mp4").write_bytes(b"video")
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"kind": "primary_video", "relative_path": "video.mp4"}],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.episode_result["status"] == "error"
    assert projected.episode_result["simulator_success"] is False


def test_latest_ledger_attempt_is_authoritative(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    old = _result(manifest, case, "attempt-old")
    latest = _result(manifest, case, "attempt-latest", status="error")
    _append_jsonl(root / "results.jsonl", old.to_payload(), latest.to_payload())
    old_root = _attempt_root(root, case.case_id, old.attempt_id)
    _write_artifacts(old_root, manifest, case)
    latest_root = _attempt_root(root, case.case_id, latest.attempt_id)
    _write_artifacts(latest_root, manifest, case, status="degraded", issues=["latest"])

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.attempt_id == "attempt-latest"
    assert projected.episode_result["status"] == "error"
    assert projected.artifact_health.issues == ("recording artifact issue",)


def test_case_detail_selects_one_retained_attempt_without_cross_attempt_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    old = _result(manifest, case, "attempt-old")
    latest = _result(manifest, case, "attempt-latest", status="error")
    _append_jsonl(root / "results.jsonl", old.to_payload(), latest.to_payload())
    old_root = _attempt_root(root, case.case_id, old.attempt_id)
    latest_root = _attempt_root(root, case.case_id, latest.attempt_id)
    _write_artifacts(old_root, manifest, case, status="complete")
    _write_artifacts(
        latest_root,
        manifest,
        case,
        status="degraded",
        issues=["latest-only"],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=old.attempt_id,
    )

    assert projected.attempt_id == old.attempt_id
    assert projected.episode_result is not None
    assert projected.episode_result["status"] == "success"
    assert projected.artifact_health is not None
    assert projected.artifact_health.status == "complete"
    assert [item.attempt_id for item in projected.attempts] == [
        old.attempt_id,
        latest.attempt_id,
    ]
    assert projected.attempts[0].summary.status == "success"
    assert projected.attempts[1].summary.status == "error"


def test_new_resultless_attempt_overrides_old_terminal_ledger_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    old = _result(manifest, case, "attempt-old")
    _write_result(root, old)
    active = _attempt_root(root, case.case_id, "attempt-new")
    _write_case_projection(
        active, manifest, case, started_at="2026-08-06T00:00:03+00:00"
    )
    _append_jsonl(active / "events.jsonl", _event(manifest, case, "attempt-new", 1, 7))

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.attempt_id == "attempt-new"
    assert projected.execution_state == "running"
    assert projected.episode_result is None
    assert projected.current_step == 7
    assert projected.started_at == "2026-08-06T00:00:03+00:00"
    assert projected.finished_at is None


def test_terminal_case_projects_stable_result_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    _write_result(root, _result(manifest, case, "attempt-1"))

    projected = (
        EvaluationRepository((root.parent,))
        .get_evaluation(manifest.evaluation_id)
        .cases[0]
    )

    assert projected.started_at == "2026-08-06T00:00:00+00:00"
    assert projected.finished_at == "2026-08-06T00:00:02+00:00"


def test_multiple_resultless_attempts_with_same_timestamp_are_unavailable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    for attempt_id, step in (("attempt-z", 9), ("attempt-a", 1)):
        attempt = _attempt_root(root, case.case_id, attempt_id)
        _write_case_projection(
            attempt, manifest, case, started_at="2026-08-06T00:00:03+00:00"
        )
        _append_jsonl(
            attempt / "events.jsonl",
            _event(manifest, case, attempt_id, 1, step),
        )

    repository = EvaluationRepository((root.parent,))
    with pytest.raises(CurrentAttemptUnavailable, match="unavailable"):
        repository.get_case(manifest.evaluation_id, case.case_id)


def test_old_orphan_attempt_does_not_override_newer_terminal_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    latest = _result(manifest, case, "attempt-terminal")
    _write_result(root, latest)
    orphan = _attempt_root(root, case.case_id, "attempt-orphan")
    _write_case_projection(
        orphan, manifest, case, started_at="2026-08-06T00:00:01+00:00"
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.attempt_id == "attempt-terminal"
    assert projected.episode_result["status"] == "success"


def test_resultless_attempts_with_ambiguous_newest_are_unavailable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    for attempt_id, started_at in (
        ("attempt-z-old", "2026-08-06T00:00:01+00:00"),
        ("attempt-a", "2026-08-06T00:00:03+00:00"),
        ("attempt-z", "2026-08-06T00:00:03+00:00"),
    ):
        attempt = _attempt_root(root, case.case_id, attempt_id)
        _write_case_projection(attempt, manifest, case, started_at=started_at)

    repository = EvaluationRepository((root.parent,))
    with pytest.raises(CurrentAttemptUnavailable, match="unavailable"):
        repository.get_case(manifest.evaluation_id, case.case_id)


def test_legacy_attempt_uses_case_projection_mtime_for_chronology(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    latest = _result(manifest, case, "attempt-terminal")
    _write_result(root, latest)
    legacy = _attempt_root(root, case.case_id, "attempt-legacy")
    _write_case_projection(legacy, manifest, case)
    timestamp = 1_785_979_203  # 2026-08-06T00:00:03+00:00
    os.utime(legacy / "case.json", (timestamp, timestamp))

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.attempt_id == "attempt-legacy"
    assert projected.episode_result is None


def test_events_ignore_non_newline_tail_and_report_current_step(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    first = _event(manifest, case, result.attempt_id, 1, 4)
    incomplete = _event(manifest, case, result.attempt_id, 2, 99)
    _append_jsonl(
        attempt / "events.jsonl",
        first,
        tail=json.dumps(incomplete).encode(),
    )

    repository = EvaluationRepository((root.parent,))
    page = repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    projected = repository.get_case(manifest.evaluation_id, case.case_id)

    assert [event["sequence"] for event in page.events] == [1]
    assert page.last_sequence == 1
    assert page.incomplete_tail is True
    assert projected.current_step == 4
    assert projected.last_sequence == 1


def test_current_step_skips_tail_events_without_progress(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    transition = _event(manifest, case, result.attempt_id, 1, 4)
    null_events = []
    for sequence in range(2, 10):
        event = _event(manifest, case, result.attempt_id, sequence, 0)
        event["step_id"] = None
        event["event_type"] = "planning_note"
        null_events.append(event)
    _append_jsonl(attempt / "events.jsonl", transition, *null_events)

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.execution_state == "terminal"
    assert projected.current_step == 4
    assert projected.last_sequence == 9


def test_operation_completed_last_step_is_progress(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    event = _event(manifest, case, result.attempt_id, 1, 0)
    event["step_id"] = None
    event["event_type"] = "operation_completed"
    event["payload"] = {"last_step_id": 6}
    _append_jsonl(attempt / "events.jsonl", event)

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.current_step == 6


def test_operations_project_one_summary_per_physical_tool_call(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    started = _event(manifest, case, result.attempt_id, 1, 0)
    started.update(
        event_type="operation_started",
        step_id=None,
        operation_id="operation-1",
        execution_id="execution-1",
        payload={"tool": "move_to_pose"},
        recorded_at="2026-08-06T00:00:01+00:00",
    )
    first = _event(manifest, case, result.attempt_id, 2, 10)
    first.update(operation_id="operation-1", execution_id="execution-1")
    second = _event(manifest, case, result.attempt_id, 3, 11)
    second.update(operation_id="operation-1", execution_id="execution-1")
    completed = _event(manifest, case, result.attempt_id, 4, 0)
    completed.update(
        event_type="operation_completed",
        step_id=None,
        operation_id="operation-1",
        execution_id="execution-1",
        payload={
            "tool": "move_to_pose",
            "first_step_id": 10,
            "last_step_id": 11,
            "executed_actions": 2,
            "result": {
                "final_dist_m": 0.31,
                "steps_used": 2,
                "max_steps": 1200,
            },
        },
        recorded_at="2026-08-06T00:00:03.400000+00:00",
    )
    clip = _event(manifest, case, result.attempt_id, 5, 0)
    clip.update(
        event_type="action_clip_available",
        step_id=None,
        payload={"artifact_id": "action-step-1", "step": 1},
    )
    _append_jsonl(attempt / "events.jsonl", started, first, second, completed, clip)

    page = EvaluationRepository((root.parent,)).operations(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
    )

    assert page.first_step_id == 10
    assert page.latest_step_id == 11
    assert page.last_sequence == 5
    assert page.incomplete_tail is False
    assert [operation.model_dump() for operation in page.operations] == [
        {
            "operation_id": "operation-1",
            "execution_id": "execution-1",
            "tool": "move_to_pose",
            "plan_step_id": None,
            "status": "succeeded",
            "started_sequence": 1,
            "finished_sequence": 4,
            "first_step_id": 10,
            "last_step_id": 11,
            "executed_actions": 2,
            "result": {
                "final_dist_m": 0.31,
                "steps_used": 2,
                "max_steps": 1200,
            },
            "elapsed_seconds": 2.4,
            "error": None,
            "clip_artifact_id": "action-step-1",
            "has_physical_transitions": True,
            "camera_clips": (),
        }
    ]


def test_plan_step_operation_binding_does_not_require_verification(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events = []
    for sequence, operation_id in enumerate(("operation-1", "operation-2"), 1):
        started = _event(manifest, case, result.attempt_id, sequence, 0)
        started.update(
            event_type="operation_started",
            step_id=None,
            operation_id=operation_id,
            execution_id=f"execution-{sequence}",
            payload={"tool": "move_to_pose", "plan_step_id": "step-1"},
        )
        events.append(started)
    _append_jsonl(attempt / "events.jsonl", *events)

    page = EvaluationRepository((root.parent,)).operations(
        manifest.evaluation_id, case.case_id, attempt_id=result.attempt_id
    )

    assert [operation.plan_step_id for operation in page.operations] == [
        "step-1",
        "step-1",
    ]


def test_operations_ignore_transitions_without_operation_identity(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    unowned = _event(manifest, case, result.attempt_id, 1, 7)
    unowned["payload"] = {
        "transition": {
            "operation_id": "nested-operation-id",
            "state": {"log": {"command": {"action": "close_gripper"}}}
        }
    }
    _append_jsonl(attempt / "events.jsonl", unowned)

    page = EvaluationRepository((root.parent,)).operations(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
    )

    assert page.operations == ()
    assert page.first_step_id is None
    assert page.latest_step_id is None


def test_events_reject_foreign_evaluation_identity(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    foreign = _event(manifest, case, result.attempt_id, 1, 4)
    foreign["evaluation_id"] = "foreign"
    _append_jsonl(attempt / "events.jsonl", foreign)

    with pytest.raises(ValueError, match="identity"):
        EvaluationRepository((root.parent,)).events_since(
            manifest.evaluation_id, case.case_id, since=0
        )


def test_projects_queued_and_running_without_a_result(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))

    queued = repository.get_case(manifest.evaluation_id, case.case_id)
    assert queued.execution_state == "queued"
    assert queued.last_sequence == 0

    from rpent.evaluation.attempt_registration import register_attempt

    register_attempt(
        manifest, case, "attempt-live", "2026-08-06T00:00:03+00:00"
    )
    attempt = root / "cases" / case.case_id / "attempts" / "attempt-live"
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, "attempt-live", 1, 3),
    )

    running = repository.get_case(manifest.evaluation_id, case.case_id)
    assert running.execution_state == "running"
    assert running.episode_result is None
    assert running.current_step == 3
    assert running.last_sequence == 1


def test_active_recording_discovery_is_strict_and_cannot_override_artifact_health(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-live")
    _write_case_projection(attempt, manifest, case)
    _write_recording_discovery(attempt, manifest, case)

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.recording_profile is not None
    assert projected.recording_profile.model_dump(mode="json") == {
        "mode": "all",
        "camera_ids": [],
    }
    assert projected.retained_camera_ids == ("main",)
    assert projected.artifacts == ()
    assert projected.artifact_health == ArtifactHealthView(
        status="unavailable", issues=("artifacts.json is missing",)
    )

    _write_recording_discovery(
        attempt, manifest, case, evaluation_id="foreign-evaluation"
    )
    degraded = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )
    assert degraded.recording_profile is None
    assert degraded.cameras == ()
    assert degraded.retained_camera_ids == ()
    assert degraded.artifact_health == projected.artifact_health


def test_historical_case_projects_data_schema_from_binding_and_descriptors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-live")
    _write_case_projection(attempt, manifest, case)
    _write_recording_discovery(attempt, manifest, case, camera_id="main")

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.data_schema.schema_version == 1
    assert [item.id for item in projected.data_schema.cameras] == ["main"]
    assert projected.data_schema.action.shape == (12,)
    assert projected.data_schema.action.dimensions[0].name == "action.base_motion[0]"
    assert projected.data_schema.state.dimensions[14].name == "state.gripper_qpos[0]"
    assert projected.data_schema.signals[0].id == "environment.success"


def test_finalized_discovery_corruption_degrades_health_but_keeps_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    discovery = _camera_discovery_payload("main")
    discovery["retained_camera_ids"] = ["foreign-camera"]
    _write_artifacts(
        attempt,
        manifest,
        case,
        status="complete",
        artifacts=[{"kind": "events", "relative_path": "events.jsonl"}],
        identity=discovery,
    )
    (attempt / "events.jsonl").write_text("")

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.episode_result == result.to_payload()
    assert projected.recording_profile is None
    assert projected.cameras == ()
    assert projected.retained_camera_ids == ()
    assert [artifact.kind for artifact in projected.artifacts] == ["events"]
    assert projected.artifact_health is not None
    assert projected.artifact_health.status == "degraded"
    assert "recording discovery unavailable" in projected.artifact_health.issues
    assert projected.run_summary is not None
    assert projected.run_summary.artifact_health == projected.artifact_health


def test_artifact_health_does_not_expose_oserror_paths(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"kind": "secret", "relative_path": "x" * 300}],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.artifact_health == ArtifactHealthView(
        status="unavailable", issues=("artifact metadata unavailable",)
    )


def test_cache_key_uses_path_mtime_ns_and_size(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events_path = attempt / "events.jsonl"
    _append_jsonl(events_path, _event(manifest, case, result.attempt_id, 1, 4))
    repository = EvaluationRepository((root.parent,))
    assert repository.get_case(manifest.evaluation_id, case.case_id).current_step == 4
    first_stat = events_path.stat()

    _append_jsonl(events_path, _event(manifest, case, result.attempt_id, 1, 5))
    os.utime(
        events_path,
        ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns + 1_000_000),
    )

    assert repository.get_case(manifest.evaluation_id, case.case_id).current_step == 5
    assert not any(key[0] == events_path.resolve() for key in repository._cache)
    result_keys = [
        key for key in repository._cache if key[0] == (root / "results.jsonl").resolve()
    ]
    assert result_keys == [
        (
            (root / "results.jsonl").resolve(),
            (root / "results.jsonl").stat().st_dev,
            (root / "results.jsonl").stat().st_ino,
            (root / "results.jsonl").stat().st_mtime_ns,
            (root / "results.jsonl").stat().st_size,
        )
    ]


def test_inode_generation_reloads_same_size_atomic_replacement(tmp_path: Path) -> None:
    boundary = tmp_path / "evaluations"
    path = boundary / "value.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"value": 1}')
    repository = EvaluationRepository((boundary,))
    assert repository._cached(path, boundary, json.loads, max_bytes=128) == {
        "value": 1
    }
    original = path.stat()
    replacement = boundary / "replacement.json"
    replacement.write_text('{"value": 2}')
    os.utime(
        replacement,
        ns=(replacement.stat().st_atime_ns, original.st_mtime_ns),
    )
    os.replace(replacement, path)
    current = path.stat()

    assert (current.st_size, current.st_mtime_ns) == (
        original.st_size,
        original.st_mtime_ns,
    )
    assert current.st_ino != original.st_ino
    assert repository._cached(path, boundary, json.loads, max_bytes=128) == {
        "value": 2
    }


def test_index_fingerprint_reports_each_configured_index(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest, _ = _write_evaluation(
        first / "run",
        persist_index=False,
    )
    repository = EvaluationRepository((first, second))

    assert repository.index_fingerprint() == (None, None)

    repository.register_evaluation(first / "run", manifest)
    info = (first / ".rpent-evaluations.json").stat()
    assert repository.index_fingerprint() == (
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
        None,
    )


def test_concurrent_cached_loads_publish_one_valid_value(tmp_path: Path) -> None:
    boundary = tmp_path / "evaluations"
    path = boundary / "value.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"value": 1}')
    repository = EvaluationRepository((boundary,))
    first_loader_entered = threading.Event()
    release_first_loader = threading.Event()
    loader_calls = 0
    loader_lock = threading.Lock()

    def coordinated_loader(raw: bytes) -> dict[str, int]:
        nonlocal loader_calls
        with loader_lock:
            loader_calls += 1
            call = loader_calls
        if call == 1:
            first_loader_entered.set()
            assert release_first_loader.wait(timeout=2)
        return {"value": call, "bytes": len(raw)}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            repository._cached,
            path,
            boundary,
            coordinated_loader,
            max_bytes=128,
        )
        assert first_loader_entered.wait(timeout=2)
        second = executor.submit(
            repository._cached,
            path,
            boundary,
            coordinated_loader,
            max_bytes=128,
        )
        second_value = second.result(timeout=2)
        release_first_loader.set()
        first_value = first.result(timeout=2)

    assert first_value == second_value == {"value": 2, "bytes": len(path.read_bytes())}
    assert len(repository._cache) == 1


def test_delayed_old_cache_parse_cannot_evict_a_newer_generation(
    tmp_path: Path,
) -> None:
    boundary = tmp_path / "evaluations"
    path = boundary / "value.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"generation": 1}')
    old_stat = path.stat()
    repository = EvaluationRepository((boundary,))
    old_parse_started = threading.Event()
    release_old_parse = threading.Event()

    def coordinated_loader(raw: bytes) -> dict[str, int]:
        loaded = json.loads(raw)
        if loaded["generation"] == 1:
            old_parse_started.set()
            assert release_old_parse.wait(timeout=2)
        return loaded

    with ThreadPoolExecutor(max_workers=1) as executor:
        old = executor.submit(
            repository._cached,
            path,
            boundary,
            coordinated_loader,
            max_bytes=128,
        )
        assert old_parse_started.wait(timeout=2)
        path.write_text('{"generation": 2}')
        current_stat = path.stat()
        os.utime(
            path,
            ns=(current_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000),
        )
        try:
            newer = repository._cached(
                path,
                boundary,
                coordinated_loader,
                max_bytes=128,
            )
        finally:
            release_old_parse.set()
        delayed = old.result(timeout=2)

    current_stat = path.stat()
    current_key = (
        path.resolve(),
        current_stat.st_dev,
        current_stat.st_ino,
        current_stat.st_mtime_ns,
        current_stat.st_size,
    )
    assert delayed == newer == {"generation": 2}
    assert repository._cache == {current_key: {"generation": 2}}


def test_concurrent_event_pages_serialize_checkpoint_mutation(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events_path = attempt / "events.jsonl"
    _append_jsonl(
        events_path,
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 5)
        ),
    )
    repository = EvaluationRepository((root.parent,), max_event_page_size=2)
    checkpoint_barrier = threading.Barrier(2)

    class CoordinatedOffsets(OrderedDict):
        def __iter__(self):
            keys = tuple(super().__iter__())

            def coordinated_keys():
                yield from keys
                try:
                    checkpoint_barrier.wait(timeout=0.25)
                except threading.BrokenBarrierError:
                    pass

            return coordinated_keys()

    repository._event_offsets = CoordinatedOffsets(
        {
            (
                events_path.resolve(),
                0,
                0,
                0,
                0,
                0,
                0,
            ): 0
        }
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        pages = tuple(
            executor.map(
                lambda _: repository.events_since(
                    manifest.evaluation_id, case.case_id, since=0
                ),
                range(2),
            )
        )

    assert all([event["sequence"] for event in page.events] == [1, 2] for page in pages)
    assert len(repository._event_offsets) <= repository.event_checkpoint_capacity


def test_missing_artifact_is_projected_without_hiding_result(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        status="partial",
        issues=["video missing"],
        artifacts=[{"kind": "primary_video", "relative_path": "missing.mp4"}],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.episode_result["status"] == "success"
    assert projected.artifact_health.status == "partial"
    assert projected.artifacts[0].available is False


def test_qualification_health_survives_missing_artifact_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    (attempt / "qualification.json").write_text(
        json.dumps(
            {
                "artifact_health": {
                    "status": "unavailable",
                    "issues": ["/Users/alice/private/model-key"],
                }
            }
        )
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.episode_result["status"] == "success"
    assert projected.artifact_health.status == "unavailable"
    assert projected.artifact_health.issues == ("recording artifact issue",)


def test_legacy_artifact_health_is_sanitized_before_api_projection(tmp_path: Path):
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        status="degraded",
        issues=["/Users/alice/private/model-key"],
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.artifact_health is not None
    assert projected.artifact_health.issues == ("recording artifact issue",)


@pytest.mark.parametrize(
    ("artifact_id", "kind"),
    [
        ("health-alias", "health"),
        ("health", "health-kind-alias"),
        ("health-alias", "health-kind-alias"),
    ],
)
def test_open_health_artifact_sanitizes_aliases_by_canonical_path(
    tmp_path: Path, artifact_id: str, kind: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "relative_path": "health.jsonl",
            }
        ],
    )
    secret = "/Users/alice/private/model-key"
    source = attempt / "health.jsonl"
    _append_jsonl(
        source,
        {
            "evaluation_id": manifest.evaluation_id,
            "case_id": case.case_id,
            "attempt_id": result.attempt_id,
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": f"camera failed: {secret}",
        },
    )
    original = source.read_bytes()

    with EvaluationRepository((root.parent,)).open_artifact(
        manifest.evaluation_id, case.case_id, artifact_id
    ) as stream:
        sanitized = stream.read()

    assert secret.encode() not in sanitized
    assert b'"issue": "recording artifact issue"' in sanitized
    assert source.read_bytes() == original


def test_duplicate_health_path_aliases_are_all_sanitized(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": artifact_id,
                "kind": f"kind-{index}",
                "relative_path": "health.jsonl",
            }
            for index, artifact_id in enumerate(("health-one", "health-two"))
        ],
    )
    secret = "/Users/alice/private/model-key"
    source = attempt / "health.jsonl"
    _append_jsonl(
        source,
        {
            "evaluation_id": manifest.evaluation_id,
            "case_id": case.case_id,
            "attempt_id": result.attempt_id,
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "issue": secret,
        },
    )
    repository = EvaluationRepository((root.parent,))

    for artifact_id in ("health-one", "health-two"):
        with repository.open_artifact(
            manifest.evaluation_id, case.case_id, artifact_id
        ) as stream:
            assert secret.encode() not in stream.read()


@pytest.mark.parametrize("failure", ["malformed", "incomplete", "bytes", "records"])
def test_open_health_artifact_rejects_invalid_or_unbounded_journal(
    tmp_path: Path, failure: str
) -> None:
    root = tmp_path / f"evaluations/{failure}"
    manifest, case = _write_evaluation(root, evaluation_id=f"evaluation-{failure}")
    result = _result(manifest, case, "attempt-final")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "health-alias",
                "kind": "health-kind-alias",
                "relative_path": "health.jsonl",
            }
        ],
    )
    row = {
        "evaluation_id": manifest.evaluation_id,
        "case_id": case.case_id,
        "attempt_id": result.attempt_id,
        "recorded_at": "2026-08-07T00:00:00+00:00",
        "issue": "camera extraction failed",
    }
    source = attempt / "health.jsonl"
    if failure == "malformed":
        source.write_bytes(b"not-json\n")
    elif failure == "incomplete":
        source.write_text(json.dumps(row))
    else:
        if failure == "bytes":
            row["issue"] = "x" * 8192
        _append_jsonl(source, row, row)
    repository = EvaluationRepository(
        (root.parent,),
        max_jsonl_bytes=4096 if failure == "bytes" else 1024 * 1024,
        max_jsonl_records=1 if failure == "records" else 100,
    )

    with pytest.raises(OSError, match="health journal unavailable"):
        repository.open_artifact(manifest.evaluation_id, case.case_id, "health-alias")


def test_resolve_artifact_accepts_only_declared_safe_existing_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    media = attempt / "media/video.mp4"
    media.parent.mkdir()
    media.write_bytes(b"video")
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "camera-main-video",
                "kind": "camera_video",
                "relative_path": "media/video.mp4",
            }
        ],
    )
    repository = EvaluationRepository((root.parent,))

    assert (
        repository.resolve_artifact(
            manifest.evaluation_id, case.case_id, "camera-main-video"
        )
        == media.resolve()
    )
    with pytest.raises(KeyError):
        repository.resolve_artifact(manifest.evaluation_id, case.case_id, "unknown")


@pytest.mark.parametrize("relative_path", ["../secret", "/tmp/secret"])
def test_resolve_artifact_rejects_unsafe_declared_paths(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "artifact_id": "unsafe",
                "kind": "log",
                "relative_path": relative_path,
            }
        ],
    )

    with pytest.raises(ValueError):
        EvaluationRepository((root.parent,)).resolve_artifact(
            manifest.evaluation_id, case.case_id, "unsafe"
        )


def test_resolve_artifact_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (attempt / "escape").symlink_to(outside)
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"artifact_id": "escape", "kind": "log", "relative_path": "escape"}],
    )

    with pytest.raises(ValueError):
        EvaluationRepository((root.parent,)).resolve_artifact(
            manifest.evaluation_id, case.case_id, "escape"
        )


def test_invalid_manifest_is_not_discovered(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/bad"
    root.mkdir(parents=True)
    (root / "evaluation_manifest.json").write_text(
        json.dumps({"evaluation_id": "spoofed", "manifest_id": "wrong"})
    )

    assert EvaluationRepository((tmp_path / "evaluations",)).list_evaluations() == ()


def test_versioned_completed_fixture_is_readable(tmp_path: Path) -> None:
    target = tmp_path / "evaluations/completed"
    shutil.copytree(FIXTURE, target)
    manifest = EvaluationManifest.load(target / "evaluation_manifest.json")
    EvaluationRepository((target.parent,)).register_evaluation(target, manifest)

    evaluation = EvaluationRepository((target.parent,)).list_evaluations()[0]

    assert evaluation.evaluation_id == "fixture-evaluation"
    assert evaluation.execution_state == "terminal"


@pytest.mark.parametrize("name", ["results.jsonl", "summary.json"])
def test_evaluation_files_cannot_symlink_outside_root(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    outside = tmp_path / name
    outside.write_text("{}\n")
    (root / name).symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)


def test_preparation_cannot_claim_a_foreign_evaluation(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    PreparationProgress.create(
        root / "preparation.json", "foreign-evaluation", manifest.manifest_id
    )

    with pytest.raises(ValueError, match="evaluation_id"):
        EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)


def test_preparation_file_cannot_symlink_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    outside = tmp_path / "preparation.json"
    PreparationProgress.create(outside, manifest.evaluation_id, manifest.manifest_id)
    (root / "preparation.json").symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)


def test_event_journal_cannot_symlink_outside_evaluation(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    outside = tmp_path / "events.jsonl"
    _append_jsonl(outside, _event(manifest, case, result.attempt_id, 1, 99))
    (attempt / "events.jsonl").symlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        EvaluationRepository((root.parent,)).get_case(
            manifest.evaluation_id, case.case_id
        )


def test_artifact_manifest_symlink_is_unavailable_not_decoded(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    outside = tmp_path / "artifacts.json"
    _write_artifacts(outside.parent, manifest, case)
    (attempt / "artifacts.json").symlink_to(outside)

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.artifact_health.status == "unavailable"
    assert projected.artifacts == ()


def test_identityless_artifact_manifest_is_unavailable(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    (attempt / "video.mp4").write_bytes(b"video")
    (attempt / "artifacts.json").write_text(
        json.dumps(
            {
                "health": {"status": "complete", "issues": []},
                "artifacts": [{"kind": "video", "relative_path": "video.mp4"}],
            }
        )
    )

    projected = EvaluationRepository((root.parent,)).get_case(
        manifest.evaluation_id, case.case_id
    )

    assert projected.artifact_health.status == "unavailable"
    assert projected.artifacts == ()


@pytest.mark.parametrize("corrupt_name", ["results.jsonl", "events.jsonl"])
def test_list_isolates_corrupt_evaluation(tmp_path: Path, corrupt_name: str) -> None:
    evaluations = tmp_path / "evaluations"
    healthy, _ = _write_evaluation(evaluations / "healthy", evaluation_id="healthy")
    broken_root = evaluations / "broken"
    broken, case = _write_evaluation(broken_root, evaluation_id="broken")
    if corrupt_name == "results.jsonl":
        (broken_root / corrupt_name).write_text("not-json\n")
    else:
        result = _result(broken, case, "attempt-1")
        _write_result(broken_root, result)
        attempt = _attempt_root(broken_root, case.case_id, result.attempt_id)
        (attempt / corrupt_name).write_text("not-json\n")

    repository = EvaluationRepository((evaluations,))

    assert [item.evaluation_id for item in repository.list_evaluations()] == [
        broken.evaluation_id,
        healthy.evaluation_id,
    ]
    if corrupt_name == "results.jsonl":
        with pytest.raises(ValueError):
            repository.get_evaluation(broken.evaluation_id)
    else:
        projected = repository.get_evaluation(broken.evaluation_id)
        assert projected.cases[0].attempt_id == "attempt-1"
        assert projected.cases[0].artifact_health.status == "unavailable"


def test_list_isolates_recursively_deep_json(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    healthy, _ = _write_evaluation(evaluations / "healthy", evaluation_id="healthy")
    broken_root = evaluations / "broken"
    broken, _ = _write_evaluation(broken_root, evaluation_id="broken")
    (broken_root / "results.jsonl").write_text(
        '{"nested":' + "[" * 2_000 + "0" + "]" * 2_000 + "}\n"
    )
    repository = EvaluationRepository((evaluations,))

    assert [item.evaluation_id for item in repository.list_evaluations()] == [
        broken.evaluation_id,
        healthy.evaluation_id,
    ]
    with pytest.raises(ValueError, match="JSON"):
        repository.get_evaluation(broken.evaluation_id)


def test_summary_rejects_foreign_manifest_and_mutation_cannot_poison_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    summary_path = root / "summary.json"
    payload = {
        "manifest_id": "foreign",
        "evaluation_id": manifest.evaluation_id,
        "scheduled": 1,
        "completed": 0,
        "successful": 0,
        "success_rate": 0.0,
        "errors": 0,
        "timeouts": 0,
        "status_counts": {"success": 0, "error": 0},
        "per_task": {
            "pick mug": {
                "scheduled": 1,
                "completed": 0,
                "successful": 0,
                "success_rate": 0.0,
            }
        },
    }
    summary_path.write_text(json.dumps(payload))
    repository = EvaluationRepository((root.parent,))

    with pytest.raises(ValueError, match="summary identity"):
        repository.get_evaluation(manifest.evaluation_id)

    payload["manifest_id"] = manifest.manifest_id
    summary_path.write_text(json.dumps(payload))
    first = repository.get_evaluation(manifest.evaluation_id)
    assert isinstance(first.summary, EvaluationSummaryView)
    with pytest.raises(ValidationError, match="frozen"):
        first.summary.completed = 99
    second = repository.get_evaluation(manifest.evaluation_id)
    assert second.summary.completed == 0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"scheduled": "1"}, "integer"),
        ({"status_counts": {"success": -1}}, "greater than or equal"),
        ({"scheduled": 0, "completed": 1}, "completed"),
    ],
)
def test_summary_strictly_rejects_coercion_negative_and_bad_totals(
    tmp_path: Path, change: dict, match: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    payload = {
        "manifest_id": manifest.manifest_id,
        "scheduled": 1,
        "completed": 0,
        "successful": 0,
        "success_rate": 0.0,
        "errors": 0,
        "timeouts": 0,
        "status_counts": {"success": 0},
        "per_task": {},
        **change,
    }
    (root / "summary.json").write_text(json.dumps(payload))

    with pytest.raises((ValueError, ValidationError), match=match):
        EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)


def test_summary_task_mapping_key_cannot_be_overridden(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "manifest_id": manifest.manifest_id,
                "scheduled": 1,
                "completed": 0,
                "successful": 0,
                "success_rate": 0.0,
                "errors": 0,
                "timeouts": 0,
                "status_counts": {"success": 0},
                "per_task": {
                    "pick mug": {
                        "task_name": "spoofed",
                        "scheduled": 1,
                        "completed": 0,
                        "successful": 0,
                        "success_rate": 0.0,
                    }
                },
            }
        )
    )

    summary = (
        EvaluationRepository((root.parent,))
        .get_evaluation(manifest.evaluation_id)
        .summary
    )

    assert summary.per_task[0].task_name == "pick mug"


@pytest.mark.parametrize(
    "variant", ["scheduled", "task_names", "task_scheduled", "empty_tasks"]
)
def test_summary_must_match_manifest_schedule(tmp_path: Path, variant: str) -> None:
    root = tmp_path / "evaluations/run"
    manifest, _ = _write_evaluation(root)
    payload = {
        "manifest_id": manifest.manifest_id,
        "scheduled": 1,
        "completed": 0,
        "successful": 0,
        "success_rate": 0.0,
        "errors": 0,
        "timeouts": 0,
        "status_counts": {"success": 0},
        "per_task": {
            "pick mug": {
                "scheduled": 1,
                "completed": 0,
                "successful": 0,
                "success_rate": 0.0,
            }
        },
    }
    if variant == "scheduled":
        payload["scheduled"] = 2
        payload["per_task"]["pick mug"]["scheduled"] = 2
    elif variant == "task_names":
        payload["per_task"] = {"other": payload["per_task"]["pick mug"]}
    elif variant == "task_scheduled":
        payload["per_task"]["pick mug"]["scheduled"] = 2
        payload["scheduled"] = 2
    else:
        payload["per_task"] = {}
    (root / "summary.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="manifest"):
        EvaluationRepository((root.parent,)).get_evaluation(manifest.evaluation_id)


def test_open_artifact_rejects_file_replaced_by_symlink(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    media = attempt / "video.mp4"
    media.write_bytes(b"safe")
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {"artifact_id": "video", "kind": "video", "relative_path": "video.mp4"}
        ],
    )
    repository = EvaluationRepository((root.parent,))
    repository.resolve_artifact(manifest.evaluation_id, case.case_id, "video")
    media.unlink()
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")
    media.symlink_to(outside)

    with pytest.raises(OSError):
        repository.open_artifact(manifest.evaluation_id, case.case_id, "video")


def test_secure_open_closes_final_fd_when_fdopen_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"data")
    captured: list[int] = []

    def fail_fdopen(descriptor: int, _mode: str):
        captured.append(descriptor)
        raise RuntimeError("fdopen failed")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    with pytest.raises(RuntimeError, match="fdopen failed"):
        _open_beneath(tmp_path, path)

    assert len(captured) == 1
    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_jsonl_limits_and_event_page_bound_work(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, result.attempt_id, 1, 1),
        _event(manifest, case, result.attempt_id, 2, 2),
    )
    repository = EvaluationRepository((root.parent,), max_event_page_size=1)

    page = repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    assert [event["sequence"] for event in page.events] == [1]
    assert page.has_more is True

    bounded = EvaluationRepository((root.parent,), max_jsonl_bytes=10)
    with pytest.raises(ValueError, match="byte limit"):
        bounded.get_case(manifest.evaluation_id, case.case_id)


def test_large_event_journal_keeps_list_visible_and_pages_incrementally(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 31)
        ),
    )
    repository = EvaluationRepository(
        (root.parent,), max_jsonl_records=5, max_event_page_size=3
    )

    assert repository.list_evaluations()[0].evaluation_id == manifest.evaluation_id
    assert repository.get_case(manifest.evaluation_id, case.case_id).current_step == 30
    page = repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    assert [event["sequence"] for event in page.events] == [1, 2, 3]
    assert page.has_more is True


def test_timeline_window_maps_transition_step_to_authoritative_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    rows = []
    for sequence, event_type, step in (
        (1, "planning_note", None),
        (2, "transition_recorded", 3),
        (3, "planning_note", None),
        (4, "harness_verification", None),
        (5, "transition_recorded", 9),
    ):
        row = _event(manifest, case, result.attempt_id, sequence, step or 0)
        row["event_type"] = event_type
        row["step_id"] = step
        rows.append(row)
    _append_jsonl(attempt / "events.jsonl", *rows)

    repository = EvaluationRepository((root.parent,))
    early = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=3,
    )
    latest = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )

    assert early.selected_transition is not None
    assert (early.selected_transition.step_id, early.selected_transition.sequence) == (
        3,
        2,
    )
    assert early.event_window is not None
    assert early.event_window.from_sequence <= 2 <= early.event_window.to_sequence
    assert early.first_transition.step_id == 3
    assert early.latest_transition.step_id == 9
    assert latest.selected_transition == latest.latest_transition
    assert latest.transition_count == 2


def test_timeline_window_preserves_missing_step_and_empty_timeline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        _event(manifest, case, result.attempt_id, 1, 3),
        _event(manifest, case, result.attempt_id, 2, 9),
    )
    repository = EvaluationRepository((root.parent,))

    missing = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=6,
    )
    assert missing.requested_step == 6
    assert missing.selected_transition is None
    assert missing.gap.model_dump() == {"from_step": 4, "to_step": 8}

    (attempt / "events.jsonl").write_text("")
    empty = EvaluationRepository((root.parent,)).timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )
    assert empty.transition_count == 0
    assert empty.selected_transition is None
    assert empty.first_transition is None
    assert empty.latest_transition is None
    assert empty.event_window is None
    assert empty.gap is None


def test_timeline_index_continues_append_and_incomplete_tail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    path = attempt / "events.jsonl"
    first = _event(manifest, case, result.attempt_id, 1, 4)
    second = _event(manifest, case, result.attempt_id, 2, 8)
    encoded_second = json.dumps(second).encode()
    _append_jsonl(path, first, tail=encoded_second[: len(encoded_second) // 2])
    repository = EvaluationRepository((root.parent,))

    incomplete = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )
    assert incomplete.latest_transition.step_id == 4
    assert incomplete.incomplete_tail is True
    initial_scanned = repository._timeline_index.last_update_records

    with path.open("ab") as stream:
        stream.write(encoded_second[len(encoded_second) // 2 :] + b"\n")
    appended = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )
    assert appended.latest_transition.step_id == 8
    assert appended.incomplete_tail is False
    assert repository._timeline_index.last_update_records == 1
    assert repository._timeline_index.last_update_records <= initial_scanned


def test_timeline_index_rebuilds_when_partial_tail_is_truncated_then_replaced(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    path = attempt / "events.jsonl"
    first = json.dumps(_event(manifest, case, result.attempt_id, 1, 1)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first.encode() + b"{" + b"x" * 4_096)
    repository = EvaluationRepository((root.parent,))
    initial = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )
    assert initial.latest_transition.step_id == 1
    assert initial.incomplete_tail is True

    second = json.dumps(_event(manifest, case, result.attempt_id, 2, 2)) + "\n"
    path.write_text(first + second)
    assert (
        path.stat().st_size
        < repository._timeline_index._cache[
            (manifest.evaluation_id, case.case_id, result.attempt_id, path.resolve())
        ].observed_size
    )
    rebuilt = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )
    assert rebuilt.latest_transition.step_id == 2
    assert rebuilt.incomplete_tail is False


@pytest.mark.parametrize("total", [100_000, 120_000])
def test_timeline_index_is_sparse_and_seek_scan_is_bounded(
    tmp_path: Path, total: int
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence * 2)
            for sequence in range(1, total + 1)
        ),
    )
    repository = EvaluationRepository(
        (root.parent,),
        max_jsonl_records=total + 1,
        timeline_index_capacity=64,
        max_timeline_seek_records=4096,
    )

    selected = repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=123_456,
    )

    assert selected.selected_transition.sequence == 61_728
    assert repository._timeline_index.checkpoint_count <= 64
    assert repository._timeline_index.last_seek_records <= 4096
    assert repository._timeline_index.last_seek_bytes <= 8 * 1024 * 1024
    assert repository._timeline_index.cache_size == 1


@pytest.mark.parametrize("requested_step", [-1, JSON_SAFE_INTEGER_MAX + 1, True])
def test_timeline_window_rejects_unsafe_step(
    tmp_path: Path, requested_step: object
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    with pytest.raises(ValueError, match="requested_step"):
        EvaluationRepository((root.parent,)).timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=requested_step,
        )


def test_timeline_window_rejects_foreign_malformed_deep_and_oversized_records(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    path = attempt / "events.jsonl"

    foreign = _event(manifest, case, result.attempt_id, 1, 1)
    foreign["case_id"] = "foreign"
    _append_jsonl(path, foreign)
    with pytest.raises(ValueError, match="identity mismatch"):
        EvaluationRepository((root.parent,)).timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=None,
        )

    path.write_bytes(b"{malformed}\n")
    with pytest.raises(ValueError, match="invalid event JSON"):
        EvaluationRepository((root.parent,)).timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=None,
        )

    path.write_bytes(b'{"x":' * 1_100 + b"null" + b"}" * 1_100 + b"\n")
    with pytest.raises(ValueError, match="JSON nesting limit|invalid event JSON"):
        EvaluationRepository((root.parent,)).timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=None,
        )

    path.write_bytes(b"{" + b"x" * 256 + b"}\n")
    with pytest.raises(ValueError, match="record byte limit"):
        EvaluationRepository(
            (root.parent,), max_event_record_bytes=128
        ).timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=None,
        )


def test_timeline_index_rejects_changed_append_seam(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    path = attempt / "events.jsonl"
    first = json.dumps(_event(manifest, case, result.attempt_id, 1, 1))
    _append_jsonl(path, json.loads(first))
    repository = EvaluationRepository((root.parent,))
    repository.timeline_window(
        manifest.evaluation_id,
        case.case_id,
        attempt_id=result.attempt_id,
        requested_step=None,
    )

    tampered = first.replace('"step_id": 1', '"step_id": 2')
    assert len(tampered) == len(first)
    path.write_text(
        tampered
        + "\n"
        + json.dumps(_event(manifest, case, result.attempt_id, 2, 3))
        + "\n"
    )

    with pytest.raises(ValueError, match="append seam changed"):
        repository.timeline_window(
            manifest.evaluation_id,
            case.case_id,
            attempt_id=result.attempt_id,
            requested_step=None,
        )


def test_event_pages_reuse_offsets_across_pages_and_append(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events_path = attempt / "events.jsonl"
    _append_jsonl(
        events_path,
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 31)
        ),
    )
    repository = EvaluationRepository(
        (root.parent,),
        max_jsonl_records=5,
        max_event_page_size=3,
        event_checkpoint_capacity=2,
    )

    first = repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    second = repository.events_since(
        manifest.evaluation_id, case.case_id, since=first.last_sequence
    )
    third = repository.events_since(
        manifest.evaluation_id, case.case_id, since=second.last_sequence
    )
    assert [
        [event["sequence"] for event in page.events] for page in (first, second, third)
    ] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert len(repository._event_offsets) <= 2

    cursor = third.last_sequence
    final = third
    while final.has_more:
        final = repository.events_since(
            manifest.evaluation_id, case.case_id, since=cursor
        )
        cursor = final.last_sequence
    assert [event["sequence"] for event in final.events] == [28, 29, 30]
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(_event(manifest, case, result.attempt_id, 31, 31)) + "\n"
        )
    appended = repository.events_since(manifest.evaluation_id, case.case_id, since=30)
    assert [event["sequence"] for event in appended.events] == [31]


def test_event_offset_checkpoint_invalidates_after_file_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events_path = attempt / "events.jsonl"
    events = tuple(
        _event(manifest, case, result.attempt_id, sequence, sequence)
        for sequence in range(1, 11)
    )
    _append_jsonl(events_path, *events)
    repository = EvaluationRepository(
        (root.parent,), max_jsonl_records=5, max_event_page_size=3
    )
    repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    replacement = attempt / "replacement.jsonl"
    _append_jsonl(replacement, *events)
    os.replace(replacement, events_path)

    rebuilt = repository.events_since(manifest.evaluation_id, case.case_id, since=3)
    assert [event["sequence"] for event in rebuilt.events] == [4, 5, 6]


@pytest.mark.parametrize(
    ("tamper", "error"),
    (("identity", "identity mismatch"), ("sequence", "sequence mismatch")),
)
def test_event_offset_checkpoint_invalidates_after_in_place_prefix_tamper(
    tmp_path: Path, tamper: str, error: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    events_path = attempt / "events.jsonl"
    _append_jsonl(
        events_path,
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 11)
        ),
    )
    repository = EvaluationRepository(
        (root.parent,), max_jsonl_records=5, max_event_page_size=3
    )
    repository.events_since(manifest.evaluation_id, case.case_id, since=0)
    before = events_path.stat()

    if tamper == "identity":
        tampered = events_path.read_bytes().replace(
            manifest.evaluation_id.encode(), b"evaluation-rouue", 1
        )
    else:
        tampered = events_path.read_bytes().replace(
            b'"sequence": 1', b'"sequence": 9', 1
        )
    with events_path.open("r+b") as stream:
        stream.write(tampered)
    os.utime(
        events_path,
        ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
    )
    after = events_path.stat()
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)

    with pytest.raises(ValueError, match=f"event journal {error}"):
        repository.events_since(manifest.evaluation_id, case.case_id, since=3)


def test_cold_event_cursor_rebuilds_prefix_outside_page_budget(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 31)
        ),
    )

    for repository in (
        EvaluationRepository(
            (root.parent,), max_jsonl_records=5, max_event_page_size=3
        ),
        EvaluationRepository(
            (root.parent,), max_jsonl_records=5, max_event_page_size=3
        ),
    ):
        page = repository.events_since(manifest.evaluation_id, case.case_id, since=6)
        assert [event["sequence"] for event in page.events] == [7, 8, 9]

    missing = EvaluationRepository((root.parent,)).events_since(
        manifest.evaluation_id, case.case_id, since=99
    )
    assert missing.events == ()
    assert missing.last_sequence == 30
    assert missing.has_more is False


def test_open_live_frame_uses_current_attempt_and_native_camera_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    attempt = _attempt_root(root, case.case_id, "attempt-live")
    _write_case_projection(attempt, manifest, case)
    _write_recording_discovery(attempt, manifest, case, camera_id="agent/view")
    component = recording_media.camera_storage_component("agent/view")
    legacy_component = f"camera-{'agent/view'.encode().hex()}"
    live = attempt / "live" / f"{legacy_component}.png"
    live.parent.mkdir()
    live.write_bytes(b"live-png")

    with EvaluationRepository((root.parent,)).open_live_frame(
        manifest.evaluation_id, case.case_id, component
    ) as stream:
        assert stream.read() == b"live-png"
    with pytest.raises(KeyError):
        EvaluationRepository((root.parent,)).open_live_frame(
            manifest.evaluation_id,
            case.case_id,
            recording_media.camera_storage_component("other/case"),
        )


def test_read_tool_image_uses_the_exact_tool_result_overlay_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    overlay = attempt / "segments/segment_overlay_03_00.png"
    overlay.parent.mkdir()
    overlay_bytes = b"\x89PNG\r\n\x1a\nretained-overlay"
    overlay.write_bytes(overlay_bytes)
    _append_jsonl(
        attempt / "events.jsonl",
        _tool_result_event(
            manifest,
            case,
            result.attempt_id,
            1,
            "images/image_03.png",
            "segments/segment_overlay_03_00.png",
        ),
    )

    selected = EvaluationRepository((root.parent,)).read_tool_image(
        manifest.evaluation_id,
        case.case_id,
        1,
        attempt_id=result.attempt_id,
    )

    assert selected.data == overlay_bytes
    assert selected.media_type == "image/png"


@pytest.mark.parametrize(
    ("event_type", "reference", "contents", "error"),
    [
        ("tool_result", "../outside.png", b"\x89PNG\r\n\x1a\n", ValueError),
        ("tool_result", "segments/missing.png", None, FileNotFoundError),
        ("transition_recorded", "segments/image.png", b"\x89PNG\r\n\x1a\n", KeyError),
        ("tool_result", "segments/result.json", b"{}", ValueError),
    ],
)
def test_read_tool_image_rejects_untrusted_or_non_image_references(
    tmp_path: Path,
    event_type: str,
    reference: str,
    contents: bytes | None,
    error: type[Exception],
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    if contents is not None and ".." not in Path(reference).parts:
        path = attempt / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    event = _tool_result_event(manifest, case, result.attempt_id, 1, reference)
    event["event_type"] = event_type
    _append_jsonl(attempt / "events.jsonl", event)

    with pytest.raises(error):
        EvaluationRepository((root.parent,)).read_tool_image(
            manifest.evaluation_id,
            case.case_id,
            1,
            attempt_id=result.attempt_id,
        )


def test_event_checkpoint_uses_current_attempt_complete_journal_tail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    _append_jsonl(
        attempt / "events.jsonl",
        *(
            _event(manifest, case, result.attempt_id, sequence, sequence)
            for sequence in range(1, 4)
        ),
        tail=json.dumps(_event(manifest, case, result.attempt_id, 4, 4)).encode(),
    )

    checkpoint = EvaluationRepository((root.parent,)).event_checkpoint(
        manifest.evaluation_id
    )

    assert checkpoint == {
        case.case_id: {"attempt_id": result.attempt_id, "sequence": 3}
    }


def test_open_indexed_frame_accepts_only_strict_index_secondary_declaration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    component = recording_media.camera_storage_component("agent/view")
    legacy_component = f"camera-{'agent/view'.encode().hex()}"
    camera_root = attempt / "trajectory/cameras" / legacy_component
    segment = camera_root / "segments/00000003.png"
    segment.parent.mkdir(parents=True)
    segment.write_bytes(b"retained-png")
    index = camera_root / "index.json"
    index.write_text(
        json.dumps(
            {
                "camera": {
                    "camera_id": "agent/view",
                    "label": "Agent",
                    "role": "primary",
                    "width": 8,
                    "height": 8,
                    "channels": 3,
                    "fps": 20.0,
                    "media_type": "rgb",
                },
                "frames": [
                    {
                        "camera_id": "agent/view",
                        "frame_index": 3,
                        "step_id": 4,
                        "timestamp_s": 0.2,
                        "relative_path": segment.relative_to(attempt).as_posix(),
                    }
                ],
            }
        )
    )
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[
            {
                "kind": "camera_agent/view_index",
                "relative_path": index.relative_to(attempt).as_posix(),
            }
        ],
        identity=_camera_discovery_payload("agent/view"),
    )

    with EvaluationRepository((root.parent,)).open_indexed_frame(
        manifest.evaluation_id, case.case_id, component, 3
    ) as stream:
        assert stream.read() == b"retained-png"


def test_old_attempt_retained_resources_remain_attempt_scoped(tmp_path: Path) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    old_result = _result(manifest, case, "attempt-old")
    current_result = _result(manifest, case, "attempt-current")
    _append_jsonl(
        root / "results.jsonl",
        old_result.to_payload(),
        current_result.to_payload(),
    )
    old = _attempt_root(root, case.case_id, old_result.attempt_id)
    current = _attempt_root(root, case.case_id, current_result.attempt_id)
    _append_jsonl(
        old / "events.jsonl",
        _event(manifest, case, old_result.attempt_id, 1, 4),
    )
    (current / "events.jsonl").write_text("")
    artifact = old / "old.png"
    artifact.write_bytes(b"old-artifact")
    frame = old / "frame.png"
    frame.write_bytes(b"old-frame")
    index = old / "index.json"
    index.write_text(
        json.dumps(
            {
                "camera": {
                    "camera_id": "main",
                    "label": "Main",
                    "role": "primary",
                    "width": 8,
                    "height": 8,
                    "channels": 3,
                    "fps": 20.0,
                    "media_type": "rgb",
                },
                "frames": [
                    {
                        "camera_id": "main",
                        "frame_index": 3,
                        "step_id": 4,
                        "timestamp_s": 0.2,
                        "relative_path": "frame.png",
                    }
                ],
            }
        )
    )
    _write_artifacts(
        old,
        manifest,
        case,
        artifacts=[
            {"artifact_id": "old", "kind": "preview", "relative_path": "old.png"},
            {"kind": "camera_main_index", "relative_path": "index.json"},
        ],
        identity=_camera_discovery_payload("main"),
    )
    _write_artifacts(current, manifest, case)
    repository = EvaluationRepository((root.parent,))

    assert (
        repository.events_since(
            manifest.evaluation_id,
            case.case_id,
            since=0,
            attempt_id=old_result.attempt_id,
        ).attempt_id
        == old_result.attempt_id
    )
    assert (
        repository.events_since(
            manifest.evaluation_id, case.case_id, since=0
        ).attempt_id
        == current_result.attempt_id
    )
    with repository.open_artifact(
        manifest.evaluation_id,
        case.case_id,
        "old",
        attempt_id=old_result.attempt_id,
    ) as stream:
        assert stream.read() == b"old-artifact"
    with repository.open_indexed_frame(
        manifest.evaluation_id,
        case.case_id,
        recording_media.camera_storage_component("main"),
        3,
        attempt_id=old_result.attempt_id,
    ) as stream:
        assert stream.read() == b"old-frame"
    with pytest.raises(ValueError, match="JSON safe integer"):
        repository.events_since(
            manifest.evaluation_id,
            case.case_id,
            since=JSON_SAFE_INTEGER_MAX + 1,
            attempt_id=old_result.attempt_id,
        )
    with pytest.raises(ValueError, match="JSON safe integer"):
        repository.open_indexed_frame(
            manifest.evaluation_id,
            case.case_id,
            recording_media.camera_storage_component("main"),
            JSON_SAFE_INTEGER_MAX + 1,
            attempt_id=old_result.attempt_id,
        )


@pytest.mark.parametrize("attempt_id", ["../escape", "..", "missing-attempt"])
def test_attempt_scoped_resources_reject_unsafe_or_missing_attempt(
    tmp_path: Path, attempt_id: str
) -> None:
    root = tmp_path / "evaluations/run"
    manifest, case = _write_evaluation(root)
    result = _result(manifest, case, "attempt-current")
    _write_result(root, result)
    _attempt_root(root, case.case_id, result.attempt_id)
    repository = EvaluationRepository((root.parent,))

    with pytest.raises((KeyError, ValueError)):
        repository.events_since(
            manifest.evaluation_id,
            case.case_id,
            since=0,
            attempt_id=attempt_id,
        )
    with pytest.raises((KeyError, ValueError)):
        repository.open_artifact(
            manifest.evaluation_id,
            case.case_id,
            "artifact",
            attempt_id=attempt_id,
        )
    with pytest.raises((KeyError, ValueError)):
        repository.open_indexed_frame(
            manifest.evaluation_id,
            case.case_id,
            recording_media.camera_storage_component("main"),
            0,
            attempt_id=attempt_id,
        )


@pytest.mark.parametrize("mutation", ["foreign_camera", "duplicate", "escape"])
def test_open_indexed_frame_rejects_corrupt_secondary_declarations(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / f"evaluations/{mutation}"
    manifest, case = _write_evaluation(root, evaluation_id=f"evaluation-{mutation}")
    result = _result(manifest, case, "attempt-1")
    _write_result(root, result)
    attempt = _attempt_root(root, case.case_id, result.attempt_id)
    index = attempt / "index.json"
    frame = {
        "camera_id": "main" if mutation != "foreign_camera" else "other",
        "frame_index": 0,
        "step_id": 1,
        "timestamp_s": 0.1,
        "relative_path": "../secret" if mutation == "escape" else "frame.png",
    }
    index.write_text(
        json.dumps(
            {
                "camera": {
                    "camera_id": "main",
                    "label": "Main",
                    "role": "primary",
                    "width": 8,
                    "height": 8,
                    "channels": 3,
                    "fps": 20.0,
                    "media_type": "rgb",
                },
                "frames": [frame, frame] if mutation == "duplicate" else [frame],
            }
        )
    )
    _write_artifacts(
        attempt,
        manifest,
        case,
        artifacts=[{"kind": "camera_main_index", "relative_path": "index.json"}],
        identity=_camera_discovery_payload("main"),
    )

    with pytest.raises((KeyError, OSError, ValueError)):
        EvaluationRepository((root.parent,)).open_indexed_frame(
            manifest.evaluation_id,
            case.case_id,
            recording_media.camera_storage_component("main"),
            0,
        )
