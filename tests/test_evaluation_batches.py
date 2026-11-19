from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from robots.robocasa.tasks import RoboCasaTask
from rpent.evaluation.batches import (
    TaskBatchAttempt,
    TaskBatchEvidence,
    TaskBatchProcessOutcome,
    append_task_batch_attempt,
    append_task_batch_process_outcome,
    load_task_batch_attempts,
    load_task_batch_process_outcomes,
    select_complete_task_batch,
    validate_case_attempt_id,
    validate_task_batch_attempt_id,
)
from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.results import EpisodeResult
from rpent.evaluation.worker import HarnessRunReport, run_episode_case


def _manifest() -> EvaluationManifest:
    task = RoboCasaTask(
        name="CloseFridge",
        registry_id="CloseFridge_D0",
        instruction="close the fridge",
        family="atomic",
        category="appliance",
        split="pretrain",
        upstream_horizon=500,
        registry_repository="https://example.invalid/robocasa",
        registry_commit="b" * 40,
        registry_digest="sha256:" + "1" * 64,
    )
    protocol = EpisodeProtocol(
        environment="robocasa",
        preset_name="robocasa365-pandaomron-smolvla-v1",
        horizon_ceiling=None,
        checkpoint_repo_id="lerobot/smolvla_base",
        checkpoint_revision="c" * 40,
        binding_name="robocasa-smolvla-v1",
    )
    cases = expand_episode_cases(
        (task,),
        protocol,
        episodes=3,
        base_seed=7,
        seed_policy="shared_environment_sequence",
    )
    return EvaluationManifest(
        rpent_commit="a" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "robocasa"},
        simulator={"name": "robocasa365"},
        policy={"checkpoint": "fixed"},
        task_selector=task.name,
        cases=cases,
        schedule={"worker_grouping": "task", "base_seed": 7},
        contracts={"action": {"shape": [12]}},
        runtime={"evaluation_root": "/tmp/evaluation"},
        created_at="2026-08-09T00:00:00+00:00",
    )


def _attempt(manifest: EvaluationManifest, status: str, completed=()):
    expected = tuple(case.case_id for case in manifest.cases)
    return TaskBatchAttempt(
        manifest_id=manifest.manifest_id,
        task_name="CloseFridge",
        batch_attempt_id="batch-001",
        status=status,
        expected_case_ids=expected,
        completed_case_ids=tuple(completed),
        started_at="2026-08-09T00:00:00+00:00",
        finished_at=(None if status == "running" else "2026-08-09T01:00:00+00:00"),
    )


def test_batch_ledger_accepts_one_running_to_complete_transition(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "batch.jsonl"
    running = _attempt(manifest, "running")
    complete = _attempt(
        manifest,
        "complete",
        (case.case_id for case in manifest.cases),
    )

    append_task_batch_attempt(path, manifest, running)
    append_task_batch_attempt(path, manifest, complete)

    assert load_task_batch_attempts(path, manifest) == (running, complete)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"manifest_id": "f" * 64}, "foreign manifest"),
        ({"task_name": "OpenDrawer"}, "task"),
        ({"expected_case_ids": ("unknown",)}, "exact ordered"),
        ({"completed_case_ids": ("duplicate", "duplicate")}, "duplicate"),
    ],
)
def test_batch_ledger_rejects_foreign_or_drifting_identity(
    tmp_path, mutation, message
) -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match=message):
        record = replace(_attempt(manifest, "running"), **mutation)
        append_task_batch_attempt(tmp_path / "batch.jsonl", manifest, record)


def test_batch_ledger_requires_exact_prefix_and_complete_membership(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "batch.jsonl"
    expected = tuple(case.case_id for case in manifest.cases)
    append_task_batch_attempt(path, manifest, _attempt(manifest, "running"))

    with pytest.raises(ValueError, match="ordered prefix"):
        append_task_batch_attempt(
            path,
            manifest,
            _attempt(manifest, "failed", (expected[1],)),
        )
    with pytest.raises(ValueError, match="complete.*every expected"):
        append_task_batch_attempt(
            path,
            manifest,
            _attempt(manifest, "complete", expected[:2]),
        )


def test_batch_ledger_rejects_illegal_or_duplicate_transitions(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "batch.jsonl"
    running = _attempt(manifest, "running")

    with pytest.raises(ValueError, match="start with running"):
        append_task_batch_attempt(path, manifest, _attempt(manifest, "failed"))
    append_task_batch_attempt(path, manifest, running)
    with pytest.raises(ValueError, match="duplicate running"):
        append_task_batch_attempt(path, manifest, running)
    append_task_batch_attempt(path, manifest, _attempt(manifest, "failed"))
    with pytest.raises(ValueError, match="terminal"):
        append_task_batch_attempt(path, manifest, _attempt(manifest, "cancelled"))


def test_batch_terminal_record_cannot_change_started_identity(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "batch.jsonl"
    append_task_batch_attempt(path, manifest, _attempt(manifest, "running"))
    terminal = replace(
        _attempt(manifest, "failed"),
        started_at="2026-08-09T00:00:01+00:00",
    )

    with pytest.raises(ValueError, match="started_at"):
        append_task_batch_attempt(path, manifest, terminal)


def test_batch_terminal_finished_at_cannot_precede_started_at() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="finished_at.*started_at"):
        replace(
            _attempt(manifest, "failed"),
            finished_at="2026-08-08T23:59:59+00:00",
        )


@pytest.mark.parametrize(
    "tail",
    [
        b'{"valid":"json"}',
        b'{"partial":',
    ],
)
def test_batch_ledger_recovers_any_final_line_without_newline(tmp_path, tail) -> None:
    manifest = _manifest()
    path = tmp_path / "batch.jsonl"
    running = _attempt(manifest, "running")
    append_task_batch_attempt(path, manifest, running)
    with path.open("ab") as stream:
        stream.write(tail)

    assert load_task_batch_attempts(path, manifest) == (running,)
    append_task_batch_attempt(path, manifest, _attempt(manifest, "failed"))
    assert load_task_batch_attempts(path, manifest)[-1].status == "failed"


def test_episode_result_batch_identity_is_optional_and_validated() -> None:
    manifest = _manifest()
    case = manifest.cases[0]
    values = {
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "attempt_id": "case-attempt",
        "status": "terminal_failure",
        "environment": case.environment,
        "preset_name": case.preset_name,
        "task_name": case.task_name,
        "task_index": case.task_index,
        "seed": case.seed,
        "episode_index": case.episode_index,
        "worker_pid": 1,
        "worker_exit_code": 0,
        "started_at": "2026-08-09T00:00:00+00:00",
        "finished_at": "2026-08-09T00:01:00+00:00",
        "artifact_dir": "cases/attempt",
        "simulator_success": False,
        "contract_success": False,
        "terminated": True,
        "truncated": False,
        "step_count": 1,
        "action_count": 1,
        "checkpoint_repo_id": case.checkpoint_repo_id,
        "checkpoint_revision": case.checkpoint_revision,
        "binding_name": case.binding_name,
    }

    legacy = EpisodeResult.from_payload(values)
    batched = EpisodeResult.from_payload(
        {**values, "task_batch_attempt_id": "batch-001"}
    )

    assert legacy.task_batch_attempt_id is None
    assert batched.task_batch_attempt_id == "batch-001"
    with pytest.raises(ValueError, match="task_batch_attempt_id"):
        EpisodeResult.from_payload({**values, "task_batch_attempt_id": ""})


def test_batch_attempt_id_cannot_escape_its_artifact_directory(tmp_path) -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="batch_attempt_id"):
        record = replace(
            _attempt(manifest, "running"),
            batch_attempt_id="..",
        )
        append_task_batch_attempt(tmp_path / "batch.jsonl", manifest, record)


def test_run_episode_case_stamps_batch_identity_on_result_and_artifacts(
    tmp_path,
) -> None:
    manifest = replace(_manifest(), runtime={"evaluation_root": str(tmp_path)})
    case = manifest.cases[0]
    result = run_episode_case(
        case,
        manifest,
        attempt_id="batch-001-000-case",
        task_batch_attempt_id="batch-001",
        harness_runner=lambda *_args: HarnessRunReport(
            exit_code=0,
            terminal_classification="FAILED",
            harness_state="terminal_failure",
            simulator_success=False,
            contract_success=False,
            terminated=True,
            truncated=False,
            step_count=1,
            action_count=1,
            error_type=None,
            error_message=None,
        ),
    )

    assert result.task_batch_attempt_id == "batch-001"
    assert Path(result.artifact_dir).name == "batch-001-000-case"


@pytest.mark.parametrize("batch_id", [123, "..", "batch/escape"])
def test_run_episode_case_rejects_invalid_batch_identity_before_harness(
    batch_id, tmp_path
) -> None:
    manifest = replace(_manifest(), runtime={"evaluation_root": str(tmp_path)})
    calls = []

    with pytest.raises(ValueError, match="task_batch_attempt_id"):
        run_episode_case(
            manifest.cases[0],
            manifest,
            attempt_id="case-attempt",
            task_batch_attempt_id=batch_id,
            harness_runner=lambda *_args: calls.append(True),
        )

    assert calls == []
    assert not (tmp_path / "cases").exists()


def test_complete_selection_requires_coordinator_observed_zero_exit() -> None:
    missing = TaskBatchEvidence("task", "missing", "complete", (), None)
    rejected = TaskBatchEvidence("task", "rejected", "complete", (), 9)
    accepted = TaskBatchEvidence("task", "accepted", "complete", (), 0)

    assert select_complete_task_batch((missing, rejected)) is None
    assert select_complete_task_batch((missing, rejected, accepted)) == accepted


def test_coordinator_process_outcome_ledger_is_unique_and_recovers_tail(
    tmp_path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "coordinator_batch_outcomes.jsonl"
    outcome = TaskBatchProcessOutcome(
        manifest.manifest_id, manifest.cases[0].task_name, "batch-001", 0
    )
    append_task_batch_process_outcome(path, manifest, outcome)
    with path.open("ab") as stream:
        stream.write(b'{"partial":')

    assert load_task_batch_process_outcomes(path, manifest) == (outcome,)
    with pytest.raises(ValueError, match="duplicate"):
        append_task_batch_process_outcome(path, manifest, outcome)


def test_coordinator_process_outcome_ledger_rejects_internal_corruption(
    tmp_path,
) -> None:
    manifest = _manifest()
    path = tmp_path / "coordinator_batch_outcomes.jsonl"
    path.write_text('{"broken":}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        load_task_batch_process_outcomes(path, manifest)


def test_batch_and_case_attempt_ids_have_independent_length_budgets() -> None:
    batch_id = "b" * 128

    assert validate_task_batch_attempt_id(batch_id) == batch_id
    with pytest.raises(ValueError, match="task_batch_attempt_id"):
        validate_task_batch_attempt_id("b" * 129)
    derived = f"{batch_id}-049-abcdefghijkl"
    assert validate_case_attempt_id(derived) == derived
