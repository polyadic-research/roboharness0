from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rpent.evaluation.batches import TaskBatchEvidence
from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.official_metrics import aggregate_official_metrics
from rpent.evaluation.results import EpisodeResult


def _manifest(tmp_path: Path) -> EvaluationManifest:
    cases = tuple(
        EpisodeCase(
            ordinal=task_index * 2 + episode_index,
            environment="robocasa",
            preset_name="preset",
            task_name=task_name,
            task_index=task_index,
            instruction="do it",
            seed=7,
            episode_index=episode_index,
            horizon=10,
            checkpoint_repo_id="repo/model",
            checkpoint_revision="a" * 40,
            binding_name="binding",
        )
        for task_index, task_name in enumerate(("task-a", "task-b", "task-c"))
        for episode_index in range(2)
    )
    return EvaluationManifest(
        rpent_commit="b" * 40,
        rpent_dirty=False,
        cli_arguments={"argv": []},
        environment={"name": "robocasa"},
        simulator={"revision": "c" * 40},
        policy={"revision": "d" * 40},
        task_selector="all",
        cases=cases,
        schedule={
            "episodes_per_task": 2,
            "base_seed": 7,
            "worker_grouping": "task",
            "task_sets": ("seen", "unseen"),
            "task_set_membership": {
                "seen": ("task-a", "task-b"),
                "unseen": ("task-c",),
            },
        },
        contracts={"action": {"shape": [12]}},
        runtime={"evaluation_root": str(tmp_path)},
        evaluation_id="evaluation",
        created_at="2026-08-09T00:00:00+00:00",
    )


def _result(manifest, case, batch_id, *, simulator, contract):
    attempt_id = f"{batch_id}-{case.episode_index:03d}-{case.case_id[:12]}"
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        task_batch_attempt_id=batch_id,
        status=("success" if simulator and contract else "error"),
        environment=case.environment,
        preset_name=case.preset_name,
        task_name=case.task_name,
        task_index=case.task_index,
        seed=case.seed,
        episode_index=case.episode_index,
        worker_pid=1,
        worker_exit_code=0,
        started_at="2026-08-09T00:00:00+00:00",
        finished_at="2026-08-09T00:00:01+00:00",
        artifact_dir=str(
            Path(manifest.runtime["evaluation_root"])
            / "cases"
            / case.case_id
            / "attempts"
            / attempt_id
        ),
        simulator_success=simulator,
        contract_success=contract,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
        error_type=None,
        error_message=None,
    )


def _evidence(manifest, task_name, signals, *, exit_code=0, status="complete"):
    cases = tuple(case for case in manifest.cases if case.task_name == task_name)
    batch_id = f"batch-{task_name}"
    results = tuple(
        _result(manifest, case, batch_id, simulator=signal[0], contract=signal[1])
        for case, signal in zip(cases, signals, strict=True)
    )
    return TaskBatchEvidence(task_name, batch_id, status, results, exit_code)


def test_official_metrics_use_native_success_and_unweighted_task_means(tmp_path):
    manifest = _manifest(tmp_path)
    evidences = (
        _evidence(manifest, "task-a", ((True, True), (True, True))),
        _evidence(manifest, "task-b", ((False, False), (False, False))),
        _evidence(manifest, "task-c", ((True, False), (False, True))),
    )
    results = tuple(result for evidence in evidences for result in evidence.results)

    metrics = aggregate_official_metrics(manifest, results, evidences)

    assert dict(metrics.task_success_rates) == {
        "task-a": 1.0,
        "task-b": 0.0,
        "task-c": 0.5,
    }
    assert dict(metrics.split_success_rates) == {"seen": 0.5, "unseen": 0.5}
    assert metrics.overall_success_rate == 0.5
    assert metrics.completed_rollouts == metrics.scheduled_rollouts == 6
    assert metrics.is_complete is True


def test_partial_and_mixed_attempts_keep_full_denominators_and_are_not_final(
    tmp_path,
):
    manifest = _manifest(tmp_path)
    accepted = _evidence(manifest, "task-a", ((True, True), (True, True)))
    rejected = _evidence(
        manifest,
        "task-b",
        ((True, True), (True, True)),
        exit_code=9,
    )
    partial = replace(
        _evidence(manifest, "task-c", ((True, True), (True, True))),
        status="failed",
        results=_evidence(manifest, "task-c", ((True, True), (True, True))).results[:1],
    )
    evidences = (accepted, rejected, partial)
    results = tuple(result for evidence in evidences for result in evidence.results)

    metrics = aggregate_official_metrics(manifest, results, evidences)

    assert metrics.scheduled_rollouts == 6
    assert metrics.completed_rollouts == 2
    assert metrics.task_success_rates["task-a"] == 1.0
    assert metrics.task_success_rates["task-b"] == 0.0
    assert metrics.task_success_rates["task-c"] == 0.0
    assert metrics.overall_success_rate == pytest.approx(1 / 3)
    assert metrics.is_complete is False
    assert "qualified" not in metrics.to_payload()


def test_multiple_accepted_complete_attempts_for_one_task_fail_closed(tmp_path):
    manifest = _manifest(tmp_path)
    first = _evidence(manifest, "task-a", ((True, True), (True, True)))
    second = replace(
        first,
        batch_attempt_id="batch-task-a-second",
        results=tuple(
            _result(
                manifest,
                case,
                "batch-task-a-second",
                simulator=True,
                contract=True,
            )
            for case in manifest.cases
            if case.task_name == "task-a"
        ),
    )

    with pytest.raises(ValueError, match="multiple complete"):
        aggregate_official_metrics(
            manifest,
            (*first.results, *second.results),
            (first, second),
        )
