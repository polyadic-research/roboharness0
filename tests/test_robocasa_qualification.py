from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    RoboCasaDependencyReport,
)
from robots.robocasa.qualification import (
    QualificationCheck,
    RoboCasaQualification,
    _canonical_worker_planning_profile,
    _validate_required_projections,
    qualify_robocasa_run,
    write_qualification_bundle,
)
from robots.robocasa.submission import SubmissionMetadata
from rpent.evaluation.batches import TaskBatchEvidence
from rpent.evaluation.coordinator import EvaluationCoordinator
from rpent.evaluation.diagnostics import aggregate_rpent_diagnostics
from rpent.evaluation.front_controller import (
    _worker_base_argv,
    build_evaluation_manifest,
)
from rpent.evaluation.results import EpisodeResult, aggregate_results
from rpent.planning.manifest import canonical_json
from rpent.recording.contracts import ArtifactHealth, QualificationOutcome
from rpent.recording.media import atomic_write_json
from tests.test_robocasa_competition import _official_build_inputs


def test_xiaomi_qualification_requires_runtime_health_identity():
    from robots.robocasa.qualification import _policy_binding_identity_matches
    from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
    from rpent.policies.lerobot_smolvla_artifacts import binding_identity
    from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
    from rpent.policies.xiaomi_mibot_server import SERVER_VERSION

    policy = {
        "policy_family": "xiaomi-mibot", "protocol_version": 1,
        "checkpoint": XIAOMI_MIBOT_CHECKPOINT.to_payload(),
        "binding_digest": binding_identity(make_xiaomi_policy_binding()),
        "server_version": SERVER_VERSION, "artifact_identity": "a" * 64,
        "remote_code_revision": XIAOMI_MIBOT_CHECKPOINT.revision,
    }
    assert not _policy_binding_identity_matches(policy)
    policy["health"] = dict(policy)
    policy["health"]["binding_identity"] = policy["binding_digest"]
    assert _policy_binding_identity_matches(policy)
    for field in ("protocol_version", "checkpoint", "binding_digest", "server_version",
                  "artifact_identity", "remote_code_revision"):
        assert not _policy_binding_identity_matches({**policy, field: "wrong"})
    assert not _policy_binding_identity_matches({**policy, "health": {**policy["health"], "artifact_identity": "b" * 64}})


def _metadata() -> dict:
    return SubmissionMetadata(
        model_name="RPent SmolVLA",
        policy_family="SmolVLA",
        date="2026-08-09",
        submission_source="external",
        code_url="https://example.com/code",
        commit_hash="abcdef1234567890",
        paper_link="N/A",
        open_source="yes",
        training_config={"batch_size": 64, "num_training_steps": 100_000},
        notes="N/A",
    ).to_payload()


def _manifest(tmp_path: Path):
    _profile, _selector, args, env_spec, launch = _official_build_inputs(tmp_path)
    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa"],
        git_identity=lambda: ("a" * 40, False),
        launch=launch,
    )
    return replace(
        manifest,
        qualification={
            **manifest.qualification,
            "submission_metadata": _metadata(),
        },
    )


def _with_planning_profile(manifest, planning_profile: str):
    return replace(
        manifest,
        cli_arguments={
            **manifest.cli_arguments,
            "argv": [
                *_worker_base_argv(manifest.cli_arguments["argv"]),
                "--planning-profile",
                planning_profile,
            ],
        },
        policy={**manifest.policy, "planning_profile": planning_profile},
        schedule={
            **manifest.schedule,
            "effective_inputs": {
                **manifest.schedule["effective_inputs"],
                "planning_profile": planning_profile,
            },
        },
        launch={
            **manifest.launch,
            "submitted": {
                **manifest.launch["submitted"],
                "planning_profile": planning_profile,
            },
            "effective": {
                **manifest.launch["effective"],
                "planning_profile": planning_profile,
            },
        },
    )


def _dependencies(*, registries=("lightwheel",)):
    return RoboCasaDependencyReport(
        interpreter="/opt/robocasa/bin/python",
        robocasa_version="1.0.1",
        robocasa_commit=ROBOCASA_COMMIT,
        robosuite_version="1.5.1",
        robosuite_commit=ROBOSUITE_COMMIT,
        mujoco_version="3.2.0",
        mujoco_gl="egl",
        default_assets_ready=True,
        available_registries=frozenset(registries),
    )


def _result(manifest, case, batch_id):
    attempt_id = f"{batch_id}-{case.episode_index:03d}-{case.case_id[:12]}"
    return EpisodeResult(
        evaluation_id=manifest.evaluation_id,
        manifest_id=manifest.manifest_id,
        case_id=case.case_id,
        attempt_id=attempt_id,
        task_batch_attempt_id=batch_id,
        status="success",
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
        simulator_success=True,
        contract_success=True,
        terminated=True,
        truncated=False,
        step_count=1,
        action_count=1,
        checkpoint_repo_id=case.checkpoint_repo_id,
        checkpoint_revision=case.checkpoint_revision,
        binding_name=case.binding_name,
    )


def _complete_evidence(manifest):
    evidences = []
    for task_index, task_name in enumerate(
        dict.fromkeys(case.task_name for case in manifest.cases)
    ):
        batch_id = f"batch-{task_index:02d}"
        results = tuple(
            _result(manifest, case, batch_id)
            for case in manifest.cases
            if case.task_name == task_name
        )
        for result in results:
            artifact = Path(result.artifact_dir)
            artifact.mkdir(parents=True)
            outcome = QualificationOutcome.evaluate(
                result, ArtifactHealth("complete"), True
            )
            (artifact / "qualification.json").write_text(
                json.dumps(outcome.to_payload()), encoding="utf-8"
            )
        evidences.append(TaskBatchEvidence(task_name, batch_id, "complete", results, 0))
    return tuple(evidences)


def _write_authority_projections(manifest, results, evidences):
    root = Path(manifest.runtime["evaluation_root"])
    manifest.write_new(root / "evaluation_manifest.json")
    (root / "results.jsonl").write_text(
        "".join(canonical_json(result.to_payload()) + "\n" for result in results),
        encoding="utf-8",
    )
    atomic_write_json(
        root / "summary.json", aggregate_results(manifest, results).to_payload()
    )
    atomic_write_json(
        root / "official_metrics.json",
        qualify_robocasa_run(
            manifest, results, evidences, dependencies=_dependencies()
        )[1].to_payload(),
    )
    atomic_write_json(
        root / "rpent_diagnostics.json",
        aggregate_rpent_diagnostics(manifest, results, evidences).to_payload(),
    )


def _check(report, check_id):
    return next(check for check in report.checks if check.id == check_id)


@pytest.mark.parametrize(
    "planning_profile",
    [
        "robocasa-hooked-gated-v1",
        "robocasa-authoritative-gated-v1",
        "libero-hooked-gated-v3",
    ],
)
def test_qualification_accepts_registered_consistent_planning_identity(
    tmp_path: Path,
    planning_profile: str,
) -> None:
    manifest = _with_planning_profile(_manifest(tmp_path), planning_profile)

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is True
    assert _check(qualification, "official_launch_runtime").accepted is True


def test_qualification_rejects_consistent_unknown_planning_identity(
    tmp_path: Path,
) -> None:
    manifest = _with_planning_profile(_manifest(tmp_path), "unknown-planning-profile")

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is False


@pytest.mark.parametrize("section", ["submitted", "effective"])
def test_qualification_rejects_launch_planning_identity_drift(
    tmp_path: Path,
    section: str,
) -> None:
    manifest = _manifest(tmp_path)
    drifted = replace(
        manifest,
        launch={
            **manifest.launch,
            section: {
                **manifest.launch[section],
                "planning_profile": "libero-hooked-gated-v3",
            },
        },
    )

    qualification, _ = qualify_robocasa_run(
        drifted, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is False


def test_qualification_accepts_canonicalized_legacy_submitted_planning_identity(
    tmp_path: Path,
) -> None:
    base = _with_planning_profile(_manifest(tmp_path), "authoritative-gated-v1")
    manifest = replace(
        base,
        launch={
            **base.launch,
            "submitted": {
                **base.launch["submitted"],
                "planning_profile": "libero-authoritative-gated-v1",
            },
        },
    )

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is True


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], False),
        (["--planning-profile", "robocasa-authoritative-gated-v1"], True),
        (["--planning-profile=robocasa-authoritative-gated-v1"], True),
    ],
)
def test_qualification_accepts_worker_planning_profile_cli_semantics(
    tmp_path: Path,
    argv: list[str],
    expected: bool,
) -> None:
    base = _manifest(tmp_path)
    manifest = replace(
        base,
        cli_arguments={**base.cli_arguments, "argv": argv},
    )

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert (
        _check(qualification, "policy_binding_planner_identity").accepted is expected
    )


def test_qualification_rejects_worker_argv_only_planning_identity_drift(
    tmp_path: Path,
) -> None:
    base = _with_planning_profile(_manifest(tmp_path), "libero-hooked-gated-v3")
    manifest = replace(
        base,
        cli_arguments={**base.cli_arguments, "argv": []},
    )

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--planning-profile", "unknown-planning-profile"],
        ["--planning-profile"],
        ["--planning-profile="],
        [
            "--planning-profile",
            "authoritative-gated-v1",
            "--planning-profile=libero-hooked-gated-v3",
        ],
    ],
)
def test_qualification_rejects_invalid_worker_planning_identity(
    tmp_path: Path,
    argv: list[str],
) -> None:
    base = _manifest(tmp_path)
    manifest = replace(
        base,
        cli_arguments={**base.cli_arguments, "argv": argv},
    )

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "policy_binding_planner_identity").accepted is False


@pytest.mark.parametrize(
    ("planning_profile", "expected"),
    [
        ("robocasa-horizon-gated-v1", True),
        ("robocasa-hooked-gated-v1", False),
        ("robocasa-authoritative-gated-v1", False),
        ("libero-hooked-gated-v3", False),
    ],
)
def test_qualification_uses_default_for_omitted_submitted_planning_identity(
    tmp_path: Path,
    planning_profile: str,
    expected: bool,
) -> None:
    base = _with_planning_profile(_manifest(tmp_path), planning_profile)
    manifest = replace(
        base,
        launch={
            **base.launch,
            "submitted": {
                key: value
                for key, value in base.launch["submitted"].items()
                if key != "planning_profile"
            },
        },
    )

    qualification, _ = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert (
        _check(qualification, "policy_binding_planner_identity").accepted is expected
    )


def test_qualification_worker_profile_fallback_is_robocasa_horizon_gated() -> None:
    assert _canonical_worker_planning_profile([]) == "robocasa-horizon-gated-v1"


@pytest.mark.parametrize(
    ("check_id", "mutate", "dependencies"),
    [
        (
            "profile_identity",
            lambda m: replace(
                m,
                qualification={**m.qualification, "official_profile": "drift"},
            ),
            _dependencies(),
        ),
        (
            "rpent_identity",
            lambda m: replace(m, rpent_dirty=True),
            _dependencies(),
        ),
        (
            "official_launch_runtime",
            lambda m: replace(m, launch={**m.launch, "runner": "drift"}),
            _dependencies(),
        ),
        (
            "official_launch_runtime",
            lambda m: replace(
                m,
                runtime={**m.runtime, "qualification_module": "drift.module"},
            ),
            _dependencies(),
        ),
        (
            "official_launch_runtime",
            lambda m: replace(
                m,
                qualification={
                    **m.qualification,
                    "require_workload_confirmation": False,
                },
            ),
            _dependencies(),
        ),
        (
            "official_launch_runtime",
            lambda m: replace(
                m,
                launch={
                    **m.launch,
                    "effective": {**m.launch["effective"], "seed": 8},
                },
            ),
            _dependencies(),
        ),
        (
            "task_membership_digest",
            lambda m: replace(
                m, schedule={**m.schedule, "task_registry_digest": "sha256:bad"}
            ),
            _dependencies(),
        ),
        (
            "task_membership_digest",
            lambda m: replace(
                m,
                schedule={
                    **m.schedule,
                    "task_set_membership": {
                        "atomic_seen": m.schedule["task_set_membership"]["atomic_seen"][
                            :-1
                        ],
                        "composite_seen": (
                            m.schedule["task_set_membership"]["atomic_seen"][-1],
                            *m.schedule["task_set_membership"]["composite_seen"],
                        ),
                        "composite_unseen": m.schedule["task_set_membership"][
                            "composite_unseen"
                        ],
                    },
                },
            ),
            _dependencies(),
        ),
        (
            "scene_split",
            lambda m: replace(m, simulator={**m.simulator, "scene_split": "target"}),
            _dependencies(),
        ),
        (
            "seed_reset_protocol",
            lambda m: replace(m, schedule={**m.schedule, "seed_policy": "per_case"}),
            _dependencies(),
        ),
        ("installed_dependencies", lambda m: m, None),
        (
            "installed_dependencies",
            lambda m: m,
            replace(_dependencies(), mujoco_gl="osmesa"),
        ),
        ("object_registries", lambda m: m, _dependencies(registries=())),
        (
            "upstream_horizons",
            lambda m: replace(
                m, contracts={**m.contracts, "horizon_rule": "lower_ceiling"}
            ),
            _dependencies(),
        ),
        (
            "policy_binding_planner_identity",
            lambda m: replace(m, policy={**m.policy, "planning_profile": "drift"}),
            _dependencies(),
        ),
        (
            "policy_binding_planner_identity",
            lambda m: replace(
                m,
                policy={
                    key: value
                    for key, value in m.policy.items()
                    if key != "planning_profile"
                },
            ),
            _dependencies(),
        ),
        (
            "policy_binding_planner_identity",
            lambda m: replace(
                m,
                policy={
                    **m.policy,
                    "processor_digests": {
                        **m.policy["processor_digests"],
                        "preprocessor": "sha256:drift",
                    },
                },
            ),
            _dependencies(),
        ),
        (
            "native_success_authority",
            lambda m: replace(m, schedule={**m.schedule, "success_source": "contract"}),
            _dependencies(),
        ),
        ("accepted_task_batches", lambda m: m, _dependencies()),
        (
            "selective_rerun_disabled",
            lambda m: replace(
                m, qualification={**m.qualification, "allow_selective_case_rerun": True}
            ),
            _dependencies(),
        ),
        ("complete_artifacts", lambda m: m, _dependencies()),
    ],
)
def test_qualification_rejects_each_locked_invariant(
    tmp_path, check_id, mutate, dependencies
):
    manifest = mutate(_manifest(tmp_path))

    qualification, _metrics = qualify_robocasa_run(
        manifest, (), (), dependencies=dependencies
    )

    assert _check(qualification, check_id).accepted is False
    assert qualification.accepted is False


def test_qualification_accepts_exact_2500_result_single_attempt_cohort(
    monkeypatch, tmp_path
):
    manifest = _manifest(tmp_path)
    evidences = _complete_evidence(manifest)
    results = tuple(result for evidence in evidences for result in evidence.results)

    qualification, metrics = qualify_robocasa_run(
        manifest, results, evidences, dependencies=_dependencies()
    )

    assert qualification.accepted is True
    assert qualification.status == "qualified"
    assert metrics.is_complete is True
    assert all(check.accepted for check in qualification.checks)

    monkeypatch.setattr(
        "robots.robocasa.qualification.inspect_robocasa_dependencies",
        lambda *args, **kwargs: _dependencies(),
    )
    _write_authority_projections(manifest, results, evidences)
    for name in (
        "evaluation_manifest.json",
        "results.jsonl",
        "summary.json",
        "official_metrics.json",
        "rpent_diagnostics.json",
    ):
        path = tmp_path / name
        original = path.read_bytes()
        if name == "results.jsonl":
            path.write_bytes(original.splitlines(keepends=True)[0])
        else:
            drift = json.loads(original)
            drift[
                "evaluation_id" if name == "evaluation_manifest.json" else "manifest_id"
            ] = "foreign"
            atomic_write_json(path, drift)
        with pytest.raises((OSError, ValueError), match=name):
            _validate_required_projections(
                tmp_path, manifest, results, evidences, metrics
            )
        path.write_bytes(original)
    written = write_qualification_bundle(manifest, results, evidences, tmp_path)

    assert written.accepted is True
    assert {
        "evaluation_manifest.json",
        "results.jsonl",
        "summary.json",
        "official_metrics.json",
        "rpent_diagnostics.json",
        "qualification.json",
        "leaderboard_submission.json",
        "reproduction.md",
    }.issubset(path.name for path in tmp_path.iterdir())


def test_qualification_rejects_extra_attempt_and_central_retry(tmp_path):
    manifest = _manifest(tmp_path)
    first = manifest.cases[0]
    failed = TaskBatchEvidence(first.task_name, "retry-batch", "failed", (), 1)
    retry = _result(manifest, first, "retry-batch")

    qualification, _metrics = qualify_robocasa_run(
        manifest, (retry,), (failed,), dependencies=_dependencies()
    )

    evidence = _check(qualification, "accepted_task_batches").evidence
    assert evidence["attempts"] == 1
    assert evidence["central_results"] == 1
    assert qualification.accepted is False


def test_artifact_qualification_rejects_fixed_child_symlink(tmp_path):
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    result = _result(manifest, case, "batch-00")
    artifact = Path(result.artifact_dir)
    artifact.mkdir(parents=True)
    alternate = artifact / "alternate.json"
    alternate.write_text(
        json.dumps(
            QualificationOutcome.evaluate(
                result, ArtifactHealth("complete"), True
            ).to_payload()
        ),
        encoding="utf-8",
    )
    (artifact / "qualification.json").symlink_to(alternate.name)
    evidence = TaskBatchEvidence(case.task_name, "batch-00", "failed", (result,), 1)

    qualification, _metrics = qualify_robocasa_run(
        manifest, (result,), (evidence,), dependencies=_dependencies()
    )

    assert _check(qualification, "complete_artifacts").evidence["complete"] == 0


def test_qualification_persists_reverifiable_runtime_dependency_inspection(tmp_path):
    manifest = _manifest(tmp_path)

    qualification, _metrics = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "installed_dependencies").evidence == {
        "schema_version": 1,
        "interpreter": "/opt/robocasa/bin/python",
        "robocasa_version": "1.0.1",
        "robocasa_commit": ROBOCASA_COMMIT,
        "robosuite_version": "1.5.1",
        "robosuite_commit": ROBOSUITE_COMMIT,
        "mujoco_version": "3.2.0",
        "mujoco_gl": "egl",
        "default_assets_ready": True,
        "available_registries": ["lightwheel"],
    }


def test_official_cli_launch_identity_is_equivalent_to_hybrid_launch(tmp_path):
    _profile, _selector, args, env_spec, _launch = _official_build_inputs(tmp_path)
    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        ["--env", "robocasa", "--evaluation-profile", args.evaluation_profile],
        git_identity=lambda: ("a" * 40, False),
        launch=None,
    )
    manifest = replace(
        manifest,
        qualification={
            **manifest.qualification,
            "submission_metadata": _metadata(),
        },
    )

    qualification, _metrics = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert _check(qualification, "official_launch_runtime").accepted is True
    assert _check(qualification, "policy_binding_planner_identity").accepted is True


def test_official_cli_launch_accepts_explicit_hooked_planning_identity(tmp_path):
    planning_profile = "libero-hooked-gated-v3"
    _profile, _selector, args, env_spec, _launch = _official_build_inputs(
        tmp_path, planning_profile
    )
    manifest, _ = build_evaluation_manifest(
        args,
        env_spec,
        [
            "--env",
            "robocasa",
            "--evaluation-profile",
            args.evaluation_profile,
            "--planning-profile",
            planning_profile,
        ],
        git_identity=lambda: ("a" * 40, False),
        launch=None,
    )
    manifest = replace(
        manifest,
        qualification={
            **manifest.qualification,
            "submission_metadata": _metadata(),
        },
    )

    qualification, _metrics = qualify_robocasa_run(
        manifest, (), (), dependencies=_dependencies()
    )

    assert manifest.launch["submitted"] == {}
    assert manifest.cli_arguments["argv"] == (
        "--planning-profile",
        planning_profile,
    )
    assert _check(qualification, "official_launch_runtime").accepted is True
    assert _check(qualification, "policy_binding_planner_identity").accepted is True


def test_coordinator_invokes_manifest_owned_qualification_projector(
    monkeypatch, tmp_path
):
    manifest = _manifest(tmp_path)
    calls = []
    monkeypatch.setattr(
        "robots.robocasa.qualification.write_qualification_bundle",
        lambda selected, results, batches, root: calls.append(
            (selected, results, batches, root)
        ),
    )
    coordinator = EvaluationCoordinator(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        policy_daemon=None,
    )

    coordinator._write_projections(aggregate_results(manifest, ()), ())

    assert calls == [(manifest, (), (), tmp_path)]


def test_partial_projection_does_not_claim_or_repeat_runtime_inspection(
    monkeypatch, tmp_path
):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        "robots.robocasa.qualification.inspect_robocasa_dependencies",
        lambda *args, **kwargs: pytest.fail("partial runs must not inspect runtime"),
    )

    (tmp_path / "leaderboard_submission.json").write_text("stale")
    (tmp_path / "reproduction.md").write_text("stale")

    qualification = write_qualification_bundle(manifest, (), (), tmp_path)

    assert qualification.status == "partial"
    assert (
        json.loads((tmp_path / "qualification.json").read_text())["accepted"] is False
    )
    assert not (tmp_path / "leaderboard_submission.json").exists()
    assert not (tmp_path / "reproduction.md").exists()


def test_export_failure_cannot_publish_accepted_qualification(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    old = b'{"accepted":false,"status":"partial"}\n'
    (tmp_path / "qualification.json").write_bytes(old)
    metrics = replace(
        qualify_robocasa_run(manifest, (), (), dependencies=None)[1],
        completed_rollouts=2_500,
        completed_tasks=50,
        is_complete=True,
    )
    accepted = RoboCasaQualification(
        manifest.schedule["evaluation_profile"],
        "qualified",
        True,
        (QualificationCheck("all", True, "ok", "none"),),
    )
    monkeypatch.setattr(
        "robots.robocasa.qualification.qualify_robocasa_run",
        lambda *args, **kwargs: (accepted, metrics),
    )
    monkeypatch.setattr(
        "robots.robocasa.qualification.inspect_robocasa_dependencies",
        lambda *args, **kwargs: _dependencies(),
    )
    monkeypatch.setattr(
        "robots.robocasa.qualification._validate_required_projections",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "robots.robocasa.submission.write_submission_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_qualification_bundle(manifest, (), (), tmp_path)

    assert (tmp_path / "qualification.json").read_bytes() == old


def test_terminal_rejection_revokes_prior_submission(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    metrics = replace(
        qualify_robocasa_run(manifest, (), (), dependencies=None)[1],
        completed_rollouts=2_500,
        completed_tasks=50,
        is_complete=True,
    )
    rejected = RoboCasaQualification(
        manifest.schedule["evaluation_profile"],
        "rejected",
        False,
        (QualificationCheck("failed", False, "drift", "rerun"),),
    )
    monkeypatch.setattr(
        "robots.robocasa.qualification.qualify_robocasa_run",
        lambda *args, **kwargs: (rejected, metrics),
    )
    monkeypatch.setattr(
        "robots.robocasa.qualification.inspect_robocasa_dependencies",
        lambda *args, **kwargs: _dependencies(),
    )
    (tmp_path / "leaderboard_submission.json").write_text("stale")
    (tmp_path / "reproduction.md").write_text("stale")

    result = write_qualification_bundle(manifest, (), (), tmp_path)

    assert result.status == "rejected"
    assert not (tmp_path / "leaderboard_submission.json").exists()
    assert not (tmp_path / "reproduction.md").exists()
