from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from robots.robocasa import get_env_spec
from robots.robocasa.competition import LEADERBOARD_PROFILE_ID
from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    RoboCasaDependencyReport,
)
from robots.robocasa.submission import (
    LEADERBOARD_SUBMISSION_SCHEMA,
    SubmissionMetadata,
    build_leaderboard_submission,
    validate_leaderboard_submission,
    write_submission_bundle,
)
from rpent.envs.launch import LaunchValidationError
from rpent.evaluation.front_controller import _worker_base_argv
from rpent.evaluation.official_metrics import OfficialMetrics
from rpent.policies.smolvla_catalog import SMOLVLA_CHECKPOINTS
from tests.test_robocasa_competition_worker import _manifest


def _metadata() -> SubmissionMetadata:
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
    )


def _metrics(manifest, *, complete=True) -> OfficialMetrics:
    return OfficialMetrics(
        manifest.manifest_id,
        {"task": 0.5},
        {
            "atomic_seen": 0.5,
            "composite_seen": 0.25,
            "composite_unseen": 0.75,
        },
        0.5,
        2_500 if complete else 2_499,
        2_500,
        50 if complete else 49,
        50,
        complete,
    )


def _dependencies() -> RoboCasaDependencyReport:
    return RoboCasaDependencyReport(
        interpreter="/opt/robocasa/bin/python",
        robocasa_version="1.0.1",
        robocasa_commit=ROBOCASA_COMMIT,
        robosuite_version="1.5.1",
        robosuite_commit=ROBOSUITE_COMMIT,
        mujoco_version="3.2.0",
        mujoco_gl="egl",
        default_assets_ready=True,
        available_registries=frozenset({"lightwheel"}),
    )


def _submission_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    return replace(
        manifest,
        qualification={
            **manifest.qualification,
            "submission_metadata": _metadata().to_payload(),
        },
        policy={"checkpoint": SMOLVLA_CHECKPOINTS["robocasa"].to_payload()},
    )


def test_submission_matches_strict_official_schema_without_internal_fields(tmp_path):
    manifest = replace(
        _manifest(tmp_path),
        policy={"checkpoint": SMOLVLA_CHECKPOINTS["robocasa"].to_payload()},
    )

    payload = build_leaderboard_submission(
        manifest, _metrics(manifest), _metadata(), _dependencies()
    )

    assert set(payload) == {
        "model_name",
        "policy_family",
        "date",
        "submission_source",
        "robocasa_version",
        "atomic_seen_success",
        "composite_seen_success",
        "composite_unseen_success",
        "code_url",
        "commit_hash",
        "checkpoint_url",
        "paper_link",
        "open_source",
        "training_config",
        "notes",
    }
    assert payload["atomic_seen_success"] == 50.0
    assert payload["composite_seen_success"] == 25.0
    assert payload["composite_unseen_success"] == 75.0
    assert payload["submission_source"] == "external"
    assert payload["open_source"] == "yes"
    assert "overall" not in payload
    assert "rpent_commit" not in payload
    assert "vla_checkpoint" not in payload


@pytest.mark.parametrize(
    "mutation",
    [
        {"unknown": "field"},
        {"date": "09/08/2026"},
        {"date": "20260809"},
        {"submission_source": "simulation"},
        {"open_source": True},
        {"training_config": {"batch_size": True, "num_training_steps": 1}},
    ],
)
def test_submission_metadata_rejects_unknown_or_schema_invalid_values(mutation):
    payload = {**_metadata().to_payload(), **mutation}

    with pytest.raises(ValueError):
        SubmissionMetadata.from_payload(payload)


def test_submission_metadata_file_is_bounded_and_canonical(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_metadata().to_payload()), encoding="utf-8")

    loaded = SubmissionMetadata.from_json_path(path)

    assert loaded == _metadata()
    path.write_bytes(b"{" + b" " * (64 * 1024) + b"}")
    with pytest.raises(ValueError, match="64 KiB"):
        SubmissionMetadata.from_json_path(path)


def test_submission_metadata_accepts_official_na_commit_hash():
    payload = {**_metadata().to_payload(), "commit_hash": "N/A"}

    assert SubmissionMetadata.from_payload(payload).commit_hash == "N/A"


@pytest.mark.parametrize(
    "arguments",
    (
        ("--submission-metadata-json", "/tmp/deleted.json", "--planner", "api"),
        ("--submission-metadata-json=/tmp/deleted.json", "--planner", "api"),
    ),
)
def test_worker_argv_never_reopens_submission_metadata_source(arguments):
    assert _worker_base_argv(arguments) == ["--planner", "api"]


@pytest.mark.parametrize(
    "arguments",
    (
        ("--planning-profile", "libero-hooked-gated-v3", "--planner", "api"),
        ("--planning-profile=episode-hooked-gated-v1", "--planner", "api"),
    ),
)
def test_worker_base_argv_strips_submitted_planning_profile(arguments):
    assert _worker_base_argv(arguments) == ["--planner", "api"]


def test_local_validator_locks_official_schema_contract():
    payload = {
        **_metadata().to_payload(),
        "robocasa_version": "1.0.1",
        "atomic_seen_success": 50.0,
        "composite_seen_success": 25.0,
        "composite_unseen_success": 75.0,
        "checkpoint_url": "https://huggingface.co/org/model/tree/revision",
    }

    assert LEADERBOARD_SUBMISSION_SCHEMA["additionalProperties"] is False
    assert LEADERBOARD_SUBMISSION_SCHEMA["properties"]["submission_source"]["enum"] == (
        "external",
        "internal",
    )
    assert LEADERBOARD_SUBMISSION_SCHEMA["properties"]["open_source"]["enum"] == (
        "yes",
        "no",
    )
    validate_leaderboard_submission(payload)
    for mutation in (
        {**payload, "unknown": True},
        {key: value for key, value in payload.items() if key != "model_name"},
        {**payload, "submission_source": "simulation"},
        {**payload, "open_source": True},
        {**payload, "date": "20260809"},
        {**payload, "atomic_seen_success": "50"},
    ):
        with pytest.raises(ValueError, match="official leaderboard schema"):
            validate_leaderboard_submission(mutation)


def test_launch_preview_validates_submission_metadata_with_cli_equivalent_schema(
    tmp_path,
):
    path = tmp_path / "metadata.json"
    path.write_text('{"unknown": true}', encoding="utf-8")
    preset = get_env_spec().launch_preset()

    with pytest.raises(LaunchValidationError) as error:
        preset.normalize(
            "hybrid",
            {
                "evaluation_profile": LEADERBOARD_PROFILE_ID,
                "submission_metadata_json": str(path),
            },
        )

    assert set(error.value.errors) == {"submission_metadata_json"}


def test_unqualified_or_partial_run_writes_no_submission_bundle(tmp_path):
    manifest = _submission_manifest(tmp_path)

    with pytest.raises(ValueError, match="qualified terminal"):
        write_submission_bundle(
            tmp_path,
            manifest,
            _metrics(manifest, complete=False),
            SimpleNamespace(accepted=False),
            _dependencies(),
        )

    assert not (tmp_path / "leaderboard_submission.json").exists()
    assert not (tmp_path / "reproduction.md").exists()


def test_qualified_bundle_writes_submission_and_reproduction_atomically(tmp_path):
    manifest = _submission_manifest(tmp_path)

    write_submission_bundle(
        tmp_path,
        manifest,
        _metrics(manifest),
        SimpleNamespace(accepted=True),
        _dependencies(),
    )

    payload = json.loads((tmp_path / "leaderboard_submission.json").read_text())
    reproduction = (tmp_path / "reproduction.md").read_text()
    assert payload["commit_hash"] == _metadata().commit_hash
    assert manifest.rpent_commit in reproduction
    assert SMOLVLA_CHECKPOINTS["robocasa"].revision in reproduction
    assert not list(tmp_path.glob("*.tmp"))


def test_bundle_staging_failure_publishes_neither_file(monkeypatch, tmp_path):
    manifest = _submission_manifest(tmp_path)
    monkeypatch.setattr(
        "robots.robocasa.submission._stage_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_submission_bundle(
            tmp_path,
            manifest,
            _metrics(manifest),
            SimpleNamespace(accepted=True),
            _dependencies(),
        )

    assert not (tmp_path / "leaderboard_submission.json").exists()
    assert not (tmp_path / "reproduction.md").exists()


def test_real_staging_failure_cleans_temporary_files(monkeypatch, tmp_path):
    from robots.robocasa.submission import _stage_bytes

    monkeypatch.setattr(
        "robots.robocasa.submission.os.fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _stage_bytes(tmp_path, {"reproduction.md": b"content"})

    assert list(tmp_path.iterdir()) == []


def test_second_publish_failure_rolls_back_first_file(monkeypatch, tmp_path):
    manifest = _submission_manifest(tmp_path)
    real_replace = __import__("os").replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish failed")
        real_replace(source, target)

    monkeypatch.setattr("robots.robocasa.submission.os.replace", fail_second)

    with pytest.raises(OSError, match="publish failed"):
        write_submission_bundle(
            tmp_path,
            manifest,
            _metrics(manifest),
            SimpleNamespace(accepted=True),
            _dependencies(),
        )

    assert not (tmp_path / "leaderboard_submission.json").exists()
    assert not (tmp_path / "reproduction.md").exists()
