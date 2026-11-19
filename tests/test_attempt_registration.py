from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpent.evaluation.attempt_registration import (
    MAX_CURRENT_ATTEMPT_BYTES,
    load_attempt_registration,
    register_attempt,
)
from rpent.ui.repository import CurrentAttemptUnavailable, EvaluationRepository
from tests.test_episode_worker import _manifest
from tests.test_ui_repository import _write_evaluation

REGISTERED_AT = "2026-08-13T01:02:03+00:00"


def test_registration_atomically_commits_case_before_current_locator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    writes: list[Path] = []

    from rpent.recording import media

    original = media.atomic_write_json

    def record(path: Path, payload: object) -> None:
        writes.append(path)
        original(path, payload)

    monkeypatch.setattr(media, "atomic_write_json", record)

    registration = register_attempt(manifest, case, "attempt-1", REGISTERED_AT)

    attempt_root = tmp_path / "cases" / case.case_id / "attempts" / "attempt-1"
    locator = tmp_path / "cases" / case.case_id / "current_attempt.json"
    assert writes == [attempt_root / "case.json", locator]
    assert registration.registered_at == REGISTERED_AT
    case_payload = json.loads((attempt_root / "case.json").read_text())
    locator_payload = json.loads(locator.read_text())
    assert case_payload["case"] == case.to_payload()
    assert locator_payload == {
        "schema_version": 1,
        "evaluation_id": manifest.evaluation_id,
        "manifest_id": manifest.manifest_id,
        "case_id": case.case_id,
        "attempt_id": "attempt-1",
        "registered_at": REGISTERED_AT,
    }


def test_existing_matching_case_registration_is_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    first = register_attempt(manifest, case, "attempt-1", REGISTERED_AT)
    case_path = (
        tmp_path / "cases" / case.case_id / "attempts" / "attempt-1" / "case.json"
    )
    before = case_path.read_bytes()

    second = register_attempt(manifest, case, "attempt-1", REGISTERED_AT)

    assert second == first
    assert case_path.read_bytes() == before


def test_existing_mismatched_case_registration_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    register_attempt(manifest, case, "attempt-1", REGISTERED_AT)

    with pytest.raises(ValueError, match="registration mismatch"):
        register_attempt(
            manifest,
            case,
            "attempt-1",
            "2026-08-13T01:02:04+00:00",
        )


def test_orphan_before_locator_commit_is_not_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    from rpent.recording import media

    original = media.atomic_write_json

    def fail_locator(path: Path, payload: object) -> None:
        if path.name == "current_attempt.json":
            raise OSError("locator commit failed")
        original(path, payload)

    monkeypatch.setattr(media, "atomic_write_json", fail_locator)
    with pytest.raises(OSError, match="locator commit failed"):
        register_attempt(manifest, case, "orphan", REGISTERED_AT)

    repository = EvaluationRepository((root.parent,))
    identity = repository.current_attempt_identity(
        manifest.evaluation_id, case.case_id
    )
    assert identity.attempt_id is None


def test_explicit_attempt_requires_ledger_locator_or_startup_legacy_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    register_attempt(manifest, case, "current", REGISTERED_AT)
    unregistered = root / "cases" / case.case_id / "attempts" / "directory-only"
    unregistered.mkdir(parents=True)
    repository = EvaluationRepository((root.parent,))

    assert repository.authorize_attempt(
        manifest.evaluation_id, case.case_id, "current"
    ).attempt_id == "current"
    with pytest.raises(KeyError):
        repository.authorize_attempt(
            manifest.evaluation_id, case.case_id, "directory-only"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"attempt_id": "foreign"}, "attempt_id"),
        ({"extra": True}, "fields"),
    ],
)
def test_locator_is_strict_bounded_versioned_and_identity_checked(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.cases[0]
    registration = register_attempt(manifest, case, "attempt-1", REGISTERED_AT)
    path = tmp_path / "cases" / case.case_id / "current_attempt.json"
    payload = registration.to_payload() | mutation
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_attempt_registration(
            path,
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            case_id=case.case_id,
            attempt_id="attempt-1",
        )

    path.write_bytes(b"{" + b" " * MAX_CURRENT_ATTEMPT_BYTES + b"}")
    with pytest.raises(ValueError, match="byte limit"):
        load_attempt_registration(
            path,
            evaluation_id=manifest.evaluation_id,
            manifest_id=manifest.manifest_id,
            case_id=case.case_id,
            attempt_id="attempt-1",
        )


def test_current_identity_uses_locator_without_attempt_directory_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    register_attempt(manifest, case, "attempt-current", REGISTERED_AT)
    repository = EvaluationRepository((root.parent,))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal identity resolution must not enumerate Attempts")

    monkeypatch.setattr("rpent.ui.repository._list_directory_beneath", forbidden)

    identity = repository.current_attempt_identity(
        manifest.evaluation_id, case.case_id
    )

    assert identity.attempt_id == "attempt-current"


def test_present_malformed_locator_fails_closed_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    register_attempt(manifest, case, "attempt-current", REGISTERED_AT)
    locator = root / "cases" / case.case_id / "current_attempt.json"
    locator.write_text('{"schema_version":2}', encoding="utf-8")
    repository = EvaluationRepository((root.parent,))

    with pytest.raises(CurrentAttemptUnavailable, match="current Attempt"):
        repository.current_attempt_identity(manifest.evaluation_id, case.case_id)


def test_startup_legacy_scan_validates_4096_entries_and_fails_on_4097(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluations" / "run"
    manifest, case = _write_evaluation(root)
    repository = EvaluationRepository((root.parent,))
    opened = 0

    monkeypatch.setattr(
        "rpent.ui.repository._iter_directory_beneath",
        lambda *_args: (f"attempt-{index}" for index in range(4_097)),
    )

    def projection(path: Path, _boundary: Path):
        nonlocal opened
        opened += 1
        return {
            "schema_version": 1,
            "evaluation_id": manifest.evaluation_id,
            "manifest_id": manifest.manifest_id,
            "case_id": case.case_id,
            "attempt_id": path.parent.name,
            "started_at": "2026-08-13T00:00:00+00:00",
            "case": case.to_payload(),
        }

    monkeypatch.setattr(repository, "_optional_json", projection)

    with pytest.raises(CurrentAttemptUnavailable, match="scan limit"):
        repository._legacy_active_attempt_scan(root, manifest, case, set(), None)

    assert opened == 4_096
