from __future__ import annotations

import json

import pytest

from rpent.evaluation.preparation_progress import (
    PREPARATION_PHASES,
    PreparationError,
    PreparationProgress,
    advance_preparation_from_environment,
    load_preparation,
)


def _clock(*values: str):
    return iter(values).__next__


def test_preparation_progress_persists_phase_updates(tmp_path):
    path = tmp_path / "preparation.json"
    progress = PreparationProgress.create(
        path,
        "evaluation-1",
        "manifest-1",
        clock=_clock(
            "2026-08-09T00:00:00+00:00",
            "2026-08-09T00:00:01+00:00",
            "2026-08-09T00:00:03+00:00",
        ),
    )

    progress.advance("sam3_loading", "loading")
    progress.advance("sam3_loading", "ready", detail="cold")

    loaded = load_preparation(
        path,
        evaluation_id="evaluation-1",
        manifest_id="manifest-1",
    )
    assert loaded is not None
    assert loaded.status == "preparing"
    assert loaded.current_phase == "sam3_loading"
    assert tuple(item.phase for item in loaded.phases) == PREPARATION_PHASES
    phase = next(item for item in loaded.phases if item.phase == "sam3_loading")
    assert (phase.status, phase.started_at, phase.finished_at, phase.detail) == (
        "ready",
        "2026-08-09T00:00:01+00:00",
        "2026-08-09T00:00:03+00:00",
        "cold",
    )


def test_preparation_progress_records_safe_terminal_failure(tmp_path):
    path = tmp_path / "preparation.json"
    progress = PreparationProgress.create(
        path,
        "evaluation-1",
        "manifest-1",
        clock=_clock(
            "2026-08-09T00:00:00+00:00",
            "2026-08-09T00:00:01+00:00",
            "2026-08-09T00:00:02+00:00",
        ),
    )
    progress.advance("sam3_loading", "loading")
    progress.advance(
        "sam3_loading",
        "failed",
        error=PreparationError("TimeoutError", "Pi0.5 did not become ready"),
    )

    loaded = load_preparation(
        path,
        evaluation_id="evaluation-1",
        manifest_id="manifest-1",
    )
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error == PreparationError(
        "TimeoutError", "Pi0.5 did not become ready"
    )
    assert "Traceback" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("phase", "status"),
    (("unknown", "loading"), ("vla_loading", "unknown")),
)
def test_preparation_progress_rejects_unknown_phase_or_status(
    tmp_path, phase, status
):
    progress = PreparationProgress.create(
        tmp_path / "preparation.json", "evaluation-1", "manifest-1"
    )

    with pytest.raises(ValueError, match="phase|status"):
        progress.advance(phase, status)


def test_preparation_progress_rejects_illegal_phase_transition(tmp_path):
    progress = PreparationProgress.create(
        tmp_path / "preparation.json", "evaluation-1", "manifest-1"
    )

    with pytest.raises(ValueError, match="transition"):
        progress.advance("sam3_loading", "ready")


def test_load_preparation_rejects_wrong_identity_and_nonfinite_json(tmp_path):
    path = tmp_path / "preparation.json"
    PreparationProgress.create(path, "evaluation-1", "manifest-1")

    with pytest.raises(ValueError, match="evaluation_id"):
        load_preparation(
            path,
            evaluation_id="evaluation-2",
            manifest_id="manifest-1",
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phases"][0]["detail"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON|constant"):
        load_preparation(
            path,
            evaluation_id="evaluation-1",
            manifest_id="manifest-1",
        )


def test_load_preparation_rejects_noncanonical_phase_order(tmp_path):
    path = tmp_path / "preparation.json"
    PreparationProgress.create(path, "evaluation-1", "manifest-1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phases"].reverse()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical order"):
        load_preparation(
            path,
            evaluation_id="evaluation-1",
            manifest_id="manifest-1",
        )


def test_environment_hook_updates_only_an_identity_bound_existing_journal(
    monkeypatch, tmp_path
):
    path = tmp_path / "preparation.json"
    PreparationProgress.create(path, "evaluation-1", "manifest-1")
    monkeypatch.setenv("RPENT_PREPARATION_PATH", str(path))
    monkeypatch.setenv("RPENT_EVALUATION_ID", "evaluation-1")
    monkeypatch.setenv("RPENT_MANIFEST_ID", "manifest-1")

    assert advance_preparation_from_environment("sam3_loading", "skipped") is True
    loaded = load_preparation(
        path, evaluation_id="evaluation-1", manifest_id="manifest-1"
    )
    assert loaded is not None
    assert loaded.phases[1].status == "skipped"

    monkeypatch.setenv("RPENT_EVALUATION_ID", "foreign")
    with pytest.raises(ValueError, match="evaluation_id"):
        advance_preparation_from_environment("vla_loading", "skipped")


def test_environment_hook_treats_repeated_worker_phase_updates_as_idempotent(
    monkeypatch, tmp_path
):
    path = tmp_path / "preparation.json"
    PreparationProgress.create(path, "evaluation-1", "manifest-1")
    monkeypatch.setenv("RPENT_PREPARATION_PATH", str(path))
    monkeypatch.setenv("RPENT_EVALUATION_ID", "evaluation-1")
    monkeypatch.setenv("RPENT_MANIFEST_ID", "manifest-1")

    assert advance_preparation_from_environment("sam3_loading", "skipped") is True
    assert advance_preparation_from_environment("sam3_loading", "skipped") is True
    assert advance_preparation_from_environment("vla_loading", "skipped") is True
    assert advance_preparation_from_environment("simulator_starting", "loading") is True
    assert advance_preparation_from_environment("simulator_starting", "ready") is True
    assert advance_preparation_from_environment("simulator_starting", "ready") is True

    loaded = load_preparation(
        path, evaluation_id="evaluation-1", manifest_id="manifest-1"
    )
    assert loaded is not None
    assert [phase.status for phase in loaded.phases[:4]] == [
        "ready",
        "skipped",
        "skipped",
        "ready",
    ]


def test_environment_hook_is_noop_without_private_worker_identity(monkeypatch):
    for name in (
        "RPENT_PREPARATION_PATH",
        "RPENT_EVALUATION_ID",
        "RPENT_MANIFEST_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    assert advance_preparation_from_environment("sam3_loading", "skipped") is False
