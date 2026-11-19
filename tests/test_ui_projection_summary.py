from __future__ import annotations

import pytest
from pydantic import ValidationError

from rpent.ui.projection_summary import build_attempt_summary


def test_attempt_summary_projects_optional_runtime_metrics() -> None:
    summary = build_attempt_summary(
        {"status": "success"},
        artifact_health=None,
        usage=None,
        planner_elapsed_seconds=12.5,
        vla_chunks=7,
        retained_frame_count=91,
    )

    assert summary is not None
    assert summary.planner_elapsed_seconds == 12.5
    assert summary.vla_chunks == 7
    assert summary.retained_frame_count == 91


def test_attempt_summary_keeps_missing_runtime_metrics_optional() -> None:
    summary = build_attempt_summary(
        {"status": "success"}, artifact_health=None, usage=None
    )

    assert summary is not None
    assert summary.planner_elapsed_seconds is None
    assert summary.vla_chunks is None
    assert summary.retained_frame_count is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("planner_elapsed_seconds", -0.1),
        ("planner_elapsed_seconds", "12.5"),
        ("vla_chunks", -1),
        ("vla_chunks", "7"),
        ("retained_frame_count", -1),
        ("retained_frame_count", "91"),
    ],
)
def test_attempt_summary_rejects_invalid_runtime_metrics(
    name: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        build_attempt_summary(
            {"status": "error"},
            artifact_health=None,
            usage=None,
            **{name: value},
        )
