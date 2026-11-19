"""Tests that keep generated Harness lifecycle documents synchronized."""

from __future__ import annotations

from pathlib import Path

from rpent.harness.lifecycle import HarnessLifecycleState, LifecycleEvent
from scripts.generate_harness_docs import (
    GENERATED_DOCUMENTS,
    documents_are_current,
    render_lifecycle_markdown,
    render_mermaid,
)


def test_generated_reference_contains_every_state_and_event() -> None:
    rendered = render_lifecycle_markdown()
    for state in HarnessLifecycleState:
        assert f"`{state.value}`" in rendered
    for event in LifecycleEvent:
        assert f"`{event.value}`" in rendered


def test_generated_reference_documents_bootstrap_boundary() -> None:
    rendered = render_lifecycle_markdown()
    assert "`start_instruction` is a one-shot pre-lifecycle bootstrap" in rendered
    assert "unstarted (`None`)" in rendered
    assert "successful plan registration" in rendered
    assert "rejected after the session has started" in rendered


def test_handwritten_governance_matches_stage_boundary_operations() -> None:
    readme = Path("docs/harness/README.md").read_text(encoding="utf-8")
    assert "`action_required` or `verification_required`" in readme
    assert "At an `observation_required` stage boundary" in readme
    assert "only `submit_observation`" in readme
    assert "`request_exception` is not\navailable there" in readme


def test_generated_mermaid_contains_every_transition() -> None:
    rendered = render_mermaid()
    assert rendered.startswith(
        "stateDiagram-v2\n    [*] --> action_required: successful start_instruction\n"
    )
    for event in LifecycleEvent:
        assert f": {event.value}" in rendered


def test_document_check_detects_missing_and_stale_files(tmp_path: Path) -> None:
    expected = {Path("lifecycle.md"): "current\n"}
    assert not documents_are_current(tmp_path, expected)
    (tmp_path / "lifecycle.md").write_text("stale\n", encoding="utf-8")
    assert not documents_are_current(tmp_path, expected)
    (tmp_path / "lifecycle.md").write_text("current\n", encoding="utf-8")
    assert documents_are_current(tmp_path, expected)


def test_committed_generated_documents_are_current() -> None:
    assert documents_are_current(Path("docs/harness"), GENERATED_DOCUMENTS)
