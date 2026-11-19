"""Contract tests for prompt placeholder syntax."""

from __future__ import annotations

import pytest

from rpent.utils.templates import substitute_text


def test_strict_render_rejects_single_brace_runtime_placeholder() -> None:
    with pytest.raises(
        ValueError,
        match=r"single-brace.*task.*\{\{task\}\}",
    ):
        substitute_text("Run {task}.", {"task": "CloseFridge"}, strict=True)


def test_strict_render_preserves_unbound_literal_single_brace_example() -> None:
    text = 'Use "pick up the {object}" as an example.'

    assert substitute_text(text, {"task": "CloseFridge"}, strict=True) == text
