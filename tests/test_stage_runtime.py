"""Tests for atomic remaining-stage replacement."""

from dataclasses import dataclass

import pytest

from rpent.harness.stages import StageRuntime


@dataclass(frozen=True)
class Stage:
    id: str
    intent: str
    outcome_ids: tuple[str, ...]


def _runtime():
    return StageRuntime(
        (
            Stage("stage-1", "done", ("outcome-1",)),
            Stage("stage-2", "old", ("outcome-2", "outcome-3")),
            Stage("stage-3", "old", ("outcome-4",)),
        ),
        active_index=1,
    )


def test_exposes_runtime_position_without_transferring_ownership():
    runtime = _runtime()

    assert runtime.active_index == 1
    assert runtime.completed == (runtime.stages[0],)
    assert runtime.has_next is True


def test_advance_requires_caller_confirmed_stage_completion():
    runtime = _runtime()
    before = (runtime.active_index, runtime.active, runtime.stages)

    with pytest.raises(ValueError, match="complete"):
        runtime.advance(stage_complete=False)

    assert (runtime.active_index, runtime.active, runtime.stages) == before


def test_advance_moves_atomically_to_the_next_stage():
    runtime = _runtime()

    active = runtime.advance(stage_complete=True)

    assert active == runtime.stages[2]
    assert runtime.active_index == 2
    assert runtime.completed == runtime.stages[:2]
    assert runtime.has_next is False


def test_advance_rejects_moving_beyond_the_final_stage_without_mutation():
    only_stage = Stage("only", "finish", ("outcome-1",))
    runtime = StageRuntime((only_stage,))
    before = (runtime.active_index, runtime.active, runtime.stages)

    with pytest.raises(ValueError, match="final"):
        runtime.advance(stage_complete=True)

    assert (runtime.active_index, runtime.active, runtime.stages) == before


def test_replaces_only_remaining_stages_atomically():
    runtime = _runtime()
    replacement = (
        Stage("replacement-1", "recover", ("outcome-2",)),
        Stage("replacement-2", "finish", ("outcome-3", "outcome-4")),
    )

    stages = runtime.replace_remaining(
        replacement, ("outcome-2", "outcome-3", "outcome-4")
    )

    assert stages[0].id == "stage-1"
    assert stages[1:] == replacement
    assert runtime.active.id == "replacement-1"


def test_replacement_starts_at_current_stage_when_nothing_is_completed():
    runtime = StageRuntime(
        (
            Stage("stage-1", "old", ("outcome-1",)),
            Stage("stage-2", "old", ("outcome-2",)),
        )
    )
    replacement = (Stage("replacement", "new", ("outcome-1", "outcome-2")),)

    stages = runtime.replace_remaining(replacement, ("outcome-1", "outcome-2"))

    assert stages == replacement
    assert runtime.active == replacement[0]
    assert runtime.active_index == 0
    assert runtime.completed == ()


@pytest.mark.parametrize(
    "replacement",
    (
        (Stage("bad", "missing", ("outcome-2", "outcome-3")),),
        (
            Stage("a", "duplicate", ("outcome-2", "outcome-3")),
            Stage("b", "duplicate", ("outcome-3", "outcome-4")),
        ),
        (Stage("bad", "unknown", ("outcome-2", "outcome-X", "outcome-4")),),
        (Stage("bad", "reordered", ("outcome-3", "outcome-2", "outcome-4")),),
    ),
    ids=("missing", "duplicate", "unknown", "order-mismatch"),
)
def test_invalid_partition_is_rejected_without_mutation(replacement):
    runtime = _runtime()
    before = runtime.stages

    with pytest.raises(ValueError, match="partition"):
        runtime.replace_remaining(replacement, ("outcome-2", "outcome-3", "outcome-4"))

    assert runtime.stages == before


def test_replacement_stage_ids_are_unique_across_completed_and_new_stages():
    runtime = _runtime()
    before = (runtime.active_index, runtime.active, runtime.stages)

    with pytest.raises(ValueError, match="unique"):
        runtime.replace_remaining(
            (Stage("stage-1", "collision", ("outcome-2", "outcome-3", "outcome-4")),),
            ("outcome-2", "outcome-3", "outcome-4"),
        )

    assert (runtime.active_index, runtime.active, runtime.stages) == before
