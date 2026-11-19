from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from rpent.evaluation.cases import (
    EpisodeCase,
    EpisodeProtocol,
    expand_episode_cases,
)


def _task(index: int, *, horizon: int = 500):
    return SimpleNamespace(
        index=index,
        name=f"task_{index}",
        language=f"instruction {index}",
        upstream_horizon=horizon,
        display_label=f"Task {index}",
        output_path=f"ignored/{index}",
    )


def _protocol(**overrides) -> EpisodeProtocol:
    values = {
        "environment": "synthetic",
        "preset_name": "robocerebra-libero10-smolvla-v1",
        "horizon_ceiling": 480,
        "checkpoint_repo_id": "lerobot/smolvla_robocerebra",
        "checkpoint_revision": "7" * 40,
        "binding_name": "robocerebra-smolvla-v1",
    }
    values.update(overrides)
    return EpisodeProtocol(**values)


def test_one_task_and_default_episode_count_produce_one_case() -> None:
    cases = expand_episode_cases([_task(3)], _protocol(), base_seed=11)

    assert len(cases) == 1
    assert cases[0].ordinal == 0
    assert cases[0].task_index == 3
    assert cases[0].seed == 11
    assert cases[0].episode_index == 0
    assert cases[0].horizon == 480


def test_missing_horizon_ceiling_preserves_upstream_horizon() -> None:
    cases = expand_episode_cases(
        [_task(3, horizon=7200)],
        _protocol(horizon_ceiling=None),
        base_seed=11,
    )

    assert cases[0].horizon == 7200


def test_explicit_horizon_ceiling_clips_upstream_horizon() -> None:
    cases = expand_episode_cases(
        [_task(3, horizon=7200)],
        _protocol(horizon_ceiling=1000),
        base_seed=11,
    )

    assert cases[0].horizon == 1000


def test_ten_tasks_by_ten_episodes_have_stable_registry_then_episode_order() -> None:
    cases = expand_episode_cases(
        [_task(index) for index in range(10)],
        _protocol(),
        episodes=10,
        base_seed=100,
    )

    assert len(cases) == 100
    assert [(case.task_index, case.episode_index) for case in cases[:12]] == [
        *((0, episode) for episode in range(10)),
        (1, 0),
        (1, 1),
    ]
    for task_index in range(10):
        task_cases = [case for case in cases if case.task_index == task_index]
        assert [case.seed for case in task_cases] == list(range(100, 110))
    assert len({case.case_id for case in cases}) == 100


def test_shared_environment_sequence_keeps_seed_and_unique_case_identity() -> None:
    cases = expand_episode_cases(
        [_task(3)],
        _protocol(),
        episodes=50,
        base_seed=7,
        seed_policy="shared_environment_sequence",
    )

    assert [case.episode_index for case in cases] == list(range(50))
    assert {case.seed for case in cases} == {7}
    assert len({case.case_id for case in cases}) == 50


def test_case_identity_is_canonical_and_execution_sensitive() -> None:
    original = expand_episode_cases([_task(3)], _protocol(), base_seed=11)[0]
    repeated = expand_episode_cases([_task(3)], _protocol(), base_seed=11)[0]

    assert original == repeated
    assert len(original.case_id) == 64
    for field, value in (
        ("environment", "other"),
        ("preset_name", "other-preset"),
        ("task_name", "other-task"),
        ("task_index", 4),
        ("instruction", "other instruction"),
        ("seed", 12),
        ("episode_index", 1),
        ("horizon", 479),
        ("checkpoint_repo_id", "other/checkpoint"),
        ("checkpoint_revision", "8" * 40),
        ("binding_name", "other-binding"),
    ):
        changed = replace(original, **{field: value})
        assert changed.case_id != original.case_id


def test_non_execution_task_labels_do_not_change_identity() -> None:
    first = _task(3)
    second = _task(3)
    second.display_label = "Renamed display only"
    second.output_path = "different/output"

    first_case = expand_episode_cases([first], _protocol(), base_seed=0)[0]
    second_case = expand_episode_cases([second], _protocol(), base_seed=0)[0]

    assert first_case.case_id == second_case.case_id


def test_case_and_expansion_validation_fail_closed() -> None:
    case = expand_episode_cases([_task(0)], _protocol(), base_seed=0)[0]
    with pytest.raises(FrozenInstanceError):
        case.seed = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="episodes"):
        expand_episode_cases([_task(0)], _protocol(), episodes=0, base_seed=0)
    with pytest.raises(ValueError, match="horizon"):
        expand_episode_cases(
            [_task(0, horizon=0)],
            _protocol(),
            base_seed=0,
        )
    with pytest.raises(ValueError, match="duplicate"):
        expand_episode_cases(
            [_task(0), _task(0)],
            _protocol(),
            base_seed=0,
        )
    with pytest.raises(ValueError, match="seed_policy"):
        expand_episode_cases(
            [_task(0)],
            _protocol(),
            base_seed=0,
            seed_policy="implicit_magic",  # type: ignore[arg-type]
        )


def test_episode_case_recomputes_digest_instead_of_accepting_spoofed_id() -> None:
    values = {
        "case_id": "spoofed",
        "ordinal": 0,
        "environment": "synthetic",
        "preset_name": "preset",
        "task_name": "task",
        "task_index": 0,
        "instruction": "instruction",
        "seed": 0,
        "episode_index": 0,
        "horizon": 10,
        "checkpoint_repo_id": "owner/model",
        "checkpoint_revision": "7" * 40,
        "binding_name": "binding",
    }

    with pytest.raises(TypeError, match="case_id"):
        EpisodeCase(**values)
