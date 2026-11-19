from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.manifest import EvaluationManifest
from rpent.evaluation.preparation import PreparedEvaluation
from tests.synthetic_tasks import SyntheticTask


def _cases():
    tasks = [
        SyntheticTask(
            suite="libero_10",
            index=index,
            name=f"task_{index}",
            language=f"instruction {index}",
            bddl_identifier=f"libero_10/task_{index}.bddl",
            upstream_horizon=500,
        )
        for index in range(2)
    ]
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="robocerebra-libero10-smolvla-v1",
        horizon_ceiling=480,
        checkpoint_repo_id="lerobot/smolvla_robocerebra",
        checkpoint_revision="7" * 40,
        binding_name="robocerebra-smolvla-v1",
    )
    return expand_episode_cases(tasks, protocol, episodes=2, base_seed=10)


def _manifest(**overrides) -> EvaluationManifest:
    values = {
        "rpent_commit": "a" * 40,
        "rpent_dirty": False,
        "cli_arguments": {
            "env": "synthetic",
            "task": "all",
            "episodes": 2,
            "seed": 10,
        },
        "environment": {
            "name": "synthetic",
            "preset_name": "robocerebra-libero10-smolvla-v1",
            "protocol": "rpent-episode-v1",
        },
        "simulator": {
            "name": "libero",
            "revision": "b" * 40,
            "suite": "libero_10",
            "variant": "standard",
        },
        "policy": {
            "protocol_version": 1,
            "lerobot_revision": "c" * 40,
            "checkpoint": {
                "repo_id": "lerobot/smolvla_robocerebra",
                "revision": "7" * 40,
            },
            "processor_digests": {"config.json": "d" * 64},
            "binding_digest": "e" * 64,
        },
        "task_selector": "all",
        "cases": _cases(),
        "schedule": {
            "episodes_per_task": 2,
            "base_seed": 10,
            "seed_policy": "base_seed_plus_episode_index",
        },
        "contracts": {
            "control_hz": 20,
            "images": {
                "count": 2,
                "shape": [256, 256, 3],
                "dtype": "uint8",
            },
            "action": {"shape": [7], "dtype": "float32", "range": [-1, 1]},
            "horizon_rule": "min(upstream_horizon,safety_ceiling)",
        },
        "runtime": {
            "main_interpreter": "/opt/rpent/bin/python",
            "policy_interpreter": "/opt/smolvla/bin/python",
            "rendering_backend": "egl",
        },
        "created_at": "2026-08-05T12:00:00+10:00",
        "launch": {
            "execution_mode": "hybrid",
            "preset_version": 1,
            "runner": "robocerebra-hybrid-v1",
            "submitted": {"task": "all"},
            "effective": {"task": "all", "max_turns": 100},
        },
    }
    values.update(overrides)
    return EvaluationManifest(**values)


def test_manifest_contains_complete_provenance_and_ordered_cases() -> None:
    manifest = _manifest()
    payload = manifest.to_payload()

    assert payload["schema_version"] == 3
    assert payload["evaluation_id"] == manifest.evaluation_id
    assert len(payload["manifest_id"]) == 64
    assert payload["rpent"] == {"commit": "a" * 40, "dirty": False}
    assert payload["cli_arguments"]["episodes"] == 2
    assert payload["environment"]["preset_name"].startswith("robocerebra-")
    assert payload["simulator"]["revision"] == "b" * 40
    assert payload["policy"]["lerobot_revision"] == "c" * 40
    assert payload["task_selector"] == "all"
    assert [case["ordinal"] for case in payload["cases"]] == [0, 1, 2, 3]
    assert payload["schedule"]["seed_policy"] == ("base_seed_plus_episode_index")
    assert payload["contracts"]["action"]["shape"] == [7]
    assert payload["runtime"]["rendering_backend"] == "egl"
    assert payload["launch"]["execution_mode"] == "hybrid"


def test_schema_v3_effective_config_changes_manifest_id() -> None:
    original = _manifest()
    changed = _manifest(
        launch={
            **dict(original.launch),
            "effective": {"task": "all", "max_turns": 120},
        }
    )

    assert changed.manifest_id != original.manifest_id


def test_schema_v3_resolved_runtime_endpoints_do_not_change_identity() -> None:
    original = _manifest()
    moved = _manifest(
        runtime={
            **dict(original.runtime),
            "policy_endpoint": "http://127.0.0.1:19001",
            "sam3_endpoint": "http://127.0.0.1:19002",
        }
    )
    moved_again = _manifest(
        runtime={
            **dict(original.runtime),
            "policy_endpoint": "http://127.0.0.1:29001",
            "sam3_endpoint": "http://127.0.0.1:29002",
        }
    )

    assert moved.manifest_id == moved_again.manifest_id


def test_schema_v2_loads_with_empty_launch(tmp_path) -> None:
    path = tmp_path / "schema-v2.json"
    legacy = _manifest(schema_version=2, launch={})
    path.write_text(json.dumps(legacy.to_payload()), encoding="utf-8")

    loaded = EvaluationManifest.load(path)

    assert loaded.source_schema_version == 2
    assert dict(loaded.launch) == {}


def test_timestamp_does_not_change_manifest_identity() -> None:
    first = _manifest(created_at="2026-08-05T12:00:00+10:00")
    second = _manifest(created_at="2026-08-06T12:00:00+10:00")

    assert first.manifest_id == second.manifest_id


def test_reordered_cases_and_execution_provenance_change_identity() -> None:
    original = _manifest()
    assert (
        _manifest(cases=tuple(reversed(_cases()))).manifest_id != original.manifest_id
    )

    changes = (
        {"rpent_commit": "f" * 40},
        {"rpent_dirty": True},
        {"cli_arguments": {**dict(original.cli_arguments), "seed": 11}},
        {"environment": {**dict(original.environment), "protocol": "v2"}},
        {"simulator": {**dict(original.simulator), "revision": "f" * 40}},
        {"policy": {**dict(original.policy), "binding_digest": "f" * 64}},
        {"task_selector": "3"},
        {"schedule": {**dict(original.schedule), "base_seed": 11}},
        {"contracts": {**dict(original.contracts), "control_hz": 10}},
        {"runtime": {**dict(original.runtime), "rendering_backend": "osmesa"}},
    )
    for change in changes:
        assert _manifest(**change).manifest_id != original.manifest_id


def test_manifest_is_frozen_and_nested_inputs_are_defensively_copied() -> None:
    cli = {"env": "synthetic"}
    manifest = _manifest(cli_arguments=cli)
    cli["env"] = "mutated"

    assert manifest.cli_arguments["env"] == "synthetic"
    with pytest.raises(FrozenInstanceError):
        manifest.task_selector = "3"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.cli_arguments["env"] = "mutated"  # type: ignore[index]


def test_prepared_evaluation_defaults_and_deeply_freezes_profile_facts() -> None:
    task = SyntheticTask(
        suite="libero_10",
        index=0,
        name="task_0",
        language="instruction 0",
        bddl_identifier="libero_10/task_0.bddl",
        upstream_horizon=500,
    )
    prepared = PreparedEvaluation(
        tasks=(task,),
        protocol=EpisodeProtocol(
            environment="synthetic",
            preset_name="robocerebra-libero10-smolvla-v1",
            horizon_ceiling=480,
            checkpoint_repo_id="lerobot/smolvla_robocerebra",
            checkpoint_revision="7" * 40,
            binding_name="robocerebra-smolvla-v1",
        ),
        environment={},
        simulator={},
        policy={},
        contracts={},
        runtime={},
        schedule={"groups": [{"name": "official"}]},
        qualification={"checks": {"complete": True}},
    )

    assert (
        dict(
            PreparedEvaluation(
                tasks=(task,),
                protocol=prepared.protocol,
                environment={},
                simulator={},
                policy={},
                contracts={},
                runtime={},
            ).schedule
        )
        == {}
    )
    with pytest.raises(TypeError):
        prepared.schedule["groups"][0]["name"] = "changed"
    with pytest.raises(TypeError):
        prepared.qualification["checks"]["complete"] = False


def test_write_new_reuses_same_identity_and_refuses_different_manifest(
    tmp_path,
) -> None:
    path = tmp_path / "evaluation_manifest.json"
    first = _manifest()
    first.write_new(path)
    original = path.read_bytes()

    reused = _manifest(created_at="2026-08-06T12:00:00+10:00").write_new(path)

    assert reused.manifest_id == first.manifest_id
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="different manifest"):
        _manifest(rpent_commit="f" * 40).write_new(path)
    assert path.read_bytes() == original


def test_schema_v1_first_write_returns_normalized_disk_identity(tmp_path) -> None:
    path = tmp_path / "legacy-manifest.json"
    submitted = _manifest(schema_version=1, evaluation_id="not-persisted")

    written = submitted.write_new(path)
    reloaded = EvaluationManifest.load(path)

    assert written == reloaded
    assert written.evaluation_id == written.manifest_id
    assert written.source_schema_version == 1
    assert written.legacy_identity is True


def test_manifest_round_trip_verifies_manifest_and_case_digests(tmp_path) -> None:
    path = tmp_path / "evaluation_manifest.json"
    manifest = _manifest()
    manifest.write_new(path)

    loaded = EvaluationManifest.load(path)
    assert loaded.manifest_id == manifest.manifest_id
    assert loaded.cases == manifest.cases

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="case_id|manifest_id"):
        EvaluationManifest.load(path)
