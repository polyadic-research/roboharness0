from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.evaluation.cases import EpisodeProtocol, expand_episode_cases
from rpent.evaluation.front_controller import build_evaluation_manifest
from rpent.evaluation.manifest import EvaluationManifest
from rpent.planning.manifest import canonical_json, content_hash
from tests.synthetic_tasks import SyntheticTask


def _tasks(count: int = 1):
    return tuple(
        SyntheticTask(
            suite="libero_10",
            index=index,
            name=f"task_{index}",
            language=f"instruction {index}",
            bddl_identifier=f"task_{index}.bddl",
            upstream_horizon=20,
        )
        for index in range(count)
    )


def _cases(count: int = 1):
    tasks = _tasks(count)
    protocol = EpisodeProtocol(
        environment="synthetic",
        preset_name="preset",
        horizon_ceiling=20,
        checkpoint_repo_id="repo/checkpoint",
        checkpoint_revision="7" * 40,
        binding_name="binding-v1",
    )
    return expand_episode_cases(tasks, protocol, base_seed=0)


def _manifest(**overrides) -> EvaluationManifest:
    values = {
        "rpent_commit": "a" * 40,
        "rpent_dirty": False,
        "cli_arguments": {"argv": []},
        "environment": {"name": "synthetic"},
        "simulator": {"revision": "b" * 40},
        "policy": {"revision": "c" * 40},
        "task_selector": "all",
        "cases": _cases(),
        "schedule": {"episodes_per_task": 1},
        "contracts": {"action": {"shape": [7]}},
        "runtime": {"evaluation_root": "/tmp/evaluation"},
        "created_at": "2026-08-07T00:00:00+00:00",
    }
    values.update(overrides)
    return EvaluationManifest(**values)


def test_new_evaluations_have_distinct_ids_but_same_config_digest() -> None:
    first = _manifest()
    second = _manifest()

    assert first.evaluation_id != second.evaluation_id
    assert first.manifest_id == second.manifest_id
    assert first.to_payload()["schema_version"] == 3


def test_recording_and_qualification_are_immutable_identity_inputs() -> None:
    original = _manifest(
        recording={"schema_version": 1, "mode": "all", "camera_ids": []},
        qualification={"require_complete_artifacts": False},
    )

    assert (
        _manifest(
            recording={"schema_version": 1, "mode": "primary", "camera_ids": []},
            qualification={"require_complete_artifacts": False},
        ).manifest_id
        != original.manifest_id
    )
    assert (
        _manifest(
            recording={"schema_version": 1, "mode": "all", "camera_ids": []},
            qualification={"require_complete_artifacts": True},
        ).manifest_id
        != original.manifest_id
    )


def test_schema_v1_load_uses_verified_manifest_id_as_legacy_evaluation_id(
    tmp_path: Path,
) -> None:
    current = _manifest()
    identity = current.identity_payload()
    identity["schema_version"] = 1
    identity.pop("launch")
    identity.pop("recording")
    identity.pop("qualification")
    legacy_id = content_hash(identity)
    payload = {
        **identity,
        "manifest_id": legacy_id,
        "created_at": current.created_at,
    }
    path = tmp_path / "legacy.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    before = path.read_bytes()

    loaded = EvaluationManifest.load(path)

    assert loaded.schema_version == 1
    assert loaded.manifest_id == legacy_id
    assert loaded.evaluation_id == legacy_id
    assert loaded.source_schema_version == 1
    assert loaded.legacy_identity is True
    assert path.read_bytes() == before


def test_runtime_output_locations_do_not_change_reproducible_identity() -> None:
    first = _manifest(
        runtime={
            "main_interpreter": "/opt/rpent/python",
            "evaluation_root": "/runs/one",
            "output_dir": "/runs/one/output",
            "rendering_backend": "egl",
        }
    )
    second = _manifest(
        runtime={
            "main_interpreter": "/opt/rpent/python",
            "evaluation_root": "/runs/two",
            "output_dir": "/runs/two/output",
            "rendering_backend": "egl",
        }
    )

    assert first.manifest_id == second.manifest_id
    assert first.to_payload()["runtime"] != second.to_payload()["runtime"]
    assert (
        _manifest(
            runtime={
                "main_interpreter": "/different/python",
                "evaluation_root": "/runs/one",
                "output_dir": "/runs/one/output",
                "rendering_backend": "egl",
            }
        ).manifest_id
        != first.manifest_id
    )


class _Spec:
    def __init__(self, camera_count: int | None = 3) -> None:
        self.camera_count = camera_count

    def parse_config(self, args):
        return SimpleNamespace(output_dir=args.output_dir)

    def prepare_evaluation(self, args, run_config, resolved_tasks):
        output_dir = args.output_dir
        del run_config
        return SimpleNamespace(
            tasks=tuple(resolved_tasks),
            protocol=EpisodeProtocol(
                environment="synthetic",
                preset_name="preset",
                horizon_ceiling=20,
                checkpoint_repo_id="repo/checkpoint",
                checkpoint_revision="7" * 40,
                binding_name="binding-v1",
            ),
            environment={"name": "synthetic"},
            simulator={"task_resolution": "packaged_contract"},
            policy={},
            contracts=(
                {} if self.camera_count is None
                else {"images": {"count": self.camera_count}}
            ),
            runtime={"evaluation_root": str(output_dir)},
        )


def _args(tmp_path: Path, *, mode=None, camera_ids=()):
    return Namespace(
        env_name="synthetic",
        task="all",
        episodes=1,
        seed=0,
        output_dir=tmp_path,
        recording_mode=mode,
        recording_camera=list(camera_ids),
        require_complete_artifacts=False,
    )


def test_front_controller_defaults_single_to_all_and_multiple_to_primary(
    tmp_path: Path,
) -> None:
    single, _ = build_evaluation_manifest(
        _args(tmp_path),
        _Spec(),
        [],
        resolved_tasks=_tasks(1),
        git_identity=lambda: ("a", False),
    )
    multiple, _ = build_evaluation_manifest(
        _args(tmp_path),
        _Spec(),
        [],
        resolved_tasks=_tasks(2),
        git_identity=lambda: ("a", False),
    )

    assert single.recording["mode"] == "all"
    assert multiple.recording["mode"] == "primary"


@pytest.mark.parametrize("mode", [None, "none"])
def test_robocasa_defaults_to_exact_llm_images_without_control_step_frames(tmp_path: Path, mode) -> None:
    from dataclasses import replace

    class RoboCasaSpec(_Spec):
        def prepare_evaluation(self, *args, **kwargs):
            prepared = super().prepare_evaluation(*args, **kwargs)
            prepared.protocol = replace(prepared.protocol, environment="robocasa")
            prepared.environment = {"name": "robocasa"}
            return prepared

    args = _args(tmp_path)
    args.env_name = "robocasa"
    args.recording_mode = mode
    manifest, _ = build_evaluation_manifest(args, RoboCasaSpec(), [], resolved_tasks=_tasks(1),
                                             git_identity=lambda: ("a", False))
    assert manifest.recording["mode"] == "none"
    assert manifest.recording["llm_inputs_only"] is True
    assert manifest.recording["llm_images"]["width"] == manifest.recording["llm_images"]["height"] == 256
    assert tuple(manifest.recording["llm_images"]["cameras"]) == (
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    )
    args.recording_mode = "all"
    explicit, _ = build_evaluation_manifest(args, RoboCasaSpec(), [], resolved_tasks=_tasks(1),
                                             git_identity=lambda: ("a", False))
    assert explicit.recording["mode"] == "all"
    assert "llm_inputs_only" not in explicit.recording


def test_robocasa_dashboard_launch_preset_defaults_to_llm_image_retention() -> None:
    from robots.robocasa.launch import build_launch_preset

    fields = {field.name: field for field in build_launch_preset().fields}
    assert fields["recording_mode"].default == "none"


def test_llm_only_image_setting_rejects_inconsistent_control_step_retention() -> None:
    with pytest.raises(ValueError, match="LLM-only"):
        _manifest(recording={"schema_version": 1, "mode": "all", "camera_ids": [], "llm_inputs_only": True})


def test_front_controller_preserves_explicit_selected_profile(tmp_path: Path) -> None:
    manifest, _ = build_evaluation_manifest(
        _args(tmp_path, mode="selected", camera_ids=("wrist",)),
        _Spec(),
        [],
        resolved_tasks=_tasks(2),
        git_identity=lambda: ("a", False),
    )

    assert manifest.recording["schema_version"] == 1
    assert manifest.recording["mode"] == "selected"
    assert manifest.recording["camera_ids"] == ("wrist",)
    assert dict(manifest.recording["storage_estimate"]) == {
        "minimum_bytes": 327680,
        "maximum_bytes": 20971520,
        "basis": manifest.recording["storage_estimate"]["basis"],
    }
    assert dict(manifest.recording["storage_estimate"]["basis"]) == {
        "total_horizon_steps": 40,
        "retained_camera_count_min": 1,
        "retained_camera_count_max": 1,
        "bytes_per_camera_step_min": 8192,
        "bytes_per_camera_step_max": 524288,
        "contract_camera_count": 3,
        "task_resolution": "packaged_contract",
        "note": "generic compressed-frame range; simulator output may vary",
    }
    assert {
        "schema_version": 1,
        "mode": "selected",
        "camera_ids": ("wrist",),
    }.items() <= dict(manifest.recording).items()
    assert json.loads(canonical_json(manifest.to_payload()))["recording"][
        "camera_ids"
    ] == ["wrist"]


def test_all_recording_uses_exact_contract_camera_count(tmp_path: Path) -> None:
    manifest, _ = build_evaluation_manifest(
        _args(tmp_path, mode="all"),
        _Spec(camera_count=3),
        [],
        resolved_tasks=_tasks(1),
        git_identity=lambda: ("a", False),
    )

    estimate = manifest.recording["storage_estimate"]
    assert estimate["basis"]["retained_camera_count_min"] == 3
    assert estimate["basis"]["retained_camera_count_max"] == 3
    assert estimate["minimum_bytes"] == 20 * 3 * 8192
    assert estimate["maximum_bytes"] == 20 * 3 * 524288


@pytest.mark.parametrize("camera_count", [0, None])
def test_default_recording_without_cameras_is_none(
    tmp_path: Path,
    camera_count: int | None,
) -> None:
    single, _ = build_evaluation_manifest(
        _args(tmp_path),
        _Spec(camera_count=camera_count),
        [],
        resolved_tasks=_tasks(1),
        git_identity=lambda: ("a", False),
    )
    multiple, _ = build_evaluation_manifest(
        _args(tmp_path),
        _Spec(camera_count=camera_count),
        [],
        resolved_tasks=_tasks(2),
        git_identity=lambda: ("a", False),
    )

    assert single.recording["mode"] == "none"
    assert multiple.recording["mode"] == "none"
    assert single.recording["storage_estimate"]["maximum_bytes"] == 0
    assert multiple.recording["storage_estimate"]["maximum_bytes"] == 0


@pytest.mark.parametrize("mode", ["none", "all", "primary"])
def test_explicit_recording_without_cameras_is_preserved(tmp_path, mode) -> None:
    manifest, _ = build_evaluation_manifest(
        _args(tmp_path, mode=mode),
        _Spec(camera_count=0),
        [],
        resolved_tasks=_tasks(1),
        git_identity=lambda: ("a", False),
    )

    assert manifest.recording["mode"] == mode


def test_selected_camera_count_cannot_exceed_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contract camera count"):
        build_evaluation_manifest(
            _args(tmp_path, mode="selected", camera_ids=("one", "two")),
            _Spec(camera_count=1),
            [],
            resolved_tasks=_tasks(1),
            git_identity=lambda: ("a", False),
        )


def test_manifest_rejects_invalid_optional_recording_storage_estimate() -> None:
    with pytest.raises(ValueError, match="storage_estimate"):
        _manifest(
            recording={
                "schema_version": 1,
                "mode": "all",
                "camera_ids": [],
                "storage_estimate": {
                    "minimum_bytes": -1,
                    "maximum_bytes": 10,
                    "basis": {},
                },
            }
        )


def test_manifest_accepts_pre_contract_storage_estimate_basis() -> None:
    manifest = _manifest(
        recording={
            "schema_version": 1,
            "mode": "all",
            "camera_ids": [],
            "storage_estimate": {
                "minimum_bytes": 8192,
                "maximum_bytes": 524288,
                "basis": {
                    "total_horizon_steps": 1,
                    "retained_camera_count_min": 1,
                    "retained_camera_count_max": 1,
                    "bytes_per_camera_step_min": 8192,
                    "bytes_per_camera_step_max": 524288,
                    "note": "legacy estimate",
                },
            },
        }
    )

    assert manifest.recording["storage_estimate"]["basis"]["note"] == (
        "legacy estimate"
    )
