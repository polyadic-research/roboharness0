from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa import tasks as robocasa_tasks
from robots.robocasa.dependencies import ROBOCASA_COMMIT
from robots.robocasa.tasks import (
    TASK_GROUP_FIXTURE,
    packaged_registry_snapshot,
    resolve_robocasa_tasks,
    resolve_robocasa_tasks_from_contract,
    resolve_selector_from_snapshot,
)


def test_task_selector_suggestions_start_with_groups_and_include_tasks() -> None:
    suggestions = robocasa_tasks.task_selector_suggestions()

    assert suggestions[:7] == (
        "atomic_seen",
        "composite_seen",
        "composite_unseen",
        "pretrain50",
        "pretrain100",
        "pretrain200",
        "pretrain300",
    )
    assert "CloseFridge" in suggestions


def test_contract_export_omits_helpers_but_live_resolution_preserves_them(
    monkeypatch, tmp_path
) -> None:
    class Kitchen:
        def __init__(self, horizon=1000):
            pass

        def _check_success(self):
            return False

    class AtomicTask(Kitchen):
        def _check_success(self):
            return True

    AtomicTask.__module__ = "robocasa.environments.kitchen.atomic.kitchen_sink"
    module = SimpleNamespace(__file__=str(tmp_path / "robocasa/__init__.py"))
    monkeypatch.setitem(sys.modules, "robocasa", module)
    monkeypatch.setitem(
        sys.modules,
        "robocasa.environments.kitchen.kitchen",
        SimpleNamespace(
            Kitchen=Kitchen,
            REGISTERED_KITCHEN_ENVS={
                "AdjustWaterTemperature": AtomicTask,
                "ManipulateSinkFaucet": AtomicTask,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.dataset_registry",
        SimpleNamespace(
            ATOMIC_TASK_DATASETS={"AdjustWaterTemperature": {"horizon": 450}},
            COMPOSITE_TASK_DATASETS={},
            TASK_SET_REGISTRY={name: [] for name in robocasa_tasks._GROUP_NAMES},
        ),
    )
    monkeypatch.setattr(
        robocasa_tasks, "_distribution_metadata", lambda _: ("1.0.1", ROBOCASA_COMMIT)
    )
    monkeypatch.setattr(
        robocasa_tasks,
        "_load_descriptions",
        lambda _: {"AdjustWaterTemperature": "Adjust the water temperature."},
    )
    live = robocasa_tasks._upstream_registry_snapshot()
    assert set(live["tasks"]) == {"AdjustWaterTemperature", "ManipulateSinkFaucet"}
    contract = robocasa_tasks._contract_payload()
    assert set(contract["tasks"]) == {"AdjustWaterTemperature"}
    assert contract["tasks"]["AdjustWaterTemperature"]["family"] == "atomic"
    assert contract["tasks"]["AdjustWaterTemperature"]["upstream_horizon"] == 450


def _fixture() -> dict:
    return json.loads(TASK_GROUP_FIXTURE.read_text(encoding="utf-8"))


def _registry_digest(groups: dict) -> str:
    encoded = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _snapshot() -> dict:
    fixture = _fixture()
    names = {name for group in fixture["groups"].values() for name in group["tasks"]}
    names.update({"OpenBlenderLid", "PickPlaceCoffee"})
    tasks = {
        name: {
            "registry_id": f"robocasa.fake:{name}",
            "instruction": f"Perform {name}.",
            "family": "atomic"
            if name in {"CloseFridge", "OpenBlenderLid", "PickPlaceCoffee"}
            else "composite",
            "category": "test",
            "upstream_horizon": 1000,
        }
        for name in names
    }
    tasks["CloseFridge"] = {
        "registry_id": (
            "robocasa.environments.kitchen.atomic.kitchen_doors:CloseFridge"
        ),
        "instruction": "Close the fridge door(s).",
        "family": "atomic",
        "category": "doors",
        "upstream_horizon": 900,
    }
    return {
        "repository": "https://github.com/robocasa/robocasa.git",
        "commit": ROBOCASA_COMMIT,
        "registry_digest": fixture["registry_digest"],
        "groups": fixture["groups"],
        "tasks": tasks,
    }


def _interpreter(tmp_path: Path) -> Path:
    executable = tmp_path / "robocasa-python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _response(selector: str, interpreter: Path) -> dict:
    tasks = resolve_selector_from_snapshot(selector, _snapshot())
    split = tasks[0].split
    return {
        "schema_version": 1,
        "interpreter": str(interpreter.absolute()),
        "selector": selector,
        "repository": tasks[0].registry_repository,
        "commit": tasks[0].registry_commit,
        "registry_digest": tasks[0].registry_digest,
        "split": split,
        "tasks": [
            {
                "name": task.name,
                "registry_id": task.registry_id,
                "instruction": task.instruction,
                "family": task.family,
                "category": task.category,
                "split": task.split,
                "upstream_horizon": task.upstream_horizon,
            }
            for task in tasks
        ],
    }


def test_packaged_registry_is_the_only_provenance_marked_task_contract() -> None:
    fixture = _fixture()

    assert fixture["schema_version"] == 1
    assert fixture["provenance"] == {
        "repository": "https://github.com/robocasa/robocasa.git",
        "commit": ROBOCASA_COMMIT,
        "source": "robocasa/utils/dataset_registry.py:TASK_SET_REGISTRY",
        "task_metadata_sources": [
            "robocasa/utils/dataset_registry.py",
            "robocasa/environments/kitchen",
            "docs/composite_tasks/task_attributes.json",
        ],
        "extraction_command": (
            "RPENT_ROBOCASA_PYTHON -m robots.robocasa.tasks --dump-contract-json"
        ),
        "extracted_at": "2026-09-29",
    }
    assert fixture["registry_digest"] == _registry_digest(fixture["groups"])
    encoded_tasks = json.dumps(
        fixture["tasks"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert len(fixture["tasks"]) == 365
    assert fixture["task_metadata_digest"] == (
        f"sha256:{hashlib.sha256(encoded_tasks).hexdigest()}"
    )
    assert TASK_GROUP_FIXTURE == (Path("robots/robocasa/task_groups.json").resolve())
    assert not Path("tests/fixtures/robocasa/task_groups.json").exists()


def test_packaged_registry_snapshot_returns_a_defensive_copy() -> None:
    snapshot = packaged_registry_snapshot()
    original = list(snapshot["groups"]["atomic_seen"]["tasks"])
    try:
        snapshot["groups"]["atomic_seen"]["tasks"].pop()

        fresh = packaged_registry_snapshot()

        assert fresh["groups"]["atomic_seen"]["tasks"] == original
    finally:
        snapshot["groups"]["atomic_seen"]["tasks"] = original


def test_packaged_contract_resolves_without_the_simulator_interpreter() -> None:
    tasks = resolve_robocasa_tasks_from_contract("CloseFridge,OpenCabinet")

    assert [task.name for task in tasks] == ["CloseFridge", "OpenCabinet"]
    assert tasks[0].instruction == "Close the fridge door(s)."
    assert tasks[0].upstream_horizon == 900
    assert {task.registry_commit for task in tasks} == {ROBOCASA_COMMIT}


def test_packaged_contract_scene_split_override_preserves_membership_and_provenance() -> (
    None
):
    default = resolve_robocasa_tasks_from_contract("atomic_seen")

    overridden = resolve_robocasa_tasks_from_contract(
        "atomic_seen",
        scene_split="pretrain",
    )

    assert [task.name for task in overridden] == [task.name for task in default]
    assert {task.split for task in default} == {"target"}
    assert {task.split for task in overridden} == {"pretrain"}
    assert [
        (task.registry_id, task.registry_commit, task.registry_digest)
        for task in overridden
    ] == [
        (task.registry_id, task.registry_commit, task.registry_digest)
        for task in default
    ]


def test_explicit_selector_normalizes_whitespace_and_preserves_order() -> None:
    tasks = resolve_selector_from_snapshot(
        " CloseFridge , OpenBlenderLid, PickPlaceCoffee ",
        _snapshot(),
    )

    assert [task.name for task in tasks] == [
        "CloseFridge",
        "OpenBlenderLid",
        "PickPlaceCoffee",
    ]
    assert {task.split for task in tasks} == {"target"}


def test_duplicate_explicit_tasks_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate.*CloseFridge"):
        resolve_selector_from_snapshot(
            "CloseFridge,OpenDrawer,CloseFridge",
            _snapshot(),
        )


@pytest.mark.parametrize(
    ("group_name", "split", "cardinality"),
    [
        ("atomic_seen", "target", 18),
        ("composite_seen", "target", 16),
        ("composite_unseen", "target", 16),
        ("pretrain50", "pretrain", 50),
        ("pretrain100", "pretrain", 100),
        ("pretrain200", "pretrain", 200),
        ("pretrain300", "pretrain", 300),
    ],
)
def test_group_order_split_and_cardinality_match_upstream_fixture(
    group_name: str,
    split: str,
    cardinality: int,
) -> None:
    fixture = _fixture()

    tasks = resolve_selector_from_snapshot(group_name, _snapshot())

    assert [task.name for task in tasks] == fixture["groups"][group_name]["tasks"]
    assert len(tasks) == cardinality
    assert {task.split for task in tasks} == {split}


def test_single_task_carries_upstream_metadata_and_provenance() -> None:
    task = resolve_selector_from_snapshot("CloseFridge", _snapshot())[0]

    assert task.name == "CloseFridge"
    assert task.registry_id.endswith(":CloseFridge")
    assert task.instruction == "Close the fridge door(s)."
    assert task.family == "atomic"
    assert task.category == "doors"
    assert task.split == "target"
    assert task.upstream_horizon == 900
    assert task.registry_commit == ROBOCASA_COMMIT
    assert task.registry_digest == _fixture()["registry_digest"]


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("ClsoeFridge", "unknown.*CloseFridge"),
        ("closefridge", "case mismatch.*CloseFridge"),
        ("Open", "ambiguous.*OpenBlenderLid"),
    ],
)
def test_unknown_ambiguous_and_case_mismatched_names_have_suggestions(
    selector: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_selector_from_snapshot(selector, _snapshot())


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "CloseFridge,",
        ",CloseFridge",
        "CloseFridge,,OpenDrawer",
    ],
)
def test_empty_selector_elements_are_rejected(selector: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        resolve_selector_from_snapshot(selector, _snapshot())


def test_group_shortcut_cannot_be_mixed_with_explicit_tasks() -> None:
    with pytest.raises(ValueError, match="group shortcut.*explicit"):
        resolve_selector_from_snapshot(
            "atomic_seen,CloseFridge",
            _snapshot(),
        )


def test_public_resolver_executes_in_selected_interpreter_and_returns_plain_json(
    tmp_path: Path,
) -> None:
    interpreter = _interpreter(tmp_path)
    calls = []
    selector = "CloseFridge,OpenBlenderLid,PickPlaceCoffee"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_response(selector, interpreter)),
            stderr="",
        )

    before = {name for name in sys.modules if name == "robocasa"}
    tasks = resolve_robocasa_tasks(selector, interpreter=interpreter, runner=run)

    assert [task.name for task in tasks] == [
        "CloseFridge",
        "OpenBlenderLid",
        "PickPlaceCoffee",
    ]
    assert calls[0][0] == [
        str(interpreter.resolve()),
        "-m",
        "robots.robocasa.tasks",
        "--resolve-json",
        selector,
    ]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["capture_output"] is True
    assert {name for name in sys.modules if name == "robocasa"} == before


def test_public_resolver_scene_split_override_preserves_membership_and_provenance(
    tmp_path: Path,
) -> None:
    interpreter = _interpreter(tmp_path)
    selector = "atomic_seen"

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_response(selector, interpreter)),
            stderr="",
        )

    tasks = resolve_robocasa_tasks(
        selector,
        scene_split="pretrain",
        interpreter=interpreter,
        runner=run,
    )
    original = resolve_selector_from_snapshot(selector, _snapshot())

    assert [task.name for task in tasks] == [task.name for task in original]
    assert {task.split for task in tasks} == {"pretrain"}
    assert [
        (task.registry_id, task.registry_commit, task.registry_digest) for task in tasks
    ] == [
        (task.registry_id, task.registry_commit, task.registry_digest)
        for task in original
    ]


@pytest.mark.parametrize("scene_split", ["", "train", "TARGET"])
def test_scene_split_override_rejects_unknown_values(scene_split: str) -> None:
    with pytest.raises(ValueError, match="scene_split"):
        resolve_robocasa_tasks_from_contract(
            "CloseFridge",
            scene_split=scene_split,
        )


def test_public_resolver_uses_configured_benchmark_preview_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = _interpreter(tmp_path)
    calls = []
    monkeypatch.setenv("RPENT_BENCHMARK_PREVIEW_TIMEOUT_S", "137.5")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_response("CloseFridge", interpreter)),
            stderr="",
        )

    resolve_robocasa_tasks("CloseFridge", interpreter=interpreter, runner=run)

    assert calls[0][1]["timeout"] == 137.5


def test_public_resolver_preserves_virtualenv_interpreter_symlink(
    tmp_path: Path,
) -> None:
    target = _interpreter(tmp_path)
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(target)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_response("CloseFridge", virtualenv_python)),
            stderr="",
        )

    resolve_robocasa_tasks(
        "CloseFridge",
        interpreter=virtualenv_python,
        runner=run,
    )

    assert calls[0][0][0] == str(virtualenv_python)


def test_formal_resolver_fails_closed_on_registry_contract_drift(
    tmp_path: Path,
) -> None:
    interpreter = _interpreter(tmp_path)
    response = _response("CloseFridge", interpreter)
    response["registry_digest"] = "sha256:" + "0" * 64

    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(response),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="registry digest.*mismatch"):
        resolve_robocasa_tasks(
            "CloseFridge",
            interpreter=interpreter,
            runner=run,
        )


def test_task_resolver_subprocess_error_is_actionable(tmp_path: Path) -> None:
    interpreter = _interpreter(tmp_path)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="RuntimeError: registry unavailable",
        )

    with pytest.raises(RuntimeError, match="cannot resolve.*registry unavailable"):
        resolve_robocasa_tasks(
            "CloseFridge",
            interpreter=interpreter,
            runner=fail,
        )
