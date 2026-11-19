from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa.object_registry_preflight import (
    ObjectRegistryCoverage,
    ObjectRegistryGap,
    _has_candidates,
    _inspection_payload,
    _recommend_registries,
    inspect_object_registry_coverage,
)


def _interpreter(tmp_path: Path) -> Path:
    executable = tmp_path / "robocasa-python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _runner(payload: dict):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    run.calls = calls
    return run


def _payload(
    interpreter: Path,
    *,
    selected_registries: list[str],
    gaps: list[dict],
    recommended_registries: list[str],
) -> dict:
    return {
        "schema_version": 1,
        "interpreter": str(interpreter.absolute()),
        "selector": "AddIceCubes",
        "selected_registries": selected_registries,
        "gaps": gaps,
        "recommended_registries": recommended_registries,
    }


def test_add_ice_cubes_lightwheel_reports_bowl_and_objaverse(
    tmp_path: Path,
) -> None:
    interpreter = _interpreter(tmp_path)
    runner = _runner(
        _payload(
            interpreter,
            selected_registries=["lightwheel"],
            gaps=[{"task": "AddIceCubes", "object_groups": ["bowl"]}],
            recommended_registries=["lightwheel", "objaverse"],
        )
    )

    report = inspect_object_registry_coverage(
        interpreter,
        selector="AddIceCubes",
        selected_registries=("lightwheel",),
        seed=0,
        runner=runner,
    )

    assert report == ObjectRegistryCoverage(
        selected_registries=("lightwheel",),
        gaps=(ObjectRegistryGap("AddIceCubes", ("bowl",)),),
        recommended_registries=("lightwheel", "objaverse"),
    )
    command, kwargs = runner.calls[0]
    assert command == [
        str(interpreter.absolute()),
        "-m",
        "robots.robocasa.object_registry_preflight",
        "--inspect-json",
        "--selector",
        "AddIceCubes",
        "--object-registries",
        "lightwheel",
        "--seed",
        "0",
    ]
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_add_ice_cubes_lightwheel_objaverse_passes(tmp_path: Path) -> None:
    interpreter = _interpreter(tmp_path)
    runner = _runner(
        _payload(
            interpreter,
            selected_registries=["lightwheel", "objaverse"],
            gaps=[],
            recommended_registries=["lightwheel", "objaverse"],
        )
    )

    report = inspect_object_registry_coverage(
        interpreter,
        selector="AddIceCubes",
        selected_registries=("lightwheel", "objaverse"),
        seed=0,
        runner=runner,
    )

    assert report.gaps == ()
    assert report.recommended_registries == ("lightwheel", "objaverse")


@pytest.mark.parametrize(
    "registries",
    [(), ("lightwheel", "lightwheel"), ("unknown",)],
)
def test_invalid_selected_registries_do_not_spawn(
    tmp_path: Path,
    registries: tuple[str, ...],
) -> None:
    interpreter = _interpreter(tmp_path)
    runner = _runner({})

    with pytest.raises(ValueError, match="registr"):
        inspect_object_registry_coverage(
            interpreter,
            selector="AddIceCubes",
            selected_registries=registries,
            seed=0,
            runner=runner,
        )

    assert runner.calls == []


def test_report_identity_must_match_requested_interpreter(tmp_path: Path) -> None:
    interpreter = _interpreter(tmp_path)
    runner = _runner(
        _payload(
            interpreter,
            selected_registries=["lightwheel"],
            gaps=[],
            recommended_registries=["lightwheel"],
        )
        | {"interpreter": "/different/python"}
    )

    with pytest.raises(RuntimeError, match="different interpreter"):
        inspect_object_registry_coverage(
            interpreter,
            selector="AddIceCubes",
            selected_registries=("lightwheel",),
            seed=0,
            runner=runner,
        )


def test_subprocess_failure_names_interpreter(tmp_path: Path) -> None:
    interpreter = _interpreter(tmp_path)

    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=[str(interpreter)], timeout=1, stderr="probe timed out"
        )

    with pytest.raises(RuntimeError, match=str(interpreter.absolute())):
        inspect_object_registry_coverage(
            interpreter,
            selector="AddIceCubes",
            selected_registries=("lightwheel",),
            seed=0,
            runner=runner,
        )


def test_recommendation_appends_the_first_registry_covering_every_gap() -> None:
    support = {
        "bowl": frozenset({"objaverse", "aigen"}),
        "ice_cube": frozenset({"lightwheel"}),
    }

    assert _recommend_registries(
        ("lightwheel",),
        ("bowl", "ice_cube"),
        support,
    ) == ("lightwheel", "objaverse")


def test_recommendation_finds_the_smallest_multi_registry_addition() -> None:
    support = {
        "bowl": frozenset({"objaverse"}),
        "butter_stick": frozenset({"aigen"}),
    }

    assert _recommend_registries(
        ("lightwheel",),
        ("bowl", "butter_stick"),
        support,
    ) == ("lightwheel", "objaverse", "aigen")


def test_inspection_payload_aggregates_task_gaps_in_task_order() -> None:
    tasks = (
        SimpleNamespace(name="AddIceCubes", split="target"),
        SimpleNamespace(name="CloseFridge", split="target"),
    )
    probe_calls = []

    def probe(task, split, registries, seed):
        probe_calls.append((task, split, registries, seed))
        return ("bowl", "bowl") if task == "AddIceCubes" else ()

    payload = _inspection_payload(
        "AddIceCubes,CloseFridge",
        ("lightwheel",),
        7,
        resolve_tasks=lambda _selector: tasks,
        probe_task=probe,
        registry_support=lambda group: {"bowl": frozenset({"objaverse", "aigen"})}[
            group
        ],
    )

    assert payload == {
        "schema_version": 1,
        "interpreter": str(Path(sys.executable).absolute()),
        "selector": "AddIceCubes,CloseFridge",
        "selected_registries": ["lightwheel"],
        "gaps": [{"task": "AddIceCubes", "object_groups": ["bowl"]}],
        "recommended_registries": ["lightwheel", "objaverse"],
    }
    assert probe_calls == [
        ("AddIceCubes", "target", ("lightwheel",), 7),
        ("CloseFridge", "target", ("lightwheel",), 7),
    ]


def test_inspection_payload_keeps_passing_selection() -> None:
    payload = _inspection_payload(
        "AddIceCubes",
        ("lightwheel", "objaverse"),
        0,
        resolve_tasks=lambda _selector: (
            SimpleNamespace(name="AddIceCubes", split="target"),
        ),
        probe_task=lambda *_args: (),
        registry_support=lambda _group: frozenset(),
    )

    assert payload["gaps"] == []
    assert payload["recommended_registries"] == ["lightwheel", "objaverse"]


class _ObjectMetadata:
    def __init__(self, paths: tuple[str, ...], **properties: bool) -> None:
        self.mjcf_paths = paths
        for name in (
            "graspable",
            "washable",
            "microwavable",
            "cookable",
            "fridgable",
            "freezable",
            "dishwashable",
        ):
            setattr(self, name, properties.get(name, False))


def test_candidate_check_matches_add_ice_cubes_registry_coverage() -> None:
    categories = {
        "bowl": {
            "objaverse": _ObjectMetadata(tuple(f"bowl-{i}" for i in range(8))),
            "aigen": _ObjectMetadata(tuple(f"aigen-bowl-{i}" for i in range(8))),
        },
        "ice_cube": {
            "lightwheel": _ObjectMetadata(
                tuple(f"ice-{i}" for i in range(8)), graspable=True
            )
        },
    }
    groups = {"bowl": ["bowl"], "ice_cube": ["ice_cube"]}

    assert not _has_candidates(
        "bowl", None, ("lightwheel",), "target", categories, groups, {}
    )
    assert _has_candidates(
        "bowl",
        None,
        ("lightwheel", "objaverse"),
        "target",
        categories,
        groups,
        {},
    )
    assert _has_candidates(
        "ice_cube",
        None,
        ("lightwheel",),
        "target",
        categories,
        groups,
        {"graspable": True},
    )


def test_candidate_check_honors_properties_and_split_membership() -> None:
    categories = {
        "bowl": {
            "objaverse": _ObjectMetadata(("only-object",), washable=False),
        }
    }
    groups = {"bowl": ["bowl"]}

    assert not _has_candidates(
        "bowl",
        None,
        ("objaverse",),
        "target",
        categories,
        groups,
        {"washable": True},
    )
    assert not _has_candidates(
        "bowl",
        None,
        ("objaverse",),
        "target",
        categories,
        groups,
        {},
    )
    assert _has_candidates(
        "bowl",
        None,
        ("objaverse",),
        "pretrain",
        categories,
        groups,
        {},
    )


def test_candidate_check_accepts_any_selected_registry_matching_properties() -> None:
    categories = {
        "bowl": {
            "lightwheel": _ObjectMetadata(("lightwheel-bowl",), washable=False),
            "objaverse": _ObjectMetadata(("objaverse-bowl",), washable=True),
        }
    }
    groups = {"bowl": ["bowl"]}

    assert _has_candidates(
        "bowl",
        None,
        ("lightwheel", "objaverse"),
        None,
        categories,
        groups,
        {"washable": True},
    )


def test_candidate_check_requires_explicit_xml_in_a_selected_registry(
    tmp_path: Path,
) -> None:
    model = tmp_path / "object" / "model.xml"
    model.parent.mkdir()
    model.write_text("<mujoco />", encoding="utf-8")
    categories = {
        "bowl": {
            "objaverse": _ObjectMetadata((str(model),)),
        }
    }

    assert not _has_candidates(
        str(model), None, ("lightwheel",), None, categories, {}, {}
    )
    assert _has_candidates(str(model), None, ("objaverse",), None, categories, {}, {})
