"""Preflight RoboCasa task object groups against selected registries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.robocasa.dependencies import _extract_json, _resolve_interpreter
from rpent.utils.config import get_benchmark_preview_timeout_s

_SCHEMA_VERSION = 1
_REGISTRY_ORDER = ("objaverse", "lightwheel", "aigen")


@dataclass(frozen=True, slots=True)
class ObjectRegistryGap:
    task: str
    object_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObjectRegistryCoverage:
    selected_registries: tuple[str, ...]
    gaps: tuple[ObjectRegistryGap, ...]
    recommended_registries: tuple[str, ...]


def _validated_registries(registries: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(registries)
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(_REGISTRY_ORDER)
    ):
        raise ValueError("invalid RoboCasa object registries")
    return selected


def inspect_object_registry_coverage(
    interpreter: str | Path | None = None,
    *,
    selector: str,
    selected_registries: Sequence[str],
    seed: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ObjectRegistryCoverage:
    """Run the object-registry probe in the dedicated RoboCasa interpreter."""

    selected = _resolve_interpreter(interpreter)
    registries = _validated_registries(selected_registries)
    command = [
        str(selected),
        "-m",
        "robots.robocasa.object_registry_preflight",
        "--inspect-json",
        "--selector",
        selector,
        "--object-registries",
        ",".join(registries),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    try:
        completed = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=get_benchmark_preview_timeout_s(),
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"cannot inspect RoboCasa object registries with {selected}: {detail}"
        ) from exc
    payload = _extract_json(completed.stdout)
    return _coverage_from_payload(
        payload,
        interpreter=selected,
        selector=selector,
        selected_registries=registries,
    )


def _coverage_from_payload(
    payload: Mapping[str, Any],
    *,
    interpreter: Path,
    selector: str,
    selected_registries: tuple[str, ...],
) -> ObjectRegistryCoverage:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("unsupported RoboCasa object-registry report schema")
    reported = Path(
        os.path.abspath(Path(str(payload.get("interpreter") or "")).expanduser())
    )
    if reported != interpreter:
        raise RuntimeError(
            "RoboCasa object-registry report used a different interpreter: "
            f"expected {interpreter}, got {reported}"
        )
    if payload.get("selector") != selector:
        raise RuntimeError("RoboCasa object-registry report changed the task selector")
    if tuple(payload.get("selected_registries") or ()) != selected_registries:
        raise RuntimeError(
            "RoboCasa object-registry report changed selected registries"
        )
    raw_gaps = payload.get("gaps")
    raw_recommended = payload.get("recommended_registries")
    if not isinstance(raw_gaps, list) or not isinstance(raw_recommended, list):
        raise RuntimeError("invalid RoboCasa object-registry report")
    try:
        gaps = tuple(
            ObjectRegistryGap(
                task=str(item["task"]),
                object_groups=tuple(str(group) for group in item["object_groups"]),
            )
            for item in raw_gaps
        )
        recommended = _validated_registries(tuple(raw_recommended))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid RoboCasa object-registry report") from exc
    if recommended[: len(selected_registries)] != selected_registries:
        raise RuntimeError("RoboCasa registry recommendation changed selected order")
    return ObjectRegistryCoverage(selected_registries, gaps, recommended)


def _recommend_registries(
    selected_registries: tuple[str, ...],
    missing_groups: Sequence[str],
    support: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    missing = tuple(dict.fromkeys(missing_groups))
    if not missing:
        return selected_registries
    remaining = tuple(
        registry for registry in _REGISTRY_ORDER if registry not in selected_registries
    )
    for count in range(1, len(remaining) + 1):
        for addition in itertools.combinations(remaining, count):
            enabled = set(selected_registries + addition)
            if all(enabled.intersection(support.get(group, ())) for group in missing):
                return selected_registries + addition
    rendered = ", ".join(missing)
    raise RuntimeError(f"no RoboCasa object registry covers groups: {rendered}")


def _registry_support(group: str) -> frozenset[str]:
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS

    categories = OBJ_GROUPS.get(group)
    if not categories:
        return frozenset()
    return frozenset(
        registry
        for registry in _REGISTRY_ORDER
        if any(registry in OBJ_CATEGORIES[category] for category in categories)
    )


def _resolved_tasks(selector: str):
    from robots.robocasa.tasks import resolve_robocasa_tasks_from_contract

    return resolve_robocasa_tasks_from_contract(selector)


def _inspection_payload(
    selector: str,
    selected_registries: tuple[str, ...],
    seed: int,
    *,
    resolve_tasks: Callable[[str], Sequence[Any]] = _resolved_tasks,
    probe_task: Callable[[str, str, tuple[str, ...], int], Sequence[str]] | None = None,
    registry_support: Callable[[str], frozenset[str]] = _registry_support,
) -> dict[str, Any]:
    probe = _probe_task if probe_task is None else probe_task
    gaps: list[ObjectRegistryGap] = []
    missing_groups: list[str] = []
    for task in resolve_tasks(selector):
        groups = tuple(
            dict.fromkeys(
                str(group)
                for group in probe(task.name, task.split, selected_registries, seed)
            )
        )
        if groups:
            gaps.append(ObjectRegistryGap(str(task.name), groups))
            missing_groups.extend(groups)
    support = {
        group: registry_support(group) for group in dict.fromkeys(missing_groups)
    }
    recommended = _recommend_registries(
        selected_registries,
        missing_groups,
        support,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "interpreter": str(Path(sys.executable).absolute()),
        "selector": selector,
        "selected_registries": list(selected_registries),
        "gaps": [
            {"task": gap.task, "object_groups": list(gap.object_groups)} for gap in gaps
        ],
        "recommended_registries": list(recommended),
    }


def _probe_task(
    task: str,
    split: str,
    selected_registries: tuple[str, ...],
    seed: int,
) -> tuple[str, ...]:
    from robocasa.models.objects import kitchen_object_utils
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
    from robocasa.utils.env_utils import create_env

    missing: list[str] = []
    original = kitchen_object_utils.sample_kitchen_object_helper

    def traced(groups, *args, **kwargs):
        registries = tuple(kwargs.get("obj_registries") or selected_registries)
        if _has_candidates(
            groups,
            kwargs.get("exclude_groups"),
            registries,
            kwargs.get("split"),
            OBJ_CATEGORIES,
            OBJ_GROUPS,
            kwargs,
        ):
            return original(groups, *args, **kwargs)
        group_names = _group_names(groups)
        missing.extend(group for group in group_names if group not in missing)
        fallback = tuple(dict.fromkeys(registries + _REGISTRY_ORDER))
        if not _has_candidates(
            groups,
            kwargs.get("exclude_groups"),
            fallback,
            kwargs.get("split"),
            OBJ_CATEGORIES,
            OBJ_GROUPS,
            kwargs,
        ):
            raise RuntimeError(
                f"RoboCasa has no loaded object candidates for {group_names}"
            )
        return original(groups, *args, **(kwargs | {"obj_registries": fallback}))

    kitchen_object_utils.sample_kitchen_object_helper = traced
    environment = None
    try:
        environment = create_env(
            env_name=task,
            split=split,
            seed=seed,
            obj_registries=selected_registries,
            camera_names=[],
            camera_widths=1,
            camera_heights=1,
        )
    finally:
        kitchen_object_utils.sample_kitchen_object_helper = original
        if environment is not None:
            environment.close()
    return tuple(missing)


def _group_names(groups: Any) -> tuple[str, ...]:
    if isinstance(groups, str):
        return (groups,)
    return tuple(str(group) for group in groups)


def _has_candidates(
    groups: Any,
    exclude_groups: Any,
    registries: tuple[str, ...],
    split: str | None,
    categories: Mapping[str, Mapping[str, Any]],
    object_groups: Mapping[str, Sequence[str]],
    constraints: Mapping[str, Any],
) -> bool:
    names = _group_names(groups)
    if len(names) == 1 and names[0].endswith(".xml"):
        model_xml = str(Path(names[0]).with_name("model.xml"))
        return any(
            model_xml in category[registry].mjcf_paths
            for category in categories.values()
            for registry in registries
            if registry in category
        )
    excluded_names = () if exclude_groups is None else _group_names(exclude_groups)
    excluded = {
        category for group in excluded_names for category in object_groups[group]
    }
    properties = (
        "graspable",
        "washable",
        "microwavable",
        "cookable",
        "fridgable",
        "freezable",
        "dishwashable",
    )
    for group in names:
        for category in object_groups[group]:
            if category in excluded:
                continue
            available = [
                categories[category][registry]
                for registry in registries
                if registry in categories[category]
            ]
            for metadata in available:
                if any(
                    constraints.get(prop) is True
                    and getattr(metadata, prop) is not True
                    for prop in properties
                ):
                    continue
                paths = tuple(metadata.mjcf_paths)
                if split is not None:
                    threshold = max(len(paths) - 5, math.ceil(len(paths) / 2))
                    paths = (
                        paths[:threshold] if split == "pretrain" else paths[threshold:]
                    )
                if paths:
                    return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", action="store_true", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--object-registries", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    payload = _inspection_payload(
        args.selector,
        _validated_registries(args.object_registries.split(",")),
        args.seed,
    )
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ObjectRegistryCoverage",
    "ObjectRegistryGap",
    "inspect_object_registry_coverage",
]
