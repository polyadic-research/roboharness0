"""Resolve RoboCasa365 task selectors from the pinned upstream registry."""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    _distribution_metadata,
    _extract_json,
    _git_commit,
    _resolve_interpreter,
)
from rpent.utils.config import get_benchmark_preview_timeout_s

ROBOCASA_REPOSITORY = "https://github.com/robocasa/robocasa.git"
TASK_GROUP_FIXTURE = Path(__file__).resolve().with_name("task_groups.json")

_SCHEMA_VERSION = 1
_GROUP_NAMES = (
    "atomic_seen",
    "composite_seen",
    "composite_unseen",
    "pretrain50",
    "pretrain100",
    "pretrain200",
    "pretrain300",
)


@dataclass(frozen=True, slots=True)
class RoboCasaTask:
    """One immutable upstream RoboCasa task selected for evaluation."""

    name: str
    registry_id: str
    instruction: str
    family: str
    category: str
    split: str
    upstream_horizon: int
    registry_repository: str
    registry_commit: str
    registry_digest: str


@lru_cache(maxsize=1)
def _contract_fixture() -> dict[str, Any]:
    try:
        fixture = json.loads(TASK_GROUP_FIXTURE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot load RoboCasa task-group contract {TASK_GROUP_FIXTURE}: {exc}"
        ) from exc
    if fixture.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("unsupported RoboCasa task-group contract schema")
    return fixture


def task_selector_suggestions() -> tuple[str, ...]:
    """Return packaged group shortcuts followed by canonical task names."""
    tasks = _contract_fixture().get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise RuntimeError("invalid RoboCasa task-group contract tasks")
    return _GROUP_NAMES + tuple(sorted(str(name) for name in tasks))


def _registry_digest(groups: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        groups,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _task_metadata_digest(tasks: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        tasks,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _parse_selector(
    selector: str,
    group_names: Sequence[str],
) -> tuple[str, ...]:
    rendered = str(selector)
    if not rendered.strip():
        raise ValueError("RoboCasa task selector contains an empty element")
    parts = tuple(part.strip() for part in rendered.split(","))
    if any(not part for part in parts):
        raise ValueError("RoboCasa task selector contains an empty element")
    duplicates = sorted({part for part in parts if parts.count(part) > 1})
    if duplicates:
        raise ValueError(
            f"RoboCasa task selector contains duplicate tasks: {', '.join(duplicates)}"
        )
    selected_groups = [part for part in parts if part in group_names]
    if len(parts) > 1 and selected_groups:
        raise ValueError(
            "a RoboCasa group shortcut cannot be mixed with explicit tasks"
        )
    return parts


def _canonical_task_name(rendered: str, names: Sequence[str]) -> str:
    if rendered in names:
        return rendered

    case_matches = [name for name in names if name.casefold() == rendered.casefold()]
    if case_matches:
        raise ValueError(f"RoboCasa task name case mismatch; use '{case_matches[0]}'")

    partial_matches = sorted(
        name for name in names if name.casefold().startswith(rendered.casefold())
    )
    if len(partial_matches) > 1:
        raise ValueError(
            f"ambiguous RoboCasa task '{rendered}'; matches: "
            f"{', '.join(partial_matches[:8])}"
        )

    suggestions = difflib.get_close_matches(rendered, names, n=3, cutoff=0.5)
    suffix = (
        f"; nearest canonical names: {', '.join(suggestions)}" if suggestions else ""
    )
    raise ValueError(f"unknown RoboCasa task '{rendered}'{suffix}")


def _task_from_registry(
    name: str,
    *,
    split: str,
    snapshot: Mapping[str, Any],
) -> RoboCasaTask:
    registry = snapshot.get("tasks")
    if not isinstance(registry, Mapping) or name not in registry:
        raise RuntimeError(f"upstream RoboCasa registry has no metadata for {name}")
    metadata = registry[name]
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"invalid upstream RoboCasa task metadata for {name}")

    registry_id = str(metadata.get("registry_id") or "").strip()
    instruction = str(metadata.get("instruction") or "").strip()
    family = str(metadata.get("family") or "").strip()
    category = str(metadata.get("category") or "").strip()
    try:
        horizon = int(metadata.get("upstream_horizon"))
    except (TypeError, ValueError):
        horizon = 0
    if (
        not registry_id
        or not instruction
        or family not in {"atomic", "composite"}
        or not category
        or horizon <= 0
    ):
        raise RuntimeError(f"incomplete upstream RoboCasa task metadata for {name}")
    provenance = snapshot.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}

    return RoboCasaTask(
        name=name,
        registry_id=registry_id,
        instruction=instruction,
        family=family,
        category=category,
        split=split,
        upstream_horizon=horizon,
        registry_repository=str(
            provenance.get("repository") or snapshot.get("repository") or ""
        ),
        registry_commit=str(provenance.get("commit") or snapshot.get("commit") or ""),
        registry_digest=str(snapshot.get("registry_digest") or ""),
    )


def resolve_selector_from_snapshot(
    selector: str,
    snapshot: Mapping[str, Any],
    *,
    scene_split: str | None = None,
) -> tuple[RoboCasaTask, ...]:
    """Resolve a selector against a JSON-safe upstream registry snapshot."""

    groups = snapshot.get("groups")
    registry = snapshot.get("tasks")
    if not isinstance(groups, Mapping) or not isinstance(registry, Mapping):
        raise RuntimeError("invalid upstream RoboCasa registry snapshot")
    actual_digest = _registry_digest(groups)
    if snapshot.get("registry_digest") != actual_digest:
        raise RuntimeError("upstream RoboCasa registry snapshot digest is invalid")

    parts = _parse_selector(selector, tuple(groups))
    if len(parts) == 1 and parts[0] in groups:
        group = groups[parts[0]]
        if not isinstance(group, Mapping):
            raise RuntimeError(f"invalid upstream RoboCasa group {parts[0]}")
        split = str(group.get("split") or "")
        names = tuple(str(name) for name in group.get("tasks", ()))
        if split not in {"target", "pretrain"} or not names:
            raise RuntimeError(f"incomplete upstream RoboCasa group {parts[0]}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"upstream RoboCasa group {parts[0]} has duplicates")
    else:
        split = "target"
        available_names = tuple(str(name) for name in registry)
        names = tuple(_canonical_task_name(part, available_names) for part in parts)

    resolved = tuple(
        _task_from_registry(name, split=split, snapshot=snapshot) for name in names
    )
    return _with_scene_split(resolved, scene_split)


def _with_scene_split(
    tasks: tuple[RoboCasaTask, ...],
    scene_split: str | None,
) -> tuple[RoboCasaTask, ...]:
    if scene_split is None:
        return tasks
    if scene_split not in {"target", "pretrain"}:
        raise ValueError("scene_split must be 'target' or 'pretrain'")
    return tuple(replace(task, split=scene_split) for task in tasks)


def _expected_selection(selector: str) -> tuple[tuple[str, ...], str]:
    groups = _contract_fixture()["groups"]
    parts = _parse_selector(selector, tuple(groups))
    if len(parts) == 1 and parts[0] in groups:
        group = groups[parts[0]]
        return tuple(group["tasks"]), str(group["split"])
    return parts, "target"


def _response_tasks(payload: Mapping[str, Any]) -> tuple[RoboCasaTask, ...]:
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise RuntimeError("RoboCasa task resolver returned no tasks")
    snapshot = {
        "repository": payload.get("repository"),
        "commit": payload.get("commit"),
        "registry_digest": payload.get("registry_digest"),
        "tasks": {
            str(item.get("name")): item for item in raw_tasks if isinstance(item, dict)
        },
    }
    resolved = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise RuntimeError("RoboCasa task resolver returned invalid task metadata")
        name = str(item.get("name") or "")
        resolved.append(
            _task_from_registry(
                name,
                split=str(item.get("split") or ""),
                snapshot=snapshot,
            )
        )
    return tuple(resolved)


def resolve_robocasa_tasks(
    selector: str,
    *,
    scene_split: str | None = None,
    interpreter: str | Path | None = None,
    formal: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[RoboCasaTask, ...]:
    """Resolve tasks inside the dedicated RoboCasa interpreter."""

    if scene_split is not None and scene_split not in {"target", "pretrain"}:
        raise ValueError("scene_split must be 'target' or 'pretrain'")
    expected_names, expected_split = _expected_selection(selector)
    selected = _resolve_interpreter(interpreter)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    command = [
        str(selected),
        "-m",
        "robots.robocasa.tasks",
        "--resolve-json",
        str(selector),
    ]
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
            f"cannot resolve RoboCasa tasks with {selected}: {detail}"
        ) from exc

    try:
        payload = _extract_json(completed.stdout)
    except RuntimeError as exc:
        raise RuntimeError(f"cannot resolve RoboCasa tasks: {exc}") from exc
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("unsupported RoboCasa task resolver schema")
    reported_interpreter = Path(
        os.path.abspath(Path(str(payload.get("interpreter") or "")).expanduser())
    )
    if reported_interpreter != selected:
        raise RuntimeError(
            "RoboCasa task resolver used a different interpreter: "
            f"expected {selected}, got {reported_interpreter}"
        )
    if payload.get("selector") != str(selector):
        raise RuntimeError("RoboCasa task resolver returned a different selector")

    fixture = _contract_fixture()
    commit = str(payload.get("commit") or "")
    digest = str(payload.get("registry_digest") or "")
    if formal and commit != ROBOCASA_COMMIT:
        raise RuntimeError(
            "RoboCasa task registry commit is unavailable or mismatched: "
            f"expected {ROBOCASA_COMMIT}, got {commit or 'unavailable'}"
        )
    if formal and digest != fixture["registry_digest"]:
        raise RuntimeError(
            "RoboCasa task registry digest mismatch: "
            f"expected {fixture['registry_digest']}, got {digest or 'unavailable'}"
        )

    tasks = _response_tasks(payload)
    actual_names = tuple(task.name for task in tasks)
    actual_splits = {task.split for task in tasks}
    if actual_names != expected_names:
        raise RuntimeError(
            "RoboCasa task resolver changed the requested task order or denominator"
        )
    if actual_splits != {expected_split} or payload.get("split") != expected_split:
        raise RuntimeError(
            "RoboCasa task resolver returned an unexpected dataset split"
        )
    return _with_scene_split(tasks, scene_split)


def resolve_robocasa_tasks_from_contract(
    selector: str,
    *,
    scene_split: str | None = None,
) -> tuple[RoboCasaTask, ...]:
    """Resolve a dry-run selector from the packaged pinned registry snapshot."""

    fixture = _contract_fixture()
    tasks = fixture.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise RuntimeError("packaged RoboCasa task metadata is unavailable")
    expected_digest = str(fixture.get("task_metadata_digest") or "")
    actual_digest = _task_metadata_digest(tasks)
    if expected_digest != actual_digest:
        raise RuntimeError("packaged RoboCasa task metadata digest is invalid")
    provenance = fixture["provenance"]
    return resolve_selector_from_snapshot(
        selector,
        {
            "repository": provenance["repository"],
            "commit": provenance["commit"],
            "registry_digest": fixture["registry_digest"],
            "groups": fixture["groups"],
            "tasks": tasks,
        },
        scene_split=scene_split,
    )


def packaged_registry_snapshot() -> Mapping[str, Any]:
    """Return the pinned registry contract used by profile validation."""

    return copy.deepcopy(_contract_fixture())


def _load_descriptions(robocasa_root: Path) -> dict[str, str]:
    attributes_path = (
        robocasa_root.parent / "docs" / "composite_tasks" / "task_attributes.json"
    )
    try:
        payload = json.loads(attributes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot load pinned RoboCasa task descriptions; use the documented "
            f"editable clone installation ({attributes_path}): {exc}"
        ) from exc
    descriptions = {
        str(item["name"]): str(item["description"]).strip()
        for item in payload.get("tasks", ())
        if item.get("name") and item.get("description")
    }
    if len(descriptions) != 365:
        raise RuntimeError(
            "pinned RoboCasa task descriptions must contain exactly 365 tasks"
        )
    return descriptions


def _category_for_class(task_class: type[Any]) -> str:
    parts = task_class.__module__.split(".")
    if "composite" in parts:
        index = parts.index("composite")
        if index + 1 < len(parts):
            return parts[index + 1]
    leaf = parts[-1]
    return leaf.removeprefix("kitchen_")


def _humanize(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip() + "."


def _upstream_registry_snapshot(*, benchmark_only: bool = False) -> dict[str, Any]:
    import robocasa
    from robocasa.environments.kitchen.kitchen import (
        REGISTERED_KITCHEN_ENVS,
        Kitchen,
    )
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
        TASK_SET_REGISTRY,
    )

    robocasa_root = Path(robocasa.__file__).resolve().parent
    _, direct_commit = _distribution_metadata("robocasa")
    commit = direct_commit or _git_commit(robocasa_root) or ""
    descriptions = _load_descriptions(robocasa_root)

    groups = {
        name: {
            "split": "target" if name in _GROUP_NAMES[:3] else "pretrain",
            "tasks": list(TASK_SET_REGISTRY[name]),
        }
        for name in _GROUP_NAMES
    }
    digest = _registry_digest(groups)

    datasets = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}
    missing_classes = sorted(set(datasets).difference(REGISTERED_KITCHEN_ENVS))
    if missing_classes:
        raise RuntimeError(
            "RoboCasa dataset registry contains unregistered environments: "
            f"{', '.join(missing_classes[:8])}"
        )

    default_horizon = int(
        inspect.signature(Kitchen.__init__).parameters["horizon"].default
    )
    tasks: dict[str, dict[str, Any]] = {}
    for name, task_class in REGISTERED_KITCHEN_ENVS.items():
        # The upstream registry also includes helper/base environments. Keep
        # them available to explicit live queries, but the packaged benchmark
        # contract must contain only the 365 officially described tasks.
        if benchmark_only and name not in descriptions:
            continue
        if name not in datasets and task_class._check_success is Kitchen._check_success:
            continue
        module = task_class.__module__
        if name in ATOMIC_TASK_DATASETS:
            family = "atomic"
        elif name in COMPOSITE_TASK_DATASETS:
            family = "composite"
        elif ".atomic." in module:
            family = "atomic"
        elif ".composite." in module:
            family = "composite"
        else:
            continue
        tasks[name] = {
            "registry_id": f"{module}:{task_class.__qualname__}",
            "instruction": descriptions.get(name, _humanize(name)),
            "family": family,
            "category": _category_for_class(task_class),
            "upstream_horizon": int(
                datasets.get(name, {}).get("horizon", default_horizon)
            ),
        }

    if benchmark_only and set(tasks) != set(descriptions):
        missing = sorted(set(descriptions).difference(tasks))
        raise RuntimeError(f"RoboCasa benchmark tasks are not registered: {missing}")

    return {
        "repository": ROBOCASA_REPOSITORY,
        "commit": commit,
        "registry_digest": digest,
        "groups": groups,
        "tasks": tasks,
    }


def _resolution_payload(selector: str) -> dict[str, Any]:
    snapshot = _upstream_registry_snapshot()
    tasks = resolve_selector_from_snapshot(selector, snapshot)
    return {
        "schema_version": _SCHEMA_VERSION,
        "interpreter": os.path.abspath(sys.executable),
        "selector": selector,
        "repository": snapshot["repository"],
        "commit": snapshot["commit"],
        "registry_digest": snapshot["registry_digest"],
        "split": tasks[0].split,
        "tasks": [
            {
                key: value
                for key, value in asdict(task).items()
                if not key.startswith("registry_") or key == "registry_id"
            }
            for task in tasks
        ],
    }


def _contract_payload() -> dict[str, Any]:
    snapshot = _upstream_registry_snapshot(benchmark_only=True)
    return {
        "schema_version": _SCHEMA_VERSION,
        "provenance": {
            "repository": snapshot["repository"],
            "commit": snapshot["commit"],
            "source": "robocasa/utils/dataset_registry.py:TASK_SET_REGISTRY",
            "task_metadata_sources": [
                "robocasa/utils/dataset_registry.py",
                "robocasa/environments/kitchen",
                "docs/composite_tasks/task_attributes.json",
            ],
        },
        "registry_digest": snapshot["registry_digest"],
        "groups": snapshot["groups"],
        "task_metadata_digest": _task_metadata_digest(snapshot["tasks"]),
        "tasks": snapshot["tasks"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resolve-json", metavar="SELECTOR")
    mode.add_argument("--dump-contract-json", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        _contract_payload()
        if args.dump_contract_json
        else _resolution_payload(args.resolve_json)
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROBOCASA_REPOSITORY",
    "TASK_GROUP_FIXTURE",
    "RoboCasaTask",
    "task_selector_suggestions",
    "resolve_robocasa_tasks",
    "resolve_robocasa_tasks_from_contract",
    "resolve_selector_from_snapshot",
    "packaged_registry_snapshot",
]
