"""Simulator discovery and launch-schema projection."""

from __future__ import annotations

import argparse
import pkgutil
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

import robots
from rpent.envs.base import get_env_spec
from rpent.envs.launch import LaunchPreset
from rpent.ui.schemas import LaunchSchemaView, SimulatorSummaryView

router = APIRouter(prefix="/api/simulators", tags=["simulators"])


@router.get("", response_model=list[SimulatorSummaryView])
def list_simulators(request: Request):
    return [
        {
            "simulator_id": simulator_id,
            "capabilities": dict(item.get("capabilities", {})),
        }
        for simulator_id, item in sorted(request.app.state.simulator_catalog.items())
    ]


@router.get(
    "/{simulator_id}/launch-schema",
    response_model=LaunchSchemaView,
    response_model_exclude_none=True,
)
def launch_schema(simulator_id: str, request: Request):
    item = request.app.state.simulator_catalog.get(simulator_id)
    if item is None:
        raise HTTPException(status_code=404, detail="simulator not found")
    return item["launch_schema"]


def build_simulator_catalog(
    simulator_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected = tuple(simulator_ids or _discover_simulators())
    catalog: dict[str, dict[str, Any]] = {}
    for requested in selected:
        spec = get_env_spec(requested)
        if spec.launch_preset is None:
            raise ValueError(f"simulator {spec.name!r} has no launch preset")
        preset = spec.launch_preset()
        _check_runtime_fields(spec, preset)
        catalog[spec.name] = {
            "simulator_id": spec.name,
            "capabilities": {},
            "launch_schema": _preset_schema(preset),
        }
    return catalog


def _discover_simulators() -> tuple[str, ...]:
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(robots.__path__)
            if not module.name.startswith("_")
        )
    )


def _preset_schema(preset: LaunchPreset) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for field in sorted(preset.fields, key=lambda item: (item.group, item.order)):
        item: dict[str, Any] = {
            "title": field.label,
            "type": field.type,
            "group": field.group,
            "order": field.order,
            "visible_in": field.visible_in,
        }
        if field.description:
            item["description"] = field.description
        if field.choices:
            item["enum"] = list(field.choices)
        if field.suggestions:
            item["suggestions"] = list(field.suggestions)
        if field.suggestions_when:
            item["suggestions_when"] = [
                {
                    "field": rule.field,
                    "values": list(rule.values),
                    "suggestions": list(rule.suggestions),
                }
                for rule in field.suggestions_when
            ]
        if field.choice_labels:
            item["choice_labels"] = list(field.choice_labels)
        if field.selection_mode:
            item["selection_mode"] = field.selection_mode
            item["allow_custom_value"] = field.allow_custom_value
        if field.exclusive_choices:
            item["exclusive_choices"] = list(field.exclusive_choices)
        if field.step is not None:
            item["step"] = field.step
        if field.display_scale != 1:
            item["display_scale"] = field.display_scale
        if field.minimum is not None:
            item["minimum"] = field.minimum
        if field.maximum is not None:
            item["maximum"] = field.maximum
        if field.has_default:
            item["default"] = field.default
        if field.owned_when is not None:
            item["owned_when"] = {
                "field": field.owned_when.field,
                "values": list(field.owned_when.values),
            }
        if field.visible_when is not None:
            item["visible_when"] = {
                "field": field.visible_when.field,
                "values": list(field.visible_when.values),
            }
        if field.type == "array":
            item["items"] = {"type": "string"}
        if field.requires_value:
            required.append(field.name)
        properties[field.name] = item
    return {
        "version": preset.version,
        "default_mode": preset.default_mode,
        "modes": preset.modes,
        "groups": tuple(
            {
                "id": group.id,
                "label": group.label,
                "order": group.order,
                "collapsed": group.collapsed,
            }
            for group in sorted(preset.groups, key=lambda item: item.order)
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _check_runtime_fields(spec: Any, preset: LaunchPreset) -> None:
    from rpent.cli.main import _build_argparser

    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    destinations = {action.dest for action in parser._actions}
    assembler_owned = {
        "reuse_inference",
        "reuse_policy_service",
        "vla_max_chunks",
    }
    missing = {
        field.name for field in preset.fields
        if field.name not in destinations and field.name not in assembler_owned
    }
    if missing:
        raise ValueError(
            f"simulator {spec.name!r} launch fields have no runtime destination: "
            + ", ".join(sorted(missing))
        )


def _parser_schema(parser: argparse.ArgumentParser) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        field: dict[str, Any] = {"title": action.dest.replace("_", " ").title()}
        if action.help and action.help is not argparse.SUPPRESS:
            field["description"] = action.help
        value_type = action.type
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            field["type"] = "boolean"
        elif isinstance(action, argparse._AppendAction):
            items = _type_schema(value_type)
            if action.choices is not None:
                items["enum"] = list(action.choices)
            field.update({"type": "array", "items": items})
        else:
            field.update(_type_schema(value_type))
            if action.choices is not None:
                field["enum"] = list(action.choices)
        if action.default is not argparse.SUPPRESS and action.default is not None:
            field["default"] = action.default
        if action.required:
            required.append(action.dest)
        properties[action.dest] = field
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _type_schema(value_type: Any) -> dict[str, Any]:
    if getattr(value_type, "__name__", None) == "_positive_integer":
        return {"type": "integer", "minimum": 1}
    if getattr(value_type, "__name__", None) == "_nonnegative_integer":
        return {"type": "integer", "minimum": 0}
    if value_type is int:
        return {"type": "integer"}
    if value_type is float:
        return {"type": "number"}
    return {"type": "string"}


__all__ = ["build_simulator_catalog", "router"]
