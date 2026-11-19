"""Simulator-owned launch configuration and presentation contract."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

FieldType = Literal["string", "integer", "number", "boolean", "array"]
SelectionMode = Literal["single", "multiple"]
PrerequisiteStatus = Literal["ready", "blocked", "warning", "not-applicable"]
_MISSING = object()


@dataclass(frozen=True, slots=True)
class LaunchOwnership:
    """A field becomes preset-owned for selected controller values."""

    field: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("ownership field must be non-empty")
        if not self.values or any(
            value in self.values[:index] for index, value in enumerate(self.values)
        ):
            raise ValueError("ownership values must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class LaunchSuggestionsWhen:
    """Suggestions selected by the current value of another field."""

    field: str
    values: tuple[Any, ...]
    suggestions: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("conditional suggestions field must be non-empty")
        if not self.values or any(
            value in self.values[:index] for index, value in enumerate(self.values)
        ):
            raise ValueError(
                "conditional suggestions values must be non-empty and unique"
            )
        if any(
            value in self.suggestions[:index]
            for index, value in enumerate(self.suggestions)
        ):
            raise ValueError("conditional suggestions must be unique")


BUILTIN_PLANNER_MODEL_SUGGESTIONS = (
    LaunchSuggestionsWhen("planner", ("api",), (
        "anthropic:claude-opus-4-7",
        "anthropic:claude-opus-4-8",
        "anthropic:claude-sonnet-4-5",
        "openai:gpt-5.5",
        "openai-chat:glm-5.2",
    )),
    LaunchSuggestionsWhen(
        "planner", ("claude_code",), ("claude-opus-4-7", "sonnet", "opus")
    ),
    LaunchSuggestionsWhen("planner", ("codex",), ()),
)


@dataclass(frozen=True, slots=True)
class LaunchField:
    name: str
    label: str
    type: FieldType
    group: str
    order: int
    default: Any = _MISSING
    required: bool = False
    description: str | None = None
    choices: tuple[Any, ...] = ()
    suggestions: tuple[Any, ...] = ()
    choice_labels: tuple[str, ...] = ()
    selection_mode: SelectionMode | None = None
    exclusive_choices: tuple[Any, ...] = ()
    allow_custom_value: bool = False
    step: float | None = None
    display_scale: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    visible_in: tuple[str, ...] = ()
    owned_when: LaunchOwnership | None = None
    visible_when: LaunchOwnership | None = None
    suggestions_when: tuple[LaunchSuggestionsWhen, ...] = ()

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    @property
    def requires_value(self) -> bool:
        """Whether the runtime contract always needs a non-empty value."""
        return self.required or (
            self.has_default
            and self.type != "boolean"
            and not _is_missing(self.default)
        )


@dataclass(frozen=True, slots=True)
class LaunchGroup:
    id: str
    label: str
    order: int
    collapsed: bool = False


@dataclass(frozen=True, slots=True)
class LaunchPrerequisite:
    id: str
    label: str
    status: PrerequisiteStatus
    message: str


class LaunchValidationError(ValueError):
    """One or more submitted fields violate the selected launch preset."""

    def __init__(self, errors: Mapping[str, str]):
        self.errors = dict(errors)
        fields = ", ".join(self.errors) or "configuration"
        super().__init__(f"invalid launch configuration: {fields}")


def validate_api_planner_model(
    mode: str, values: dict[str, Any]
) -> dict[str, Any]:
    """Require the provider:model form used by the API planner."""
    if mode == "hybrid" and values.get("planner") == "api":
        model = values.get("model")
        if (
            not isinstance(model, str)
            or not model.strip()
            or ":" not in model
            or not all(part.strip() for part in model.split(":", 1))
        ):
            raise LaunchValidationError({
                "model": "API planner model requires a provider prefix, for example openai:gpt-5.5."
            })
    return values


DeriveLaunch = Callable[[str, dict[str, Any]], dict[str, Any]]
CanonicalizeLaunch = Callable[[str, dict[str, Any]], dict[str, Any]]
CheckPrerequisites = Callable[[str, Mapping[str, Any]], tuple[LaunchPrerequisite, ...]]


@dataclass(frozen=True, slots=True)
class LaunchPreset:
    version: int
    default_mode: str
    modes: tuple[str, ...]
    groups: tuple[LaunchGroup, ...]
    fields: tuple[LaunchField, ...]
    canonicalize: CanonicalizeLaunch | None = None
    derive: DeriveLaunch | None = None
    check_prerequisites: CheckPrerequisites | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("launch preset version must be positive")
        if not self.modes or self.default_mode not in self.modes:
            raise ValueError("default launch mode must be supported")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("launch modes must be unique")
        group_ids = {group.id for group in self.groups}
        if len(group_ids) != len(self.groups):
            raise ValueError("launch groups must be unique")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("launch fields must be unique")
        fields = {field.name: field for field in self.fields}
        if any(field.group not in group_ids for field in self.fields):
            raise ValueError("launch fields must reference declared groups")
        if any(
            mode not in self.modes for field in self.fields for mode in field.visible_in
        ):
            raise ValueError("launch field visibility uses an unsupported mode")
        for field in self.fields:
            for relationship, condition in (
                ("ownership", field.owned_when),
                ("visibility", field.visible_when),
            ):
                if condition is None:
                    continue
                controller = fields.get(condition.field)
                if controller is None or controller.name == field.name:
                    raise ValueError(
                        f"launch field {field.name!r} has invalid {relationship} controller"
                    )
                if controller.type not in {
                    "string",
                    "integer",
                    "number",
                    "boolean",
                }:
                    raise ValueError(
                        f"launch field {field.name!r} has unsupported {relationship} "
                        "controller type"
                    )
                if any(
                    not _matches_type(controller.type, value)
                    for value in condition.values
                ):
                    raise ValueError(
                        f"launch field {field.name!r} has invalid {relationship} value"
                    )
                if controller.choices and any(
                    value not in controller.choices for value in condition.values
                ):
                    raise ValueError(
                        f"launch field {field.name!r} has invalid {relationship} value"
                    )
            suggestion_values: set[Any] = set()
            for rule in field.suggestions_when:
                controller = fields.get(rule.field)
                if controller is None or controller.name == field.name:
                    raise ValueError(
                        f"launch field {field.name!r} has invalid conditional suggestions controller"
                    )
                if controller.type not in {"string", "integer", "number", "boolean"}:
                    raise ValueError(
                        f"launch field {field.name!r} has unsupported conditional suggestions controller type"
                    )
                if any(not _matches_type(controller.type, value) for value in rule.values):
                    raise ValueError(
                        f"launch field {field.name!r} has invalid conditional suggestions value"
                    )
                if controller.choices and any(
                    value not in controller.choices for value in rule.values
                ):
                    raise ValueError(
                        f"launch field {field.name!r} has invalid conditional suggestions value"
                    )
                if any(value in suggestion_values for value in rule.values):
                    raise ValueError(
                        f"launch field {field.name!r} has ambiguous conditional suggestions"
                    )
                suggestion_values.update(rule.values)
                if any(not _matches_type(field.type, value) for value in rule.suggestions):
                    raise ValueError(
                        f"launch field {field.name!r} has invalid conditional suggestions"
                    )
            if not field.description or not field.description.strip():
                raise ValueError(f"launch field {field.name!r} requires help text")
            if field.choice_labels and len(field.choice_labels) != len(field.choices):
                raise ValueError(
                    f"launch field {field.name!r} choice labels must match choices"
                )
            if any(not label.strip() for label in field.choice_labels):
                raise ValueError(
                    f"launch field {field.name!r} choice labels must be non-empty"
                )
            if (field.suggestions or field.suggestions_when) and field.type not in {
                "string", "integer", "number"
            }:
                raise ValueError(f"launch field {field.name!r} cannot use suggestions")
            if field.selection_mode is not None and (
                field.selection_mode not in {"single", "multiple"}
                or field.type != "string"
            ):
                raise ValueError(
                    f"launch field {field.name!r} has invalid selection mode"
                )
            if field.selection_mode == "multiple" and field.choices:
                raise ValueError(
                    f"launch field {field.name!r} multiple selection must use suggestions"
                )
            if field.allow_custom_value and (
                field.selection_mode is None
                or not (
                    field.suggestions
                    or any(rule.suggestions for rule in field.suggestions_when)
                )
                or bool(field.choices)
            ):
                raise ValueError(
                    f"launch field {field.name!r} custom selection must use suggestions"
                )
            candidates = field.choices + field.suggestions
            if field.exclusive_choices and (
                field.selection_mode != "multiple"
                or any(value not in candidates for value in field.exclusive_choices)
            ):
                raise ValueError(
                    f"launch field {field.name!r} has invalid exclusive choices"
                )
            if field.step is not None and (
                field.type not in {"integer", "number"}
                or not isfinite(field.step)
                or field.step <= 0
            ):
                raise ValueError(
                    f"launch field {field.name!r} step must be a positive number"
                )
            if not isfinite(field.display_scale) or field.display_scale <= 0:
                raise ValueError(
                    f"launch field {field.name!r} display scale must be positive"
                )
            if field.display_scale != 1 and field.type not in {"integer", "number"}:
                raise ValueError(
                    f"launch field {field.name!r} cannot be display-scaled"
                )
        visibility_edges = {
            field.name: field.visible_when.field
            for field in self.fields
            if field.visible_when is not None
        }
        for start in visibility_edges:
            seen: set[str] = set()
            current = start
            while current in visibility_edges:
                if current in seen:
                    raise ValueError("launch field visibility cycle")
                seen.add(current)
                current = visibility_edges[current]

    def normalize(
        self,
        mode: str,
        submitted: Mapping[str, Any],
    ) -> dict[str, Any]:
        if mode not in self.modes:
            raise LaunchValidationError(
                {"execution_mode": f"Unsupported execution mode: {mode}."}
            )
        if not isinstance(submitted, Mapping):
            raise LaunchValidationError({"config": "Must be an object."})
        fields = {field.name: field for field in self.fields}
        canonical = self._canonicalized(mode, submitted, fields)
        visible = _visible_fields(fields, mode, canonical)
        errors: dict[str, str] = {}
        for name in submitted:
            if name not in fields:
                errors[str(name)] = "Unknown launch field."
            elif name not in visible:
                errors[name] = f"Not available in {_mode_label(mode)} mode."
        for name in canonical:
            if name not in visible:
                errors[str(name)] = f"Not available in {_mode_label(mode)} mode."
        effective: dict[str, Any] = {}
        for name, field in visible.items():
            if name in canonical:
                if field.requires_value and _is_missing(canonical[name]):
                    errors[name] = "Required."
                else:
                    effective[name] = deepcopy(canonical[name])
            elif field.has_default:
                effective[name] = deepcopy(field.default)
            elif field.requires_value:
                errors[name] = "Required."
        errors.update(_value_errors(visible, effective))
        if errors:
            raise LaunchValidationError(errors)
        if self.derive is not None:
            effective = self.derive(mode, dict(effective))
            derived_unknown = set(effective).difference(visible)
            if derived_unknown:
                raise ValueError(
                    "launch derivation returned unknown fields: "
                    + ", ".join(sorted(derived_unknown))
                )
            errors = _value_errors(visible, effective)
            if errors:
                raise LaunchValidationError(errors)
        return effective

    def _canonicalized(
        self,
        mode: str,
        submitted: Mapping[str, Any],
        fields: Mapping[str, LaunchField],
    ) -> dict[str, Any]:
        values = deepcopy(dict(submitted))
        if self.canonicalize is None:
            return values
        result = self.canonicalize(mode, values)
        if not isinstance(result, Mapping):
            raise TypeError("launch canonicalization must return a mapping")
        unknown = set(result).difference(fields)
        if unknown:
            raise ValueError(
                "launch canonicalization returned unknown fields: "
                + ", ".join(sorted(str(name) for name in unknown))
            )
        return deepcopy(dict(result))

    def preflight(
        self,
        mode: str,
        effective: Mapping[str, Any],
    ) -> tuple[LaunchPrerequisite, ...]:
        if mode not in self.modes:
            raise LaunchValidationError(
                {"execution_mode": f"Unsupported execution mode: {mode}."}
            )
        if self.check_prerequisites is None:
            return ()
        result = tuple(self.check_prerequisites(mode, effective))
        if len({item.id for item in result}) != len(result):
            raise ValueError("launch prerequisites must have unique IDs")
        return result

    def cli_preflight(
        self, args: argparse.Namespace
    ) -> tuple[LaunchPrerequisite, ...]:
        """Project parsed CLI values through the canonical hybrid preflight."""
        submitted = {
            field.name: getattr(args, field.name)
            for field in self.fields
            if hasattr(args, field.name) and getattr(args, field.name) is not None
        }
        return self.preflight("hybrid", submitted)

    def partial_preflight(
        self, mode: str, submitted: Mapping[str, Any]
    ) -> tuple[LaunchPrerequisite, ...]:
        if mode not in self.modes:
            raise LaunchValidationError(
                {"execution_mode": f"Unsupported execution mode: {mode}."}
            )
        if not isinstance(submitted, Mapping):
            raise LaunchValidationError({"config": "Must be an object."})
        fields = {field.name: field for field in self.fields}
        canonical = self._canonicalized(mode, submitted, fields)
        visible = _visible_fields(fields, mode, canonical)
        errors: dict[str, str] = {}
        for name in submitted:
            if name not in fields:
                errors[str(name)] = "Unknown launch field."
            elif name not in visible:
                errors[name] = f"Not available in {_mode_label(mode)} mode."
        for name in canonical:
            if name not in visible:
                errors[str(name)] = f"Not available in {_mode_label(mode)} mode."
        effective: dict[str, Any] = {}
        for name, field in visible.items():
            if name in canonical and not (
                field.requires_value and _is_missing(canonical[name])
            ):
                effective[name] = deepcopy(canonical[name])
            elif field.has_default:
                effective[name] = deepcopy(field.default)
        errors.update(_value_errors(visible, effective))
        if errors:
            raise LaunchValidationError(errors)
        return self.preflight(mode, effective)


def _visible_fields(
    fields: Mapping[str, LaunchField],
    mode: str,
    submitted: Mapping[str, Any],
) -> dict[str, LaunchField]:
    values = {
        name: deepcopy(field.default)
        for name, field in fields.items()
        if field.has_default
    }
    values.update(submitted)

    def visible(name: str) -> bool:
        field = fields[name]
        if field.visible_in and mode not in field.visible_in:
            return False
        condition = field.visible_when
        return condition is None or (
            visible(condition.field)
            and values.get(condition.field) in condition.values
        )

    return {name: field for name, field in fields.items() if visible(name)}


def _value_errors(
    fields: Mapping[str, LaunchField],
    values: Mapping[str, Any],
) -> dict[str, str]:
    errors: dict[str, str] = {}
    for name, value in values.items():
        field = fields[name]
        if not _matches_type(field.type, value):
            errors[name] = f"Must be {_type_phrase(field.type)}."
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if field.minimum is not None and value < field.minimum:
                errors[name] = f"Must be at least {_number(field.minimum)}."
                continue
            if field.maximum is not None and value > field.maximum:
                errors[name] = f"Must be at most {_number(field.maximum)}."
                continue
        if field.choices and value not in field.choices:
            errors[name] = "Must use an allowed value."
    return errors


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _matches_type(field_type: FieldType, value: Any) -> bool:
    if field_type == "string":
        return isinstance(value, str)
    if field_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        )
    if field_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, list)


def _type_phrase(field_type: FieldType) -> str:
    return {
        "string": "a string",
        "integer": "an integer",
        "number": "a finite number",
        "boolean": "boolean",
        "array": "an array",
    }[field_type]


def _number(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _mode_label(mode: str) -> str:
    return "VLA-only" if mode == "vla-only" else mode.replace("-", " ").title()


__all__ = [
    "LaunchField",
    "LaunchGroup",
    "LaunchPrerequisite",
    "LaunchPreset",
    "LaunchValidationError",
]
