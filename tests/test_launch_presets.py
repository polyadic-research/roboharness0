from __future__ import annotations

from dataclasses import fields, replace

import pytest

from rpent.envs.launch import (
    LaunchField,
    LaunchGroup,
    LaunchOwnership,
    LaunchPrerequisite,
    LaunchPreset,
    LaunchSuggestionsWhen,
    LaunchValidationError,
)


def test_launch_field_exposes_presentation_metadata() -> None:
    names = {field.name for field in fields(LaunchField)}

    assert {
        "suggestions",
        "choice_labels",
        "step",
        "display_scale",
        "selection_mode",
        "exclusive_choices",
        "allow_custom_value",
        "owned_when",
        "visible_when",
        "suggestions_when",
    } <= names


def test_conditional_suggestions_require_valid_unambiguous_metadata() -> None:
    group = LaunchGroup("planning", "Planning", 10)
    controller = LaunchField(
        "planner", "Planner", "string", group.id, 1,
        choices=("api", "claude_code", "codex"),
        description="Planner backend.",
    )
    model = LaunchField(
        "model", "Model", "string", group.id, 2,
        description="Planner model.",
        selection_mode="single",
        allow_custom_value=True,
        suggestions_when=(
            LaunchSuggestionsWhen("planner", ("api",), ("openai:gpt-5.5",)),
            LaunchSuggestionsWhen("planner", ("codex",), ()),
        ),
    )
    preset = LaunchPreset(
        version=1, default_mode="hybrid", modes=("hybrid",), groups=(group,),
        fields=(controller, model),
    )
    assert preset.fields[1].suggestions_when == model.suggestions_when

    invalid_rules = (
        (LaunchSuggestionsWhen("missing", ("api",), ("x",)),),
        (LaunchSuggestionsWhen("model", ("api",), ("x",)),),
        (LaunchSuggestionsWhen("planner", ("other",), ("x",)),),
        (
            LaunchSuggestionsWhen("planner", ("api",), ("x",)),
            LaunchSuggestionsWhen("planner", ("api", "codex"), ("y",)),
        ),
    )
    for rules in invalid_rules:
        with pytest.raises(ValueError, match="conditional suggestions"):
            LaunchPreset(
                version=1, default_mode="hybrid", modes=("hybrid",),
                groups=(group,), fields=(controller, replace(model, suggestions_when=rules)),
            )


@pytest.mark.parametrize("values", [(), ("api", "api")])
def test_conditional_suggestions_reject_empty_or_duplicate_controller_values(values) -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        LaunchSuggestionsWhen("planner", values, ("model",))


def test_conditional_suggestions_reject_empty_controller_and_duplicate_suggestions() -> None:
    with pytest.raises(ValueError, match="field must be non-empty"):
        LaunchSuggestionsWhen("", ("api",), ("model",))
    with pytest.raises(ValueError, match="suggestions must be unique"):
        LaunchSuggestionsWhen("planner", ("api",), ("model", "model"))


def test_visibility_requires_a_valid_controller_and_value() -> None:
    group = LaunchGroup("planning", "Planning", 10)
    controller = LaunchField(
        "planner", "Planner", "string", group.id, 1,
        choices=("api", "claude_code", "codex"),
        description="Planner backend.",
    )
    model = LaunchField(
        "model", "Model", "string", group.id, 2,
        description="Planner model.",
        visible_when=LaunchOwnership("planner", ("api",)),
    )

    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(group,),
        fields=(controller, model),
    )
    assert preset.fields[1].visible_when == LaunchOwnership("planner", ("api",))

    for invalid in (
        replace(model, visible_when=LaunchOwnership("missing", ("api",))),
        replace(model, visible_when=LaunchOwnership("model", ("api",))),
        replace(model, visible_when=LaunchOwnership("planner", ("other",))),
    ):
        with pytest.raises(ValueError, match="visibility"):
            LaunchPreset(
                version=1,
                default_mode="hybrid",
                modes=("hybrid",),
                groups=(group,),
                fields=(controller, invalid),
            )


@pytest.mark.parametrize("values", [(), ("api", "api")])
def test_visibility_rejects_empty_or_duplicate_values(values) -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        LaunchOwnership("planner", values)


def test_visibility_rejects_an_empty_controller_name() -> None:
    with pytest.raises(ValueError, match="field must be non-empty"):
        LaunchOwnership("", ("api",))


def test_visibility_rejects_incompatible_controller_types_and_cycles() -> None:
    group = LaunchGroup("planning", "Planning", 10)
    with pytest.raises(ValueError, match="visibility controller type"):
        LaunchPreset(
            version=1,
            default_mode="hybrid",
            modes=("hybrid",),
            groups=(group,),
            fields=(
                LaunchField(
                    "controllers", "Controllers", "array", group.id, 1,
                    description="Controller list.",
                ),
                LaunchField(
                    "model", "Model", "string", group.id, 2,
                    description="Planner model.",
                    visible_when=LaunchOwnership("controllers", ("api",)),
                ),
            ),
        )

    with pytest.raises(ValueError, match="visibility cycle"):
        LaunchPreset(
            version=1,
            default_mode="hybrid",
            modes=("hybrid",),
            groups=(group,),
            fields=(
                LaunchField(
                    "first", "First", "string", group.id, 1,
                    description="First selector.",
                    visible_when=LaunchOwnership("second", ("yes",)),
                ),
                LaunchField(
                    "second", "Second", "string", group.id, 2,
                    description="Second selector.",
                    visible_when=LaunchOwnership("first", ("yes",)),
                ),
            ),
        )


def test_normalize_enforces_conditional_visibility_without_ownership() -> None:
    group = LaunchGroup("planning", "Planning", 10)
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(group,),
        fields=(
            LaunchField(
                "planner", "Planner", "string", group.id, 1,
                default="api", choices=("api", "claude_code"),
                description="Planner backend.",
            ),
            LaunchField(
                "budget", "Budget", "number", group.id, 2,
                default=1.0, description="Claude budget.",
                visible_when=LaunchOwnership("planner", ("claude_code",)),
            ),
        ),
    )

    assert preset.normalize("hybrid", {"planner": "api"}) == {"planner": "api"}
    with pytest.raises(LaunchValidationError, match="budget"):
        preset.normalize("hybrid", {"planner": "api", "budget": 2.0})
    assert preset.normalize(
        "hybrid", {"planner": "claude_code", "budget": 2.0}
    ) == {"planner": "claude_code", "budget": 2.0}


def test_conditional_ownership_requires_a_valid_controller_and_value() -> None:
    ownership = LaunchOwnership("profile", ("official",))
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "profile",
                "Profile",
                "string",
                "execution",
                1,
                choices=("development", "official"),
                description="Profile selector.",
            ),
            LaunchField(
                "task",
                "Task",
                "string",
                "execution",
                2,
                required=True,
                description="Task selector.",
                owned_when=ownership,
            ),
        ),
    )

    assert preset.fields[1].owned_when == ownership
    with pytest.raises(ValueError, match="ownership controller"):
        LaunchPreset(
            version=1,
            default_mode="hybrid",
            modes=("hybrid",),
            groups=(LaunchGroup("execution", "Execution", 10),),
            fields=(
                replace(
                    preset.fields[1], owned_when=LaunchOwnership("missing", ("x",))
                ),
            ),
        )


def test_conditional_ownership_accepts_boolean_and_integer_controllers() -> None:
    group = LaunchGroup("execution", "Execution", 10)
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(group,),
        fields=(
            LaunchField(
                "enabled",
                "Enabled",
                "boolean",
                group.id,
                1,
                description="Boolean controller.",
            ),
            LaunchField(
                "level",
                "Level",
                "integer",
                group.id,
                2,
                description="Integer controller.",
            ),
            LaunchField(
                "first",
                "First",
                "string",
                group.id,
                3,
                description="Boolean-owned field.",
                owned_when=LaunchOwnership("enabled", (True,)),
            ),
            LaunchField(
                "second",
                "Second",
                "string",
                group.id,
                4,
                description="Integer-owned field.",
                owned_when=LaunchOwnership("level", (7,)),
            ),
        ),
    )

    assert preset.fields[2].owned_when == LaunchOwnership("enabled", (True,))
    assert preset.fields[3].owned_when == LaunchOwnership("level", (7,))


@pytest.mark.parametrize(
    ("controller_type", "value", "message"),
    [
        ("integer", True, "ownership value"),
        ("boolean", 1, "ownership value"),
        ("array", ["official"], "ownership controller type"),
    ],
)
def test_conditional_ownership_rejects_type_mismatch_and_array_controller(
    controller_type, value, message
) -> None:
    group = LaunchGroup("execution", "Execution", 10)

    with pytest.raises(ValueError, match=message):
        LaunchPreset(
            version=1,
            default_mode="hybrid",
            modes=("hybrid",),
            groups=(group,),
            fields=(
                LaunchField(
                    "controller",
                    "Controller",
                    controller_type,
                    group.id,
                    1,
                    description="Ownership controller.",
                ),
                LaunchField(
                    "task",
                    "Task",
                    "string",
                    group.id,
                    2,
                    description="Owned field.",
                    owned_when=LaunchOwnership("controller", (value,)),
                ),
            ),
        )


def _preset() -> LaunchPreset:
    return LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid", "vla-only"),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "episodes",
                "Episodes",
                "integer",
                "execution",
                10,
                default=1,
                description="Number of episodes per task.",
                minimum=1,
            ),
            LaunchField(
                "max_turns",
                "Planner turn budget",
                "integer",
                "execution",
                20,
                default=100,
                description="Maximum planner turns.",
                visible_in=("hybrid",),
            ),
            LaunchField(
                "max_chunks",
                "VLA chunk budget",
                "integer",
                "execution",
                30,
                default=48,
                description="Maximum policy action chunks.",
                visible_in=("vla-only",),
            ),
        ),
        check_prerequisites=lambda mode, values: (
            LaunchPrerequisite(
                id="planner-auth",
                label="Planner authorization",
                status="ready" if mode == "hybrid" else "not-applicable",
                message=f"episodes={values['episodes']}",
            ),
        ),
    )


def test_preset_applies_only_visible_defaults() -> None:
    assert _preset().normalize("hybrid", {"episodes": 2}) == {
        "episodes": 2,
        "max_turns": 100,
    }


def test_preset_canonicalizes_before_required_fields_and_defaults() -> None:
    submitted: dict[str, object] = {}

    def canonicalize(mode: str, values: dict[str, object]) -> dict[str, object]:
        assert mode == "hybrid"
        values["task"] = "profile-owned-task"
        values["episodes"] = 50
        return values

    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "task",
                "Task",
                "string",
                "execution",
                10,
                required=True,
                description="Required task.",
            ),
            LaunchField(
                "episodes",
                "Episodes",
                "integer",
                "execution",
                20,
                default=1,
                description="Episodes per task.",
            ),
        ),
        canonicalize=canonicalize,
    )

    assert preset.normalize("hybrid", submitted) == {
        "task": "profile-owned-task",
        "episodes": 50,
    }
    assert submitted == {}


def test_preset_rejects_unknown_fields_returned_by_canonicalizer() -> None:
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "task",
                "Task",
                "string",
                "execution",
                10,
                required=True,
                description="Required task.",
            ),
        ),
        canonicalize=lambda _mode, _values: {
            "task": "profile-owned-task",
            "undeclared": True,
        },
    )

    with pytest.raises(ValueError, match="canonicalization returned unknown fields"):
        preset.normalize("hybrid", {})


@pytest.mark.parametrize("operation", ["normalize", "partial_preflight"])
def test_preset_rejects_hidden_fields_returned_by_canonicalizer(
    operation: str,
) -> None:
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid", "vla-only"),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "max_chunks",
                "VLA chunks",
                "integer",
                "execution",
                10,
                default=48,
                description="VLA action chunk budget.",
                visible_in=("vla-only",),
            ),
        ),
        canonicalize=lambda _mode, values: {**values, "max_chunks": 12},
    )

    with pytest.raises(LaunchValidationError) as error:
        getattr(preset, operation)("hybrid", {})

    assert error.value.errors == {"max_chunks": "Not available in Hybrid mode."}


def test_preset_treats_non_empty_scalar_defaults_as_required_runtime_values() -> None:
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "task",
                "Task",
                "string",
                "execution",
                10,
                required=True,
                description="Required task.",
            ),
            LaunchField(
                "episodes",
                "Episodes",
                "integer",
                "execution",
                20,
                default=1,
                description="Episodes per task.",
            ),
            LaunchField(
                "model",
                "Model",
                "string",
                "execution",
                30,
                description="Optional model override.",
            ),
            LaunchField(
                "enabled",
                "Enabled",
                "boolean",
                "execution",
                40,
                default=False,
                description="Optional feature switch.",
            ),
            LaunchField(
                "cameras",
                "Cameras",
                "array",
                "execution",
                50,
                default=[],
                description="Optional selected cameras.",
            ),
        ),
    )

    assert [field.name for field in preset.fields if field.requires_value] == [
        "task",
        "episodes",
    ]
    assert preset.normalize("hybrid", {"task": "pick"})["episodes"] == 1
    with pytest.raises(LaunchValidationError) as error:
        preset.normalize("hybrid", {"task": "", "episodes": ""})
    assert error.value.errors == {"task": "Required.", "episodes": "Required."}


def test_partial_preflight_allows_missing_required_fields_and_validates_supplied_values() -> (
    None
):
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "task",
                "Task",
                "string",
                "execution",
                10,
                required=True,
                description="Required task.",
                choices=("pick",),
            ),
            LaunchField(
                "device",
                "Device",
                "string",
                "execution",
                20,
                default="cpu",
                description="Runtime device.",
            ),
        ),
        check_prerequisites=lambda mode, values: (
            LaunchPrerequisite("runtime", "Runtime", "ready", str(values["device"])),
        ),
    )

    assert preset.partial_preflight("hybrid", {})[0].message == "cpu"
    assert preset.partial_preflight("hybrid", {"task": ""})[0].message == "cpu"
    with pytest.raises(LaunchValidationError, match="device"):
        preset.partial_preflight("hybrid", {"device": 1})


def test_presentation_metadata_does_not_change_normalization() -> None:
    preset = LaunchPreset(
        version=1,
        default_mode="hybrid",
        modes=("hybrid",),
        groups=(LaunchGroup("execution", "Execution", 10),),
        fields=(
            LaunchField(
                "quality",
                "Video quality",
                "integer",
                "execution",
                10,
                default=512,
                description="Rendered simulator frame size.",
                choices=(512, 256),
                choice_labels=("HD", "Standard"),
                step=256,
            ),
            LaunchField(
                "task",
                "Task selector",
                "string",
                "execution",
                20,
                description="Task group, index, or canonical name.",
                suggestions=("all", "0"),
            ),
            LaunchField(
                "timeout_s",
                "Timeout (minutes)",
                "integer",
                "execution",
                30,
                default=3600,
                description="Whole-run wall-clock limit displayed in minutes.",
                display_scale=60,
            ),
        ),
    )

    assert preset.normalize("hybrid", {"task": "custom-task"}) == {
        "quality": 512,
        "task": "custom-task",
        "timeout_s": 3600,
    }
    with pytest.raises(LaunchValidationError):
        preset.normalize("hybrid", {"quality": 128, "task": "0"})


@pytest.mark.parametrize(
    "field",
    [
        LaunchField(
            "x",
            "X",
            "string",
            "execution",
            1,
            description="X.",
            choices=("a",),
            choice_labels=("A", "B"),
        ),
        LaunchField(
            "x",
            "X",
            "string",
            "execution",
            1,
            description="X.",
            choice_labels=("A",),
        ),
        LaunchField(
            "x",
            "X",
            "integer",
            "execution",
            1,
            description="X.",
            step=0,
        ),
        LaunchField(
            "x",
            "X",
            "integer",
            "execution",
            1,
            description="X.",
            display_scale=0,
        ),
        LaunchField(
            "x",
            "X",
            "boolean",
            "execution",
            1,
            description="X.",
            suggestions=(True,),
        ),
        LaunchField(
            "x",
            "X",
            "integer",
            "execution",
            1,
            description="X.",
            selection_mode="multiple",
        ),
        LaunchField(
            "x",
            "X",
            "string",
            "execution",
            1,
            description="X.",
            suggestions=("a",),
            selection_mode="multiple",
            exclusive_choices=("missing",),
        ),
        LaunchField(
            "x",
            "X",
            "string",
            "execution",
            1,
            description="X.",
            choices=("a",),
            selection_mode="multiple",
        ),
        LaunchField("x", "X", "string", "execution", 1, description=" "),
    ],
)
def test_preset_rejects_invalid_presentation_metadata(field: LaunchField) -> None:
    with pytest.raises(ValueError):
        LaunchPreset(
            version=1,
            default_mode="hybrid",
            modes=("hybrid",),
            groups=(LaunchGroup("execution", "Execution", 10),),
            fields=(field,),
        )


def test_preset_rejects_hidden_mode_field() -> None:
    with pytest.raises(LaunchValidationError) as error:
        _preset().normalize("hybrid", {"max_chunks": 9})
    assert error.value.errors == {"max_chunks": "Not available in Hybrid mode."}


def test_preset_reports_field_addressable_type_and_range_errors() -> None:
    with pytest.raises(LaunchValidationError) as error:
        _preset().normalize("hybrid", {"episodes": 0, "max_turns": "many"})
    assert error.value.errors == {
        "episodes": "Must be at least 1.",
        "max_turns": "Must be an integer.",
    }


def test_preset_preflight_is_side_effect_free_projection() -> None:
    effective = _preset().normalize("hybrid", {})
    assert _preset().preflight("hybrid", effective) == (
        LaunchPrerequisite(
            id="planner-auth",
            label="Planner authorization",
            status="ready",
            message="episodes=1",
        ),
    )
