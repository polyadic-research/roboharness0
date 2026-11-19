"""Pi0.5 review decisions must preserve the effective instruction unless opted in."""

import pytest

from robots.robocasa.toolkit import RoboCasaToolkit
from tests.test_robocasa_toolkit import _BudgetController, _initial_pick_observation

POLICY_TOOLS = ("execute_vla", "continue_vla", "smolvla_pick")


def _toolkit():
    controller = _BudgetController()
    toolkit = RoboCasaToolkit(
        controller,
        initial_observation=_initial_pick_observation(),
        instruction="Open the stand mixer head.",
        forward_step_guidance=True,
        actions_per_policy_request=5,
    )
    return toolkit, controller


def _call(toolkit, tool, **arguments):
    result = toolkit.execute_tool(tool, arguments).result
    assert "error" not in result, result
    return result


@pytest.mark.parametrize("tool", POLICY_TOOLS)
def test_review_can_keep_original_instruction_without_supplying_guidance(tool):
    toolkit, controller = _toolkit()
    result = _call(toolkit, tool, intervene=False)
    assert controller.instructions
    assert set(controller.instructions) == {"Open the stand mixer head."}
    assert result["instruction"] == "Open the stand mixer head."
    assert result["intervened"] is False


@pytest.mark.parametrize("tool", POLICY_TOOLS)
def test_omitting_intervention_defaults_to_preserving_instruction(tool):
    toolkit, controller = _toolkit()
    _call(toolkit, tool)
    assert set(controller.instructions) == {"Open the stand mixer head."}


@pytest.mark.parametrize("tool", POLICY_TOOLS)
def test_review_without_intervention_cannot_forward_audit_text(tool):
    toolkit, controller = _toolkit()
    _call(toolkit, tool, intervene=False, instruction="Approach the mixer.")
    assert set(controller.instructions) == {"Open the stand mixer head."}


@pytest.mark.parametrize("update_tool", POLICY_TOOLS)
def test_explicit_intervention_persists_across_tools_and_later_reviews(update_tool):
    toolkit, controller = _toolkit()
    guidance = "Lift the cream stand mixer head."
    first = _call(toolkit, update_tool, intervene=True, instruction=guidance)
    assert first["intervened"] is True
    assert first["instruction"] == guidance
    assert set(controller.instructions) == {guidance}
    for tool in POLICY_TOOLS:
        result = _call(toolkit, tool, intervene=False)
        assert result["instruction"] == guidance
        assert result["intervened"] is False
    assert set(controller.instructions) == {guidance}
    assert toolkit.observe()["task_language"] == "Open the stand mixer head."
    _call(toolkit, "execute_vla", intervene=True, instruction="Open the stand mixer head.")
    assert controller.instructions[-1] == "Open the stand mixer head."


@pytest.mark.parametrize("tool", POLICY_TOOLS)
@pytest.mark.parametrize("guidance", (None, "", "   ", "x" * 1001))
def test_invalid_intervention_cannot_execute_or_replace_current_instruction(tool, guidance):
    toolkit, controller = _toolkit()
    retained = "Lift the cream stand mixer head."
    _call(toolkit, "execute_vla", intervene=True, instruction=retained)
    previous_calls = len(controller.instructions)
    result = toolkit.execute_tool(tool, {"intervene": True, "instruction": guidance}).result
    assert "error" in result
    assert len(controller.instructions) == previous_calls
    _call(toolkit, "execute_vla", intervene=False)
    assert controller.instructions[-1] == retained


@pytest.mark.parametrize("tool", POLICY_TOOLS)
@pytest.mark.parametrize("value", ("false", 0, 1, None))
def test_intervention_requires_an_actual_boolean(tool, value):
    toolkit, controller = _toolkit()
    result = toolkit.execute_tool(tool, {"intervene": value}).result
    assert "error" in result
    assert controller.instructions == []


def test_closed_empty_recovery_without_intervention_keeps_current_instruction():
    toolkit, controller = _toolkit()
    retained = "Lift the cream stand mixer head."
    _call(toolkit, "execute_vla", intervene=True, instruction=retained)
    toolkit._consecutive_closed_empty = 1
    _call(toolkit, "smolvla_pick", intervene=False, instruction="A different recovery proposal.")
    assert set(controller.instructions) == {retained}


def test_pi05_tool_schemas_offer_optional_intervention_at_every_review():
    toolkit, _ = _toolkit()
    specs = {item["name"]: item for item in toolkit.get_tools_spec()}
    for name in POLICY_TOOLS:
        schema = specs[name]["input_schema"]
        assert "instruction" not in schema.get("required", [])
        assert schema["properties"]["intervene"]["type"] == "boolean"
        assert schema["properties"]["intervene"]["default"] is False
        assert "unchanged" in schema["properties"]["intervene"]["description"]


@pytest.mark.parametrize("max_actions", (0, -1, False, "5"))
def test_invalid_execution_budget_cannot_apply_a_proposed_intervention(max_actions):
    toolkit, controller = _toolkit()
    result = toolkit.execute_tool("execute_vla", {
        "intervene": True, "instruction": "Lift the mixer head.",
        "max_actions": max_actions,
    }).result
    assert "error" in result
    assert controller.instructions == []
    _call(toolkit, "execute_vla", intervene=False)
    assert controller.instructions[-1] == "Open the stand mixer head."


def test_blocked_grasp_cannot_apply_a_proposed_intervention():
    toolkit, controller = _toolkit()
    toolkit._consecutive_closed_empty = 2
    result = _call(toolkit, "smolvla_pick", intervene=True, instruction="Lift the mixer head.")
    assert result["executed_actions"] == 0
    assert result["intervened"] is False
    assert controller.instructions == []
    _call(toolkit, "execute_vla", intervene=False)
    assert controller.instructions[-1] == "Open the stand mixer head."
