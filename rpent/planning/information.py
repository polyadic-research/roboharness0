"""Public, text-only scene evidence exchanged with a nested planner."""

from copy import deepcopy
from typing import Any

from rpent.harness.lifecycle import PLANNING_INFORMATION_SCOPE

OBJECT_STATES_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 64,
    "description": (
        "Complete snapshot of task targets, source/destination fixtures and relevant obstacles. "
        "Include identity, location, task-relevant states (door/access/contents/grasp/support), "
        "relations and actual evidence. Explicitly write unknown for unconfirmed values; "
        "previous model verdicts are not fresh evidence."
    ),
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "identity": {"type": "string"},
            "location": {"type": "string"},
            "states": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {"type": "string"},
            },
            "relations": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "observation_step": {"type": "integer", "minimum": 0},
            "uncertainty": {"type": "string"},
        },
        "required": [
            "id",
            "identity",
            "location",
            "states",
            "relations",
            "evidence",
            "observation_step",
            "uncertainty",
        ],
    },
}

INFORMATION_REQUEST_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "maxItems": 8,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "object_id": {"type": "string"},
            "fields": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
                "description": "Top-level fields or states keys, such as location, door or contents.",
            },
            "question": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["object_id", "fields", "question", "reason"],
    },
}

PLANNING_INFORMATION_GUIDANCE = (
    "When the exposed create_plan/replan schemas require object_states, the nested planner "
    "receives text, not camera attachments. Before create_plan "
    "or replan, supply object_states for every task target, source/destination fixture, and "
    "relevant obstacle. Give identity, location, relevant current states, relations, evidence "
    "observation steps and uncertainty. Naming a refrigerator without its door/access state "
    "is incomplete. Include container contents, holding/support and destination accessibility "
    "when relevant, without reading hidden simulator state. Separate observed facts from "
    "inferences and mark unconfirmed values unknown. Refresh affected states after actions; "
    "do not promote a previous LLM verdict into a measurement. "
    + PLANNING_INFORMATION_SCOPE
    + " When information_required is "
    "returned, answer every requested field using existing evidence or read-only observation, "
    "then call provide_planner_information with the exact request_id, the complete updated "
    "object_states and observations. This resumes the original goal and planning context. "
    "It neither registers a plan nor authorizes physical motion. If obtaining a new view "
    "requires motion, answer unknown; a grounded observation action would need its own "
    "registered plan and normal action/verification flow. Information questions have a "
    "separate bounded budget; exhaustion never makes unknown facts known or satisfies "
    "an outcome. On information_budget_exhausted report the missing evidence and leave "
    "the run INCOMPLETE without further planning calls or physical actions."
)


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def validate_object_states(value: Any) -> list[dict[str, Any]]:
    """Validate evidence shape, without inventing or verifying physical facts."""
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("object_states must contain 1-64 task-relevant objects")
    ids = set()
    for obj in value:
        if not isinstance(obj, dict):
            raise ValueError("object_states entries must be objects")
        for field in ("id", "identity", "location", "uncertainty"):
            _text(obj.get(field), field)
        if obj["id"] in ids:
            raise ValueError("object_states IDs must be unique")
        ids.add(obj["id"])
        states = obj.get("states")
        if not isinstance(states, dict) or not states:
            raise ValueError(
                "object states must name relevant states or explicitly unknown"
            )
        for key, state in states.items():
            _text(key, "state field")
            _text(state, f"states.{key}")
        for field in ("relations", "evidence"):
            items = obj.get(field)
            if not isinstance(items, list):
                raise ValueError(f"{field} must be an array")
            for item in items:
                _text(item, field)
        if not obj["evidence"] and any(s.lower() != "unknown" for s in states.values()):
            raise ValueError("known object states require actual evidence")
        step = obj.get("observation_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("observation_step must be a non-negative integer")
    return deepcopy(value)


def validate_information_request(value: Any) -> list[dict[str, Any]]:
    """Require actionable questions tied to objects and missing state fields."""
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("information_request must contain 1-8 questions")
    for question in value:
        if not isinstance(question, dict):
            raise ValueError("information_request entries must be objects")
        for field in ("object_id", "question", "reason"):
            _text(question.get(field), field)
        fields = question.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("information_request.fields must identify missing states")
        for field in fields:
            _text(field, "requested field")
    return deepcopy(value)
