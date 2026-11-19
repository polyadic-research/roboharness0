"""Generate validated staged plan proposals through an episode runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from typing import Any

from robots.robocasa.control_modes import IDENTITY_FIELDS
from rpent.harness.lifecycle import (
    COMPLETED_STEP_REVIEW_GUIDANCE,
    PHYSICAL_ATTEMPT_BUDGET_GUIDANCE,
    PLANNING_INFORMATION_SCOPE,
    PLANNING_STEP_GRANULARITY,
)
from rpent.harness.state import MAX_STEP_ATTEMPTS, control_step_schema, parse_steps
from rpent.planning.coverage import (
    parse_outcome_ids,
    parse_step_outcome_coverage,
    validate_outcome_coverage,
)
from rpent.planning.generator import PlanGenerator
from rpent.planning.information import (
    INFORMATION_REQUEST_SCHEMA,
    validate_information_request,
    validate_object_states,
)
from rpent.planning.instructions import parse_numbered_outcomes
from rpent.planning.outcome_program import OutcomeProgram, compile_outcome_program
from rpent.planning.schemas import Stage
from rpent.runtime.base import AgentRuntime
from rpent.tools.toolkit import Toolkit


class RuntimePlanGenerator(PlanGenerator):
    """Use a dedicated runtime instance for one nested planning request."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        max_turns: int = 4,
        use_chinese: bool = False,
        require_outcome_program: bool = True,
        action_guidance: str = "",
        allow_information_requests: bool = False,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._allow_information_requests = allow_information_requests
        self._runtime = runtime
        self._max_turns = max_turns
        self._use_chinese = bool(use_chinese)
        self._require_outcome_program = bool(require_outcome_program)
        self._action_guidance = action_guidance.strip()
        self._action_history_source: Callable[[], dict[str, Any]] | None = None
        self._control_config: dict[str, Any] = {}

    def configure_control_modes(
        self, *, policy_family: str | None, task_name: str | None,
        transition_takeover_enabled: bool = True, op_takeover_enabled: bool = True,
        required_monitor_operations: set[Any] | None = None,
        escalated_operations: set[Any] | None = None,
        control_constraints: dict[str, Any] | None = None,
    ) -> None:
        """Bind trusted episode/config values for candidate parsing."""
        self._control_config = {
            "policy_family": policy_family, "task_name": task_name,
            "transition_takeover_enabled": transition_takeover_enabled,
            "op_takeover_enabled": op_takeover_enabled,
            "required_monitor_operations": set(required_monitor_operations or ()),
            "escalated_operations": set(escalated_operations or ()),
            "control_constraints": dict(control_constraints or {}),
        }

    def set_action_history_source(self, source: Callable[[], dict[str, Any]]) -> None:
        """Bind the active scene's non-privileged command history."""
        self._action_history_source = source

    def generate(
        self,
        *,
        goal: str,
        observations: list[str],
        constraints: list[str],
        physical_tools: set[str],
        object_states: list[dict[str, Any]] | None = None,
        information_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        numbered = (
            parse_numbered_outcomes(goal) if self._require_outcome_program else ()
        )
        explicit_contract = [
            {"id": item.id, "number": item.number, "text": item.text}
            for item in numbered
        ]
        payload: dict[str, Any] = {
            "request_type": "initial_plan",
            "goal": goal,
            "observations": observations,
            "constraints": constraints,
            "available_physical_tools": sorted(physical_tools),
        }
        self._add_information(payload, object_states, information_history)
        if self._require_outcome_program:
            payload["outcome_contract"] = explicit_contract
        if explicit_contract:
            payload["required_outcome_ids"] = [item["id"] for item in explicit_contract]
        system_prompt = (
            "You are a robot task planner. Produce a concise, grounded physical "
            "execution plan from the supplied observations. Do not execute any "
            "actions. Use only available_physical_tools in allowed_tools. Each "
            "step must describe one auditable skill-level action and concrete "
            "post-action evidence. Group repeated waypoints, alignment "
            "corrections, or other same-purpose motions into one step; the executor "
            "can report progress and remain on that step until its evidence is "
            "satisfied."
            + _attempt_budget_guidance(physical_tools)
            + " Do not create a "
            "separate plan step for every small geometric waypoint or observation. "
            "Set max_attempts only when a bounded same-step retry is safe. Call the "
            "finish tool exactly once with the strategy and steps; do not answer "
            "with prose." + _episode_continuity_requirement()
        )
        if self._require_outcome_program:
            system_prompt += (
                " First compile every independently verifiable requested outcome "
                "into an immutable outcome contract. Preserve numbered outcome IDs "
                "and text exactly; use unique IDs for natural-language outcomes. "
                "Partition all outcomes into stages exactly once and in instruction "
                "order. Return detailed steps only for the first stage; every step "
                "must name its stage_id and cover exactly one outcome; flattened "
                "covers_outcomes must exactly match that stage's outcome_ids. Call "
                "the finish tool with the strategy, outcomes, stages, and first-stage "
                "steps." + _outcome_lifecycle_requirement()
            )
        system_prompt += self._language_requirement()
        return self._generate_candidate(
            goal=goal,
            payload=payload,
            system_prompt=system_prompt,
            physical_tools=physical_tools,
            explicit_contract=(
                explicit_contract if self._require_outcome_program else None
            ),
            require_contract=self._require_outcome_program,
        )

    def generate_replan(
        self,
        *,
        goal: str,
        reason: str,
        observations: list[str],
        constraints: list[str],
        current_plan: dict[str, Any],
        physical_tools: set[str],
        object_states: list[dict[str, Any]] | None = None,
        information_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        required_outcome_ids: tuple[str, ...] | None = None
        deferred_outcome_ids: tuple[str, ...] | None = None
        explicit_legacy_opt_out = False
        has_outcome_program = current_plan.get("outcome_program") is not None
        if "required_outcome_ids" in current_plan:
            parsed = parse_outcome_ids(
                current_plan["required_outcome_ids"],
                field="current_plan.required_outcome_ids",
            )
            # Empty IDs are a legacy opt-out only when no immutable contract exists.
            explicit_legacy_opt_out = not parsed and not has_outcome_program
            required_outcome_ids = None if explicit_legacy_opt_out else parsed
        if "deferred_outcome_ids" in current_plan:
            deferred_outcome_ids = parse_outcome_ids(
                current_plan["deferred_outcome_ids"],
                field="current_plan.deferred_outcome_ids",
            )

        program = (
            _load_outcome_program(current_plan.get("outcome_program"))
            if not explicit_legacy_opt_out
            else None
        )
        contract_aware = program is not None
        if contract_aware:
            assert program is not None
            completed = parse_outcome_ids(
                current_plan.get("completed_outcome_ids", []),
                field="current_plan.completed_outcome_ids",
            )
            required_outcome_ids = _validate_remaining_outcomes(
                program,
                required_outcome_ids=required_outcome_ids,
                completed_outcome_ids=completed,
                deferred_outcome_ids=deferred_outcome_ids,
            )

        payload: dict[str, Any] = {
            "request_type": "replan",
            "goal": goal,
            "failure_reason": reason,
            "observations": observations,
            "constraints": constraints,
            "current_plan": current_plan,
            "available_physical_tools": sorted(physical_tools),
        }
        self._add_information(payload, object_states, information_history)
        if required_outcome_ids is not None:
            payload["required_outcome_ids"] = list(required_outcome_ids)
        if deferred_outcome_ids is not None:
            payload["deferred_outcome_ids"] = list(deferred_outcome_ids)
        if contract_aware:
            assert program is not None
            payload["immutable_outcome_program"] = program.to_payload()

        completion_review = bool(current_plan.get("completion_review", {}).get("required"))
        system_prompt = (
            "You are a robot task replanner. Produce a concise, grounded "
            + (
                "repair of the original goal using completed_step_reviews. Earlier "
                "passed steps may need repair; completed_steps is history, not an "
                "exclusion from planning. "
                if completion_review else
                "replacement for only the unfinished portion of the supplied plan. "
            )
            + "Preserve the goal and account for the failure reason, observations, "
            "completed steps, and constraints. "
            + (
                "Do not blindly repeat completed actions or " if completion_review else
                "Do not repeat completed actions or "
            )
            + "execute anything. Use only available_physical_tools in allowed_tools. "
            "Each step must describe one auditable skill-level action and concrete "
            "post-action evidence. Use as few replacement steps as the remaining "
            "goal permits and never exceed 16. Group repeated waypoints, alignment "
            "corrections, or other same-purpose motions into one step; the executor "
            "can report progress and remain on that step."
            + _attempt_budget_guidance(physical_tools)
            + " Do not create a separate plan step for every small "
            "geometric waypoint or observation. Set max_attempts only when a "
            "bounded same-step retry is safe. Call the finish tool exactly once "
            "with the replacement strategy and steps; do not answer with prose."
            + _episode_continuity_requirement()
            + " In strategy, state the observed condition behind failure_reason "
            "and the executable condition the replacement changes: a different "
            "tool or stop condition, a grounded pose, a different task phase "
            "supported by fresh observations, or more policy time for a step "
            "whose observations show progress. Renaming a step as a final "
            "attempt is not a change, and rewording the same tool call is not a "
            "change. Keep the "
            "goal's target identity and source constraint; update their "
            "grounding only from fresh observations."
        )
        if completion_review:
            system_prompt += " " + COMPLETED_STEP_REVIEW_GUIDANCE
        if current_plan.get("verified_facts"):
            system_prompt += (
                " verified_facts contains only locally confirmed results from "
                "the completed-step review. Preserve them unless fresh evidence "
                "contradicts them; never promote the review to native success."
                if completion_review or current_plan.get("completed_step_reviews") else
                " Treat verified_facts as sticky facts established by passed "
                "verification. An ordinary action failure or ambiguous observation "
                "must not rename, replace, or negate a verified target or object "
                "relation; preserve those facts while repairing unfinished work."
            )
        if required_outcome_ids is not None:
            system_prompt += _coverage_requirement()
        if deferred_outcome_ids is not None:
            system_prompt += (
                " Do not cover deferred_outcome_ids; they remain immutable and "
                "will be planned by a later authorized scope."
            )
        if contract_aware:
            system_prompt += (
                " The immutable_outcome_program is authoritative. Do not return "
                "or rewrite outcomes or outcome_program. Repartition every "
                "required_outcome_id into replacement stages exactly once and in "
                "order. Detailed steps cover only the first replacement stage, "
                "name its stage_id, cover exactly one outcome per step, and never "
                "cover completed or invented IDs."
            )
        system_prompt += self._language_requirement()
        return self._generate_candidate(
            goal=goal,
            payload=payload,
            system_prompt=system_prompt,
            physical_tools=physical_tools,
            required_outcome_ids=required_outcome_ids,
            immutable_program=program if contract_aware else None,
        )

    @staticmethod
    def _add_information(
        payload: dict[str, Any],
        object_states: list[dict[str, Any]] | None,
        information_history: list[dict[str, Any]] | None,
    ) -> None:
        if object_states is not None:
            payload["object_states"] = validate_object_states(object_states)
        if information_history:
            payload["information_history"] = deepcopy(information_history)

    def _generate_candidate(
        self,
        *,
        goal: str,
        payload: dict[str, Any],
        system_prompt: str,
        physical_tools: set[str],
        required_outcome_ids: tuple[str, ...] | None = None,
        explicit_contract: list[dict[str, Any]] | None = None,
        immutable_program: OutcomeProgram | None = None,
        require_contract: bool = False,
    ) -> dict[str, Any]:
        system_prompt += "\n\n" + PLANNING_STEP_GRANULARITY
        if self._allow_information_requests:
            system_prompt += (
                "\n\nUse object_states and their evidence to assess prerequisites. Do not "
                "schedule a redundant action for a condition already satisfied (for example, "
                "opening an already open door). Missing or unknown state is not closed, "
                "empty, absent or failed. "
                + PLANNING_INFORMATION_SCOPE
                + " For VLA skill-level execution, exact waypoints, a complete transport "
                "route, and a predetermined destination pose are not planning inputs "
                "unless the selected tool or task actually requires them. For example, "
                "visible foods and an empty bowl permit food acquisition and placement "
                "before selecting a suitable space in an already open fridge. A later "
                "fridge-placement step must require fresh destination/access evidence, "
                "not assert that its space or route is already clear. If a prerequisite "
                "for the next grounded action is missing, conflicting or uncertain, "
                "call finish with information_request instead of a plan: for "
                "each question name object_id, fields, question and why it changes the "
                "next action in reason. You may ask about an omitted relevant object. "
                "Do not return strategy/steps/outcomes/stages with a question. The "
                "executor will answer from public evidence and resume this request. "
                "Use information_history as prior questions and answers, not sensor "
                "measurements; never treat a model assertion as new evidence. If the "
                "next-action prerequisite is still unknown, request clarification rather than "
                "guessing or fabricating a physical prerequisite."
            )
        if self._action_guidance:
            system_prompt += (
                "\n\nEnvironment action constraints shared with the executor. "
                "Build a plan the executor can follow under these constraints; "
                "do not schedule a tool whose required grounding is unavailable:\n"
                + self._action_guidance
            )
        history = (
            deepcopy(self._action_history_source())
            if self._action_history_source is not None
            else None
        )
        if history is not None:
            payload["action_history"] = _recent_action_history(history)
            system_prompt += (
                "\n\nThe action_history contains native commands, not measured motion or "
                "proof of contact/success. Its recent chunks are included automatically; "
                "use read_action_history to inspect older chunks by start_chunk. "
                "Follow next_chunk until it is null when the whole history is needed. "
                "Use step_ids and executed_actions to distinguish confirmed commands "
                "from unexecuted suffixes and unconfirmed failed attempts."
            )
        toolkit = _PlanOutputToolkit(
            physical_tools,
            require_contract=require_contract,
            require_stages=immutable_program is not None,
            action_history=history,
            allow_information_requests=self._allow_information_requests,
        )
        user_message = json.dumps(
            payload,
            ensure_ascii=False,
            **({"separators": (",", ":")} if history is not None else {"indent": 2}),
        )
        # The outer embodied runtime may be waiting on this nested request and
        # provider SDKs may already own an event loop on the dispatch thread.
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="plan-generator",
        ) as pool:
            result = None
            prompts = (
                system_prompt,
                system_prompt
                + " Your previous response did not invoke the finish tool. A JSON "
                "object in assistant text is invalid. Retry the identical request "
                "and invoke the available finish tool exactly once.",
            )
            for candidate_prompt in prompts:
                result = pool.submit(
                    copy_context().run,
                    self._runtime.solve,
                    system_prompt=candidate_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=self._max_turns,
                    input_queue=None,
                ).result()
                if result.error or isinstance(result.finish_result, dict):
                    break

        assert result is not None
        if result.error:
            raise RuntimeError(f"planning LLM failed: {result.error}")
        output = result.finish_result
        if not isinstance(output, dict):
            raise RuntimeError("planning LLM did not call finish with a plan")
        if "information_request" in output:
            if not self._allow_information_requests:
                raise ValueError(
                    "information requests are not enabled for this planner"
                )
            if any(
                output.get(key) is not None
                for key in ("strategy", "steps", "outcomes", "stages")
            ):
                raise ValueError("information_request cannot also return a plan")
            questions = validate_information_request(output["information_request"])
            if self._use_chinese and any(
                not _contains_cjk(question[field])
                for question in questions
                for field in ("question", "reason")
            ):
                raise ValueError(
                    "planner questions/reasons must use Simplified Chinese when --use_chines is set"
                )
            return {"information_request": questions}
        if require_contract and not explicit_contract:
            output = _canonicalize_natural_outcome_ids(output)

        strategy = str(output.get("strategy", "")).strip()
        if not strategy:
            raise ValueError("planning LLM returned an empty strategy")
        raw_steps = output.get("steps")
        parsed_steps = parse_steps(raw_steps, physical_tools, **self._control_config)
        coverage: tuple[tuple[str, ...], ...] | None = None
        stages: tuple[Stage, ...] = ()
        program: OutcomeProgram | None = immutable_program

        if require_contract:
            raw_outcomes = output.get("outcomes")
            raw_stages = output.get("stages")
            if not isinstance(raw_outcomes, list) or not raw_outcomes:
                raise ValueError("initial plans require non-empty immutable outcomes")
            if not isinstance(raw_stages, list):
                raise ValueError("initial plans require non-empty stages")
            program = compile_outcome_program(
                goal,
                raw_outcomes,
                raw_stages,
                explicit_contract=explicit_contract,
            )
            required_outcome_ids = tuple(item.id for item in program.outcomes)
            stages = _stages_from_program(program)
            coverage = _validate_first_stage_steps(
                raw_steps,
                stages=stages,
                required_outcome_ids=required_outcome_ids,
            )
        elif immutable_program is not None:
            if "outcomes" in output or "outcome_program" in output:
                raise ValueError("replans cannot replace outcomes or outcome_program")
            stages = _parse_stages(output.get("stages"), required_outcome_ids or ())
            coverage = _validate_first_stage_steps(
                raw_steps,
                stages=stages,
                required_outcome_ids=required_outcome_ids or (),
            )
        elif required_outcome_ids is not None:
            coverage = parse_step_outcome_coverage(raw_steps)
            validate_outcome_coverage(coverage, required_outcome_ids)

        if self._use_chinese:
            values = [strategy]
            for step in parsed_steps:
                values.extend(
                    [step.intent, *step.preconditions, *step.expected_evidence]
                )
                if step.on_failure != "replan":
                    values.append(step.on_failure)
            invalid = [value for value in values if value and not _contains_cjk(value)]
            if invalid:
                raise ValueError(
                    "planning LLM returned non-Chinese planning text while "
                    f"--use_chines is set: {invalid[0]!r}"
                )

        steps = [
            {
                "id": step.id,
                "intent": step.intent,
                "allowed_tools": list(step.allowed_tools),
                "preconditions": list(step.preconditions),
                "expected_evidence": list(step.expected_evidence),
                "on_failure": step.on_failure,
                "max_attempts": step.max_attempts,
            }
            for step in parsed_steps
        ]
        for raw, parsed, payload in zip(raw_steps, parsed_steps, steps, strict=True):
            if any(field in raw for field in ("skill_row", "tier", "control_mode")):
                payload.update(
                    skill_row=parsed.skill_row, tier=parsed.tier,
                    control_mode=parsed.control_mode, defaulted=parsed.defaulted,
                )
            for field in IDENTITY_FIELDS:
                if getattr(parsed, field) is not None:
                    payload[field] = getattr(parsed, field)
            if "step_type" in raw:
                payload["step_type"] = parsed.step_type
        if coverage is not None:
            assert isinstance(raw_steps, list)
            for raw, step, covered_ids in zip(raw_steps, steps, coverage, strict=True):
                step["covers_outcomes"] = list(covered_ids)
                if "stage_id" in raw:
                    step["stage_id"] = str(raw["stage_id"]).strip()

        candidate: dict[str, Any] = {
            "goal": goal,
            "strategy": strategy,
            "steps": steps,
        }
        if program is not None:
            program_payload = json.loads(
                json.dumps(program.to_payload(), ensure_ascii=False)
            )
            candidate.update(
                {
                    "outcome_program": program_payload,
                    "outcomes": deepcopy(program_payload["outcomes"]),
                    "contract_id": program.contract_id,
                    "outcome_program_schema_version": program.schema_version,
                    "required_outcome_ids": list(required_outcome_ids or ()),
                    "stages": [
                        {
                            "id": stage.id,
                            "intent": stage.intent,
                            "outcome_ids": list(stage.outcome_ids),
                        }
                        for stage in stages
                    ],
                }
            )
        return candidate

    def _language_requirement(self) -> str:
        if not self._use_chinese:
            return ""
        return (
            " Write every natural-language planning field in Simplified Chinese, "
            "including strategy, intent, preconditions, expected_evidence, and "
            "on_failure, plus information_request question/reason when enabled. "
            "Keep tool names, JSON keys, step identifiers, and numeric "
            "coordinates unchanged. Do not use a bare machine expression as an "
            "entire natural-language field; add a Chinese description, for "
            "example `环境成功条件尚未满足（environment_success=false）`."
        )


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff" for char in text
    )


def _attempt_budget_guidance(physical_tools: set[str]) -> str:
    guidance = " " + PHYSICAL_ATTEMPT_BUDGET_GUIDANCE
    if {"pi0_pick", "set_gripper"}.issubset(physical_tools):
        guidance += (
            " For example, pi0_pick followed by set_gripper requires "
            "max_attempts>=2 without a retry and >=3 when preserving one safe "
            "missed-grasp retry."
        )
    return guidance


def _episode_continuity_requirement() -> str:
    return (
        " The finish tool only returns this candidate plan to the executor; it "
        "neither executes the plan nor ends the episode. Every step must "
        "physically pursue the goal: never add a step whose purpose is to "
        "acknowledge, record, or declare failure, and write on_failure as the "
        "recovery to attempt next, never as a request to stop the episode."
    )


def _coverage_requirement() -> str:
    return (
        " Each step must include covers_outcomes as a string array. Across all "
        "steps, those IDs must cover required_outcome_ids exactly once and in "
        "the supplied order. One step may cover multiple consecutive outcomes."
    )


def _outcome_lifecycle_requirement() -> str:
    return (
        " Each outcome requires description, success_predicate, depends_on, and "
        "lifecycle. Use event for a milestone intentionally consumed later, "
        "persistent for a condition that must remain true, and terminal for a "
        "predicate checked at final completion."
    )


class _PlanOutputToolkit(Toolkit):
    """Minimal toolkit used to obtain one structured planning result."""

    def __init__(
        self,
        physical_tools: set[str],
        *,
        require_contract: bool = False,
        require_stages: bool = False,
        action_history: dict[str, Any] | None = None,
        allow_information_requests: bool = False,
    ) -> None:
        self._plan_physical_tools = sorted(physical_tools)
        self._require_contract = require_contract
        self._require_stages = require_stages
        super().__init__()
        if action_history is not None:
            self.add_tool(
                "read_action_history",
                {
                    "name": "read_action_history",
                    "description": (
                        "Read an immutable page of this scene's recorded native command chunks. "
                        "This is read-only and cannot execute or repeat actions. Follow next_chunk "
                        "for the next page; the current plan request includes recent chunks."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "start_chunk": {
                                "type": "integer",
                                "minimum": 0,
                                "default": 0,
                            }
                        },
                    },
                },
                lambda start_chunk=0: _action_history_page(action_history, start_chunk),
            )
        spec = self._finish_spec()
        if allow_information_requests:
            schema = spec["input_schema"]
            plan_required = schema.pop("required")
            schema["properties"]["information_request"] = deepcopy(
                INFORMATION_REQUEST_SCHEMA
            )
            schema["anyOf"] = [
                {"required": plan_required},
                {"required": ["information_request"]},
            ]
            spec["description"] += (
                " Or request missing scene information from the executor, without returning a plan."
            )
        self.add_tool("finish", spec, self._finish)

    def _register_common_tools(self) -> None:
        """Do not expose filesystem or environment tools to the nested planner."""

    def _finish_spec(self) -> dict[str, Any]:
        allowed_tool_items: dict[str, Any] = {"type": "string"}
        if self._plan_physical_tools:
            allowed_tool_items["enum"] = self._plan_physical_tools
        covers_outcomes_schema: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string"},
        }
        if self._require_contract or self._require_stages:
            covers_outcomes_schema.update(
                {
                    "minItems": 1,
                    "maxItems": 1,
                    "description": (
                        "For a staged outcome contract, the detailed step must "
                        "cover exactly one outcome."
                    ),
                }
            )
        required = ["strategy", "steps"]
        if self._require_contract:
            required.extend(["outcomes", "stages"])
        elif self._require_stages:
            required.append("stages")
        return {
            "name": "finish",
            "description": (
                "Return the completed candidate plan. This does not execute or "
                "register the plan."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "intent": {"type": "string"},
                                **control_step_schema(),
                                "allowed_tools": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": allowed_tool_items,
                                },
                                "preconditions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "expected_evidence": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "covers_outcomes": covers_outcomes_schema,
                                "stage_id": {"type": "string"},
                                "on_failure": {"type": "string"},
                                "max_attempts": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": MAX_STEP_ATTEMPTS,
                                    "description": PHYSICAL_ATTEMPT_BUDGET_GUIDANCE + " Default 2.",
                                },
                            },
                            "required": [
                                "id",
                                "intent",
                                "allowed_tools",
                                "expected_evidence",
                            ],
                        },
                    },
                    "outcomes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "instruction_index": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "description": {"type": "string"},
                                "success_predicate": {"type": "string"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "lifecycle": {
                                    "type": "string",
                                    "enum": ["event", "persistent", "terminal"],
                                },
                            },
                            "required": [
                                "id",
                                "instruction_index",
                                "description",
                                "success_predicate",
                                "depends_on",
                                "lifecycle",
                            ],
                        },
                    },
                    "stages": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "intent": {"type": "string"},
                                "outcome_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "intent", "outcome_ids"],
                        },
                    },
                },
                "required": required,
            },
        }

    @staticmethod
    def _finish(
        strategy: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        outcomes: list[dict[str, Any]] | None = None,
        stages: list[dict[str, Any]] | None = None,
        information_request: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if information_request is not None:
            if any(value is not None for value in (strategy, steps, outcomes, stages)):
                raise ValueError("information_request cannot also return a plan")
            return {
                "_finish": True,
                "information_request": validate_information_request(
                    information_request
                ),
            }
        return {
            "_finish": True,
            "strategy": strategy,
            "steps": steps,
            **({"outcomes": outcomes} if outcomes is not None else {}),
            **({"stages": stages} if stages is not None else {}),
        }


# A planning request must stay under the gateway prompt limit (CodexGateway
# CODEX_MAX_PROMPT_CHARS, 200k characters) next to a ~70k-character system
# prompt; 1.0.11 run03 replans failed at ~202k with 122k of history.
_HISTORY_WINDOW_ACTIONS = 250
_HISTORY_WINDOW_CHARS = 40_000
_HISTORY_PAGE_CHARS = 20_000


def _sliceable(record: dict[str, Any]) -> bool:
    """A merged privileged record: one action, one step and one native chunk per row."""
    rows = len(record.get("actions", []))
    return rows > 1 and record.get("chunk_count", 1) == rows and len(record.get("step_ids", [])) == rows


def _record_rows(record: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Rows [start, stop) of a sliceable record, as a record of its own native chunks."""
    part = dict(record)
    part["actions"] = record["actions"][start:stop]
    part["step_ids"] = record["step_ids"][start:stop]
    part["chunk_index"] = record.get("chunk_index", 0) + start
    part["chunk_count"] = part["requested_actions"] = part["executed_actions"] = stop - start
    return part


def _fit_rows(record: dict[str, Any], max_chars: int, keep: str) -> dict[str, Any]:
    """The most rows of an oversized sliceable record that fit the bounds, newest ("tail") or oldest ("head")."""
    rows = len(record["actions"])

    def part(count: int) -> dict[str, Any]:
        if keep == "tail":
            fitted = _record_rows(record, rows - count, rows)
            fitted["rows_omitted_before"] = rows - count
            fitted["rows_note"] = (f"Only the last {count} of this record's {rows} rows fit; read_action_history "
                                   f"from start_chunk={record.get('chunk_index', 0)} pages the earlier rows.")
        else:
            fitted = _record_rows(record, 0, count)
            fitted["rows_omitted_after"] = rows - count
        return fitted

    low, high = 1, min(rows, _HISTORY_WINDOW_ACTIONS)
    while low < high:
        mid = (low + high + 1) // 2
        # +2 for the list brackets around a one-record window or page
        if len(json.dumps(part(mid), separators=(",", ":"), default=str)) + 2 <= max_chars:
            low = mid
        else:
            high = mid - 1
    return part(low)


def _bounded_records(records: list[dict[str, Any]], max_chars: int, keep: str = "head") -> list[dict[str, Any]]:
    """Whole records up to the action and character bounds.

    The first record is always included; when it alone exceeds the bounds and is a
    merged privileged record, only the rows that fit are kept (its newest rows for
    keep="tail"), so one long takeover cannot overflow the planner request (D10,
    1.2.0 full eval: one 835-action move_eef_path record pushed a replan to 218k).
    """
    kept, actions, chars = [], 0, 0
    for record in records:
        count = len(record["actions"])
        size = len(json.dumps(record, separators=(",", ":"), default=str))
        if kept and (actions + count > _HISTORY_WINDOW_ACTIONS or chars + size > max_chars):
            break
        if not kept and (count > _HISTORY_WINDOW_ACTIONS or size > max_chars) and _sliceable(record):
            kept.append(_fit_rows(record, max_chars, keep))
            break
        kept.append(record)
        actions += count
        chars += size
    return kept


def _recent_action_history(history: dict[str, Any]) -> dict[str, Any]:
    """Keep a bounded recent window; older exact commands remain pageable."""
    recent = _bounded_records(list(reversed(history["records"])), _HISTORY_WINDOW_CHARS, keep="tail")
    return {**history, "scope": "recent_chunks", "records": list(reversed(recent))}


def _action_history_page(history: dict[str, Any], start_chunk: int) -> dict[str, Any]:
    if (
        isinstance(start_chunk, bool)
        or not isinstance(start_chunk, int)
        or start_chunk < 0
    ):
        raise ValueError("start_chunk must be a nonnegative integer")
    records = history["records"]
    if start_chunk > history.get("total_chunks", len(records)):
        raise ValueError("start_chunk exceeds the available history")
    # start_chunk and next_chunk count native chunks (chunk_index); a merged
    # record covers chunk_count of them.
    position = next(
        (i for i, record in enumerate(records)
         if record.get("chunk_index", i) + record.get("chunk_count", 1) > start_chunk),
        len(records),
    )
    view = records[position:]
    if view:
        # A page may start inside a merged privileged record (one row per native chunk).
        offset = start_chunk - view[0].get("chunk_index", position)
        if offset > 0 and _sliceable(view[0]):
            view = [_record_rows(view[0], offset, len(view[0]["actions"]))] + view[1:]
    page = _bounded_records(view, _HISTORY_PAGE_CHARS)
    after = position + len(page)
    if page and page[-1].get("rows_omitted_after"):
        next_chunk = page[-1]["chunk_index"] + page[-1]["chunk_count"]
    else:
        next_chunk = records[after].get("chunk_index", after) if after < len(records) else None
    return {
        "action_history": {
            **history,
            "scope": "chunk_page",
            "records": page,
            "next_chunk": next_chunk,
        }
    }


def _canonicalize_natural_outcome_ids(output: dict[str, Any]) -> dict[str, Any]:
    raw_outcomes = output.get("outcomes")
    if not isinstance(raw_outcomes, list) or not raw_outcomes:
        return output
    old_ids = [
        str(item.get("id", "")).strip() if isinstance(item, dict) else ""
        for item in raw_outcomes
    ]
    if not all(old_ids) or len(old_ids) != len(set(old_ids)):
        return output
    mapping = {
        old_id: f"outcome-{index}" for index, old_id in enumerate(old_ids, start=1)
    }
    normalized = deepcopy(output)
    for index, outcome in enumerate(normalized["outcomes"], start=1):
        outcome["id"] = f"outcome-{index}"
        outcome["instruction_index"] = index
        outcome["depends_on"] = [
            mapping.get(str(value), str(value))
            for value in outcome.get("depends_on", [])
        ]
    for field in ("stages", "steps"):
        key = "outcome_ids" if field == "stages" else "covers_outcomes"
        for item in normalized.get(field) or []:
            if isinstance(item, dict):
                item[key] = [
                    mapping.get(str(value), str(value)) for value in item.get(key, [])
                ]
    return normalized


def _stages_from_program(program: OutcomeProgram) -> tuple[Stage, ...]:
    return tuple(
        Stage(id=item.id, intent=item.intent, outcome_ids=item.outcome_ids)
        for item in program.stages
    )


def _parse_stages(
    raw_stages: object, outcome_ids: tuple[str, ...]
) -> tuple[Stage, ...]:
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("remaining stages must be a non-empty array")
    stages: list[Stage] = []
    for item in raw_stages:
        if not isinstance(item, dict):
            raise ValueError("remaining stage must be an object")
        stages.append(
            Stage(
                id=str(item.get("id", "")).strip(),
                intent=str(item.get("intent", "")).strip(),
                outcome_ids=tuple(
                    str(value).strip() for value in item.get("outcome_ids", ())
                ),
            )
        )
    if len({stage.id for stage in stages}) != len(stages):
        raise ValueError("remaining stage IDs must be unique")
    validate_outcome_coverage(tuple(stage.outcome_ids for stage in stages), outcome_ids)
    return tuple(stages)


def _validate_first_stage_steps(
    raw_steps: object,
    *,
    stages: tuple[Stage, ...],
    required_outcome_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be an array")
    first = stages[0]
    coverage = parse_step_outcome_coverage(raw_steps)
    validate_outcome_coverage(coverage, first.outcome_ids)
    if any(len(covered_ids) != 1 for covered_ids in coverage):
        raise ValueError("each staged plan step must cover exactly one outcome")
    for index, step in enumerate(raw_steps):
        if str(step.get("stage_id", "")).strip() != first.id:
            raise ValueError(f"step {index} must use first stage ID {first.id!r}")
    unknown = sorted(set(first.outcome_ids) - set(required_outcome_ids))
    if unknown:
        raise ValueError(f"first stage contains unknown outcome IDs: {unknown}")
    return coverage


def _load_outcome_program(value: object) -> OutcomeProgram | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("current_plan.outcome_program must be an object")
    if value.get("schema_version") != "outcome-program/v1":
        raise ValueError("unsupported outcome program schema_version")
    outcomes = value.get("outcomes")
    stages = value.get("stages")
    if not isinstance(outcomes, (list, tuple)) or not isinstance(stages, (list, tuple)):
        raise ValueError("outcome program requires outcomes and stages arrays")
    diagnostics = value.get("compile_diagnostics")
    explicit_contract = None
    if isinstance(diagnostics, dict) and diagnostics.get("confidence") == 1.0:
        explicit_contract = [
            {
                "id": item.get("id"),
                "number": item.get("instruction_index"),
                "text": item.get("source_text"),
            }
            for item in outcomes
        ]
    compiled = compile_outcome_program(
        str(value.get("source_instruction", "")),
        list(outcomes),
        list(stages),
        explicit_contract=explicit_contract,
    )
    if compiled.contract_id != value.get("contract_id"):
        raise ValueError("outcome_program contract_id does not match its content")
    return compiled


def _validate_remaining_outcomes(
    program: OutcomeProgram,
    *,
    required_outcome_ids: tuple[str, ...] | None,
    completed_outcome_ids: tuple[str, ...],
    deferred_outcome_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    all_ids = tuple(item.id for item in program.outcomes)
    if len(completed_outcome_ids) != len(set(completed_outcome_ids)):
        raise ValueError("completed_outcome_ids contains duplicates")
    unknown_completed = sorted(set(completed_outcome_ids) - set(all_ids))
    if unknown_completed:
        raise ValueError(
            f"completed_outcome_ids contains unknown IDs: {unknown_completed}"
        )
    expected_remaining = tuple(
        outcome_id for outcome_id in all_ids if outcome_id not in completed_outcome_ids
    )
    if deferred_outcome_ids is not None:
        if required_outcome_ids is None or not required_outcome_ids:
            raise ValueError(
                "required_outcome_ids cannot be empty when deferred outcomes exist"
            )
        declared_remaining = required_outcome_ids + deferred_outcome_ids
    else:
        declared_remaining = required_outcome_ids
    if declared_remaining is not None and declared_remaining != expected_remaining:
        raise ValueError(
            "completed, required, and deferred outcome IDs must partition the "
            "immutable outcome contract exactly once and in order"
        )
    if not expected_remaining:
        raise ValueError("cannot replan because no unresolved outcomes remain")
    if deferred_outcome_ids is not None:
        assert required_outcome_ids is not None
        return required_outcome_ids
    return expected_remaining


__all__ = ["RuntimePlanGenerator"]
