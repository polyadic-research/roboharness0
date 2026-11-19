"""Small Harness-facing toolkit for RoboCasa365."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from robots.robocasa.binding import CHECKPOINT_CHUNK_SIZE
from robots.robocasa.control_modes import PRIVILEGED_TOOLS as _PRIVILEGED_TOOL_NAMES
from robots.robocasa.grasp_progress import GraspProgress
from robots.robocasa.outcomes import RoboCasaSkillOutcomeAdapter
from robots.robocasa.perception import localize_rgbd_points
from robots.robocasa.task_review import TaskActionReviewMixin
from rpent.episode.contracts import POLICY_REVIEW_PAUSE_REASON, StepResult
from rpent.episode.controller import _validate_max_actions
from rpent.episode.liveness import EpisodeBudget
from rpent.harness.lifecycle import (
    PRECLOSE_MAX_CONSECUTIVE_REJECTIONS,
    ROBOCASA_EPISODE_MAX_LLM_REQUESTS,
    ROBOCASA_EPISODE_MAX_SECONDS,
)
from rpent.recording.display import png_bytes as _png_bytes
from rpent.tools.toolkit import Toolkit, ToolResult, _write_run_artifact
from rpent.tools.vla import VLAOperation

_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
_OVERVIEW_CAMERA = "robot0_agentview_center"
_MAX_STEP_GUIDANCE_LENGTH = 1000
_POSITION_ACTION_SCALE_M = 0.05
_ROTATION_ACTION_SCALE_RAD = 0.5
_POSITION_TOLERANCE_M = 0.005
_ROTATION_TOLERANCE_RAD = 0.08
_OPEN_GRIPPER_THRESHOLD = 0.06
_RECOVERY_CLOSE_INTENT_STEPS = 5
_RECOVERY_CLOSE_MAX_STEPS = 20
_RECOVERY_LIFT_MAX_STEPS = 25
_RECOVERY_LIFT_M = 0.06
_RECOVERY_CANDIDATE_POSITION_TOLERANCE_M = 0.015
_RECOVERY_CANDIDATE_ROTATION_TOLERANCE_RAD = 0.15
_DEFAULT_TRANSITION_BUDGET = 250
_DEFAULT_MAX_CHUNKS = 10
_MAX_CHUNKS = 12
# Successful KettleBoiling demonstrations either start a sustained final close
# or descend toward the control in 150 actions; the shortest final close is 23.
_BURNER_APPROACH_GUARD_ACTIONS = 150
_BURNER_APPROACH_MIN_DESCENT_M = 0.02
_BURNER_CONTROL_CLOSE_INTENT_STEPS = 10
_BURNER_CONTROL_RESTORE_MAX_STEPS = 25
_BURNER_CONTROL_POLICY_TASKS = frozenset({"KettleBoiling"})
_PERCEPTION_STANDOFF_DISTANCE_M = 0.1
_ADJUST_MAX_TRANSLATION_M = 0.05
_ADJUST_MAX_ROTATION_DEG = 15.0
_RESUME_MAX_ACTIONS = 10
# A closure the LLM attributes to a scene-fixed control (lever, knob, button,
# fixture handle) acquires no movable object, so it needs no alignment review.
# The exemption covers repeated close/open cycles at that control while the
# base stays put (1.2.1).
_CLOSURE_TARGETS = ("movable_object", "fixed_control")
_FIXED_CONTROL_RADIUS_M = 0.06
_FIXED_CONTROL_BASE_TOLERANCE_M = 0.02
_FIXED_CONTROL_BASE_ROTATION_TOLERANCE_RAD = 0.05
_GRASP_FEEDBACK_KEYS = frozenset({
    "grasp_events", "grasp_status", "review_required", "interruption_scope", "grasp_lost_next",
    "_image_bytes", "_image_right_bytes", "_image_wrist_bytes", "_image_camera_ids",
})


class RoboCasaToolkit(TaskActionReviewMixin, Toolkit):
    """Expose one policy-neutral physical operation and safe observation."""

    PHYSICAL_TOOLS = frozenset(
        {"execute_vla", "servo_eef", "smolvla_pick", "continue_vla",
         "adjust_eef", "move_base", "adjust_torso", "resume_vla",
         "navigate_to", "move_eef_path", "adjust_torso_to", "pour_into", "place_container"}
    )

    def __init__(
        self,
        controller,
        *,
        initial_observation: dict[str, Any],
        instruction: str,
        policy_instruction: str | None = None,
        forward_step_guidance: bool = False,
        actions_per_policy_request: int | None = None,
        perception_source: Any | None = None,
        max_transitions_per_tool_call: int = _DEFAULT_TRANSITION_BUDGET,
        episode_horizon: int | None = None,
        agent_artifact_root: Path | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        if agent_artifact_root is not None:
            supplied_artifact_root = Path(agent_artifact_root)
            resolved_artifact_root = (
                supplied_artifact_root.parent.resolve()
                / supplied_artifact_root.name
            )
            write_spec = {
                **self._tools["write_text_file"][0],
                "description": (
                    "Write an agent-authored audit artifact. Relative paths are "
                    "confined to the current attempt's agent_artifacts directory."
                ),
            }

            def write_agent_artifact(path: str, content: str) -> dict[str, Any]:
                return _write_run_artifact(resolved_artifact_root, path, content)

            self.add_tool("write_text_file", write_spec, write_agent_artifact)
        if (
            isinstance(max_transitions_per_tool_call, bool)
            or not isinstance(max_transitions_per_tool_call, int)
            or max_transitions_per_tool_call < 1
        ):
            raise ValueError("max_transitions_per_tool_call must be a positive integer")
        self._instruction = instruction
        self._policy_instruction = policy_instruction or instruction
        self._active_policy_instruction = self._policy_instruction
        self._forward_step_guidance = forward_step_guidance
        self._continue_vla_step_intents = ("task_progress",)
        if self._policy_instruction in _BURNER_CONTROL_POLICY_TASKS:
            self._continue_vla_step_intents += ("burner_control",)
        self._actions_per_policy_request = actions_per_policy_request
        self._perception_source = perception_source
        self._max_transitions_per_tool_call = max_transitions_per_tool_call
        if episode_horizon is not None and (
            isinstance(episode_horizon, bool)
            or not isinstance(episode_horizon, int)
            or episode_horizon < 1
        ):
            raise ValueError("episode_horizon must be a positive integer")
        self._episode_horizon = episode_horizon
        self._episode_terminal = False
        self._consecutive_closed_empty = 0
        self._pick_progress: GraspProgress | None = None
        initial_state = np.asarray(initial_observation.get("state", ()))
        if initial_state.shape == (16,) and np.isfinite(initial_state).all():
            self._pick_progress = GraspProgress(
                initial_state, descent_threshold_m=0.05, ascent_threshold_m=0.05,
                closed_threshold=0.03, empty_threshold=0.002,
                open_threshold=_OPEN_GRIPPER_THRESHOLD,
            )
        self._pick_episode_id: str | None = None
        self._last_pick_step = -1
        self._confirmed_grasp_step: int | None = None
        self._pending_grasp_events: list[dict[str, Any]] = []
        self._burner_control_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self._burner_control_start_z: float | None = None
        self._burner_control_peak_z: float | None = None
        self._burner_control_policy_actions = 0
        self._burner_control_close_run = 0
        self._burner_control_max_close_run = 0
        self._precontact_grounding: dict[str, Any] | None = None
        # Robot observation preconditions, not a second Harness lifecycle.
        self._last_observed_step: int | None = None
        self._motion_observe_required = False
        self._resume_after_adjustment = False
        self._preclose_active = False
        self._preclose_pending = False
        self._preclose_closing = False
        self._preclose_approval: dict[str, Any] | None = None
        self._preclose_rejections = 0
        self._fixed_control_zones: list[dict[str, Any]] = []
        self._grasp_operation_key: tuple[str, str] | None = None
        self._last_executed_gripper: str | None = None
        self.episode_budget = EpisodeBudget(
            max_requests=ROBOCASA_EPISODE_MAX_LLM_REQUESTS,
            max_seconds=ROBOCASA_EPISODE_MAX_SECONDS,
        )
        self._preclose_recovery_in_progress = False
        install_guard = getattr(controller, "set_policy_action_guard", None)
        self._preclose_review_enabled = callable(install_guard)
        if self._preclose_review_enabled:
            install_guard(self._guard_policy_action)
        self._operation = VLAOperation(
            controller,
            initial_observation=initial_observation,
            instruction=self._policy_instruction,
            capture_action_history=True,
            transition_guard=self._observe_pick_transition,
        )
        self._action_history_cursor = 0
        self._init_task_review()
        self.set_skill_outcome_adapter(RoboCasaSkillOutcomeAdapter())
        self.configure_harness_physical_tools(set(self.PHYSICAL_TOOLS))
        self.configure_harness_evidence_tools(
            {"observe", "execute_vla", "back_project"}
        )
        forwarded_instruction_schema = {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_STEP_GUIDANCE_LENGTH,
            "description": (
                "One plain task or subtask sentence, forwarded only when "
                "intervene=true to replace the current policy instruction. "
                "Otherwise this optional text is audit-only. Preserve the "
                "original target and final goal; do not add instruction labels."
                " Switch on entry to a new registered long-task operation, or after "
                "an authorized semantic review when fresh current-goal evidence "
                "justifies an explicit language intervention in the same operation. "
                "Preserve operation identity, counters and original outcomes. "
                "Internal phase bookkeeping and physical retries alone never reword it."
            ),
        }
        intervention_properties = {
            "intervene": {
                "type": "boolean",
                "default": False,
                "description": (
                    "False or omitted keeps the current policy instruction "
                    "unchanged, including after replanning or recovery. True "
                    "requires instruction and explicitly replaces it."
                    " True is for entering a new registered long-task operation or "
                    "a grounded language intervention after authorized semantic review. "
                    "Internal phases or physical retries alone do not require it."
                ),
            },
        } if forward_step_guidance else {}
        self.add_tool(
            "observe",
            {
                "name": "observe",
                "description": "Read the current non-privileged RoboCasa observation.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.observe,
        )
        self.add_tool(
            "back_project",
            {
                "name": "back_project",
                "description": (
                    "Back-project 1-32 pixels from one current RoboCasa RGB-D "
                    "camera into world and robot-relative base coordinates without "
                    "advancing the simulator. Use several interior target pixels "
                    "from observe, use the numbered camera label of that same "
                    "image, and pass that exact observe step. The result also "
                    "contains a task-neutral conservative standoff pose derived from "
                    "the measured surface and current end-effector pose."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "camera": {
                            "type": "string",
                            "enum": list(_CAMERAS),
                        },
                        "expected_step": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "points": {
                            "type": "array",
                            "description": (
                                "Integer [row, column] coordinates in the named "
                                "observe image."
                            ),
                            "minItems": 1,
                            "maxItems": 32,
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                    "required": ["camera", "expected_step", "points"],
                },
            },
            self.back_project,
        )
        self.add_tool(
            "execute_vla",
            {
                "name": "execute_vla",
                "description": (
                    "Execute one policy action chunk containing multiple validated "
                    "12D low-level actions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        **({"instruction": dict(forwarded_instruction_schema)}
                           if forward_step_guidance else {}),
                        **intervention_properties,
                        "max_actions": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "description": (
                                "Optional cap on low-level actions in this chunk. "
                                "Omit or pass null to use the configured execution "
                                "horizon; smaller values may further cap the request."
                            ),
                        },
                    },
                },
            },
            self.execute_vla,
        )
        self.add_tool(
            "review_grasp",
            {
                "name": "review_grasp",
                "description": (
                    "Record your fresh robot0_eye_in_hand inspection before grasp closure. "
                    "Check the intended object is between both fingers and the finger "
                    "closing axis spans its narrower graspable dimension. Judge only what "
                    "the image shows: approve both checks only when the image shows them. "
                    "If the target cannot be seen well enough to judge (occluded, out of "
                    "frame, unclear), zoom on the gap between the fingertips; if it is "
                    "still not visible, set target_visible=false: an unseen target is not "
                    "misaligned, so the pose is left alone and the policy may close. This "
                    "performs no motion and does not prove contact or grasp success. If "
                    "the image shows either check failing, correct with adjust_eef keeping "
                    "the gripper open, observe and review again, resolve Harness "
                    "verification, then resume_vla. After two "
                    "consecutive rejections, policy_close_allowed=true permits a fresh "
                    "policy close without visual approval; do not repeat reviews. "
                    "closure_target=fixed_control records that the fingers are at a part "
                    "fixed to the scene (lever, knob, button, switch, fixture door or "
                    "handle), so the closure acquires no movable object: it needs no "
                    "alignment, and closures within 6 cm of this hand position stay "
                    "unreviewed for the rest of the operation while the base stays put."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expected_step": {"type": "integer", "minimum": 0},
                        "object_between_fingers": {"type": "boolean"},
                        "narrow_axis_aligned": {"type": "boolean"},
                        "evidence": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "closure_target": {
                            "type": "string",
                            "enum": list(_CLOSURE_TARGETS),
                            "default": "movable_object",
                            "description": (
                                "movable_object (default) for an object the closure will "
                                "acquire; fixed_control for a scene-fixed control the "
                                "fingers press, push, pull or turn. Never fixed_control "
                                "for anything that will be lifted or carried."
                            ),
                        },
                        "target_visible": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "false when the intended object cannot be seen well enough "
                                "in the current wrist image (after zooming on the finger gap) "
                                "to judge either check; the two checks are then ignored and "
                                "the policy may close at its own pose without a correction."
                            ),
                        },
                    },
                    "required": ["expected_step", "object_between_fingers", "narrow_axis_aligned", "evidence"],
                },
            },
            self.review_grasp,
        )
        self.add_tool(
            "smolvla_pick",
            {
                "name": "smolvla_pick",
                "description": (
                    "Run repeated policy chunks for one grasp attempt, retaining "
                    "motion evidence across bounded calls in the same episode. "
                    "An already confirmed and still closed grasp returns without "
                    "executing more actions. Pauses before the first native close "
                    "command with preclose_review_required; observe the wrist image "
                    "and call review_grasp before continuing. Approval without "
                    "state, instruction or operation changes resumes the retained "
                    "actions without another policy prediction. Rejection or "
                    "correction discards them. Returns local "
                    "success only after ordered descent, non-empty gripper closure, "
                    "and lift; official task success remains environment.success."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_STEP_GUIDANCE_LENGTH,
                            **({"description": forwarded_instruction_schema["description"]}
                               if forward_step_guidance else {}),
                        },
                        **intervention_properties,
                        "max_chunks": {
                            "type": "integer",
                            "minimum": _DEFAULT_MAX_CHUNKS,
                            "maximum": _MAX_CHUNKS,
                            "default": _DEFAULT_MAX_CHUNKS,
                            "description": (
                                "Omit this optional field during evaluation. Values "
                                "below the default can end a grasp attempt before "
                                "physical closure evidence is available."
                            ),
                        },
                    },
                    "required": [] if forward_step_guidance else ["instruction"],
                },
            },
            self.smolvla_pick,
        )
        self.add_tool(
            "servo_eef",
            {
                "name": "servo_eef",
                "description": (
                    "Move the end effector toward one explicit relative-base pose "
                    "using bounded proprioceptive feedback while holding the requested "
                    "gripper command. Use only with a pose supplied by the task "
                    "override or established from prior non-privileged trajectory data."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_position": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "minimum": -2.0,
                                "maximum": 2.0,
                            },
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "target_quaternion": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "minimum": -1.0,
                                "maximum": 1.0,
                            },
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "gripper": {"type": "string", "enum": ["open", "close"]},
                        "max_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 25,
                            "default": 25,
                        },
                        "settle_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 1,
                        },
                    },
                    "required": [
                        "target_position",
                        "target_quaternion",
                        "gripper",
                    ],
                },
            },
            self.servo_eef,
        )
        self.add_tool(
            "adjust_eef",
            {
                "name": "adjust_eef",
                "description": (
                    "Directly correct pre-close grasp alignment, a stalled motion, or lost contact from a fresh "
                    "observe. Add a small translation and axis-angle rotation to the "
                    "measured EEF pose in the robot base frame; no world coordinates "
                    "or guessed absolute pose. Do not use for productive contact with the intended target; assess "
                    "task-direction progress and typical-step timing under ACTION and TASK SCOPE. "
                    "Limits: translation norm 0.05 m, "
                    "rotation norm 15 degrees, at most 25 feedback actions. Ground "
                    "direction in current images/proprioception and back_project "
                    "when useful. Keeps the policy instruction. Pre-close corrections "
                    "must keep the gripper open and invalidate the previous grasp review. Then observe and "
                    "resolve Harness verification before resume_vla. target_reached "
                    "describes only the EEF, not object motion or restored contact."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expected_step": {"type": "integer", "minimum": 0},
                        "delta_position": {
                            "type": "array", "minItems": 3, "maxItems": 3,
                            "items": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "description": "[dx, dy, dz] metres in robot base axes; vector norm <=0.05.",
                        },
                        "rotation_degrees": {
                            "type": "array", "minItems": 3, "maxItems": 3,
                            "items": {"type": "number", "minimum": -15, "maximum": 15},
                            "default": [0, 0, 0],
                            "description": (
                                "Axis-angle vector in robot base axes: direction is the "
                                "rotation axis, norm is angle in degrees, <=15. "
                                "For example [0,0,5] rotates +5 degrees about base Z. "
                                "Not Euler angles; does not specify an object hinge angle."
                            ),
                        },
                        "gripper": {
                            "type": "string", "enum": ["keep", "open", "close"], "default": "keep",
                            "description": (
                                "keep preserves the last confirmed native binary gripper "
                                "command. With no command history choose open or close explicitly."
                            ),
                        },
                        "max_steps": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10},
                    },
                    "required": ["expected_step", "delta_position"],
                },
            },
            self.adjust_eef,
        )
        self._register_body_motion_tools()
        from robots.robocasa.privileged_tools import register_privileged_tools
        register_privileged_tools(self)
        self.add_tool(
            "resume_vla",
            {
                "name": "resume_vla",
                "description": (
                    "After adjust_eef, move_base, or adjust_torso, a fresh observe, and Harness verification, "
                    "infer one new VLA chunk from the corrected observation. Keeps "
                    "the current policy instruction exactly. Executes at most 10 "
                    "actions (or the smaller configured policy horizon), then requires "
                    "observe and verification again. If progress resumes, use normal "
                    "policy tools; if it does not, reassess the observed contact."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "max_actions": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                },
            },
            self.resume_vla,
        )
        self.add_tool(
            "continue_vla",
            {
                "name": "continue_vla",
                "description": (
                    "Run repeated policy chunks for transport, placement, release, "
                    "or contact. Only environment.success is reported as physical "
                    "success. "
                    + (
                        "The policy keeps its current instruction unless "
                        "intervene=true explicitly replaces it."
                        if forward_step_guidance else
                        "The checkpoint adapter keeps the checkpoint task's "
                        "policy instruction canonical and handles supported retry setup."
                    )
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_STEP_GUIDANCE_LENGTH,
                            "description": (
                                forwarded_instruction_schema["description"]
                                if forward_step_guidance else
                                "Planner audit-only description of this physical "
                                "step; never forwarded to the policy."
                            ),
                        },
                        **intervention_properties,
                        "max_chunks": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_CHUNKS,
                            "default": _DEFAULT_MAX_CHUNKS,
                        },
                        "step_intent": {
                            "type": "string",
                            "enum": list(self._continue_vla_step_intents),
                            "default": "task_progress",
                            "description": (
                                "Structured physical intent independent of the "
                                "optional policy-instruction intervention."
                                if forward_step_guidance else
                                "Structured audit and retry intent for the checkpoint "
                                "adapter. It never changes the policy instruction."
                            ),
                        },
                    },
                    "required": [] if forward_step_guidance else ["instruction"],
                },
            },
            self.continue_vla,
        )

    def _register_body_motion_tools(self) -> None:
        common_properties = {
            "expected_step": {"type": "integer", "minimum": 0},
            "gripper": {
                "type": "string", "enum": ["keep", "open", "close"], "default": "keep",
                "description": (
                    "keep preserves the last confirmed native gripper command; "
                    "without history choose open or close. Pre-close motion requires open."
                ),
            },
            "max_steps": {
                "type": "integer", "minimum": 2, "maximum": 25, "default": 5,
                "description": "Total control steps including one final zero body command; 20 Hz.",
            },
        }
        for name, command_properties, description, handler in (
            (
                "move_base",
                {"velocity": {
                    "type": "array", "minItems": 3, "maxItems": 3,
                    "items": {"type": "number", "minimum": -1, "maximum": 1},
                    "description": (
                        "Native [forward, lateral, yaw] velocity commands in current robot "
                        "base axes. Positive: forward, left, counterclockwise about +Z. "
                        "Values are normalized controls, not requested displacement or angle."
                    ),
                }},
                "Directly translate or turn the mobile base using action indices 0-2; torso command is zero. ",
                self.move_base,
            ),
            (
                "adjust_torso",
                {"command": {
                    "type": "number", "minimum": -1, "maximum": 1,
                    "description": (
                        "Native torso position-increment command (index 3): positive raises, "
                        "negative lowers. Each motion step requests command * 0.05 m relative "
                        "to the achieved joint position; this is not measured travel."
                    ),
                }},
                "Directly raise or lower the torso using action index 3; base velocity commands are zero. ",
                self.adjust_torso,
            ),
        ):
            self.add_tool(name, {
                "name": name,
                "description": description + (
                    "Requires a fresh observe at expected_step. Ground a small command in "
                    "current images and public robot state when body position obstructs the "
                    "task goal, subject to the same progress/timing criteria as EEF correction. "
                    "Do not use for productive contact with the intended target; assess "
                    "task-direction progress and typical-step timing under ACTION and TASK SCOPE. "
                    "Keeps zero arm deltas in native base mode and preserves the "
                    "chosen gripper command. Use short windows and inspect clearance; no "
                    "automatic collision avoidance or target convergence. A normal window "
                    "ends with one zero body command; terminal states, grasp loss and faults "
                    "interrupt it immediately. Then observe, resolve Harness verification, "
                    "and resume_vla with the current policy instruction unchanged."
                ),
                "input_schema": {
                    "type": "object", "properties": {**common_properties, **command_properties},
                    "required": ["expected_step", *command_properties],
                },
            }, handler)

    def execute_vla(
        self,
        max_actions: int | None = None,
        instruction: str | None = None,
        *,
        intervene: bool = False,
    ) -> dict[str, Any]:
        self._sync_grasp_operation()
        self._require_motion_review_complete()
        max_actions = _validate_max_actions(max_actions)
        policy_instruction = None
        step_guidance = self._validate_policy_review(instruction, intervene)
        if self._forward_step_guidance:
            policy_instruction = self._reviewed_policy_instruction(step_guidance, intervene)
        elif instruction is not None:
            raise TypeError("execute_vla instruction is not supported by this policy")
        if self._burner_control_anchor is None:
            self._precontact_grounding = None
        limit = self._actions_per_policy_request
        if limit is not None:
            max_actions = limit if max_actions is None else min(max_actions, limit)
        max_actions = self._motion_action_budget(
            self._max_transitions_per_tool_call if max_actions is None else max_actions
        )
        result = self._operation.execute(
            max_actions=max_actions, instruction=policy_instruction,
        )
        self._update_pick_progress(result)
        if self._forward_step_guidance:
            result.update(
                instruction=policy_instruction, step_guidance=step_guidance,
                intervened=intervene,
            )
        return result

    def attach_harness(self, harness: Any) -> None:
        super().attach_harness(harness)
        self._bind_episode_horizon(harness)

    def attach_harness_session(self, session: Any, evidence_root: Path) -> None:
        super().attach_harness_session(session, evidence_root)
        self._bind_episode_horizon(session.controller)

    def _bind_episode_horizon(self, harness: Any) -> None:
        """LLM segment quotas keep horizon for the policy fallback (1.0.11 B4)."""
        state = getattr(harness, "state", None)
        bind = getattr(state, "set_episode_horizon", None)
        if callable(bind):
            bind(self._episode_horizon)
        bind_instruction = getattr(state, "set_policy_instruction", None)
        if callable(bind_instruction):
            bind_instruction(self._policy_instruction)

    def _whole_task_policy_first(self) -> bool:
        """Use the plan authority's live scope, not an inferred capability tier."""
        harness = self._harness_session.controller if self._harness_session is not None else self._harness
        state = getattr(harness, "state", None)
        return getattr(state, "whole_task_policy_first_active", False) is True

    def _grasp_review_status(self) -> dict[str, Any]:
        self._sync_grasp_operation()
        state = _robocasa_state(self._operation.observation)
        at_fixed_control = self._at_fixed_control(state)
        policy_first = self._whole_task_policy_first()
        return {
            "enabled": self._preclose_review_enabled and not policy_first,
            "active": self._preclose_active and not policy_first,
            "required": self._preclose_pending and not at_fixed_control and not policy_first,
            **({"whole_task_policy_first": True} if policy_first else {}),
            "camera": _CAMERAS[2],
            "approved_step": self._preclose_approval["step"] if self._preclose_approval else None,
            "consecutive_rejections": self._preclose_rejections,
            "max_consecutive_rejections": PRECLOSE_MAX_CONSECUTIVE_REJECTIONS,
            "policy_close_allowed": (
                self._preclose_rejections >= PRECLOSE_MAX_CONSECUTIVE_REJECTIONS
                or at_fixed_control
                or self._preclose_approval is not None
                and self._preclose_pose_is_reviewed(state)
            ),
            "fixed_control_zones": len(self._fixed_control_zones),
            "at_fixed_control": at_fixed_control,
            "evidence_scope": "LLM visual alignment assessment only; not contact or grasp success",
        }

    def _at_fixed_control(self, state: np.ndarray) -> bool:
        """Whether the hand is at a control the LLM recorded in this operation."""
        return any(
            np.linalg.norm(state[0:3] - zone["base_position"]) <= _FIXED_CONTROL_BASE_TOLERANCE_M
            and _base_rotation_within(
                zone["base_quaternion"], state[3:7], _FIXED_CONTROL_BASE_ROTATION_TOLERANCE_RAD,
            )
            and np.linalg.norm(state[7:10] - zone["position"]) <= _FIXED_CONTROL_RADIUS_M
            for zone in self._fixed_control_zones
        )

    def _preclose_pose_is_reviewed(self, state: np.ndarray) -> bool:
        approval = self._preclose_approval
        return bool(
            approval is not None
            and approval["remaining_actions"] > 0
            and np.linalg.norm(state[7:10] - approval["position"]) <= _POSITION_TOLERANCE_M
            and np.linalg.norm(_quaternion_error_rotvec(
                approval["quaternion"], state[10:14],
            )) <= _ROTATION_TOLERANCE_RAD
        )

    def _guard_grasp_close(self, observation: dict[str, Any], action: Any) -> str | None:
        """Pause the first close until its wrist pose is reviewed."""
        self.episode_budget.check()
        if self._whole_task_policy_first():
            return None
        self._sync_grasp_operation()
        if not self._preclose_active:
            return None
        if self._pick_progress is not None and self._pick_progress.confirmed:
            self._preclose_active = False
            self._preclose_pending = False
            self._preclose_approval = None
            self._preclose_rejections = 0
            # A later open is no longer tracked here; a stale closing flag
            # would report "already closed" and skip the next review.
            self._preclose_closing = False
            return None
        closing = float(action[11]) >= 0.5
        # Keeping an existing closure (a held object after a step change) is
        # not a new closure and needs no review (1.0.11 A2).
        if closing and not self._preclose_closing and self._last_gripper_command() != "close":
            state = _robocasa_state(observation)
            if (
                self._preclose_rejections < PRECLOSE_MAX_CONSECUTIVE_REJECTIONS
                and not self._preclose_pose_is_reviewed(state)
                and not self._at_fixed_control(state)
            ):
                self._preclose_pending = True
                self._preclose_approval = None
                return POLICY_REVIEW_PAUSE_REASON
        return None

    def _observe_grasp_command(self, action: Any) -> None:
        """Update review state only for confirmed policy or direct commands."""
        if not self._preclose_active or action is None:
            return
        values = np.asarray(action)
        if values.shape != (12,):
            return
        closing = float(values[11]) >= 0.5
        if closing:
            self._preclose_closing = True
            self._preclose_pending = False
            self._preclose_approval = None
            self._preclose_rejections = 0
        elif not closing and self._preclose_closing:
            self._preclose_closing = False
            self._preclose_pending = not (
                self._pick_progress is not None
                and self._pick_progress.sustained_close_command
            )
            self._preclose_approval = None
            self._preclose_rejections = 0
        elif self._preclose_approval is not None:
            self._preclose_approval["remaining_actions"] -= 1

    def review_grasp(
        self, expected_step: int, object_between_fingers: bool,
        narrow_axis_aligned: bool, evidence: str,
        closure_target: str = "movable_object", target_visible: bool = True,
    ) -> dict[str, Any]:
        """Record an explicit LLM inspection of the current public wrist image."""
        self._sync_grasp_operation()
        current_step = int(self._operation.metrics["step_count"])
        if (
            isinstance(expected_step, bool) or not isinstance(expected_step, int)
            or expected_step != current_step or self._last_observed_step != current_step
        ):
            raise ValueError("review_grasp requires a fresh observe and its exact expected_step")
        if self._preclose_closing or self._last_gripper_command() == "close":
            raise ValueError(
                "the gripper is already closed; keeping a closure needs no review "
                "(use gripper=keep). Review only before closing an open gripper."
            )
        if not isinstance(object_between_fingers, bool) or not isinstance(narrow_axis_aligned, bool):
            raise ValueError("grasp alignment checks must be booleans; unknown must be false")
        if closure_target not in _CLOSURE_TARGETS:
            raise ValueError(f"closure_target must be one of {', '.join(_CLOSURE_TARGETS)}")
        if not isinstance(target_visible, bool):
            raise ValueError("target_visible must be a boolean")
        evidence = self._validate_step_guidance(evidence)
        state = _robocasa_state(self._operation.observation)
        if np.linalg.norm(state[10:14]) <= 1e-8:
            raise ValueError("review_grasp requires a valid observed EEF quaternion")
        # A takeover's direct close opens its own review; no policy pick has to
        # activate it first (1.0.11 A1).
        self._preclose_active = True
        if closure_target == "fixed_control":
            self._operation.discard_pending_policy_actions()
            self._fixed_control_zones.append({
                "step": current_step, "position": state[7:10].copy(),
                "base_position": state[0:3].copy(), "base_quaternion": state[3:7].copy(),
            })
            self._preclose_rejections = 0
            self._preclose_pending = False
            self._preclose_approval = None
            return {
                "step": current_step, "camera": _CAMERAS[2], "approved": False,
                "closure_target": closure_target, "policy_close_allowed": True,
                "next_action": (
                    "Resolve pending verification as progress, then let the policy continue "
                    "with resume_vla or continue_vla. Closures at this control need no further "
                    "review in this operation; a direct close here is also permitted. This "
                    "records no control effect: judge that from later observations."
                ),
                "object_between_fingers": object_between_fingers,
                "narrow_axis_aligned": narrow_axis_aligned, "evidence": evidence,
                "grasp_review": self._grasp_review_status(),
            }
        if not target_visible:
            self._operation.discard_pending_policy_actions()
            # 1.2.2: an unseen target is not a misaligned one. In the 1.2.1 full eval
            # (D05) the policy paused 1.5 cm from the sugar cube; the review zoomed on
            # the wrong region, treated the occluded cube as misaligned, lifted the hand
            # and re-approached, and the later closure held only the cube's edge and
            # dropped it during the carry. 1.2.0 approved the same pose and succeeded.
            # Leave the pose alone and let the policy close; retention is checked after
            # the lift as for any closure. This is the two-rejection release (policy close
            # only, never a direct close), reached without correcting the pose.
            self._preclose_rejections = PRECLOSE_MAX_CONSECUTIVE_REJECTIONS
            self._preclose_pending = False
            self._preclose_approval = None
            return {
                "step": current_step, "camera": _CAMERAS[2], "approved": False,
                "closure_target": closure_target, "target_visible": False,
                "policy_close_allowed": True, "release_basis": "target_not_visible",
                "next_action": (
                    "The target is not visible enough to judge, which is not evidence of "
                    "misalignment: do not correct the pose. Resolve pending verification as "
                    "progress and let the policy close with smolvla_pick or resume_vla; after "
                    "the lift, verify retention from the wrist image and the measured opening. "
                    "A direct close stays unauthorized."
                ),
                "object_between_fingers": object_between_fingers,
                "narrow_axis_aligned": narrow_axis_aligned, "evidence": evidence,
                "grasp_review": self._grasp_review_status(),
            }
        approved = object_between_fingers and narrow_axis_aligned
        if not approved:
            self._operation.discard_pending_policy_actions()
        self._preclose_rejections = 0 if approved else self._preclose_rejections + 1
        policy_close_allowed = approved or self._preclose_rejections >= PRECLOSE_MAX_CONSECUTIVE_REJECTIONS
        self._preclose_pending = not policy_close_allowed
        self._preclose_approval = {
            "step": current_step, "position": state[7:10].copy(),
            "quaternion": state[10:14].copy(), "remaining_actions": _RESUME_MAX_ACTIONS,
        } if approved else None
        return {
            "step": current_step, "camera": _CAMERAS[2], "approved": approved,
            "closure_target": closure_target,
            "policy_close_allowed": policy_close_allowed,
            "next_action": (
                "Resolve pending verification, then allow a fresh policy close; alignment remains "
                "unconfirmed and a direct close stays unauthorized."
                if policy_close_allowed and not approved else
                "Resolve pending verification, then close at this reviewed pose: continue policy "
                "execution, or in a takeover call move_eef_path with gripper=close and the current "
                "position as its only waypoint."
                if approved else "Resolve pending verification, then correct and review the pose."
            ),
            "object_between_fingers": object_between_fingers,
            "narrow_axis_aligned": narrow_axis_aligned, "evidence": evidence,
            "grasp_review": self._grasp_review_status(),
        }

    def _last_gripper_command(self) -> str | None:
        """Last confirmed native gripper command (None before any command).

        Before this toolkit executes anything, defer to
        ``_resolve_gripper_command("keep")``: a restored state (fork) supplies
        its archived command there, so a held object is not mistaken for an
        open gripper.
        """
        if self._last_executed_gripper is not None:
            return self._last_executed_gripper
        try:
            return self._resolve_gripper_command("keep")[0]
        except ValueError:
            return None

    def _require_reviewed_direct_close(self, gripper: str) -> None:
        """Every direct open-to-close needs a review of the current pose (1.0.11 A1).

        Keeping an existing closure, for example a held object, needs none (A2).
        """
        self._check_task_gripper(gripper)
        harness = self._harness_session.controller if self._harness_session else self._harness
        state = getattr(harness, "state", None)
        finishing = getattr(state, "finishing_action_budget", lambda: None)()
        if finishing is not None and gripper == "close" and self._last_gripper_command() != "close":
            raise ValueError("finishing lease cannot start a new grasp after release")
        if gripper != "close" or self._preclose_closing or self._last_gripper_command() == "close":
            return
        state = _robocasa_state(self._operation.observation)
        if not self._preclose_pose_is_reviewed(state) and not self._at_fixed_control(state):
            raise ValueError(
                "closing an open gripper needs a review of the current pose: observe the "
                "wrist image and call review_grasp, then close at that pose"
            )

    def _update_pick_progress(self, result: dict[str, Any]) -> None:
        """Reconcile batch-only controllers and expose alerts before feedback."""
        environment = result.get("environment")
        if not isinstance(environment, dict):
            environment = {}
        if environment.get("terminated") or environment.get("truncated"):
            self._episode_terminal = True
        actions = self._operation.last_actions
        for index, transition in enumerate(self._operation.last_transitions):
            action = actions[index] if index < len(actions) else None
            self._observe_pick_transition(action, transition)
        self._attach_grasp_feedback(result)
        if result.get("stop_reason") in {
            "preclose_review_required", "prerelease_review_required", "task_conflict_review_required",
        }:
            metrics = self._operation.metrics
            result.setdefault("environment", {
                "step_id": int(metrics["step_count"]),
                "success": bool(metrics["simulator_success"]),
                "terminated": bool(metrics["terminated"]),
                "truncated": bool(metrics["truncated"]),
            })
            result.update(success=False, failure_reason=None, review_required=True,
                          grasp_review=self._grasp_review_status())

    def _observe_pick_transition(self, action: Any, transition: StepResult) -> str | None:
        """Run at the authoritative control boundary, before another action."""
        self._observe_task_action(action)
        if action is not None and np.asarray(action).shape == (12,):
            # Inside a policy chunk the action history lags until the chunk
            # ends; the close guard needs the command executed just now.
            self._last_executed_gripper = "close" if float(np.asarray(action)[11]) >= 0.5 else "open"
        if self._whole_task_policy_first():
            # Width-based acquisition heuristics are advisory, not authority to
            # cancel a native whole-task chunk or inject recovery movements.
            return None
        progress = self._pick_progress
        if progress is None:
            return None
        if self._pick_episode_id != transition.episode_id:
            if self._pick_episode_id is not None:
                progress.reset(np.asarray(transition.observation.get("state", ())))
                self._pending_grasp_events.clear()
                self._fixed_control_zones.clear()
                self._confirmed_grasp_step = None
            self._pick_episode_id = transition.episode_id
            self._last_pick_step = -1
        if transition.step_id <= self._last_pick_step:
            return None
        self._last_pick_step = transition.step_id
        self._observe_grasp_command(action)
        was_confirmed = progress.confirmed
        event = progress.observe(np.asarray(transition.observation.get("state", ())), action)
        if event is not None:
            event.update(
                episode_id=transition.episode_id, step=transition.step_id,
                confirmed_grasp_step=self._confirmed_grasp_step,
            )
            self._pending_grasp_events.append(event)
            self._confirmed_grasp_step = None
            return "grasp_lost"
        if progress.confirmed and not was_confirmed:
            self._confirmed_grasp_step = transition.step_id
        return None

    def _attach_grasp_feedback(self, result: dict[str, Any]) -> None:
        if self._pick_progress is not None:
            result["grasp_status"] = self._pick_progress.diagnostics()
        if not self._pending_grasp_events:
            return
        observation = self._operation.observation
        image_step = int(self._operation.metrics["step_count"])
        events = [{**event, "image_step": image_step} for event in self._pending_grasp_events]
        self._pending_grasp_events.clear()
        result.setdefault("grasp_events", []).extend(events)
        result["interruption_scope"] = (
            "control_step" if result.get("stop_reason") == "grasp_lost" else "after_chunk"
        )
        result["stop_reason"] = "grasp_lost"
        result["review_required"] = True
        # 1.2.2 (D05, 1.2.1 full eval): after this event the policy carried on, the cube
        # fell to the counter 25 cm from its source, and the recovery only searched the
        # source and the cabinet until the horizon.
        result["grasp_lost_next"] = (
            "The held object has left the hand or is slipping out (empty closure width). Do "
            "not assume it is still held, at its source or at its destination: find it in "
            "these images and fresh side views (zoom, back_project) before any further "
            "acquisition, carry or placement. If it still hangs at the fingertips, do not "
            "carry it on; set it down or re-grasp it where it is."
        )
        for camera, key in zip(_CAMERAS, ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes")):
            if camera in observation:
                result[key] = _png_bytes(observation[camera])
        result["_image_camera_ids"] = {
            camera: key for camera, key in zip(
                _CAMERAS, ("_image_bytes", "_image_right_bytes", "_image_wrist_bytes")
            ) if key in result
        }

    def back_project(
        self,
        camera: str,
        expected_step: int,
        points: list[list[int]],
    ) -> dict[str, Any]:
        """Localize selected current pixels without executing a physical action."""

        if camera not in _CAMERAS:
            raise ValueError(f"not a planner observation camera: {camera!r}")
        if self._perception_source is None:
            raise RuntimeError("RoboCasa perception source is not configured")
        if isinstance(expected_step, bool) or not isinstance(expected_step, int):
            raise ValueError("expected_step must be an integer")
        current_step = int(self._operation.metrics["step_count"])
        if expected_step != current_step:
            raise ValueError(
                f"expected_step {expected_step} does not match current observation "
                f"step {current_step}"
            )
        frame = self._perception_source.rgbd(camera)
        result = localize_rgbd_points(
            frame,
            self._operation.observation["state"],
            expected_step=expected_step,
            points=points,
        )
        state = _robocasa_state(self._operation.observation)
        surface = np.asarray(result["relative_base_xyz"], dtype=np.float64)
        current_position = np.asarray(state[7:10], dtype=np.float64)
        retreat = current_position - surface
        retreat_norm = float(np.linalg.norm(retreat))
        if retreat_norm <= 1e-8:
            raise ValueError(
                "cannot derive a standoff direction from an end effector on the "
                "measured surface"
            )
        standoff = surface + (
            retreat / retreat_norm * _PERCEPTION_STANDOFF_DISTANCE_M
        )
        quaternion = np.asarray(state[10:14], dtype=np.float64)
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1e-8:
            raise ValueError("cannot derive standoff from an invalid EEF quaternion")
        quaternion /= quaternion_norm
        result.update(
            recommended_standoff_xyz=standoff.tolist(),
            recommended_standoff_quaternion=quaternion.tolist(),
            recommended_standoff_distance_m=_PERCEPTION_STANDOFF_DISTANCE_M,
        )
        episode_id = frame.get("episode_id")
        if episode_id is not None:
            result["episode_id"] = episode_id
        self._precontact_grounding = {
            "source_step": expected_step,
            "standoff_position": standoff,
            "standoff_quaternion": quaternion,
            "servo_step": None,
            "observed": False,
        }
        return result

    def _guided_instruction(self, instruction: str) -> str:
        guidance = self._validate_step_guidance(instruction)
        return f"{self._policy_instruction}\nCurrent action-step guidance: {guidance}"

    def _validate_policy_review(
        self, instruction: str | None, intervene: bool,
    ) -> str | None:
        if not isinstance(intervene, bool):
            raise ValueError("intervene must be a boolean")
        harness = self._harness_session.controller if self._harness_session else self._harness
        semantic = getattr(getattr(harness, "state", None), "semantic_policy_replanning_allowed", False)
        if self._whole_task_policy_first() and not semantic and intervene and instruction != self._policy_instruction:
            raise ValueError("whole-task monitor must preserve the complete original instruction")
        if intervene and not self._forward_step_guidance:
            raise ValueError("this policy does not support instruction intervention")
        if instruction is None:
            if intervene:
                raise ValueError("instruction is required when intervene=true")
            return None
        return self._validate_step_guidance(instruction)

    def _reviewed_policy_instruction(
        self, instruction: str | None, intervene: bool,
    ) -> str:
        """Persist an explicit review decision across physical tool calls."""
        if intervene:
            assert instruction is not None
            if instruction != self._active_policy_instruction:
                self._reset_grasp_operation()
            self._active_policy_instruction = instruction
        return self._active_policy_instruction

    def _reset_grasp_operation(self) -> None:
        """Start fresh acquisition evidence without resetting policy or physics.

        A closure that is still commanded survives the step change: a held
        object keeps its retention watch (grasp-loss alerts and the holding
        speed limit of privileged motion) and needs no new review (1.0.11 A2).
        """
        self._operation.discard_pending_policy_actions()
        closed = self._last_gripper_command() == "close"
        self._preclose_pending = False
        self._preclose_closing = closed and self._preclose_active
        self._preclose_approval = None
        self._preclose_rejections = 0
        self._fixed_control_zones.clear()
        self._preclose_recovery_in_progress = False
        self._consecutive_closed_empty = 0
        self._confirmed_grasp_step = None
        if self._pick_progress is not None:
            retained = closed and self._pick_progress.retention_monitor_active
            self._pick_progress.reset(_robocasa_state(self._operation.observation))
            self._pick_progress.retention_monitor_active = retained

    def _sync_grasp_operation(self) -> None:
        """Use the existing plan authority in both legacy and strict sessions."""
        harness = (
            self._harness_session.controller
            if self._harness_session is not None else self._harness
        )
        active = harness.state.active_step if harness is not None else None
        if active is None:
            return
        key = (active.id, active.intent)
        if self._grasp_operation_key is not None and key != self._grasp_operation_key:
            self._reset_grasp_operation()
        self._grasp_operation_key = key

    @staticmethod
    def _validate_step_guidance(instruction: str) -> str:
        if not isinstance(instruction, str):
            raise ValueError("instruction must be a string")
        guidance = instruction.strip()
        if not guidance:
            raise ValueError("instruction must not be empty")
        if len(guidance) > _MAX_STEP_GUIDANCE_LENGTH:
            raise ValueError(
                f"instruction must contain at most {_MAX_STEP_GUIDANCE_LENGTH} characters"
            )
        return guidance

    def _execute_vla_until(
        self,
        *,
        instruction: str | None,
        max_chunks: int,
        checker: Callable[[dict[str, Any], Any | None], bool],
        failure_checker: (
            Callable[[dict[str, Any], Any | None], str | None] | None
        ) = None,
        guide_policy: bool = False,
        intervene: bool = False,
        transition_budget: int | None = None,
    ) -> dict[str, Any]:
        self._sync_grasp_operation()
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or not 1 <= max_chunks <= _MAX_CHUNKS
        ):
            raise ValueError(f"max_chunks must be between 1 and {_MAX_CHUNKS}")
        step_guidance = self._validate_policy_review(instruction, intervene)
        effective_transition_budget = (
            self._max_transitions_per_tool_call
            if transition_budget is None
            else min(transition_budget, self._max_transitions_per_tool_call)
        )
        if self._forward_step_guidance:
            effective_instruction = self._reviewed_policy_instruction(step_guidance, intervene)
        elif guide_policy:
            assert step_guidance is not None
            effective_instruction = self._guided_instruction(step_guidance)
        else:
            effective_instruction = self._policy_instruction
        requested_actions = 0
        executed_actions = 0
        cancelled_actions = 0
        paused_actions = 0
        resumed_actions = 0
        environment = None

        for chunk_index in range(max_chunks):
            remaining_transitions = (
                effective_transition_budget - executed_actions
            )
            if remaining_transitions <= 0:
                break
            request_limit = remaining_transitions
            if self._actions_per_policy_request is not None:
                request_limit = min(request_limit, self._actions_per_policy_request)
            try:
                reviewed_action = self._take_reviewed_release()
                result = (
                    self._operation.execute_reviewed_policy_action(reviewed_action, effective_instruction)
                    if reviewed_action is not None else
                    self._operation.execute(max_actions=request_limit, instruction=effective_instruction)
                )
            except Exception as exc:
                # Earlier chunks moved the robot; keep them countable.
                exc.rpent_executed_actions = executed_actions
                raise
            self._update_pick_progress(result)
            if result.get("stop_reason") == "prerelease_review_required" and self._task_release_candidate:
                self._task_release_candidate["step"] = int(self._operation.metrics["step_count"])
            requested_actions += int(result["requested_actions"])
            executed_actions += int(result["executed_actions"])
            cancelled_actions += int(result["cancelled_actions"])
            paused_actions += int(result.get("paused_actions", 0))
            resumed_actions += int(result.get("resumed_actions", 0))
            environment = result.get("environment")
            common = {
                "instruction": effective_instruction,
                "step_guidance": step_guidance,
                **({"intervened": intervene} if self._forward_step_guidance else {}),
                "chunks_used": chunk_index + 1,
                "max_chunks": max_chunks,
                "requested_actions": requested_actions,
                "executed_actions": executed_actions,
                "cancelled_actions": cancelled_actions,
                "paused_actions": paused_actions,
                "resumed_actions": resumed_actions,
                "transitions_used": executed_actions,
                "transition_budget": effective_transition_budget,
                **{key: value for key, value in result.items() if key in _GRASP_FEEDBACK_KEYS},
            }
            if environment is not None:
                common["environment"] = environment
                if environment["success"]:
                    return self._finish_repeated_vla_result(
                        common,
                        success=True,
                        failure_reason=None,
                        stop_reason="success",
                    )
                if environment["terminated"] or environment["truncated"]:
                    return self._finish_repeated_vla_result(
                        common,
                        success=False,
                        failure_reason="episode_terminal",
                        stop_reason="episode_terminal",
                    )
            if result.get("grasp_events"):
                return self._finish_repeated_vla_result(
                    common, success=False, failure_reason="grasp_lost", stop_reason="grasp_lost",
                )
            if result.get("stop_reason") == "preclose_review_required":
                common.update(review_required=True, grasp_review=self._grasp_review_status())
                return self._finish_repeated_vla_result(
                    common, success=False, failure_reason=None, stop_reason="preclose_review_required",
                )
            if result.get("stop_reason") in {"task_conflict_review_required", "prerelease_review_required"}:
                common.update(review_required=True, task_action_review=self._task_review_snapshot())
                return self._finish_repeated_vla_result(
                    common, success=False, failure_reason=None, stop_reason=result["stop_reason"],
                )
            if reviewed_action is not None:
                return self._finish_repeated_vla_result(
                    common, success=False, failure_reason=None, stop_reason="reviewed_release_executed",
                )
            actions = self._operation.last_actions
            for observation_index, observation in enumerate(
                self._operation.last_observations
            ):
                action = (
                    actions[observation_index]
                    if observation_index < len(actions)
                    else None
                )
                if checker(observation, action):
                    return self._finish_repeated_vla_result(
                        common,
                        success=True,
                        failure_reason=None,
                        stop_reason="success",
                    )
                if failure_checker is not None:
                    failure_reason = failure_checker(observation, action)
                    if failure_reason is not None:
                        return self._finish_repeated_vla_result(
                            common,
                            success=False,
                            failure_reason=failure_reason,
                            stop_reason=failure_reason,
                        )
            if executed_actions >= effective_transition_budget:
                return self._finish_repeated_vla_result(
                    common,
                    success=False,
                    failure_reason="budget_exhausted",
                    stop_reason="transition_budget_exhausted",
                )

        result = {
            "instruction": effective_instruction,
            "step_guidance": step_guidance,
            **({"intervened": intervene} if self._forward_step_guidance else {}),
            "success": False,
            "failure_reason": "budget_exhausted",
            "chunks_used": max_chunks,
            "max_chunks": max_chunks,
            "requested_actions": requested_actions,
            "executed_actions": executed_actions,
            "cancelled_actions": cancelled_actions,
            "paused_actions": paused_actions,
            "resumed_actions": resumed_actions,
            "transitions_used": executed_actions,
            "transition_budget": effective_transition_budget,
        }
        if environment is not None:
            result["environment"] = environment
        return self._finish_repeated_vla_result(
            result,
            success=False,
            failure_reason="budget_exhausted",
            stop_reason="max_chunks_exhausted",
        )

    def _finish_repeated_vla_result(
        self,
        result: dict[str, Any],
        *,
        success: bool,
        failure_reason: str | None,
        stop_reason: str,
    ) -> dict[str, Any]:
        state = _robocasa_state(self._operation.observation)
        environment = result.get("environment")
        step = (
            environment.get("step_id")
            if isinstance(environment, dict)
            else self._operation.metrics["step_count"]
        )
        return {
            **result,
            "success": success,
            "failure_reason": failure_reason,
            "stop_reason": stop_reason,
            "observation_summary": {
                "step": step,
                "eef_z": float(state[9]),
                "gripper_opening": float(np.abs(state[14:16]).sum()),
            },
        }

    def smolvla_pick(
        self,
        instruction: str | None = None,
        *,
        intervene: bool = False,
        max_chunks: int = _DEFAULT_MAX_CHUNKS,
        descent_threshold_m: float = 0.05,
        ascent_threshold_m: float = 0.05,
        closed_threshold: float = 0.03,
        empty_threshold: float = 0.002,
    ) -> dict[str, Any]:
        """Run repeated policy chunks until proprioception supports a grasp."""
        self._sync_grasp_operation()
        self._require_motion_review_complete()
        if self._burner_control_anchor is None:
            self._precontact_grounding = None
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or not 1 <= max_chunks <= _MAX_CHUNKS
        ):
            raise ValueError(f"max_chunks must be between 1 and {_MAX_CHUNKS}")
        step_guidance = self._validate_policy_review(instruction, intervene)
        if not self._forward_step_guidance and step_guidance is None:
            raise ValueError("instruction is required for this policy")
        initial_state = np.asarray(self._operation.observation.get("state", ()))
        if initial_state.shape != (16,):
            raise ValueError("RoboCasa pick requires a 16D state")
        thresholds = (
            descent_threshold_m,
            ascent_threshold_m,
            closed_threshold,
            empty_threshold,
            _OPEN_GRIPPER_THRESHOLD,
        )
        progress = self._pick_progress
        if progress is None or progress.thresholds != thresholds:
            progress = GraspProgress(
                initial_state,
                descent_threshold_m=descent_threshold_m,
                ascent_threshold_m=ascent_threshold_m,
                closed_threshold=closed_threshold,
                empty_threshold=empty_threshold,
                open_threshold=_OPEN_GRIPPER_THRESHOLD,
            )
            self._pick_progress = progress
        partial_closure_state: np.ndarray | None = None
        partial_closure_run = 0
        is_closed_empty_recovery = self._consecutive_closed_empty == 1
        if is_closed_empty_recovery and not self._preclose_recovery_in_progress:
            progress.reset(initial_state)
        diagnostics: dict[str, Any] = {**progress.diagnostics(), "recovery_pose": None}
        blocked_reason = None
        if self._episode_terminal:
            blocked_reason = "episode_terminal"
        elif self._consecutive_closed_empty >= 2:
            blocked_reason = "closed_empty_retry_exhausted"
        if blocked_reason is not None:
            return {
                "instruction": self._active_policy_instruction,
                "step_guidance": step_guidance,
                **({"intervened": False} if self._forward_step_guidance else {}),
                "success": False,
                "failure_reason": (
                    "closed_empty"
                    if blocked_reason == "closed_empty_retry_exhausted"
                    else blocked_reason
                ),
                "stop_reason": blocked_reason,
                "retry_allowed": False,
                "chunks_used": 0,
                "max_chunks": max_chunks,
                "requested_actions": 0,
                "executed_actions": 0,
                "cancelled_actions": 0,
                "transitions_used": 0,
                "transition_budget": self._max_transitions_per_tool_call,
                "observation_summary": {
                    "step": self._operation.metrics["step_count"],
                    "eef_z": float(initial_state[9]),
                    "gripper_opening": float(np.abs(initial_state[14:16]).sum()),
                    "grasp": dict(diagnostics),
                },
                "diagnostics": diagnostics,
            }

        if self._forward_step_guidance:
            self._reviewed_policy_instruction(step_guidance, intervene)
            is_closed_empty_recovery = self._consecutive_closed_empty == 1
            diagnostics.update(progress.diagnostics())

        def grasp_confirmed(
            observation: dict[str, Any],
            action: Any | None,
        ) -> bool:
            nonlocal partial_closure_run
            nonlocal partial_closure_state
            state = np.asarray(observation.get("state", ()))
            if state.shape != (16,):
                raise ValueError("RoboCasa pick requires a 16D state")
            final_opening = float(np.abs(state[14:16]).sum())
            if progress.descent_confirmed:
                if (
                    is_closed_empty_recovery
                    and not progress.empty_closure_seen
                    and action is not None
                    and np.asarray(action).shape == (12,)
                    and float(np.asarray(action)[11]) > 0.0
                ):
                    partial_closure_run += 1
                    if (
                        partial_closure_run >= _RECOVERY_CLOSE_INTENT_STEPS
                        and closed_threshold < final_opening <= _OPEN_GRIPPER_THRESHOLD
                    ):
                        partial_closure_state = state.copy()
                else:
                    partial_closure_run = 0
            diagnostics.update(progress.diagnostics())
            return progress.confirmed

        def empty_grasp_confirmed(
            _observation: dict[str, Any],
            _action: Any | None,
        ) -> str | None:
            if progress.closed_empty:
                return "closed_empty"
            if partial_closure_state is not None:
                current_state = _robocasa_state(self._operation.observation)
                position_delta = np.linalg.norm(
                    current_state[7:10] - partial_closure_state[7:10]
                )
                rotation_delta = np.linalg.norm(
                    _quaternion_error_rotvec(
                        current_state[10:14],
                        partial_closure_state[10:14],
                    )
                )
                if (
                    position_delta <= _RECOVERY_CANDIDATE_POSITION_TOLERANCE_M
                    and rotation_delta
                    <= _RECOVERY_CANDIDATE_ROTATION_TOLERANCE_RAD
                ):
                    return "partial_closure"
            return None

        if progress.confirmed or progress.closed_empty:
            result = self._finish_repeated_vla_result(
                {
                    "instruction": self._active_policy_instruction,
                    "step_guidance": step_guidance,
                    **({"intervened": False} if self._forward_step_guidance else {}),
                    "chunks_used": 0,
                    "max_chunks": max_chunks,
                    "requested_actions": 0,
                    "executed_actions": 0,
                    "cancelled_actions": 0,
                    "transitions_used": 0,
                    "transition_budget": self._max_transitions_per_tool_call,
                },
                success=progress.confirmed,
                failure_reason=None if progress.confirmed else "closed_empty",
                stop_reason="grasp_already_confirmed" if progress.confirmed else "closed_empty",
            )
        else:
            if self._preclose_review_enabled and not self._preclose_active:
                self._preclose_active = True
                self._preclose_closing = False
            result = self._execute_vla_until(
                instruction=instruction,
                max_chunks=max_chunks,
                checker=grasp_confirmed,
                failure_checker=empty_grasp_confirmed,
                guide_policy=is_closed_empty_recovery,
                intervene=intervene,
                transition_budget=(
                    max(
                        1,
                        self._max_transitions_per_tool_call
                        - _RECOVERY_CLOSE_MAX_STEPS
                        - _RECOVERY_LIFT_MAX_STEPS,
                    )
                    if is_closed_empty_recovery
                    else None
                ),
            )
        diagnostics.update(progress.diagnostics())
        closure_state = progress.closure_state
        recovery_pose_source = "first_confirmed_closure_pose"
        if (
            result["failure_reason"] == "partial_closure"
            and partial_closure_state is not None
        ):
            result, closure_state = self._persist_recovery_close_and_lift(
                result,
                candidate_state=partial_closure_state,
                diagnostics=diagnostics,
                min_z=progress.min_z,
                ascent_threshold_m=ascent_threshold_m,
                closed_threshold=closed_threshold,
                empty_threshold=empty_threshold,
            )
            diagnostics.update(progress.diagnostics())
            environment = result.get("environment", {})
            if result["success"] and not (
                environment.get("success") or progress.confirmed
            ):
                result = self._finish_repeated_vla_result(
                    result,
                    success=False,
                    failure_reason="recovery_lift_exhausted",
                    stop_reason="recovery_lift_exhausted",
                )
            recovery_pose_source = "policy_partial_closure_pose"
        diagnostics["recovery_pose"] = (
            {
                "position": closure_state[7:10].tolist(),
                "quaternion": closure_state[10:14].tolist(),
                "source": recovery_pose_source,
            }
            if (
                result["failure_reason"] != "closed_empty"
                and diagnostics["descent_confirmed"]
                and diagnostics["closure_confirmed"]
                and closure_state is not None
            )
            else None
        )
        environment = result.get("environment")
        if isinstance(environment, dict) and (
            environment.get("terminated") or environment.get("truncated")
        ):
            self._episode_terminal = True
        if result.get("stop_reason") == "preclose_review_required":
            self._preclose_recovery_in_progress = is_closed_empty_recovery
            result["retry_allowed"] = self._consecutive_closed_empty < 2
        elif result["failure_reason"] == "closed_empty":
            self._preclose_recovery_in_progress = False
            self._consecutive_closed_empty += 1
            result["retry_allowed"] = self._consecutive_closed_empty == 1
            if not result["retry_allowed"]:
                result["stop_reason"] = "closed_empty_retry_exhausted"
        elif is_closed_empty_recovery and not result["success"]:
            self._preclose_recovery_in_progress = False
            self._consecutive_closed_empty = 2
            result["retry_allowed"] = False
        else:
            self._preclose_recovery_in_progress = False
            self._consecutive_closed_empty = 0
            result["retry_allowed"] = not self._episode_terminal
        result["diagnostics"] = diagnostics
        result["observation_summary"]["grasp"] = dict(diagnostics)
        return result

    def _persist_recovery_close_and_lift(
        self,
        result: dict[str, Any],
        *,
        candidate_state: np.ndarray,
        diagnostics: dict[str, Any],
        min_z: float,
        ascent_threshold_m: float,
        closed_threshold: float,
        empty_threshold: float,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        """Boundedly finish a policy-initiated close pulse, then verify a lift."""
        remaining = self._max_transitions_per_tool_call - int(
            result["executed_actions"]
        )
        if remaining < 1:
            return result, None
        close_result = self._policy_owned_servo_eef(
            target_position=candidate_state[7:10].tolist(),
            target_quaternion=candidate_state[10:14].tolist(),
            gripper="close",
            max_steps=min(_RECOVERY_CLOSE_MAX_STEPS, remaining),
            settle_steps=2,
            _closed_threshold=closed_threshold,
        )
        result = _merge_physical_result(result, close_result)
        result["transition_budget"] = self._max_transitions_per_tool_call
        close_state = _robocasa_state(self._operation.observation).copy()
        close_opening = float(np.abs(close_state[14:16]).sum())
        diagnostics.update(
            final_gripper_opening=close_opening,
            closure_confirmed=(
                empty_threshold < close_opening <= closed_threshold
            ),
        )
        result["recovery_close_persistence"] = {
            "trigger": "five_consecutive_policy_close_actions",
            "candidate_position": candidate_state[7:10].tolist(),
            "candidate_quaternion": candidate_state[10:14].tolist(),
            "candidate_gripper_opening": float(
                np.abs(candidate_state[14:16]).sum()
            ),
            "close": {
                "target_reached": bool(close_result["target_reached"]),
                "executed_actions": int(close_result["executed_actions"]),
                "final_gripper_opening": close_opening,
            },
            "lift": None,
        }
        environment = close_result.get("environment")
        if isinstance(environment, dict) and (
            environment["success"]
            or environment["terminated"]
            or environment["truncated"]
        ):
            return self._finish_repeated_vla_result(
                result,
                success=bool(environment["success"]),
                failure_reason=(None if environment["success"] else "episode_terminal"),
                stop_reason=("success" if environment["success"] else "episode_terminal"),
            ), close_state if environment["success"] else None
        if close_result.get("grasp_events"):
            return self._finish_repeated_vla_result(
                result, success=False, failure_reason="grasp_lost", stop_reason="grasp_lost",
            ), None
        if close_opening <= empty_threshold:
            return self._finish_repeated_vla_result(
                result,
                success=False,
                failure_reason="closed_empty",
                stop_reason="closed_empty",
            ), None
        if not close_result["target_reached"] or close_opening > closed_threshold:
            return self._finish_repeated_vla_result(
                result,
                success=False,
                failure_reason="recovery_closure_exhausted",
                stop_reason="recovery_closure_exhausted",
            ), None

        remaining = self._max_transitions_per_tool_call - int(
            result["executed_actions"]
        )
        if remaining < 1:
            return self._finish_repeated_vla_result(
                result,
                success=False,
                failure_reason="recovery_lift_exhausted",
                stop_reason="recovery_lift_exhausted",
            ), close_state
        lift_position = close_state[7:10].copy()
        lift_position[2] += _RECOVERY_LIFT_M
        lift_result = self.servo_eef(
            target_position=lift_position.tolist(),
            target_quaternion=close_state[10:14].tolist(),
            gripper="close",
            max_steps=min(_RECOVERY_LIFT_MAX_STEPS, remaining),
            settle_steps=2,
            _closed_threshold=closed_threshold,
        )
        result = _merge_physical_result(result, lift_result)
        result["transition_budget"] = self._max_transitions_per_tool_call
        final_state = _robocasa_state(self._operation.observation)
        final_opening = float(np.abs(final_state[14:16]).sum())
        ascent = float(final_state[9]) - min_z
        diagnostics.update(
            post_descent_peak_eef_z=float(final_state[9]),
            final_gripper_opening=final_opening,
            post_descent_ascent_m=ascent,
            closure_confirmed=(
                empty_threshold < final_opening <= closed_threshold
            ),
        )
        result["recovery_close_persistence"]["lift"] = {
            "target_reached": bool(lift_result["target_reached"]),
            "executed_actions": int(lift_result["executed_actions"]),
            "final_gripper_opening": final_opening,
            "post_descent_ascent_m": ascent,
        }
        environment = lift_result.get("environment")
        if lift_result.get("grasp_events") and not (environment or {}).get("success"):
            return self._finish_repeated_vla_result(
                result, success=False, failure_reason="grasp_lost", stop_reason="grasp_lost",
            ), None
        if isinstance(environment, dict) and (
            not environment["success"]
            and (environment["terminated"] or environment["truncated"])
        ):
            return self._finish_repeated_vla_result(
                result,
                success=False,
                failure_reason="episode_terminal",
                stop_reason="episode_terminal",
            ), None
        success = bool(
            (isinstance(environment, dict) and environment["success"])
            or (
                lift_result["target_reached"]
                and diagnostics["closure_confirmed"]
                and ascent >= ascent_threshold_m
            )
        )
        if success:
            return self._finish_repeated_vla_result(
                result,
                success=True,
                failure_reason=None,
                stop_reason="recovery_close_persisted",
            ), close_state
        failure_reason = (
            "closed_empty"
            if final_opening <= empty_threshold
            else "recovery_lift_exhausted"
        )
        return self._finish_repeated_vla_result(
            result,
            success=False,
            failure_reason=failure_reason,
            stop_reason=failure_reason,
        ), None

    def servo_eef(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        gripper: str,
        *,
        max_steps: int = 25,
        settle_steps: int = 1,
        _closed_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Servo to an explicit relative-base pose using public proprioception."""
        self._require_reviewed_direct_close(gripper)
        return self._policy_owned_servo_eef(
            target_position, target_quaternion, gripper,
            max_steps=max_steps, settle_steps=settle_steps,
            _closed_threshold=_closed_threshold,
        )

    def _policy_owned_servo_eef(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        gripper: str,
        *,
        max_steps: int = 25,
        settle_steps: int = 1,
        _closed_threshold: float | None = None,
    ) -> dict[str, Any]:
        """servo_eef for toolkit-owned motions: a recovery that persists the
        policy's own close pulse is not a direct LLM closure and is not reviewed."""
        self._require_motion_review_complete()
        return self._servo_eef(
            target_position, target_quaternion, gripper,
            max_steps=max_steps, settle_steps=settle_steps,
            _closed_threshold=_closed_threshold,
        )

    def _servo_eef(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        gripper: str,
        *,
        max_steps: int,
        settle_steps: int = 1,
        _closed_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Shared servo implementation; callers own motion-review preconditions."""
        position = _finite_vector(target_position, size=3, label="target_position")
        quaternion = _finite_vector(
            target_quaternion,
            size=4,
            label="target_quaternion",
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1e-8:
            raise ValueError("target_quaternion must have non-zero norm")
        quaternion /= quaternion_norm
        if gripper not in {"open", "close"}:
            raise ValueError("gripper must be 'open' or 'close'")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise ValueError("max_steps must be an integer")
        if not 1 <= max_steps <= 25:
            raise ValueError("max_steps must be between 1 and 25")
        if isinstance(settle_steps, bool) or not isinstance(settle_steps, int):
            raise ValueError("settle_steps must be an integer")
        if not 1 <= settle_steps <= min(20, max_steps):
            raise ValueError("settle_steps must be between 1 and max_steps")
        max_steps = self._motion_action_budget(max_steps)

        grounding = (
            self._precontact_grounding
            if self._burner_control_anchor is None
            else None
        )
        grounding_servo = bool(
            grounding is not None
            and int(self._operation.metrics["step_count"])
            == int(grounding["source_step"])
            and np.linalg.norm(position - grounding["standoff_position"])
            <= _POSITION_TOLERANCE_M
            and np.linalg.norm(
                _quaternion_error_rotvec(
                    quaternion,
                    grounding["standoff_quaternion"],
                )
            )
            <= _ROTATION_TOLERANCE_RAD
            and gripper == "open"
        )
        if grounding is not None and not grounding_servo:
            self._precontact_grounding = None

        requested_actions = 0
        executed_actions = 0
        cancelled_actions = 0
        environment = None
        settled = 0
        grasp_feedback: dict[str, Any] = {}
        for _ in range(max_steps):
            state = _robocasa_state(self._operation.observation)
            position_error = position - state[7:10]
            rotation_error = _quaternion_error_rotvec(state[10:14], quaternion)
            action = np.zeros(12, dtype=np.float32)
            action[4] = -1.0
            action[5:8] = np.clip(
                position_error / _POSITION_ACTION_SCALE_M,
                -1.0,
                1.0,
            )
            action[8:11] = np.clip(
                rotation_error / _ROTATION_ACTION_SCALE_RAD,
                -1.0,
                1.0,
            )
            action[11] = -1.0 if gripper == "open" else 1.0
            step_result = self._operation.execute_actions((action,))
            self._update_pick_progress(step_result)
            requested_actions += int(step_result["requested_actions"])
            executed_actions += int(step_result["executed_actions"])
            cancelled_actions += int(step_result["cancelled_actions"])
            environment = step_result.get("environment")
            if step_result.get("grasp_events"):
                grasp_feedback = {key: value for key, value in step_result.items() if key in _GRASP_FEEDBACK_KEYS}

            state = _robocasa_state(self._operation.observation)
            position_error = position - state[7:10]
            rotation_error = _quaternion_error_rotvec(state[10:14], quaternion)
            pose_reached = bool(
                np.linalg.norm(position_error) <= _POSITION_TOLERANCE_M
                and np.linalg.norm(rotation_error) <= _ROTATION_TOLERANCE_RAD
            )
            gripper_reached = bool(
                (
                    _closed_threshold is None
                    or np.abs(state[14:16]).sum() <= _closed_threshold
                )
                if gripper == "close"
                else np.abs(state[14:16]).sum() >= _OPEN_GRIPPER_THRESHOLD
            )
            settled = settled + 1 if pose_reached and gripper_reached else 0
            if environment is not None and (
                environment["success"]
                or environment["terminated"]
                or environment["truncated"]
            ):
                break
            if settled >= settle_steps:
                break
            if grasp_feedback:
                break

        state = _robocasa_state(self._operation.observation)
        position_error = position - state[7:10]
        rotation_error = _quaternion_error_rotvec(state[10:14], quaternion)
        result: dict[str, Any] = {
            **grasp_feedback,
            "target_reached": bool(
                np.linalg.norm(position_error) <= _POSITION_TOLERANCE_M
                and np.linalg.norm(rotation_error) <= _ROTATION_TOLERANCE_RAD
                and (
                    (
                        _closed_threshold is None
                        or np.abs(state[14:16]).sum() <= _closed_threshold
                    )
                    if gripper == "close"
                    else np.abs(state[14:16]).sum() >= _OPEN_GRIPPER_THRESHOLD
                )
            ),
            "settled_steps": settled,
            "requested_actions": requested_actions,
            "executed_actions": executed_actions,
            "cancelled_actions": cancelled_actions,
            "final_position": state[7:10].tolist(),
            "final_quaternion": state[10:14].tolist(),
            "final_gripper_opening": float(np.abs(state[14:16]).sum()),
            "position_error_m": float(np.linalg.norm(position_error)),
            "rotation_error_rad": float(np.linalg.norm(rotation_error)),
        }
        if environment is not None:
            result["environment"] = environment
        if grasp_feedback:
            result["stop_reason"] = "grasp_lost"
        if grounding_servo and grounding is not None:
            if result["target_reached"]:
                grounding["servo_step"] = int(
                    self._operation.metrics["step_count"]
                )
                grounding["observed"] = False
            else:
                self._precontact_grounding = None
        return result

    def continue_vla(
        self,
        instruction: str | None = None,
        *,
        intervene: bool = False,
        max_chunks: int = _DEFAULT_MAX_CHUNKS,
        step_intent: str = "task_progress",
        policy_phase: str | None = None,
    ) -> dict[str, Any]:
        """Continue one physical intent until official success or budget exhaustion."""
        self._require_motion_review_complete()
        if policy_phase is not None:
            raise ValueError(
                "checkpoint does not support policy phase overrides: "
                f"{self._policy_instruction!r} -> {policy_phase!r}"
            )
        if (
            step_intent == "burner_control"
            and self._policy_instruction not in _BURNER_CONTROL_POLICY_TASKS
        ):
            raise ValueError(
                f"checkpoint task {self._policy_instruction!r} does not support "
                "step_intent='burner_control'"
            )
        if step_intent not in self._continue_vla_step_intents:
            raise ValueError(
                "step_intent must be one of "
                f"{', '.join(self._continue_vla_step_intents)}"
            )
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or not 1 <= max_chunks <= _MAX_CHUNKS
        ):
            raise ValueError(f"max_chunks must be between 1 and {_MAX_CHUNKS}")
        step_guidance = self._validate_policy_review(instruction, intervene)
        if not self._forward_step_guidance and step_guidance is None:
            raise ValueError("instruction is required for this policy")
        if step_intent == "task_progress":
            self._precontact_grounding = None
            self._burner_control_anchor = None
            self._burner_control_start_z = None
            self._burner_control_peak_z = None
            self._burner_control_policy_actions = 0
            self._burner_control_close_run = 0
            self._burner_control_max_close_run = 0
            result = self._execute_vla_until(
                instruction=step_guidance,
                max_chunks=max_chunks,
                checker=lambda _observation, _action: False,
                intervene=intervene,
            )
            result["step_intent"] = step_intent
            return result

        if self._perception_source is not None and self._burner_control_anchor is None:
            grounding = self._precontact_grounding
            if grounding is None:
                raise ValueError(
                    "burner_control requires current RGB-D grounding before contact"
                )
            servo_step = grounding["servo_step"]
            if servo_step is None:
                raise ValueError(
                    "burner_control requires the recommended servo_eef standoff"
                )
            current_step = int(self._operation.metrics["step_count"])
            if current_step != int(servo_step):
                self._precontact_grounding = None
                raise ValueError(
                    "burner_control requires new RGB-D grounding after another "
                    "physical transition"
                )
            if not grounding["observed"]:
                raise ValueError(
                    "burner_control requires a fresh observe after the servo_eef "
                    "standoff"
                )

        policy_transition_budget = self._burner_control_policy_transition_budget(
            max_chunks=max_chunks,
            available_transitions=self._max_transitions_per_tool_call,
        )
        if policy_transition_budget <= 0:
            raise ValueError(
                "burner_control requires enough policy requests to drain one "
                f"{CHECKPOINT_CHUNK_SIZE}-action checkpoint chunk"
            )

        if self._burner_control_anchor is None:
            state = _robocasa_state(self._operation.observation)
            if float(np.abs(state[14:16]).sum()) < _OPEN_GRIPPER_THRESHOLD:
                raise ValueError("burner_control requires an open gripper")
            quaternion = np.array(state[10:14], copy=True)
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm <= 1e-8:
                raise ValueError("burner_control requires a valid EEF quaternion")
            quaternion /= quaternion_norm
            self._burner_control_anchor = (
                np.array(state[7:10], copy=True),
                quaternion,
            )
            self._burner_control_start_z = float(state[9])
            self._burner_control_peak_z = float(state[9])

        approach_guard: dict[str, int | float] = {
            "policy_actions": self._burner_control_policy_actions,
            "max_close_run": self._burner_control_max_close_run,
            "eef_descent_m": 0.0,
            "peak_descent_m": 0.0,
        }

        def approach_not_converging(
            observation: dict[str, Any],
            action: Any | None,
        ) -> str | None:
            state = _robocasa_state(observation)
            self._burner_control_policy_actions += 1
            close_commanded = bool(
                action is not None
                and np.asarray(action).shape == (12,)
                and float(np.asarray(action)[11]) > 0.0
            )
            self._burner_control_close_run = (
                self._burner_control_close_run + 1 if close_commanded else 0
            )
            self._burner_control_max_close_run = max(
                self._burner_control_max_close_run,
                self._burner_control_close_run,
            )
            start_z = self._burner_control_start_z
            if start_z is None:
                raise RuntimeError("burner control attempt has no start EEF height")
            current_z = float(state[9])
            peak_z = self._burner_control_peak_z
            if peak_z is None:
                raise RuntimeError("burner control attempt has no peak EEF height")
            peak_z = max(peak_z, current_z)
            self._burner_control_peak_z = peak_z
            eef_descent_m = start_z - current_z
            peak_descent_m = peak_z - current_z
            approach_guard.update(
                policy_actions=self._burner_control_policy_actions,
                max_close_run=self._burner_control_max_close_run,
                eef_descent_m=eef_descent_m,
                peak_descent_m=peak_descent_m,
            )
            if (
                self._burner_control_policy_actions
                == _BURNER_APPROACH_GUARD_ACTIONS
                and self._max_transitions_per_tool_call
                > _BURNER_APPROACH_GUARD_ACTIONS
                and self._burner_control_max_close_run
                < _BURNER_CONTROL_CLOSE_INTENT_STEPS
                and peak_descent_m < _BURNER_APPROACH_MIN_DESCENT_M
            ):
                return "burner_approach_not_converging"
            return None

        result = self._execute_vla_until(
            instruction=step_guidance,
            max_chunks=max_chunks,
            checker=lambda _observation, _action: False,
            failure_checker=approach_not_converging,
            transition_budget=policy_transition_budget,
        )
        anchor_restore: dict[str, Any] = {"attempted": False}
        if result["failure_reason"] == "burner_approach_not_converging":
            remaining = self._max_transitions_per_tool_call - int(
                result["executed_actions"]
            )
            anchor_position, anchor_quaternion = self._burner_control_anchor
            restore_result = self.servo_eef(
                target_position=anchor_position.tolist(),
                target_quaternion=anchor_quaternion.tolist(),
                gripper="open",
                max_steps=min(_BURNER_CONTROL_RESTORE_MAX_STEPS, remaining),
            )
            result = _merge_physical_result(result, restore_result)
            anchor_restore = {
                "attempted": True,
                "target_reached": bool(restore_result["target_reached"]),
                "executed_actions": int(restore_result["executed_actions"]),
                "position_error_m": float(restore_result["position_error_m"]),
                "rotation_error_rad": float(restore_result["rotation_error_rad"]),
            }
            environment = restore_result.get("environment")
            if isinstance(environment, dict) and environment["success"]:
                result = self._finish_repeated_vla_result(
                    result,
                    success=True,
                    failure_reason=None,
                    stop_reason="success",
                )
            elif isinstance(environment, dict) and (
                environment["terminated"] or environment["truncated"]
            ):
                result = self._finish_repeated_vla_result(
                    result,
                    success=False,
                    failure_reason="episode_terminal",
                    stop_reason="episode_terminal",
                )
            elif restore_result["target_reached"]:
                state = _robocasa_state(self._operation.observation)
                self._burner_control_start_z = float(state[9])
                self._burner_control_peak_z = float(state[9])
                self._burner_control_policy_actions = 0
                self._burner_control_close_run = 0
                self._burner_control_max_close_run = 0
                result = self._finish_repeated_vla_result(
                    result,
                    success=False,
                    failure_reason="burner_approach_not_converging",
                    stop_reason="burner_approach_not_converging",
                )
            else:
                result = self._finish_repeated_vla_result(
                    result,
                    success=False,
                    failure_reason="burner_anchor_restore_failed",
                    stop_reason="burner_anchor_restore_failed",
                )
        result["step_intent"] = step_intent
        result["anchor_restore"] = anchor_restore
        result["approach_guard"] = dict(approach_guard)
        result["policy_transition_budget"] = policy_transition_budget
        return result

    def _burner_control_policy_transition_budget(
        self,
        *,
        max_chunks: int,
        available_transitions: int,
    ) -> int:
        capacity = available_transitions
        if self._actions_per_policy_request is not None:
            capacity = min(
                capacity,
                max_chunks * self._actions_per_policy_request,
            )
        return capacity - capacity % CHECKPOINT_CHUNK_SIZE

    def _motion_review(self) -> dict[str, Any]:
        return {
            "observe_required": self._motion_observe_required,
            "resume_required": self._resume_after_adjustment,
            "max_resume_actions": _RESUME_MAX_ACTIONS,
        }

    def _require_motion_review_complete(self) -> None:
        self._sync_grasp_operation()
        if self._motion_observe_required:
            raise ValueError("motion correction/handoff requires a fresh observe before another physical action")
        if self._resume_after_adjustment:
            raise ValueError("after adjust_eef/move_base/adjust_torso, use resume_vla for a short policy handoff")

    def _motion_action_budget(self, requested: int) -> int:
        self._sync_grasp_operation()
        metrics = self._operation.metrics
        if metrics["simulator_success"] or metrics["terminated"] or metrics["truncated"]:
            # Terminal, not a refusal: finish must stay possible (1.0.11 H1).
            raise RuntimeError("motion correction/handoff cannot execute after the episode ended")
        limit = min(requested, self._max_transitions_per_tool_call)
        if self._episode_horizon is not None:
            limit = min(limit, self._episode_horizon - int(metrics["step_count"]))
        harness = self._harness_session.controller if self._harness_session else self._harness
        state = getattr(harness, "state", None)
        finishing_budget = getattr(state, "finishing_action_budget", lambda: None)()
        if finishing_budget is not None:
            limit = min(limit, finishing_budget)
        if limit < 1:
            raise RuntimeError("motion correction/handoff has no remaining episode action budget or finishing allowance")
        return limit

    def _resolve_gripper_command(self, gripper: str) -> tuple[str, str]:
        if gripper not in {"keep", "open", "close"}:
            raise ValueError("gripper must be 'keep', 'open', or 'close'")
        if gripper != "keep":
            return gripper, "explicit"
        for record in reversed(self._operation.action_history_since()):
            count = min(record["executed_actions"], len(record["actions"]))
            if count:
                command = "close" if record["actions"][count - 1][11] >= 0.5 else "open"
                return command, "last_confirmed_command"
        raise ValueError("no confirmed gripper command; choose gripper='open' or 'close'")

    def move_base(
        self, expected_step: int, velocity: list[float], *,
        gripper: str = "keep", max_steps: int = 5,
    ) -> dict[str, Any]:
        """Send a short native base-velocity window from a fresh public observation."""
        command = _finite_vector(velocity, size=3, label="velocity")
        if np.any(np.abs(command) > 1) or not np.any(command):
            raise ValueError("velocity must be nonzero with each value in [-1,1]")
        return self._execute_body_motion(
            "move_base", expected_step, np.concatenate((command, [0])),
            gripper=gripper, max_steps=max_steps,
        )

    def adjust_torso(
        self, expected_step: int, command: float, *,
        gripper: str = "keep", max_steps: int = 5,
    ) -> dict[str, Any]:
        """Send a short native torso-increment window without base velocity."""
        if (
            isinstance(command, bool) or not isinstance(command, (int, float))
            or not np.isfinite(command) or not 0 < abs(command) <= 1
        ):
            raise ValueError("command must be a finite nonzero number in [-1,1]")
        return self._execute_body_motion(
            "adjust_torso", expected_step, np.array([0, 0, 0, command]),
            gripper=gripper, max_steps=max_steps,
        )

    def _execute_body_motion(
        self, tool: str, expected_step: int, command: np.ndarray, *,
        gripper: str, max_steps: int,
    ) -> dict[str, Any]:
        if isinstance(expected_step, bool) or not isinstance(expected_step, int):
            raise ValueError("expected_step must be an integer")
        current_step = int(self._operation.metrics["step_count"])
        if expected_step != current_step or self._last_observed_step != current_step:
            raise ValueError(f"{tool} requires a fresh observe and its exact expected_step")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 2 <= max_steps <= 25:
            raise ValueError("max_steps must be an integer between 2 and 25")
        gripper, gripper_source = self._resolve_gripper_command(gripper)
        self._check_task_gripper(gripper)
        if self._preclose_active and not self._preclose_closing and gripper != "open":
            raise ValueError("pre-close body motion must keep the gripper open")
        limit = self._motion_action_budget(max_steps)
        if limit < 2:
            raise ValueError("body motion needs two remaining actions including its zero stop command")
        before = _robocasa_state(self._operation.observation).copy()
        actions = np.zeros((limit, 12), dtype=np.float32)
        actions[:-1, :4] = command
        # HYBRID_MOBILE_BASE tracks the desired arm target as the body moves.
        actions[:, 4] = 1.0
        actions[:, 11] = -1.0 if gripper == "open" else 1.0
        self._preclose_approval = None
        if self._preclose_active and not self._preclose_closing:
            self._preclose_pending = True
        self._precontact_grounding = None
        self._burner_control_anchor = None
        # A partial fault must also require observation before another action.
        self._motion_observe_required = True
        self._resume_after_adjustment = True
        result = self._operation.execute_actions(tuple(actions))
        self._update_pick_progress(result)
        after = _robocasa_state(self._operation.observation)
        result.update(
            policy_instruction=self._active_policy_instruction,
            transitions_used=result["executed_actions"], transition_budget=limit,
            body_motion={
                "tool": tool, "source_step": expected_step,
                "native_body_command": command.tolist(),
                "command_frame": "current_robot_base",
                "motion_steps_requested": limit - 1,
                "stop_command_executed": result["executed_actions"] == limit,
                "before_base_position": before[:3].tolist(),
                "after_base_position": after[:3].tolist(),
                "before_base_quaternion": before[3:7].tolist(),
                "after_base_quaternion": after[3:7].tolist(),
                "measured_base_delta_position": (after[:3] - before[:3]).tolist(),
                "measurement_frame": "world",
                "gripper_command": gripper, "gripper_source": gripper_source,
                "evidence_scope": (
                    "public base pose only; not a torso joint measurement, collision "
                    "clearance, stopped velocity, object motion/contact or task success"
                ),
            },
        )
        return result

    def adjust_eef(
        self,
        expected_step: int,
        delta_position: list[float],
        *,
        rotation_degrees: list[float] | None = None,
        gripper: str = "keep",
        max_steps: int = 10,
    ) -> dict[str, Any]:
        """Ground a small base-frame correction in the currently observed pose."""
        if isinstance(expected_step, bool) or not isinstance(expected_step, int):
            raise ValueError("expected_step must be an integer")
        current_step = int(self._operation.metrics["step_count"])
        if expected_step != current_step or self._last_observed_step != current_step:
            raise ValueError("adjust_eef requires a fresh observe and its exact expected_step")
        delta = _finite_vector(delta_position, size=3, label="delta_position")
        rotation = _finite_vector(
            [0, 0, 0] if rotation_degrees is None else rotation_degrees,
            size=3, label="rotation_degrees",
        )
        if np.linalg.norm(delta) > _ADJUST_MAX_TRANSLATION_M + 1e-9:
            raise ValueError("delta_position norm must be <=0.05 metres")
        if np.linalg.norm(rotation) > _ADJUST_MAX_ROTATION_DEG + 1e-9:
            raise ValueError("rotation_degrees norm must be <=15 degrees")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 25:
            raise ValueError("max_steps must be an integer between 1 and 25")
        gripper, gripper_source = self._resolve_gripper_command(gripper)
        self._check_task_gripper(gripper)
        limit = self._motion_action_budget(max_steps)
        before = _robocasa_state(self._operation.observation).copy()
        current_quaternion = before[10:14]
        norm = float(np.linalg.norm(current_quaternion))
        if norm <= 1e-8:
            raise ValueError("adjust_eef requires a valid observed EEF quaternion")
        current_quaternion = current_quaternion / norm
        target_quaternion = _rotate_base_quaternion(current_quaternion, np.deg2rad(rotation))
        target_position = before[7:10] + delta
        if self._preclose_active and not self._preclose_closing:
            if gripper != "open":
                raise ValueError("pre-close adjust_eef must keep the gripper open")
            self._preclose_approval = None
            self._preclose_pending = True
        # Set before execution so even a partial physical fault requires observation.
        self._motion_observe_required = True
        self._resume_after_adjustment = True
        result = self._servo_eef(
            target_position.tolist(), target_quaternion.tolist(), gripper,
            max_steps=limit,
        )
        after = _robocasa_state(self._operation.observation)
        result.update(
            policy_instruction=self._active_policy_instruction,
            transitions_used=result["executed_actions"], transition_budget=limit,
            motion_adjustment={
                "source_step": expected_step, "frame": "robot_base",
                "delta_position": delta.tolist(), "rotation_degrees": rotation.tolist(),
                "before_position": before[7:10].tolist(),
                "before_quaternion": current_quaternion.tolist(),
                "target_position": target_position.tolist(),
                "target_quaternion": target_quaternion.tolist(),
                "measured_delta_position": (after[7:10] - before[7:10]).tolist(),
                "measured_rotation_degrees": np.rad2deg(
                    _quaternion_error_rotvec(current_quaternion, after[10:14])
                ).tolist(),
                "gripper_command": gripper, "gripper_source": gripper_source,
                "evidence_scope": "measured EEF motion only; object motion/contact requires observation",
            },
        )
        return result

    def resume_vla(self, max_actions: int = 5) -> dict[str, Any]:
        """Re-infer from the corrected state, with one short unchanged-language chunk."""
        self._sync_grasp_operation()
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or not 1 <= max_actions <= _RESUME_MAX_ACTIONS:
            raise ValueError("max_actions must be an integer between 1 and 10")
        last_source = next((
            record.get("control_source") for record in reversed(self._operation._action_history)
            if record["executed_actions"]
        ), None)
        if not self._resume_after_adjustment and last_source in _PRIVILEGED_TOOL_NAMES:
            # 1.0.11 C3: name the handback path instead of only the requirement.
            raise ValueError(
                f"resume_vla follows only adjust_eef/move_base/adjust_torso; after {last_source} "
                "observe, resolve verification and hand back with continue_vla: intervene=true "
                "with the next operation's frozen sentence, or intervene=false when the policy "
                "keeps the sentence already active"
            )
        if self._motion_observe_required:
            raise ValueError("resume_vla requires a fresh observe after adjust_eef/move_base/adjust_torso")
        if not self._resume_after_adjustment:
            raise ValueError("resume_vla requires a preceding adjust_eef/move_base/adjust_torso")
        limit = self._motion_action_budget(max_actions)
        if self._actions_per_policy_request is not None:
            limit = min(limit, self._actions_per_policy_request)
        source_step = int(self._operation.metrics["step_count"])
        if self._last_observed_step != source_step:
            raise ValueError("resume_vla requires a current observe")
        try:
            result = self._operation.execute(
                max_actions=limit, instruction=self._active_policy_instruction,
            )
            self._update_pick_progress(result)
        finally:
            if int(self._operation.metrics["step_count"]) != source_step:
                self._resume_after_adjustment = False
                self._motion_observe_required = True
        result.update(
            instruction=self._active_policy_instruction, intervened=False,
            transitions_used=result["executed_actions"], transition_budget=limit,
            handoff={"source_step": source_step, "fresh_policy_inference": True},
        )
        if result.get("stop_reason") in {
            "preclose_review_required", "prerelease_review_required", "task_conflict_review_required",
        }:
            self._resume_after_adjustment = False
            self._motion_observe_required = True
        return result

    def observe(self) -> dict[str, Any]:
        observation = self._operation.observation
        state = np.asarray(observation.get("state", ()))
        current_step = int(self._operation.metrics["step_count"])
        self._last_observed_step = current_step
        self._motion_observe_required = False
        grounding = self._precontact_grounding
        if (
            grounding is not None
            and grounding["servo_step"] is not None
            and int(grounding["servo_step"]) == current_step
        ):
            grounding["observed"] = True
        return {
            "step": current_step,
            **({"task_action_review": self._task_review_snapshot()}
               if self._current_task_conflict() or self._task_release_watch else {}),
            "grasp_review": self._grasp_review_status(),
            "grasp_status": self._pick_progress.diagnostics() if self._pick_progress else None,
            "environment": {
                "success": bool(self._operation.metrics["simulator_success"]),
                "terminated": bool(self._operation.metrics["terminated"]),
                "truncated": bool(self._operation.metrics["truncated"]),
            },
            "state": {"proprioception": state.tolist()},
            "state_shape": list(state.shape),
            "image_keys": list(_CAMERAS),
            "task_language": self._instruction,
            "policy_instruction": self._active_policy_instruction,
            "motion_review": self._motion_review(),
            "_image_center_bytes": _png_bytes(observation[_OVERVIEW_CAMERA]),
            "_image_bytes": _png_bytes(observation[_CAMERAS[0]]),
            "_image_right_bytes": _png_bytes(observation[_CAMERAS[1]]),
            "_image_wrist_bytes": _png_bytes(observation[_CAMERAS[2]]),
            "_image_camera_ids": {
                _OVERVIEW_CAMERA: "_image_center_bytes",
                _CAMERAS[0]: "_image_bytes",
                _CAMERAS[1]: "_image_right_bytes",
                _CAMERAS[2]: "_image_wrist_bytes",
            },
        }

    def action_history_snapshot(self, cursor: int = 0) -> dict[str, Any]:
        """Return this scene's command history for independent planning calls.

        Records from ``cursor`` on are returned with each direct or privileged
        tool call's single-step records merged (``_compact_history``); totals
        cover the whole episode.
        """
        records = self._operation.action_history_since()
        return {
            "schema": "robocasa-action-history/v1",
            "action_order": [
                "base_x", "base_y", "base_yaw", "torso", "control_mode",
                "eef_dx", "eef_dy", "eef_dz", "eef_drx", "eef_dry", "eef_drz", "gripper",
            ],
            "semantics": {
                "values": "Validated native PandaOmron control commands, normalized to [-1,1].",
                "eef": "Delta pose targets relative to the robot base; commands are not measured motion.",
                "gripper": "Native gripper value below 0.5 means open, otherwise close.",
                "execution": (
                    "step_ids identify confirmed transitions for the first executed_actions rows. "
                    "Later rows are unexecuted suffixes; a faulted attempted_action_index is an "
                    "unconfirmed attempt unless it has a corresponding confirmed step_id."
                ),
                "sources": (
                    "vla denotes policy output; direct denotes explicit servo/recovery control; "
                    "a privileged tool name denotes that tool's motion."
                ),
                "compaction": (
                    "Consecutive single-step records of one direct or privileged tool call are "
                    "merged into one record; chunk_count gives the merged native chunks, and "
                    "step_ids lists every confirmed transition."
                ),
            },
            "total_chunks": len(records),
            "total_executed_actions": sum(r["executed_actions"] for r in records),
            "records": _compact_history(records[cursor:]),
        }

    def _build_tool_result(
        self, name: str, result: dict[str, Any], execution_id: str,
    ) -> ToolResult:
        # Preserve an alert even if another part of the handler raised after a
        # confirmed transition; regular handlers attach it before normalization.
        self._attach_grasp_feedback(result)
        if result.get("grasp_events"):
            result = {"step": int(self._operation.metrics["step_count"]), **result}
        result = {**result, "motion_review": self._motion_review(),
                  "grasp_review": self._grasp_review_status(),
                  "episode_liveness_budget": self.episode_budget.snapshot()}
        history_count = self._operation.action_history_count
        if history_count > self._action_history_cursor:
            history = self.action_history_snapshot(self._action_history_cursor)
            result = {**result, "action_history": {**history, "scope": "since_previous_tool_result"}}
        # Counters come from confirmed simulator transitions, including on
        # rejected/observation/planning calls that execute no physical action.
        if self._episode_horizon is not None:
            used = int(self._operation.metrics["step_count"])
            result = {**result, "episode_budget": {
                "scope": "episode",
                "horizon": self._episode_horizon,
                "steps_used": used,
                "steps_remaining": max(0, self._episode_horizon - used),
                "exhausted": used >= self._episode_horizon,
            }}
        if "transition_budget" in result:
            result = {**result, "transition_budget_scope": "tool_call"}
        tool_result = super()._build_tool_result(name, result, execution_id)
        self._action_history_cursor = history_count
        return tool_result

    def write_recipe(self, recipe_tag: str) -> None:
        del recipe_tag
        return None

    @property
    def evaluation_metrics(self) -> dict[str, int | bool]:
        return self._operation.metrics

    def close(self) -> None:
        return None


__all__ = ["RoboCasaToolkit"]


def _single_step_record(record: dict[str, Any]) -> bool:
    """A fully executed record whose every action has a confirmed step."""
    return bool(
        record["control_source"] != "vla" and record["status"] == "completed"
        and record["stop_reason"] is None and record["attempted_action_index"] is None
        and record["cancelled_actions"] == 0 and record["step_ids"]
        and record["executed_actions"] == record["requested_actions"]
        == len(record["actions"]) == len(record["step_ids"])
    )


def _compact_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge contiguous single-step records of one direct/privileged control source.

    Privileged motion executes one native action per chunk, and each chunk
    record repeats about 400 characters of metadata; one 150-step path would
    otherwise cost about 75k characters of model context. Policy chunks,
    faulted, cancelled or interrupted records stay separate; commands and
    step IDs are kept exactly.
    """
    merged: list[dict[str, Any]] = []
    for record in records:
        previous = merged[-1] if merged else None
        if (
            previous is not None and _single_step_record(previous) and _single_step_record(record)
            and all(previous[key] == record[key] for key in
                    ("control_source", "episode_id", "instruction", "actions_scope"))
            and record["step_ids"][0] == previous["step_ids"][-1] + 1
        ):
            previous["actions"] = previous["actions"] + record["actions"]
            previous["step_ids"] = previous["step_ids"] + record["step_ids"]
            for key in ("requested_actions", "executed_actions"):
                previous[key] += record[key]
            previous["chunk_count"] = previous.get("chunk_count", 1) + 1
        else:
            merged.append(dict(record))
    return merged


def _merge_physical_result(
    result: dict[str, Any],
    physical_result: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(result)
    for key in ("requested_actions", "executed_actions", "cancelled_actions"):
        merged[key] = int(merged[key]) + int(physical_result[key])
    merged["transitions_used"] = merged["executed_actions"]
    if "environment" in physical_result:
        merged["environment"] = physical_result["environment"]
    for key in _GRASP_FEEDBACK_KEYS:
        if key in physical_result:
            merged[key] = physical_result[key]
    if physical_result.get("stop_reason") == "grasp_lost":
        merged["stop_reason"] = "grasp_lost"
        merged["failure_reason"] = "grasp_lost"
        merged["success"] = False
    return merged


def _finite_vector(value: Any, *, size: int, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _robocasa_state(observation: dict[str, Any]) -> np.ndarray:
    state = np.asarray(observation.get("state", ()), dtype=np.float64)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("RoboCasa servo requires a finite 16D state")
    return state


def _base_rotation_within(first: np.ndarray, second: np.ndarray, tolerance_rad: float) -> bool:
    """Compare two base orientations; an unreported (zero) quaternion matches only itself."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_norm, second_norm = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return bool(np.allclose(first, second, atol=1e-8))
    cosine = min(1.0, abs(float(np.dot(first, second))) / (first_norm * second_norm))
    return 2.0 * float(np.arccos(cosine)) <= tolerance_rad


def _quaternion_error_rotvec(
    current_xyzw: np.ndarray,
    target_xyzw: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_xyzw, dtype=np.float64)
    current_norm = float(np.linalg.norm(current))
    if current.shape != (4,) or current_norm <= 1e-8:
        raise ValueError("RoboCasa servo requires a valid current quaternion")
    current /= current_norm
    x1, y1, z1, w1 = target_xyzw
    x2, y2, z2, w2 = (-current[0], -current[1], -current[2], current[3])
    error = np.array(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dtype=np.float64,
    )
    if error[3] < 0.0:
        error *= -1.0
    vector_norm = float(np.linalg.norm(error[:3]))
    if vector_norm <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, max(float(error[3]), 0.0))
    return error[:3] * (angle / vector_norm)


def _rotate_base_quaternion(current_xyzw: np.ndarray, rotation_rad: np.ndarray) -> np.ndarray:
    """Left-compose a base-frame axis-angle rotation onto an observed XYZW pose."""
    angle = float(np.linalg.norm(rotation_rad))
    if angle <= 1e-8:
        return current_xyzw.copy()
    xyz = rotation_rad / angle * np.sin(angle / 2)
    w = np.cos(angle / 2)
    current_xyz, current_w = current_xyzw[:3], current_xyzw[3]
    return np.concatenate((
        w * current_xyz + current_w * xyz + np.cross(xyz, current_xyz),
        [w * current_w - np.dot(xyz, current_xyz)],
    ))
