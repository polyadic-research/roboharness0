"""Single public entrypoint for versioned planning behavior.

``PlanningStack`` replaces the old two-stage
``EffectivePlanningConfig -> ResolvedPlanningComponents`` flow.  One resolved
stack now owns the effective Harness settings and concrete context projector
used by a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from rpent.harness.lifecycle import EPISODE_LIVENESS_GUIDANCE
from rpent.harness.recovery import RecoveryPolicy
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_POLICY_SPEC

if TYPE_CHECKING:
    from rpent.context import ContextPolicy
    from rpent.harness import Harness, HarnessSession
    from rpent.harness.verification import PredicateVerifier
    from rpent.planning.events import EventRecorder
    from rpent.planning.generator import PlanGenerator


AUTHORITATIVE_PLANNING_PROFILE = "authoritative-gated-v1"
ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE = "robocasa-authoritative-gated-v1"
ROBOCASA_PRECISE_PLANNING_PROFILE = "robocasa-authoritative-gated-v2"
ROBOCASA_HOOKED_GATED_PLANNING_PROFILE = "robocasa-hooked-gated-v1"
ROBOCASA_HORIZON_PLANNING_PROFILE = "robocasa-horizon-gated-v1"
ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE = "robocasa-xiaomi-mibot-gated-v1"
LEGACY_AUTHORITATIVE_PLANNING_PROFILE = "libero-authoritative-gated-v1"
_PROFILE_ALIASES = {
    LEGACY_AUTHORITATIVE_PLANNING_PROFILE: AUTHORITATIVE_PLANNING_PROFILE,
}


@dataclass(frozen=True)
class PlanningProfile:
    """Stable identifiers and defaults for one planning experiment."""

    id: str
    description: str
    planning_algorithm_version: str
    planner_prompt_version: str
    verifier_prompt_version: str
    instruction_parser_version: str
    observation_version: str
    context_policy_version: str
    history_policy_version: str
    agent_runtime_version: str
    harness_mode: str
    max_replans: int | None
    environment_horizon_only: bool = False
    require_execution_id: bool = False
    auto_verify_tool_signals: bool = False
    require_environment_success: bool = False
    outcome_program_schema_version: str = "outcome-program/v1"
    require_outcome_program: bool = False
    authoritative_session: bool = False
    verification_max_observation_age_s: float = 30.0
    verification_min_confidence: float = 0.8
    recovery_policy: RecoveryPolicy | None = None
    policy_family: str | None = None
    checkpoint_repo_id: str | None = None
    checkpoint_revision: str | None = None
    binding_name: str | None = None
    observation_history: int | None = None
    observation_interval: int | None = None
    crop_ratio: float | None = None
    generated_chunk_size: int | None = None
    native_action_shape: tuple[int, ...] | None = None
    canonical_task_input: bool | None = None
    supports_articulated_control: bool | None = None

    def __post_init__(self) -> None:
        policy = self.recovery_policy
        if policy is None:
            return
        scoped_replans = (
            policy.local_replans_per_outcome
            + policy.stage_replans_per_stage
            + policy.remaining_stage_repartitions_per_run
        )
        if self.max_replans is not None and scoped_replans > self.max_replans:
            raise ValueError(
                "recovery replan budget cannot exceed max_replans"
            )

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable profile metadata."""
        metadata = asdict(self)
        if not self.environment_horizon_only:
            metadata.pop("environment_horizon_only")
        if not self.authoritative_session:
            metadata.pop("authoritative_session")
            metadata.pop("verification_max_observation_age_s")
            metadata.pop("verification_min_confidence")
        if self.recovery_policy is None:
            metadata.pop("recovery_policy")
        for field_name in (
            "policy_family",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "binding_name",
            "observation_history",
            "observation_interval",
            "crop_ratio",
            "generated_chunk_size",
            "native_action_shape",
            "canonical_task_input",
            "supports_articulated_control",
        ):
            if getattr(self, field_name) is None:
                metadata.pop(field_name)
        return metadata


@dataclass(frozen=True)
class PlanningStack:
    """A profile resolved into all components needed by one run."""

    profile: PlanningProfile
    harness_mode: str
    max_replans: int | None
    context_policy: ContextPolicy
    recovery_policy: RecoveryPolicy
    use_chinese: bool = False
    require_environment_success: bool = False
    transition_takeover_enabled: bool = True
    op_takeover_enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.transition_takeover_enabled) is not bool or type(self.op_takeover_enabled) is not bool:
            raise ValueError("control-mode ablation switches must be booleans")

    @property
    def id(self) -> str:
        """Return the selected profile id."""
        return self.profile.id

    @property
    def config(self) -> PlanningStack:
        """Compatibility view for old ``ResolvedPlanningComponents`` callers."""
        return self

    @property
    def uses_authoritative_session(self) -> bool:
        """Whether this profile uses the lifecycle-owned session protocol."""
        return self.profile.authoritative_session

    def metadata(self) -> dict[str, object]:
        """Return the selected profile and effective runtime values."""
        effective = {
            "harness_mode": self.harness_mode,
            "max_replans": self.max_replans,
            "transition_takeover_enabled": self.transition_takeover_enabled,
            "op_takeover_enabled": self.op_takeover_enabled,
            "use_chinese": self.use_chinese,
            "require_execution_id": self.profile.require_execution_id,
            "auto_verify_tool_signals": self.profile.auto_verify_tool_signals,
            "require_environment_success": self.require_environment_success,
            "outcome_program_schema_version": (
                self.profile.outcome_program_schema_version
            ),
            "recovery_policy": asdict(self.recovery_policy),
        }
        if self.uses_authoritative_session:
            effective["authoritative_session"] = True
        if self.profile.environment_horizon_only:
            effective["environment_horizon_only"] = True
            # This profile uses the hooked controller, not RecoveryTracker.
            effective["recovery_policy"] = None
        return {
            "profile_id": self.profile.id,
            "profile": self.profile.metadata(),
            "effective": effective,
        }

    def component_metadata(self) -> dict[str, object]:
        """Return concrete component identities for trace/debug UIs."""
        metadata = {
            "context_policy": self.context_policy.id,
            "harness_mode": self.harness_mode,
            "max_replans": self.max_replans,
            "use_chinese": self.use_chinese,
            "require_execution_id": self.profile.require_execution_id,
            "auto_verify_tool_signals": self.profile.auto_verify_tool_signals,
            "require_environment_success": (self.require_environment_success),
            "outcome_program_schema_version": (
                self.profile.outcome_program_schema_version
            ),
            "planning_algorithm": self.profile.planning_algorithm_version,
            "agent_runtime": self.profile.agent_runtime_version,
        }
        if self.uses_authoritative_session:
            metadata["authoritative_session"] = True
        return metadata

    def prompt_context(self) -> str:
        """Render version identity and Harness protocol for the LLM."""
        identity = (
            f"PLANNING PROFILE: {self.profile.id}; "
            f"algorithm={self.profile.planning_algorithm_version}; "
            f"planner_prompt={self.profile.planner_prompt_version}; "
            f"verifier_prompt={self.profile.verifier_prompt_version}; "
            f"observation={self.profile.observation_version}; "
            f"context={self.profile.context_policy_version}; "
            f"history={self.profile.history_policy_version}; "
            f"runtime={self.profile.agent_runtime_version}."
            f" outcome_program={self.profile.outcome_program_schema_version}."
        )
        speed_protocol = ""
        if self.profile.planner_prompt_version in {
            "libero-fast-v2",
            "libero-hooked-v3",
            "episode-hooked-v1",
            "episode-hooked-v2",
        }:
            speed_protocol = (
                " FAST ITERATION: The system prompt and tool schemas are "
                "self-contained. Do not read MEMORY.md, guides, previous run "
                "artifacts, or reference recipes unless a concrete runtime "
                "error cannot be resolved otherwise. Keep decisions concise. "
            )
            if self.profile.planner_prompt_version in {
                "libero-fast-v2",
                "libero-hooked-v3",
            }:
                speed_protocol += (
                    "When localizing multiple pixels from one camera/step, use one "
                    "back_project call with points=[[row,col], ...], never repeated "
                    "single-pixel calls. Prefer one region query or one segmentation "
                    "call when it provides equivalent evidence."
                )
        if self.harness_mode == "reactive":
            protocol = (
                f"{identity} Use the normal reactive tool loop. After each "
                "physical action, inspect the returned state before choosing "
                f"the next action.{speed_protocol}"
            )
        elif self.uses_authoritative_session:
            protocol = (
                f"{identity} HARNESS MODE: {self.harness_mode}; authoritative "
                "HarnessSession protocol. Begin with read-only scene observation "
                "and grounded localization, then call start_instruction exactly "
                "once with the original goal and those observations. After startup "
                "and after every operation, inspect execution_status. Perform only "
                "a Harness-controlled operation listed in "
                "execution_status.allowed_operations; read-only inspection and "
                "run-output audit writing utilities are available outside the "
                "lifecycle because they cannot change the robot. When "
                "execution_status.required_operation is present, perform that exact "
                "operation next, except required_operation='verify_outcomes': first "
                "obtain one new read-only observation after the preceding "
                "submit_observation, then call verify_outcomes using only its new "
                "capture. Never reuse the previous observation result; unchanged "
                "content-addressed refs may repeat only because the new observation "
                "records a later capture time. Physical actions must use the active "
                "step's "
                "allowed_tools. configured_physical_tool is a category placeholder "
                "in allowed_operations, never a literal tool name. In "
                "ACTION_REQUIRED, choose exactly one actual tool from "
                "execution_status.allowed_tools. Each action must be followed by "
                "the required "
                "fresh observation or verification operation. When grounded "
                "evidence shows the current path needs an exception, call "
                "request_exception; once recovery_required is reported, call "
                "apply_recovery and continue following execution_status. Contract "
                "completion is determined by the immutable OutcomeProgram and its "
                "authoritative predicate verifier. Call "
                "finish(status='success', summary=...) only when "
                "execution_status reports state='complete' and "
                "contract_success=true. For terminal_failure, save the audit and "
                "call finish(status='failure', summary=...). Never use "
                "status='stuck'; normalize every unsuccessful terminal outcome "
                "to status='failure'."
                + speed_protocol
            )
        elif self.profile.environment_horizon_only:
            protocol = (
                f"{identity} HARNESS MODE: plan-gated; environment-horizon budget. "
                "Observe the scene, call create_plan with grounded observations, "
                "then register_plan. Physical actions must follow the active "
                "step's allowed_tools. Inspect the returned harness execution_mode "
                "after each action. In verify mode call verify_step with the exact "
                "pending_execution_id and fresh evidence. Use progress for an "
                "unfinished step, retry for a recoverable miss, and pass only for "
                "a verified local result. Ground local progress in current/historical "
                "images and returned textual feedback. When current images are "
                "inconclusive, use fresh measurements and confirmed execution history; "
                "mark inferred progress and keep final success authoritative. "
                "Uncertain evidence requires informative read-only history inspection "
                "or observation before resolving the pending verification. Repeating "
                "an unchanged occluded view is not new evidence. Never bypass "
                "verification to execute an unchecked action. In plan mode call replan "
                "and register its proposal before continuing. After a resolved retry, "
                "replan is also available without another physical action or verification. "
                "There is no independent replan or physical-tool-call limit. A step "
                "attempt limit requests replanning; it does not end the episode. "
                "If the local plan completes without native task success, follow "
                "completion_review: observe again and supply evidence-backed "
                "completed_step_reviews and review_observation_id to replan. Recheck "
                "earlier accepted results against the original goal, not just the "
                "last action. Local passes can be revised and never replace native "
                "success. All motion, including recovery, consumes the "
                "same task-specific upstream environment horizon. Read episode_budget for "
                "the actual horizon and steps_remaining. A per-call "
                "transition_budget/max_chunks limit is not that horizon. "
                + EPISODE_LIVENESS_GUIDANCE + " "
                "TERMINATION: finish is accepted only in three cases. Native "
                "environment success ends physical execution immediately; "
                "finish(status='success') requires that signal. On environment "
                "truncation/termination without success, finish(status='failure'). "
                "A controller error may abort execution and must remain visible as "
                "an error. In every other state the Harness rejects "
                "finish(status='failure'): while "
                "steps_remaining>0, environment.success=false, and the environment "
                "has not terminated, "
                "continue the physical goal or replan it; do not replace the goal "
                "with a step to acknowledge failure. "
                "Do not give up merely because a grasp, subgoal, or recovery failed "
                "or a tool refuses another retry of itself; tool-local and step "
                "limits never end the episode. After a rejected finish, do not "
                "resend it and never reply without a tool call: such a reply ends "
                "the run as an INCOMPLETE protocol error, which is worse than any "
                "remaining attempt. Take the operation execution_mode allows: "
                "verify_step in verify mode, replan then register_plan in plan "
                "mode, or one allowed physical tool in act or recover mode. When "
                "no new grounded strategy remains, keep the environment's policy "
                "continuation working on the original task until success or the "
                "horizon ends. Never reset the "
                "scene or the step counter to recover."
                + speed_protocol
            )
        elif self.profile.planner_prompt_version in {
            "libero-hooked-v3",
            "episode-hooked-v1",
            "episode-hooked-v2",
        }:
            signal_sources = (
                "Definitive controller, environment, or Pi0-pick signals"
                if self.profile.planner_prompt_version == "libero-hooked-v3"
                else "Definitive normalized controller or environment signals"
            )
            protocol = (
                f"{identity} HARNESS MODE: {self.harness_mode}; planning-loop-v3. "
                "Start in read-only observation/planning mode. Call create_plan "
                "with grounded observations, inspect the proposal, and activate "
                "it with register_plan. In this legacy loop, register_plan "
                "activates a one-shot full-task contract, not an appendable "
                "stage. Before registration, reject any proposal that omits a "
                "physical phase required for official task success and call "
                "create_plan again. For pick-and-place, the proposal must cover "
                "acquisition, transport, supported placement, and release. There "
                "is no continuation or stage-advance after a plan completes. "
                "Physical actions are single-writer and "
                "must follow the active step's allowed_tools. After each physical "
                "action inspect harness.execution_mode and pending_execution_id. "
                f"{signal_sources} may be verified automatically. Only when "
                "execution_mode='verify', call "
                "verify_step with that exact execution_id and grounded evidence. "
                "Use verdict='progress' when the action helped but the step is "
                "unfinished, 'retry' for a bounded same-step retry, 'pass' only "
                "when expected evidence is satisfied, and fail/uncertain when the "
                "remaining plan is invalid. Replan returns an inactive repair "
                "proposal that must be registered. finish(status='success') is "
                + (
                    "blocked until the official environment predicate is true. "
                    "Official environment success is authoritative terminal "
                    "evidence: if it terminates the episode before later "
                    "procedural steps run, do not execute those now-redundant "
                    "steps; save the audit and finish successfully."
                    if self.require_environment_success
                    else (
                        "allowed after every registered plan step has grounded "
                        "verification; the official environment predicate is "
                        "audit-only for this run."
                    )
                )
                + speed_protocol
            )
            if self.require_environment_success:
                protocol += (
                    " TERMINATION PROTOCOL: After the official environment "
                    "predicate becomes true, save the required audit and call "
                    "finish(status='success', summary=...). If execution becomes "
                    "unrecoverable, the episode is truncated, or the replan "
                    "budget is exhausted, save the audit and call "
                    "finish(status='failure', summary=...). Do not use "
                    "status='stuck'; normalize every unsuccessful terminal "
                    "outcome to status='failure'. A failed plan step or exhausted "
                    "same-step attempt budget is not by itself terminal: when "
                    "replans_remaining>0 and the episode is still safely "
                    "recoverable, call replan and register_plan instead of finish."
                )
            else:
                protocol += (
                    " CUSTOM-SEQUENCE TERMINATION PROTOCOL: "
                    "(1) After every registered plan step has grounded "
                    "verification, save the required audit and call "
                    "finish(status='success', summary=...). "
                    "(2) If execution becomes unrecoverable, "
                    "episode_truncated=true, or the replan budget is exhausted, "
                    "save the audit and call "
                    "finish(status='failure', summary=...). "
                    "A failed plan step or exhausted same-step attempt budget is "
                    "not by itself terminal: when replans_remaining>0 and the "
                    "episode is still safely recoverable, call replan and "
                    "register_plan instead of finish. "
                    "(3) If base_task_success=true while environment termination "
                    "is ignored, do not call finish; continue the remaining "
                    "registered plan. "
                    "(4) Never use status='stuck'; normalize every unsuccessful "
                    "terminal outcome to status='failure'."
                )
        else:
            strictness = (
                "mandatory"
                if self.harness_mode == "plan-gated"
                else "strongly recommended"
            )
            protocol = (
                f"{identity} HARNESS MODE: {self.harness_mode}. Structured planning "
                f"is {strictness}. After inspecting the initial state, call "
                "create_plan with grounded observations, inspect its candidate "
                "strategy and steps, then call register_plan with the returned "
                "proposal_id before physical actions. After EVERY physical action, "
                "inspect the returned state and call verify_step with concrete "
                "image/proprioception/geometry evidence. A fail or uncertain "
                "verdict requires calling replan with grounded observations, "
                "inspecting the repair proposal, and registering its proposal_id "
                "before another physical action. Never "
                f"claim hidden simulator state as evidence.{speed_protocol}"
            )
        if self.use_chinese:
            recovery_text = (
                "recovery reason"
                if self.uses_authoritative_session
                else "replan reason"
            )
            protocol += (
                " LANGUAGE REQUIREMENT: Use Simplified Chinese for all "
                "user-visible and audit-facing planning text, including strategy, "
                "step intent, preconditions, expected evidence, observations, "
                "constraints, verify_step evidence/summary, "
                f"{recovery_text}, and "
                "finish summary. Keep tool names, JSON keys, identifiers, numeric "
                "coordinates, and policy prompts unchanged when exact syntax or "
                "model compatibility requires it."
            )
        return protocol

    def startup_instruction(self) -> str:
        """Return the first-step workflow selected by this stack."""
        if self.uses_authoritative_session:
            return """Call `view_driver_state({"step":0})`, inspect the returned
high-resolution images, and localize the named entities with batched
back_project/segment calls. Then call start_instruction exactly once with the
original goal and grounded observation references. Inspect execution_status and
follow its allowed_operations and required_operation. Do not read MEMORY.md or
guides during this run."""
        if self.profile.planner_prompt_version in {
            "libero-fast-v2",
            "libero-hooked-v3",
        }:
            return """Call `view_driver_state({"step":0})`, inspect the returned
high-resolution images, localize the named entities with batched
back_project/segment calls, then call create_plan with those grounded
observations, register_plan with its proposal_id, and execute. Do not read
MEMORY.md or guides during this run."""
        if self.profile.planner_prompt_version in {
            "episode-hooked-v1", "episode-hooked-v2",
        }:
            return """Use the available read-only environment observation tools to gather
grounded scene facts. Then call create_plan with those observations, inspect the
proposal, register_plan with its proposal_id, and execute only the active step's
allowed_tools. Use another read-only observation when physical feedback lacks
fresh evidence. Do not read MEMORY.md or guides during this run."""
        return """read MEMORY.md and the guides, then call
`view_driver_state({"step":0})` and inspect the returned high-resolution images.
Localize the target, call create_plan with the grounded observations, register
the returned proposal, then execute."""

    def build_harness(
        self,
        *,
        trace_path: str | Path,
        event_recorder: EventRecorder | None = None,
        plan_generator: PlanGenerator | None = None,
    ) -> Harness | None:
        """Build the selected Harness, or ``None`` for reactive mode."""
        if self.harness_mode == "reactive":
            return None
        from rpent.harness import Harness

        return Harness(
            mode=self.harness_mode,
            trace_path=trace_path,
            max_replans=self.max_replans,
            event_recorder=event_recorder,
            plan_generator=plan_generator,
            use_chinese=self.use_chinese,
            require_execution_id=self.profile.require_execution_id,
            auto_verify_tool_signals=self.profile.auto_verify_tool_signals,
            require_environment_success=self.require_environment_success,
            environment_horizon_only=self.profile.environment_horizon_only,
            transition_takeover_enabled=self.transition_takeover_enabled,
            op_takeover_enabled=self.op_takeover_enabled,
        )

    def build_harness_session(
        self,
        *,
        trace_path: str | Path,
        predicate_verifiers: Sequence[PredicateVerifier],
        event_recorder: EventRecorder | None = None,
        plan_generator: PlanGenerator | None = None,
    ) -> HarnessSession:
        """Build the authoritative session selected by this profile."""
        if not self.uses_authoritative_session:
            raise ValueError(
                f"planning profile {self.id!r} does not use authoritative "
                "HarnessSession"
            )
        if not predicate_verifiers:
            raise ValueError(
                "authoritative HarnessSession requires a predicate verifier"
            )
        from rpent.harness import HarnessSession
        from rpent.harness.verification import (
            VerificationCoordinator,
            VerificationPolicy,
        )

        controller = self.build_harness(
            trace_path=trace_path,
            event_recorder=event_recorder,
            plan_generator=plan_generator,
        )
        if controller is None:  # Defensive against an invalid future profile.
            raise ValueError("authoritative HarnessSession requires a Harness")
        coordinator = VerificationCoordinator(
            predicate_verifiers,
            policy=VerificationPolicy(
                max_observation_age_s=(
                    self.profile.verification_max_observation_age_s
                ),
                min_confidence=self.profile.verification_min_confidence,
            ),
        )
        return HarnessSession(
            controller,
            verification_coordinator=coordinator,
            recovery_policy=self.recovery_policy,
        )


DEFAULT_PLANNING_PROFILE = "libero-reactive-v1"

_PROFILES = {
    "libero-reactive-v1": PlanningProfile(
        id="libero-reactive-v1",
        description="Compatibility profile for the existing reactive agent loop.",
        planning_algorithm_version="tool-loop-v1",
        planner_prompt_version="libero-planner-v1",
        verifier_prompt_version="inline-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-driver-state-v1",
        context_policy_version="raw-tool-result-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="reactive",
        max_replans=2,
    ),
    "libero-trace-v1": PlanningProfile(
        id="libero-trace-v1",
        description=(
            "Collect structured plan/verification/replan traces without blocking "
            "physical actions."
        ),
        planning_algorithm_version="plan-generate-register-verify-repair-v2",
        planner_prompt_version="libero-planner-v2",
        verifier_prompt_version="harness-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-driver-state-v1",
        context_policy_version="raw-tool-result-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="trace",
        max_replans=3,
    ),
    "libero-plan-gated-v1": PlanningProfile(
        id="libero-plan-gated-v1",
        description=(
            "Require an explicit plan and evidence-based verification around "
            "every physical action."
        ),
        planning_algorithm_version="plan-generate-register-verify-repair-v2",
        planner_prompt_version="libero-planner-v2",
        verifier_prompt_version="harness-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-driver-state-v1",
        context_policy_version="raw-tool-result-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="plan-gated",
        max_replans=3,
    ),
    "libero-compact-trace-v1": PlanningProfile(
        id="libero-compact-trace-v1",
        description=(
            "Use normalized compact physical-action feedback while retaining "
            "full inspection-tool payloads; trace plan protocol violations."
        ),
        planning_algorithm_version="plan-generate-register-verify-repair-v2",
        planner_prompt_version="libero-planner-v2",
        verifier_prompt_version="harness-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="trace",
        max_replans=3,
    ),
    "libero-fast-trace-v1": PlanningProfile(
        id="libero-fast-trace-v1",
        description=(
            "Fast iteration profile: compact feedback, no repeated guide reads, "
            "and batched perception calls in trace mode."
        ),
        planning_algorithm_version="plan-generate-register-verify-repair-v2",
        planner_prompt_version="libero-fast-v2",
        verifier_prompt_version="harness-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="trace",
        max_replans=2,
    ),
    "libero-hooked-gated-v3": PlanningProfile(
        id="libero-hooked-gated-v3",
        description=(
            "Event-hooked planning loop with execution-bound verification, "
            "bounded retries, tool-signal verification, and strict success gating."
        ),
        planning_algorithm_version="mode-hook-feedback-loop-v3",
        planner_prompt_version="libero-hooked-v3",
        verifier_prompt_version="structured-feedback-verifier-v2",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="plan-gated",
        max_replans=2,
        require_execution_id=True,
        auto_verify_tool_signals=True,
        require_environment_success=True,
    ),
    "episode-hooked-gated-v1": PlanningProfile(
        id="episode-hooked-gated-v1",
        description=(
            "Simulator-neutral event-hooked planning loop with execution-bound "
            "verification, bounded retries, tool-signal verification, and strict "
            "environment-success gating."
        ),
        planning_algorithm_version="mode-hook-feedback-loop-v3",
        planner_prompt_version="episode-hooked-v1",
        verifier_prompt_version="structured-feedback-verifier-v2",
        instruction_parser_version="instruction-override-v1",
        observation_version="episode-observation-summary-v1",
        context_policy_version="episode-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="plan-gated",
        max_replans=2,
        require_execution_id=True,
        auto_verify_tool_signals=True,
        require_environment_success=True,
    ),
    ROBOCASA_HOOKED_GATED_PLANNING_PROFILE: PlanningProfile(
        id=ROBOCASA_HOOKED_GATED_PLANNING_PROFILE,
        description=(
            "RoboCasa event-hooked planning loop using the legacy Harness, "
            "simulator-neutral Episode context, and strict environment-success "
            "gating."
        ),
        planning_algorithm_version="mode-hook-feedback-loop-v3",
        planner_prompt_version="episode-hooked-v1",
        verifier_prompt_version="structured-feedback-verifier-v2",
        instruction_parser_version="instruction-override-v1",
        observation_version="episode-observation-summary-v1",
        context_policy_version="episode-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="legacy-agent-runtime-v1",
        harness_mode="plan-gated",
        max_replans=8,
        require_execution_id=True,
        auto_verify_tool_signals=True,
        require_environment_success=True,
    ),
    AUTHORITATIVE_PLANNING_PROFILE: PlanningProfile(
        id=AUTHORITATIVE_PLANNING_PROFILE,
        description=(
            "Authoritative HarnessSession loop with immutable outcome contracts, "
            "lifecycle-owned operations, recovery, and API-runtime predicate "
            "verification."
        ),
        planning_algorithm_version="authoritative-outcome-session-v1",
        planner_prompt_version="authoritative-harness-v1",
        verifier_prompt_version="authoritative-predicate-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="authoritative-harness-session-v1",
        harness_mode="plan-gated",
        max_replans=2,
        require_execution_id=True,
        auto_verify_tool_signals=False,
        require_environment_success=False,
        require_outcome_program=True,
        authoritative_session=True,
        verification_max_observation_age_s=30.0,
        verification_min_confidence=0.8,
    ),
    ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE: PlanningProfile(
        id=ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE,
        description=(
            "RoboCasa authoritative HarnessSession loop with immutable outcome "
            "contracts, lifecycle-owned operations, bounded recovery, and "
            "API-runtime predicate verification."
        ),
        planning_algorithm_version="authoritative-outcome-session-v1",
        planner_prompt_version="authoritative-harness-v1",
        verifier_prompt_version="authoritative-predicate-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="authoritative-harness-session-v1",
        harness_mode="plan-gated",
        max_replans=8,
        require_execution_id=True,
        auto_verify_tool_signals=False,
        require_environment_success=False,
        require_outcome_program=True,
        authoritative_session=True,
        verification_max_observation_age_s=30.0,
        verification_min_confidence=0.8,
        recovery_policy=RecoveryPolicy(
            local_replans_per_outcome=6,
            stage_replans_per_stage=1,
            remaining_stage_repartitions_per_run=1,
            max_total_physical_actions_per_run=9,
        ),
    ),
    ROBOCASA_PRECISE_PLANNING_PROFILE: PlanningProfile(
        id=ROBOCASA_PRECISE_PLANNING_PROFILE,
        description=(
            "RoboCasa authoritative HarnessSession loop with 25-action "
            "receding-horizon recovery coverage for a full 450-step episode."
        ),
        planning_algorithm_version="authoritative-outcome-session-v1",
        planner_prompt_version="authoritative-harness-v1",
        verifier_prompt_version="authoritative-predicate-verifier-v1",
        instruction_parser_version="instruction-override-v1",
        observation_version="libero-observation-summary-v1",
        context_policy_version="libero-compact-v1",
        history_policy_version="backend-native-v1",
        agent_runtime_version="authoritative-harness-session-v1",
        harness_mode="plan-gated",
        max_replans=17,
        require_execution_id=True,
        auto_verify_tool_signals=False,
        require_environment_success=False,
        require_outcome_program=True,
        authoritative_session=True,
        verification_max_observation_age_s=30.0,
        verification_min_confidence=0.8,
        recovery_policy=RecoveryPolicy(
            local_replans_per_outcome=15,
            stage_replans_per_stage=1,
            remaining_stage_repartitions_per_run=1,
            max_total_physical_actions_per_run=18,
        ),
    ),
}

_PROFILES[ROBOCASA_HORIZON_PLANNING_PROFILE] = replace(
    _PROFILES[ROBOCASA_HOOKED_GATED_PLANNING_PROFILE],
    id=ROBOCASA_HORIZON_PLANNING_PROFILE,
    description=(
        "RoboCasa hooked planning bounded by the upstream task horizon, with "
        "native success, uncapped recovery, and independent episode liveness budgets."
    ),
    planning_algorithm_version="environment-horizon-hooked-v1",
    planner_prompt_version="episode-hooked-v2",
    verifier_prompt_version="multimodal-feedback-verifier-v3",
    max_replans=None,
    environment_horizon_only=True,
)

_PROFILES[ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE] = replace(
    _PROFILES[ROBOCASA_HOOKED_GATED_PLANNING_PROFILE],
    id=ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE,
    description="RoboCasa hooked planning with the pinned Xiaomi MiBoT policy.",
    policy_family="xiaomi-mibot",
    checkpoint_repo_id="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
    checkpoint_revision="0d1aa76d0d82debc9b611e4d1e231096434d5be4",
    binding_name="robocasa-xiaomi-mibot-v1",
    observation_history=XIAOMI_MIBOT_POLICY_SPEC.observation_history,
    observation_interval=XIAOMI_MIBOT_POLICY_SPEC.observation_interval,
    crop_ratio=XIAOMI_MIBOT_POLICY_SPEC.crop_ratio,
    generated_chunk_size=XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size,
    native_action_shape=XIAOMI_MIBOT_POLICY_SPEC.native_action_shape,
    canonical_task_input=True,
    supports_articulated_control=True,
)


def list_planning_profiles() -> tuple[PlanningProfile, ...]:
    """Return registered profiles in stable UI/CLI order."""
    return tuple(_PROFILES.values())


def list_planning_profile_ids(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """Return canonical profile IDs and optional accepted input aliases."""
    profile_ids = tuple(_PROFILES)
    if not include_aliases:
        return profile_ids
    return (*profile_ids, *_PROFILE_ALIASES)


def get_planning_profile(profile_id: str) -> PlanningProfile:
    """Resolve a profile id or raise a user-facing error."""
    canonical_id = _PROFILE_ALIASES.get(profile_id, profile_id)
    try:
        return _PROFILES[canonical_id]
    except KeyError as exc:
        choices = ", ".join(list_planning_profile_ids(include_aliases=True))
        raise ValueError(
            f"unknown planning profile {profile_id!r}; choose one of: {choices}"
        ) from exc


def resolve_required_planning_profile(profile_id: object) -> str:
    """Require an explicit profile and return its canonical registered ID."""
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(
            "--planning-profile is required for non-UI evaluation"
        )
    return get_planning_profile(profile_id.strip()).id


def resolve_planning_stack(
    profile_id: str,
    *,
    harness_mode: str | None = None,
    max_replans: int | None = None,
    use_chinese: bool = False,
    require_environment_success: bool | None = None,
    transition_takeover_enabled: bool = True,
    op_takeover_enabled: bool = True,
) -> PlanningStack:
    """Resolve one profile directly into its concrete per-run stack."""
    from rpent.context import resolve_context_policy

    profile = get_planning_profile(profile_id)
    if profile.environment_horizon_only and (
        max_replans is not None
        or harness_mode not in {None, "plan-gated"}
        or require_environment_success is False
    ):
        raise ValueError(
            "environment-horizon profile requires plan-gated native-success "
            "evaluation without a --max-replans override"
        )
    effective_mode = harness_mode or profile.harness_mode
    if effective_mode not in {"reactive", "trace", "plan-gated"}:
        raise ValueError(f"unsupported harness mode: {effective_mode!r}")
    if profile.authoritative_session and effective_mode != "plan-gated":
        raise ValueError(
            f"authoritative profile {profile.id!r} requires harness mode "
            "'plan-gated'"
        )
    effective_environment_success = (
        profile.require_environment_success
        if require_environment_success is None
        else bool(require_environment_success)
    )
    if profile.authoritative_session and effective_environment_success:
        raise ValueError(
            "authoritative HarnessSession does not implement an environment "
            "success gate"
        )
    effective_replans = profile.max_replans if max_replans is None else max_replans
    if effective_replans is not None and effective_replans < 0:
        raise ValueError("max_replans must be >= 0")
    if profile.recovery_policy is not None:
        scoped_replans = (
            profile.recovery_policy.local_replans_per_outcome
            + profile.recovery_policy.stage_replans_per_stage
            + profile.recovery_policy.remaining_stage_repartitions_per_run
        )
        if effective_replans is not None and scoped_replans > effective_replans:
            raise ValueError(
                "recovery replan budget cannot exceed effective max_replans"
            )
    return PlanningStack(
        profile=profile,
        harness_mode=effective_mode,
        max_replans=effective_replans,
        context_policy=resolve_context_policy(profile.context_policy_version),
        recovery_policy=profile.recovery_policy or RecoveryPolicy(),
        use_chinese=bool(use_chinese),
        require_environment_success=effective_environment_success,
        transition_takeover_enabled=transition_takeover_enabled,
        op_takeover_enabled=op_takeover_enabled,
    )


# Compatibility names. New code should use PlanningStack/resolve_planning_stack.
EffectivePlanningConfig = PlanningStack


def resolve_planner_turn_limit(profile_id: str, requested: int | None) -> int | None:
    """Keep infrastructure timeouts separate from the episode's action budget."""
    if get_planning_profile(profile_id).environment_horizon_only:
        if requested is not None:
            raise ValueError("environment-horizon profile does not accept --max-turns")
        return None
    if requested is not None and requested < 1:
        raise ValueError("max_turns must be >= 1")
    return 100 if requested is None else requested


def resolve_planning_config(
    profile_id: str,
    *,
    harness_mode: str | None = None,
    max_replans: int | None = None,
    use_chinese: bool = False,
    require_environment_success: bool | None = None,
    transition_takeover_enabled: bool = True,
    op_takeover_enabled: bool = True,
) -> PlanningStack:
    """Compatibility wrapper for the former config-only resolver."""
    return resolve_planning_stack(
        profile_id,
        harness_mode=harness_mode,
        max_replans=max_replans,
        use_chinese=use_chinese,
        require_environment_success=require_environment_success,
        transition_takeover_enabled=transition_takeover_enabled,
        op_takeover_enabled=op_takeover_enabled,
    )


__all__ = [
    "AUTHORITATIVE_PLANNING_PROFILE",
    "DEFAULT_PLANNING_PROFILE",
    "EffectivePlanningConfig",
    "LEGACY_AUTHORITATIVE_PLANNING_PROFILE",
    "PlanningProfile",
    "PlanningStack",
    "ROBOCASA_AUTHORITATIVE_PLANNING_PROFILE",
    "ROBOCASA_HOOKED_GATED_PLANNING_PROFILE",
    "ROBOCASA_HORIZON_PLANNING_PROFILE",
    "ROBOCASA_XIAOMI_MIBOT_PLANNING_PROFILE",
    "get_planning_profile",
    "resolve_planner_turn_limit",
    "list_planning_profiles",
    "list_planning_profile_ids",
    "resolve_planning_config",
    "resolve_required_planning_profile",
    "resolve_planning_stack",
]
