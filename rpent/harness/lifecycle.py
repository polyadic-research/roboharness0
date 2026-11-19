"""Executable contract for the ChatGPT-visible Harness lifecycle."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping

# Legacy nested planning may ask the executor for evidence before returning an
# inactive candidate. These read-only exchanges do not transition this lifecycle,
# consume recovery attempts, register a plan, or authorize physical actions.
PLANNING_INFORMATION_MAX_ROUNDS: Final[int] = 2

POLICY_FAILURE_TAKEOVER_GUIDANCE: Final[str] = (
    "Observation, semantic replanning and direct motor takeover are separate. "
    "Two explicitly typed, verified physical VLA failures permit bounded takeover; "
    "they never require it while fresh evidence supports native continuation. "
    "One observed destructive failure or explicit wrong-target/destination/allocation "
    "conflict may permit bounded recovery immediately. Uncertainty, missing effect, "
    "a false task predicate, allocation timeout, guard pause or rejected/zero-action "
    "call alone never establishes a physical failure. Resolve pending verification "
    "before any recovery. Preserve original outcomes, budgets and success authority."
)

UNMEASURED_COMPOSITE_GUIDANCE: Final[str] = (
    "An unmeasured composite task (no own composite capability row) starts as "
    "one whole-task monitor operation: skill_row=none, tier=none. Unknown is "
    "not weak and is not an estimated success rate. Preserve the complete live "
    "original task instruction verbatim and use continue_vla or execute_vla "
    "with intervene=false. Do not preemptively decompose, navigate, restore, "
    "correct a grasp, or replace the instruction. Observation and verification "
    "remain read-only; advisory grasp checks must not discard or replace the "
    "policy's action stream. A call limit, waiting, missing capability data or "
    "a 120-step no-progress window alone does not authorize motor intervention. "
    "A verified current-goal phase boundary, an explicitly typed first physical "
    "failure or 120 executed steps without fresh current-goal progress permits "
    "semantic replanning into policy-only monitor steps; this does not authorize "
    "motor corrections. Preserve operation/subtask identity and all counters. "
    "Replanning may keep the original instruction: do not change it merely because "
    "the internal phase changed. After semantic review is authorized, fresh "
    "current-goal evidence may justify an explicit language intervention within "
    "the same registered operation, preserving its identity and counters. "
    "Internal phase bookkeeping alone never requires this language switch. "
    "Two physical failures, or one destructive "
    "failure/explicit target conflict, permit existing bounded motor recovery. "
    "Preserve every original outcome and authoritative final success checks."
)

POLICY_PHYSICAL_FAILURE_TYPES: Final[frozenset[str]] = frozenset({
    "grasp_missed", "object_dropped", "unsafe_trajectory", "motion_blocked",
})
POLICY_PROGRESS_PHASES: Final[tuple[str, ...]] = (
    "approach", "acquire", "transport", "release", "withdraw", "articulate", "verify",
)
POLICY_PROGRESS_GUIDANCE: Final[str] = (
    "For composite policy execution, verify_step may include progress_evidence "
    "with a stable public target_id, phase (approach/acquire/transport/release/"
    "withdraw/articulate/verify), exact observation_step, observed_change, optional "
    "measured target_distance_m and phase_complete. Use the current unfinished "
    "goal's target and phase, not a historical completed fact. Repeated holding, "
    "an old placement, changed wording or the same phase completion cannot reset "
    "the stall clock. A new distance key establishes a baseline; later measured "
    "decrease or a newly evidenced phase completion counts once. Missing structured "
    "evidence does not invent progress. Public evidence is advisory, never hidden "
    "predicate truth or authority to declare task success. Semantic review may "
    "change the internal monitor plan while preserving the VLA instruction."
)

# Bounded finishing leases spend only a capped part of the policy reserve.
FINISHING_TOOLS: Final[frozenset[str]] = frozenset({"move_eef_path"})
FINISHING_LEASE_STEPS: Final[int] = 50
FINISHING_LEASE_CALLS: Final[int] = 2
FINISHING_MAX_BORROW_STEPS: Final[int] = 200
FINISHING_FALLBACK_FLOOR_STEPS: Final[int] = 50
FINISHING_PROGRESS_CALLS: Final[int] = 2
FINISHING_PROGRESS_M: Final[float] = 0.003
FINISHING_WIDTH_DRIFT_M: Final[float] = 0.003
FINISHING_RESERVE_GUIDANCE: Final[str] = (
    "At a takeover quota boundary, the program may grant a short finishing lease "
    "after two consecutive physically executed, verified progress calls with "
    "measured arm movement toward their endpoints and stable retained-load width. "
    "Holding still or an unsupported progress verdict cannot qualify. Each lease "
    "allows at most 50 actions and two physical calls; all calls still require "
    "fresh verification. Only bounded pose/gripper primitives are allowed: no "
    "base travel, policy action, multiphase macro or new grasp. "
    "All calls still require "
    "verification. Only the current operation may spend its shared reserve, at "
    "most 200 actions and always leaving at least 50 actions for policy fallback. "
    "The configured original episode horizon remains binding. An observed stable "
    "commanded release after qualifying transport can retain one final withdrawal "
    "lease; it never authorizes a new acquisition. Unexpected loss, unstable width, "
    "semantic conflict or a physical call without verified measured progress ends "
    "a finishing lease and returns to the existing monitor fallback. Replanning "
    "and split subtasks cannot replenish the same operation's borrowed reserve. "
    "No lease establishes object support, retention or task success: inspect fresh "
    "public evidence and retain all existing outcome and recovery authority."
)


TASK_CONFLICT_TYPES: Final[tuple[str, ...]] = (
    "wrong_target", "wrong_destination", "wrong_allocation",
)

TASK_ACTION_REVIEW_GUIDANCE: Final[str] = (
    "During protected whole-task policy-first execution, task-action conflict "
    "evidence may permit bounded recovery, but a release watch or speculative "
    "alignment judgment cannot interrupt the native stream. An uncertain or "
    "merely incomplete current phase permits only semantic policy review. "
    "Keep normal delegation and retry thresholds. On fresh, explicit evidence "
    "of a wrong object, destination or quantity/container assignment, call "
    "review_task_action(verdict=conflict) with expected_step, phase, conflict_type, "
    "expected_target, observed_target and evidence. This pauses policy execution "
    "and offers bounded correction immediately, even before any physical failure. "
    "Resolve pending verification before physical correction; use the existing "
    "replan/recovery protocol and unchanged operation identity and budgets. "
    "Correct only the conflicting action, then observe and review aligned before "
    "handing back to the policy. Clear semantic errors never count as progress. "
    "Occlusion, uncertain alignment, slow productive contact and historical rates "
    "alone are not semantic conflicts. For a placement whose original goal has "
    "multiple distinguishable containers or per-container quantities, arm a "
    "release watch with review_task_action(phase=release, verdict=watch) before "
    "continuing transport. It pauses immediately before opening; observe and "
    "review aligned, conflict or uncertain at that pose. Aligned release approval "
    "is local and consumed by one release. Do not arm watches for ordinary "
    "unambiguous placements. A semantic veto is independent of the two rejected "
    "alignment reviews that allow a policy close; neither rejection counts nor "
    "replanning can erase it. Reviews are public-evidence judgments, never hidden "
    "predicate truth or success authority."
)

COMPLETED_STEP_REVIEW_VERDICTS: Final[tuple[str, ...]] = (
    "confirmed", "needs_repair", "uncertain",
)

COMPLETED_STEP_REVIEW_GUIDANCE: Final[str] = (
    "When completion_review.required=true, all local steps passed but native "
    "environment success is still false. Call observe again after the last local "
    "acceptance, then call replan with its completion_review.observation_id as "
    "review_observation_id and completed_step_reviews covering every listed "
    "fact_id exactly once. For each historical acceptance, inspect its original "
    "expected_evidence and current public evidence; report confirmed, needs_repair, "
    "or uncertain with concrete evidence. Check all original task targets, "
    "quantities, container assignments, supported release, placement and posture "
    "constraints, not just the last action. A prior pass is historical evidence, "
    "not an irrevocable fact. Repair earlier work contradicted by fresh evidence; "
    "unknown or occluded conditions remain uncertain. Preserve confirmed progress "
    "and target identity. If every local result is confirmed, inspect the original "
    "goal for omitted conditions rather than repeating the final action. These "
    "reviews cannot authorize finish(success) or expose hidden predicate truth."
)

# Liveness budgets are independent of confirmed physical actions. In particular,
# a review-only pause must not turn the environment horizon into an infinite run.
PRECLOSE_MAX_CONSECUTIVE_REJECTIONS: Final[int] = 2
PRECLOSE_CONTINUATION_GUIDANCE: Final[str] = (
    "A pre-close review pauses and retains the unexecuted policy actions. "
    "Approval alone never moves the robot: resolve pending verification, then "
    "explicitly continue the policy tool. If the observation, instruction and "
    "operation are unchanged, that call resumes the original actions without "
    "another policy prediction. A rejected review, correction, changed state, "
    "instruction or operation discards the retained actions; subsequent policy "
    "execution uses fresh inference. Existing action budgets and guards still apply."
)
# Review budgets and approvals belong to one acquisition/operation. A changed
# effective instruction or active plan operation starts fresh; audit notes and
# bounded corrections within the current operation do not. Sustained commanded
# closure followed by opening is release, not another pre-close review boundary.
GRASP_STABLE_CLOSE_STEPS: Final[int] = 5
GRASP_STABLE_WIDTH_TOLERANCE_M: Final[float] = 0.002
ROBOCASA_EPISODE_MAX_LLM_REQUESTS: Final[int] = 500
ROBOCASA_EPISODE_MAX_SECONDS: Final[int] = 90 * 60
EPISODE_LIVENESS_GUIDANCE: Final[str] = (
    "Each RoboCasa episode has an independent liveness budget: 500 observable "
    "API model requests across execution and nested planning, or 90 minutes of "
    "wall time, whichever is reached first. Native SDK provider requests are "
    "opaque and use the wall-time ceiling. Exhaustion ends the run as FAILED "
    "without claiming environment success or fabricating a finish. After two "
    "consecutive rejected pre-close reviews in the same closure cycle, the "
    "next policy close is permitted without claiming visual approval; direct "
    "closure still requires approval. Resolve pending verification first. "
    "After a verified retry, either use its bounded same-step retry or request "
    "a replan directly; do not verify the already resolved execution again."
)

PLANNING_STEP_GRANULARITY: Final[str] = (
    "Use the smallest number of auditable skill-level steps needed for the goal. "
    "An atomic task may use a single complete-operation step when the selected "
    "tool can execute that operation. Add steps for distinct dependencies, "
    "independently verifiable outcomes, or different necessary tool operations. "
    "Do not split one reliable operation into approach, close, lift, or waypoint "
    "steps merely to reach a step-count target. Preserve every requested outcome "
    "and prerequisite, and never exceed 16 steps."
)

PHYSICAL_ATTEMPT_BUDGET_GUIDANCE: Final[str] = (
    "max_attempts is the total physical tool-call budget while a step is active, "
    "including progress calls and retries; it is not just a retry count. A completed "
    "zero-action pre-close review pause consumes no attempt, but still requires "
    "verification before another physical action. Calls that execute any action "
    "or fail still count. Count every physical call needed for the expected "
    "sequence except this review-only pause, and preserve one additional attempt "
    "only when a safe retry is justified. Reaching max_attempts returns control "
    "for replanning or recovery; by itself it never ends the episode."
)

_PRECLOSE_REVIEW_TOOLS: Final[frozenset[str]] = frozenset(
    {"smolvla_pick", "continue_vla", "execute_vla", "resume_vla"}
)


def consumes_physical_attempt(
    *, tool_name: str, execution_status: str, result: Mapping[str, object],
) -> bool:
    """Exempt only a completed pre-close pause with consistent zero-action counts."""
    executed = result.get("executed_actions")
    requested = result.get("requested_actions")
    cancelled = result.get("cancelled_actions")
    paused = result.get("paused_actions", 0)
    transitions = result.get("transitions_used", 0)
    environment = result.get("environment")
    review_only = (
        tool_name in _PRECLOSE_REVIEW_TOOLS
        and execution_status == "completed"
        and not result.get("error")
        and result.get("stop_reason") in {
            "preclose_review_required", "prerelease_review_required", "task_conflict_review_required",
        }
        and result.get("review_required") is True
        and type(executed) is int and executed == 0
        and type(requested) is int and requested > 0
        and type(cancelled) is int and cancelled >= 0
        and type(paused) is int and paused >= 0
        and cancelled + paused == requested
        and type(transitions) is int and transitions == 0
        and isinstance(environment, Mapping)
        and all(environment.get(field) is False for field in ("success", "terminated", "truncated"))
    )
    return not review_only

# Clarification gates the next action, not all implementation choices in an
# entire episode. An unspecified preference is not missing sensor evidence.
# Later-step prerequisites remain explicit and are checked before that action.
PLANNING_INFORMATION_SCOPE: Final[str] = (
    "Request clarification only for missing evidence that blocks the next grounded "
    "action with its chosen tool. An unspecified task preference is freedom to "
    "choose a suitable option, not a missing answer: do not ask the executor to "
    "invent a required shelf, exact placement position, or route when the goal "
    "does not specify one. Unknown physical state remains unknown; never assert "
    "clearance or support without evidence. Missing evidence needed only by a "
    "later step must not block earlier grounded actions. Keep the complete "
    "original goal and plan later actions at skill level with explicit fresh "
    "observation preconditions; the executor must check them before that action "
    "and replan if blocked. Do not treat expected future evidence as already "
    "observed, delete the later goal, or declare success from a partial plan."
)


class HarnessLifecycleState(str, Enum):
    """Stable external states returned by the Harness runtime."""

    ACTION_REQUIRED = "action_required"
    # Advisory physical alerts (including grasp_lost) use the normal
    # ACTION_PROGRESSED -> VERIFICATION_REQUIRED path. They cannot establish
    # object identity/drop, authorize another action, or spend recovery budget.
    VERIFICATION_REQUIRED = "verification_required"
    OBSERVATION_REQUIRED = "observation_required"
    RECOVERY_REQUIRED = "recovery_required"
    COMPLETE = "complete"
    TERMINAL_FAILURE = "terminal_failure"


class LifecycleEvent(str, Enum):
    """Events that can move the external lifecycle between states."""

    ACTION_PROGRESSED = "action_progressed"
    ACTION_FAILED = "action_failed"
    EXCEPTION_REQUESTED = "exception_requested"
    VERIFICATION_UNCERTAIN = "verification_uncertain"
    OUTCOME_SATISFIED_AWAITING_FINAL_VERIFICATION = (
        "outcome_satisfied_awaiting_final_verification"
    )
    OUTCOME_SATISFIED = "outcome_satisfied"
    STAGE_COMPLETED = "stage_completed"
    CONTRACT_COMPLETED = "contract_completed"
    VERIFICATION_FAILED = "verification_failed"
    STAGE_ADVANCED = "stage_advanced"
    RECOVERY_APPLIED = "recovery_applied"
    BUDGET_EXHAUSTED = "budget_exhausted"


_TRANSITIONS: dict[
    HarnessLifecycleState, dict[LifecycleEvent, HarnessLifecycleState]
] = {
    HarnessLifecycleState.ACTION_REQUIRED: {
        LifecycleEvent.ACTION_PROGRESSED: HarnessLifecycleState.VERIFICATION_REQUIRED,
        LifecycleEvent.ACTION_FAILED: HarnessLifecycleState.RECOVERY_REQUIRED,
        LifecycleEvent.EXCEPTION_REQUESTED: HarnessLifecycleState.RECOVERY_REQUIRED,
        LifecycleEvent.BUDGET_EXHAUSTED: HarnessLifecycleState.TERMINAL_FAILURE,
    },
    HarnessLifecycleState.VERIFICATION_REQUIRED: {
        LifecycleEvent.VERIFICATION_UNCERTAIN: (
            HarnessLifecycleState.VERIFICATION_REQUIRED
        ),
        LifecycleEvent.OUTCOME_SATISFIED_AWAITING_FINAL_VERIFICATION: (
            HarnessLifecycleState.VERIFICATION_REQUIRED
        ),
        LifecycleEvent.OUTCOME_SATISFIED: HarnessLifecycleState.ACTION_REQUIRED,
        LifecycleEvent.STAGE_COMPLETED: HarnessLifecycleState.OBSERVATION_REQUIRED,
        LifecycleEvent.CONTRACT_COMPLETED: HarnessLifecycleState.COMPLETE,
        LifecycleEvent.VERIFICATION_FAILED: HarnessLifecycleState.RECOVERY_REQUIRED,
        LifecycleEvent.EXCEPTION_REQUESTED: HarnessLifecycleState.RECOVERY_REQUIRED,
        LifecycleEvent.BUDGET_EXHAUSTED: HarnessLifecycleState.TERMINAL_FAILURE,
    },
    HarnessLifecycleState.OBSERVATION_REQUIRED: {
        LifecycleEvent.STAGE_ADVANCED: HarnessLifecycleState.ACTION_REQUIRED,
    },
    HarnessLifecycleState.RECOVERY_REQUIRED: {
        LifecycleEvent.RECOVERY_APPLIED: HarnessLifecycleState.ACTION_REQUIRED,
        LifecycleEvent.BUDGET_EXHAUSTED: HarnessLifecycleState.TERMINAL_FAILURE,
    },
    HarnessLifecycleState.COMPLETE: {},
    HarnessLifecycleState.TERMINAL_FAILURE: {},
}

TRANSITIONS: Final[
    Mapping[HarnessLifecycleState, Mapping[LifecycleEvent, HarnessLifecycleState]]
] = MappingProxyType(
    {state: MappingProxyType(events) for state, events in _TRANSITIONS.items()}
)

ALLOWED_OPERATIONS: Final[Mapping[HarnessLifecycleState, tuple[str, ...]]] = (
    MappingProxyType(
        {
            HarnessLifecycleState.ACTION_REQUIRED: (
                "configured_physical_tool",
                "request_exception",
                "execution_status",
            ),
            HarnessLifecycleState.VERIFICATION_REQUIRED: (
                "verify_step",
                "verify_observation",
                "submit_observation",
                "verify_outcomes",
                "execution_status",
            ),
            HarnessLifecycleState.OBSERVATION_REQUIRED: (
                "submit_observation",
                "execution_status",
            ),
            HarnessLifecycleState.RECOVERY_REQUIRED: (
                "apply_recovery",
                "execution_status",
            ),
            HarnessLifecycleState.COMPLETE: ("finish", "execution_status"),
            HarnessLifecycleState.TERMINAL_FAILURE: (
                "finish",
                "execution_status",
            ),
        }
    )
)

RECOVERY_ACTION_PENDING_BLOCKED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"request_exception"}
)

TERMINAL_STATES: Final[frozenset[HarnessLifecycleState]] = frozenset(
    {HarnessLifecycleState.COMPLETE, HarnessLifecycleState.TERMINAL_FAILURE}
)


def allowed_operations(
    state: HarnessLifecycleState | str,
    *,
    recovery_action_pending: bool = False,
) -> tuple[str, ...]:
    """Return operations available after applying conditional lifecycle guards."""
    source = HarnessLifecycleState(state)
    operations = ALLOWED_OPERATIONS[source]
    if not recovery_action_pending:
        return operations
    return tuple(
        operation
        for operation in operations
        if operation not in RECOVERY_ACTION_PENDING_BLOCKED_OPERATIONS
    )


def transition(
    state: HarnessLifecycleState | str,
    event: LifecycleEvent | str,
) -> HarnessLifecycleState:
    """Return the declared target state or reject an illegal transition."""
    try:
        source = HarnessLifecycleState(state)
    except ValueError as exc:
        raise ValueError(f"unknown Harness lifecycle state: {state!r}") from exc
    try:
        trigger = LifecycleEvent(event)
    except ValueError as exc:
        raise ValueError(f"unknown Harness lifecycle event: {event!r}") from exc
    try:
        return TRANSITIONS[source][trigger]
    except KeyError as exc:
        raise ValueError(
            f"illegal Harness lifecycle transition: {source.value} + {trigger.value}"
        ) from exc


__all__ = [
    "TASK_CONFLICT_TYPES",
    "TASK_ACTION_REVIEW_GUIDANCE",
    "COMPLETED_STEP_REVIEW_VERDICTS",
    "COMPLETED_STEP_REVIEW_GUIDANCE",

    "ALLOWED_OPERATIONS",
    "EPISODE_LIVENESS_GUIDANCE",
    "PRECLOSE_MAX_CONSECUTIVE_REJECTIONS",
    "ROBOCASA_EPISODE_MAX_LLM_REQUESTS",
    "ROBOCASA_EPISODE_MAX_SECONDS",
    "PHYSICAL_ATTEMPT_BUDGET_GUIDANCE",
    "PLANNING_INFORMATION_MAX_ROUNDS",
    "PLANNING_INFORMATION_SCOPE",
    "PLANNING_STEP_GRANULARITY",
    "RECOVERY_ACTION_PENDING_BLOCKED_OPERATIONS",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "HarnessLifecycleState",
    "LifecycleEvent",
    "allowed_operations",
    "consumes_physical_attempt",
    "transition",
]
