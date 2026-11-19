"""Composable execution hooks for planning-loop-v3.

Hooks keep deterministic safety, state tracking, tool-signal verification, and
completion checks outside the LLM-facing planning controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rpent.harness.gate import GateDecision, PlanSafetyGate
from rpent.harness.state import PlanStateMachine
from rpent.planning.schemas import ActionFeedback, StepVerdict


@dataclass(frozen=True)
class ToolHookContext:
    """Identity and arguments shared by pre/post tool hooks."""

    execution_id: str
    name: str
    input_dict: dict[str, Any]
    physical: bool


class PreToolHook(Protocol):
    """Inspect a proposed tool call before it changes external state."""

    def before_tool(
        self,
        context: ToolHookContext,
        state: PlanStateMachine,
    ) -> GateDecision | None:
        """Return a decision or ``None`` when the hook has no opinion."""
        ...


class PostToolHook(Protocol):
    """Consume normalized feedback after a tool finishes."""

    def after_tool(
        self,
        context: ToolHookContext,
        feedback: ActionFeedback,
        state: PlanStateMachine,
    ) -> StepVerdict | None:
        """Optionally return an automatic structured verdict."""
        ...


class HarnessHookPipeline:
    """Run ordered pre/post hooks with deny-first semantics."""

    def __init__(
        self,
        *,
        pre_hooks: tuple[PreToolHook, ...] = (),
        post_hooks: tuple[PostToolHook, ...] = (),
    ) -> None:
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks

    def before_tool(
        self,
        context: ToolHookContext,
        state: PlanStateMachine,
    ) -> GateDecision:
        warning: str | None = None
        for hook in self.pre_hooks:
            decision = hook.before_tool(context, state)
            if decision is None:
                continue
            if not decision.allowed:
                return decision
            warning = decision.warning or warning
        return GateDecision(allowed=True, warning=warning)

    def after_tool(
        self,
        context: ToolHookContext,
        feedback: ActionFeedback,
        state: PlanStateMachine,
    ) -> tuple[StepVerdict, ...]:
        verdicts: list[StepVerdict] = []
        for hook in self.post_hooks:
            verdict = hook.after_tool(context, feedback, state)
            if verdict is not None:
                verdicts.append(verdict)
        return tuple(verdicts)


class PlanGateHook:
    """Apply the existing plan/tool safety policy as a pre-action hook."""

    def __init__(self, *, gate: PlanSafetyGate, control_tools: set[str]) -> None:
        self.gate = gate
        self.control_tools = set(control_tools)

    def before_tool(
        self,
        context: ToolHookContext,
        state: PlanStateMachine,
    ) -> GateDecision:
        return self.gate.evaluate(
            name=context.name,
            input_dict=context.input_dict,
            state=state,
            control_tools=self.control_tools,
        )


def _refused(feedback: Any) -> bool:
    """Whether a guard or argument check of the tool layer refused the call.

    The toolkit marks such errors ``refused``; policy, environment and
    end-of-episode errors are not refusals and stay terminal.
    """
    raw = getattr(feedback, "raw_result", None)
    if not isinstance(raw, dict) or raw.get("refused") is not True:
        return False
    executed = raw.get("executed_actions")
    return not (isinstance(executed, int) and not isinstance(executed, bool) and executed > 0)


class FinishGuardHook:
    """Prevent unsupported or prematurely terminal finish calls.

    A call the tool refused (a guard, a missing review, bad arguments) never
    justifies finish(failure) while horizon and replans remain (1.0.11 H1:
    D06 gave up at step 679 of 2100 on a review guard).
    """

    _TERMINAL_FAILURE_TYPES = {
        "controller_error",
        "timeout",
        "unsafe_trajectory",
        "recovery_exhausted",
        "object_dropped",
    }

    def __init__(
        self,
        *,
        mode: str,
        require_environment_success: bool,
    ) -> None:
        self.mode = mode
        self.require_environment_success = require_environment_success

    def before_tool(
        self,
        context: ToolHookContext,
        state: PlanStateMachine,
    ) -> GateDecision | None:
        if context.name != "finish":
            return None
        if self.mode != "plan-gated":
            return None
        status = context.input_dict.get("status")
        if status == "failure":
            feedback = state.last_feedback
            if state.environment_horizon_only and not (
                feedback is not None
                and (
                    feedback.environment_terminated
                    or feedback.environment_truncated
                    or (feedback.execution_status != "completed" and not _refused(feedback))
                )
            ):
                return GateDecision(
                    allowed=False,
                    reason=(
                        "environment-horizon budget remains; continue the task "
                        "or replan instead of declaring failure"
                    ),
                )
            failure_type = (
                state.last_verdict.failure_type
                if state.last_verdict is not None
                else None
            )
            if (
                state.needs_replan
                and (state.replans_remaining is None or state.replans_remaining > 0)
                and not (
                    state.environment_horizon_only
                    and feedback is not None
                    and (feedback.environment_terminated or feedback.environment_truncated)
                )
                and not (
                    failure_type in self._TERMINAL_FAILURE_TYPES and not _refused(feedback)
                )
            ):
                return GateDecision(
                    allowed=False,
                    reason=(
                        "cannot finish failure while the current failure is "
                        "recoverable and replan budget remains; call replan and "
                        "register_plan first "
                        f"(replans_remaining={state.replans_remaining})"
                    ),
                )
            return GateDecision(allowed=True)
        if status != "success":
            return None
        if state.awaiting_verification:
            return GateDecision(
                allowed=False,
                reason="cannot finish success while a physical action is unverified",
            )
        if not state.completed and not (
            self.require_environment_success and state.environment_success
        ):
            return GateDecision(
                allowed=False,
                reason="cannot finish success before the plan completes",
            )
        if self.require_environment_success and not state.environment_success:
            return GateDecision(
                allowed=False,
                reason=(
                    "cannot finish success without the environment success "
                    "predicate or an attached goal verifier"
                ),
            )
        return GateDecision(allowed=True)


class PhysicalStateHook:
    """Make normalized physical feedback authoritative for plan state."""

    def after_tool(
        self,
        context: ToolHookContext,
        feedback: ActionFeedback,
        state: PlanStateMachine,
    ) -> None:
        state.record_action_feedback(feedback, input_dict=context.input_dict)
        return None


class StructuredFeedbackVerifierHook:
    """Auto-verify only definitive controller or environment signals.

    Ambiguous visual outcomes intentionally return ``None`` so the agent or a
    future multimodal verifier must submit evidence through ``verify_step``.
    """

    def __init__(self, *, use_chinese: bool = False) -> None:
        self.use_chinese = bool(use_chinese)

    def after_tool(
        self,
        context: ToolHookContext,
        feedback: ActionFeedback,
        state: PlanStateMachine,
    ) -> StepVerdict | None:
        if not context.physical:
            return None
        step_id = state.active_step.id if state.active_step is not None else None
        if feedback.execution_status != "completed":
            return StepVerdict(
                verdict="fail",
                evidence=self._controller_error_evidence(feedback),
                summary=(
                    "物理工具没有成功完成执行。"
                    if self.use_chinese
                    else "The physical tool did not complete successfully."
                ),
                failure_type=feedback.failure_type or "controller_error",
                recommended_recovery=(
                    "重新规划前先检查控制器错误。"
                    if self.use_chinese
                    else "Inspect the controller error before replanning."
                ),
                execution_id=feedback.execution_id,
                step_id=step_id,
                confidence=1.0,
                source="hook",
            )
        if feedback.environment_success:
            return StepVerdict(
                verdict="pass",
                evidence=(
                    (
                        "环境成功谓词=true"
                        if self.use_chinese
                        else "environment_success=true"
                    ),
                ),
                summary=(
                    "已满足环境提供的正式成功条件。"
                    if self.use_chinese
                    else "The official environment success predicate was reached."
                ),
                execution_id=feedback.execution_id,
                step_id=step_id,
                confidence=1.0,
                source="tool-signal",
            )

        action_result = _action_result(feedback.raw_result)
        # Pi0 pick exposes a bounded, documented grasp heuristic. Other skills
        # may use `success=False` merely to mean the overall task is unfinished,
        # so they remain subject to explicit visual/geometric verification.
        if context.name == "pi0_pick" and isinstance(
            action_result.get("success"), bool
        ):
            succeeded = bool(action_result["success"])
            metrics = _numeric_values(action_result)
            needs_followup = bool(
                succeeded
                and state.active_step is not None
                and any(
                    tool != "pi0_pick"
                    for tool in state.active_step.allowed_tools
                )
            )
            return StepVerdict(
                verdict="progress" if needs_followup else (
                    "pass" if succeeded else "retry"
                ),
                evidence=self._pick_evidence(succeeded, metrics),
                summary=(
                    (
                        "Pi0 抓取启发式信号确认物体已被提起；继续执行当前"
                        "步骤中要求的锁紧或稳定动作。"
                    )
                    if self.use_chinese and needs_followup
                    else (
                        "The Pi0 grasp heuristic confirmed a lifted grasp; "
                        "continue with the lock or stabilization action required "
                        "by the current step."
                    )
                    if needs_followup
                    else self._pick_summary(succeeded)
                ),
                failure_type=None if succeeded else "grasp_missed",
                recommended_recovery=(
                    None
                    if succeeded
                    else (
                        "重新观察目标，然后重试当前抓取步骤。"
                        if self.use_chinese
                        else (
                            "Re-observe the target and retry the current "
                            "grasp step."
                        )
                    )
                ),
                metrics=metrics,
                execution_id=feedback.execution_id,
                step_id=step_id,
                confidence=0.9,
                source="tool-signal",
            )
        return None

    def _controller_error_evidence(
        self,
        feedback: ActionFeedback,
    ) -> tuple[str, ...]:
        if self.use_chinese:
            return (
                f"执行状态={feedback.execution_status}",
                f"错误={feedback.error or '未知'}",
            )
        return (
            f"execution_status={feedback.execution_status}",
            f"error={feedback.error or 'unknown'}",
        )

    def _pick_evidence(
        self,
        succeeded: bool,
        metrics: dict[str, float],
    ) -> tuple[str, ...]:
        if self.use_chinese:
            return (
                f"Pi0 抓取成功标志={str(succeeded).lower()}",
                *(f"指标 {key}={value}" for key, value in sorted(metrics.items())),
            )
        return (
            f"pi0_pick.success={str(succeeded).lower()}",
            *(f"{key}={value}" for key, value in sorted(metrics.items())),
        )

    def _pick_summary(self, succeeded: bool) -> str:
        if self.use_chinese:
            return (
                "Pi0 抓取启发式信号确认物体已被提起。"
                if succeeded
                else "Pi0 抓取启发式信号尚未确认抓取成功。"
            )
        return (
            "The Pi0 grasp heuristic confirmed a lifted grasp."
            if succeeded
            else "The Pi0 grasp heuristic did not confirm a grasp."
        )


def _action_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    log = raw_result.get("log")
    if isinstance(log, dict) and isinstance(log.get("result"), dict):
        return log["result"]
    return raw_result


def _numeric_values(result: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values[str(key)] = float(value)
    return values


__all__ = [
    "FinishGuardHook",
    "HarnessHookPipeline",
    "PhysicalStateHook",
    "PlanGateHook",
    "PostToolHook",
    "PreToolHook",
    "StructuredFeedbackVerifierHook",
    "ToolHookContext",
]
