"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""

from __future__ import annotations

import hashlib
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from rpent.context.policy import ContextPolicy, ContextProjection
from rpent.context.projectors import RawToolContextV1
from rpent.episode.controller import EpisodeExecutionError
from rpent.episode.liveness import EpisodeBudgetExceeded, budget_scope, current_budget
from rpent.episode.recording import (
    NULL_TRANSITION_OBSERVER,
    TransitionObserver,
    TransitionOperation,
    operation_scope,
)
from rpent.recording.live_events import record_tool_call, record_tool_result
from rpent.utils.logging import get_logger
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.harness import Harness
    from rpent.harness.session import HarnessSession
    from rpent.planning.events import EventRecorder
    from rpent.planning.outcomes import SkillOutcomeAdapter


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None
    context_policy: ContextPolicy | None = field(default=None, repr=False)

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    projection: ContextProjection = field(init=False, repr=False)
    is_finish: bool = field(default=False, init=False)

    #: Legacy name; this caps characters in the emitted text block.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        policy = self.context_policy or RawToolContextV1()
        self.projection = policy.project_tool_result(
            tool_name=self.name,
            result=self.result,
            max_text_chars=self.MAX_TEXT_BYTES_IN_RESULT,
        )
        self.content_blocks = self.projection.content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying LiberoPrimitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side primitives / servers at the end of the run.
    """

    def __init__(self, *, dashboard: Any = None) -> None:
        from rpent.planning.outcomes import UNKNOWN_OUTCOME_ADAPTER

        # name -> (spec, handler)
        self._tools: dict[
            str, tuple[dict[str, Any], Callable[..., dict[str, Any]]]
        ] = {}
        self._dashboard = dashboard
        self._harness: Harness | None = None
        self._harness_session: HarnessSession | None = None
        self._harness_physical_tools: set[str] = set()
        self._harness_evidence_tools: set[str] = set()
        self._strict_evidence_root: Path | None = None
        self._trusted_evidence_paths: set[Path] = set()
        self._trusted_evidence_captures: dict[Path, str] = {}
        self._trusted_artifact_tools: set[str] = set()
        self._event_recorder: EventRecorder | None = None
        self._transition_observer: TransitionObserver = NULL_TRANSITION_OBSERVER
        self._context_policy: ContextPolicy = RawToolContextV1()
        self._skill_outcome_adapter = UNKNOWN_OUTCOME_ADAPTER
        self._register_common_tools()

    def configure_harness_physical_tools(self, tools: set[str]) -> None:
        """Declare the env-specific tools that can change the physical world."""
        if self._harness_session is not None:
            raise RuntimeError(
                "physical tool classification is frozen by the attached session"
            )
        self._harness_physical_tools = set(tools)
        if self._harness is not None:
            self._harness.set_physical_tools(self._harness_physical_tools)

    def configure_harness_evidence_tools(self, tools: set[str]) -> None:
        """Declare environment tools allowed to register verifier artifacts."""
        if self._harness_session is not None:
            raise RuntimeError(
                "evidence tool classification is frozen by the attached session"
            )
        self._harness_evidence_tools = set(tools)

    def attach_harness(self, harness: Harness) -> None:
        """Attach an optional plan/verification harness after env setup.

        Attaching after construction keeps the environment factory contract
        stable: existing robot packages do not need a new constructor argument.
        """
        if self._harness is not None or self._harness_session is not None:
            raise RuntimeError("a harness is already attached")
        self._harness = harness
        self._harness.set_physical_tools(self._harness_physical_tools)
        for name, spec, handler in self._harness.tool_definitions():
            self.add_tool(name, spec, handler)

    def protocol_continuation(self) -> str | None:
        """Return the Harness reminder for a reply that made no tool call."""
        if self._harness is None:
            return None
        return self._harness.protocol_continuation()

    def attach_harness_session(
        self,
        session: HarnessSession,
        evidence_root: Path,
    ) -> None:
        """Attach the authoritative session protocol and trusted evidence root."""
        from rpent.harness.evidence import EvidenceResolver
        from rpent.tools.harness_session import tool_definitions

        if self._harness is not None or self._harness_session is not None:
            raise RuntimeError("a harness is already attached")
        session.controller.set_physical_tools(self._harness_physical_tools)
        resolved_evidence_root = Path(evidence_root).resolve()
        resolver = EvidenceResolver(resolved_evidence_root)
        trusted_evidence_paths: set[Path] = set()
        trusted_evidence_captures: dict[Path, str] = {}
        definitions = tool_definitions(
            session,
            resolver,
            trusted_evidence_paths=trusted_evidence_paths,
            trusted_evidence_captures=trusted_evidence_captures,
        )
        candidate_tools = dict(self._tools)
        unknown_evidence_tools = self._harness_evidence_tools - set(candidate_tools)
        if unknown_evidence_tools:
            raise ValueError(
                "Harness evidence tools are not registered: "
                f"{sorted(unknown_evidence_tools)}"
            )
        trusted_artifact_tools = set(self._harness_evidence_tools)
        write_entry = self._tools.get("write_text_file")
        if write_entry is not None:
            agent_artifact_root = resolved_evidence_root / "agent_artifacts"

            def write_run_artifact(path: str, content: str) -> dict[str, Any]:
                return _write_run_artifact(
                    agent_artifact_root,
                    path,
                    content,
                )

            write_spec = deepcopy(write_entry[0])
            write_spec["description"] = (
                "Write an agent-authored audit artifact. Relative paths are "
                "confined to {{output_dir}}/agent_artifacts and can never be "
                "used as outcome-verification evidence."
            )
            candidate_tools["write_text_file"] = (
                write_spec,
                write_run_artifact,
            )
        for name, spec, handler in definitions:
            candidate_tools[name] = (spec, handler)
        self._tools = candidate_tools
        self._strict_evidence_root = resolved_evidence_root
        self._trusted_evidence_paths = trusted_evidence_paths
        self._trusted_evidence_captures = trusted_evidence_captures
        self._trusted_artifact_tools = trusted_artifact_tools
        self._harness_session = session

    def attach_event_recorder(self, recorder: EventRecorder) -> None:
        """Attach the append-only planning trace used across all backends."""
        if self._event_recorder is not None:
            raise RuntimeError("an event recorder is already attached")
        self._event_recorder = recorder

    def attach_transition_observer(self, observer: TransitionObserver) -> None:
        """Join physical tool operations to controller transition records."""
        self._transition_observer = observer

    def set_context_policy(self, policy: ContextPolicy) -> None:
        """Select the model-facing tool-result projection for this run."""
        self._context_policy = policy

    def set_skill_outcome_adapter(self, adapter: SkillOutcomeAdapter) -> None:
        """Select how successful physical tool calls are classified."""
        self._skill_outcome_adapter = adapter

    @property
    def skill_outcome_adapter_id(self) -> str:
        """Return the stable identifier recorded for the selected adapter."""
        return self._skill_outcome_adapter.id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict.
        """
        if self._harness_session is not None:
            raise RuntimeError("strict session tool registry is frozen")
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        if self._harness_session is None:
            specs = [spec for spec, _ in self._tools.values()]
        else:
            exposed = self._strict_exposed_tool_names()
            specs = [spec for name, (spec, _) in self._tools.items() if name in exposed]
        return substitute(specs)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        budget = current_budget(self)
        with budget_scope(budget):
            if budget is not None:
                try:
                    budget.check()
                except EpisodeBudgetExceeded as exc:
                    return self._build_tool_result(name, {
                        "error": str(exc), "stop_reason": "episode_budget_exhausted",
                        "episode_budget": budget.snapshot(),
                    }, uuid4().hex)
            return self._execute_tool(name, input_dict)

    def _execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        from rpent.planning.feedback import action_feedback_from_result

        execution_id = uuid4().hex
        physical = name in self._harness_physical_tools
        self._record_live_event(
            record_tool_call,
            tool=name,
            arguments=input_dict,
            call_id=execution_id,
        )
        self._record_event(
            "tool_call_proposed",
            execution_id=execution_id,
            tool=name,
            physical=physical,
            args=input_dict,
            control_mode=self._active_control_mode(),
        )
        if self._harness_session is not None:
            exposed = self._strict_exposed_tool_names()
            entry = self._tools.get(name)
            if entry is None or name not in exposed:
                result = {"error": f"unknown or unexposed tool: {name}"}
                return self._build_tool_result(name, result, execution_id)
            if physical:
                return self._execute_session_physical_tool(
                    name,
                    input_dict,
                    execution_id=execution_id,
                    handler=entry[1],
                )
        if self._harness is not None:
            decision = self._harness.before_tool(
                name,
                input_dict,
                execution_id=execution_id,
            )
            if not decision["allowed"]:
                result = {
                    "error": f"harness blocked {name}: {decision['reason']}",
                    "harness": self._harness.snapshot(),
                }
                feedback = action_feedback_from_result(
                    execution_id=execution_id,
                    tool_name=name,
                    physical=physical,
                    result=result,
                    rejected=True,
                    outcome_adapter=self._skill_outcome_adapter,
                )
                self._record_event(
                    "tool_call_blocked",
                    execution_id=execution_id,
                    tool=name,
                    reason=decision["reason"],
                )
                self._record_event("action_feedback", feedback=feedback)
                if self._dashboard is not None:
                    self._dashboard.on_tool_result(name, result)
                return self._build_tool_result(name, result, execution_id)
        entry = self._tools.get(name)
        if entry is None:
            result = {"error": f"unknown tool: {name}"}
            feedback = action_feedback_from_result(
                execution_id=execution_id,
                tool_name=name,
                physical=physical,
                result=result,
                outcome_adapter=self._skill_outcome_adapter,
            )
            if self._harness is not None:
                result["harness"] = self._harness.after_tool(
                    name,
                    input_dict,
                    result,
                    execution_id=execution_id,
                    feedback=feedback,
                )
            self._record_event(
                "tool_execution_finished",
                execution_id=execution_id,
                tool=name,
                feedback=feedback,
            )
            self._record_event("action_feedback", feedback=feedback)
            return self._build_tool_result(name, result, execution_id)
        handler = entry[1]
        self._record_event(
            "tool_execution_started",
            execution_id=execution_id,
            tool=name,
            physical=physical,
        )
        if physical:
            result = self._execute_physical_handler(
                name, input_dict, execution_id, handler
            )
        else:
            try:
                result = handler(**input_dict)
            except TypeError as e:
                result = {
                    "error": f"bad arguments for {name}: {e}",
                    "error_type": "TypeError",
                    "refused": True,
                    "got": input_dict,
                }
            except Exception as e:
                result = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "refused": _refusal(e),
                    "traceback": traceback.format_exc(),
                }
        self._register_trusted_artifacts(name, result)
        feedback = action_feedback_from_result(
            execution_id=execution_id,
            tool_name=name,
            physical=physical,
            result=result,
            outcome_adapter=self._skill_outcome_adapter,
        )
        if self._harness is not None:
            result["harness"] = self._harness.after_tool(
                name,
                input_dict,
                result,
                execution_id=execution_id,
                feedback=feedback,
            )
        self._record_event(
            "tool_execution_finished",
            execution_id=execution_id,
            tool=name,
            feedback=feedback,
        )
        self._record_event("action_feedback", feedback=feedback)
        if feedback.observation is not None:
            self._record_event(
                "observation_created",
                execution_id=execution_id,
                observation=feedback.observation,
            )
        if self._dashboard is not None:
            self._dashboard.on_tool_result(name, result)
        return self._build_tool_result(name, result, execution_id)

    def _strict_exposed_tool_names(self) -> set[str]:
        """Return the static strict-session schema set for this environment."""
        from rpent.tools.harness_session import (
            LEGACY_CONTROL_TOOLS,
            SESSION_TOOL_NAMES,
        )

        read_only = {
            name
            for name in self._tools
            if name not in self._harness_physical_tools
            and name not in LEGACY_CONTROL_TOOLS
            and name != "finish"
        }
        registered_physical = self._harness_physical_tools & self._tools.keys()
        return set(SESSION_TOOL_NAMES) | read_only | set(registered_physical)

    def _execute_session_physical_tool(
        self,
        name: str,
        input_dict: dict[str, Any],
        *,
        execution_id: str,
        handler: Callable[..., dict[str, Any]],
    ) -> ToolResult:
        """Dispatch one leased physical action through the strict session."""
        from rpent.planning.feedback import action_feedback_from_result

        session = self._harness_session
        assert session is not None
        try:
            decision = session.physical_action_allowed(name, execution_id)
        except Exception as exc:
            result = {"error": f"physical action preflight failed: {exc}"}
            return self._build_tool_result(name, result, execution_id)
        if not decision.get("allowed"):
            result = dict(decision)
            result.setdefault("error", "Harness rejected physical action")
            self._record_event(
                "tool_call_blocked",
                execution_id=execution_id,
                tool=name,
                reason=result["error"],
            )
            return self._build_tool_result(name, result, execution_id)

        self._record_event(
            "tool_execution_started",
            execution_id=execution_id,
            tool=name,
            physical=True,
        )
        operation_id = uuid4().hex
        control_mode = self._active_control_mode()
        operation = TransitionOperation(operation_id, execution_id)
        self._record_event(
            "operation_started",
            operation_id=operation_id,
            execution_id=execution_id,
            tool=name,
            plan_step_id=decision.get("plan_step_id"),
        )
        try:
            with operation_scope(
                self._transition_observer, operation_id, execution_id
            ) as operation:
                result = handler(**input_dict)
        except Exception as exc:
            self._record_operation_end(
                name, operation, _operation_error(exc), decision.get("plan_step_id")
            )
            return self._abort_session_physical_tool(name, execution_id, exc)
        if control_mode is not None:
            result = {**result, "control_mode": control_mode}
        self._record_operation_end(name, operation, result, decision.get("plan_step_id"))

        try:
            self._register_trusted_artifacts(name, result)
            feedback = action_feedback_from_result(
                execution_id=execution_id,
                tool_name=name,
                physical=True,
                result=result,
                outcome_adapter=self._skill_outcome_adapter,
            )
        except Exception as exc:
            return self._abort_session_physical_tool(name, execution_id, exc)

        try:
            harness_result = session.record_action_feedback(feedback)
        except Exception as exc:
            return self._abort_session_physical_tool(name, execution_id, exc)
        result = dict(result)
        result["harness"] = harness_result
        self._record_event(
            "tool_execution_finished",
            execution_id=execution_id,
            tool=name,
            feedback=feedback,
        )
        self._record_event("action_feedback", feedback=feedback)
        if feedback.observation is not None:
            self._record_event(
                "observation_created",
                execution_id=execution_id,
                observation=feedback.observation,
            )
        if self._dashboard is not None:
            self._dashboard.on_tool_result(name, result)
        return self._build_tool_result(name, result, execution_id)

    def _execute_physical_handler(
        self,
        name: str,
        input_dict: dict[str, Any],
        execution_id: str,
        handler: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        operation_id = uuid4().hex
        control_mode = self._active_control_mode()
        self._record_event(
            "operation_started",
            operation_id=operation_id,
            execution_id=execution_id,
            tool=name,
        )
        with operation_scope(
            self._transition_observer, operation_id, execution_id
        ) as operation:
            try:
                result = handler(**input_dict)
            except TypeError as exc:
                result = {
                    "error": f"bad arguments for {name}: {exc}",
                    "error_type": "TypeError",
                    "refused": True,
                    "got": input_dict,
                }
            except Exception as exc:
                result = _operation_error(exc)
        if control_mode is not None:
            result = {**result, "control_mode": control_mode}
        self._record_operation_end(name, operation, result)
        return result

    def _record_operation_end(
        self,
        name: str,
        operation: TransitionOperation,
        result: dict[str, Any],
        plan_step_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "operation_id": operation.operation_id,
            "execution_id": operation.execution_id,
            "tool": name,
        }
        if plan_step_id is not None:
            payload["plan_step_id"] = plan_step_id
        for key in (
            "requested_actions",
            "executed_actions",
            "cancelled_actions",
            "paused_actions",
            "resumed_actions",
        ):
            if key in result:
                payload[key] = result[key]
        log = result.get("log")
        result_summary = log.get("result") if isinstance(log, dict) else None
        if isinstance(result_summary, dict):
            payload["result"] = deepcopy(result_summary)
        step_ids = operation.step_ids
        if step_ids:
            payload["first_step_id"] = step_ids[0]
            payload["last_step_id"] = step_ids[-1]
        else:
            environment = result.get("environment")
            executed = result.get("executed_actions")
            if isinstance(environment, dict) and isinstance(
                environment.get("step_id"), int
            ):
                last_step = environment["step_id"]
                payload["last_step_id"] = last_step
                if isinstance(executed, int) and executed > 0:
                    payload["first_step_id"] = last_step - executed + 1
        if "error" in result:
            payload["error"] = result["error"]
            event = "operation_failed"
        else:
            event = "operation_completed"
        self._record_event(event, **payload)
        submit = getattr(self._transition_observer, "submit_operation", None)
        if callable(submit) and operation.step_ids:
            try:
                submit(operation.operation_id, tuple(operation.step_ids))
            except Exception as exc:
                get_logger("toolkit").warning(
                    "operation clip submission failed: %s", exc
                )

    def _register_trusted_artifacts(self, tool_name: str, result: Any) -> None:
        """Register only files attested by an environment-owned tool result."""
        root = self._strict_evidence_root
        if (
            self._harness_session is None
            or root is None
            or tool_name not in self._trusted_artifact_tools
            or not isinstance(result, dict)
            or "error" in result
        ):
            return
        raw_paths = list(_result_artifact_paths(result))
        retain_observation = getattr(
            self._event_recorder, "retain_observation_artifacts", None
        )
        if callable(retain_observation):
            raw_paths.extend(str(path) for path in retain_observation(result))
        refs: list[str] = []
        for raw_path in raw_paths:
            supplied = Path(raw_path)
            candidate = supplied if supplied.is_absolute() else root / supplied
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (FileNotFoundError, ValueError):
                continue
            if resolved.is_file():
                if resolved in self._trusted_evidence_paths:
                    previous_raw = self._trusted_evidence_captures.get(resolved)
                    previous = (
                        datetime.fromisoformat(previous_raw)
                        if previous_raw is not None
                        else datetime.fromtimestamp(
                            resolved.stat().st_mtime, timezone.utc
                        )
                    )
                    captured = datetime.now(timezone.utc)
                    if captured <= previous:
                        captured = previous + timedelta(microseconds=1)
                    self._trusted_evidence_captures[resolved] = captured.isoformat()
                self._trusted_evidence_paths.add(resolved)
                ref = resolved.relative_to(root).as_posix()
                if ref not in refs:
                    refs.append(ref)
        if refs:
            result["observation_refs"] = refs

    def _abort_session_physical_tool(
        self,
        name: str,
        execution_id: str,
        exc: Exception,
    ) -> ToolResult:
        """Consume a dispatched action lease when no valid feedback exists."""
        session = self._harness_session
        assert session is not None
        reason = f"{type(exc).__name__}: {exc}"
        result: dict[str, Any] = {"error": reason, "traceback": traceback.format_exc()}
        try:
            result["harness"] = session.abort_physical_action(
                execution_id,
                reason,
                ["runtime could not produce valid physical action feedback"],
            )
        except Exception as abort_exc:
            result["abort_error"] = f"{type(abort_exc).__name__}: {abort_exc}"
        self._record_event(
            "tool_execution_aborted",
            execution_id=execution_id,
            tool=name,
            reason=reason,
        )
        if self._dashboard is not None:
            self._dashboard.on_tool_result(name, result)
        return self._build_tool_result(name, result, execution_id)

    def _build_tool_result(
        self,
        name: str,
        result: dict[str, Any],
        execution_id: str,
    ) -> ToolResult:
        self._record_live_event(
            record_tool_result,
            tool=name,
            result=result,
            is_error="error" in result,
            call_id=execution_id,
        )
        tool_result = ToolResult(
            name=name,
            result=result,
            call_id=execution_id,
            context_policy=self._context_policy,
        )
        self._record_event(
            "context_projected",
            execution_id=execution_id,
            tool=name,
            context_policy=self._context_policy.id,
            text_sha256=hashlib.sha256(
                tool_result.projection.text.encode("utf-8")
            ).hexdigest(),
            text_chars=len(tool_result.projection.text),
            images=len(tool_result.projection.images),
        )
        return tool_result

    def _record_live_event(self, callback: Callable[..., None], **payload: Any) -> None:
        try:
            recorder = self._event_recorder
            if callback is record_tool_call and recorder is not None:
                mode = self._active_control_mode()
                recorder = SimpleNamespace(record=lambda event, **data: self._event_recorder.record(event, **data, control_mode=mode))
            callback(recorder, **payload)
        except Exception as exc:
            get_logger("toolkit").warning(
                "failed to record live monitor event: %s", exc
            )

    def _active_control_mode(self) -> str | None:
        harness = self._harness_session.controller if self._harness_session else self._harness
        state = getattr(harness, "state", None)
        active = getattr(state, "active_step", None)
        return active.control_mode if active else None

    def _record_event(self, event: str, **payload: Any) -> None:
        if self._event_recorder is not None:
            try:
                self._event_recorder.record(event, **payload)
            except Exception as exc:
                # An action may already have changed the physical world. Trace
                # I/O must never hide its result from the planner.
                get_logger("toolkit").warning(
                    "failed to record planning event %s: %s",
                    event,
                    exc,
                )

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by env toolkits)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the env-side primitives / servers at end of run. Default: no-op."""

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None


def _write_run_artifact(root: Path, path: str, content: str) -> dict[str, Any]:
    """Write one text artifact without allowing escape from the run directory."""
    supplied = Path(path)
    candidate = supplied if supplied.is_absolute() else root / supplied
    target = candidate.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run output: {path}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside run output: {path}") from exc
    target = resolved_parent / target.name
    if target.exists() or target.is_symlink():
        target = target.resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path is outside run output: {path}") from exc
    if target.is_dir():
        raise ValueError(f"artifact path is a directory: {path}")
    target.write_text(str(content))
    return {"path": str(target), "bytes_written": len(str(content).encode("utf-8"))}


def _result_artifact_paths(value: Any, *, key: str = "") -> tuple[str, ...]:
    """Extract path-labelled strings from trusted environment tool output."""
    found: list[str] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            found.extend(
                _result_artifact_paths(child, key=str(child_key).strip().lower())
            )
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_result_artifact_paths(child, key=key))
    elif isinstance(value, str) and "path" in key:
        found.append(value)
    return tuple(found)


# Argument and state guards of the RoboCasa tools are refusals, not faults (1.0.11
# H1); pour_tool.py joined in 1.2.0, its guards had read as terminal faults.
_TOOL_LAYER_FILES = {"toolkit.py", "privileged_tools.py", "pour_tool.py"}


def _refusal(exc: Exception) -> bool:
    """A guard or argument check of the tool layer refused the call.

    Only ValueError/TypeError raised directly in a toolkit or privileged-tool
    module counts; policy, observation and environment faults stay terminal.
    """
    if not isinstance(exc, (ValueError, TypeError)):
        return False
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and Path(frames[-1].filename).name in _TOOL_LAYER_FILES


def _operation_error(exc: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "refused": _refusal(exc),
        "traceback": traceback.format_exc(),
    }
    earlier = getattr(exc, "rpent_executed_actions", 0)
    if isinstance(earlier, int) and earlier > 0:
        result.update(executed_actions=earlier, refused=False)
    if isinstance(exc, EpisodeExecutionError):
        executed = len(exc.confirmed_transitions)
        result.update(
            requested_actions=exc.requested_actions,
            executed_actions=executed + max(earlier, 0) if isinstance(earlier, int) else executed,
            cancelled_actions=exc.requested_actions - executed,
        )
    return result
