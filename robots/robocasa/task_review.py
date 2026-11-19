"""Public-evidence target reviews; no LLM motion generation or trajectory ranking."""

from __future__ import annotations

import numpy as np

from rpent.harness.lifecycle import TASK_ACTION_REVIEW_GUIDANCE, TASK_CONFLICT_TYPES
from rpent.harness.state import subtask_key


class TaskActionReviewMixin:
    def _init_task_review(self):
        self._task_conflicts = {}
        self._task_release_watch = None
        self._task_release_candidate = None
        self._task_release_approval = None
        self._task_release_executing = False
        self.add_tool("review_task_action", {
            "name": "review_task_action", "description": TASK_ACTION_REVIEW_GUIDANCE,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expected_step": {"type": "integer", "minimum": 0},
                    "phase": {"type": "string", "enum": ["grasp", "release"]},
                    "verdict": {"type": "string", "enum": ["aligned", "conflict", "uncertain", "watch"]},
                    "conflict_type": {"type": ["string", "null"], "enum": [*TASK_CONFLICT_TYPES, None]},
                    "expected_target": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "observed_target": {"type": "string", "minLength": 1, "maxLength": 1000},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 1000},
                },
                "required": ["expected_step", "phase", "verdict", "expected_target", "observed_target", "evidence"],
                "additionalProperties": False,
            },
        }, self.review_task_action)

    def _task_harness(self):
        return self._harness_session.controller if self._harness_session else self._harness

    def _task_scope(self):
        harness = self._task_harness()
        active = harness.state.active_step if harness else None
        return subtask_key(active) if active else (("unbound",), "whole")

    def _current_task_conflict(self):
        key = self._task_scope()
        harness = self._task_harness()
        ledger = harness.state._subtasks.get(key, {}) if harness else {}
        return self._task_conflicts.get(key) or ledger.get("task_conflict")

    def review_task_action(self, expected_step, phase, verdict, expected_target,
                           observed_target, evidence, conflict_type=None):
        self.episode_budget.check()
        if self._whole_task_policy_first() and verdict in {"watch", "aligned"}:
            raise ValueError("protected whole-task policy-first execution keeps task-action reviews advisory; "
                             "the existing physical-failure recovery gate must open before intervention")
        current = int(self._operation.metrics["step_count"])
        if type(expected_step) is not int or expected_step != current or self._last_observed_step != current:
            raise ValueError("task review requires a fresh observe at its exact expected_step")
        if phase not in {"grasp", "release"} or verdict not in {"aligned", "conflict", "uncertain", "watch"}:
            raise ValueError("invalid task review phase or verdict")
        expected_target = self._validate_step_guidance(expected_target)
        observed_target = self._validate_step_guidance(observed_target)
        evidence = self._validate_step_guidance(evidence)
        if conflict_type is not None and (verdict != "conflict" or conflict_type not in TASK_CONFLICT_TYPES):
            raise ValueError("conflict_type belongs only to a confirmed semantic conflict")
        if verdict == "conflict" and (
            conflict_type not in TASK_CONFLICT_TYPES
            or (phase == "grasp" and conflict_type != "wrong_target")
            or expected_target == observed_target
        ):
            raise ValueError("name a supported conflict_type for this action")
        if verdict == "watch" and phase != "release":
            raise ValueError("watch is only for an ambiguous container/quantity release")
        key = self._task_scope()
        watch = self._task_release_watch
        if phase == "release" and watch and watch["scope"] == key and watch["expected_target"] != expected_target:
            raise ValueError("review the watched destination/assignment unchanged")
        conflict = self._current_task_conflict()
        if conflict and ((conflict["expected_target"] is not None and expected_target != conflict["expected_target"])
                         or phase != conflict["phase"]):
            raise ValueError("review the same conflicting target and phase; replanning cannot erase a veto")
        harness = self._task_harness()
        if harness and harness.state.active_step is None:
            raise ValueError("task review requires an active physical plan step")
        candidate = self._task_release_candidate
        if verdict == "aligned" and phase == "release" and not conflict and (
            not candidate or candidate["step"] != current or candidate["scope"] != key
        ):
            raise ValueError("aligned release needs a currently paused policy release candidate")
        if verdict == "conflict":
            conflict = {"phase": phase, "conflict_type": conflict_type, "expected_target": expected_target,
                        "observed_target": observed_target, "evidence": evidence, "observation_step": current}
            if harness:
                harness.state.report_task_conflict(conflict)
            self._task_conflicts[key] = conflict
            self._task_release_approval = None
        elif verdict == "aligned":
            if harness:
                harness.state.report_task_conflict(None)
            self._task_conflicts.pop(key, None)
            if phase == "release":
                self._task_release_approval = current if (
                    candidate and candidate["step"] == current and candidate["scope"] == key
                ) else None
        elif verdict == "uncertain":
            self._task_release_approval = None
        else:
            if conflict:
                raise ValueError("a watch cannot clear a confirmed semantic conflict")
            if not self._preclose_review_enabled:
                raise ValueError("this controller has no per-action policy guard")
            self._task_release_watch = {"scope": key, "expected_target": expected_target}
            self._task_release_approval = None
        return {"step": current, "verdict": verdict, "task_action_review": self._task_review_snapshot(),
                "next_action": "Resolve pending verification, use bounded correction/recovery, then observe and review aligned."
                if self._current_task_conflict() else "Continue the existing plan after pending verification."}

    def _task_review_snapshot(self):
        candidate = self._task_release_candidate
        return {"conflict": self._current_task_conflict(),
                "release_watch": dict(self._task_release_watch) if self._task_release_watch else None,
                "release_pending": candidate is not None,
                "candidate": {
                    "observation_step": candidate["step"],
                    "eef_position_base_m": candidate["eef_position_base_m"],
                    "commanded_eef_delta_base_m": (candidate["action"][5:8] * 0.05).tolist(),
                    "evidence_scope": "Current public EEF pose and unexecuted native VLA command; not measured displacement or destination identity.",
                } if candidate else None}

    def _guard_policy_action(self, observation, action):
        if self._whole_task_policy_first():
            return None
        if self._current_task_conflict():
            return "task_conflict_review_required"
        watch = self._task_release_watch
        if watch and watch["scope"] == self._task_scope() and float(action[11]) < 0.5:
            if self._last_gripper_command() == "close":
                # Save the actual policy candidate before execution. Approval
                # resumes this exact action; it never resamples another chunk.
                self._task_release_candidate = {
                    "action": action.copy(), "step": int(self._operation.metrics["step_count"]),
                    "scope": self._task_scope(), "instruction": self._active_policy_instruction,
                    "eef_position_base_m": np.asarray(observation["state"])[7:10].tolist(),
                }
                self._task_release_approval = None
                return "prerelease_review_required"
        return self._guard_grasp_close(observation, action)

    def _take_reviewed_release(self):
        if self._whole_task_policy_first():
            return None
        candidate = self._task_release_candidate
        if self._task_release_approval is None:
            return None
        if (not candidate or candidate["step"] != int(self._operation.metrics["step_count"])
                or candidate["scope"] != self._task_scope()
                or candidate["instruction"] != self._active_policy_instruction
                or self._current_task_conflict()):
            self._task_release_approval = None
            self._task_release_candidate = None
            raise ValueError("reviewed policy candidate is stale; observe and review a new release")
        self._task_release_approval = None
        self._task_release_executing = True
        return candidate["action"].copy()

    def _check_task_gripper(self, gripper):
        conflict = self._current_task_conflict()
        changing = gripper != self._last_gripper_command()
        if conflict and changing and ((conflict["phase"] == "grasp" and gripper == "close")
                                      or (conflict["phase"] == "release" and gripper == "open")):
            raise ValueError("confirmed task conflict blocks this gripper action; correct and review the target first")

    def _observe_task_action(self, action):
        if action is None:
            return
        values = np.asarray(action)
        if values.shape == (12,) and float(values[11]) < 0.5 and self._task_release_executing:
            self._task_release_watch = None
            self._task_release_candidate = None
            self._task_release_approval = None
            self._task_release_executing = False
