"""Episode-local grasp motion evidence from non-privileged robot state."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from rpent.harness.lifecycle import (
    GRASP_STABLE_CLOSE_STEPS,
    GRASP_STABLE_WIDTH_TOLERANCE_M,
)


class GraspProgress:
    """Retain an ordered grasp cycle across bounded physical tool calls.

    Finger width and motion are advisory evidence. They cannot establish object
    identity, actual contact, or the environment's task success predicate.
    """

    def __init__(
        self,
        state: np.ndarray,
        *,
        descent_threshold_m: float,
        ascent_threshold_m: float,
        closed_threshold: float,
        empty_threshold: float,
        open_threshold: float,
        width_jump_threshold_m: float = 0.01,
    ) -> None:
        self.thresholds = (
            descent_threshold_m,
            ascent_threshold_m,
            closed_threshold,
            empty_threshold,
            open_threshold,
        )
        if not all(np.isfinite(value) for value in self.thresholds):
            raise ValueError("grasp thresholds must be finite")
        if not (
            descent_threshold_m > 0
            and ascent_threshold_m > 0
            and 0 <= empty_threshold < closed_threshold < open_threshold
        ):
            raise ValueError("invalid grasp motion or gripper thresholds")
        self.cycle = 0
        if isinstance(width_jump_threshold_m, bool) or not np.isfinite(width_jump_threshold_m) or width_jump_threshold_m <= 0:
            raise ValueError("width jump threshold must be finite and positive")
        self.width_jump_threshold_m = width_jump_threshold_m
        self.reset(state)

    def reset(self, state: np.ndarray) -> None:
        """Discard a released, empty, or explicitly retried grasp cycle."""
        state = self._state(state)
        self.cycle += 1
        self.start_z = float(state[9])
        self.min_z = self.start_z
        self.post_min_peak_z = self.start_z
        self.current_z = self.start_z
        self.final_opening = float(np.abs(state[14:16]).sum())
        self.empty_closure_seen = False
        self.closure_state: np.ndarray | None = None
        self.post_closure_peak_z: float | None = None
        self.retention_monitor_active = False
        self.close_command_steps = 0
        self._close_widths: deque[float] = deque(maxlen=GRASP_STABLE_CLOSE_STEPS)
        self._max_opening = self.final_opening
        self._command_closure = False

    @property
    def sustained_close_command(self) -> bool:
        return self.close_command_steps >= GRASP_STABLE_CLOSE_STEPS

    @staticmethod
    def _state(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("RoboCasa pick requires a finite 16D state")
        return state

    def observe(self, state: np.ndarray, action: Any | None = None) -> dict[str, Any] | None:
        """Consume every executed transition, including a chunk's final state."""
        state = self._state(state)
        opening = float(np.abs(state[14:16]).sum())
        previous_opening = self.final_opening
        action_array = np.asarray(action) if action is not None else None
        release_commanded = (
            bool(float(action_array[11]) < 0.5)
            if action_array is not None and action_array.shape == (12,)
            else None
        )
        closing_commanded = bool(
            action_array is not None
            and action_array.shape == (12,)
            and float(action_array[11]) >= 0.5
        )
        if (
            self.retention_monitor_active
            and release_commanded
            and opening >= self.thresholds[4] - 1e-8
        ):
            # A completed commanded release ends retention even if gradual
            # opening already discarded the old closure state. Allow float32
            # sensor rounding at the open threshold.
            self.reset(state)
        if self.retention_monitor_active and (
            opening <= self.thresholds[3] + 1e-9
            or (
                closing_commanded
                and opening - previous_opening >= self.width_jump_threshold_m - 1e-9
            )
        ):
            event = {
                "type": "grasp_lost",
                "reason": "empty_closure" if opening <= self.thresholds[3] + 1e-9 else "width_jump",
                "grasp_cycle": self.cycle,
                "previous_gripper_opening": previous_opening,
                "gripper_opening": opening,
                "width_change_m": opening - previous_opening,
                "empty_threshold": self.thresholds[3],
                "width_jump_threshold_m": self.width_jump_threshold_m,
                "release_commanded": release_commanded,
                "evidence_scope": "proprioceptive_grasp_loss",
                "message": (
                    "Previously confirmed grasp lost its width evidence. Review current images "
                    "to distinguish intended release, slip/drop, and changed contact; "
                    "gripper state alone does not prove object identity or an object drop."
                ),
            }
            self.reset(state)
            return event
        if (
            (
                self.closure_state is not None
                and opening > self.thresholds[2]
                and (not closing_commanded or (
                    not self._command_closure and opening >= self.thresholds[4]
                ))
            )
            or (self.empty_closure_seen and opening >= self.thresholds[4])
        ):
            # Keep watching gradual width changes until a completed commanded
            # release, alert, explicit retry, or new episode resets retention.
            retained = self.retention_monitor_active
            self.reset(state)
            self.retention_monitor_active = retained
        self.current_z = float(state[9])
        self.final_opening = opening
        self._max_opening = max(self._max_opening, opening)
        if closing_commanded:
            self.close_command_steps += 1
            self._close_widths.append(opening)
        else:
            self.close_command_steps = 0
            self._close_widths.clear()
        if (
            self._command_closure
            and not self.retention_monitor_active
            and self.closure_state is not None
            and abs(opening - float(np.abs(self.closure_state[14:16]).sum()))
            > GRASP_STABLE_WIDTH_TOLERANCE_M + 1e-8
        ):
            self.closure_state = None
            self.post_closure_peak_z = None
            self._command_closure = False
        if self.current_z < self.min_z:
            self.min_z = self.current_z
            self.post_min_peak_z = self.current_z
        else:
            self.post_min_peak_z = max(self.post_min_peak_z, self.current_z)
        if self.descent_confirmed:
            if opening <= self.thresholds[3]:
                self.empty_closure_seen = True
                self.closure_state = None
                self.post_closure_peak_z = None
            elif (
                not self.empty_closure_seen
                and opening <= self.thresholds[2]
                and self.closure_state is None
            ):
                self.closure_state = state.copy()
                self.post_closure_peak_z = self.current_z
            elif (
                not self.empty_closure_seen
                and self.closure_state is None
                and self.sustained_close_command
                and max(self._close_widths) - min(self._close_widths)
                <= GRASP_STABLE_WIDTH_TOLERANCE_M + 1e-8
                and self._max_opening - opening >= GRASP_STABLE_WIDTH_TOLERANCE_M
            ):
                # A wide object may block closure above the historical 3 cm
                # cutoff, even above the open threshold. Require actual width
                # reduction and five stable confirmed close commands. Only a
                # subsequent lift can confirm grasp motion.
                self.closure_state = state.copy()
                self.post_closure_peak_z = self.current_z
                self._command_closure = True
        if self.closure_state is not None:
            self.post_closure_peak_z = max(
                float(self.post_closure_peak_z), self.current_z
            )
        if self.confirmed:
            self.retention_monitor_active = True
        return None

    @property
    def descent_confirmed(self) -> bool:
        return self.start_z - self.min_z >= self.thresholds[0]

    @property
    def closure_confirmed(self) -> bool:
        return bool(
            self.closure_state is not None
            and not self.empty_closure_seen
            and self.thresholds[3] < self.final_opening
            and (self._command_closure or self.final_opening <= self.thresholds[2])
        )

    @property
    def post_closure_ascent_m(self) -> float:
        if self.closure_state is None or self.post_closure_peak_z is None:
            return 0.0
        return self.post_closure_peak_z - float(self.closure_state[9])

    @property
    def confirmed(self) -> bool:
        return bool(
            self.descent_confirmed
            and self.closure_confirmed
            and self.post_closure_ascent_m >= self.thresholds[1]
        )

    @property
    def closed_empty(self) -> bool:
        return bool(
            self.empty_closure_seen
            and self.descent_confirmed
            and self.post_min_peak_z - self.min_z >= self.thresholds[1]
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "start_eef_z": self.start_z,
            "min_eef_z": self.min_z,
            "post_descent_peak_eef_z": self.post_min_peak_z,
            "final_gripper_opening": self.final_opening,
            "descent_confirmed": self.descent_confirmed,
            "closure_confirmed": self.closure_confirmed,
            "closure_evidence": (
                "stable_close_command" if self.closure_confirmed and self._command_closure
                else "narrow_width" if self.closure_confirmed else None
            ),
            "close_command_steps": self.close_command_steps,
            "stable_close_steps_required": GRASP_STABLE_CLOSE_STEPS,
            "stable_width_tolerance_m": GRASP_STABLE_WIDTH_TOLERANCE_M,
            "post_descent_ascent_m": self.post_min_peak_z - self.min_z,
            "post_closure_ascent_m": self.post_closure_ascent_m,
            "grasp_motion_confirmed": self.confirmed,
            "grasp_cycle": self.cycle,
            "evidence_scope": "continuous_episode_grasp_cycle",
            "retention_monitor_active": self.retention_monitor_active,
            "width_jump_threshold_m": self.width_jump_threshold_m,
        }
