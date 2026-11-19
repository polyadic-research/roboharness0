"""Canonical single-action simulator session for RoboCasa365."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.observation import CAMERA_SHAPES, copy_robocasa_observation
from rpent.episode.contracts import StepResult


def _response_mapping(value: Any, operation: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"RoboCasa {operation} response must be a mapping")
    return value


def _response_step_id(value: Any, expected: int, operation: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"RoboCasa {operation} step_id must be an integer")
    if value != expected:
        raise RuntimeError(
            f"RoboCasa {operation} step_id mismatch: expected {expected}, got {value}"
        )
    return value


def _response_bool(response: Mapping[str, Any], field: str) -> bool:
    value = response.get(field)
    if not isinstance(value, bool):
        raise RuntimeError(f"RoboCasa {field} must be a boolean")
    return value


def _copy_rgbd_response(
    response: Mapping[str, Any],
    camera_name: str,
) -> dict[str, Any]:
    if response.get("camera_name") != camera_name:
        raise RuntimeError("RoboCasa RGB-D camera mismatch")
    try:
        expected_rgb = CAMERA_SHAPES[camera_name]
    except KeyError as exc:
        raise ValueError(f"unknown RoboCasa RGB-D camera: {camera_name!r}") from exc
    rgb = np.asarray(response.get("rgb"))
    depth_m = np.asarray(response.get("depth_m"))
    intrinsic = np.asarray(response.get("intrinsic_K"))
    extrinsic = np.asarray(response.get("extrinsic_cam2world"))
    if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
        raise RuntimeError("RoboCasa RGB-D returned an invalid RGB frame")
    if (
        depth_m.shape != expected_rgb[:2]
        or not np.issubdtype(depth_m.dtype, np.number)
        or not np.isfinite(depth_m).all()
        or np.any(depth_m <= 0.0)
    ):
        raise RuntimeError("RoboCasa RGB-D returned an invalid metric depth frame")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError("RoboCasa RGB-D returned invalid camera intrinsics")
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise RuntimeError("RoboCasa RGB-D returned invalid camera extrinsics")
    return {
        "episode_id": response["episode_id"],
        "step_id": response["step_id"],
        "camera_name": camera_name,
        "rgb": np.array(rgb, dtype=np.uint8, copy=True),
        "depth_m": np.array(depth_m, dtype=np.float32, copy=True),
        "intrinsic_K": np.array(intrinsic, dtype=np.float64, copy=True),
        "extrinsic_cam2world": np.array(extrinsic, dtype=np.float64, copy=True),
    }


class RoboCasaSession:
    """Validate RoboCasa RPC transitions against ``SimulatorSession``."""

    def __init__(
        self,
        client: RoboCasaEnvClient,
        *,
        task: str,
        seed: int,
    ) -> None:
        self._client = client
        self.task = str(task)
        self._seed = int(seed)
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self._effective_horizon: int | None = None
        self._step_id = 0
        self._terminal = False
        self._closed = False

    @property
    def episode_id(self) -> str:
        if self._episode_id is None:
            raise RuntimeError("episode_id is unavailable before reset")
        return self._episode_id

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            raise RuntimeError("instruction is unavailable before reset")
        return self._instruction

    @property
    def effective_horizon(self) -> int:
        if self._effective_horizon is None:
            raise RuntimeError("effective_horizon is unavailable before reset")
        return self._effective_horizon

    def reset(self, task: str, seed: int) -> dict[str, Any]:
        """Reset this single-use simulator worker exactly once."""
        if self._episode_id is not None:
            raise RuntimeError("RoboCasa session has already been reset")
        if task != self.task:
            raise ValueError(
                f"RoboCasa task mismatch: expected {self.task!r}, got {task!r}"
            )
        if int(seed) != self._seed:
            raise ValueError(
                f"RoboCasa seed mismatch: expected {self._seed}, got {seed}"
            )
        response = _response_mapping(self._client.reset(task, seed), "reset")
        episode_id = response.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError("RoboCasa reset episode_id must be non-empty")
        _response_step_id(response.get("step_id"), 0, "reset")
        instruction = response.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError("RoboCasa reset instruction must be non-empty")
        horizon = response.get("effective_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise RuntimeError("RoboCasa effective_horizon must be positive")
        observation = copy_robocasa_observation(
            _response_mapping(response.get("observation"), "reset observation")
        )
        self._episode_id = episode_id
        self._instruction = instruction.strip()
        self._effective_horizon = horizon
        self._step_id = 0
        self._terminal = False
        return observation

    def observe(self) -> dict[str, Any]:
        """Read and validate the authoritative current remote observation."""
        if self._episode_id is None:
            raise RuntimeError("observation is unavailable before reset")
        response = _response_mapping(self._client.observe(), "observe")
        self._validate_identity(
            response, expected_step=self._step_id, operation="observe"
        )
        if response.get("instruction") != self._instruction:
            raise RuntimeError("RoboCasa observe instruction mismatch")
        return copy_robocasa_observation(
            _response_mapping(response.get("observation"), "observe observation")
        )

    def rgbd(self, camera_name: str) -> dict[str, Any]:
        """Read one current RGB-D frame without advancing or changing terminal state."""

        if self._episode_id is None:
            raise RuntimeError("RoboCasa RGB-D is unavailable before reset")
        response = _response_mapping(self._client.rgbd(camera_name), "RGB-D")
        self._validate_identity(
            response,
            expected_step=self._step_id,
            operation="RGB-D",
        )
        return _copy_rgbd_response(response, camera_name)

    def step(self, action: np.ndarray) -> StepResult:
        """Send exactly one validated 12D action and validate its transition."""
        if self._episode_id is None:
            raise RuntimeError("cannot step RoboCasa before reset")
        if self._terminal:
            raise RuntimeError("cannot step RoboCasa after a terminal transition")
        validated_action = self._validate_action(action)
        try:
            response = _response_mapping(self._client.step(validated_action), "step")
            next_step = self._step_id + 1
            self._validate_identity(response, expected_step=next_step, operation="step")
            observation = copy_robocasa_observation(
                _response_mapping(response.get("observation"), "step observation")
            )
            reward = float(response.get("reward"))
            if not np.isfinite(reward):
                raise RuntimeError("RoboCasa reward must be finite")
            result = StepResult(
                episode_id=self._episode_id,
                step_id=next_step,
                observation=observation,
                reward=reward,
                success=_response_bool(response, "success"),
                terminated=_response_bool(response, "terminated"),
                truncated=_response_bool(response, "truncated"),
            )
        except Exception:
            # A malformed remote transition leaves simulator state uncertain.
            self._terminal = True
            raise
        self._step_id = next_step
        self._terminal = result.terminal
        return result

    def close(self) -> None:
        """Close this session once; the client decides remote ownership."""
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def _validate_identity(
        self,
        response: Mapping[str, Any],
        *,
        expected_step: int,
        operation: str,
    ) -> None:
        if response.get("episode_id") != self._episode_id:
            raise RuntimeError(
                f"RoboCasa {operation} episode_id mismatch: expected "
                f"{self._episode_id!r}, got {response.get('episode_id')!r}"
            )
        _response_step_id(response.get("step_id"), expected_step, operation)

    @staticmethod
    def _validate_action(action: Any) -> np.ndarray:
        if not isinstance(action, np.ndarray):
            raise TypeError("RoboCasa action must be a numpy.ndarray")
        if action.shape != (12,):
            raise ValueError(
                f"RoboCasa action must have shape (12,), got {action.shape}"
            )
        if action.dtype != np.float32:
            raise ValueError("RoboCasa action dtype must be float32")
        if not np.isfinite(action).all():
            raise ValueError("RoboCasa action must contain finite values")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("RoboCasa action values must be in [-1, 1]")
        return np.array(action, dtype=np.float32, copy=True)


__all__ = ["RoboCasaSession"]
