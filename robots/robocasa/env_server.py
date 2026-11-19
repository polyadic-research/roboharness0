"""Versioned RPC facade for one direct upstream RoboCasa environment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from robots.robocasa.observation import (
    CAMERA_SHAPES,
    OBSERVATION_SCHEMA,
    OVERVIEW_CAMERA_KEY,
    assemble_robocasa_observation,
    copy_robocasa_observation,
)
from rpent.utils.rpc import RpcFacade

ROBOCASA_ENV_PROTOCOL = "rpent-robocasa-env-v2"
ACTION_SCHEMA = "robocasa-pandaomron-action-v1"
_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
_RGBD_CAMERA_NAMES = (OVERVIEW_CAMERA_KEY, *_CAMERA_NAMES)
_ALLOWED_REGISTRIES = {"lightwheel", "objaverse", "aigen"}
_LOGGER = logging.getLogger(__name__)


def _save_failed_frame(
    reason: str, camera_name: str, raw_rgb: Any, raw_depth: Any
) -> Path | None:
    """Retain exact renderer outputs without masking the original frame error."""
    configured = os.environ.get("RPENT_ROBOCASA_DIAGNOSTICS_DIR")
    if not configured:
        return None
    try:
        rgb, depth = np.asarray(raw_rgb), np.asarray(raw_depth)
        recorded_at = datetime.now(timezone.utc)
        directory = Path(configured) / (
            recorded_at.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
        )
        directory.mkdir(parents=True, exist_ok=False)
        expected = CAMERA_SHAPES[camera_name]
        checks = []
        if rgb.shape != expected:
            checks.append("rgb_shape")
        if rgb.dtype != np.uint8:
            checks.append("rgb_dtype")
        if depth.shape != expected[:2]:
            checks.append("depth_shape")
        numeric = np.issubdtype(depth.dtype, np.integer) or np.issubdtype(
            depth.dtype, np.floating
        )
        depth_stats = {"shape": list(depth.shape), "dtype": str(depth.dtype)}
        if numeric:
            finite = np.isfinite(depth)
            below, above = depth < 0.0, depth > 1.0
            depth_stats.update(
                nonfinite_count=int(np.count_nonzero(~finite)),
                below_zero_count=int(np.count_nonzero(below)),
                above_one_count=int(np.count_nonzero(above)),
                finite_min=float(depth[finite].min()) if finite.any() else None,
                finite_max=float(depth[finite].max()) if finite.any() else None,
            )
            if not finite.all():
                checks.append("depth_nonfinite")
            if below.any() or above.any():
                checks.append("depth_out_of_range")
        else:
            checks.append("depth_dtype")
        metadata = {
            "recorded_at_utc": recorded_at.isoformat(),
            "pid": os.getpid(),
            "camera_name": camera_name,
            "reason": reason,
            "failed_checks": checks,
            "rgb": {"shape": list(rgb.shape), "dtype": str(rgb.dtype)},
            "depth": depth_stats,
            "expected_rgb_shape": list(expected),
            "expected_depth_shape": list(expected[:2]),
            "raw_orientation": "unaltered renderer output",
            "preview_orientation": "vertically flipped to match observation",
            "depth_preview": "[0,1] grayscale; invalid values magenta; exact values in raw_depth.npy",
            "save_errors": [],
        }

        def write(name, writer):
            try:
                writer(directory / name)
            except Exception as exc:
                metadata["save_errors"].append(
                    {"file": name, "error_type": type(exc).__name__}
                )

        write("raw_rgb.npy", lambda path: np.save(path, rgb, allow_pickle=False))
        write("raw_depth.npy", lambda path: np.save(path, depth, allow_pickle=False))
        import imageio.v3 as iio

        if rgb.ndim == 3 and rgb.shape[-1] in (3, 4) and rgb.dtype == np.uint8:
            write("rgb.png", lambda path: iio.imwrite(path, rgb[::-1]))
        if numeric and depth.ndim == 2:
            gray = (np.clip(np.where(finite, depth, 0), 0, 1) * 255).astype(np.uint8)
            preview = np.repeat(gray[..., None], 3, axis=-1)
            preview[~finite | below | above] = [255, 0, 255]
            write("depth_preview.png", lambda path: iio.imwrite(path, preview[::-1]))
        (directory / "diagnostic.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _LOGGER.warning("Saved failed RoboCasa frame: %s", directory)
        return directory
    except Exception as exc:
        _LOGGER.warning("Could not save failed RoboCasa frame (%s)", type(exc).__name__)
        return None


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    """Validated state returned by one explicit upstream reset."""

    observation: Mapping[str, Any]
    info: Mapping[str, Any]
    instruction: str


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "task",
        "split",
        "seed",
        "robot",
        "camera_names",
        "camera_size",
        "object_registries",
        "rendering_mode",
        "upstream_horizon",
        "safety_ceiling",
        "task_registry_commit",
        "task_registry_digest",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"RoboCasa config is missing: {', '.join(missing)}")
    normalized = dict(config)
    normalized["task"] = str(config["task"]).strip()
    normalized["split"] = str(config["split"]).strip()
    normalized["seed"] = int(config["seed"])
    normalized["camera_names"] = list(config["camera_names"])
    normalized["camera_size"] = int(config["camera_size"])
    normalized["object_registries"] = list(config["object_registries"])
    normalized["upstream_horizon"] = int(config["upstream_horizon"])
    normalized["safety_ceiling"] = int(config["safety_ceiling"])

    if not normalized["task"]:
        raise ValueError("RoboCasa task must be non-empty")
    if normalized["split"] not in {"target", "pretrain"}:
        raise ValueError("RoboCasa split must be target or pretrain")
    if normalized["robot"] != "PandaOmron":
        raise ValueError("RoboCasa robot must be PandaOmron")
    if tuple(normalized["camera_names"]) != _CAMERA_NAMES:
        raise ValueError(f"RoboCasa cameras must be {_CAMERA_NAMES}")
    if normalized["camera_size"] != 256:
        raise ValueError("RoboCasa camera size must be 256")
    registries = normalized["object_registries"]
    if (
        not registries
        or len(registries) != len(set(registries))
        or not set(registries).issubset(_ALLOWED_REGISTRIES)
    ):
        raise ValueError("invalid RoboCasa object registries")
    if normalized["rendering_mode"] != "headless":
        raise ValueError("RoboCasa rendering_mode must be headless")
    if normalized["upstream_horizon"] <= 0 or normalized["safety_ceiling"] <= 0:
        raise ValueError("RoboCasa horizons must be positive")
    if not str(normalized["task_registry_commit"]).strip():
        raise ValueError("RoboCasa task registry commit is required")
    if not str(normalized["task_registry_digest"]).startswith("sha256:"):
        raise ValueError("RoboCasa task registry digest must be sha256")
    return normalized


def _validate_action(action: Any) -> np.ndarray:
    try:
        value = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("RoboCasa action must be numeric") from exc
    if value.shape != (12,):
        raise ValueError(f"RoboCasa action must have shape (12,), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("RoboCasa action must contain finite values")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ValueError("RoboCasa action values must be in [-1, 1]")
    return np.array(value, dtype=np.float32, copy=True)


def _bool_signal(value: Any, label: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"RoboCasa {label} signal must be scalar")
    return bool(array.reshape(-1)[0])


def _camera_geometry(
    sim: Any,
    camera_name: str,
    height: int,
    width: int,
    normalized_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    return {
        "depth_m": np.asarray(
            get_real_depth_map(sim, normalized_depth),
            dtype=np.float32,
        ),
        "intrinsic_K": np.asarray(
            get_camera_intrinsic_matrix(
                sim=sim,
                camera_name=camera_name,
                camera_height=height,
                camera_width=width,
            ),
            dtype=np.float64,
        ),
        "extrinsic_cam2world": np.asarray(
            get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name),
            dtype=np.float64,
        ),
    }


def _copy_rgbd_payload(
    camera_name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if camera_name not in _RGBD_CAMERA_NAMES:
        raise ValueError(f"unknown RoboCasa RGB-D camera: {camera_name!r}")
    if payload.get("camera_name") != camera_name:
        raise RuntimeError("RoboCasa RGB-D camera identity is inconsistent")
    expected_rgb_shape = CAMERA_SHAPES[camera_name]
    rgb = np.asarray(payload.get("rgb"))
    depth_m = np.asarray(payload.get("depth_m"))
    intrinsic = np.asarray(payload.get("intrinsic_K"))
    extrinsic = np.asarray(payload.get("extrinsic_cam2world"))
    if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
        raise RuntimeError("RoboCasa RGB-D returned an invalid RGB frame")
    if depth_m.shape != expected_rgb_shape[:2]:
        raise RuntimeError("RoboCasa RGB-D returned an invalid depth frame")
    if not np.issubdtype(depth_m.dtype, np.number) or not np.isfinite(depth_m).all():
        raise RuntimeError("RoboCasa RGB-D depth must contain finite numbers")
    if np.any(depth_m <= 0.0):
        raise RuntimeError("RoboCasa RGB-D metric depth must be positive")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError("RoboCasa RGB-D returned invalid camera intrinsics")
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise RuntimeError("RoboCasa RGB-D returned invalid camera extrinsics")
    return {
        "camera_name": camera_name,
        "rgb": np.array(rgb, dtype=np.uint8, copy=True),
        "depth_m": np.array(depth_m, dtype=np.float32, copy=True),
        "intrinsic_K": np.array(intrinsic, dtype=np.float64, copy=True),
        "extrinsic_cam2world": np.array(extrinsic, dtype=np.float64, copy=True),
    }


class RoboCasaEnvFacade(RpcFacade):
    """Own one loaded task environment across explicit sequential episodes."""

    def __init__(
        self,
        *,
        environment_factory: Callable[[Mapping[str, Any]], Any],
        dependency_identity: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self._environment_factory = environment_factory
        self._dependency_identity = dict(dependency_identity)
        self._config: dict[str, Any] | None = None
        self._environment: Any | None = None
        self._episode_id: str | None = None
        self._instruction: str | None = None
        self._observation: dict[str, np.ndarray] | None = None
        self._step_id = 0
        self._terminal = False
        self._closed = False

    @property
    def effective_horizon(self) -> int:
        if self._config is None:
            raise RuntimeError("RoboCasa environment is not loaded")
        return min(
            self._config["upstream_horizon"],
            self._config["safety_ceiling"],
        )

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        operations = {
            "env.health": self.health,
            "env.load": self.load,
            "env.reset": self.reset,
            "env.observe": self.observe,
            "env.rgbd": self.rgbd,
            "env.step": self.step,
            "env.close": self.close,
        }
        operation = operations.get(method)
        if operation is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        return operation(*args, **kwargs)

    def health(self) -> dict[str, Any]:
        config = self._config or {}
        return {
            "protocol": ROBOCASA_ENV_PROTOCOL,
            "loaded": self._config is not None,
            "task": config.get("task"),
            "split": config.get("split"),
            "object_registries": list(config.get("object_registries", ())),
            "task_registry_commit": config.get("task_registry_commit"),
            "task_registry_digest": config.get("task_registry_digest"),
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "dependencies": dict(self._dependency_identity),
        }

    def load(self, config: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("RoboCasa environment facade is closed")
        if self._config is not None:
            raise RuntimeError("RoboCasa environment has already been loaded")
        self._config = _validate_config(config)
        return self.health()

    def reset(self, task: str, seed: int) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("RoboCasa environment must be loaded before reset")
        if self._closed:
            raise RuntimeError("RoboCasa environment facade is closed")
        if task != self._config["task"]:
            raise ValueError(
                f"RoboCasa task mismatch: expected {self._config['task']!r}, "
                f"got {task!r}"
            )
        if int(seed) != self._config["seed"]:
            raise ValueError(
                f"RoboCasa seed mismatch: expected {self._config['seed']}, got {seed}"
            )

        self._episode_id = None
        self._instruction = None
        self._observation = None
        self._step_id = 0
        self._terminal = False
        if self._environment is None:
            self._environment = self._environment_factory(self._config)
        reset = getattr(self._environment, "reset", None)
        if not callable(reset):
            raise RuntimeError("RoboCasa environment exposes no reset operation")
        prepared = reset()
        if not isinstance(prepared, PreparedEnvironment):
            raise RuntimeError("RoboCasa environment reset returned invalid state")
        instruction = str(prepared.instruction).strip()
        if not instruction:
            raise RuntimeError("RoboCasa reset returned no upstream instruction")
        observation = assemble_robocasa_observation(prepared.observation)
        self._observation = observation
        self._instruction = instruction
        self._episode_id = str(uuid4())
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "observation": copy_robocasa_observation(observation),
            "instruction": instruction,
            "effective_horizon": self.effective_horizon,
        }

    def observe(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("RoboCasa environment facade is closed")
        if self._episode_id is None or self._observation is None:
            raise RuntimeError("RoboCasa observation is unavailable before reset")
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "observation": copy_robocasa_observation(self._observation),
            "instruction": self._instruction,
        }

    def rgbd(self, camera_name: str) -> dict[str, Any]:
        """Return the latest aligned RGB-D frame without advancing the episode."""

        if self._closed:
            raise RuntimeError("RoboCasa environment facade is closed")
        if self._episode_id is None or self._environment is None:
            raise RuntimeError("RoboCasa RGB-D is unavailable before reset")
        if camera_name not in _RGBD_CAMERA_NAMES:
            raise ValueError(f"unknown RoboCasa RGB-D camera: {camera_name!r}")
        read_rgbd = getattr(self._environment, "rgbd", None)
        if not callable(read_rgbd):
            raise RuntimeError("RoboCasa environment exposes no RGB-D operation")
        payload = read_rgbd(camera_name)
        if not isinstance(payload, Mapping):
            raise RuntimeError("RoboCasa RGB-D response must be a mapping")
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            **_copy_rgbd_payload(camera_name, payload),
        }

    def step(self, action: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("RoboCasa environment facade is closed")
        if self._episode_id is None or self._environment is None:
            raise RuntimeError("cannot step RoboCasa before reset")
        if self._terminal:
            raise RuntimeError("cannot step RoboCasa after a terminal transition")
        validated_action = _validate_action(action)
        transition = self._environment.step(validated_action)
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise RuntimeError("RoboCasa step must return a Gymnasium 5-tuple")
        raw_observation, reward, term, trunc, info = transition
        observation = assemble_robocasa_observation(raw_observation)

        underlying = getattr(self._environment, "env", self._environment)
        predicate = getattr(underlying, "_check_success", None)
        if callable(predicate):
            success = bool(predicate())
        elif isinstance(info, Mapping) and "success" in info:
            success = _bool_signal(info["success"], "success")
        else:
            raise RuntimeError(
                "RoboCasa environment exposes no official task success predicate"
            )
        terminated = _bool_signal(term, "terminated")
        truncated = _bool_signal(trunc, "truncated")
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        if not np.isfinite(reward_value):
            raise RuntimeError("RoboCasa reward must be finite")

        self._step_id += 1
        horizon_reached = self._step_id >= self.effective_horizon
        truncated = truncated or horizon_reached
        self._observation = observation
        self._terminal = success or terminated or truncated
        return {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "observation": copy_robocasa_observation(observation),
            "reward": reward_value,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "horizon_reached": horizon_reached,
        }

    def close(self) -> dict[str, bool]:
        if self._closed:
            return {"closed": True}
        self._closed = True
        if self._environment is not None:
            self._environment.close()
        return {"closed": True}


class _GymEnvironmentAdapter:
    """Translate the checkpoint's 12D vector to upstream Gym action fields."""

    def __init__(self, gym_environment: Any) -> None:
        self._gym_environment = gym_environment
        self.env = gym_environment.env
        self._latest_observation: dict[str, Any] | None = None

    def _with_overview(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        from rpent.recording.narrated.hd import capture_environment_state

        capture_environment_state(self)
        height, width, _channels = CAMERA_SHAPES[OVERVIEW_CAMERA_KEY]
        rendered = self.env.sim.render(
            height=height,
            width=width,
            camera_name=OVERVIEW_CAMERA_KEY,
            depth=True,
        )
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            _save_failed_frame(
                "RoboCasa overview camera returned no depth frame",
                OVERVIEW_CAMERA_KEY,
                rendered,
                None,
            )
            raise RuntimeError("RoboCasa overview camera returned no depth frame")
        raw_rgb, raw_depth = rendered
        frame = np.asarray(raw_rgb)[::-1]
        depth = np.asarray(raw_depth)[::-1]
        if frame.shape != CAMERA_SHAPES[OVERVIEW_CAMERA_KEY] or frame.dtype != np.uint8:
            _save_failed_frame(
                "RoboCasa overview camera returned invalid RGB frame",
                OVERVIEW_CAMERA_KEY,
                raw_rgb,
                raw_depth,
            )
            raise RuntimeError("RoboCasa overview camera returned invalid RGB frame")
        if (
            depth.shape != CAMERA_SHAPES[OVERVIEW_CAMERA_KEY][:2]
            or not np.issubdtype(depth.dtype, np.number)
            or not np.isfinite(depth).all()
            or np.any(depth < 0.0)
            or np.any(depth > 1.0)
        ):
            _save_failed_frame(
                "RoboCasa overview camera returned invalid depth frame",
                OVERVIEW_CAMERA_KEY,
                raw_rgb,
                raw_depth,
            )
            raise RuntimeError("RoboCasa overview camera returned invalid depth frame")
        combined = {
            **observation,
            f"video.{OVERVIEW_CAMERA_KEY}": np.array(frame, copy=True),
            f"video.{OVERVIEW_CAMERA_KEY}_depth": np.array(
                depth,
                dtype=np.float32,
                copy=True,
            ),
        }
        self._latest_observation = combined
        return combined

    def rgbd(self, camera_name: str) -> dict[str, Any]:
        if camera_name not in _RGBD_CAMERA_NAMES:
            raise ValueError(f"unknown RoboCasa RGB-D camera: {camera_name!r}")
        if self._latest_observation is None:
            raise RuntimeError("RoboCasa RGB-D is unavailable before reset")
        expected_rgb_shape = CAMERA_SHAPES[camera_name]
        rgb_key = f"video.{camera_name}"
        depth_key = f"video.{camera_name}_depth"
        if rgb_key not in self._latest_observation:
            raise RuntimeError(f"RoboCasa RGB-D is missing {rgb_key}")
        if depth_key not in self._latest_observation:
            # RoboCasa's wrapper supplies its own camera_depths default. Render
            # missing depth on demand from this unchanged physics state, keeping
            # the actual policy RGB bytes and avoiding three extra renders on
            # every control step. reset/step invalidate this observation cache.
            rendered = self.env.sim.render(
                height=expected_rgb_shape[0],
                width=expected_rgb_shape[1],
                camera_name=camera_name,
                depth=True,
            )
            if not isinstance(rendered, tuple) or len(rendered) != 2:
                _save_failed_frame(
                    "RoboCasa RGB-D returned no depth frame",
                    camera_name,
                    rendered,
                    None,
                )
                raise RuntimeError("RoboCasa RGB-D returned no depth frame")
            raw_rgb, raw_depth = rendered
            depth = np.asarray(raw_depth)
            if (
                depth.shape != expected_rgb_shape[:2]
                or not np.issubdtype(depth.dtype, np.number)
                or not np.isfinite(depth).all()
                or np.any(depth < 0.0)
                or np.any(depth > 1.0)
            ):
                _save_failed_frame(
                    "RoboCasa RGB-D returned an invalid depth frame",
                    camera_name,
                    raw_rgb,
                    raw_depth,
                )
                raise RuntimeError("RoboCasa RGB-D returned an invalid depth frame")
            self._latest_observation[depth_key] = np.array(
                depth[::-1], dtype=np.float32, copy=True
            )
        rgb = np.asarray(self._latest_observation[rgb_key])
        normalized_depth = np.asarray(
            self._latest_observation[depth_key],
            dtype=np.float32,
        )
        if normalized_depth.shape == (*expected_rgb_shape[:2], 1):
            normalized_depth = normalized_depth[..., 0]
        if normalized_depth.shape != expected_rgb_shape[:2]:
            raise RuntimeError("RoboCasa RGB-D returned an invalid depth frame")
        if (
            not np.isfinite(normalized_depth).all()
            or np.any(normalized_depth < 0.0)
            or np.any(normalized_depth > 1.0)
        ):
            raise RuntimeError("RoboCasa RGB-D normalized depth must be in [0, 1]")
        geometry = _camera_geometry(
            self.env.sim,
            camera_name,
            expected_rgb_shape[0],
            expected_rgb_shape[1],
            normalized_depth,
        )
        return _copy_rgbd_payload(
            camera_name,
            {
                "camera_name": camera_name,
                "rgb": rgb,
                **geometry,
            },
        )

    def reset(self) -> PreparedEnvironment:
        self._latest_observation = None
        transition = self._gym_environment.reset()
        if not isinstance(transition, tuple) or len(transition) != 2:
            raise RuntimeError("RoboCasa reset must return a Gymnasium 2-tuple")
        observation, info = transition
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise RuntimeError("RoboCasa reset observation and info must be mappings")
        instruction = str(
            observation.get("annotation.human.task_description", "")
        ).strip()
        return PreparedEnvironment(
            observation=self._with_overview(observation),
            info=info,
            instruction=instruction,
        )

    def step(self, action: np.ndarray):
        self._latest_observation = None
        action_dict = {
            "action.base_motion": np.array(action[0:4], copy=True),
            "action.control_mode": np.array(action[4:5], copy=True),
            "action.end_effector_position": np.array(action[5:8], copy=True),
            "action.end_effector_rotation": np.array(action[8:11], copy=True),
            "action.gripper_close": np.array(action[11:12], copy=True),
        }
        transition = self._gym_environment.step(action_dict)
        if not isinstance(transition, tuple) or len(transition) != 5:
            return transition
        observation, reward, terminated, truncated, info = transition
        if not isinstance(observation, Mapping):
            return transition
        return (
            self._with_overview(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self._latest_observation = None
        capture = getattr(self, "_narration_capture", None)
        if capture is not None:
            capture.close()
        self._gym_environment.close()


def make_upstream_environment(config: Mapping[str, Any]) -> _GymEnvironmentAdapter:
    """Construct one pinned RoboCasa task environment without owning episodes."""

    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    gym_environment = RoboCasaGymEnv(
        env_name=config["task"],
        split=config["split"],
        robots=config["robot"],
        seed=config["seed"],
        camera_widths=config["camera_size"],
        camera_heights=config["camera_size"],
        obj_registries=tuple(config["object_registries"]),
        horizon=config["upstream_horizon"],
        enable_render=True,
    )
    return _GymEnvironmentAdapter(gym_environment)


def _dependency_identity() -> dict[str, Any]:
    from robots.robocasa.dependencies import _inspect_current_environment

    report = _inspect_current_environment()
    return {
        "robocasa_commit": report["robocasa"]["commit"],
        "robosuite_commit": report["robosuite"]["commit"],
        "mujoco_version": report["mujoco"]["version"],
        "mujoco_gl": report["mujoco_gl"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    os.environ.setdefault(
        "RPENT_ROBOCASA_DIAGNOSTICS_DIR",
        str(args.config.resolve().parent / "failed_frames"),
    )
    facade = RoboCasaEnvFacade(
        environment_factory=make_upstream_environment,
        dependency_identity=_dependency_identity(),
    )
    facade.load(config)
    facade.serve(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_SCHEMA",
    "ROBOCASA_ENV_PROTOCOL",
    "PreparedEnvironment",
    "RoboCasaEnvFacade",
    "make_upstream_environment",
]
