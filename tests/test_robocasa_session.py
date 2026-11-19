from __future__ import annotations

import inspect
from copy import deepcopy

import numpy as np
import pytest

from robots.robocasa.session import RoboCasaSession
from rpent.episode.contracts import SimulatorSession


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "state": np.full(16, value, dtype=np.float32),
        "robot0_agentview_center": np.full(
            (512, 768, 3), value, dtype=np.uint8
        ),
        "robot0_agentview_left": np.full((256, 256, 3), value, dtype=np.uint8),
        "robot0_agentview_right": np.full((256, 256, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand": np.full((256, 256, 3), value, dtype=np.uint8),
    }


class _Client:
    def __init__(self) -> None:
        self.env_meta = {"task": "CloseFridge", "split": "target"}
        self.reset_calls: list[tuple[str, int]] = []
        self.observe_calls = 0
        self.step_actions: list[np.ndarray] = []
        self.rgbd_calls: list[str] = []
        self.close_calls = 0
        self.episode_id = "episode-7"
        self.step_id = 0
        self.step_overrides: dict = {}

    def reset(self, task: str, seed: int) -> dict:
        self.reset_calls.append((task, seed))
        return {
            "episode_id": self.episode_id,
            "step_id": 0,
            "observation": _observation(),
            "instruction": "Close the fridge door.",
            "effective_horizon": 3,
        }

    def observe(self) -> dict:
        self.observe_calls += 1
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation": _observation(self.step_id),
            "instruction": "Close the fridge door.",
        }

    def step(self, action: np.ndarray) -> dict:
        self.step_actions.append(np.array(action, copy=True))
        self.step_id += 1
        result = {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "observation": _observation(self.step_id),
            "reward": 0.25,
            "success": False,
            "terminated": False,
            "truncated": False,
            "horizon_reached": False,
        }
        result.update(self.step_overrides)
        return result

    def rgbd(self, camera_name: str) -> dict:
        self.rgbd_calls.append(camera_name)
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "camera_name": camera_name,
            "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
            "depth_m": np.ones((256, 256), dtype=np.float32),
            "intrinsic_K": np.eye(3, dtype=np.float64),
            "extrinsic_cam2world": np.eye(4, dtype=np.float64),
        }

    def close(self) -> None:
        self.close_calls += 1


def test_session_has_exact_simulator_session_surface() -> None:
    expected = {name for name in SimulatorSession.__dict__ if not name.startswith("_")}
    actual = {
        name
        for name, value in RoboCasaSession.__dict__.items()
        if not name.startswith("_") and (callable(value) or isinstance(value, property))
    }
    assert actual == expected | {"instruction", "effective_horizon", "rgbd"}
    assert inspect.signature(RoboCasaSession.step).parameters["action"].annotation in {
        "np.ndarray",
        np.ndarray,
    }


def test_constructor_does_not_reset_and_identity_requires_reset() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)

    assert client.reset_calls == []
    with pytest.raises(RuntimeError, match="before reset"):
        _ = session.episode_id
    with pytest.raises(RuntimeError, match="before reset"):
        _ = session.instruction
    with pytest.raises(RuntimeError, match="before reset"):
        session.observe()


def test_reset_forwards_canonical_task_and_seed_once() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)

    result = session.reset("CloseFridge", 7)

    assert client.reset_calls == [("CloseFridge", 7)]
    assert result["state"].shape == (16,)
    assert session.episode_id == "episode-7"
    assert session.instruction == "Close the fridge door."
    assert session.effective_horizon == 3
    with pytest.raises(RuntimeError, match="already been reset"):
        session.reset("CloseFridge", 7)
    assert len(client.reset_calls) == 1


@pytest.mark.parametrize(
    ("task", "seed", "message"),
    [("OpenDrawer", 7, "task"), ("CloseFridge", 8, "seed")],
)
def test_reset_rejects_case_mismatch_before_rpc(task, seed, message) -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)

    with pytest.raises(ValueError, match=message):
        session.reset(task, seed)
    assert client.reset_calls == []


def test_observe_uses_rpc_and_validates_current_identity_and_schema() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.reset("CloseFridge", 7)

    observed = session.observe()

    assert client.observe_calls == 1
    assert observed["state"].dtype == np.float32
    client.episode_id = "stale-episode"
    with pytest.raises(RuntimeError, match="episode_id"):
        session.observe()


def test_rgbd_forwards_camera_and_returns_an_owned_current_frame() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)

    with pytest.raises(RuntimeError, match="before reset"):
        session.rgbd("robot0_agentview_left")

    session.reset("CloseFridge", 7)
    frame = session.rgbd("robot0_agentview_left")

    assert frame["step_id"] == 0
    assert frame["camera_name"] == "robot0_agentview_left"
    assert frame["depth_m"].shape == (256, 256)
    assert client.rgbd_calls == ["robot0_agentview_left"]
    frame["depth_m"][:] = 99
    assert np.all(client.rgbd("robot0_agentview_left")["depth_m"] == 1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_id", "stale-episode", "episode_id"),
        ("step_id", 1, "step_id"),
        ("camera_name", "robot0_eye_in_hand", "camera"),
    ],
)
def test_rgbd_rejects_stale_or_mismatched_identity(field, value, message) -> None:
    client = _Client()
    original_rgbd = client.rgbd

    def malformed(camera_name):
        result = original_rgbd(camera_name)
        result[field] = value
        return result

    client.rgbd = malformed
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.reset("CloseFridge", 7)

    with pytest.raises(RuntimeError, match=message):
        session.rgbd("robot0_agentview_left")


def test_step_builds_monotonic_result_and_preserves_terminal_signals() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.reset("CloseFridge", 7)
    client.step_overrides = {"success": True, "terminated": False, "truncated": False}

    result = session.step(np.zeros(12, dtype=np.float32))

    assert result.episode_id == "episode-7"
    assert result.step_id == 1
    assert result.reward == pytest.approx(0.25)
    assert result.success is True
    assert result.terminated is False
    assert result.truncated is False
    assert len(client.step_actions) == 1
    with pytest.raises(RuntimeError, match="terminal"):
        session.step(np.zeros(12, dtype=np.float32))
    assert len(client.step_actions) == 1


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(11, dtype=np.float32),
        np.zeros(12, dtype=np.float64),
        np.full(12, np.nan, dtype=np.float32),
        np.full(12, 1.01, dtype=np.float32),
        [0.0] * 12,
    ],
)
def test_invalid_action_fails_before_rpc(action) -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.reset("CloseFridge", 7)

    with pytest.raises((TypeError, ValueError), match="action"):
        session.step(action)
    assert client.step_actions == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"episode_id": "stale"}, "episode_id"),
        ({"step_id": 3}, "step_id"),
        ({"observation": {"state": np.zeros(16, dtype=np.float32)}}, "keys"),
        ({"success": 1}, "success"),
    ],
)
def test_malformed_transition_fails_closed(overrides, message) -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.reset("CloseFridge", 7)
    client.step_overrides = deepcopy(overrides)

    with pytest.raises((RuntimeError, ValueError), match=message):
        session.step(np.zeros(12, dtype=np.float32))


def test_close_is_idempotent() -> None:
    client = _Client()
    session = RoboCasaSession(client, task="CloseFridge", seed=7)
    session.close()
    session.close()
    assert client.close_calls == 1


def test_each_attached_case_uses_a_fresh_single_episode_session() -> None:
    first_client = _Client()
    first = RoboCasaSession(first_client, task="CloseFridge", seed=7)
    first.reset("CloseFridge", 7)
    first.close()

    second_client = _Client()
    second_client.episode_id = "episode-8"
    second = RoboCasaSession(second_client, task="CloseFridge", seed=7)
    second.reset("CloseFridge", 7)

    assert first_client.close_calls == 1
    assert first_client.reset_calls == [("CloseFridge", 7)]
    assert second_client.reset_calls == [("CloseFridge", 7)]
    assert second.episode_id == "episode-8"
    with pytest.raises(RuntimeError, match="already been reset"):
        second.reset("CloseFridge", 7)
