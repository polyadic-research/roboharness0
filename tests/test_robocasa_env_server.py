from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pytest

import robots.robocasa.env_server as env_server
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.env_server import (
    ROBOCASA_ENV_PROTOCOL,
    PreparedEnvironment,
    RoboCasaEnvFacade,
    _GymEnvironmentAdapter,
    make_upstream_environment,
)


def _upstream_observation(value: int = 0) -> dict:
    return {
        "state.base_position": np.full(3, value, dtype=np.float32),
        "state.base_rotation": np.full(4, value, dtype=np.float32),
        "state.end_effector_position_relative": np.full(3, value, dtype=np.float32),
        "state.end_effector_rotation_relative": np.full(4, value, dtype=np.float32),
        "state.gripper_qpos": np.full(2, value, dtype=np.float32),
        "video.robot0_agentview_center": np.full((512, 768, 3), value, dtype=np.uint8),
        "video.robot0_agentview_left": np.full((256, 256, 3), value, dtype=np.uint8),
        "video.robot0_agentview_left_depth": np.full(
            (256, 256, 1), 0.25, dtype=np.float32
        ),
        "video.robot0_agentview_right": np.full((256, 256, 3), value, dtype=np.uint8),
        "video.robot0_agentview_right_depth": np.full(
            (256, 256, 1), 0.5, dtype=np.float32
        ),
        "video.robot0_eye_in_hand": np.full((256, 256, 3), value, dtype=np.uint8),
        "video.robot0_eye_in_hand_depth": np.full(
            (256, 256, 1), 0.75, dtype=np.float32
        ),
    }


def _config(**overrides) -> dict:
    values = {
        "task": "CloseFridge",
        "split": "target",
        "seed": 7,
        "robot": "PandaOmron",
        "camera_names": [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        "camera_size": 256,
        "object_registries": ["lightwheel"],
        "rendering_mode": "headless",
        "upstream_horizon": 5,
        "safety_ceiling": 3,
        "task_registry_commit": ROBOCASA_COMMIT,
        "task_registry_digest": "sha256:" + "1" * 64,
    }
    values.update(overrides)
    return values


@dataclass
class _Transition:
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0


class _FakeEnvironment:
    def __init__(self, transitions: list[_Transition]) -> None:
        self.transitions = list(transitions)
        self.step_calls: list[np.ndarray] = []
        self.close_calls = 0
        self.reset_calls = 0
        self._current_success = False
        self.rgbd_calls: list[str] = []

    def reset(self) -> PreparedEnvironment:
        self.reset_calls += 1
        return PreparedEnvironment(
            observation=_upstream_observation(self.reset_calls),
            info={"success": False},
            instruction=f"Close the fridge door. Reset {self.reset_calls}.",
        )

    def step(self, action: np.ndarray):
        self.step_calls.append(np.array(action, copy=True))
        transition = self.transitions.pop(0) if self.transitions else _Transition()
        self._current_success = transition.success
        return (
            _upstream_observation(len(self.step_calls)),
            transition.reward,
            transition.terminated,
            transition.truncated,
            {"success": not transition.success},
        )

    def _check_success(self) -> bool:
        return self._current_success

    def rgbd(self, camera_name: str) -> dict:
        self.rgbd_calls.append(camera_name)
        if camera_name == "robot0_agentview_center":
            height, width = 512, 768
        else:
            height = width = 256
        value = len(self.step_calls)
        return {
            "camera_name": camera_name,
            "rgb": np.full((height, width, 3), value, dtype=np.uint8),
            "depth_m": np.full((height, width), value + 1.0, dtype=np.float32),
            "intrinsic_K": np.eye(3, dtype=np.float64),
            "extrinsic_cam2world": np.eye(4, dtype=np.float64),
        }

    def close(self) -> None:
        self.close_calls += 1


def _facade(fake: _FakeEnvironment) -> tuple[RoboCasaEnvFacade, list[dict]]:
    factory_calls = []

    def factory(config):
        factory_calls.append(dict(config))
        return fake

    facade = RoboCasaEnvFacade(
        environment_factory=factory,
        dependency_identity={
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "mujoco_version": "3.3.2",
        },
    )
    return facade, factory_calls


def test_load_configures_without_constructing_or_resetting_environment() -> None:
    fake = _FakeEnvironment([])
    facade, factory_calls = _facade(fake)

    loaded = facade.load(_config())

    assert loaded["loaded"] is True
    assert factory_calls == []
    assert fake.reset_calls == 0


def test_reset_constructs_once_and_returns_owned_16d_four_camera_observation() -> None:
    fake = _FakeEnvironment([])
    facade, factory_calls = _facade(fake)
    facade.load(_config())

    result = facade.reset("CloseFridge", 7)

    assert fake.reset_calls == 1
    assert len(factory_calls) == 1
    assert result["episode_id"]
    assert result["step_id"] == 0
    assert result["instruction"] == "Close the fridge door. Reset 1."
    assert result["effective_horizon"] == 3
    assert result["observation"]["state"].shape == (16,)
    assert result["observation"]["robot0_agentview_center"].shape == (
        512,
        768,
        3,
    )
    assert result["observation"]["robot0_agentview_left"].shape == (
        256,
        256,
        3,
    )
    assert result["observation"]["robot0_agentview_right"].shape == (
        256,
        256,
        3,
    )
    assert result["observation"]["robot0_eye_in_hand"].shape == (256, 256, 3)

    second = facade.reset("CloseFridge", 7)
    assert fake.reset_calls == 2
    assert len(factory_calls) == 1
    assert second["episode_id"] != result["episode_id"]
    assert second["instruction"] == "Close the fridge door. Reset 2."
    assert np.all(second["observation"]["state"] == 2)


def test_fifty_resets_reuse_one_seeded_environment_in_order() -> None:
    fake = _FakeEnvironment([])
    facade, factory_calls = _facade(fake)
    facade.load(_config(seed=7))

    resets = [facade.reset("CloseFridge", 7) for _ in range(50)]

    assert len(factory_calls) == 1
    assert factory_calls[0]["seed"] == 7
    assert fake.reset_calls == 50
    assert len({result["episode_id"] for result in resets}) == 50
    assert [result["instruction"] for result in resets] == [
        f"Close the fridge door. Reset {index}." for index in range(1, 51)
    ]


def test_reset_clears_terminal_step_and_observation_state() -> None:
    fake = _FakeEnvironment([_Transition(success=True), _Transition()])
    facade, _ = _facade(fake)
    facade.load(_config())
    first = facade.reset("CloseFridge", 7)
    terminal = facade.step(np.zeros(12, dtype=np.float32))

    second = facade.reset("CloseFridge", 7)
    transition = facade.step(np.zeros(12, dtype=np.float32))

    assert terminal["success"] is True
    assert second["episode_id"] != first["episode_id"]
    assert second["step_id"] == 0
    assert np.all(second["observation"]["state"] == 2)
    assert transition["episode_id"] == second["episode_id"]
    assert transition["step_id"] == 1
    assert transition["success"] is False


def test_failed_repeated_reset_invalidates_the_previous_episode() -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    def fail_reset():
        raise RuntimeError("upstream reset failed")

    fake.reset = fail_reset

    with pytest.raises(RuntimeError, match="upstream reset failed"):
        facade.reset("CloseFridge", 7)
    with pytest.raises(RuntimeError, match="before reset"):
        facade.observe()
    with pytest.raises(RuntimeError, match="before reset"):
        facade.step(np.zeros(12, dtype=np.float32))


@pytest.mark.parametrize(
    "malformed",
    [
        object(),
        PreparedEnvironment(
            observation=_upstream_observation(),
            info={},
            instruction="",
        ),
        PreparedEnvironment(
            observation={"state.base_position": np.zeros(3, dtype=np.float32)},
            info={},
            instruction="Fresh instruction.",
        ),
    ],
)
def test_malformed_repeated_reset_invalidates_the_previous_episode(
    malformed,
) -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)
    fake.reset = lambda: malformed

    with pytest.raises((RuntimeError, ValueError)):
        facade.reset("CloseFridge", 7)
    with pytest.raises(RuntimeError, match="before reset"):
        facade.observe()
    with pytest.raises(RuntimeError, match="before reset"):
        facade.step(np.zeros(12, dtype=np.float32))


@pytest.mark.parametrize(
    ("task", "seed", "message"),
    [("OpenDrawer", 7, "task mismatch"), ("CloseFridge", 8, "seed mismatch")],
)
def test_repeated_reset_rejects_task_or_seed_change(task, seed, message) -> None:
    fake = _FakeEnvironment([])
    facade, factory_calls = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    with pytest.raises(ValueError, match=message):
        facade.reset(task, seed)

    assert len(factory_calls) == 1
    assert fake.reset_calls == 1


def test_reset_after_close_and_step_before_reset_are_rejected() -> None:
    fake = _FakeEnvironment([])
    facade, factory_calls = _facade(fake)
    facade.load(_config())

    with pytest.raises(RuntimeError, match="before reset"):
        facade.step(np.zeros(12, dtype=np.float32))
    facade.close()
    with pytest.raises(RuntimeError, match="closed"):
        facade.reset("CloseFridge", 7)

    assert factory_calls == []
    assert fake.close_calls == 0


def test_step_sends_exactly_one_12d_action_and_advances_once() -> None:
    fake = _FakeEnvironment([_Transition(reward=0.25)])
    facade, _ = _facade(fake)
    facade.load(_config())
    reset = facade.reset("CloseFridge", 7)

    result = facade.step(np.zeros(12, dtype=np.float32))

    assert len(fake.step_calls) == 1
    assert fake.step_calls[0].shape == (12,)
    assert result["episode_id"] == reset["episode_id"]
    assert result["step_id"] == 1
    assert result["reward"] == 0.25
    assert result["success"] is False
    assert result["terminated"] is False
    assert result["truncated"] is False


def test_step_still_rejects_non_gymnasium_transition_shape() -> None:
    fake = _FakeEnvironment([])
    fake.step = lambda action: (_upstream_observation(), 0.0, False, {})
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    with pytest.raises(RuntimeError, match="Gymnasium 5-tuple"):
        facade.step(np.zeros(12, dtype=np.float32))


@pytest.mark.parametrize(
    ("transition", "terminal_field"),
    [
        (_Transition(success=True, reward=0.0), "success"),
        (_Transition(terminated=True, reward=0.0), "terminated"),
        (_Transition(truncated=True, reward=0.0), "truncated"),
    ],
)
def test_success_termination_and_truncation_are_independent_and_latch_terminal(
    transition: _Transition,
    terminal_field: str,
) -> None:
    fake = _FakeEnvironment([transition, _Transition()])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    result = facade.step(np.zeros(12, dtype=np.float32))

    assert result[terminal_field] is True
    for field in {"success", "terminated", "truncated"} - {terminal_field}:
        assert result[field] is False
    with pytest.raises(RuntimeError, match="terminal"):
        facade.step(np.zeros(12, dtype=np.float32))
    assert len(fake.step_calls) == 1
    assert fake.reset_calls == 1


def test_effective_horizon_truncates_without_auto_reset() -> None:
    fake = _FakeEnvironment([_Transition(), _Transition(), _Transition()])
    facade, factory_calls = _facade(fake)
    facade.load(_config(upstream_horizon=4, safety_ceiling=3))
    facade.reset("CloseFridge", 7)

    first = facade.step(np.zeros(12, dtype=np.float32))
    second = facade.step(np.zeros(12, dtype=np.float32))
    final = facade.step(np.zeros(12, dtype=np.float32))

    assert first["truncated"] is False
    assert second["truncated"] is False
    assert final["truncated"] is True
    assert final["horizon_reached"] is True
    assert fake.reset_calls == 1
    assert len(factory_calls) == 1


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(11, dtype=np.float32),
        np.full(12, np.nan, dtype=np.float32),
        np.full(12, 1.1, dtype=np.float32),
    ],
)
def test_invalid_action_fails_before_upstream_step(action: np.ndarray) -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    with pytest.raises(ValueError, match="action"):
        facade.step(action)
    assert fake.step_calls == []


def test_observe_does_not_advance_and_close_is_idempotent() -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    reset = facade.reset("CloseFridge", 7)

    observed = facade.observe()
    facade.close()
    facade.close()

    assert observed["episode_id"] == reset["episode_id"]
    assert observed["step_id"] == 0
    assert fake.step_calls == []
    assert fake.close_calls == 1


def test_rgbd_is_step_identified_read_only_and_available_after_terminal() -> None:
    fake = _FakeEnvironment([_Transition(success=True)])
    facade, _ = _facade(fake)
    facade.load(_config())

    with pytest.raises(RuntimeError, match="before reset"):
        facade.rgbd("robot0_agentview_left")

    reset = facade.reset("CloseFridge", 7)
    initial = facade._dispatch("env.rgbd", ("robot0_agentview_left",), {})
    terminal = facade.step(np.zeros(12, dtype=np.float32))
    final = facade.rgbd("robot0_agentview_left")

    assert initial["episode_id"] == reset["episode_id"]
    assert initial["step_id"] == 0
    assert initial["depth_m"].shape == (256, 256)
    assert terminal["success"] is True
    assert final["episode_id"] == reset["episode_id"]
    assert final["step_id"] == 1
    assert float(final["depth_m"][0, 0]) == pytest.approx(2.0)
    assert fake.rgbd_calls == [
        "robot0_agentview_left",
        "robot0_agentview_left",
    ]


def test_rgbd_rejects_unknown_camera_before_environment_access() -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    with pytest.raises(ValueError, match="camera"):
        facade.rgbd("fixture_truth_camera")

    assert fake.rgbd_calls == []


def test_close_invalidates_episode_for_observe_and_step() -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade.load(_config())
    facade.reset("CloseFridge", 7)

    facade.close()

    with pytest.raises(RuntimeError, match="closed"):
        facade.observe()
    with pytest.raises(RuntimeError, match="closed"):
        facade.step(np.zeros(12, dtype=np.float32))


def test_health_and_rpc_dispatch_expose_versioned_identity_without_chunk_step() -> None:
    fake = _FakeEnvironment([])
    facade, _ = _facade(fake)
    facade._dispatch("env.load", (_config(),), {})

    health = facade._dispatch("env.health", (), {})

    assert health["protocol"] == ROBOCASA_ENV_PROTOCOL == "rpent-robocasa-env-v2"
    assert health["task"] == "CloseFridge"
    assert health["split"] == "target"
    assert health["object_registries"] == ["lightwheel"]
    assert health["task_registry_commit"] == ROBOCASA_COMMIT
    assert health["observation_schema"] == "robocasa-pandaomron-observation-v2"
    assert health["action_schema"] == "robocasa-pandaomron-action-v1"
    assert health["dependencies"]["robosuite_commit"] == ROBOSUITE_COMMIT

    with pytest.raises(ValueError, match="unknown RPC method"):
        facade._dispatch("env.chunk_step", (np.zeros((2, 12)),), {})


def test_gym_adapter_preserves_checkpoint_actions_and_caches_latest_rgbd(
    monkeypatch,
) -> None:
    overview = np.zeros((512, 768, 3), dtype=np.uint8)
    overview[0] = 11
    overview[-1] = 22
    overview_depth = np.full((512, 768), 0.5, dtype=np.float32)
    overview_depth[0] = 0.25
    overview_depth[-1] = 0.75

    def camera_geometry(sim, camera_name, height, width, normalized_depth):
        del sim, camera_name
        assert normalized_depth.shape == (height, width)
        return {
            "depth_m": normalized_depth + 10.0,
            "intrinsic_K": np.eye(3, dtype=np.float64),
            "extrinsic_cam2world": np.eye(4, dtype=np.float64),
        }

    monkeypatch.setattr(env_server, "_camera_geometry", camera_geometry, raising=False)

    class Sim:
        def __init__(self) -> None:
            self.calls = []

        def render(self, **kwargs):
            self.calls.append(kwargs)
            return overview, overview_depth

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return {
                **_upstream_observation(4),
                "annotation.human.task_description": "Fresh instruction.",
            }, {"success": False}

        def step(self, action):
            self.action = action
            return _upstream_observation(5), 0.5, False, False, {"success": False}

    gym = Gym()
    adapter = _GymEnvironmentAdapter(gym)
    action = np.arange(12, dtype=np.float32)

    prepared = adapter.reset()
    transition = adapter.step(action)

    assert prepared.instruction == "Fresh instruction."
    assert np.all(prepared.observation["state.base_position"] == 4)
    assert np.all(prepared.observation["video.robot0_agentview_center"][0] == 22)
    assert np.all(prepared.observation["video.robot0_agentview_center"][-1] == 11)
    assert np.all(transition[0]["video.robot0_agentview_center"][0] == 22)
    rgbd = adapter.rgbd("robot0_agentview_center")
    assert np.all(rgbd["rgb"][0] == 22)
    assert np.allclose(rgbd["depth_m"][0], 10.75)
    assert rgbd["intrinsic_K"].shape == (3, 3)
    assert rgbd["extrinsic_cam2world"].shape == (4, 4)
    assert gym.env.sim.calls == [
        {
            "height": 512,
            "width": 768,
            "camera_name": "robot0_agentview_center",
            "depth": True,
        },
        {
            "height": 512,
            "width": 768,
            "camera_name": "robot0_agentview_center",
            "depth": True,
        },
    ]
    overview[:] = 99
    assert np.all(prepared.observation["video.robot0_agentview_center"][0] == 22)
    assert np.array_equal(gym.action["action.base_motion"], action[0:4])
    assert np.array_equal(gym.action["action.control_mode"], action[4:5])
    assert np.array_equal(gym.action["action.end_effector_position"], action[5:8])
    assert np.array_equal(gym.action["action.end_effector_rotation"], action[8:11])
    assert np.array_equal(gym.action["action.gripper_close"], action[11:12])
    assert all(value.dtype == np.float32 for value in gym.action.values())


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((256, 256, 3), dtype=np.uint8), "invalid RGB frame"),
        (np.zeros((512, 768, 3), dtype=np.float32), "invalid RGB frame"),
    ],
)
def test_gym_adapter_fails_closed_on_malformed_overview(frame, message) -> None:
    class Sim:
        def render(self, **kwargs):
            return frame, np.zeros((512, 768), dtype=np.float32)

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return _upstream_observation(), {}

    with pytest.raises(RuntimeError, match=message):
        _GymEnvironmentAdapter(Gym()).reset()


def test_gym_adapter_fails_closed_when_overview_render_raises() -> None:
    class Sim:
        def render(self, **kwargs):
            raise RuntimeError("render failed")

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return _upstream_observation(), {}

    with pytest.raises(RuntimeError, match="render failed"):
        _GymEnvironmentAdapter(Gym()).reset()


@pytest.mark.parametrize("failure", ["nan", "out_of_range", "shape"])
def test_overview_failure_retains_raw_frames_and_diagnostics(
    monkeypatch, tmp_path, failure
) -> None:
    directory = tmp_path / "failed_frames"
    monkeypatch.setenv("RPENT_ROBOCASA_DIAGNOSTICS_DIR", str(directory))
    rgb = np.zeros((512, 768, 3), dtype=np.uint8)
    rgb[0] = 123
    depth = np.full((512, 768), 0.5, dtype=np.float32)
    expected_check = {
        "nan": "depth_nonfinite",
        "out_of_range": "depth_out_of_range",
        "shape": "depth_shape",
    }[failure]
    if failure == "nan":
        depth[0, 0] = np.nan
    elif failure == "out_of_range":
        depth[0, 0] = np.nextafter(np.float32(1), np.float32(2))
    else:
        depth = depth[:3]

    class Sim:
        def render(self, **kwargs):
            return rgb, depth

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return _upstream_observation(), {}

    with pytest.raises(
        RuntimeError, match="overview camera returned invalid depth frame"
    ):
        _GymEnvironmentAdapter(Gym()).reset()

    saved = list(directory.iterdir())
    assert len(saved) == 1
    record = saved[0]
    np.testing.assert_array_equal(np.load(record / "raw_rgb.npy"), rgb)
    np.testing.assert_array_equal(np.load(record / "raw_depth.npy"), depth)
    assert (record / "rgb.png").is_file()
    assert (record / "depth_preview.png").is_file()
    metadata = json.loads((record / "diagnostic.json").read_text())
    assert expected_check in metadata["failed_checks"]
    assert metadata["depth"]["shape"] == list(depth.shape)
    assert metadata["depth"]["dtype"] == "float32"
    assert metadata["depth"]["nonfinite_count"] == (1 if failure == "nan" else 0)
    assert metadata["depth"]["above_one_count"] == (
        1 if failure == "out_of_range" else 0
    )


def test_frame_save_failure_preserves_original_simulator_error(
    monkeypatch, tmp_path, caplog
) -> None:
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied")
    monkeypatch.setenv("RPENT_ROBOCASA_DIAGNOSTICS_DIR", str(blocked))

    class Sim:
        def render(self, **kwargs):
            return np.zeros((512, 768, 3), dtype=np.uint8), np.full(
                (512, 768), np.nan, dtype=np.float32
            )

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return _upstream_observation(), {}

    with pytest.raises(
        RuntimeError, match="overview camera returned invalid depth frame"
    ):
        _GymEnvironmentAdapter(Gym()).reset()
    assert "Could not save failed RoboCasa frame" in caplog.text


def test_make_upstream_environment_uses_upstream_camera_depth_default(
    monkeypatch,
) -> None:
    captured = {}

    class Gym:
        def __init__(self, **kwargs):
            # RoboCasa 1.0.1 sets this default inside create_env. Passing it
            # through kwargs raises before an environment can be constructed.
            dict(camera_depths=False, **kwargs)
            captured.update(kwargs)
            self.env = object()

    robocasa = ModuleType("robocasa")
    wrappers = ModuleType("robocasa.wrappers")
    gym_wrapper = ModuleType("robocasa.wrappers.gym_wrapper")
    gym_wrapper.RoboCasaGymEnv = Gym
    monkeypatch.setitem(sys.modules, "robocasa", robocasa)
    monkeypatch.setitem(sys.modules, "robocasa.wrappers", wrappers)
    monkeypatch.setitem(sys.modules, "robocasa.wrappers.gym_wrapper", gym_wrapper)

    adapter = make_upstream_environment(_config())

    assert isinstance(adapter, _GymEnvironmentAdapter)
    assert "camera_depths" not in captured


@pytest.mark.parametrize(
    "camera",
    [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
)
def test_policy_depth_is_rendered_read_only_cached_and_invalidated_per_step(
    monkeypatch, camera
):
    class Sim:
        def __init__(self):
            self.step = 0
            self.calls = []

        def render(self, *, height, width, camera_name, depth):
            assert depth is True
            self.calls.append((self.step, camera_name))
            rgb = np.full((height, width, 3), 99, dtype=np.uint8)
            z = np.full((height, width), 0.4 + self.step / 10, dtype=np.float32)
            z[0] = 0.2
            return rgb, z

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def observation(self):
            return {
                key: value
                for key, value in _upstream_observation(7).items()
                if not key.endswith("_depth")
            }

        def reset(self):
            return self.observation(), {}

        def step(self, action):
            self.env.sim.step += 1
            return self.observation(), 0, False, False, {}

    monkeypatch.setattr(
        env_server,
        "_camera_geometry",
        lambda *args: {
            "depth_m": args[-1] + 1,
            "intrinsic_K": np.eye(3),
            "extrinsic_cam2world": np.eye(4),
        },
    )
    gym = Gym()
    adapter = _GymEnvironmentAdapter(gym)
    prepared = adapter.reset()
    first = adapter.rgbd(camera)
    second = adapter.rgbd(camera)
    assert np.array_equal(first["rgb"], prepared.observation["video." + camera])
    assert np.all(first["rgb"] == 7)  # Original policy RGB, not the new render.
    assert np.allclose(first["depth_m"][0], 1.4)
    assert np.allclose(first["depth_m"][-1], 1.2)  # Same vertical orientation.
    assert np.array_equal(first["depth_m"], second["depth_m"])
    assert gym.env.sim.step == 0
    assert gym.env.sim.calls.count((0, camera)) == 1
    adapter.step(np.zeros(12, dtype=np.float32))
    refreshed = adapter.rgbd(camera)
    assert np.allclose(refreshed["depth_m"][0], 1.5)
    assert gym.env.sim.calls.count((1, camera)) == 1


@pytest.mark.parametrize("failure", ["missing", "shape", "nan", "range"])
def test_on_demand_depth_fails_closed_and_is_not_cached(failure):
    class Sim:
        def render(self, *, height, width, **kwargs):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            if height != 512:
                if failure == "missing":
                    return rgb
                if failure == "shape":
                    depth = depth[:1]
                elif failure == "nan":
                    depth[0, 0] = np.nan
                elif failure == "range":
                    depth[0, 0] = 1.1
            return rgb, depth

    class Gym:
        env = type("Env", (), {"sim": Sim()})()

        def reset(self):
            return {
                key: value
                for key, value in _upstream_observation().items()
                if not key.endswith("_depth")
            }, {}

    adapter = _GymEnvironmentAdapter(Gym())
    adapter.reset()
    with pytest.raises(RuntimeError, match="depth frame"):
        adapter.rgbd("robot0_agentview_left")
    assert "video.robot0_agentview_left_depth" not in adapter._latest_observation
