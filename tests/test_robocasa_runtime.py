from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.daemon import RoboCasaDaemon
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.env_server import ACTION_SCHEMA, ROBOCASA_ENV_PROTOCOL
from robots.robocasa.observation import OBSERVATION_SCHEMA
from robots.robocasa.runtime import (
    _default_policy_daemon_builder,
    build_episode_bootstrap,
)
from rpent.episode import open_episode
from rpent.episode.contracts import StepResult


class _Dependency:
    def __init__(self, name, events, resource=None):
        self.name = name
        self.events = events
        self.resource = resource
        self.started = False

    def start(self):
        if not self.started:
            self.events.append(f"start:{self.name}")
            self.started = True
        return self.resource

    def stop(self):
        self.events.append(f"stop:{self.name}")


class _Simulator:
    episode_id = "episode-1"
    instruction = "Live upstream instruction."
    effective_horizon = 300

    def __init__(self, events):
        self.events = events
        self.reset_calls = []
        self.observation = {
            "state": np.zeros(16, dtype=np.float32),
            "robot0_agentview_center": np.zeros(
                (512, 768, 3), dtype=np.uint8
            ),
            "robot0_agentview_left": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_agentview_right": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand": np.zeros((256, 256, 3), dtype=np.uint8),
        }

    def reset(self, task, seed):
        self.events.append(f"reset:simulator:{task}:{seed}")
        self.reset_calls.append((task, seed))
        return self.observation

    def observe(self):
        return self.observation

    def step(self, action):
        return StepResult(
            self.episode_id, 1, self.observation, 0.0, False, False, False
        )

    def close(self):
        self.events.append("close:simulator")


class _Policy:
    def __init__(self, events):
        self.events = events
        self.reset_ids = []

    def reset(self, episode_id):
        self.events.append(f"reset:policy:{episode_id}")
        self.reset_ids.append(episode_id)

    def predict(self, observation, instruction, *, max_actions=None):
        del observation, instruction, max_actions
        return np.zeros((1, 12), dtype=np.float32)

    def close(self):
        self.events.append("close:policy")


class _Toolkit:
    def __init__(self, events, instruction):
        self.events = events
        self.instruction = instruction

    def close(self):
        self.events.append("close:toolkit")


def _args():
    return Namespace(
        planning_profile="robocasa-hooked-gated-v1",
        task="CloseFridge",
        seed=7,
        max_episode_steps=300,
        object_registries=("lightwheel",),
        env_endpoint=None,
        policy_endpoint=None,
        policy_device="cuda:0",
        policy_dtype="float32",
        actions_per_policy_request=25,
        output_dir=None,
    )


def _task():
    return SimpleNamespace(
        name="CloseFridge",
        registry_id="module:CloseFridge",
        instruction="Registry instruction.",
        family="atomic",
        category="fixtures",
        split="target",
        upstream_horizon=500,
        registry_repository="https://github.com/robocasa/robocasa.git",
        registry_commit="456174f62b89b8fca99eaaf33949c29fec9cfc2a",
        registry_digest="sha256:" + "1" * 64,
    )


@pytest.mark.parametrize("family", ["xiaomi-mibot"])
@pytest.mark.parametrize("disabled", [False, True])
def test_policy_language_and_guidance_are_family_selected(tmp_path, family, disabled):
    args = _args()
    args.vla = family
    args.no_forward_step_guidance = disabled
    args.actions_per_policy_request = 16 if family == "xiaomi-mibot" else 5
    simulator = _Simulator([])
    captured = []
    bootstrap = build_episode_bootstrap(
        args, tmp_path, None,
        task_resolver=lambda *a, **kw: (_task(),),
        environment_builder=lambda **kw: (None, lambda: simulator, {}),
        policy_daemon_builder=lambda **kw: _Dependency("policy", [], _Policy([])),
        toolkit_builder=lambda *a, **kw: captured.append(kw),
    )
    bootstrap.toolkit_factory(None, simulator.observation)
    language = family in {"pi05", "xiaomi-mibot"}
    assert captured[0]["policy_instruction"] == (simulator.instruction if language else _task().name)
    assert captured[0]["forward_step_guidance"] is (language and not disabled)
    assert bootstrap.runtime_metadata["policy_instruction_source"] == (
        "environment_language" if language else "training_task_id")
    assert bootstrap.runtime_metadata["policy_instruction"] == captured[0]["policy_instruction"]


def test_xiaomi_default_profile_selects_daemon_and_requires_snapshot(monkeypatch, tmp_path):
    from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
    from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon

    args = _args()
    args.vla = "xiaomi-mibot"
    args.policy_dtype = "bfloat16"
    monkeypatch.delenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", raising=False)
    with pytest.raises(RuntimeError, match="RPENT_XIAOMI_MIBOT_SNAPSHOT"):
        _default_policy_daemon_builder(args=args, binding=make_xiaomi_policy_binding(), output_dir=tmp_path)
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    daemon = _default_policy_daemon_builder(args=args, binding=make_xiaomi_policy_binding(), output_dir=tmp_path)
    assert isinstance(daemon, XiaomiMiBoTDaemon)
    assert daemon.dtype == args.policy_dtype
    assert daemon.log_path.name == "xiaomi_policy_server.log"




def test_runtime_composes_direct_simulator_shared_policy_binding_and_toolkit(
    tmp_path,
) -> None:
    events = []
    simulator = _Simulator(events)
    policy = _Policy(events)
    env_dependency = _Dependency("robocasa", events, simulator)
    policy_dependency = _Dependency("smolvla", events, policy)
    toolkit_arguments = []

    bootstrap = build_episode_bootstrap(
        _args(),
        tmp_path,
        None,
        task_resolver=lambda selector, **kwargs: (_task(),),
        environment_builder=lambda **kwargs: (
            env_dependency,
            lambda: simulator,
            kwargs,
        ),
        policy_daemon_builder=lambda **kwargs: policy_dependency,
        toolkit_builder=lambda controller, observation, **kwargs: (
            toolkit_arguments.append(kwargs),
            _Toolkit(events, kwargs["instruction"]),
        )[1],
    )

    assert bootstrap.preset_name == "robocasa365-pandaomron-smolvla-v1"
    assert bootstrap.task == "CloseFridge"
    assert bootstrap.instruction == "Registry instruction."
    assert bootstrap.binding.name == "robocasa-smolvla-v1"
    assert tuple(
        camera.camera_id for camera in bootstrap.display_adapter.descriptors
    ) == (
        "robot0_agentview_center",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    assert bootstrap.dependencies == (env_dependency, policy_dependency)
    metadata = dict(bootstrap.runtime_metadata or {})
    assert metadata["execution_mode"] == "rpent_harness"
    assert metadata["task"] == "CloseFridge"
    assert metadata["policy_instruction"] == "CloseFridge"
    assert metadata["policy_instruction_source"] == "training_task_id"
    assert metadata["forward_step_guidance"] is False
    assert metadata["family"] == "atomic"
    assert metadata["category"] == "fixtures"
    assert metadata["split"] == "target"
    assert metadata["upstream_horizon"] == 500
    assert metadata["effective_horizon"] == 300
    assert metadata["object_registries"] == ["lightwheel"]
    assert metadata["simulator_commit"] == _task().registry_commit
    assert metadata["observation_schema"] == OBSERVATION_SCHEMA
    assert metadata["checkpoint"]["revision"] == (
        "38d38bf58e1217092c5a81418fcc8eef1a7c3a6f"
    )

    with open_episode(bootstrap) as runtime:
        assert runtime.simulator is simulator
        assert runtime.policy is policy
        assert simulator.reset_calls == [("CloseFridge", 7)]
        assert policy.reset_ids == ["episode-1"]
        assert toolkit_arguments == [
            {
                "instruction": "Live upstream instruction.",
                "policy_instruction": "CloseFridge",
                "forward_step_guidance": False,
                "perception_source": simulator,
                "actions_per_policy_request": 25,
                "episode_horizon": 300,
                "agent_artifact_root": tmp_path / "agent_artifacts",
                "dashboard": None,
            }
        ]

    assert events == [
        "start:robocasa",
        "start:smolvla",
        "reset:simulator:CloseFridge:7",
        "reset:policy:episode-1",
        "close:toolkit",
        "close:policy",
        "close:simulator",
        "stop:smolvla",
        "stop:robocasa",
    ]


def test_xiaomi_runtime_selects_native_binding_and_canonical_task(tmp_path) -> None:
    args = _args()
    args.planning_profile = "robocasa-xiaomi-mibot-gated-v1"
    args.actions_per_policy_request = 16
    captured = {}

    bootstrap = build_episode_bootstrap(
        args,
        tmp_path,
        None,
        task_resolver=lambda selector, **kwargs: (_task(),),
        environment_builder=lambda **kwargs: (
            None,
            lambda: _Simulator([]),
            kwargs,
        ),
        policy_daemon_builder=lambda **kwargs: captured.update(kwargs) or None,
        toolkit_builder=lambda controller, observation, **kwargs: object(),
    )

    assert captured["binding"].name == "robocasa-xiaomi-mibot-v1"
    assert bootstrap.binding.name == "robocasa-xiaomi-mibot-v1"
    assert bootstrap.preset_name == "robocasa365-pandaomron-xiaomi-mibot-v1"
    assert bootstrap.runtime_metadata["policy_family"] == "xiaomi-mibot"
    assert bootstrap.runtime_metadata["policy_instruction"] == "Registry instruction."


def test_runtime_defaults_to_upstream_horizon_without_ceiling(tmp_path) -> None:
    args = _args()
    args.max_episode_steps = None
    environment_calls = []

    bootstrap = build_episode_bootstrap(
        args,
        tmp_path,
        None,
        task_resolver=lambda selector, **kwargs: (_task(),),
        environment_builder=lambda **kwargs: (
            environment_calls.append(kwargs),
            lambda: object(),
            {},
        ),
        policy_daemon_builder=lambda **kwargs: None,
        toolkit_builder=lambda controller, observation, **kwargs: object(),
    )

    metadata = dict(bootstrap.runtime_metadata or {})
    assert environment_calls[0]["horizon"] == 500
    assert metadata["upstream_horizon"] == 500
    assert metadata["requested_safety_ceiling"] is None
    assert metadata["effective_horizon"] == 500


def test_environment_config_carries_exact_task_split_registry_and_horizons(
    tmp_path,
) -> None:
    captured = []

    class Daemon:
        name = "robocasa"

        def __init__(self, **kwargs):
            captured.append(kwargs)

        def start(self):
            return object()

        def stop(self):
            return None

    from robots.robocasa.runtime import _default_environment_builder

    dependency, factory, metadata = _default_environment_builder(
        args=_args(),
        task=_task(),
        horizon=300,
        output_dir=tmp_path,
        daemon_factory=Daemon,
    )

    config = captured[0]["config"]
    assert captured[0]["endpoint"] is None
    assert captured[0]["close_remote"] is True
    assert config["task"] == "CloseFridge"
    assert config["split"] == "target"
    assert config["object_registries"] == ["lightwheel"]
    assert config["upstream_horizon"] == 500
    assert config["safety_ceiling"] == 300
    assert config["task_registry_commit"] == _task().registry_commit
    assert config["task_registry_digest"] == _task().registry_digest
    assert dependency is not None
    assert factory() is not None
    assert metadata["effective_horizon"] == 300


def test_shared_environment_endpoint_attaches_without_remote_ownership(
    tmp_path, monkeypatch
) -> None:
    captured = []

    class Daemon:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def start(self):
            return object()

    from robots.robocasa.runtime import _default_environment_builder

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", "http://127.0.0.1:19001")
    args = _args()
    dependency, factory, metadata = _default_environment_builder(
        args=args,
        task=_task(),
        horizon=300,
        output_dir=tmp_path,
        daemon_factory=Daemon,
    )

    assert captured[0]["endpoint"] == "http://127.0.0.1:19001"
    assert captured[0]["close_remote"] is False
    assert dependency is not None
    assert factory() is not None
    assert metadata["endpoint"] == "http://127.0.0.1:19001"


def test_shared_environment_scene_split_reaches_task_resolution(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def resolver(selector, *, scene_split=None):
        calls.append((selector, scene_split))
        values = vars(_task()) | {"split": scene_split or "target"}
        return (SimpleNamespace(**values),)

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", "http://127.0.0.1:19001")
    monkeypatch.setenv("RPENT_SHARED_ENV_SCENE_SPLIT", "pretrain")

    bootstrap = build_episode_bootstrap(
        _args(),
        tmp_path,
        None,
        task_resolver=resolver,
        environment_builder=lambda **kwargs: (None, lambda: object(), {}),
        policy_daemon_builder=lambda **kwargs: None,
        toolkit_builder=lambda controller, observation, **kwargs: object(),
    )

    assert calls == [("CloseFridge", "pretrain")]
    assert bootstrap.runtime_metadata["split"] == "pretrain"


@pytest.mark.parametrize("split", ["", "TARGET", "training", "../pretrain"])
def test_shared_environment_scene_split_is_fail_closed(
    split, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", "http://127.0.0.1:19001")
    monkeypatch.setenv("RPENT_SHARED_ENV_SCENE_SPLIT", split)

    with pytest.raises(ValueError, match="RPENT_SHARED_ENV_SCENE_SPLIT"):
        build_episode_bootstrap(
            _args(),
            tmp_path,
            None,
            task_resolver=lambda selector: (_task(),),
            environment_builder=lambda **kwargs: (None, lambda: object(), {}),
            policy_daemon_builder=lambda **kwargs: None,
            toolkit_builder=lambda controller, observation, **kwargs: object(),
        )


def test_shared_scene_split_requires_a_shared_environment_endpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RPENT_SHARED_ENV_SCENE_SPLIT", "pretrain")

    with pytest.raises(ValueError, match="requires RPENT_SHARED_ENV_ENDPOINT"):
        build_episode_bootstrap(
            _args(),
            tmp_path,
            None,
            task_resolver=lambda selector: (_task(),),
            environment_builder=lambda **kwargs: (None, lambda: object(), {}),
            policy_daemon_builder=lambda **kwargs: None,
            toolkit_builder=lambda controller, observation, **kwargs: object(),
        )


def test_shared_environment_runtime_daemon_client_chain_closes_only_transport(
    tmp_path, monkeypatch
) -> None:
    class Rpc:
        def __init__(self):
            self.calls = []
            self.close_calls = 0

        def call(self, method, **kwargs):
            self.calls.append((method, kwargs))
            if method == "env.health":
                return {
                    "protocol": ROBOCASA_ENV_PROTOCOL,
                    "loaded": True,
                    "task": "CloseFridge",
                    "split": "target",
                    "object_registries": ["lightwheel"],
                    "task_registry_commit": ROBOCASA_COMMIT,
                    "task_registry_digest": _task().registry_digest,
                    "observation_schema": OBSERVATION_SCHEMA,
                    "action_schema": ACTION_SCHEMA,
                    "dependencies": {
                        "robocasa_commit": ROBOCASA_COMMIT,
                        "robosuite_commit": ROBOSUITE_COMMIT,
                        "mujoco_gl": "egl",
                    },
                }
            raise AssertionError(f"unexpected RPC call: {method}")

        def close(self):
            self.close_calls += 1

    rpc = Rpc()

    def daemon_factory(**kwargs):
        return RoboCasaDaemon(
            **kwargs,
            client_factory=lambda endpoint: rpc,
            wait_ready=lambda *args, **kwargs: None,
        )

    from robots.robocasa.runtime import _default_environment_builder

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", "http://127.0.0.1:19001")
    dependency, simulator_factory, _ = _default_environment_builder(
        args=_args(),
        task=_task(),
        horizon=300,
        output_dir=tmp_path,
        daemon_factory=daemon_factory,
    )

    simulator_factory()
    dependency.stop()

    assert [method for method, _ in rpc.calls] == ["env.health"]
    assert rpc.close_calls == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:19001",
        "http://localhost:19001",
        "http://[::1]:19001",
        "socket://127.0.0.1:19001",
        "socket://localhost:19001",
    ],
)
def test_shared_environment_endpoint_preserves_exact_canonical_origin(
    endpoint, tmp_path, monkeypatch
) -> None:
    captured = []

    class Daemon:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def start(self):
            return object()

    from robots.robocasa.runtime import _default_environment_builder

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", endpoint)
    _, _, metadata = _default_environment_builder(
        args=_args(),
        task=_task(),
        horizon=300,
        output_dir=tmp_path,
        daemon_factory=Daemon,
    )

    assert captured[0]["endpoint"] == endpoint
    assert metadata["endpoint"] == endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "ftp://127.0.0.1:19001",
        "http://example.com:19001",
        "http://user:secret@127.0.0.1:19001",
        "http://127.0.0.1:19001/base",
        "http://127.0.0.1:19001/",
        "http://127.0.0.1:19001?query=yes",
        "http://127.0.0.1:19001#fragment",
        "http://127.0.0.1:not-a-port",
        "HTTP://127.0.0.1:19001",
        "http://LOCALHOST:19001",
        "http://127.0.0.1:01901",
        "http://127.0.0.1:65536",
        "socket://[::1]:19001",
    ],
)
def test_shared_environment_endpoint_rejects_untrusted_values(
    endpoint, tmp_path, monkeypatch
) -> None:
    from robots.robocasa.runtime import _default_environment_builder

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="RPENT_SHARED_ENV_ENDPOINT"):
        _default_environment_builder(
            args=_args(),
            task=_task(),
            horizon=300,
            output_dir=tmp_path,
            daemon_factory=lambda **kwargs: object(),
        )


def test_shared_environment_endpoint_rejects_conflicting_cli_endpoint(
    tmp_path, monkeypatch
) -> None:
    from robots.robocasa.runtime import _default_environment_builder

    monkeypatch.setenv("RPENT_SHARED_ENV_ENDPOINT", "http://127.0.0.1:19001")
    args = _args()
    args.env_endpoint = "http://127.0.0.1:29001"

    with pytest.raises(ValueError, match="conflicts"):
        _default_environment_builder(
            args=args,
            task=_task(),
            horizon=300,
            output_dir=tmp_path,
            daemon_factory=lambda **kwargs: object(),
        )
