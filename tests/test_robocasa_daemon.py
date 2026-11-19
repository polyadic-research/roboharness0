from __future__ import annotations

import errno
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.daemon import RoboCasaDaemon
from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.env_server import ACTION_SCHEMA, ROBOCASA_ENV_PROTOCOL
from robots.robocasa.observation import CAMERA_SHAPES, OBSERVATION_SCHEMA


def _config() -> dict:
    return {
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
        "upstream_horizon": 500,
        "safety_ceiling": 300,
        "task_registry_commit": ROBOCASA_COMMIT,
        "task_registry_digest": "sha256:" + "1" * 64,
    }


def _health() -> dict:
    config = _config()
    return {
        "protocol": ROBOCASA_ENV_PROTOCOL,
        "loaded": True,
        "task": config["task"],
        "split": config["split"],
        "object_registries": config["object_registries"],
        "task_registry_commit": config["task_registry_commit"],
        "task_registry_digest": config["task_registry_digest"],
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "dependencies": {
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "mujoco_version": "3.3.2",
            "mujoco_gl": "egl",
        },
    }


def test_default_startup_timeout_allows_cold_simulator_load() -> None:
    daemon = RoboCasaDaemon(config=_config())

    assert daemon.timeout_s == 600.0


class _Process:
    instances = []

    def __init__(self, name, cmd, **kwargs):
        self.name = name
        self.cmd = cmd
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=15.0):
        self.stopped = True
        self.stop_timeout = timeout

    def failure_reason(self):
        return getattr(self, "exit_reason", None)


class _Rpc:
    instances = []
    health_overrides = {}

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, dict(kwargs or {}), timeout_s))
        if method == "healthz":
            return {"status": "ok"}
        if method == "env.health":
            return {**deepcopy(_health()), **self.health_overrides}
        if method in {"env.close", "shutdown"}:
            return {"ok": True}
        raise AssertionError(method)

    def close(self):
        self.close_calls += 1


def _report(interpreter: str):
    return SimpleNamespace(
        interpreter=interpreter,
        robocasa_commit=ROBOCASA_COMMIT,
        robosuite_commit=ROBOSUITE_COMMIT,
        mujoco_version="3.3.2",
        mujoco_gl="egl",
    )


def _make_daemon(tmp_path: Path, **overrides) -> RoboCasaDaemon:
    values = {
        "config": _config(),
        "timeout_s": 42,
        "log_path": tmp_path / "robocasa.log",
        "config_path": tmp_path / "robocasa.json",
        "daemon_factory": _Process,
        "client_factory": _Rpc,
        "port_picker": lambda: 17892,
        "wait_ready": lambda client, **kwargs: client.call("healthz"),
        "preflight": lambda interpreter, **kwargs: _report(interpreter),
    }
    values.update(overrides)
    return RoboCasaDaemon(**values)


def test_local_launch_uses_selected_interpreter_module_and_immutable_config(
    monkeypatch, tmp_path
) -> None:
    _Process.instances = []
    _Rpc.instances = []
    monkeypatch.setenv("RPENT_ROBOCASA_PYTHON", "/opt/robocasa/bin/python")
    checked = []
    daemon = _make_daemon(
        tmp_path,
        preflight=lambda interpreter, **kwargs: (
            checked.append((interpreter, kwargs)),
            _report(interpreter),
        )[1],
    )

    session = daemon.start()

    process = _Process.instances[0]
    assert process.started is True
    assert process.cmd[:3] == [
        "/opt/robocasa/bin/python",
        "-m",
        "robots.robocasa.env_server",
    ]
    assert process.cmd[-2:] == ["--config", str(tmp_path / "robocasa.json")]
    assert json.loads((tmp_path / "robocasa.json").read_text()) == _config()
    assert (tmp_path / "robocasa.json").stat().st_mode & 0o222 == 0
    assert checked == [
        (
            "/opt/robocasa/bin/python",
            {"requested_registries": ("lightwheel",), "formal": True},
        )
    ]
    assert session.task == "CloseFridge"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol": "wrong"}, "protocol"),
        ({"task_registry_commit": "0" * 40}, "task_registry_commit"),
        ({"task": "OpenDrawer"}, "task"),
        ({"split": "pretrain"}, "split"),
        ({"object_registries": ["objaverse"]}, "object_registries"),
        ({"observation_schema": "wrong"}, "observation_schema"),
        ({"action_schema": "wrong"}, "action_schema"),
    ],
)
def test_readiness_requires_full_environment_identity(tmp_path, override, message) -> None:
    _Process.instances = []
    _Rpc.instances = []
    _Rpc.health_overrides = override
    daemon = _make_daemon(tmp_path, interpreter="/opt/robocasa/bin/python")
    try:
        with pytest.raises(RuntimeError, match=message):
            daemon.start()
        assert _Rpc.instances[0].close_calls == 1
        assert _Process.instances[0].stopped is True
    finally:
        _Rpc.health_overrides = {}


def test_external_endpoint_skips_local_preflight_process_and_shutdown(tmp_path) -> None:
    _Process.instances = []
    _Rpc.instances = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("local dependency must not be used")

    daemon = _make_daemon(
        tmp_path,
        endpoint="http://simulator.example:9001",
        preflight=forbidden,
        port_picker=forbidden,
    )

    daemon.start()
    daemon.stop()

    assert _Process.instances == []
    assert _Rpc.instances[0].endpoint == "http://simulator.example:9001"
    methods = [call[0] for rpc in _Rpc.instances for call in rpc.calls]
    assert "env.close" in methods
    assert "shutdown" not in methods


def test_stop_is_idempotent_and_bounds_process_cleanup(tmp_path) -> None:
    _Process.instances = []
    _Rpc.instances = []
    daemon = _make_daemon(tmp_path, interpreter="/opt/robocasa/bin/python")
    daemon.start()

    daemon.stop()
    daemon.stop()

    process = _Process.instances[0]
    methods = [call[0] for rpc in _Rpc.instances for call in rpc.calls]
    assert methods.count("env.close") == 1
    assert methods.count("shutdown") == 1
    assert process.stopped is True
    assert process.stop_timeout == 15.0


def test_startup_failure_reports_interpreter_and_captured_stderr(tmp_path) -> None:
    log_path = tmp_path / "robocasa.log"
    log_path.write_text("EGL initialization failed\n", encoding="utf-8")

    def timeout(_client, **_kwargs):
        raise TimeoutError("not ready")

    daemon = _make_daemon(
        tmp_path,
        interpreter="/opt/robocasa/bin/python",
        wait_ready=timeout,
    )

    with pytest.raises(
        RuntimeError,
        match=r"/opt/robocasa/bin/python.*EGL initialization failed",
    ):
        daemon.start()


def test_daemon_import_does_not_load_lerobot() -> None:
    before = {name for name in sys.modules if name.startswith("lerobot")}
    __import__("robots.robocasa.daemon")
    assert {name for name in sys.modules if name.startswith("lerobot")} == before


def test_startup_timing_covers_preflight_process_transport_and_health(tmp_path) -> None:
    times = iter((2.0, 7.5))
    daemon = _make_daemon(
        tmp_path,
        interpreter="/opt/robocasa/bin/python",
        monotonic=lambda: next(times),
    )

    daemon.start()

    assert daemon.startup_timings == {"simulator_process_startup": 5500}


class _ResetRpc(_Rpc):
    failures = 0
    kill_process = False

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        if method != "env.reset":
            return super().call(method, args, kwargs, timeout_s=timeout_s)
        self.calls.append((method, args, dict(kwargs or {}), timeout_s))
        if type(self).failures:
            type(self).failures -= 1
            if self.kill_process:
                _Process.instances[-1].exit_reason = "simulator exited with code -9"
            raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
        return {
            "episode_id": "recovered-episode",
            "step_id": 0,
            "instruction": "Close the fridge door.",
            "effective_horizon": 500,
            "observation": {
                "state": np.zeros(16, dtype=np.float32),
                **{
                    camera: np.zeros(shape, dtype=np.uint8)
                    for camera, shape in CAMERA_SHAPES.items()
                },
            },
        }


@pytest.mark.parametrize("mode", ["owned-dead", "owned-alive", "external"])
def test_reset_recovers_before_episode_start_without_restarting_live_or_external_worker(
    monkeypatch, tmp_path, mode
) -> None:
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    _Process.instances = []
    _Rpc.instances = []
    monkeypatch.setattr(_ResetRpc, "failures", 1)
    monkeypatch.setattr(_ResetRpc, "kill_process", mode == "owned-dead")
    ports = iter((17892, 17893))
    daemon = _make_daemon(
        tmp_path,
        interpreter="/opt/robocasa/bin/python",
        endpoint="http://simulator.example:9001" if mode == "external" else None,
        port_picker=lambda: next(ports),
        client_factory=_ResetRpc,
    )
    session = daemon.start()
    try:
        observation = session.reset("CloseFridge", 7)
        assert observation["state"].shape == (16,)
        assert session.episode_id == "recovered-episode"
        assert session.effective_horizon == 500
        with pytest.raises(RuntimeError, match="already been reset"):
            session.reset("CloseFridge", 7)
        expected_processes = {"owned-dead": 2, "owned-alive": 1, "external": 0}
        assert len(_Process.instances) == expected_processes[mode]
        reset_calls = [
            call for rpc in _Rpc.instances for call in rpc.calls
            if call[0] == "env.reset"
        ]
        assert len(reset_calls) == 2
        assert all(call[1] == ("CloseFridge", 7) for call in reset_calls)
        if mode == "owned-dead":
            assert _Process.instances[0].stopped is True
            assert _Rpc.instances[0].close_calls == 1
            assert _Rpc.instances[1].endpoint == "http://127.0.0.1:17893"
            assert _Process.instances[1].cmd[0] == "/opt/robocasa/bin/python"
            assert json.loads((tmp_path / "robocasa.json").read_text()) == _config()
    finally:
        daemon.stop()
    assert all(process.stopped for process in _Process.instances)
    assert all(rpc.close_calls == 1 for rpc in _Rpc.instances)


def test_repeated_worker_exit_exhausts_reset_retries_and_cleans_up(
    monkeypatch, tmp_path, caplog
) -> None:
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    _Process.instances = []
    _Rpc.instances = []
    monkeypatch.setattr(_ResetRpc, "failures", 4)
    monkeypatch.setattr(_ResetRpc, "kill_process", True)
    ports = iter(range(17892, 17896))
    daemon = _make_daemon(
        tmp_path,
        interpreter="/opt/robocasa/bin/python",
        port_picker=lambda: next(ports),
        client_factory=_ResetRpc,
    )
    session = daemon.start()
    try:
        with pytest.raises(ConnectionRefusedError):
            session.reset("CloseFridge", 7)
        assert sleeps == [1.0, 2.0, 4.0]
        assert len(_Process.instances) == 4
        assert "exited with code -9" in caplog.text
        with pytest.raises(RuntimeError, match="before reset"):
            _ = session.episode_id
    finally:
        daemon.stop()
    assert all(process.stopped for process in _Process.instances)
    assert all(rpc.close_calls == 1 for rpc in _Rpc.instances)


def test_restart_readiness_failure_releases_new_transport_and_process(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    _Process.instances = []
    _Rpc.instances = []
    monkeypatch.setattr(_ResetRpc, "failures", 1)
    monkeypatch.setattr(_ResetRpc, "kill_process", True)
    checked = []

    def wait_ready(client, **_kwargs):
        checked.append(client)
        if len(checked) == 2:
            raise TimeoutError("replacement not ready")

    daemon = _make_daemon(
        tmp_path,
        interpreter="/opt/robocasa/bin/python",
        client_factory=_ResetRpc,
        wait_ready=wait_ready,
    )
    session = daemon.start()
    try:
        with pytest.raises(RuntimeError, match="replacement not ready"):
            session.reset("CloseFridge", 7)
        assert _Process.instances[1].stopped is True
        assert _Rpc.instances[1].close_calls == 1
    finally:
        daemon.stop()
    assert _Rpc.instances[1].close_calls == 1
