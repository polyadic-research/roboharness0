from __future__ import annotations

import errno
from copy import deepcopy
from threading import Thread
from urllib.error import URLError

import numpy as np
import pytest

from robots.robocasa.dependencies import ROBOCASA_COMMIT, ROBOSUITE_COMMIT
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import ACTION_SCHEMA, ROBOCASA_ENV_PROTOCOL
from robots.robocasa.observation import OBSERVATION_SCHEMA
from rpent.utils.http_rpc import HttpRpcClient, HttpRpcServer
from rpent.utils.socket_rpc import RpcError


def _health() -> dict:
    return {
        "protocol": ROBOCASA_ENV_PROTOCOL,
        "loaded": True,
        "task": "CloseFridge",
        "split": "target",
        "object_registries": ["lightwheel"],
        "task_registry_commit": ROBOCASA_COMMIT,
        "task_registry_digest": "sha256:" + "1" * 64,
        "observation_schema": OBSERVATION_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "dependencies": {
            "robocasa_commit": ROBOCASA_COMMIT,
            "robosuite_commit": ROBOSUITE_COMMIT,
            "mujoco_version": "3.3.2",
            "mujoco_gl": "egl",
        },
    }


class _Rpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict, float | None]] = []
        self.close_calls = 0
        self.responses = {
            "env.health": _health(),
            "env.reset": {"reset": True},
            "env.observe": {"observe": True},
            "env.rgbd": {
                "episode_id": "episode-7",
                "step_id": 0,
                "camera_name": "robot0_agentview_left",
                "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
                "depth_m": np.ones((256, 256), dtype=np.float32),
                "intrinsic_K": np.eye(3, dtype=np.float64),
                "extrinsic_cam2world": np.eye(4, dtype=np.float64),
            },
            "env.step": {"step": True},
            "env.close": {"closed": True},
        }

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, dict(kwargs or {}), timeout_s))
        return deepcopy(self.responses[method])

    def close(self) -> None:
        self.close_calls += 1


def test_constructor_checks_health_without_reset_and_exposes_defensive_copy() -> None:
    rpc = _Rpc()
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    assert [call[0] for call in rpc.calls] == ["env.health"]
    exposed = client.env_meta
    exposed["task"] = "OpenDrawer"
    assert client.env_meta["task"] == "CloseFridge"


def test_expected_nested_health_subset_is_checked_fail_closed() -> None:
    rpc = _Rpc()
    expected = _health()
    expected["dependencies"] = {"robocasa_commit": "0" * 40}

    try:
        RoboCasaEnvClient(rpc, expected_health=expected)
    except ValueError as exc:
        assert "dependencies.robocasa_commit" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("mismatched health was accepted")


def test_calls_use_environment_rpc_names_and_explicit_task_seed() -> None:
    rpc = _Rpc()
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    assert client.reset("CloseFridge", 7) == {"reset": True}
    assert client.observe() == {"observe": True}
    rgbd = client.rgbd("robot0_agentview_left")
    assert client.step([0.0] * 12) == {"step": True}

    reset_call = next(call for call in rpc.calls if call[0] == "env.reset")
    assert reset_call[1] == ("CloseFridge", 7)
    assert next(call for call in rpc.calls if call[0] == "env.step")[1] == (
        [0.0] * 12,
    )
    rgbd_call = next(call for call in rpc.calls if call[0] == "env.rgbd")
    assert rgbd_call[1] == ("robot0_agentview_left",)
    assert rgbd["depth_m"].shape == (256, 256)


def test_close_releases_remote_environment_and_transport_once() -> None:
    rpc = _Rpc()
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    client.close()
    client.close()

    assert [call[0] for call in rpc.calls].count("env.close") == 1
    assert rpc.close_calls == 1


def test_attached_client_can_leave_remote_environment_open() -> None:
    rpc = _Rpc()
    client = RoboCasaEnvClient(
        rpc,
        expected_health=_health(),
        close_remote=False,
    )

    client.close()

    assert "env.close" not in [call[0] for call in rpc.calls]
    assert rpc.close_calls == 1


def _refused(*, http: bool = False) -> Exception:
    error = ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")
    if not http:
        return error
    wrapped = RpcError("env.reset", "HTTP request failed")
    wrapped.__cause__ = URLError(error)
    return wrapped


class _FailingRpc(_Rpc):
    def __init__(self, failures, *, method="env.reset") -> None:
        super().__init__()
        self.failures = list(failures)
        self.method = method

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        result = super().call(method, args, kwargs, timeout_s=timeout_s)
        if method == self.method and self.failures:
            raise self.failures.pop(0)
        return result


@pytest.mark.parametrize("http", [False, True], ids=["socket", "http"])
@pytest.mark.parametrize("failures", [1, 2, 3])
def test_reset_recovers_from_connection_refusal_with_same_task_seed(
    monkeypatch, caplog, http, failures
) -> None:
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    rpc = _FailingRpc([_refused(http=http) for _ in range(failures)])
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    assert client.reset("CloseFridge", 7) == {"reset": True}

    calls = [call for call in rpc.calls if call[0] == "env.reset"]
    assert len(calls) == failures + 1
    assert all(call[1] == ("CloseFridge", 7) for call in calls)
    assert all(call[3] == 180.0 for call in calls)
    assert sleeps == [1.0, 2.0, 4.0][:failures]
    assert "retry" in caplog.text


def test_reset_stops_after_three_retries_and_keeps_last_error(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    errors = [_refused(http=True) for _ in range(4)]
    rpc = _FailingRpc(errors)
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    with pytest.raises(RpcError) as raised:
        client.reset("CloseFridge", 7)

    assert raised.value is errors[-1]
    assert sleeps == [1.0, 2.0, 4.0]
    assert [call[0] for call in rpc.calls].count("env.reset") == 4


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("response lost"),
        ConnectionResetError(errno.ECONNRESET, "reply interrupted"),
        ConnectionError("socket closed mid-frame"),
        RpcError("env.reset", "invalid JSON response"),
        RpcError("env.reset", "Connection refused", traceback="server stack"),
        ValueError("wrong seed"),
    ],
)
def test_reset_does_not_repeat_ambiguous_or_server_failures(monkeypatch, error) -> None:
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    rpc = _FailingRpc([error])
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    with pytest.raises(type(error)) as raised:
        client.reset("CloseFridge", 7)

    assert raised.value is error
    assert sleeps == []
    assert [call[0] for call in rpc.calls].count("env.reset") == 1


def test_reset_replacement_revalidates_identity_and_releases_old_transport(
    monkeypatch,
) -> None:
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    old = _FailingRpc([_refused()])
    replacement = _Rpc()
    recovered = []

    def recover(client):
        recovered.append(client)
        return replacement

    client = RoboCasaEnvClient(
        old, expected_health=_health(), recover_connection=recover
    )

    assert client.reset("CloseFridge", 7) == {"reset": True}
    assert recovered == [old]
    assert old.close_calls == 1
    assert [call[0] for call in replacement.calls] == ["env.health", "env.reset"]
    client.close()
    assert replacement.close_calls == 1
    assert old.close_calls == 1


def test_reset_retry_rejects_changed_service_identity(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    rpc = _FailingRpc([_refused()])

    def recover(client):
        client.responses["env.health"]["task"] = "OpenDrawer"
        return client

    expected = _health()
    client = RoboCasaEnvClient(
        rpc, expected_health=expected, recover_connection=recover
    )
    expected["task"] = "OpenDrawer"  # Caller mutation cannot weaken identity checks.

    with pytest.raises(ValueError, match="health.task mismatch"):
        client.reset("CloseFridge", 7)

    assert [call[0] for call in rpc.calls].count("env.reset") == 1


def test_refusal_during_retry_health_check_uses_remaining_retry_budget(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    rpc = _FailingRpc([_refused()])

    def recover(client):
        if len(sleeps) == 1:
            client.method = "env.health"
            client.failures = [_refused()]
        return client

    client = RoboCasaEnvClient(
        rpc, expected_health=_health(), recover_connection=recover
    )

    assert client.reset("CloseFridge", 7) == {"reset": True}
    assert sleeps == [1.0, 2.0]
    assert [call[0] for call in rpc.calls].count("env.reset") == 2


def test_physical_step_connection_refusal_is_never_retried(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    rpc = _FailingRpc([_refused()], method="env.step")
    client = RoboCasaEnvClient(rpc, expected_health=_health())

    with pytest.raises(ConnectionRefusedError):
        client.step([0.0] * 12)

    assert sleeps == []
    assert [call[0] for call in rpc.calls].count("env.step") == 1


def test_reset_recovers_from_real_http_listener_outage(monkeypatch) -> None:
    requests = []
    servers = []
    sleeps = []

    def dispatch(method, args, _kwargs):
        requests.append((method, args))
        if method == "env.health":
            return _health()
        if method == "env.reset":
            return {"reset": True}
        raise AssertionError(method)

    def listen(port):
        server = HttpRpcServer(("127.0.0.1", port), dispatch)
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        servers.append((server, thread))
        return server

    original = listen(0)
    port = original.server_address[1]
    client = RoboCasaEnvClient(
        HttpRpcClient(f"http://127.0.0.1:{port}"), expected_health=_health(),
        close_remote=False,
    )
    original.shutdown()
    original.server_close()
    servers[0][1].join(timeout=2)

    def recover_after_delay(delay):
        sleeps.append(delay)
        listen(port)

    monkeypatch.setattr("time.sleep", recover_after_delay)
    try:
        assert client.reset("CloseFridge", 7) == {"reset": True}
        assert sleeps == [1.0]
        assert requests == [
            ("env.health", ()),
            ("env.health", ()),
            ("env.reset", ("CloseFridge", 7)),
        ]
    finally:
        client.close()
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
