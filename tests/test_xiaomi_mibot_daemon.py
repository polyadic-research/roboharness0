from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import rpent.policies.xiaomi_mibot_daemon as daemon_module
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon


@pytest.mark.parametrize("cuda", ["cu128", "cu126"])
@pytest.mark.parametrize("flash", ["2.8.3", "2.8.3+cu12torch2.8cxx11abitrue"])
def test_preflight_accepts_official_cuda_wheel_local_version(monkeypatch, cuda, flash) -> None:
    monkeypatch.setattr(
        daemon_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                '{"python":[3,12],"torch":"2.8.0+' + cuda + '",'
                '"transformers":"4.57.1","flash_attn":"' + flash + '"}'
            )
        ),
    )

    daemon_module._preflight_interpreter("python3.12")


def test_preflight_rejects_wrong_flash_attention_abi(monkeypatch) -> None:
    monkeypatch.setattr(
        daemon_module.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=(
            '{"python":[3,12],"torch":"2.8.0+cu126",'
            '"transformers":"4.57.1","flash_attn":"2.8.3+cu12torch2.8cxx11abifalse"}'
        )),
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        daemon_module._preflight_interpreter("python3.12")


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
        self.timeout = timeout

    def failure_reason(self):
        return None


class _Rpc:
    instances = []
    overrides = {}

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        del args
        payload = dict(kwargs or {})
        self.calls.append((method, payload, timeout_s))
        if method == "healthz":
            return {"status": "ok"}
        if method in {"load", "health"}:
            return {
                "protocol_version": 1,
                "checkpoint": XIAOMI_MIBOT_CHECKPOINT.to_payload(),
                "checkpoint_identity": XIAOMI_MIBOT_CHECKPOINT.identity_digest,
                "binding_identity": "b" * 64,
                "artifact_identity": "a" * 64,
                "remote_code_revision": XIAOMI_MIBOT_CHECKPOINT.revision,
                "device": "cuda:0",
                "dtype": "bfloat16",
                "server_version": "rpent-xiaomi-mibot/1",
                "loaded": True,
                **self.overrides,
            }
        if method == "reset":
            return payload
        if method in {"close", "shutdown"}:
            return {"ok": True}
        raise AssertionError(method)

    def close(self):
        self.close_calls += 1


def _daemon(tmp_path: Path, **overrides):
    values = {
        "checkpoint": XIAOMI_MIBOT_CHECKPOINT,
        "snapshot_path": tmp_path / XIAOMI_MIBOT_CHECKPOINT.revision,
        "binding_identity": "b" * 64,
        "device": "cuda:0",
        "dtype": "bfloat16",
        "log_path": tmp_path / "xiaomi.log",
        "daemon_factory": _Process,
        "client_factory": _Rpc,
        "port_picker": lambda: 17892,
        "wait_ready": lambda rpc, **kwargs: rpc.call("healthz"),
        "preflight": lambda interpreter: None,
    }
    values.update(overrides)
    return XiaomiMiBoTDaemon(**values)


def test_local_launch_uses_dedicated_interpreter_and_pinned_identity(
    monkeypatch, tmp_path
) -> None:
    _Process.instances = []
    _Rpc.instances = []
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_PYTHON", "/opt/xiaomi-py312/bin/python")
    daemon = _daemon(tmp_path)

    session = daemon.start()

    command = _Process.instances[0].cmd
    assert command[:3] == [
        "/opt/xiaomi-py312/bin/python",
        "-m",
        "rpent.policies.xiaomi_mibot_server",
    ]
    assert XIAOMI_MIBOT_CHECKPOINT.repo_id in command
    assert XIAOMI_MIBOT_CHECKPOINT.revision in command
    assert "cuda:0" in command
    assert "bfloat16" in command
    assert session.checkpoint == XIAOMI_MIBOT_CHECKPOINT


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"protocol_version": 2}, "protocol_version"),
        ({"checkpoint_identity": "0" * 64}, "checkpoint_identity"),
        ({"binding_identity": "0" * 64}, "binding_identity"),
        ({"remote_code_revision": "0" * 40}, "remote_code_revision"),
    ],
)
def test_readiness_rejects_identity_drift(tmp_path, override, message) -> None:
    _Rpc.overrides = override
    try:
        with pytest.raises(RuntimeError, match=message):
            _daemon(tmp_path, interpreter="/opt/xiaomi/bin/python").start()
    finally:
        _Rpc.overrides = {}


def test_clear_episode_and_stop_are_idempotent(tmp_path) -> None:
    _Rpc.instances = []
    daemon = _daemon(tmp_path, interpreter="/opt/xiaomi/bin/python")
    daemon.start()

    daemon.clear_episode_state()
    daemon.stop()
    daemon.stop()

    methods = [method for rpc in _Rpc.instances for method, _, _ in rpc.calls]
    assert "reset" in methods
    assert methods.count("close") == 1
    assert methods.count("shutdown") == 1
    assert _Process.instances[-1].stopped


def test_attached_loaded_service_skips_load_and_remote_close(tmp_path) -> None:
    _Rpc.instances = []
    daemon = _daemon(
        tmp_path,
        endpoint="http://policy.test:18000",
        attach_loaded=True,
        expected_artifact_identity="a" * 64,
    )

    daemon.start()
    daemon.stop()

    methods = [method for rpc in _Rpc.instances for method, _, _ in rpc.calls]
    assert "health" in methods
    assert "load" not in methods
    assert "close" not in methods
    assert "shutdown" not in methods
