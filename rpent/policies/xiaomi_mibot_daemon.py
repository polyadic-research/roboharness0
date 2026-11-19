"""Supervise the isolated, pinned Xiaomi MiBoT policy process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rpent.policies.xiaomi_mibot_catalog import XiaomiMiBoTCheckpoint
from rpent.policies.xiaomi_mibot_client import XiaomiMiBoTPolicySession
from rpent.policies.xiaomi_mibot_protocol import XiaomiMiBoTHealthResponse
from rpent.policies.xiaomi_mibot_server import SERVER_VERSION
from rpent.utils.config import get_repo_root

_PREFLIGHT_SCRIPT = """
import importlib.metadata as metadata
import json
import sys
print(json.dumps({
    "python": list(sys.version_info[:2]),
    "torch": metadata.version("torch"),
    "transformers": metadata.version("transformers"),
    "flash_attn": metadata.version("flash-attn"),
}))
""".strip()


def _preflight_interpreter(interpreter: str) -> None:
    try:
        completed = subprocess.run(
            [interpreter, "-c", _PREFLIGHT_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError(f"Xiaomi interpreter preflight failed: {detail}") from error
    expected = {
        "python": [3, 12],
        "transformers": "4.57.1",
        "flash_attn": "2.8.3",
    }
    identities = tuple(
        {**expected, "torch": torch_version, "flash_attn": flash_version}
        for torch_version in ("2.8.0", "2.8.0+cu128", "2.8.0+cu126")
        for flash_version in ("2.8.3", "2.8.3+cu12torch2.8cxx11abitrue")
    )
    if payload not in identities:
        raise RuntimeError(f"Xiaomi interpreter identity mismatch: {payload!r}")


def _client_for_endpoint(endpoint: str):
    if endpoint.startswith("http://"):
        from rpent.utils.http_rpc import HttpRpcClient

        return HttpRpcClient(endpoint)
    if endpoint.startswith("socket://"):
        from rpent.utils.socket_rpc import SocketRpcClient

        host, separator, port = endpoint.removeprefix("socket://").rpartition(":")
        if not separator or not host:
            raise ValueError(f"invalid socket endpoint: {endpoint!r}")
        return SocketRpcClient(host, int(port))
    raise ValueError("Xiaomi endpoint must use http:// or socket://")


class XiaomiMiBoTDaemon:
    """Own one Xiaomi process, transport, loaded identity and cleanup."""

    def __init__(
        self,
        *,
        checkpoint: XiaomiMiBoTCheckpoint,
        snapshot_path: str | Path,
        binding_identity: str,
        device: str,
        dtype: str,
        physical_device: str | None = None,
        endpoint: str | None = None,
        interpreter: str | None = None,
        timeout_s: float = 600.0,
        log_path: str | Path | None = None,
        daemon_factory: Callable[..., Any] | None = None,
        client_factory: Callable[[str], Any] | None = None,
        port_picker: Callable[[], int] | None = None,
        wait_ready: Callable[..., None] | None = None,
        preflight: Callable[[str], None] = _preflight_interpreter,
        attach_loaded: bool = False,
        expected_artifact_identity: str | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.checkpoint = checkpoint
        self.snapshot_path = Path(snapshot_path)
        self.binding_identity = binding_identity
        self.device = device
        self.dtype = dtype
        self.physical_device = physical_device
        self.endpoint = endpoint
        self.interpreter = interpreter
        self.timeout_s = float(timeout_s)
        self.log_path = Path(log_path) if log_path is not None else None
        self._daemon_factory = daemon_factory
        self._client_factory = client_factory or _client_for_endpoint
        self._port_picker = port_picker
        self._wait_ready = wait_ready
        self._preflight = preflight
        self.attach_loaded = bool(attach_loaded)
        self.expected_artifact_identity = expected_artifact_identity
        self._local = endpoint is None
        self._process = None
        self._rpc = None
        self._session: XiaomiMiBoTPolicySession | None = None
        self._health: XiaomiMiBoTHealthResponse | None = None
        self._stopped = False
        if self.attach_loaded and self._local:
            raise ValueError("attach_loaded requires an external Xiaomi endpoint")
        if self.attach_loaded and not self.expected_artifact_identity:
            raise ValueError("attach_loaded requires expected_artifact_identity")

    def start(self) -> XiaomiMiBoTPolicySession:
        if self._stopped:
            raise RuntimeError("Xiaomi MiBoT daemon is stopped")
        if self._session is not None:
            return self._session
        selected = self.interpreter or os.environ.get("RPENT_XIAOMI_MIBOT_PYTHON")
        try:
            if self._local:
                interpreter = selected or sys.executable
                self._preflight(interpreter)
                self._launch(interpreter)
            assert self.endpoint is not None
            rpc = self._client_factory(self.endpoint)
            self._rpc = rpc
            self._wait_for_transport(rpc)
            session = XiaomiMiBoTPolicySession(
                rpc,
                checkpoint=self.checkpoint,
                binding_identity=self.binding_identity,
                close_remote=not self.attach_loaded,
                policy_seed=(int(os.environ["RPENT_XIAOMI_POLICY_SEED"])
                             if "RPENT_XIAOMI_POLICY_SEED" in os.environ else None),
                rng_journal=os.environ.get("RPENT_XIAOMI_RNG_JOURNAL"),
            )
            loaded = (
                None
                if self.attach_loaded
                else session.load(
                    device=self.device, dtype=self.dtype, timeout_s=self.timeout_s
                )
            )
            health_payload = rpc.call(
                "health", timeout_s=min(30.0, self.timeout_s)
            )
            health = XiaomiMiBoTHealthResponse.from_payload(health_payload)
            artifact_identity = (
                str(self.expected_artifact_identity)
                if loaded is None
                else loaded.artifact_identity
            )
            self._validate_health(health, artifact_identity=artifact_identity)
            self._health = health
            self._session = session
            return session
        except Exception as error:
            self._cleanup_failed_start()
            label = selected or self.endpoint or "active interpreter"
            raise RuntimeError(f"Xiaomi MiBoT startup failed using {label}: {error}") from error

    def ensure_ready(self) -> None:
        self.start()

    def check_health(self) -> Mapping[str, Any]:
        if self._rpc is None or self._health is None:
            raise RuntimeError("Xiaomi MiBoT daemon is not running")
        payload = self._rpc.call("health", timeout_s=min(30.0, self.timeout_s))
        health = XiaomiMiBoTHealthResponse.from_payload(payload)
        self._validate_health(health, artifact_identity=self._health.artifact_identity)
        return health.to_payload()

    def clear_episode_state(self) -> None:
        if self._session is None:
            raise RuntimeError("Xiaomi MiBoT daemon is not running")
        self._session.reset("__rpent_idle__")

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        elif self._rpc is not None:
            self._rpc.close()
        if self._local and self._process is not None and self.endpoint is not None:
            admin = None
            try:
                admin = self._client_factory(self.endpoint)
                admin.call("shutdown", timeout_s=5.0)
            except Exception:
                pass
            finally:
                if admin is not None:
                    admin.close()
            self._process.stop(timeout=15.0)
        self._rpc = None

    def _launch(self, interpreter: str) -> None:
        if self._daemon_factory is None:
            from rpent.utils.daemon import ProcessDaemon

            daemon_factory = ProcessDaemon
        else:
            daemon_factory = self._daemon_factory
        if self._port_picker is None:
            from rpent.utils.daemon import pick_free_port

            port = pick_free_port()
        else:
            port = self._port_picker()
        self.endpoint = f"http://127.0.0.1:{port}"
        if self.log_path is None:
            self.log_path = Path(tempfile.gettempdir()) / f"rpent-xiaomi-{port}.log"
        root = str(get_repo_root())
        env = os.environ.copy()
        if self.physical_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.physical_device
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (root, env.get("PYTHONPATH")) if item
        )
        command = [
            interpreter,
            "-m",
            "rpent.policies.xiaomi_mibot_server",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--snapshot-path",
            str(self.snapshot_path),
            "--binding-identity",
            self.binding_identity,
            "--checkpoint-repo-id",
            self.checkpoint.repo_id,
            "--checkpoint-revision",
            self.checkpoint.revision,
            "--binding-name",
            self.checkpoint.binding_name,
            "--robot-type",
            self.checkpoint.robot_type,
            "--device",
            self.device,
            "--dtype",
            self.dtype,
        ]
        self._process = daemon_factory(
            "xiaomi_mibot_policy_server",
            command,
            env=env,
            log_path=str(self.log_path),
            cwd=root,
        )
        self._process.start()

    def _wait_for_transport(self, rpc: Any) -> None:
        if self._wait_ready is None:
            from rpent.utils.rpc import wait_for_ready

            wait_ready = wait_for_ready
        else:
            wait_ready = self._wait_ready
        wait_ready(
            rpc,
            timeout_s=self.timeout_s,
            failure_check=self._process.failure_reason if self._process else None,
        )

    def _validate_health(
        self, health: XiaomiMiBoTHealthResponse, *, artifact_identity: str
    ) -> None:
        expected = {
            "checkpoint": self.checkpoint,
            "checkpoint_identity": self.checkpoint.identity_digest,
            "binding_identity": self.binding_identity,
            "artifact_identity": artifact_identity,
            "remote_code_revision": self.checkpoint.revision,
            "device": self.device,
            "dtype": self.dtype,
            "server_version": SERVER_VERSION,
            "loaded": True,
        }
        for field, value in expected.items():
            if getattr(health, field) != value:
                raise ValueError(f"health {field} mismatch")

    def _cleanup_failed_start(self) -> None:
        if self._rpc is not None:
            try:
                self._rpc.close()
            except Exception:
                pass
        self._rpc = None
        if self._process is not None:
            try:
                self._process.stop(timeout=15.0)
            except Exception:
                pass


__all__ = ["XiaomiMiBoTDaemon"]
