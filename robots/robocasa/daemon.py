"""Supervise one isolated, direct-upstream RoboCasa simulator worker."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    inspect_robocasa_dependencies,
)
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import (
    ACTION_SCHEMA,
    ROBOCASA_ENV_PROTOCOL,
    _validate_config,
)
from robots.robocasa.observation import OBSERVATION_SCHEMA
from robots.robocasa.session import RoboCasaSession
from rpent.utils.config import get_repo_root

_LOGGER = logging.getLogger(__name__)


def _client_for_endpoint(endpoint: str):
    if endpoint.startswith("http://"):
        from rpent.utils.http_rpc import HttpRpcClient

        return HttpRpcClient(endpoint)
    if endpoint.startswith("socket://"):
        from rpent.utils.socket_rpc import SocketRpcClient

        address = endpoint.removeprefix("socket://")
        host, separator, port = address.rpartition(":")
        if not separator or not host:
            raise ValueError(f"invalid socket endpoint: {endpoint!r}")
        return SocketRpcClient(host, int(port))
    raise ValueError("RoboCasa endpoint must use http:// or socket://")


class RoboCasaDaemon:
    """Own the simulator process, RPC identity, session, and bounded cleanup."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        endpoint: str | None = None,
        interpreter: str | None = None,
        timeout_s: float = 600.0,
        log_path: str | Path | None = None,
        config_path: str | Path | None = None,
        daemon_factory: Callable[..., Any] | None = None,
        client_factory: Callable[[str], Any] | None = None,
        port_picker: Callable[[], int] | None = None,
        wait_ready: Callable[..., None] | None = None,
        preflight: Callable[..., Any] = inspect_robocasa_dependencies,
        formal: bool = True,
        close_remote: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.config = _validate_config(config)
        self.endpoint = endpoint
        self.interpreter = interpreter
        self.timeout_s = float(timeout_s)
        self.log_path = Path(log_path) if log_path is not None else None
        self.config_path = Path(config_path) if config_path is not None else None
        self._daemon_factory = daemon_factory
        self._client_factory = client_factory or _client_for_endpoint
        self._port_picker = port_picker
        self._wait_ready = wait_ready
        self._preflight = preflight
        self._formal = bool(formal)
        self._close_remote = bool(close_remote)
        self._monotonic = monotonic
        self._startup_timings: dict[str, int] = {}
        self._local = endpoint is None
        self._process = None
        self._selected_interpreter: str | None = None
        self._rpc = None
        self._client: RoboCasaEnvClient | None = None
        self._session: RoboCasaSession | None = None
        self._stopped = False

    def start(self) -> RoboCasaSession:
        if self._stopped:
            raise RuntimeError("RoboCasa daemon is stopped")
        if self._session is not None:
            return self._session
        selected = self.interpreter or os.environ.get("RPENT_ROBOCASA_PYTHON")
        started_at = self._monotonic()
        try:
            if self._local:
                if not selected:
                    raise RuntimeError(
                        "RPENT_ROBOCASA_PYTHON must name the isolated simulator "
                        "interpreter"
                    )
                report = self._preflight(
                    selected,
                    requested_registries=tuple(self.config["object_registries"]),
                    formal=self._formal,
                )
                selected = report.interpreter
                self._selected_interpreter = selected
                self._launch(selected)
            assert self.endpoint is not None
            rpc = self._client_factory(self.endpoint)
            self._rpc = rpc
            self._wait_for_transport(rpc)
            client = RoboCasaEnvClient(
                rpc,
                expected_health=self._expected_health(),
                close_remote=self._close_remote,
                recover_connection=(
                    self._recover_reset_connection if self._local else None
                ),
            )
            self._client = client
            self._rpc = None
            self._session = RoboCasaSession(
                client,
                task=self.config["task"],
                seed=self.config["seed"],
            )
            self._startup_timings = {
                "simulator_process_startup": max(
                    0, round((self._monotonic() - started_at) * 1000)
                )
            }
            return self._session
        except Exception as exc:
            self._cleanup_failed_start()
            label = selected or self.endpoint or "unknown RoboCasa service"
            stderr = self._captured_stderr()
            suffix = f"; captured stderr: {stderr}" if stderr else ""
            raise RuntimeError(
                f"RoboCasa startup failed using {label}: {exc}{suffix}"
            ) from exc

    @property
    def startup_timings(self) -> Mapping[str, int]:
        return dict(self._startup_timings)

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
        elif self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        if self._rpc is not None:
            try:
                self._rpc.close()
            except Exception:
                pass
            self._rpc = None
        if self._local and self.endpoint is not None and self._process is not None:
            admin = None
            try:
                admin = self._client_factory(self.endpoint)
                admin.call("shutdown", timeout_s=5.0)
            except Exception:
                pass
            finally:
                if admin is not None:
                    try:
                        admin.close()
                    except Exception:
                        pass
            self._process.stop(timeout=15.0)

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
            self.log_path = Path(tempfile.gettempdir()) / f"rpent-robocasa-{port}.log"
        if self.config_path is None:
            self.config_path = (
                Path(tempfile.gettempdir()) / f"rpent-robocasa-{port}.json"
            )
        self._write_immutable_config()
        environment = os.environ.copy()
        root = str(get_repo_root())
        caller_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            root if not caller_path else os.pathsep.join((root, caller_path))
        )
        self._process = daemon_factory(
            "robocasa_env_server",
            [
                interpreter,
                "-m",
                "robots.robocasa.env_server",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--config",
                str(self.config_path),
            ],
            env=environment,
            log_path=str(self.log_path),
            cwd=root,
        )
        self._process.start()

    def _recover_reset_connection(self, rpc: Any) -> Any:
        """Restart an exited, owned worker before its first successful reset."""
        if self._stopped:
            raise RuntimeError("RoboCasa daemon is stopped")
        if not self._local or self._process is None:
            return rpc
        reason = self._process.failure_reason()
        if reason is None:
            return rpc
        assert self._selected_interpreter is not None
        _LOGGER.warning("RoboCasa reset recovery restarting owned simulator: %s", reason)
        self._process.stop(timeout=15.0)
        replacement = None
        try:
            self._launch(self._selected_interpreter)
            assert self.endpoint is not None
            replacement = self._client_factory(self.endpoint)
            self._wait_for_transport(replacement)
            return replacement
        except Exception as exc:
            if replacement is not None:
                try:
                    replacement.close()
                except Exception:
                    pass
            if self._process is not None:
                self._process.stop(timeout=15.0)
            stderr = self._captured_stderr()
            suffix = f"; captured stderr: {stderr}" if stderr else ""
            raise RuntimeError(f"RoboCasa reset recovery failed: {exc}{suffix}") from exc

    def _write_immutable_config(self) -> None:
        assert self.config_path is not None
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.config_path.name}.",
            dir=self.config_path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.config, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.config_path)
            self.config_path.chmod(0o444)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _expected_health(self) -> dict[str, Any]:
        return {
            "protocol": ROBOCASA_ENV_PROTOCOL,
            "loaded": True,
            "task": self.config["task"],
            "split": self.config["split"],
            "object_registries": self.config["object_registries"],
            "task_registry_commit": self.config["task_registry_commit"],
            "task_registry_digest": self.config["task_registry_digest"],
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "dependencies": {
                "robocasa_commit": ROBOCASA_COMMIT,
                "robosuite_commit": ROBOSUITE_COMMIT,
                **({"mujoco_gl": "egl"} if self._formal else {}),
            },
        }

    def _wait_for_transport(self, rpc: Any) -> None:
        if self._wait_ready is None:
            from rpent.utils.rpc import wait_for_ready

            wait_ready = wait_for_ready
        else:
            wait_ready = self._wait_ready
        wait_ready(
            rpc,
            timeout_s=self.timeout_s,
            failure_check=(
                self._process.failure_reason if self._process is not None else None
            ),
        )

    def _captured_stderr(self) -> str:
        if self.log_path is None:
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-4000:].strip()

    def _cleanup_failed_start(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
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


__all__ = ["RoboCasaDaemon"]
