"""Thin RPC client for the isolated RoboCasa simulator service."""

from __future__ import annotations

import errno
import logging
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any
from urllib.error import URLError

from rpent.utils.rpc import RpcClient
from rpent.utils.socket_rpc import RpcError

_LOGGER = logging.getLogger(__name__)
_RESET_RETRY_DELAYS_S = (1.0, 2.0, 4.0)

_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 180.0,
    "env.rgbd": 60.0,
    "env.step": 60.0,
}


def _connection_was_refused(error: BaseException) -> bool:
    """Recognize a refused connection before any request reached the service."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RpcError) and current.server_traceback:
            return False
        if isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            return True
        if isinstance(current, URLError) and isinstance(current.reason, BaseException):
            current = current.reason
        else:
            current = current.__cause__
    return False


def _validate_expected_subset(
    actual: Any,
    expected: Any,
    *,
    path: str = "health",
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{path} must be a mapping")
        for key, value in expected.items():
            field = f"{path}.{key}"
            if key not in actual:
                raise ValueError(f"{field} is missing")
            _validate_expected_subset(actual[key], value, path=field)
        return
    if actual != expected:
        raise ValueError(
            f"{path} mismatch: expected {expected!r}, actual {actual!r}"
        )


class RoboCasaEnvClient:
    """Expose only RPent's versioned RoboCasa environment operations."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_health: Mapping[str, Any],
        close_remote: bool = True,
        recover_connection: Callable[[RpcClient], RpcClient] | None = None,
    ) -> None:
        self._client = client
        self._closed = False
        self._close_remote = bool(close_remote)
        self._expected_health = deepcopy(dict(expected_health))
        self._recover_connection = recover_connection
        self._check_health()

    def _check_health(self) -> None:
        health = self._client.call("env.health", timeout_s=_TIMEOUT_S["default"])
        _validate_expected_subset(health, self._expected_health)
        if not isinstance(health, Mapping):
            raise ValueError("health response must be a mapping")
        self._env_meta = deepcopy(dict(health))

    @property
    def env_meta(self) -> dict[str, Any]:
        """Return a defensive copy of the verified service identity."""
        return deepcopy(self._env_meta)

    def reset(self, task: str, seed: int) -> Any:
        """Retry initialization only when the request could not reach the server."""
        args = (task, int(seed))
        for attempt in range(len(_RESET_RETRY_DELAYS_S) + 1):
            try:
                if attempt:
                    if self._recover_connection is not None:
                        previous = self._client
                        self._client = self._recover_connection(previous)
                        if self._client is not previous:
                            previous.close()
                    self._check_health()
                result = self._client.call(
                    "env.reset",
                    args=args,
                    timeout_s=_TIMEOUT_S["env.reset"],
                )
                if attempt:
                    _LOGGER.info("RoboCasa reset recovered after %d retries", attempt)
                return result
            except Exception as exc:
                if not _connection_was_refused(exc):
                    raise
                if attempt == len(_RESET_RETRY_DELAYS_S):
                    _LOGGER.error("RoboCasa reset exhausted 3 retries: %s", exc)
                    raise
                delay = _RESET_RETRY_DELAYS_S[attempt]
                _LOGGER.warning(
                    "RoboCasa reset connection refused; retry %d/3 in %.0fs "
                    "for task=%s seed=%d: %s",
                    attempt + 1,
                    delay,
                    task,
                    args[1],
                    exc,
                )
                time.sleep(delay)
        raise AssertionError("unreachable reset retry state")

    def observe(self) -> Any:
        return self._client.call("env.observe", timeout_s=_TIMEOUT_S["default"])

    def rgbd(self, camera_name: str) -> Any:
        return self._client.call(
            "env.rgbd",
            args=(camera_name,),
            timeout_s=_TIMEOUT_S["env.rgbd"],
        )

    def step(self, action: Any) -> Any:
        return self._client.call(
            "env.step",
            args=(action,),
            timeout_s=_TIMEOUT_S["env.step"],
        )

    def close(self) -> None:
        """Close the remote environment, then release the transport, once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._close_remote:
                self._client.call(
                    "env.close",
                    timeout_s=_TIMEOUT_S["default"],
                )
        finally:
            self._client.close()


__all__ = ["RoboCasaEnvClient"]
