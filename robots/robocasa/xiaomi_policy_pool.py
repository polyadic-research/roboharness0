"""Dashboard-owned reusable Xiaomi MiBoT services for RoboCasa."""

from __future__ import annotations

import atexit
import math
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
from rpent.policies.lerobot_smolvla_artifacts import binding_identity
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon
from rpent.policies.xiaomi_mibot_server import SERVER_VERSION
from rpent.utils.config import get_repo_root


@dataclass(frozen=True, slots=True)
class RoboCasaXiaomiPolicyKey:
    checkpoint_repo_id: str
    checkpoint_revision: str
    checkpoint_identity: str
    binding_identity: str
    server_version: str
    device: str
    dtype: str
    interpreter: str
    snapshot_path: str


@dataclass(slots=True)
class _Entry:
    service: Any
    leased: bool = True
    timer: Any | None = None


class RoboCasaXiaomiPolicyLease:
    def __init__(self, pool, key, service, *, reusable: bool, reused: bool) -> None:
        self._pool = pool
        self._key = key
        self.services = service
        self.reusable = reusable
        self.reused = reused
        self._released = False

    @property
    def endpoint(self) -> str:
        return self.services.endpoint

    @property
    def startup_timings(self) -> Mapping[str, int]:
        return getattr(self.services, "startup_timings", {})

    def ensure_ready(self, on_phase=None, cancelled=None) -> None:
        try:
            if cancelled is not None and cancelled():
                raise RuntimeError("Xiaomi policy startup cancelled")
            if on_phase is not None:
                on_phase("sam3_loading", "skipped", "not required")
                on_phase("vla_loading", "loading", None)
            self.services.ensure_ready()
            if on_phase is not None:
                on_phase("vla_loading", "ready", None)
        except BaseException:
            self._released = True
            self._pool._discard(self._key, self.services, self.reusable)
            raise

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._release(self._key, self.services, self.reusable)


class RoboCasaXiaomiPolicyPool:
    """Reuse one idle Xiaomi service per exact immutable runtime identity."""

    def __init__(
        self,
        *,
        starter=None,
        idle_timeout_s: float = 900.0,
        timer_factory=None,
        register_atexit: bool = True,
    ) -> None:
        if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be finite and positive")
        self._starter = starter or _start_service
        self._idle_timeout_s = float(idle_timeout_s)
        self._timer_factory = timer_factory or _daemon_timer
        self._entries: dict[RoboCasaXiaomiPolicyKey, _Entry] = {}
        self._transient: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._closed = False
        if register_atexit:
            atexit.register(self.close)

    def acquire(self, config: Mapping[str, Any], *, reuse: bool) -> RoboCasaXiaomiPolicyLease:
        key = _key(config)
        to_stop = []
        with self._lock:
            if self._closed:
                raise RuntimeError("Xiaomi policy pool is closed")
            for other_key, entry in tuple(self._entries.items()):
                if other_key != key and not entry.leased:
                    self._entries.pop(other_key)
                    _cancel_timer(entry)
                    to_stop.append(entry.service)
            entry = self._entries.get(key)
            if reuse and entry is not None and not entry.leased:
                try:
                    entry.service.check_health()
                except Exception:
                    self._entries.pop(key)
                    to_stop.append(entry.service)
                else:
                    _cancel_timer(entry)
                    entry.leased = True
                    lease = RoboCasaXiaomiPolicyLease(
                        self, key, entry.service, reusable=True, reused=True
                    )
                    for service in to_stop:
                        service.stop()
                    return lease
            service = self._starter(key, config)
            reusable = reuse and entry is None
            if reusable:
                self._entries[key] = _Entry(service)
            else:
                self._transient[id(service)] = service
            lease = RoboCasaXiaomiPolicyLease(
                self, key, service, reusable=reusable, reused=False
            )
        for item in to_stop:
            item.stop()
        return lease

    def _release(self, key, service, reusable: bool) -> None:
        if not reusable:
            with self._lock:
                self._transient.pop(id(service), None)
            service.stop()
            return
        try:
            service.clear_episode_state()
        except Exception:
            self._discard(key, service, True)
            return
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.service is service and not self._closed:
                entry.leased = False
                entry.timer = self._timer_factory(
                    self._idle_timeout_s,
                    lambda: self._expire_idle(key, service),
                )
                entry.timer.start()
                return
            self._entries.pop(key, None)
        service.stop()

    def _expire_idle(self, key, service) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.service is not service or entry.leased:
                return
            self._entries.pop(key)
            entry.timer = None
        service.stop()

    def _discard(self, key, service, reusable: bool) -> None:
        with self._lock:
            if reusable:
                entry = self._entries.get(key)
                if entry is not None and entry.service is service:
                    _cancel_timer(entry)
                    self._entries.pop(key)
            else:
                self._transient.pop(id(service), None)
        service.stop()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            services = [entry.service for entry in self._entries.values()]
            for entry in self._entries.values():
                _cancel_timer(entry)
            services.extend(self._transient.values())
            self._entries.clear()
            self._transient.clear()
        for service in services:
            service.stop()


def _key(config: Mapping[str, Any]) -> RoboCasaXiaomiPolicyKey:
    binding_digest = binding_identity(make_xiaomi_policy_binding())
    interpreter = str(
        config.get("policy_interpreter")
        or os.environ.get("RPENT_XIAOMI_MIBOT_PYTHON")
        or ""
    )
    return RoboCasaXiaomiPolicyKey(
        checkpoint_repo_id=XIAOMI_MIBOT_CHECKPOINT.repo_id,
        checkpoint_revision=XIAOMI_MIBOT_CHECKPOINT.revision,
        checkpoint_identity=XIAOMI_MIBOT_CHECKPOINT.identity_digest,
        binding_identity=binding_digest,
        server_version=SERVER_VERSION,
        device=str(config["policy_device"]),
        dtype=str(config["policy_dtype"]),
        interpreter=interpreter,
        snapshot_path=str(
            config.get("snapshot_path")
            or os.environ.get("RPENT_XIAOMI_MIBOT_SNAPSHOT")
            or ""
        ),
    )


def _start_service(key: RoboCasaXiaomiPolicyKey, config: Mapping[str, Any]):
    log_root = get_repo_root() / "logs/shared_policy/robocasa_xiaomi"
    return XiaomiMiBoTDaemon(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        snapshot_path=key.snapshot_path,
        binding_identity=key.binding_identity,
        device=key.device,
        dtype=key.dtype,
        interpreter=key.interpreter or None,
        log_path=log_root / "xiaomi_mibot.log",
    )


def _daemon_timer(delay: float, callback: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay, callback)
    timer.daemon = True
    return timer


def _cancel_timer(entry: _Entry) -> None:
    if entry.timer is not None:
        entry.timer.cancel()
        entry.timer = None


__all__ = [
    "RoboCasaXiaomiPolicyKey",
    "RoboCasaXiaomiPolicyLease",
    "RoboCasaXiaomiPolicyPool",
]
