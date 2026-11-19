from __future__ import annotations

from dataclasses import dataclass

import pytest

from robots.robocasa.xiaomi_policy_pool import RoboCasaXiaomiPolicyPool


@dataclass
class _Service:
    endpoint: str
    ready: bool = False
    stopped: bool = False
    clear_calls: int = 0

    startup_timings = {"model_loading": 12}

    def ensure_ready(self, on_phase=None, cancelled=None):
        if on_phase is not None:
            assert cancelled is not None and not cancelled()
        self.ready = True

    def check_health(self):
        if not self.ready:
            raise RuntimeError("not ready")

    def clear_episode_state(self):
        self.clear_calls += 1

    def stop(self):
        self.stopped = True


def _config(**overrides):
    return {
        "policy_device": "cuda:0",
        "policy_dtype": "bfloat16",
        "policy_interpreter": "/opt/xiaomi-py312/bin/python",
        "snapshot_path": "/models/0d1aa76d",
        **overrides,
    }


class _Timer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def _pool(*, idle_timeout_s=900.0):
    started = []
    timers = []

    def starter(key, config):
        service = _Service(f"http://127.0.0.1:{13000 + len(started)}")
        started.append(service)
        return service

    def timer_factory(delay, callback):
        timer = _Timer(delay, callback)
        timers.append(timer)
        return timer

    pool = RoboCasaXiaomiPolicyPool(
        starter=starter,
        idle_timeout_s=idle_timeout_s,
        timer_factory=timer_factory,
        register_atexit=False,
    )
    return pool, started, timers


def test_launch_config_resolves_runtime_identity_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "RPENT_XIAOMI_MIBOT_PYTHON", "/opt/xiaomi-py312/bin/python"
    )
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/models/0d1aa76d")
    observed = []
    pool = RoboCasaXiaomiPolicyPool(
        starter=lambda key, _config: observed.append(key) or _Service("http://p"),
        register_atexit=False,
    )

    lease = pool.acquire(
        {"policy_device": "cuda:0", "policy_dtype": "bfloat16"},
        reuse=True,
    )

    assert observed[0].interpreter == "/opt/xiaomi-py312/bin/python"
    assert observed[0].snapshot_path == "/models/0d1aa76d"
    lease.release()
    pool.close()


def test_lease_forwards_dashboard_readiness_contract() -> None:
    pool, _started, _timers = _pool()
    lease = pool.acquire(_config(), reuse=True)
    phases = []

    lease.ensure_ready(
        on_phase=lambda *args: phases.append(args),
        cancelled=lambda: False,
    )

    assert lease.startup_timings == {"model_loading": 12}
    lease.release()
    pool.close()


def test_identical_sequential_keys_reuse_loaded_service_and_clear_episode() -> None:
    pool, started, _timers = _pool()
    first = pool.acquire(_config(), reuse=True)
    first.ensure_ready()
    first.release()

    second = pool.acquire(_config(), reuse=True)

    assert second.endpoint == first.endpoint
    assert second.reused is True
    assert len(started) == 1
    assert started[0].clear_calls == 1


@pytest.mark.parametrize(
    "change",
    [
        {"policy_device": "cuda:1"},
        {"policy_dtype": "float32"},
        {"policy_interpreter": "/other/python"},
        {"snapshot_path": "/models/other"},
    ],
)
def test_different_runtime_identity_never_reuses_service(change) -> None:
    pool, started, _timers = _pool()
    first = pool.acquire(_config(), reuse=True)
    first.ensure_ready()
    first.release()

    second = pool.acquire(_config(**change), reuse=True)

    assert second.endpoint != first.endpoint
    assert started[0].stopped


def test_busy_service_uses_isolated_one_shot() -> None:
    pool, started, _timers = _pool()
    first = pool.acquire(_config(), reuse=True)
    second = pool.acquire(_config(), reuse=True)

    assert second.endpoint != first.endpoint
    assert second.reusable is False
    second.release()
    assert started[1].stopped


def test_close_stops_idle_and_busy_services_and_is_idempotent() -> None:
    pool, started, _timers = _pool()
    busy = pool.acquire(_config(), reuse=True)
    transient = pool.acquire(_config(), reuse=True)

    pool.close()
    pool.close()
    busy.release()
    transient.release()

    assert all(service.stopped for service in started)
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire(_config(), reuse=True)


def test_idle_timeout_stops_released_service_and_callback_is_idempotent() -> None:
    pool, started, timers = _pool(idle_timeout_s=30.0)
    lease = pool.acquire(_config(), reuse=True)
    lease.ensure_ready()
    lease.release()

    assert timers[-1].delay == 30.0
    assert timers[-1].started
    timers[-1].callback()
    timers[-1].callback()

    assert started[0].stopped
