from __future__ import annotations

import sys
import time

import pytest

from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import wait_for_ready


class _UnavailableClient:
    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        raise ConnectionRefusedError("not listening")

    def close(self) -> None:
        pass


class _ReadyClient:
    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        return {"ok": True}

    def close(self) -> None:
        pass


def test_wait_for_ready_fails_immediately_when_service_process_exits() -> None:
    with pytest.raises(RuntimeError, match=r"SAM3 exited with code 1.*sam3.log"):
        wait_for_ready(
            _UnavailableClient(),
            timeout_s=60,
            poll_interval_s=0,
            failure_check=lambda: "SAM3 exited with code 1; see sam3.log",
        )


def test_wait_for_ready_checks_failure_after_failed_health_call() -> None:
    checks = iter((None, "VLA exited with code 2"))

    with pytest.raises(RuntimeError, match="VLA exited with code 2"):
        wait_for_ready(
            _UnavailableClient(),
            timeout_s=60,
            poll_interval_s=0,
            failure_check=lambda: next(checks),
        )


def test_wait_for_ready_preserves_success_without_failure_check() -> None:
    wait_for_ready(_ReadyClient(), timeout_s=0.1, poll_interval_s=0)


def test_wait_for_ready_preserves_timeout_without_failure_check() -> None:
    with pytest.raises(TimeoutError, match="server did not become ready"):
        wait_for_ready(_UnavailableClient(), timeout_s=0, poll_interval_s=0)


def test_process_daemon_reports_early_exit_and_log_path(tmp_path) -> None:
    log_path = tmp_path / "service.log"
    daemon = ProcessDaemon(
        "test_service",
        [sys.executable, "-c", "raise SystemExit(7)"],
        log_path=str(log_path),
    )
    daemon.start()
    try:
        deadline = time.monotonic() + 2
        reason = None
        while reason is None and time.monotonic() < deadline:
            reason = daemon.failure_reason()
            time.sleep(0.01)
        assert reason == f"test_service exited with code 7; see {log_path}"
    finally:
        daemon.stop()
