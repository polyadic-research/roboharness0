"""Bounded, in-memory diagnostics for the dashboard HTTP process."""

from __future__ import annotations

import faulthandler
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anyio

_LATENCY_BUCKETS = ("le_10", "le_50", "le_200", "le_1000", "gt_1000")
_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)
_thread_dump_lock = threading.Lock()
_thread_dump_installed = False


@dataclass(frozen=True, slots=True)
class Readiness:
    ready: bool
    reasons: tuple[str, ...]


@dataclass(slots=True)
class RequestTicket:
    route_key: str
    started_ns: int


def _latencies() -> dict[str, int]:
    return dict.fromkeys(_LATENCY_BUCKETS, 0)


def _request_metric() -> dict[str, Any]:
    return {
        "in_flight": 0,
        "completed": 0,
        "rejected": 0,
        "timed_out": 0,
        "latency_ms": _latencies(),
    }


def _latency_bucket(elapsed_ms: float) -> str:
    if elapsed_ms <= 10:
        return "le_10"
    if elapsed_ms <= 50:
        return "le_50"
    if elapsed_ms <= 200:
        return "le_200"
    if elapsed_ms <= 1000:
        return "le_1000"
    return "gt_1000"


class DashboardDiagnostics:
    """Aggregate request and component state without retaining request details."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, dict[str, Any]] = {}
        self._latency_ms = _latencies()
        self._components: dict[str, dict[str, int | float | str]] = {}
        self._component_providers: dict[
            str, Callable[[], Mapping[str, int | float | str]]
        ] = {}

    def begin(self, route_key: str) -> RequestTicket:
        ticket = RequestTicket(route_key, time.perf_counter_ns())
        with self._lock:
            metric = self._requests.setdefault(route_key, _request_metric())
            metric["in_flight"] += 1
        return ticket

    def _retarget(self, ticket: RequestTicket, route_key: str) -> None:
        with self._lock:
            if ticket.route_key == route_key:
                return
            provisional = self._requests[ticket.route_key]
            provisional["in_flight"] = max(0, provisional["in_flight"] - 1)
            if not any(
                value for name, value in provisional.items() if name != "latency_ms"
            ) and not any(provisional["latency_ms"].values()):
                del self._requests[ticket.route_key]
            metric = self._requests.setdefault(route_key, _request_metric())
            metric["in_flight"] += 1
            ticket.route_key = route_key

    def finish(self, ticket: RequestTicket, *, outcome: str) -> None:
        elapsed_ms = (time.perf_counter_ns() - ticket.started_ns) / 1_000_000
        bucket = _latency_bucket(elapsed_ms)
        selected_outcome = (
            outcome if outcome in {"rejected", "timed_out"} else "completed"
        )
        with self._lock:
            metric = self._requests.setdefault(ticket.route_key, _request_metric())
            metric["in_flight"] = max(0, metric["in_flight"] - 1)
            metric[selected_outcome] += 1
            metric["latency_ms"][bucket] += 1
            self._latency_ms[bucket] += 1

    def set_component(
        self,
        name: str,
        values: Mapping[str, int | float | str],
    ) -> None:
        with self._lock:
            self._components[name] = dict(values)

    def register_component(
        self,
        name: str,
        provider: Callable[[], Mapping[str, int | float | str]],
    ) -> None:
        if not callable(provider):
            raise TypeError("component provider must be callable")
        with self._lock:
            self._component_providers[name] = provider

    def readiness(self) -> Readiness:
        return self._readiness(self._component_snapshot())

    @staticmethod
    def _readiness(
        components: Mapping[str, Mapping[str, int | float | str]],
    ) -> Readiness:
        read_gate_saturated = any(
            components.get(name, {}).get("saturated", 0)
            for name in ("read_gate", "repository_read_gate", "media_read_gate")
        )
        reasons = tuple(
            reason
            for reason, active in (
                ("read_gate_saturated", read_gate_saturated),
                (
                    "sse_saturated",
                    components.get("sse", {}).get("saturated", 0),
                ),
                (
                    "projection_stale",
                    components.get("projection", {}).get("stale", 0),
                ),
                (
                    "projection_error",
                    components.get("projection", {}).get("error", ""),
                ),
                (
                    "active_case_repair_stalled",
                    components.get("case_runtime", {}).get(
                        "stalled_active_repairs", 0
                    ),
                ),
                (
                    "shutting_down",
                    components.get("lifecycle", {}).get("shutting_down", 0),
                ),
            )
            if active
        )
        return Readiness(ready=not reasons, reasons=reasons)

    def _component_snapshot(self) -> dict[str, dict[str, int | float | str]]:
        with self._lock:
            components = {
                name: dict(values) for name, values in self._components.items()
            }
            providers = tuple(self._component_providers.items())
        for name, provider in providers:
            try:
                components[name] = dict(provider())
            except Exception:
                components[name] = {
                    "error": "snapshot unavailable",
                    "saturated": 1,
                }
        return components

    async def snapshot(self) -> dict[str, Any]:
        limiter = anyio.to_thread.current_default_thread_limiter()
        stats = limiter.statistics()
        threads = {
            "total_tokens": limiter.total_tokens,
            "borrowed_tokens": limiter.borrowed_tokens,
            "available_tokens": limiter.available_tokens,
            "borrowers_waiting": stats.tasks_waiting,
        }
        with self._lock:
            requests = {
                route: {
                    **metric,
                    "latency_ms": dict(metric["latency_ms"]),
                }
                for route, metric in self._requests.items()
            }
            latency_ms = dict(self._latency_ms)
        components = self._component_snapshot()
        state = self._readiness(components)
        return {
            "requests": requests,
            "latency_ms": latency_ms,
            "components": components,
            "readiness": {"ready": state.ready, "reasons": state.reasons},
            "anyio_threads": threads,
        }


class RequestObservationMiddleware:
    """Record bounded metrics keyed by the route template selected by Starlette."""

    def __init__(self, app: Any, *, diagnostics: DashboardDiagnostics) -> None:
        self.app = app
        self.diagnostics = diagnostics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") == "/healthz":
            await self.app(scope, receive, send)
            return
        ticket = self.diagnostics.begin(_route_key(scope))
        observed_scope = _RouteObservedScope(
            scope,
            lambda: self.diagnostics._retarget(ticket, _route_key(observed_scope)),
        )
        status_code = 500

        async def observe(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(observed_scope, receive, observe)
        finally:
            self.diagnostics._retarget(ticket, _route_key(observed_scope))
            self.diagnostics.finish(ticket, outcome=_outcome(status_code))


class _RouteObservedScope(dict[str, Any]):
    def __init__(self, scope: Mapping[str, Any], on_route: Callable[[], None]) -> None:
        self._on_route = on_route
        super().__init__(scope)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key == "route":
            self._on_route()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        if "route" in self:
            self._on_route()


def _route_key(scope: Mapping[str, Any]) -> str:
    raw_method = str(scope.get("method", "")).upper()
    method = raw_method if raw_method in _HTTP_METHODS else "OTHER"
    path = getattr(scope.get("route"), "path", None)
    return f"{method} {path}" if isinstance(path, str) else f"{method} unmatched"


def _outcome(status_code: int) -> str:
    if status_code in {408, 504}:
        return "timed_out"
    if status_code in {429, 503}:
        return "rejected"
    return "completed"


def install_thread_dump_signal() -> bool:
    """Install an idempotent POSIX all-thread dump handler when supported."""
    global _thread_dump_installed
    with _thread_dump_lock:
        if _thread_dump_installed:
            return True
        dump_signal = getattr(signal, "SIGUSR2", None)
        if dump_signal is None:
            return False
        faulthandler.register(
            dump_signal,
            file=sys.stderr,
            all_threads=True,
            chain=False,
        )
        _thread_dump_installed = True
        return True


__all__ = [
    "DashboardDiagnostics",
    "Readiness",
    "RequestObservationMiddleware",
    "RequestTicket",
    "install_thread_dump_signal",
]
