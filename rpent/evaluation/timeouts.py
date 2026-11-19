"""Reproducible worker deadlines derived from official task horizons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from rpent.evaluation.cases import EpisodeCase
from rpent.evaluation.manifest import EvaluationManifest

DEFAULT_WORKER_TIMEOUT_S = 1800
ROBOCASA_STEPS_PER_TIMEOUT_TIER = 750


def normalize_worker_timeout_s(value: Real | None) -> float | None:
    """Keep None as automatic; reject invalid explicit wall-clock budgets."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("worker_timeout_s must be a finite positive number")
    return float(value)


def worker_timeout_policy(
    environment: str, override: Real | None = None
) -> dict[str, str | float | int]:
    """Persist the selected rule alongside the immutable evaluation schedule."""
    seconds = normalize_worker_timeout_s(override)
    if seconds is not None or environment != "robocasa":
        return {
            "mode": "fixed",
            "seconds": DEFAULT_WORKER_TIMEOUT_S if seconds is None else seconds,
        }
    return {
        "mode": "horizon_tiers",
        "steps_per_tier": ROBOCASA_STEPS_PER_TIMEOUT_TIER,
        "seconds_per_tier": DEFAULT_WORKER_TIMEOUT_S,
    }


def resolve_worker_timeout_s(
    manifest: EvaluationManifest,
    case: EpisodeCase,
    *,
    override: Real | None = None,
) -> float:
    """Resolve each dispatched case, using upstream rather than clipped steps."""
    seconds = normalize_worker_timeout_s(override)
    if seconds is not None:
        return seconds
    policy = _policy(manifest, case)
    if policy.get("mode") == "fixed":
        seconds = normalize_worker_timeout_s(policy.get("seconds"))
        if seconds is None:
            raise ValueError("fixed worker timeout policy requires seconds")
        return seconds
    if policy.get("mode") != "horizon_tiers":
        raise ValueError("unknown worker timeout policy")
    steps_per_tier = policy.get("steps_per_tier")
    if (
        isinstance(steps_per_tier, bool)
        or not isinstance(steps_per_tier, int)
        or steps_per_tier < 1
    ):
        raise ValueError("worker timeout steps_per_tier must be a positive integer")
    seconds_per_tier = normalize_worker_timeout_s(policy.get("seconds_per_tier"))
    if seconds_per_tier is None:
        raise ValueError("worker timeout policy requires seconds_per_tier")
    horizon = _official_horizon(manifest, case)
    return ((horizon - 1) // steps_per_tier + 1) * seconds_per_tier


def resolve_group_worker_timeout_s(
    manifest: EvaluationManifest,
    cases: Sequence[EpisodeCase],
    *,
    override: Real | None = None,
) -> float:
    """A grouped automatic worker receives the sum of its case budgets."""
    if override is None and _policy(manifest, cases[0]).get("mode") == "horizon_tiers":
        return sum(resolve_worker_timeout_s(manifest, case) for case in cases)
    return resolve_worker_timeout_s(manifest, cases[0], override=override)


def _policy(manifest: EvaluationManifest, case: EpisodeCase) -> Mapping:
    policy = manifest.schedule.get("worker_timeout")
    if policy is None:
        return worker_timeout_policy(case.environment)
    if not isinstance(policy, Mapping):
        raise ValueError("worker timeout policy must be a mapping")
    return policy


def _official_horizon(manifest: EvaluationManifest, case: EpisodeCase) -> int:
    for task in manifest.simulator.get("tasks", ()):
        if task.get("name") == case.task_name:
            horizon = task.get("upstream_horizon", case.horizon)
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("official task horizon must be a positive integer")
            return horizon
    # Older and third-party manifests may only contain the resolved case horizon.
    return case.horizon
