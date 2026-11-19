"""Synthetic-only inputs for provider-independent control boundary tests."""

from copy import deepcopy

import pytest


@pytest.fixture
def synthetic_control_catalog(monkeypatch, request):
    """Keep algorithm thresholds independent of any measured policy's scores."""
    import robots.robocasa.capabilities as capabilities
    import rpent.harness.state as state
    from robots.robocasa.tasks import packaged_registry_snapshot

    rows = {}

    def row(name, tier, typical, family="atomic"):
        n, k = {"weak": (50, 14), "unstable": (10, 5), "reliable": (10, 8)}[tier]
        interval = {"weak": [0.1, 0.29], "unstable": [0.4, 0.6], "reliable": [0.71, 0.9]}[tier]
        rows[name] = {"family": family, "episodes": n, "successes": k, "tier": tier,
                      "wilson95": interval, "crosses_threshold": False,
                      "typical_steps": typical, "source": "synthetic CPU boundary fixture; not policy evidence"}

    # These scalar values place calls immediately before/after quota boundaries.
    for name, tier, typical in (
        ("TurnOnMicrowave", "weak", 284), ("OpenCabinet", "unstable", 200),
        ("PickPlaceSinkToCounter", "reliable", 400),
        ("PickPlaceCounterToSink", "reliable", 316),
        ("PickPlaceFridgeDrawerToShelf", "reliable", 346.5),
        ("OpenFridge", "weak", 477), ("OpenMicrowave", "weak", 426),
        ("PickPlaceCounterToMicrowave", "weak", 741.5),
        ("CloseMicrowave", "unstable", 375.5), ("TiltPan", "unstable", 366),
        ("DumpLeftovers", "weak", None),
    ):
        row(name, tier, typical)
    for name, item in packaged_registry_snapshot()["tasks"].items():
        if item["family"] == "composite":
            row(name, "weak", 900, family="composite")
    catalog = {
        "schema_version": 1, "policy": "synthetic",
        "checkpoint": {"repo_id": "local/synthetic", "path": "fixture", "revision": "fixture"},
        "thresholds": {"reliable": 0.7, "unstable": 0.3}, "rows": rows,
        "declared": {
            "PickPlaceCounterToOpenFridge": {"kind": "proxy", "proxy_row": "PickPlaceFridgeDrawerToShelf",
                "operation": "synthetic placement", "evidence": "synthetic fixture", "allowed_control_modes": ["monitor"]},
            "PlateToBowlSameCounter": {"kind": "demonstrated", "tier": "unstable",
                "operation": "synthetic transfer", "evidence": "synthetic fixture", "allowed_control_modes": ["monitor"]},
        },
        "handback": {"provisional": True, "by_fixture": {}, "by_task": {},
                     "default": {"distance_m": [0.2, 1.0], "heading_deg_max": 45}},
    }
    capabilities.validate_capabilities(catalog, policy="synthetic")
    actual = capabilities.load_capabilities

    def load(policy, **kwargs):
        return deepcopy(catalog) if policy == "synthetic" else actual(policy, **kwargs)

    monkeypatch.setattr(capabilities, "load_capabilities", load)
    monkeypatch.setattr(state, "load_capabilities", load)
    monkeypatch.setattr(request.module, "load_capabilities", load, raising=False)
