"""Optional dashboard layer for live-monitoring a RPent run.

Opt in via ``python rpent/cli/main.py --dashboard``; never imported on the normal
CLI path.
"""
from rpent.dashboard.attachment import (
    DashboardAlreadyRunning,
    DashboardDescriptor,
    DashboardLease,
)
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import State

__all__ = [
    "DashboardAlreadyRunning",
    "DashboardDescriptor",
    "DashboardLease",
    "DashboardServer",
    "State",
]
