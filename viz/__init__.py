"""
viz package — Real-time dual-panel dashboard and visualization.
"""

from viz.dashboard import Dashboard

# Alias for interface compatibility
DualPanelDashboard = Dashboard

__all__ = [
    "Dashboard",
    "DualPanelDashboard",
]
