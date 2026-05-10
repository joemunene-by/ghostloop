"""FastAPI backend for the ghostloop fleet/episode/run dashboard.

Read-only HTTP surface over the SQLite store + an optional FleetRegistry
attached to the same process. Designed as the data layer for a Next.js
+ tRPC dashboard (matching the broader ghostloop ecosystem stack).

Conditional import — package itself doesn't depend on fastapi:

  pip install ghostloop[dashboard]

Endpoints:
  GET /healthz
  GET /v1/store/stats
  GET /v1/store/episodes?limit=N
  GET /v1/store/episodes/{episode_id}
  GET /v1/store/runs?bench_name=&limit=N
  GET /v1/store/runs/{run_id}
  GET /v1/fleet                          (only if a fleet was attached)
  GET /v1/fleet/{robot_name}             (only if a fleet was attached)

The factory ``create_dashboard_app(store, fleet=None)`` returns a ready
FastAPI ASGI app suitable for ``uvicorn`` / ``hypercorn`` / Mangum on
Lambda / etc.
"""

from .api import create_dashboard_app, dashboard_available
from .production import (
    AlarmRecord,
    AlarmRegistry,
    AuthStrategy,
    ProductionConfig,
    RateLimiter,
    StaticTokenAuth,
    create_production_app,
)
from .streaming import StreamManager, attach_streaming

__all__ = [
    "create_dashboard_app",
    "dashboard_available",
    "StreamManager",
    "attach_streaming",
    "AlarmRecord",
    "AlarmRegistry",
    "AuthStrategy",
    "ProductionConfig",
    "RateLimiter",
    "StaticTokenAuth",
    "create_production_app",
]
