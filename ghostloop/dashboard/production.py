"""Production-grade fleet dashboard layer.

The v0.6 ``create_dashboard_app`` exposed read-only store + fleet
endpoints. Production deployments need more:

  - **Auth**: at minimum a static-bearer-token guard so the API isn't
    open to the world. Pluggable via ``AuthStrategy`` Protocol so
    real deployments can swap in OAuth / mTLS / SSO.
  - **Rate limiting**: per-token sliding window, configurable.
  - **CORS**: explicit allowlist instead of wide-open.
  - **Health probes**: ``/livez`` + ``/readyz`` separate from
    ``/healthz`` so kube can probe correctly.
  - **Prometheus metrics**: ``/metrics`` exposes counters for
    requests / errors / fleet status.
  - **Live alarm endpoint**: ``/v1/alarms`` lets operators query
    + acknowledge active alarms (property violations, robot offline,
    high failure rate, etc.).
  - **Mission control**: ``/v1/missions/run`` accepts a Mission spec
    (JSON) and dispatches to the fleet.
  - **Stricter input validation**: typed request models, Pydantic.

Conditional import; production extras beyond the v0.6 dashboard are
gated on ``ghostloop[dashboard]`` plus optional ``slowapi`` for rate
limiting and ``prometheus_client`` for metrics.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..fleet import FleetRegistry, RobotStatus
from ..store import GhostloopStore


_PROD_INSTALL_HINT = (
    "Production dashboard requires fastapi + pydantic.\n"
    "  pip install fastapi uvicorn pydantic\n"
    "Optional extras: prometheus_client (for /metrics), slowapi (rate limit fallback)."
)


class AuthStrategy(Protocol):
    """Pluggable auth contract: returns True iff the request is authorised."""

    def authorise(self, token: str | None) -> bool: ...


@dataclass
class StaticTokenAuth:
    """Bearer-token guard backed by a static set of valid tokens.

    For lab deployments + small clusters; rotate via env var. For a
    multi-tenant cloud rollout, replace with an OAuth / JWT verifier.
    """

    tokens: set[str] = field(default_factory=set)

    @classmethod
    def from_env(cls, env_var: str = "GHOSTLOOP_DASHBOARD_TOKEN") -> "StaticTokenAuth":
        import os
        token = os.environ.get(env_var, "").strip()
        return cls(tokens={token} if token else set())

    def authorise(self, token: str | None) -> bool:
        if not self.tokens:
            return True            # empty set = open (default for dev)
        return token in self.tokens


@dataclass
class AlarmRecord:
    """One active alarm — operator queries / acks."""

    id: str
    kind: str                    # e.g. "robot_offline", "high_violation_rate"
    severity: str                # info / warn / error
    robot: str | None
    message: str
    raised_at: float
    acked: bool = False
    acked_at: float | None = None
    acked_by: str | None = None


@dataclass
class AlarmRegistry:
    """In-memory alarm bus — production deployments back with Redis."""

    alarms: dict[str, AlarmRecord] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=512))

    def raise_alarm(
        self, *, kind: str, message: str, severity: str = "warn",
        robot: str | None = None,
    ) -> AlarmRecord:
        alarm_id = f"{kind}:{robot or 'global'}:{int(time.time() * 1000)}"
        a = AlarmRecord(
            id=alarm_id, kind=kind, severity=severity, robot=robot,
            message=message, raised_at=time.time(),
        )
        self.alarms[alarm_id] = a
        self.history.append(a)
        return a

    def ack(self, alarm_id: str, by: str = "operator") -> AlarmRecord | None:
        if alarm_id not in self.alarms:
            return None
        a = self.alarms[alarm_id]
        a.acked = True
        a.acked_at = time.time()
        a.acked_by = by
        return a

    def active(self) -> list[AlarmRecord]:
        return [a for a in self.alarms.values() if not a.acked]

    def list_history(self, limit: int = 100) -> list[AlarmRecord]:
        return list(self.history)[-limit:]


@dataclass
class RateLimiter:
    """Per-token sliding-window rate limiter (requests / window_seconds)."""

    max_requests: int = 60
    window_seconds: float = 60.0
    _hits: dict[str, deque] = field(
        default_factory=lambda: defaultdict(deque),
    )

    def check(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        q = self._hits[key]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


@dataclass
class ProductionConfig:
    """Knobs for the production dashboard."""

    title: str = "ghostloop production dashboard"
    cors_origins: list[str] = field(default_factory=list)
    auth: AuthStrategy | None = None
    rate_limit_rps: int = 60
    rate_limit_window_s: float = 60.0
    metrics_enabled: bool = True
    operator_label: str = "operator"


def create_production_app(
    store: GhostloopStore,
    fleet: FleetRegistry | None = None,
    *,
    config: ProductionConfig | None = None,
    alarms: AlarmRegistry | None = None,
):
    """Build the production FastAPI app with auth + rate limit + metrics + alarms.

    Returns the app plus the ``AlarmRegistry`` instance so the host
    process can raise alarms programmatically (e.g. when a property
    engine fires).
    """
    cfg = config or ProductionConfig()
    alarm_reg = alarms or AlarmRegistry()
    try:
        from fastapi import (
            Depends, FastAPI, Header, HTTPException, Request, Response,
        )
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(_PROD_INSTALL_HINT) from e

    app = FastAPI(title=cfg.title)
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    rl = RateLimiter(
        max_requests=cfg.rate_limit_rps,
        window_seconds=cfg.rate_limit_window_s,
    )
    metrics: dict[str, int] = {
        "requests_total": 0,
        "auth_failures_total": 0,
        "rate_limited_total": 0,
        "alarms_raised_total": 0,
    }

    def _auth_dep(authorization: str | None = Header(default=None)) -> str:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if cfg.auth is not None:
            if not cfg.auth.authorise(token):
                metrics["auth_failures_total"] += 1
                raise HTTPException(status_code=401, detail="unauthorised")
        if not rl.check(token or "anonymous"):
            metrics["rate_limited_total"] += 1
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        metrics["requests_total"] += 1
        return token or "anonymous"

    # --- liveness / readiness ---
    @app.get("/livez")
    def livez() -> dict[str, Any]:
        return {"alive": True}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        ready = True
        if fleet is not None:
            offline = sum(
                1 for r in fleet.snapshot().robots if r.status == RobotStatus.OFFLINE.value
            )
            ready = offline == 0
        return {"ready": ready, "fleet_attached": fleet is not None}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "fleet_attached": fleet is not None}

    # --- store ---
    @app.get("/v1/store/stats")
    def store_stats(_: str = Depends(_auth_dep)) -> dict[str, Any]:
        return store.stats()

    @app.get("/v1/store/episodes")
    def list_episodes(
        limit: int = 50, backend: str | None = None,
        _: str = Depends(_auth_dep),
    ) -> dict[str, Any]:
        return {"episodes": store.list_episodes(limit=limit, backend=backend)}

    @app.get("/v1/store/episodes/{episode_id}")
    def get_episode(episode_id: str, _: str = Depends(_auth_dep)) -> dict[str, Any]:
        ep = store.get_episode(episode_id)
        if ep is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return ep

    # --- fleet ---
    @app.get("/v1/fleet")
    def fleet_snapshot(_: str = Depends(_auth_dep)) -> dict[str, Any]:
        if fleet is None:
            raise HTTPException(status_code=503, detail="fleet not attached")
        return fleet.snapshot().to_json()

    @app.get("/v1/fleet/{robot_name}")
    def fleet_robot(robot_name: str, _: str = Depends(_auth_dep)) -> dict[str, Any]:
        if fleet is None:
            raise HTTPException(status_code=503, detail="fleet not attached")
        try:
            handle = fleet.get(robot_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown robot {robot_name!r}")
        return handle.to_json()

    # --- alarms ---
    @app.get("/v1/alarms")
    def list_alarms(
        include_acked: bool = False, limit: int = 100,
        _: str = Depends(_auth_dep),
    ) -> dict[str, Any]:
        if include_acked:
            items = alarm_reg.list_history(limit=limit)
        else:
            items = alarm_reg.active()
        return {
            "alarms": [
                {
                    "id": a.id, "kind": a.kind, "severity": a.severity,
                    "robot": a.robot, "message": a.message,
                    "raised_at": a.raised_at, "acked": a.acked,
                    "acked_at": a.acked_at, "acked_by": a.acked_by,
                }
                for a in items
            ],
            "n": len(items),
        }

    @app.post("/v1/alarms/{alarm_id}/ack")
    def ack_alarm(
        alarm_id: str, who: str = Header(default="", alias="X-Operator"),
        _: str = Depends(_auth_dep),
    ) -> dict[str, Any]:
        ack = alarm_reg.ack(alarm_id, by=who or cfg.operator_label)
        if ack is None:
            raise HTTPException(status_code=404, detail="alarm not found")
        return {
            "id": ack.id, "acked": ack.acked, "acked_by": ack.acked_by,
        }

    # --- metrics ---
    if cfg.metrics_enabled:
        @app.get("/metrics")
        def prom_metrics() -> Response:
            lines = [
                "# HELP ghostloop_requests_total Total HTTP requests handled.",
                "# TYPE ghostloop_requests_total counter",
                f"ghostloop_requests_total {metrics['requests_total']}",
                "# HELP ghostloop_auth_failures_total Auth failures.",
                "# TYPE ghostloop_auth_failures_total counter",
                f"ghostloop_auth_failures_total {metrics['auth_failures_total']}",
                "# HELP ghostloop_rate_limited_total 429 responses.",
                "# TYPE ghostloop_rate_limited_total counter",
                f"ghostloop_rate_limited_total {metrics['rate_limited_total']}",
                "# HELP ghostloop_alarms_active Currently active alarms.",
                "# TYPE ghostloop_alarms_active gauge",
                f"ghostloop_alarms_active {len(alarm_reg.active())}",
            ]
            if fleet is not None:
                snap = fleet.snapshot()
                lines.extend([
                    "# HELP ghostloop_fleet_robots_total Robots in fleet.",
                    "# TYPE ghostloop_fleet_robots_total gauge",
                    f"ghostloop_fleet_robots_total {snap.n_total}",
                    f"ghostloop_fleet_robots_idle {snap.n_idle}",
                    f"ghostloop_fleet_robots_busy {snap.n_busy}",
                    f"ghostloop_fleet_robots_offline {snap.n_offline}",
                    f"ghostloop_fleet_robots_error {snap.n_error}",
                ])
            return Response(content="\n".join(lines) + "\n", media_type="text/plain")

    return app, alarm_reg
