"""FastAPI dashboard app factory."""

from __future__ import annotations

from typing import Any

from ..fleet import FleetRegistry
from ..store import GhostloopStore


_DASHBOARD_INSTALL_HINT = (
    "Dashboard requires fastapi + pydantic.\n"
    "  pip install fastapi uvicorn\n"
    "or  pip install ghostloop[dashboard]"
)


def dashboard_available() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


def create_dashboard_app(
    store: GhostloopStore,
    fleet: FleetRegistry | None = None,
    *,
    title: str = "ghostloop dashboard API",
):
    """Build the FastAPI app. Raises ImportError with install hint if fastapi missing.

    The ``store`` and ``fleet`` instances are captured by reference; the
    fleet can be mutated after app creation and endpoints will reflect
    the latest state.
    """
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
    except ImportError as e:
        raise ImportError(_DASHBOARD_INSTALL_HINT) from e

    app = FastAPI(title=title)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "fleet_attached": fleet is not None}

    # --- Store endpoints ---

    @app.get("/v1/store/stats")
    def store_stats() -> dict[str, Any]:
        return store.stats()

    @app.get("/v1/store/episodes")
    def store_episodes(limit: int = 50, backend: str | None = None) -> dict[str, Any]:
        rows = store.list_episodes(backend=backend, limit=limit)
        return {
            "n": len(rows),
            "episodes": [
                {
                    "episode_id": r.episode_id,
                    "backend": r.backend,
                    "started_at": r.started_at,
                    "n_steps": r.n_steps,
                    "n_blocked": r.n_blocked,
                    "n_errored": r.n_errored,
                }
                for r in rows
            ],
        }

    @app.get("/v1/store/episodes/{episode_id}")
    def store_episode(episode_id: str) -> dict[str, Any]:
        body = store.load_episode(episode_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"unknown episode: {episode_id}")
        return body

    @app.get("/v1/store/runs")
    def store_runs(limit: int = 50, bench_name: str | None = None) -> dict[str, Any]:
        rows = store.list_runs(bench_name=bench_name, limit=limit)
        return {
            "n": len(rows),
            "runs": [
                {
                    "run_id": r.run_id,
                    "run_name": r.run_name,
                    "bench_name": r.bench_name,
                    "n": r.n,
                    "passed": r.passed,
                    "rate": r.rate,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                }
                for r in rows
            ],
        }

    @app.get("/v1/store/runs/{run_id}")
    def store_run(run_id: str) -> dict[str, Any]:
        body = store.load_run(run_id)
        if body is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return body

    # --- Fleet endpoints (conditional on fleet being attached) ---

    if fleet is not None:
        @app.get("/v1/fleet")
        def fleet_snapshot() -> dict[str, Any]:
            return fleet.snapshot().to_json()

        @app.get("/v1/fleet/{robot_name}")
        def fleet_robot(robot_name: str) -> dict[str, Any]:
            handle = fleet.get(robot_name)
            if handle is None:
                raise HTTPException(status_code=404, detail=f"unknown robot: {robot_name}")
            return handle.to_json()

    return app
