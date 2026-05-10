"""ghostloop CLI: ``python -m ghostloop <subcommand>``.

Subcommands:
  info               print version + available backends + registry primitives
  demo               run the bundled scripted pick-and-place demo
  bench              run a paired-comparison bench (default: geofence on/off)
  replay <path>      summarise a JSONL trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def cmd_info(_args: argparse.Namespace) -> int:
    from .backends import mujoco_available, pybullet_available
    from .core import MockBackend
    from .primitives import move_to, pick, place, scan
    print(f"ghostloop {__version__}")
    print()
    print("Backends:")
    print(f"  MockBackend     always available")
    print(f"  MuJoCoBackend   {'available' if mujoco_available() else 'not installed (pip install ghostloop[mujoco])'}")
    print(f"  PyBulletBackend {'available' if pybullet_available() else 'not installed (pip install ghostloop[pybullet])'}")
    print()
    print("Primitives (MockBackend bindings):")
    for prim in (move_to(), scan(), pick(), place()):
        args = ", ".join(f"{k}: {v}" for k, v in prim.arg_schema.items()) or "—"
        print(f"  {prim.name:<10} {prim.description}")
        print(f"             args: {args}")
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    """Inline copy of examples/pick_and_place.py so the CLI works post-install."""
    from .core import Intent, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
    from .policies import DenyListGate, GeofenceGate, RateLimitGate
    from .primitives import move_to, pick, place, scan

    backend = MockBackend()
    registry = PrimitiveRegistry([move_to(), scan(), pick(), place()])
    pipeline = PolicyPipeline(gates=[
        DenyListGate(denied=set()),
        RateLimitGate(per_minute=600),
        GeofenceGate(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.0)),
    ])
    runtime = Runtime(backend=backend, registry=registry, policy_pipeline=pipeline)
    plan = [
        Intent("scan", {"radius": 0.5}, rationale="initial scan"),
        Intent("move_to", {"x": 0.4, "y": 0.2, "z": 0.1}, rationale="approach widget"),
        Intent("pick", {"object_id": "widget-7"}, rationale="acquire widget"),
        Intent("move_to", {"x": -0.4, "y": 0.2, "z": 0.1}, rationale="approach drop zone"),
        Intent("place", {}, rationale="release"),
        Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}, rationale="overshoot test"),
    ]
    for intent in plan:
        result = runtime.step(intent)
        marker = {"ok": "OK ", "error": "ERR", "blocked": "BLK", "timeout": "TMO"}[result.status.value]
        print(f"  [{marker}] {intent.name:<10} -> {result.message}")
    print()
    print(f"episode {runtime.trace.episode_id} complete: {len(runtime.trace.events)} steps")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import EpisodeRunner, paired_compare, preset_geofence_smoke, summarize
    from .core import PolicyPipeline
    from .policies import GeofenceGate

    eps_a = preset_geofence_smoke()
    res_a = EpisodeRunner().run_all(eps_a)
    rep_a = summarize(res_a, run_name="no-gates", bench_name="geofence-impact")

    eps_b = preset_geofence_smoke()
    fence = PolicyPipeline(gates=[
        GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
    ])
    for ep in eps_b:
        ep.pipeline = fence
    res_b = EpisodeRunner().run_all(eps_b)
    rep_b = summarize(res_b, run_name="with-geofence", bench_name="geofence-impact")

    print(rep_a.render_md())
    print(rep_b.render_md())
    print(paired_compare(rep_a, rep_b).render_md())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .traces import summarize_trace
    summary = summarize_trace(args.path)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"Episode: {summary['episode_id']}")
    print(f"Backend: {summary['backend']}")
    print(f"Events:  {summary['n_events']}")
    print(f"Total duration: {summary['total_duration_ms']:.1f}ms")
    print()
    print("By status:")
    for k, v in sorted(summary["by_status"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10} {v}")
    print()
    print("By intent:")
    for k, v in sorted(summary["by_intent"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
    if summary["denied"]:
        print()
        print(f"Denied intents ({summary['denied']}):")
        for r in summary["deny_reasons"]:
            print(f"  - {r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ghostloop",
        description="ghostloop — the agent loop, embodied.",
    )
    p.add_argument("--version", action="version", version=f"ghostloop {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Print version + available backends + primitives")
    sub.add_parser("demo", help="Run the bundled pick-and-place demo")
    sub.add_parser("bench", help="Run the geofence-impact paired bench")

    rp = sub.add_parser("replay", help="Summarise a JSONL trace file")
    rp.add_argument("path", type=Path, help="Path to a trace JSONL written by Trace.write_jsonl()")
    rp.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    sp = sub.add_parser("store", help="Inspect the SQLite episode/run store")
    sp_sub = sp.add_subparsers(dest="store_cmd", required=True)
    sp_stats = sp_sub.add_parser("stats", help="Counts of episodes / runs / comparisons")
    sp_stats.add_argument("--db", type=Path, default=Path.home() / ".ghostloop" / "store.db")
    sp_eps = sp_sub.add_parser("episodes", help="List recent episodes")
    sp_eps.add_argument("--db", type=Path, default=Path.home() / ".ghostloop" / "store.db")
    sp_eps.add_argument("--limit", type=int, default=20)
    sp_runs = sp_sub.add_parser("runs", help="List recent run reports")
    sp_runs.add_argument("--db", type=Path, default=Path.home() / ".ghostloop" / "store.db")
    sp_runs.add_argument("--limit", type=int, default=20)

    mp = sub.add_parser("mcp", help="Run the MCP server (stdio transport for Claude Desktop / Cursor)")
    mp.add_argument("--name", default="ghostloop")

    args = p.parse_args(argv)
    handlers = {
        "info": cmd_info,
        "demo": cmd_demo,
        "bench": cmd_bench,
        "replay": cmd_replay,
        "store": cmd_store,
        "mcp": cmd_mcp,
    }
    return handlers[args.cmd](args)


def cmd_store(args: argparse.Namespace) -> int:
    from .store import GhostloopStore
    db = args.db
    if not db.parent.exists():
        db.parent.mkdir(parents=True, exist_ok=True)
    store = GhostloopStore(str(db))
    if args.store_cmd == "stats":
        s = store.stats()
        print(f"Store: {db}")
        for k, v in s.items():
            print(f"  {k:<14} {v}")
    elif args.store_cmd == "episodes":
        rows = store.list_episodes(limit=args.limit)
        if not rows:
            print("(no episodes ingested yet)")
        for r in rows:
            print(f"  {r.episode_id[:8]}  backend={r.backend:<8} "
                  f"steps={r.n_steps:<3} blocked={r.n_blocked} errored={r.n_errored}")
    elif args.store_cmd == "runs":
        rows = store.list_runs(limit=args.limit)
        if not rows:
            print("(no run reports ingested yet)")
        for r in rows:
            print(f"  {r.run_id[:8]}  bench={r.bench_name:<20} run={r.run_name:<14} "
                  f"{r.passed}/{r.n} = {r.rate:.1%}  CI=[{r.ci_low:.1%},{r.ci_high:.1%}]")
    store.close()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .core import MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
    from .mcp_server import run_mcp_server
    from .policies import GeofenceGate, RateLimitGate
    from .primitives import move_to, pick, place, scan
    runtime = Runtime(
        backend=MockBackend(),
        registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
        policy_pipeline=PolicyPipeline(gates=[
            RateLimitGate(per_minute=600),
            GeofenceGate(min_corner=(-1, -1, 0), max_corner=(1, 1, 1)),
        ]),
    )
    run_mcp_server(runtime, server_name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
