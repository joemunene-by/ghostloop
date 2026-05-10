#!/usr/bin/env python3
"""Bench harness demo: paired comparison of two policies on the same episodes.

Builds an 8-episode bench where a scripted policy tries to reach 8 different
targets, half inside the workspace, half outside. Runs the bench twice — once
with NO geofence gate, once WITH one — and uses paired comparison (Wilson
95% CIs + McNemar exact p) to quantify the safety pipeline's impact.

Expected: the no-gate run hits everything (8/8); the geofenced run blocks
the 4 out-of-bounds targets (4/8). McNemar should show p ≈ 0.06 (n=4
discordant), Cohen's h should land in the 'large' band.
"""

from __future__ import annotations

from ghostloop import Intent, MockBackend, PolicyPipeline
from ghostloop.bench import (
    Episode,
    EpisodeRunner,
    paired_compare,
    summarize,
)
from ghostloop.policies import GeofenceGate
from ghostloop.primitives import move_to, pick, place, scan


TARGETS = [
    ("inside-1", (0.3, 0.0, 0.0)),
    ("inside-2", (-0.5, 0.2, 0.1)),
    ("inside-3", (0.0, 0.7, 0.5)),
    ("inside-4", (0.4, -0.4, 0.2)),
    ("outside-1", (5.0, 0.0, 0.0)),
    ("outside-2", (-3.0, 0.0, 0.0)),
    ("outside-3", (0.0, 9.0, 0.0)),
    ("outside-4", (0.0, 0.0, 12.0)),
]


def _make_episode(name: str, target: tuple[float, float, float]) -> Episode:
    def setup():
        return MockBackend()

    def policy(_runtime):
        return [Intent("move_to", {"x": target[0], "y": target[1], "z": target[2]})]

    def success(_trace, state):
        return tuple(state["position"]) == target

    return Episode(
        name=name,
        goal=f"reach {target}",
        setup=setup,
        policy=policy,
        success_predicate=success,
        primitives=lambda: [move_to(), scan(), pick(), place()],
    )


def main() -> None:
    bench = "geofence-impact"

    # Run A: no safety gates.
    eps_a = [_make_episode(name, target) for name, target in TARGETS]
    res_a = EpisodeRunner().run_all(eps_a)
    rep_a = summarize(res_a, run_name="no-gates", bench_name=bench)

    # Run B: same episodes, geofence enabled.
    eps_b = [_make_episode(name, target) for name, target in TARGETS]
    fence = PolicyPipeline(gates=[GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))])
    for ep in eps_b:
        ep.pipeline = fence
    res_b = EpisodeRunner().run_all(eps_b)
    rep_b = summarize(res_b, run_name="with-geofence", bench_name=bench)

    print(rep_a.render_md())
    print(rep_b.render_md())

    comp = paired_compare(rep_a, rep_b)
    print(comp.render_md())


if __name__ == "__main__":
    main()
