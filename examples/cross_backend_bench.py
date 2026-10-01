#!/usr/bin/env python3
"""Cross-backend demo: the same seeded episodes on two backends, paired.

Defaults to mock vs mock so it runs anywhere. With MuJoCo installed, try the
gravity-sag sensitivity run on a UR5e:

    python examples/cross_backend_bench.py mujoco:ur5e mujoco:ur5e gravity_compensation=false

Writes report.md, report.json and per-episode traces to ./xbench-out.
"""

from __future__ import annotations

import json
import sys

from ghostloop.bench.catalogue import PRESETS
from ghostloop.bench.cross import cross_backend_compare


def main(argv: list[str]) -> int:
    a = argv[0] if argv else "mock"
    b = argv[1] if len(argv) > 1 else "mock"
    b_kwargs = {}
    for item in argv[2:]:
        key, _, raw = item.partition("=")
        b_kwargs[key] = json.loads(raw) if raw in ("true", "false") or raw.replace(".", "", 1).isdigit() else raw

    report = cross_backend_compare(
        PRESETS["workspace_grid_27"], a, b,
        seeds=[0, 1, 2], tolerance=0.01, b_kwargs=b_kwargs, preset_name="workspace_grid_27",
    )
    print(report.render_md())
    out = report.write("xbench-out")
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
