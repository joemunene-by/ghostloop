"""Conformance checks for backend authors.

Run these against your backend before sending a PR or publishing a plugin::

    from ghostloop.testing import assert_backend_conforms

    def test_my_backend():
        assert_backend_conforms("omnisim:ur5e")

Each check targets a property the bench relies on: JSON-serialisable state,
honest move_to results, fail-closed behaviour on unreachable targets, and
run-to-run determinism.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .bench.frames import workspace_for
from .core import Result, ResultStatus


@dataclass
class ConformanceReport:
    spec: str
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = [f"conformance: {self.spec}  ({'PASS' if self.ok else 'FAIL'})"]
        lines += [f"  ok    {name}" for name in self.passed]
        lines += [f"  WARN  {name}: {why}" for name, why in self.warnings]
        lines += [f"  FAIL  {name}: {why}" for name, why in self.failed]
        return "\n".join(lines)


def _move(binding: Any, xyz: tuple[float, float, float]) -> Result:
    prim = next(p for p in binding.primitives if p.name == "move_to")
    return prim.call(binding.backend, x=xyz[0], y=xyz[1], z=xyz[2])


def check_backend_conformance(
    spec: str,
    *,
    tolerance_m: float = 0.01,
    determinism_m: float = 1e-6,
    **backend_kwargs: Any,
) -> ConformanceReport:
    """Run every conformance check against backend ``spec`` and collect results."""
    from .backends.registry import create_backend

    report = ConformanceReport(spec)

    def check(name: str, ok: bool, why: str = "") -> bool:
        if ok:
            report.passed.append(name)
        else:
            report.failed.append((name, why))
        return ok

    try:
        binding = create_backend(spec, **backend_kwargs)
    except Exception as exc:  # noqa: BLE001
        check("constructs", False, f"{type(exc).__name__}: {exc}")
        return report
    check("constructs", True)

    snap = binding.backend.snapshot()
    try:
        json.dumps(snap)
        check("snapshot is JSON-serialisable", True)
    except (TypeError, ValueError) as exc:
        check("snapshot is JSON-serialisable", False, str(exc))
    pos = snap.get("position")
    if not check(
        "snapshot has a 3-float 'position'",
        isinstance(pos, (list, tuple)) and len(pos) == 3 and all(isinstance(v, (int, float)) for v in pos),
        f"got {pos!r}",
    ):
        return report

    names = {p.name for p in binding.primitives}
    move = next((p for p in binding.primitives if p.name == "move_to"), None)
    if not check("provides move_to(x, y, z)", move is not None and {"x", "y", "z"} <= set(move.arg_schema),
                 f"primitives: {sorted(names)}"):
        return report
    for p in binding.primitives:
        if not p.description:
            report.warnings.append((f"primitive {p.name}", "no description; LLM policies see an empty tool card"))

    ws = workspace_for(binding.embodiment)
    target = ws.to_robot((0.2, -0.1, 0.1))
    res = _move(binding, target)
    after = binding.backend.snapshot()["position"]
    if check("move_to reaches a central workspace target", res.ok and math.dist(after, target) <= tolerance_m,
             f"status={res.status.value} target={target} final={after} msg={res.message!r}"):
        check("result is JSON-serialisable", _serialisable(res), "Result.to_json() failed")

    far = ws.to_robot((50.0, 0.0, 0.0))
    before = binding.backend.snapshot()["position"]
    res = _move(binding, far)
    after = binding.backend.snapshot()["position"]
    if res.ok:
        check("never reports OK for a target it did not reach", math.dist(after, far) <= tolerance_m,
              f"returned OK but final position {after} is {math.dist(after, far):.3f}m from {far}")
    else:
        check("unreachable target fails without moving",
              res.status in (ResultStatus.ERROR, ResultStatus.BLOCKED) and math.dist(before, after) <= tolerance_m,
              f"status={res.status.value}, moved {math.dist(before, after):.4f}m")

    nan = _move_safely(binding, (math.nan, 0.0, 0.0))
    check("rejects non-finite targets", nan is None or not nan.ok,
          "move_to(nan, 0, 0) returned OK")

    finals = []
    for _ in range(2):
        fresh = create_backend(spec, **backend_kwargs)
        for xyz in ((0.3, 0.2, 0.1), (-0.2, 0.3, -0.1)):
            _move(fresh, ws.to_robot(xyz))
        finals.append(fresh.backend.snapshot()["position"])
    gap = math.dist(finals[0], finals[1])
    if gap > determinism_m:
        report.warnings.append(("deterministic replay", f"two identical runs ended {gap:.2e}m apart"))
    else:
        report.passed.append("deterministic replay")
    return report


def _move_safely(binding: Any, xyz: tuple[float, float, float]) -> Result | None:
    try:
        return _move(binding, xyz)
    except (ValueError, ArithmeticError):
        return None


def _serialisable(res: Result) -> bool:
    try:
        json.dumps(res.to_json())
        return True
    except (TypeError, ValueError):
        return False


def assert_backend_conforms(spec: str, **kwargs: Any) -> ConformanceReport:
    """pytest helper: raise AssertionError with the full report if any check fails."""
    report = check_backend_conformance(spec, **kwargs)
    assert report.ok, report.render()
    return report
