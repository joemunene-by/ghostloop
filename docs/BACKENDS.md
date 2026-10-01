# Writing a ghostloop backend

A backend is anything that can report the robot's state and execute primitives:
a physics engine, a simulator reached over HTTP or MCP, or real hardware. This
page covers what a backend has to provide, how to ship it as a plugin, and how
to check it before you open a PR or run a cross-backend comparison.

## The contract

A factory takes a `variant` (the part after the colon in `"yoursim:ur5e"`) plus
keyword arguments, and returns a `BackendBinding`:

```python
from ghostloop.backends import BackendBinding
from ghostloop.core import Primitive, Result, ResultStatus


class YourSimBackend:
    def __init__(self, robot: str, url: str = "http://localhost:8080"):
        self.name = f"yoursim:{robot}"
        self.client = ...  # your HTTP / MCP / SDK client

    def snapshot(self) -> dict:
        # Must be JSON-serialisable. "position" is the end-effector point in
        # the robot base frame, metres. Add whatever else helps debugging.
        state = self.client.joint_states()
        return {"backend": self.name, "position": state.ee_xyz, "qpos": state.q}

    def describe(self) -> dict:
        # Optional. Copied into bench provenance so reports say exactly what ran.
        return {"engine": "yoursim", "engine_version": self.client.version()}


def _move_to(backend: YourSimBackend, x: float, y: float, z: float) -> Result:
    ok, achieved, why = backend.client.move_ee((x, y, z))
    if not ok:
        return Result(ResultStatus.ERROR, {"target": [x, y, z]}, f"unreachable: {why}")
    return Result(ResultStatus.OK, {"target": [x, y, z], "achieved": achieved}, "reached")


def make_backend(variant: str | None = None, **kwargs) -> BackendBinding:
    """YourSim; variant is the robot name."""
    backend = YourSimBackend(robot=variant or "ur5e", **kwargs)
    move_to = Primitive(
        name="move_to",
        call=_move_to,
        description="Move the end-effector to (x, y, z) in metres, robot base frame.",
        arg_schema={"x": "float", "y": "float", "z": "float"},
    )
    return BackendBinding(backend=backend, primitives=[move_to], embodiment=variant)
```

Rules the bench relies on:

1. **`move_to` must be honest.** Return `OK` only when the end-effector actually
   got there. If the target is out of reach, return `ERROR` (or `BLOCKED`) and
   leave the robot where it was.
2. **Reject non-finite input.** The pipeline already denies NaN/inf, but a
   backend used directly should not move on `move_to(nan, 0, 0)`.
3. **Use the primitive names policies already know** (`move_to`, `scan`,
   `pick`, `place`) so the same policy runs on every backend.
4. **Set `embodiment`** (`"ur5e"`, `"panda"`, ...) when you model a known
   robot. The bench uses it to pick the shared workspace, which is what lets
   your results pair up with another backend's for the same arm.
5. **Be deterministic for a given seed** where the engine allows it. If it
   can't be, say so in `describe()`.

## Shipping it as a plugin

Put the factory in your own package and register it under the
`ghostloop.backends` entry-point group:

```toml
# pyproject.toml of ghostloop-yoursim
[project.entry-points."ghostloop.backends"]
yoursim = "ghostloop_yoursim:make_backend"
```

After `pip install ghostloop-yoursim`:

```console
$ ghostloop backends
  mock       available    [builtin] In-memory point robot; deterministic, unlimited reach.
  mujoco     available    [builtin] MuJoCo physics; variant is a Menagerie robot ...
  yoursim    available    [plugin:ghostloop-yoursim] YourSim; variant is the robot name.
```

A plugin that fails to import shows up as `failed to load` with the error; it
never breaks ghostloop for anyone else. Set `GHOSTLOOP_DISABLE_PLUGINS=1` to
skip plugin discovery entirely.

For a quick experiment without packaging, register in-process:

```python
from ghostloop.backends import register_backend
register_backend("yoursim", make_backend)
```

## Checking it

```python
from ghostloop.testing import assert_backend_conforms

def test_yoursim_conforms():
    assert_backend_conforms("yoursim:ur5e")
```

`check_backend_conformance(spec)` returns a report instead of raising. It
checks construction, a JSON-serialisable snapshot with a 3-float `position`,
`move_to(x, y, z)`, reaching a central workspace target, honest handling of a
far-away target, rejection of NaN, and that two identical runs end in the same
place. Tests that need a running simulator should skip cleanly when it isn't
reachable, the same way ghostloop's MuJoCo tests skip without MuJoCo.

## Comparing against another backend

```bash
ghostloop xbench --preset reach_8 --a mujoco:ur5e --b yoursim:ur5e \
  --seeds 0 1 2 --tolerance 0.01 --out-dir xbench-out
```

Both sides get identical seeded targets, written in a canonical frame and mapped
onto the embodiment's workspace (`ghostloop.bench.frames.WORKSPACES`). The output
folder holds `report.md`, `report.json` (stats, per-episode pairs, provenance for
both backends) and one JSONL trace per episode per side under `traces/a` and
`traces/b`.

`--tolerance` is in canonical units; for the UR5e workspace (scale 0.35) the
default 0.01 is 3.5 mm. `--a-arg` / `--b-arg key=value` pass constructor
arguments, and `--workspace` overrides the shared frame.

Two physics engines disagreeing tells you how sensitive a policy is to the
simulator. It does not tell you how the policy will do on hardware.
