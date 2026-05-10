# Changelog

## [0.1.0] — 2026-05-10 — Initial release: core runtime, three policy gates, mock backend

First public version. The thesis: the missing layer between ROS-style middleware
and VLA research codebases is a **tool-using runtime with a fail-closed safety
pipeline and structured tracing**. ghostloop is that layer, sim-first so it can
be developed and validated without hardware.

### Core abstractions (`ghostloop.core`)

  - `Intent`: structured high-level command. `name`, `args`, `rationale`.
  - `Primitive`: backend-bound callable with name, description, arg schema.
  - `PrimitiveRegistry`: name -> Primitive lookup. Enforces uniqueness.
  - `Result` + `ResultStatus`: ok / error / blocked / timeout. Carries
    observation dict, message, and execution duration.
  - `Decision` + `DecisionAction`: a single gate's verdict (allow / deny).
  - `PolicyGate` (Protocol): one safety check.
  - `PolicyPipeline`: ordered list of gates, fail-closed on first deny,
    transparent reasoning surfaced to the trace.
  - `Backend` (Protocol): execution adapter.
  - `MockBackend`: in-memory backend with 3D position + held-object state.
  - `TraceEvent` + `Trace`: append-only JSON-serialisable event log with
    `state_before` / `state_after`. Includes `write_jsonl()` for fleet ingest.
  - `Runtime`: orchestrator. `step(intent) -> Result`, also
    `run(intents) -> list[Result]`. Catches Primitive exceptions as ERROR
    Results so the loop never crashes the caller.

### Policy gates (`ghostloop.policies`)

  - `DenyListGate`: hard-block named primitives. O(1) set lookup.
  - `RateLimitGate`: per-primitive sliding-window rate limit. Default 120/min.
  - `GeofenceGate`: axis-aligned bounding-box workspace limit. Inspects
    `intent.args` for `x`/`y`/`z` or `target=(x,y,z)`. Pass-through for
    intents with no positional args.

### Primitives (`ghostloop.primitives`)

All bound to `MockBackend` for v0.1; same shape lifts cleanly to MuJoCo /
PyBullet / ROS 2 backends in upcoming releases.

  - `move_to(x, y, z)`: Cartesian teleport, returns observation with
    `from`, `to`, `distance`.
  - `scan(radius=1.0)`: read workspace, returns `center`, `radius`,
    `detections` (empty list for mock).
  - `pick(object_id)`: grasp object. Errors if already holding.
  - `place()`: release held object. Errors if empty.

### Demo (`examples/pick_and_place.py`)

Six-step end-to-end episode:
  scan -> move_to -> pick -> move_to -> place -> deliberate fence violation

The sixth intent (`x=5`) is blocked by GeofenceGate before reaching the
backend. Trace records the `BLOCKED` event with the gate name and reason
exactly as a fleet operator would consume it. Runs against `MockBackend`
with zero installs.

### Tests (`tests/test_core.py`)

  - **Registry**: register / lookup / duplicate raise (3 tests)
  - **Runtime step**: unknown primitive blocks; move_to updates position;
    pick/place round trip; pick-while-holding errors; place-when-empty
    errors; primitive exceptions become ERROR Results (6 tests)
  - **Policy pipeline**: deny-list blocks; geofence blocks outside;
    geofence allows inside; geofence pass-through for argless intents;
    rate-limit blocks after threshold; rate-limit per-primitive; pipeline
    short-circuits on first deny (7 tests)
  - **Trace**: covers every step; JSON-serialisable; state before/after
    differ for movement; `write_jsonl` round-trips (4 tests)
  - **Decision**: allow / deny factories, `to_json` (3 tests)

  **23 passed in 0.35s.**

### Files

```
ghostloop/__init__.py          public API
ghostloop/core.py              Intent / Primitive / Runtime / Trace / Decision
ghostloop/policies/            DenyListGate, RateLimitGate, GeofenceGate
ghostloop/primitives/          move_to, scan, pick, place
examples/pick_and_place.py     end-to-end demo
tests/test_core.py             23 tests
README.md                      vision, architecture, roadmap, quick start
LICENSE                        MIT
pyproject.toml                 0.1.0, no runtime deps
```

### Roadmap (next 6 months)

  v0.2 — `MuJoCoBackend` against the MuJoCo Menagerie (Franka, UR5e, Stretch).
  v0.3 — `PyBulletBackend` + bench harness with Wilson CIs and McNemar.
  v0.4 — `ForceCapGate` + `HumanInTheLoopGate`.
  v0.5 — `LLMPolicy` adapter (any OpenAI-compatible endpoint emits Intents).
  v0.6 — `VLABackend` adapter (OpenVLA / π0 / RT-2 emit primitives).
  v0.7 — `ROS2Backend` for real-hardware deployments.
  v0.8 — MCP server: every primitive becomes a callable MCP tool.
  v1.0 — Production deployment story + fleet management dashboard +
         end-to-end VLA-on-MuJoCo benchmarks.
