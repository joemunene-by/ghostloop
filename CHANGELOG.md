# Changelog

## [0.7.0] — 2026-05-10 — Gymnasium + Cooldown + TimeWindow + SDF + Composite + Missions + Streaming

Six additions that pull production infrastructure forward of the original
roadmap. The package now sits one Backend short of being a serious
RL/embodied-AI substrate (Gymnasium ecosystem unlocked), gains two
ops-grade rate gates, ships a real signed-distance-field workspace,
introduces composite primitives so policies can reason at a higher level,
adds a Mission DAG runner with retry semantics, and exposes a live
WebSocket stream of trace events for dashboards.

### GymnasiumBackend (ghostloop/backends/gymnasium.py)

Wraps any [Farama Gymnasium](https://gymnasium.farama.org) environment
(or legacy `gym`) as a ghostloop Backend. Conditional import — the
package itself doesn't depend on gymnasium; constructing a
`GymnasiumBackend` raises `ImportError` with an install hint:

  pip install ghostloop[gym]

That single bridge brings hundreds of envs into the ghostloop runtime:
classic control, MuJoCo locomotion, FetchReach / Push, Atari,
pybullet-gym ports, every research env that publishes a Gym-style
`step(action)` contract. Every safety gate, bench harness, MissionRunner,
and trace replay we ship now works against that ecosystem out of the
box. Two primitives bind to the backend: `apply_action()` (raw action
vector — what VLAPolicy and scripted RL controllers emit) and
`reset_env()` (multi-episode benches).

### CooldownGate + TimeWindowGate (ghostloop/policies/{cooldown,time_window}.py)

Two ops-grade rate gates that turned out to be the most-asked-for
extensions to the v0.4 RateLimitGate.

  CooldownGate     enforces a minimum interval between successive calls
                   of the same primitive (per-primitive overrides on top
                   of a default). Implemented with `time.monotonic()` so
                   it survives wall-clock shenanigans. Useful for hardware
                   that needs a settling period after each motion, or for
                   any primitive whose cost dominates throughput planning.

  TimeWindowGate   gate primitives by time-of-day. Common ops constraint
                   (warehouse robot 06:00-22:00, lab arm 09:00-18:00,
                   farm robot sunrise-to-sunset). Windows can cross
                   midnight for night-shift ops. ``now`` is injected as a
                   callable so tests can drive synthetic clocks.

### Convex polytope + half-space SDF (ghostloop/policies/sdf.py)

The v0.5 WorkspaceModel handles axis-aligned boxes + spheres; this
release adds half-spaces (floor / ceiling / wall), convex polytopes
(intersection of half-spaces — the right shape for a robot body acting
as an obstacle to its fleet siblings), and a `signed_distance(p, ws)`
function that returns the negative-inside / positive-outside metric
across the whole workspace. Stdlib math; no numpy. Inigo Quilez's box
SDF formula. Pairs cleanly with future proximity-based slow-down
policies.

### Composite primitives (ghostloop/primitives/composite.py)

Real robots have macros: `approach_grasp = (move_to_pre_grasp,
descend_to_grasp_pose, close_gripper, lift)`. Defining each macro as a
single `Primitive` keeps the registry small and lets policies (and LLM
tool cards) reason at the right level of abstraction. The
`composite_primitive(name, steps, ...)` factory sequences existing
primitives behind one name; sub-primitive observations are preserved
under `substep_<i>_<name>` keys; first non-OK substep stops the chain.

### Mission orchestrator (ghostloop/missions/)

DAG of Steps with dependencies, Kahn's-algorithm cycle detection, retry
semantics, required-vs-optional steps, KeyError-safe aggregation. The
`MissionRunner` runs the DAG against any `Runtime` (so the same safety
pipeline applies to multi-step missions); a `MissionResult` aggregates
per-step status (`SUCCEEDED / FAILED / SKIPPED`) and the overall mission
status (`SUCCEEDED / PARTIAL / FAILED`). Failed required steps propagate
to dependents as `SKIPPED`; failed optional steps are tolerated and
their dependents still run.

### WebSocket trace streaming (ghostloop/dashboard/streaming.py)

The v0.6 dashboard exposes the SQLite store as JSON; this adds the live
side. `StreamManager` is a sync-publish / async-subscribe fan-out with a
bounded ring buffer per robot (so a newly-connected client gets a small
replay instead of a blank screen). `attach_streaming(app, manager)`
registers `/ws/v1/stream` on a FastAPI app. Robot runtimes call
`manager.publish(robot_name, event)` synchronously; the publish path is
non-blocking (saturated subscribers drop the event rather than slowing
the publisher).

### Tests

29 new tests in `tests/test_v07_additions.py` across all six surfaces:

  TestGymnasiumConditional   2  ImportError + apply_action/reset_env primitives
  TestCooldownGate           4  default + per-primitive + non-blocking + reset
  TestTimeWindowGate         5  inside / outside / midnight / pass-through / unconfigured
  TestSDF                    5  bounds clearance, sphere SDF (in+out), half-space, polytope
  TestCompositePrimitive     2  sequential dispatch + first-failure-stops-chain
  TestMission                8  DAG topology, cycle detection, retry, optional skip
  TestStreamManager          3  publish/subscribe, history replay, drop-on-saturation

  Total package: 211 passed, 7 live-gated.

## [0.6.0] — 2026-05-10 — Fleet + Dashboard + LLMPlanner + Retry + Diff + Observations + Combinators

Eight additions across the production-ops surface. Multi-robot becomes
first-class, the dashboard backend turns the SQLite store into a real
HTTP API, LLM-driven planning lands as the bridge between VLAPolicy and
TaskPlanner, and the operator toolkit (retry, diff, observation memory,
property combinators, two new CLI subcommands) closes the gap between
"runs in lab" and "runs in production".

### Fleet (ghostloop/fleet/)

Multi-robot abstractions over single-robot Runtimes.

  RobotHandle      one robot — runtime + status + labels + last_seen
                   heartbeat metadata. step() updates lifecycle status.
  FleetRegistry    name -> RobotHandle index with selection helpers
                   (filter_by_status, filter_by_label, stale-detection).
  FleetDispatcher  load-balanced submission across robots via three
                   strategies: FIRST_IDLE / ROUND_ROBIN / LEAST_BUSY.
                   Pin to a specific robot via dispatch(intent, robot=).
  FleetSnapshot    JSON-safe view aggregating idle/busy/error/offline
                   counts plus per-robot snapshots — the data shape the
                   dashboard / fleet UI consumes.
  FleetError       raised on duplicate registration / unknown robot /
                   empty fleet dispatch.
  RobotStatus enum  IDLE / BUSY / OFFLINE / ERROR.

### Dashboard (ghostloop/dashboard/)

Read-only FastAPI HTTP surface over the SQLite store + an optional
attached FleetRegistry. Conditional install:

  pip install ghostloop[dashboard]

create_dashboard_app(store, fleet=None, title=...) returns a ready
ASGI app. Endpoints:

  GET /healthz
  GET /v1/store/stats
  GET /v1/store/episodes?limit=N&backend=...
  GET /v1/store/episodes/{episode_id}
  GET /v1/store/runs?limit=N&bench_name=...
  GET /v1/store/runs/{run_id}
  GET /v1/fleet                          (only if fleet attached)
  GET /v1/fleet/{robot_name}             (only if fleet attached)

Wired up alongside this release as the data layer for a future Next.js
+ tRPC fleet UI matching the broader ghostloop ecosystem stack.
``GhostloopStore`` now opens with ``check_same_thread=False`` so the
FastAPI thread pool can read concurrently.

### LLMPlanner (ghostloop/planning/llm_planner.py)

Sister to LLMPolicy. Where LLMPolicy emits ONE intent per turn (closed
loop), LLMPlanner asks the model for the FULL plan up front via a
single ``submit_plan(steps=[...])`` tool call, then the runtime
executes that plan under the safety pipeline.

  - Same OpenAI-compatible wire format / config as LLMPolicy.
  - tool_choice forces submit_plan, no free-form responses.
  - JSON schema enumerates the registered primitive names so the model
    can't hallucinate unknowns (or if it does, the runtime BLOCKS them
    via the resolver).
  - Per-step rationale + top-level rationale recorded in PlanResult.

Use cases: hand-curated few-shot prompts (regression tests),
hierarchical control (top-level decomposition + low-level controller),
cost optimisation (one LLM call per episode beats N calls).

### RetryPolicy (ghostloop/policies/retry.py)

Wraps a Runtime and re-dispatches transient ERROR results.

  - max_attempts cap (default 3 = 1 + 2 retries).
  - Exponential backoff with configurable factor + jitter_frac.
  - is_transient_error: matches network/timeout/busy patterns
    (timeout, temporarily, try again, rate limit, busy, connection
    reset, broken pipe, 503, 502, 429).
  - is_any_error: aggressive retry — only when you trust the API.
  - Custom predicates pluggable via ``classify``.
  - BLOCKED results NEVER retried — those came from the safety
    pipeline and the policy must respect them.

### Trace diff (ghostloop/traces/diff.py)

Side-by-side comparison of two Trace JSONLs. Step-by-step diff
classifies each row as identical / diverged / only_a / only_b. First-
divergence helper for fast triage; render_md() produces a publication-
ready table for incident postmortems and policy A/B comparisons.

### ObservationBuffer (ghostloop/observations.py)

Fixed-size deque of recent ObservationRecords (intent name + args +
status + observation + state_after + duration). Short-term memory the
runtime can populate after each step and policies can read from
without iterating the trace. capacity, latest(), n_recent(),
filter_by_intent(), n_blocked(), n_errored(), JSON-safe.

### Property combinators (ghostloop/properties/combinators.py)

  AndProperty(children)  holds iff every child holds; concatenates
                         violations with from-child tags
  OrProperty(children)   holds iff any child holds
  NotProperty(child)     inverts; useful for 'must violate' regression
                         tests

Severity defaults to max(children); explicit override available.

### CLI: 2 new subcommands

  python -m ghostloop eval --preset reach_8 [--with-geofence] [--out FILE]
  python -m ghostloop diff <a.jsonl> <b.jsonl> [--json]

eval runs any of the three catalogue presets through EpisodeRunner +
summarize and emits a Markdown report (Wilson CI rendered). diff
loads two trace files and renders the diff table directly.

### Tests

  test_v06_additions.py NEW — 37 tests:
    ObservationBuffer            5
    RetryPolicy                  5
    Property combinators         5
    TraceDiff                    4
    LLMPlanner                   2
    Fleet                        9
    Dashboard (conditional)      3 (1 live-gated on fastapi install)
    CLI: eval + diff             4

  Suite: 145 -> 182 passing, 7 skipped (live-only) in 3.16s.

### Pyproject

  version 0.5.0 -> 0.6.0
  optional-dependencies: + dashboard = fastapi + uvicorn[standard]

### What ghostloop is now

Six releases deep, ~7,000 LOC source + 3,000 LOC tests, 182 passing
tests, six optional installs (mujoco / pybullet / mcp / otel / dashboard
/ dev), three backends, three policy adapters (LLM / VLA / scripted),
six policy gates (DenyList / RateLimit / Geofence / ForceCap / HITL /
ObstacleAvoidance), four declarative properties + three combinators,
six primitives, three planners (PickAndPlace / Traverse / LLM), seven
CLI subcommands (info / demo / bench / replay / store / mcp / eval +
diff). Multi-robot fleet ops + read-only HTTP dashboard ship with this
release; what was a single-robot agent loop is now a fleet operations
platform.

---

## [0.5.0] — 2026-05-10 — VLAPolicy + properties engine + workspace + trajectories + planners

Five additions that complete the policy / safety / planning surface:
the VLA model adapter from the original roadmap, declarative trace-level
safety properties (richer than single-step gates), proper geometric
workspace + obstacle modelling, trajectory primitives + linear
interpolation, and a TaskPlanner module for goal -> intent decomposition.

### VLAPolicy (ghostloop/policies/vla.py)

Adapter for OpenVLA / pi-0 / RT-2 and any VLA model that emits action
vectors. Decoupled into:

  - VLAModel callable: ``(observation_dict) -> action_vector``. Works
    for local HF checkpoints, remote inference endpoints, modal /
    vLLM-served models, and hand-rolled scripted policies in tests.
  - ActionDecoder Protocol: maps action vector + current state to a
    structured Intent the registry can dispatch.
  - DeltaXYZDecoder: standard OpenVLA / RT-2 head — first 3 dims are
    end-effector position deltas, dim 6 is the gripper open/close.
    Emits ``move_to`` for position deltas above the deadband, ``pick``
    on gripper close transitions, ``place`` on opens. State-aware
    (tracks last_gripper) so it doesn't emit duplicate picks.
  - VLAPolicy: orchestrator with idle-detection termination.
  - vla_policy_loop: end-to-end driver, returns a summary.

The safety pipeline still gates every emitted Intent. A VLA model can't
violate the geofence — its delta gets dispatched as a normal move_to,
the GeofenceGate evaluates the resulting target, and BLOCKED travels
back through the trace just like for any LLMPolicy / scripted run.

### Properties engine (ghostloop/properties/)

Declarative invariants that span the trace, not single-step gate checks.

  Property Protocol:    name + severity + check(trace) -> PropertyResult.
  PropertyEngine:       evaluates a list of Properties, summarises with
                        held / violated / error_violations counts.
  PropertyResult:       per-property structured violation list.
  Severity:             INFO / WARN / ERROR (ERROR = ship-blocking).

Four built-in properties:

  StaysInsideWorkspace        every state_after.position must stay in box
  NeverHoldsTwoObjects        held_object must transition through None
  NoConsecutiveDuplicateIntents  catch policies stuck in tight loops
  NeverExceedsRate            per-primitive rate cap on wall-clock timestamps

Use cases: CI gates, bench scoring beyond pass/fail, post-incident
analysis.

### WorkspaceModel + ObstacleAvoidanceGate (ghostloop/policies/workspace.py)

Real geometry, not just an axis-aligned box.

  Sphere(center, radius, inflation, label):  spherical obstacle with
                                              optional safety margin
  AxisAlignedBox(min, max, inflation, label): box obstacle
  WorkspaceModel(bounds, obstacles):         outer bounds + obstacle list
  ObstacleAvoidanceGate(workspace):           policy gate that uses it

Targets are valid iff inside outer bounds AND outside every obstacle.
Inflation radius adds a safety margin without changing the physical
obstacle shape (a 5cm cup with 2cm inflation rejects targets within 7cm).

### Trajectory primitives (ghostloop/primitives/trajectory.py)

  follow_trajectory(waypoints, dwell_s=0.0):  visit a list of [x,y,z]
                                                in sequence
  linear_interpolate(start, end, n=10):       generate evenly-spaced
                                                waypoints (helper)

Useful when the policy already knows the path (planned, recorded teleop,
hand-tuned approach) and the safety pipeline should still gate every
waypoint.

### TaskPlanner module (ghostloop/planning/)

High-level goal -> Intent sequence decomposition. The bench harness
already accepts any callable that yields Intents; planners give that
callable a structured shape.

  Planner Protocol:  goal -> PlanResult (name, intents, rationale, metadata)
  PlanResult:        Intent list + JSON-serialisable metadata

Two built-in planners:

  PickAndPlacePlanner({object_id, pickup, drop}):  scan -> move -> pick
                                                    -> move -> place;
                                                    use_trajectory flag
                                                    swaps single move
                                                    for follow_trajectory
  TraversePlanner([waypoints], scan_at_each):      move (+optional scan)
                                                    per waypoint

Custom planners drop in alongside via the Protocol — LLM-backed and
search-based planners are obvious next-release adds.

### Tests

  test_v05_additions.py NEW — 32 tests:
    DeltaXYZDecoder           6
    VLAPolicy                 2
    WorkspaceModel            5
    ObstacleAvoidanceGate     3
    Trajectory primitives     4
    Properties engine         6
    Planning module           6

  Suite: 113 -> 145 passing, 6 skipped (live-only) in 1.73s.

### Pyproject

  version 0.4.0 -> 0.5.0
  no new optional deps (pure Python additions)

### Roadmap state after this push

Originally-planned v0.5 items:
  VLABackend adapter ✓ (shipped as VLAPolicy + DeltaXYZDecoder)
Originally-planned v0.6 items:
  Vision pipeline (camera primitives) ✓ (shipped in v0.4)

Not on the original roadmap but shipped this release:
  Declarative properties engine
  WorkspaceModel + ObstacleAvoidanceGate
  Trajectory primitives + linear_interpolate
  TaskPlanner module + PickAndPlacePlanner + TraversePlanner

Roadmap effectively pulled forward by 1.5 versions in one push session.

---

## [0.4.0] — 2026-05-10 — AsyncRuntime + SQLite store + vision + MCP server + OpenTelemetry

Production-infrastructure release. The pieces that turn ghostloop from
"demo-ready" into "could be deployed at a robotics startup tomorrow":
async control loops, persistent episode + run history, sensor abstraction,
MCP exposure to any agent, and OTel observability hooks.

### AsyncRuntime (ghostloop/async_runtime.py)

Coroutine-friendly runtime for control-loop and network workloads. Same
Intent / Primitive / PolicyPipeline / Backend / Trace surface as Runtime,
but ``step`` is ``async def`` and supports:

  - AsyncPolicyGate Protocol (awaitable check method).
  - AsyncPolicyPipeline that mixes sync + async gates transparently
    via inspect.isawaitable — sync gates pass through unchanged.
  - Async primitives: any Primitive whose .call is a coroutine function
    is awaited. Sync primitives stay sync.
  - control_loop(next_intent, max_steps, rate_hz): closed control loop
    driven by a callable that returns the next Intent given the previous
    Result. ``rate_hz`` paces the loop with sleep-correction.

The sync Runtime still ships unchanged for sim / scripted / single-shot
workloads. AsyncRuntime is for HITL-with-Slack-webhook, ROS 2 backends,
and real control loops.

### SQLite-backed store (ghostloop/store.py)

Single-file dependency-free persistence — sqlite3 in stdlib, no Postgres
required. Three tables:

  episodes      — every Trace ever ingested. Full JSONL stored for replay,
                  metadata extracted into indexed columns.
  run_reports   — every RunReport ever scored. Per-episode pass/fail in
                  JSON blob; aggregate rate / Wilson CI indexed.
  comparisons   — every PairedComparison ever computed.

Content-addressed: re-ingesting the same artefact is a no-op.
``GhostloopStore`` is both a context manager and direct constructor.
``EpisodeRow`` / ``RunRow`` dataclasses for typed row access.

### Vision sensors (ghostloop/sensors/)

  - Camera (Protocol): anything that produces CameraFrames on demand.
  - CameraFrame: JSON-safe metadata (intrinsics, timestamps, shapes,
    extras) PLUS opaque rgb / depth payload (numpy / PIL / bytes).
  - CameraIntrinsics: pinhole intrinsics with to_json().
  - CameraProcessor (Protocol): pluggable post-processor for object
    detection / depth refinement / segmentation.
  - MockCamera: in-memory deterministic gradient-RGB + constant-depth
    camera for tests + sim demos.
  - capture_camera() Primitive: looks up camera by name on
    ``backend.cameras``, captures, returns frame metadata in the
    observation. Falls back to MockCamera when backend has no cameras
    configured (keeps demos usable without sim setup).

### MCP server (ghostloop/mcp_server.py)

Conditional import — package itself doesn't depend on mcp. Calling
``run_mcp_server(runtime)`` requires the SDK:

  pip install ghostloop[mcp]

Exposes four general tools (list_primitives, step, recent_trace, state)
PLUS one auto-tool per Primitive in the registry. Every tool call passes
through the safety pipeline; blocked actions return structured BLOCKED
results with gate name + reason. Same FastMCP pattern as GhostLM's MCP
server.

### Telemetry / OpenTelemetry (ghostloop/telemetry.py)

  pip install ghostloop[otel]

Conditional. ``configure_otel(service_name)`` attaches a Tracer; reads
OTEL_EXPORTER_OTLP_ENDPOINT etc from env via the standard SDK auto-
configuration so any production OTel setup (Honeycomb / Jaeger / Grafana
Tempo / Datadog) just works. ``step_span(intent)`` is a context manager
that emits a per-step span with intent / decision / result attributes;
no-op when OTel isn't installed or configured. ``record_decision`` /
``record_result`` helpers attach the structured fields.

### CLI: 2 new subcommands

  python -m ghostloop store stats [--db PATH]      counts of episodes/runs/comparisons
  python -m ghostloop store episodes [--db PATH] [--limit N]
  python -m ghostloop store runs [--db PATH] [--limit N]
  python -m ghostloop mcp [--name NAME]            run the MCP server (stdio)

  Default db path: ~/.ghostloop/store.db (auto-created).

### Tests

  test_v04_additions.py NEW — 21 tests:
    AsyncRuntime              6
    GhostloopStore            5
    Camera sensors            5
    Telemetry                 2
    MCP server (conditional)  3 (1 live-gated on `mcp` install)

  Suite: 93 -> 113 passing, 6 skipped (live MuJoCo + live MCP) in 1.21s.

### Pyproject

  version 0.3.0 -> 0.4.0
  optional-dependencies:
    + mcp>=1.0
    + opentelemetry-api / -sdk / -exporter-otlp >=1.20

### Why this is the "real production" release

  - Async control loops unblock ROS 2 / gRPC / network-IO backends.
  - Persistent store lets ops ask "how have my policies performed
    over time" not just "this one bench run".
  - Camera abstraction is the missing primitive for any vision-based
    policy (VLA models, object-detection-driven planners).
  - MCP server makes Claude Desktop / Cursor / any MCP-aware agent a
    first-class robot driver.
  - OTel hooks mean a fleet operator can wire ghostloop into existing
    observability stacks with one env-var.

The runtime that pairs with VLA models is no longer hypothetical;
production-grade scaffolding ships in this release.

---

## [0.3.0] — 2026-05-10 — PyBullet + Menagerie loader + episode catalogue + replay + 5 gates + CLI

Six substantial additions in one release. Pulls forward roadmap items v0.3
(PyBullet, episode catalogue, trace replay) AND v0.4 (ForceCap + HITL
gates), plus adds the MuJoCo Menagerie loader and a `python -m ghostloop`
CLI that wasn't on the roadmap at all.

### PyBulletBackend (ghostloop/backends/pybullet.py)

Bullet Physics backend, BSD-licensed, single-wheel install on every major
OS. Conditional import: package itself imports without ``pybullet``;
backend errors with install hint at construction time. URDF loading,
end-effector link pose snapshot, joint position control + reset,
DIRECT/GUI mode, gravity + timestep wired. Bound `move_to` and `scan`
primitives shipping with names matching MockBackend + MuJoCoBackend so
policies stay backend-agnostic.

  pip install ghostloop[pybullet]

### MuJoCo Menagerie loader (ghostloop/backends/menagerie.py)

Resolves friendly model names (`"franka"`, `"ur5e"`, `"stretch"`, etc.)
to the right XML inside the MuJoCo Menagerie. Two cases:

  - Menagerie cloned locally (set MENAGERIE_PATH or pass menagerie_root):
    we find the model XML and return its absolute path.
  - User has nothing: we shallow-clone (--depth=1 --filter=blob:none)
    into ~/.cache/ghostloop/mujoco_menagerie on first use (~80 MB).
    Subsequent calls reuse the clone.

10 known model aliases shipped: franka, panda, ur5e, ur10e, stretch,
allegro, spot, aloha, shadow, sawyer. ``load_franka()`` is a one-liner
helper that returns a ready-to-use MuJoCoBackend.

### Episode catalogue (ghostloop/bench/catalogue.py)

Pre-built episode builders for the bench harness:

  reach_targets(targets)        — N-target move suite
  pick_and_place_pairs(pairs)   — pick A, place B suite
  scan_at_targets(waypoints)    — visit each waypoint and scan
  geofence_violations(in, out)  — half-inside, half-outside regression suite

Plus three convenience presets: preset_reach_8, preset_pick_and_place_4,
preset_geofence_smoke (the 8-episode suite the v0.2 demo used inline).

### Trace replay (ghostloop/traces/)

Read trace JSONL back into structured ReplayedEvent records. Three entry
points:

  load_trace(path)        full read into (TraceHeader, list[ReplayedEvent])
  iter_events(path)       streaming iterator (large traces)
  summarize_trace(path)   high-level dict with by_status / by_intent /
                          deny_reasons / total_duration_ms

The inverse of Trace.write_jsonl(); enables regression tests, fleet
ingestion, post-hoc analysis, and the new `replay` CLI subcommand.

### Two new policy gates

ForceCapGate (ghostloop/policies/force_cap.py):
  Reject intents whose declared force / torque / velocity / acceleration /
  effort / speed exceeds a configured cap. Pass-through for intents
  without those keys; uses absolute values; non-numeric values silently
  skipped (don't crash on weird input). Caps default to None (= unlimited).

HumanInTheLoopGate (ghostloop/policies/human_in_the_loop.py):
  Block selected primitives until an approver returns True. Synchronous
  by design — the runtime blocks on it. Three approvers shipped:
  always_approve, always_deny, cli_approver (stdin prompt). Production
  deployments wire Slack-webhook / dashboard-poller / queue readers.

Pipeline now ships five gates total: DenyList, RateLimit, Geofence,
ForceCap, HITL.

### CLI (ghostloop/__main__.py)

  python -m ghostloop info               — version + backends + primitives
  python -m ghostloop demo               — bundled scripted demo
  python -m ghostloop bench              — geofence-impact paired bench
  python -m ghostloop replay <trace>     — summarise a JSONL trace
  python -m ghostloop replay <trace> --json   — machine-readable output

Plus a console script: after install, `ghostloop` is on PATH.

### Tests

  test_v03_additions.py NEW — 25 tests:
    PyBullet conditional      2
    ForceCapGate              6
    HumanInTheLoopGate        5
    Episode catalogue         5
    Trace replay              3
    Menagerie loader (offline) 3
    CLI subcommands           5

  Suite: 64 -> 93 passing, 5 skipped (live MuJoCo) in 0.68s.

### pyproject

  version 0.2.0 -> 0.3.0
  optional-dependencies: + pybullet>=3.2
  Added [project.scripts] ghostloop = "ghostloop.__main__:main"

### Roadmap progress

  v0.3 originally planned: PyBullet ✓, episode catalogue ✓, replay ✓
  v0.4 originally planned: ForceCap ✓, HITL ✓
  Bonus shipped: MuJoCo Menagerie loader, CLI, console script

  Next planned (v0.4 / v0.5):
    - VLABackend adapter (OpenVLA / pi-0 / RT-2 emit primitives directly)
    - Vision pipeline: camera primitives, RGB-D fusion
    - End-to-end demo: LLM-driven pick on a real Franka in the menagerie

---

## [0.2.0] — 2026-05-10 — MuJoCoBackend + LLMPolicy + bench harness + brand assets

Three substantial additions on top of v0.1, all shipped together so the v0.2
story is "ghostloop now has a brain (LLMPolicy), a body (MuJoCoBackend), and
rigor (bench harness with Wilson CIs + McNemar)."

### LLMPolicy adapter (ghostloop/policies/llm.py)

Any OpenAI-compatible chat endpoint emits Intents through the registry's
tool schema. Same wire format the GhostLM multi-vendor server speaks
(OpenAI / Anthropic / Gemini / Ollama / vLLM all consume), so a single
adapter works across every model vendor.

  - LLMPolicyConfig: base_url, model, api_key, temperature, max_tokens.
    Defaults to Ollama localhost so demos run with no cloud spend.
  - LLMPolicy(registry, config): owns its own message history. ask(obs)
    sends the last observation, returns the next Intent (or None on
    'done').
  - llm_policy_loop(registry, runtime, goal, max_steps): end-to-end
    driver that closes the loop until the model calls 'done' or the
    step budget runs out. Returns a summary dict.
  - Hallucinated tool calls surface as Intents the runtime resolver
    blocks (and traces) — no silent failures.
  - Best-effort arg coercion: string-floats -> floats, string-bools ->
    bools, string-ints -> ints. Catches the common weak-model output.
  - Zero SDK dependency: pure urllib so the package's "no runtime
    deps" promise holds.

### MuJoCoBackend (ghostloop/backends/mujoco.py)

Real-physics backend via Google DeepMind's MuJoCo. Conditional import —
the package itself imports cleanly without mujoco; MuJoCoBackend(...)
raises ImportError with install guidance only at construction time.

  - MuJoCoBackend(model_path, end_effector, timestep, name): loads
    MJCF / URDF, holds (mj.MjModel, mj.MjData), exposes end-effector
    position + qpos via snapshot().
  - Low-level helpers: advance(duration), set_qpos(qpos),
    set_actuator(idx, value).
  - MuJoCo-bound move_to(x, y, z, duration) and scan(radius)
    primitives that share names with the MockBackend versions, so
    policies are backend-agnostic at the Intent layer.
  - mujoco_available() helper for tests / demos that need to branch.
  - Models from the MuJoCo Menagerie (Franka, UR5e, Stretch, etc.)
    drop in directly — documented in README, not vendored.

### Bench harness (ghostloop/bench/)

Statistically-rigorous episode benchmarking, mirroring the GhostBench
pattern from GhostLM.

  - Episode: declarative trial — setup() returns Backend, policy(rt)
    yields Intents (or runs its own loop), success_predicate(trace,
    state) scores it.
  - EpisodeRunner: drives every episode, returns EpisodeResult list.
  - RunReport: pass count, rate, Wilson 95% CI, Markdown rendering.
  - wilson_ci(successes, n): well-behaved at p≈0 / p≈1 / small n
    (where the Normal approximation breaks).
  - paired_compare(a, b): pairs two reports episode-by-episode,
    computes:
      McNemar exact p-value via the binomial distribution, log-space
        arithmetic for numerical stability at large n.
      Cohen's h via arcsine-transformed difference of proportions,
        labelled negligible / small / medium / large.
  - PairedComparison.render_md() produces a publication-ready table
    with both rates, both Wilson CIs, discordant-pair counts, p, and h.

### Brand assets (assets/)

Sibling design to the GhostLM mark — same Pac-Man-ghost silhouette
family for instant brand recognition, teal (#14B8A6) instead of coral
for product distinction, with a circular arrow ↻ inside the body
(negative space) instead of a G. Vertical gradient on the body for
depth, soft outer glow for polish, white eyes for face consistency.

  ghostloop_mark.png            transparent crop, ~1024px source
  ghostloop_mark_{64,128,256,512,1024}.png  square padded variants
  ghostloop_wordmark.png        full lockup ('ghost' dark + 'loop' teal)
  ghostloop_wordmark_small.png  half-height inline lockup

### Demos

  examples/pick_and_place.py                    (unchanged from v0.1)
  examples/bench_with_without_geofence.py  NEW — 8-episode bench, paired
    comparison of geofence on/off, full Wilson + McNemar + Cohen's h
    output. Reproduces in 0.1s.

### Tests

  - test_core.py             23 (unchanged)
  - test_llm_policy.py       14 NEW (urllib mocked, no live LLM needed)
  - test_bench.py            22 NEW (Wilson math, McNemar exact,
                                    Cohen's h, runner, report, compare)
  - test_mujoco_backend.py   5 offline + 5 live-gated NEW

  **64 passed, 5 skipped in 0.38s.**

### Repository layout (after v0.2)

```
ghostloop/
  __init__.py
  core.py
  policies/{deny_list,geofence,rate_limit,llm}.py
  primitives/{motion,manipulation}.py
  backends/mujoco.py
  bench/{episode,report,compare}.py
examples/{pick_and_place,bench_with_without_geofence}.py
tests/{test_core,test_llm_policy,test_bench,test_mujoco_backend}.py
assets/ghostloop_{mark*,wordmark*}.png
```

### pyproject

  version 0.1.0 -> 0.2.0
  Added optional-dependencies group: ghostloop[mujoco] -> mujoco>=3.0

### What's next (v0.3)

  PyBulletBackend (no-MuJoCo path)
  Episode catalogue: pick-place, sort, stack, navigate
  Trace replay tooling
  Then ForceCapGate + HumanInTheLoopGate (v0.4)

---

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
