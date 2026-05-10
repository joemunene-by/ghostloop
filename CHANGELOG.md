# Changelog

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
