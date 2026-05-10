<div align="center">

<img src="assets/ghostloop_wordmark.png" alt="ghostloop" width="560">

**The agent loop, embodied.**

A tool-using agent runtime, fail-closed safety pipeline, statistically-rigorous bench harness, and sim-first execution layer for embodied AI. Sister project to [GhostLM](https://github.com/joemunene-by/GhostLM).

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-v0.9.0%20%E2%80%94%20ROS2%20%2B%20smoothing%20%2B%20safe--projection%20%2B%20reward--shaper%20%2B%20sim2real-14B8A6.svg)](#)
[![Tests](https://img.shields.io/badge/tests-263%20passed%2C%208%20live--gated-14B8A6.svg)](#)

</div>

---

## Why this exists

Robotics in 2026 has two healthy ecosystems and a missing middle.

- **ROS 2** gives you middleware: a message bus, lifecycle management, drivers, navigation. It does not care about LLMs, agents, or modern eval methodology.
- **VLA models** (Open-X-Embodiment, OpenVLA, π0, RT-2) give you policies: vision-and-language conditioned action heads. They mostly live in research codebases that ship the model weights but not the runtime.

Nobody ships **the layer in between**: a runtime where a model emits high-level intents like `move_to(0.4, 0.2, 0.1)` or `pick("widget-7")`, those intents flow through a **fail-closed safety pipeline** (geofence, rate limit, deny list, eventually human-in-the-loop), the survivors execute on a backend (sim today, hardware later), and every step is captured in a structured **trace** that can be replayed, audited, or scored against a benchmark with statistical rigor.

That layer is `ghostloop`. The shape is borrowed from `GhostAgent` in [GhostLM](https://github.com/joemunene-by/GhostLM): tool registry, policy gates, structured trace, paired-comparison eval. The novel piece is binding it to robot primitives instead of CVE lookups, and making the runtime backend-agnostic so the same agent loop drives a `MockBackend` today, MuJoCo right now, and ROS 2 / direct serial later.

## Architecture

```
                policy         registry           pipeline           backend
                 emits          resolves          gates            executes
   user goal  ┌──────────┐   ┌──────────┐    ┌──────────────┐   ┌──────────┐
   ────────► │  Intent  │ ► │ Primitive│ ► │ PolicyPipeline │ ► │ Backend  │ ►
              └──────────┘   └──────────┘    └──────────────┘   └──────────┘
                                                                       │
                                                                       ▼
                                              ┌──────────────────────────────┐
                                              │   Trace (JSONL, replayable)  │
                                              └──────────────────────────────┘
                                                              │
                                                              ▼
                                              ┌──────────────────────────────┐
                                              │ Bench: Wilson CI + McNemar   │
                                              └──────────────────────────────┘
```

| Type | Role |
|---|---|
| `Intent` | A high-level structured command emitted by a policy: `name`, `args`, `rationale`. |
| `Primitive` | A backend-bound callable. Has a name, a description, an arg schema (LLM-tool-card friendly). |
| `PolicyPipeline` | Ordered list of `PolicyGate`s. Fail-closed: any deny short-circuits. |
| `Backend` | Execution adapter. v0.1: `MockBackend`. **v0.2: `MuJoCoBackend`.** |
| `Trace` | Append-only event log with `state_before` / `state_after` per step. JSONL writer included. |
| **`LLMPolicy`** *(v0.2)* | Any OpenAI-compatible chat endpoint emits Intents through the registry's tool schema. |
| **`bench`** *(v0.2)* | Episode harness with Wilson 95% CIs, McNemar exact p, Cohen's h, paired comparison. |

## What ships in v0.2.0

### Core (unchanged from v0.1)
  - 13 abstractions: `Intent` / `Primitive` / `Registry` / `Result` / `Decision` / `PolicyGate` / `PolicyPipeline` / `Backend` / `MockBackend` / `TraceEvent` / `Trace` / `Runtime` (+ enums).

### Policies — now four gates
  - `DenyListGate`: hard-block named primitives (O(1) set lookup)
  - `RateLimitGate`: per-primitive sliding-window rate limit
  - `GeofenceGate`: axis-aligned bounding-box workspace limit
  - **`LLMPolicy`** *(NEW)*: any OpenAI-compatible chat endpoint (Ollama, OpenAI, Anthropic-via-proxy, vLLM, GhostLM's multi-vendor server) emits Intents via the standard tools array. Includes a `done` pseudo-tool for graceful termination. Hallucinated tool calls surface as Intents the runtime resolver blocks (and traces). Best-effort arg coercion (string-floats become floats, string-bools become bools) for weaker models. Zero-SDK dependency: pure `urllib`.

### Backends — sim-first plus MuJoCo

  - `MockBackend`: zero-install in-memory backend.
  - **`MuJoCoBackend`** *(NEW)*: real-physics backend via Google DeepMind's MuJoCo. Loads MJCF / URDF, exposes end-effector pose + joint state via `snapshot()`, drives `mj_step()` integration. Conditional import — package itself imports cleanly without `mujoco`; `MuJoCoBackend(...)` raises `ImportError` with install guidance only at construction time. Comes with MuJoCo-bound `move_to(x, y, z, duration)` and `scan(radius)` primitives.

### Bench harness *(NEW)*

  - `Episode`: declarative trial — `setup()` returns a Backend, `policy(runtime)` yields Intents (or runs its own loop), `success_predicate(trace, state)` scores it.
  - `EpisodeRunner`: drives every episode end-to-end, returns `EpisodeResult`s.
  - `RunReport`: pass count, rate, Wilson 95% CI, Markdown rendering.
  - `paired_compare(a, b)`: pairs two reports episode-by-episode and computes:
    - **Wilson 95% CIs** for each rate (well-behaved at p≈0 / p≈1).
    - **McNemar exact p** via the binomial distribution (numerically stable in log space).
    - **Cohen's h** effect size with negligible / small / medium / large labels.

### Demos

  - `examples/pick_and_place.py`: 6-step scripted episode against `MockBackend`, includes deliberate fence violation that the safety pipeline catches.
  - **`examples/bench_with_without_geofence.py`** *(NEW)*: 8-episode bench paired-compared with and without `GeofenceGate`. Fully reproducible output with Wilson CIs, McNemar p, Cohen's h.

### Tests

  - **64 passed, 5 skipped** *(the 5 are the live MuJoCo integration tests, gated on the `mujoco` package being importable so the offline test path stays at zero install cost)*. All in **0.38s**.

## Quick start

```bash
git clone https://github.com/joemunene-by/ghostloop
cd ghostloop

# Demo 1: end-to-end agent loop with safety pipeline (zero install).
PYTHONPATH=. python3 examples/pick_and_place.py

# Demo 2: paired-comparison bench harness.
PYTHONPATH=. python3 examples/bench_with_without_geofence.py
```

The bench-comparison output:

```
# Paired comparison: with-geofence vs no-gates
Bench: geofence-impact · n=8

| Run           | Passed | Rate   | Wilson 95% CI    |
| no-gates      | 8      | 100.0% | [67.6%, 100.0%]  |
| with-geofence | 4      | 50.0%  | [21.5%, 78.5%]   |

Discordant pairs: only-A=4, only-B=0
McNemar exact p: 0.1250 (not significant — n=4 discordant)
Cohen's h: −1.571 (large)
```

## Use it programmatically

### Run an LLM-driven episode

```python
from ghostloop import Intent, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
from ghostloop.policies import GeofenceGate, LLMPolicyConfig, llm_policy_loop
from ghostloop.primitives import move_to, pick, place, scan

registry = PrimitiveRegistry([move_to(), scan(), pick(), place()])
runtime = Runtime(
    backend=MockBackend(),
    registry=registry,
    policy_pipeline=PolicyPipeline(gates=[
        GeofenceGate(min_corner=(-1, -1, 0), max_corner=(1, 1, 1)),
    ]),
)

summary = llm_policy_loop(
    registry=registry,
    runtime=runtime,
    goal="Pick widget-7 from (0.4, 0.2, 0.1) and place it at (-0.4, 0.2, 0.1).",
    config=LLMPolicyConfig(
        base_url="http://localhost:11434/v1",  # Ollama default
        model="qwen2.5:14b",
    ),
    max_steps=16,
)
print(summary["terminated"], summary["steps"])
runtime.trace.write_jsonl("episode.jsonl")
```

### Run a paired-comparison bench

```python
from ghostloop.bench import EpisodeRunner, paired_compare, summarize

a = summarize(EpisodeRunner().run_all(eps_a), run_name="no-gates", bench_name="my-bench")
b = summarize(EpisodeRunner().run_all(eps_b), run_name="with-fence", bench_name="my-bench")
print(paired_compare(a, b).render_md())
```

### Drive a real physics simulation

```python
from ghostloop import PolicyPipeline, PrimitiveRegistry, Runtime
from ghostloop.backends import MuJoCoBackend
from ghostloop.backends.mujoco import move_to, scan

backend = MuJoCoBackend(model_path="franka_panda.xml", end_effector="hand")
registry = PrimitiveRegistry([move_to(), scan()])
runtime = Runtime(backend=backend, registry=registry, policy_pipeline=PolicyPipeline())

runtime.step(Intent("move_to", {"x": 0.4, "y": 0.0, "z": 0.5, "duration": 1.0}))
runtime.step(Intent("scan", {"radius": 0.5}))
```

Models from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) drop in directly: Franka Panda, UR5e, Stretch RE3, Allegro hand, Spot, Aloha bimanual, etc.

## Roadmap

| Version | Focus |
|---|---|
| v0.1.0 | Core abstractions, MockBackend, three policy gates, runnable demo, 23 tests |
| v0.2.0 | MuJoCoBackend, LLMPolicy adapter, bench harness with Wilson CIs + McNemar + Cohen's h, paired-comparison demo, 64 tests |
| v0.3.0 | PyBulletBackend, async runtime, declarative properties engine, MCP server, scripted policies, 89 tests |
| v0.4.0 | ForceCap + HumanInTheLoop gates, episode catalogue, MuJoCo Menagerie auto-clone, replay/diff CLI, 110 tests |
| v0.5.0 | VLAPolicy adapter, sensor primitives + cameras, OpenTelemetry hooks, SQLite persistence, planner DSL, 142 tests |
| v0.6.0 | Fleet abstraction, FastAPI dashboard, LLMPlanner, RetryGate, observation buffer, property combinators, 182 tests |
| v0.7.0 | GymnasiumBackend, CooldownGate + TimeWindowGate, convex polytope SDF, composite primitives, Mission DAG runner, WebSocket trace streaming, 211 tests |
| v0.8.0 | STL temporal properties, URDF workspace builder, RandomizedBackend, trace query DSL, safe-RL harness with Lagrangian, 239 tests |
| **v0.9.0 (now)** | **`ROS2Backend` (rclpy adapter, conditional)**, **`ActionSmoothingGate` (velocity / acceleration limits)**, **safe-action projection (analytic + SDF)**, **declarative reward shaper DSL**, **`Sim2RealBench` paired transfer-gap harness**, 263 tests |
| v1.0 | Multi-modal perception (RGB-D fusion + lightweight object detection), end-to-end VLA-on-MuJoCo benchmarks reproducing OpenVLA / π0 numbers under our trace + safety regime, fleet dashboard productionised. |

## Repository layout

```
ghostloop/
  __init__.py        — public API surface
  core.py            — Intent / Primitive / Runtime / Trace / Decision
  policies/
    deny_list.py     — DenyListGate
    geofence.py      — GeofenceGate
    rate_limit.py    — RateLimitGate
    llm.py           — LLMPolicy + LLMPolicyConfig + llm_policy_loop  (v0.2)
  primitives/
    motion.py        — move_to, scan (MockBackend)
    manipulation.py  — pick, place (MockBackend)
  backends/
    mujoco.py        — MuJoCoBackend + move_to + scan (v0.2)
  bench/             — Episode / RunReport / paired_compare           (v0.2)
    episode.py       — Episode + EpisodeRunner + EpisodeResult
    report.py        — RunReport + wilson_ci + summarize
    compare.py       — PairedComparison + mcnemar_p + cohens_h
examples/
  pick_and_place.py              — scripted end-to-end demo
  bench_with_without_geofence.py — paired-comparison demo (v0.2)
tests/
  test_core.py                   — 23 tests
  test_llm_policy.py             — 14 tests (v0.2, urllib mocked)
  test_bench.py                  — 22 tests (v0.2)
  test_mujoco_backend.py         — 5 offline + 5 live-gated  (v0.2)
assets/
  ghostloop_mark*.png            — favicon-sized + social card variants
  ghostloop_wordmark*.png        — full lockup
```

## Why this is novel

There are robot frameworks. There are agent frameworks. There is no robot framework that **treats robots as a model with a tool registry, a fail-closed safety gate, a structured trace log, and statistical bench rigor** — the same shape that is now standard for LLM-driven cybersec agents (`secure-mcp`, `ghostguard`, `GhostAgent`). The thesis: as VLA models become the policy substrate, the runtime around them needs the same rigor we already apply to LLM tool use. ghostloop is that runtime.

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Joe Munene](https://github.com/joemunene-by) at [Complex Developers](https://github.com/complexdevelopers). Sibling to [GhostLM](https://github.com/joemunene-by/GhostLM), [secure-mcp](https://github.com/joemunene-by/secure-mcp), [ghostguard](https://github.com/joemunene-by/ghostguard), [CyberBench](https://github.com/joemunene-by/cyberbench).
