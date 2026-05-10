<div align="center">

# ghostloop

**The agent loop, embodied.**

A tool-using agent runtime, fail-closed safety pipeline, and sim-first execution harness for embodied AI. Sister project to [GhostLM](https://github.com/joemunene-by/GhostLM).

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-v0.1.0%20%E2%80%94%20core%20runtime%20%2B%20policy%20pipeline%20%2B%20mock%20backend%20%2B%2023%20tests-blue.svg)](#)

</div>

---

## Why this exists

Robotics in 2026 has two healthy ecosystems and a missing middle.

- **ROS 2** gives you middleware: a message bus, lifecycle management, drivers, navigation. It does not care about LLMs, agents, or modern eval methodology.
- **VLA models** (Open-X-Embodiment, OpenVLA, π0, RT-2) give you policies: vision-and-language conditioned action heads. They mostly live in research codebases that ship the model weights but not the runtime.

Nobody ships **the layer in between**: a runtime where a model emits high-level intents like `move_to(0.4, 0.2, 0.1)` or `pick("widget-7")`, those intents flow through a **fail-closed safety pipeline** (geofence, rate limit, deny list, eventually human-in-the-loop), the survivors execute on a backend (sim today, hardware later), and every step is captured in a structured **trace** that can be replayed, audited, or scored against a benchmark.

That layer is `ghostloop`. The shape is borrowed from `GhostAgent` in [GhostLM](https://github.com/joemunene-by/GhostLM): tool registry, policy gates, structured trace, paired-comparison eval. The novel piece is binding it to robot primitives instead of CVE lookups, and making the runtime backend-agnostic so the same agent loop drives a `MockBackend` today, MuJoCo and PyBullet next, and ROS 2 / direct serial later.

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
```

Five concepts, all JSON-serialisable:

| Type | Role |
|---|---|
| `Intent` | A high-level structured command emitted by a policy: `name`, `args`, `rationale`. |
| `Primitive` | A backend-bound callable. Has a name, a description, an arg schema (LLM-tool-card friendly). |
| `PolicyPipeline` | Ordered list of `PolicyGate`s. Fail-closed: any deny short-circuits. |
| `Backend` | Execution adapter. v0.1 ships `MockBackend`. v0.2 lands MuJoCo. |
| `Trace` | Append-only event log with `state_before` / `state_after` per step. JSONL writer included. |

## What ships in v0.1.0

  - **Runtime + registry + trace**: the core loop and the JSON-serialisable event log.
  - **Three policy gates** (`DenyListGate`, `RateLimitGate`, `GeofenceGate`), composable with deterministic short-circuit semantics.
  - **MockBackend**: zero-install in-memory backend with 3D position + held object — enough for tests, examples, and the bench harness.
  - **Four primitives**: `move_to`, `scan`, `pick`, `place`, all bound to `MockBackend`.
  - **Runnable end-to-end demo**: `examples/pick_and_place.py` drives the full loop, including a deliberate fence violation that the pipeline catches.
  - **23 tests** covering registry, runtime step semantics, every policy gate, trace serialisation, and edge cases (pick-while-holding, place-when-empty, primitive exceptions caught as ERROR results).

## Quick start

```bash
# zero install — run the demo straight from the repo
git clone https://github.com/joemunene-by/ghostloop
cd ghostloop
PYTHONPATH=. python3 examples/pick_and_place.py
```

```
  [OK ] scan       -> scanned 0.5m sphere from (0.0, 0.0, 0.0)
  [OK ] move_to    -> moved 0.4583 units
  [OK ] pick       -> picked 'widget-7'
  [OK ] move_to    -> moved 0.8 units
  [OK ] place      -> placed 'widget-7'
  [BLK] move_to    -> geofence: target x=5 outside workspace [-1,1]
```

The sixth intent is a deliberate overshoot. The geofence gate catches it before it reaches the backend; the runtime records a `BLOCKED` event with the gate name and reason in the trace. That's the safety pipeline doing its job.

## Use it programmatically

```python
from ghostloop import Intent, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
from ghostloop.policies import DenyListGate, GeofenceGate, RateLimitGate
from ghostloop.primitives import move_to, pick, place, scan

runtime = Runtime(
    backend=MockBackend(),
    registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
    policy_pipeline=PolicyPipeline(gates=[
        DenyListGate(denied=set()),
        RateLimitGate(per_minute=600),
        GeofenceGate(min_corner=(-1, -1, 0), max_corner=(1, 1, 1)),
    ]),
)
result = runtime.step(Intent("move_to", {"x": 0.5, "y": 0.0, "z": 0.2}))
print(result.ok, result.observation)
```

`runtime.trace.write_jsonl("episode.jsonl")` writes a header line plus one JSON-per-event for replay or fleet ingestion.

## Roadmap

| Version | Focus |
|---|---|
| v0.1.0 (now) | Core abstractions, MockBackend, three policy gates, runnable demo, 23 tests |
| v0.2 | `MuJoCoBackend` against the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda, UR5e, Stretch). Real motion physics. |
| v0.3 | `PyBulletBackend` for users without MuJoCo. Bench harness with Wilson 95% CIs and McNemar paired comparisons (the GhostBench pattern from GhostLM). |
| v0.4 | Two more gates: `ForceCapGate` (deny torque > N), `HumanInTheLoopGate` (block until an external review approves). |
| v0.5 | `LLMPolicy` adapter: any OpenAI / Anthropic / Gemini / Ollama / GhostLM endpoint emits Intents via a tool-call schema, runtime executes them. |
| v0.6 | `VLABackend` adapter: OpenVLA / π0 / RT-2 emit primitives directly; ghostloop adds the safety + trace + bench layer they're missing. |
| v0.7 | `ROS2Backend` for real-hardware deployments via DDS. |
| v0.8 | MCP server: every primitive becomes a callable MCP tool, so Claude Desktop / Cursor / any MCP client can drive the robot through the safety pipeline. |
| v1.0 | Production deployment story, fleet management dashboard (Next.js + tRPC), end-to-end VLA-on-MuJoCo benchmarks reproducing OpenVLA / π0 numbers under our trace + safety regime. |

## Repository layout

```
ghostloop/
  __init__.py        — public API surface
  core.py            — Intent / Primitive / Runtime / Trace / Decision
  policies/          — DenyListGate, RateLimitGate, GeofenceGate
  primitives/        — move_to, scan, pick, place (MockBackend-bound)
  backends/          — (v0.2: MuJoCo, v0.3: PyBullet, v0.7: ROS 2)
  traces/            — (v0.3: replay tooling, episode store)
examples/
  pick_and_place.py  — end-to-end demo; runs against MockBackend
tests/
  test_core.py       — 23 tests covering runtime, gates, trace
docs/
  (incoming v0.2 design notes for the MuJoCo backend)
```

## Why this is novel

There are robot frameworks. There are agent frameworks. There is no robot framework that **treats robots as a model with a tool registry, a fail-closed safety gate, and a structured trace log** — the same shape that is now standard for LLM-driven cybersec agents (`secure-mcp`, `ghostguard`, `GhostAgent`). The thesis: as VLA models become the policy substrate, the runtime around them needs the same rigor we already apply to LLM tool use. ghostloop is that runtime.

## License

MIT. See [LICENSE](LICENSE).

---

Built by [Joe Munene](https://github.com/joemunene-by) at [Complex Developers](https://github.com/complexdevelopers). Sibling to [GhostLM](https://github.com/joemunene-by/GhostLM), [secure-mcp](https://github.com/joemunene-by/secure-mcp), [ghostguard](https://github.com/joemunene-by/ghostguard), [CyberBench](https://github.com/joemunene-by/cyberbench).
