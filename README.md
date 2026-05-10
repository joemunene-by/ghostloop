<div align="center">

<img src="assets/ghostloop_wordmark.png" alt="ghostloop" width="560">

**The agent loop, embodied.**

A tool-using agent runtime, fail-closed safety pipeline, statistically-rigorous bench harness, and sim-first execution layer for embodied AI. Sister project to [GhostLM](https://github.com/joemunene-by/GhostLM).

**Now live:** [`pip install ghostloop`](https://pypi.org/project/ghostloop/) · [interactive demo](https://huggingface.co/spaces/Ghostgim/ghostloop-demo) · 11 releases · 333 tests · MIT.

[![PyPI](https://img.shields.io/pypi/v/ghostloop?color=14B8A6&label=pypi)](https://pypi.org/project/ghostloop/)
[![Downloads](https://static.pepy.tech/badge/ghostloop/month)](https://pepy.tech/project/ghostloop)
[![HF Space](https://img.shields.io/badge/🤗%20live%20demo-Ghostgim%2Fghostloop--demo-FFD21E)](https://huggingface.co/spaces/Ghostgim/ghostloop-demo)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-v1.0.2%20%E2%80%94%20production-14B8A6.svg)](#)
[![Tests](https://img.shields.io/badge/tests-359%20passed%2C%208%20live--gated-14B8A6.svg)](#)
[![CI](https://github.com/joemunene-by/ghostloop/actions/workflows/ci.yml/badge.svg)](https://github.com/joemunene-by/ghostloop/actions/workflows/ci.yml)

</div>

---

## Why this exists

Robotics in 2026 has two healthy ecosystems and a missing middle.

- **ROS 2** gives you middleware: a message bus, lifecycle management, drivers, navigation. It does not care about LLMs, agents, or modern eval methodology.
- **VLA models** (Open-X-Embodiment, OpenVLA, π0, RT-2) give you policies: vision-and-language conditioned action heads. They mostly live in research codebases that ship the model weights but not the runtime.

Nobody ships **the layer in between**: a runtime where a model emits high-level intents like `move_to(0.4, 0.2, 0.1)` or `pick("widget-7")`, those intents flow through a **fail-closed safety pipeline**, the survivors execute on a backend (sim or hardware), and every step is captured in a structured **trace** that can be replayed, audited, scored, mined, counterfactually re-played, or causally analysed.

That layer is `ghostloop`. The shape is borrowed from `GhostAgent` in [GhostLM](https://github.com/joemunene-by/GhostLM): tool registry, policy gates, structured trace, paired-comparison eval. The novel piece is binding it to robot primitives instead of CVE lookups, making the runtime backend-agnostic so the same agent loop drives a mock today, MuJoCo / PyBullet / Gymnasium right now, and ROS 2 / direct hardware later — and adding a layer of post-hoc analysis tooling (counterfactual replay, causal attribution, LLM-as-judge, property mining, adversarial fuzzing) that no other robotics framework ships.

## Architecture

```
                policy        registry           pipeline           backend         post-hoc
                 emits         resolves           gates           executes         analysis
   user goal  ┌────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────────────┐
   ────────► │ Intent │ ► │ Primitive │ ► │PolicyPipeline│ ► │ Backend  │ ► │ counterfactual  │
              └────────┘   └──────────┘   └──────────────┘   └──────────┘   │ causal          │
                                                                  │         │ LLM-judge       │
                                                                  ▼         │ property mining │
                                              ┌──────────────────────┐      │ adversarial     │
                                              │  Trace (JSONL)       │ ───► │ trace query DSL │
                                              └──────────────────────┘      │ energy ledger   │
                                                       │                    └─────────────────┘
                                                       ▼
                                  ┌────────────────────────────────────┐
                                  │ Bench: Wilson CI + McNemar +       │
                                  │ Cohen's h + Sim2Real transfer gap  │
                                  └────────────────────────────────────┘
```

| Type | Role |
|---|---|
| `Intent` | High-level structured command emitted by a policy: `name`, `args`, `rationale`. |
| `Primitive` | Backend-bound callable. Has a name, description, arg schema (LLM-tool-card friendly). |
| `PolicyPipeline` | Ordered list of `PolicyGate`s. Fail-closed: any deny short-circuits. |
| `Backend` | Execution adapter. `MockBackend` / `MuJoCoBackend` / `PyBulletBackend` / `GymnasiumBackend` / `ROS2Backend` / `RandomizedBackend`. |
| `Trace` | Append-only event log with `state_before` / `state_after` / `decision` / `result` per step. JSONL writer + replay + query DSL. |
| `LLMPolicy` / `VLAPolicy` | Bridge any OpenAI-compatible chat endpoint or VLA action head to the registry. |
| `Mission` | DAG of Steps with prerequisites + retry semantics. Kahn-validated. |
| `bench` | Episode harness with Wilson 95% CIs, McNemar exact p, Cohen's h, paired comparison, sim2real transfer gap. |
| `properties` | Declarative invariants over traces — `Always` / `Eventually` / `Until` STL combinators + auto-mined candidates. |
| `judges` | LLM-as-judge + heuristic rule-based trace scoring. |
| `training` | Constrained-MDP rollout collector + Lagrangian multiplier + HER relabeling. |

## What ships in v0.10.0

70+ modules across ten releases. Highlights:

### Core runtime
13 abstractions in `core.py` (Intent / Primitive / Result / Decision / PolicyGate / PolicyPipeline / Backend / MockBackend / TraceEvent / Trace / Runtime / Registry / DecisionAction). `async_runtime.py` mirrors them with awaitable gates + a `control_loop(rate_hz)`.

### Policy gates (12)
`DenyListGate`, `RateLimitGate`, `GeofenceGate`, `ForceCapGate`, `HumanInTheLoopGate`, `ObstacleAvoidanceGate`, `RetryPolicy`, `CooldownGate`, `TimeWindowGate`, `ActionSmoothingGate` (velocity / acceleration limits), plus the `LLMPolicy` and `VLAPolicy` adapters. All fail-closed.

### Backends (6)
- **`MockBackend`** — zero-install in-memory.
- **`MuJoCoBackend`** — Google DeepMind MuJoCo with Menagerie auto-clone (Franka / UR5e / Spot / Stretch / Aloha / Allegro).
- **`PyBulletBackend`** — Bullet physics for users without MuJoCo.
- **`GymnasiumBackend`** — wrap any Farama Gymnasium env (hundreds of robotics + RL envs).
- **`ROS2Backend`** — rclpy adapter for real-hardware deployments via DDS.
- **`RandomizedBackend`** — wrap any backend with reproducible noise / jitter / dropout for sim2real.

### Workspace + geometry
`WorkspaceModel` with axis-aligned boxes + spheres, `HalfSpace` / `ConvexPolytope` / `signed_distance` for SDF queries, `workspace_from_urdf(...)` to auto-build from a URDF, plus `project_to_workspace` / `project_to_sdf` for safe-action repair when a policy violates constraints.

### Bench harness
- `Episode` / `EpisodeRunner` / `RunReport` with Wilson 95% CIs.
- `paired_compare` — McNemar exact p + Cohen's h.
- `Sim2RealBench` — paired transfer-gap harness with per-primitive action-distribution KL.
- `random_seeds` / `grid_seeds` / `cma_es_seeds` — adversarial fuzzers for finding failure-prone Episode initial states.
- `RewardShaper` — declarative reward DSL (`OnPrimitive` / `OnDecision` / `OnObservation` / `StepCost` / `CustomReward`).
- Episode catalogue: `preset_reach_8` / `preset_pick_and_place_4` / `preset_geofence_smoke`.

### Properties + verification
- `PropertyEngine` with built-in invariants (`StaysInsideWorkspace`, `NeverHoldsTwoObjects`, `NeverExceedsRate`, `NoConsecutiveDuplicateIntents`).
- `Always` / `Eventually` / `Until` STL combinators over sliding windows.
- `AndProperty` / `OrProperty` / `NotProperty` boolean combinators.
- `mine_properties(traces)` — auto-discover candidate invariants from a corpus (followup transitions, numeric bounds, workspace AABBs).

### Post-hoc analysis (the v0.10 novel pillars)
- `replay_with_policy(trace, new_policy)` — counterfactual reasoning. "What would policy B have done on policy A's trace?"
- `attribute_failure(trace, property)` — leave-one-out causal attribution; ranks events by necessity.
- `minimal_cause_set` — greedy multi-event attribution.
- `LLMJudge` — score traces with an LLM against a configurable rubric.
- `HeuristicJudge` — rule-based predicate scoring for air-gapped CI.

### Missions + skills
- `Mission` / `Step` / `MissionRunner` — DAG of steps with prerequisites, retry semantics, required-vs-optional.
- `SkillGraph` — typed DAG of skills with prereq + refines edges.
- `MorphologyRegistry` — register `pick` per `(franka, ur5e, spot)` and build per robot.
- `composite_primitive` — sequence existing primitives behind one name.

### Training (constrained-MDP + HER)
- `SafeRolloutCollector` + `LagrangianMultiplier` + `train_safe` — train policies under the safety pipeline; safety violations contribute to a Lagrangian penalty.
- `hindsight_relabel(rollout, goal_extractor, reward_fn)` — classic HER (Andrychowicz et al. 2017) with FINAL / FUTURE / EPISODE / RANDOM strategies.
- `sparse_indicator_reward(threshold)` — canonical HER reward.

### Telemetry + persistence
- OpenTelemetry hooks (`step_span`, `record_decision`, `record_result`).
- `EnergyLedger` — per-primitive joule accounting with constant / linear-in-arg / linear-in-duration / linear-in-xyz models.
- `GhostloopStore` — SQLite store for episodes / runs / comparisons.
- `Trace.write_jsonl()` + `load_trace()` + `iter_events()` + `summarize_trace()`.
- `query(trace, expr)` — small DSL over traces (comparison ops + `and`/`or`/`not`/`in`).
- `diff_traces(a, b)` — structured diff for ablation studies.

### Fleet + dashboard
- `RobotHandle` / `FleetRegistry` / `FleetDispatcher` (FIRST_IDLE / ROUND_ROBIN / LEAST_BUSY).
- `create_dashboard_app(store, fleet)` — read-only FastAPI surface over the SQLite store.
- `StreamManager` + `attach_streaming(app)` — WebSocket trace streaming with bounded ring buffers.

### MCP + LLM integration
- `mcp_server.py` exposes the runtime as a FastMCP server so Claude Desktop / Cursor / any MCP client can drive a robot through the safety pipeline.
- `LLMPolicy` (closed-loop) and `LLMPlanner` (single-shot full-plan emission).

## Setup

There are three ways to run ghostloop, in order of effort. **Start with the zero-install path, prove the safety pipeline, then promote to your real arm.**

### 1. Zero-install (3 minutes)

```bash
git clone https://github.com/joemunene-by/ghostloop
cd ghostloop

# Run the canonical pick-and-place demo on MockBackend.
PYTHONPATH=. python3 examples/pick_and_place.py

# Run a paired-comparison bench (Wilson CI + McNemar + Cohen's h).
PYTHONPATH=. python3 examples/bench_with_without_geofence.py

# Full test suite — 314 pass, 8 live-gated (skip cleanly without extras).
PYTHONPATH=. python3 -m pytest tests/
```

No dependencies beyond Python 3.10+. This proves the runtime, the safety pipeline, the bench harness, and the trace recorder — exactly the same code you'll point at a real arm later.

### 2. Drive *any* robot from any chat client over MCP (10 minutes)

ghostloop ships a single MCP server (`examples/mcp_robot.py`) that works with **every MCP-aware client** — the protocol is universal, so the same server speaks to Claude Desktop, Cursor, Continue, Cline, Zed, Gemini CLI, and any future client. Pick what you control via `GHOSTLOOP_PROFILE`:

| Profile | Robot | Primitives exposed |
|---|---|---|
| `franka_arm` (default) | Franka Panda 7-DOF arm | `set_joint`, `set_gripper`, `sense`, `take_photo`, … |
| `turtlebot` | TurtleBot mobile base | `drive`, `stop`, `goto`, `rotate`, … |
| `spot` | Boston Dynamics Spot quadruped | `walk_to`, `sit`, `stand`, `lie_down`, … |
| `tello` | DJI Tello / quadcopter | `takeoff`, `land`, `fly_to`, `hover`, … |
| `stretch` | Hello Robot Stretch RE3 (mobile arm) | `drive`, `set_joint`, `set_gripper`, … |
| `humanoid_demo` | Stationary humanoid | `wave`, `look_at`, `point_at`, `nod` |
| `<path/to/your.yaml>` | **Your robot** | whatever you declare |

Each preset bundles morphology-appropriate primitives, conservative workspace + force + velocity caps, HITL on the dangerous primitives, and a robot-specific instructions block the LLM gets as a system prompt. See `examples/custom_robot.yaml` for the YAML schema and `examples/custom_robot_primitives.py` for how to plug in your own actions (`dispense_pill`, `alert_nurse`, whatever your hardware does) without forking ghostloop.

Two transports, picked via `GHOSTLOOP_TRANSPORT`:

| Transport | When to use | Clients |
|---|---|---|
| `stdio` (default) | desktop, same machine | Claude Desktop, Cursor, Continue, Cline, Zed, Gemini CLI |
| `streamable-http` | remote, mobile, browser, kiosk | any client supporting remote MCP servers |

**Step 1.** Verify the example boots (any OS, any profile):

```bash
# Default (Franka arm)
python3 examples/mcp_robot.py --selfcheck

# Quadruped
GHOSTLOOP_PROFILE=spot python3 examples/mcp_robot.py --selfcheck

# Drone
GHOSTLOOP_PROFILE=tello python3 examples/mcp_robot.py --selfcheck

# Custom YAML
GHOSTLOOP_PROFILE=examples/custom_robot.yaml python3 examples/mcp_robot.py --selfcheck
```

**Step 2.** Install the MCP transport package:

```bash
pip install ghostloop[mcp]      # or: pip install mcp
```

**Step 3.** Wire it into your client. The same `{ command, args, env }` block works for every desktop MCP client — only the **path to the config file** differs:

| Client | macOS | Windows | Linux |
|---|---|---|---|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` | `%APPDATA%\Claude\claude_desktop_config.json` | `~/.config/Claude/claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json` (or project-local `.cursor/mcp.json`) | `%USERPROFILE%\.cursor\mcp.json` | `~/.cursor/mcp.json` |
| Continue | `~/.continue/config.yaml` (under `mcpServers:`) | same | same |
| Cline | VS Code `settings.json` → `cline.mcpServers` | same | same |
| Zed | `~/.config/zed/settings.json` (under `context_servers`) | same | same |
| Gemini CLI | `~/.gemini/settings.json` (under `mcpServers`) | same | same |

Paste this block into the config file (replace the absolute path; pick a profile that matches your robot):

```jsonc
{
  "mcpServers": {
    "ghostloop": {
      "command": "python3",
      "args": ["/absolute/path/to/ghostloop/examples/mcp_robot.py"],
      "env": {
        "GHOSTLOOP_PROFILE": "franka_arm",
        "GHOSTLOOP_BACKEND": "mock",
        "GHOSTLOOP_TRANSPORT": "stdio",
        "GHOSTLOOP_INSTRUCTIONS": "Optional: extra robot-specific guidance appended to the profile's instructions block."
      }
    }
  }
}
```

> 💡 On Windows, swap `python3` for `python` (or the absolute path to your interpreter, e.g. `C:\Python313\python.exe`). On macOS, use `python3` from Homebrew or pyenv. Continue + Zed use YAML / JSONC respectively but the field shape is identical.

**Step 4.** Restart the client. New conversations get the tools: `list_primitives`, `step`, `move_to(x, y, z)`, `pick(object_id)`, `place()`, `scan(radius)`, `state`, `recent_trace(n)`. Try: *"Use the ghostloop tools to move to (0.4, 0.0, 0.5), then scan with radius 0.3, then move to (0.6, 0.2, 0.5)."* Watch the geofence reject targets outside [-0.6, 0.6].

**Upgrade from mock to real physics (MuJoCo) — one env var:**

```jsonc
"env": {
  "GHOSTLOOP_BACKEND": "mujoco",
  "GHOSTLOOP_MUJOCO_MODEL": "/absolute/path/to/franka_panda.xml"
}
```
(`pip install ghostloop[mujoco]` first.)

**Upgrade to a real arm via ROS 2:**

```jsonc
"env": {
  "GHOSTLOOP_BACKEND": "ros2",
  "GHOSTLOOP_ROS2_NODE": "ghostloop_arm",
  "GHOSTLOOP_CMD_VEL": "/franka/cmd",
  "GHOSTLOOP_JOINT_STATES": "/franka/joint_states",
  "GHOSTLOOP_FORCE_TORQUE": "/franka/wrench"
}
```

Prerequisites: ROS 2 (`apt install ros-humble-desktop` on Ubuntu, the [Robotology Mac install](https://github.com/RoboStack/ros-humble) on macOS, [WSL2 + Ubuntu](https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Install-Binary.html) on Windows), your arm's ROS 2 driver running, and `source /opt/ros/humble/setup.bash` (or `setup.zsh` / `setup.ps1`) in the same shell that launches the client so the subprocess inherits `$AMENT_PREFIX_PATH`.

> ⚠ **Before pointing this at a real robot:** edit your profile (or copy a preset to YAML and tweak it) — set `workspace_bounds` / `max_force_n` / `max_velocity` / `max_acceleration` to your hardware's safe envelope, list dangerous primitives under `hitl_primitives` so the operator approves each call interactively, and write robot-specific guidance into the `instructions:` block (e.g. *"never reach behind the base"*, *"battery below 20% triggers automatic land"*). Read the trace logs for the first dozen episodes; relax HITL only after you trust the model's behaviour.

#### Define your own robot

Two ways to add a robot ghostloop doesn't already know about:

**A. YAML profile** (no Python required) — copy `examples/custom_robot.yaml`, edit it, and point `GHOSTLOOP_PROFILE` at the path. The schema covers categories of standard primitives, your own custom primitives, composite macros, instructions for the LLM, workspace + force + velocity caps, denied / HITL operations, and the backend kind. The shipped sample defines a hospital medication-delivery robot — mobile base + arm, with custom `dispense_pill` and `alert_nurse` primitives and a `deliver_room` macro composed from existing primitives:

```yaml
name: medbot_floor3
morphology: mobile_arm
categories: [mobile_base, dexterous, sensing, generic]
instructions: |
  You are MedBot, hospital floor-3 medication delivery. NEVER drive faster
  than 0.4 m/s. ALWAYS stop before extending the arm. ...
workspace_bounds: [[-15, -15, 0], [15, 15, 1.6]]
max_velocity: 0.4
hitl_primitives: [set_gripper, dispense_pill]
custom_primitives:
  - module: examples.custom_robot_primitives
    factory: dispense_pill
  - module: examples.custom_robot_primitives
    factory: alert_nurse
composites:
  - name: deliver_room
    steps: [goto, take_photo, dispense_pill, alert_nurse]
backend:
  kind: ros2
  kwargs: { node_name: medbot, cmd_vel_topic: /medbot/cmd_vel }
```

**B. Code** — build a `RobotProfile` programmatically. Useful when your robot needs runtime state (a calibration matrix, a credential, dynamically-resolved topic names) that doesn't fit YAML:

```python
from ghostloop.profiles import RobotProfile, build_runtime_from_profile
from ghostloop.primitives import drive, set_gripper
from my_robot.primitives import dispense_pill, alert_nurse

profile = RobotProfile(
    name="medbot",
    morphology="mobile_arm",
    primitives=[drive(), set_gripper(), dispense_pill(), alert_nurse()],
    instructions="You are MedBot...",
    workspace_bounds=((-15, -15, 0), (15, 15, 1.6)),
    max_velocity=0.4,
    hitl_primitives=["dispense_pill"],
    backend_kind="ros2",
    backend_kwargs={"node_name": "medbot", "cmd_vel_topic": "/medbot/cmd_vel"},
)
runtime = build_runtime_from_profile(profile)
```

Custom Primitive factories follow a stable contract: a function returning `Primitive(name, call, description, arg_schema)`. The `call` body talks to your hardware however you need it to — ROS 2 publisher, vendor SDK, raw serial, REST endpoint. See `examples/custom_robot_primitives.py` for two worked examples (`dispense_pill`, `alert_nurse`).

### 3. Mobile + remote MCP (HTTP transport)

For mobile chat apps (and any client that doesn't run on the same machine as the robot), run ghostloop's MCP server as a long-running HTTP service on the robot host:

```bash
# macOS / Linux
GHOSTLOOP_BACKEND=mock GHOSTLOOP_TRANSPORT=streamable-http \
GHOSTLOOP_HOST=0.0.0.0 GHOSTLOOP_PORT=8765 \
  python3 examples/claude_desktop_mcp_arm.py

# Windows PowerShell
$env:GHOSTLOOP_TRANSPORT='streamable-http'; $env:GHOSTLOOP_HOST='0.0.0.0'; $env:GHOSTLOOP_PORT='8765'
python examples\claude_desktop_mcp_arm.py
```

Then configure remote-MCP-capable clients with the **URL form** (no `command`/`args`):

```jsonc
{
  "mcpServers": {
    "ghostloop": { "url": "http://your-robot-host.local:8765/mcp" }
  }
}
```

Mobile MCP clients (Claude iOS once it ships remote MCP, plus the growing crop of third-party MCP-aware iOS / Android chat apps) connect via the same URL — no app-side install. For a custom mobile app, use any MCP TypeScript / Swift / Kotlin SDK from [modelcontextprotocol.io](https://modelcontextprotocol.io). The HTTP wire format is the same.

> ⚠ Bind to `0.0.0.0` only on a private network or behind authentication. The default `127.0.0.1` is loopback-only (safer). For internet-exposed setups, put a reverse proxy with TLS + auth in front, or use the production dashboard's `StaticTokenAuth` pattern.

### 4. Without MCP — direct OpenAI-compatible function calling

Already have a model running and don't want to bother with MCP? `examples/direct_llm_arm.py` skips the protocol entirely and uses ghostloop's `LLMPolicy` to drive any OpenAI-compatible chat endpoint via native function calling. Tested against:

- **OpenAI** GPT-4o / GPT-4o-mini
- **Anthropic** Claude (via the OpenAI-compatible proxy endpoint)
- **Google Gemini** (via OpenAI-compatible adapter)
- **Groq** (Llama 3.x, DeepSeek, Mixtral)
- **Ollama** (local Qwen, Llama, Mistral, GhostLM)
- **vLLM** + **llama.cpp server** + **GhostLM's multi-vendor server**

```bash
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... \
OPENAI_MODEL=gpt-4o-mini \
  python3 examples/direct_llm_arm.py

# Or local Ollama:
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
OPENAI_MODEL=qwen2.5:14b \
  python3 examples/direct_llm_arm.py
```

Same Backend choice (Mock / MuJoCo / ROS 2), same safety pipeline, same trace recorder. Only the LLM-to-tool plumbing differs: in-process via `LLMPolicy` instead of MCP wire protocol.

### 5. Run programmatically

For everything else — bench harnesses, training loops, post-hoc analysis — use ghostloop as a library. Examples below.

## Library API examples

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
    config=LLMPolicyConfig(base_url="http://localhost:11434/v1", model="qwen2.5:14b"),
    max_steps=16,
)
runtime.trace.write_jsonl("episode.jsonl")
```

### Drive a real physics simulation

```python
from ghostloop import PolicyPipeline, PrimitiveRegistry, Runtime, Intent
from ghostloop.backends import MuJoCoBackend
from ghostloop.backends.mujoco import move_to, scan

backend = MuJoCoBackend(model_path="franka_panda.xml", end_effector="hand")
registry = PrimitiveRegistry([move_to(), scan()])
runtime = Runtime(backend=backend, registry=registry, policy_pipeline=PolicyPipeline())

runtime.step(Intent("move_to", {"x": 0.4, "y": 0.0, "z": 0.5, "duration": 1.0}))
runtime.step(Intent("scan", {"radius": 0.5}))
```

Models from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) drop in directly: Franka Panda, UR5e, Stretch RE3, Allegro hand, Spot, Aloha bimanual.

### Counterfactual replay — "what would the new policy have done?"

```python
from ghostloop.counterfactual import replay_with_policy
from ghostloop.traces import load_trace

original = load_trace("episode.jsonl")

def new_policy(state_before):
    # any callable mapping state -> Intent | None
    return Intent("scan", {"radius": 0.3})

cf = replay_with_policy(original, new_policy, new_policy_name="more-cautious")
print(cf.divergence_rate, cf.first_divergence_step)
print(cf.render_md())
```

### Causal failure attribution

```python
from ghostloop.causal import attribute_failure, minimal_cause_set
from ghostloop.properties import StaysInsideWorkspace

prop = StaysInsideWorkspace(min_corner=(-1, -1, 0), max_corner=(1, 1, 1))
analysis = attribute_failure(failing_trace, prop)
print(analysis.render_md())          # ranked top-K root causes

cause_set = minimal_cause_set(failing_trace, prop, max_set_size=3)
```

### LLM-as-judge

```python
from ghostloop.judges import LLMJudge, LLMJudgeConfig

class GhostLMClient:
    def chat(self, messages, **kwargs):
        # adapt your chat endpoint here
        ...

judge = LLMJudge(client=GhostLMClient(), config=LLMJudgeConfig(model="ghostlm-v0.9-chat"))
judgement = judge.score(trace)
print(judgement.label, judgement.score, judgement.rubric_scores)
```

### Adversarial fuzzing

```python
from ghostloop.bench import cma_es_seeds

def perturb(base_episode, sample):
    # return a copy of base_episode with backend initial state shifted by `sample`
    ...

results = cma_es_seeds(
    base_episode, perturb,
    parameter_ranges={"x0": (-1.0, 1.0), "y0": (-1.0, 1.0)},
    n_iterations=8, population_size=8, seed=42,
)
worst = results[:5]    # promote into your regression bench
```

### Property mining

```python
from ghostloop.properties import mine_properties

corpus = [load_trace(p) for p in successful_traces_paths]
candidates = mine_properties(corpus, min_support=0.9)
for mp in candidates:
    print(mp.pattern, mp.description, mp.support)
    promoted = mp.promote()        # a real Property ready for the engine
```

### Sim-to-Real bench

```python
from ghostloop.bench import Sim2RealBench

bench = Sim2RealBench(
    sim_episodes=eps_sim,
    real_episodes=eps_real,
    sim_label="mujoco", real_label="randomized_mujoco",
)
report = bench.run()
print(report.render_md())          # transfer gap + McNemar + KL action-distribution
```

### Energy ledger

```python
from ghostloop.telemetry import EnergyLedger

ledger = EnergyLedger()
print(ledger.total(trace), "J")
print(ledger.by_primitive(trace))
```

### Skill graph + cross-embodiment

```python
from ghostloop.skills import SkillGraph, skill_from_primitive
from ghostloop.primitives import MorphologyRegistry, move_to, scan

graph = SkillGraph()
graph.add(skill_from_primitive(move_to()))
graph.add(skill_from_primitive(scan(), prerequisites=["move_to"]))
graph.validate()
order = graph.topological_order()        # ['move_to', 'scan']

reg = MorphologyRegistry()
reg.register("franka", "pick", franka_pick_factory)
reg.register("ur5e",   "pick", ur5e_pick_factory)
prims = reg.build("franka", ["pick"])    # robot-specific primitives
```

## Roadmap

| Version | Focus |
|---|---|
| v0.1.0 | Core abstractions, MockBackend, three policy gates, runnable demo, 23 tests |
| v0.2.0 | MuJoCoBackend, LLMPolicy adapter, bench harness with Wilson CIs + McNemar + Cohen's h, 64 tests |
| v0.3.0 | PyBulletBackend, async runtime, declarative properties engine, MCP server, scripted policies, 89 tests |
| v0.4.0 | ForceCap + HumanInTheLoop gates, episode catalogue, MuJoCo Menagerie auto-clone, replay/diff CLI, 110 tests |
| v0.5.0 | VLAPolicy adapter, sensor primitives + cameras, OpenTelemetry hooks, SQLite persistence, planner DSL, 142 tests |
| v0.6.0 | Fleet abstraction, FastAPI dashboard, LLMPlanner, RetryGate, observation buffer, property combinators, 182 tests |
| v0.7.0 | GymnasiumBackend, CooldownGate + TimeWindowGate, convex polytope SDF, composite primitives, Mission DAG runner, WebSocket trace streaming, 211 tests |
| v0.8.0 | STL temporal properties, URDF workspace builder, RandomizedBackend, trace query DSL, safe-RL harness with Lagrangian, 239 tests |
| v0.9.0 | ROS2Backend, ActionSmoothingGate, safe-action projection, reward shaper DSL, Sim2RealBench, 263 tests |
| v0.10.0 | Counterfactual trace replay, causal failure attribution, LLM-as-judge for traces, adversarial bench generator, property mining, skill graph, hindsight relabeling, energy ledger, cross-embodiment morphology registry, 296 tests |
| **v1.0.0 (now)** | **RGB-D fusion + deproject_depth + BlobDetector + CameraProcessorPipeline**, **VLABenchmarkSuite + published-baseline catalogue (OpenVLA / π0 / RT-2 / Octo / Diffusion Policy / ACT)**, **production fleet dashboard (StaticTokenAuth / RateLimiter / AlarmRegistry / Prometheus /metrics / livez+readyz)**, 314 tests |

## Repository layout

```
ghostloop/
  __init__.py                public API surface, version
  core.py                    Intent / Primitive / Runtime / Trace / Decision / Backend / MockBackend
  async_runtime.py           AsyncRuntime + control_loop(rate_hz)
  observations.py            ObservationBuffer (deque-based short-term memory)
  store.py                   GhostloopStore — SQLite episodes / runs / comparisons
  mcp_server.py              FastMCP server exposing Runtime as MCP tools
  counterfactual.py          replay_with_policy + CounterfactualTrace        (v0.10)
  causal.py                  attribute_failure + minimal_cause_set            (v0.10)

  policies/
    deny_list.py             DenyListGate
    rate_limit.py            RateLimitGate
    geofence.py              GeofenceGate
    force_cap.py             ForceCapGate
    human_in_the_loop.py     HumanInTheLoopGate + cli_approver
    workspace.py             WorkspaceModel + ObstacleAvoidanceGate
    sdf.py                   HalfSpace / ConvexPolytope / signed_distance     (v0.7)
    urdf.py                  workspace_from_urdf                              (v0.8)
    cooldown.py              CooldownGate                                     (v0.7)
    time_window.py           TimeWindowGate + Window                          (v0.7)
    smoothing.py             ActionSmoothingGate + smooth_target              (v0.9)
    safe_projection.py       project_to_workspace + project_to_sdf            (v0.9)
    retry.py                 RetryPolicy + transient-error helpers
    llm.py                   LLMPolicy + LLMPolicyConfig + llm_policy_loop
    vla.py                   VLAPolicy + DeltaXYZDecoder

  primitives/
    motion.py                move_to / scan
    manipulation.py          pick / place
    trajectory.py            follow_trajectory + linear_interpolate
    composite.py             composite_primitive factory                     (v0.7)
    morphology.py            MorphologyRegistry — cross-embodiment           (v0.10)
    library.py               cross-morphology primitive catalogue —          (v1.0)
                             mobile_base / quadruped / humanoid / aerial /
                             dexterous / sensing / generic

  profiles/                                                                  (v1.0)
    core.py                  RobotProfile + YAML loader + runtime builder
    presets.py               franka_arm / turtlebot / spot / tello /
                             stretch / humanoid_demo

  backends/
    mujoco.py                MuJoCoBackend                                   (v0.2)
    pybullet.py              PyBulletBackend                                 (v0.3)
    gymnasium.py             GymnasiumBackend (Farama Gym ecosystem)         (v0.7)
    ros2.py                  ROS2Backend (rclpy adapter)                     (v0.9)
    randomized.py            RandomizedBackend (sim2real wrapper)            (v0.8)
    menagerie.py             MuJoCo Menagerie auto-clone                     (v0.4)

  bench/
    episode.py               Episode + EpisodeRunner + EpisodeResult         (v0.2)
    report.py                RunReport + wilson_ci + summarize               (v0.2)
    compare.py               paired_compare + mcnemar_p + cohens_h            (v0.2)
    catalogue.py             preset_reach_8 + preset_pick_and_place_4 + …    (v0.4)
    reward_shaper.py         RewardShaper + OnPrimitive / OnDecision / …     (v0.9)
    sim2real.py              Sim2RealBench + Sim2RealReport                   (v0.9)
    adversarial.py           random_seeds / grid_seeds / cma_es_seeds        (v0.10)

  properties/
    core.py                  Property + PropertyEngine + Severity            (v0.5)
    builtins.py              StaysInsideWorkspace / NeverHoldsTwoObjects/…   (v0.5)
    combinators.py           AndProperty / OrProperty / NotProperty          (v0.6)
    temporal.py              Always / Eventually / Until (STL)               (v0.8)
    mining.py                mine_properties + MinedProperty                 (v0.10)

  judges/
    llm_judge.py             LLMJudge + LLMJudgeConfig + parse_judgement     (v0.10)
    heuristic.py             HeuristicJudge + rule predicates                 (v0.10)

  skills/
    graph.py                 SkillGraph + Skill + topological order           (v0.10)

  missions/
    core.py                  Mission + Step + MissionRunner + MissionResult   (v0.7)

  fleet/
    core.py                  RobotHandle + FleetRegistry + FleetDispatcher    (v0.6)

  dashboard/
    api.py                   FastAPI factory + healthz + store endpoints      (v0.6)
    streaming.py             StreamManager + WebSocket /ws/v1/stream          (v0.7)

  planning/
    core.py                  TaskPlanner + TaskStep                          (v0.5)
    builtin.py               sequential_planner / fixed_plan                  (v0.5)
    llm_planner.py           LLMPlanner (single-shot full-plan emission)      (v0.6)

  sensors/
    camera.py                Camera Protocol + MockCamera + capture_camera   (v0.5)

  telemetry/
    otel.py                  step_span + record_decision + record_result    (v0.5)
    energy.py                EnergyLedger + PrimitiveEnergyModel             (v0.10)

  training/
    core.py                  SafeRolloutCollector + LagrangianMultiplier     (v0.8)
    hindsight.py             HER relabeling + sparse_indicator_reward        (v0.10)

  traces/
    replay.py                load_trace + iter_events + summarize_trace      (v0.4)
    diff.py                  diff_traces + StepDiff + TraceDiff              (v0.6)
    query.py                 query DSL with comparison ops + and/or/not/in   (v0.8)

examples/
  pick_and_place.py                    scripted end-to-end demo
  bench_with_without_geofence.py       paired-comparison demo
  mcp_robot.py                         general MCP server — picks profile   (v1.0)
                                       via GHOSTLOOP_PROFILE; works with
                                       arms, mobile bases, quadrupeds,
                                       drones, humanoids, custom robots
  claude_desktop_mcp_arm.py            arm-specific MCP example (legacy)    (v1.0)
  claude_desktop_config.json           cross-client + cross-OS config       (v1.0)
                                       reference (Claude Desktop / Cursor /
                                       Continue / Cline / Zed / Gemini CLI)
  custom_robot.yaml                    sample profile YAML —                (v1.0)
                                       hospital medication-delivery robot
                                       with custom primitives + composites
  custom_robot_primitives.py           sample custom Primitive factories    (v1.0)
                                       (dispense_pill, alert_nurse)
  direct_llm_arm.py                    direct OpenAI-compatible function    (v1.0)
                                       calling — works with OpenAI /
                                       Anthropic / Gemini / Groq / Ollama
                                       / vLLM / GhostLM

tests/                                  333 tests (8 live-gated)
  test_core.py                          23
  test_llm_policy.py                    14
  test_bench.py                         22
  test_mujoco_backend.py                10
  test_v03_additions.py                 25
  test_v04_additions.py                 21
  test_v05_additions.py                 32
  test_v06_additions.py                 37
  test_v07_additions.py                 29
  test_v08_additions.py                 28
  test_v09_additions.py                 25
  test_v10_additions.py                 33
  test_v10_v1_additions.py              18
  test_profiles.py                      19

assets/                                  brand mark + wordmark variants
docs/                                    architecture / migration / brand notes
```

## Why this is novel

There are robot frameworks. There are agent frameworks. There is no robot framework that **treats robots as a model with a tool registry, a fail-closed safety pipeline, a structured trace log, statistical bench rigor, AND a layer of post-hoc analysis** (counterfactual replay, causal attribution, LLM-as-judge, property mining, adversarial fuzzing) — the same shape that's now standard for LLM-driven cybersec agents (`secure-mcp`, `ghostguard`, `GhostAgent`).

The thesis: as VLA models become the policy substrate, the runtime around them needs the same rigor we already apply to LLM tool use, plus the analytical tooling — counterfactuals, causal attribution, judge models — that LLM safety has been building for years. ghostloop is that runtime.

## License

MIT. See [LICENSE](LICENSE).

---

## The ghostloop family

- **[ghostloop-ui](https://github.com/joemunene-by/ghostloop-ui)**: Next.js 15 dashboard talking to `create_production_app`. Fleet view, alarm tray, episode list, Prometheus metrics, profile-aware gamepad mapper (drone / mobile base / quadruped / arm / humanoid), `/connect` onboarding for non-coders. Public deploy at [ghostloop-ui.vercel.app](https://ghostloop-ui.vercel.app), backend on Render. No Python on the operator's machine.
- **[ghostloop-desktop](https://github.com/joemunene-by/ghostloop-desktop)** v0.2: Tauri 2 native app wrapping ghostloop-ui. Voice control via the embedded WebView's Web Speech API (Windows + Linux today, native whisper.cpp in v0.3 for macOS), gamepad rumble on safety events, native OS notifications for alarms, sidecar Python runtime, native gamepad via `gilrs` (wired and Bluetooth), system tray, e-stop hotkey. Single-file bundles for macOS / Windows / Linux on every tagged release via GitHub Actions.
- **[Live demo on HuggingFace](https://huggingface.co/spaces/Ghostgim/ghostloop-demo)**: Gradio control panel, no install. Per-primitive dispatch buttons, virtual joystick, pause / resume / e-stop, live trace pane.

Built by [Joe Munene](https://github.com/joemunene-by) at [Complex Developers](https://github.com/complexdevelopers). Sibling to [GhostLM](https://github.com/joemunene-by/GhostLM), [secure-mcp](https://github.com/joemunene-by/secure-mcp), [ghostguard](https://github.com/joemunene-by/ghostguard), [CyberBench](https://github.com/joemunene-by/cyberbench).
