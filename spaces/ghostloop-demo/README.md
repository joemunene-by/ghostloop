---
title: ghostloop demo
emoji: 🤖
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 5.12.0
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: Drive any robot through a fail-closed safety pipeline.
tags:
  - robotics
  - embodied-ai
  - agent
  - mcp
  - safety
  - mujoco
  - ros2
---

# ghostloop · live demo

The agent loop, embodied. Tool-using runtime + fail-closed safety
pipeline + sim-first execution + post-hoc analysis layer for
embodied AI / robotics. Sister project to
[GhostLM](https://github.com/joemunene-by/GhostLM).

This Space lets you pick a robot profile (Franka arm / Spot quadruped /
Tello drone / Stretch mobile arm / humanoid / TurtleBot) and dispatch
Intents through the safety pipeline that ships in the library.

- **GitHub:** https://github.com/joemunene-by/ghostloop
- **PyPI:** `pip install ghostloop`
- **arXiv:** _[link to be added once preprint is up]_

## What you can try here

1. Switch profiles to see how the same Runtime + safety pipeline shape
   covers totally different morphologies.
2. Send `move_to {"x": 5, "y": 0, "z": 0}` on the Franka profile to
   watch the GeofenceGate reject it with a structured reason.
3. Send `takeoff {"altitude": 1}` on the Tello profile to see the HITL
   gate escalate.
4. Read the trace event JSON for any call. That's the same shape the
   library emits for replay, diff, query, energy ledger, and judge
   scoring.

## Beyond the demo

The full library does much more than this Space exposes:

- 6 backends (Mock / MuJoCo / PyBullet / Gymnasium / ROS 2 / Randomized).
- 12 policy gates including STL temporal properties.
- Counterfactual trace replay, causal failure attribution, LLM-as-judge.
- VLA-on-MuJoCo benchmark harness vs OpenVLA / π0 / RT-2 / Octo numbers.
- Safe-RL training loop with Lagrangian multiplier + HER.
- Production fleet dashboard with auth + rate limit + alarms + Prometheus.
- MCP server for Claude Desktop / Cursor / Continue / Cline / Zed / Gemini CLI.

`pip install ghostloop` and clone the repo for the full kit.
