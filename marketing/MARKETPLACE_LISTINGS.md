# Marketplace + community listings

The MCP server + Python package belong in several public registries.
Each gets discovered by a different audience; the listings are not
duplicates of each other — they're distribution channels.

## 1. PyPI (Python users)

Status: ready, see `release/RELEASE_TO_PYPI.md`.

URL when live: `https://pypi.org/project/ghostloop/`.

## 2. modelcontextprotocol.io community list

The official list of MCP servers. Mostly maintained on GitHub:
https://github.com/modelcontextprotocol/servers and
https://modelcontextprotocol.io/clients

**To submit:** open a PR against the README of the official servers
repo with this entry under "Community Servers":

```markdown
- **[ghostloop](https://github.com/joemunene-by/ghostloop)** — drive
  any robot (arm, mobile base, quadruped, drone, humanoid, or your
  own custom robot) through a fail-closed safety pipeline. Six
  backends including MuJoCo + PyBullet + Gymnasium + ROS 2.
  Stdio + streamable-http transports. `pip install ghostloop[mcp]`.
```

## 3. Smithery.ai

`https://smithery.ai/` aggregates MCP servers with a one-click install
into Claude Desktop / Cursor. Submission goes through a GitHub
manifest at the top of the repo or via their dashboard.

**Manifest file** (drop at `smithery.yaml` in the repo root):

```yaml
name: ghostloop
displayName: ghostloop — robot agent runtime
description: |
  Drive any robot from any MCP-aware assistant. Tool-using runtime,
  fail-closed safety pipeline (geofence + force cap + action smoothing
  + rate limit + HITL), six backends (Mock / MuJoCo / PyBullet /
  Gymnasium / ROS 2 / Randomized), and post-hoc analysis (counterfactual
  replay, causal attribution, LLM-as-judge, property mining).
homepage: https://github.com/joemunene-by/ghostloop
license: MIT
author: Joe Munene
tags: [robotics, embodied-ai, agent, safety, mcp, mujoco, ros2]

stdio:
  command: python3
  args: ["-m", "ghostloop.mcp_server"]
  env:
    GHOSTLOOP_PROFILE: "franka_arm"
    GHOSTLOOP_BACKEND: "mock"

http:
  url: "{your-deployment-url}/mcp"
```

(needs a small `python -m ghostloop.mcp_server` entry point — not yet
shipped; this is one TODO before submission.)

## 4. Cursor / Continue / Cline / Zed / Gemini CLI

These are NOT separate listings — every MCP-aware client reads the
same MCP server. Once you're in `modelcontextprotocol.io` or
Smithery, those clients all benefit.

What you CAN do: add a "Quick install" button per client in the README
that pre-fills the right config block. Cursor in particular has a
deep-link install URL pattern:

```
cursor://anysphere.cursor-mcp/install?name=ghostloop&command=python3&args=...
```

Document this pattern in the repo's README under the cross-client config table.

## 5. ROS Index (ROS users)

`https://index.ros.org/` — the package index for ROS 2. Submit by
opening a PR against `https://github.com/ros/rosdistro` with a
`rosdep` entry pointing at the PyPI package. Visibility tier: every
ROS 2 user discovering the package via `apt search` / `rosdep`.

## 6. Awesome lists on GitHub

Get added to:

- `awesome-mcp-servers` — https://github.com/punkpeye/awesome-mcp-servers
- `awesome-robotics-libraries` — https://github.com/jslee02/awesome-robotics-libraries
- `awesome-llm-tooluse` — search; if none exists, start one
- `awesome-embodied-ai` — search; multiple exist

Each is a one-line PR with the repo description. Free distribution.

## 7. Reddit + HN posts (timing matters)

Save these for the v1.0.0 launch week:

- `r/MachineLearning` — Show post tagged `[P]` (project)
- `r/robotics` — flagged "Open Source"
- `r/LocalLLaMA` — post angle: "use any local LLM to control a robot via MCP"
- `r/Python` — angle: "I built a Python lib for safe LLM-robot control"
- HN — "Show HN: ghostloop — drive any robot through a fail-closed safety pipeline from Claude Desktop"

Stagger across 3 days, one platform per day. Don't spam — the demo
video is the hook, the repo + PyPI + HF Space are the destinations.

## 8. Anthropic / OpenAI / Google ecosystem listings

- Anthropic's MCP examples gallery — there's no formal submission yet
  but the team links example servers from the docs. Worth a tweet
  at `@AnthropicAI` once the demo video exists.
- OpenAI Assistants playground — function-calling demo via
  `examples/direct_llm_arm.py`. Submit as a community example if they
  reopen submissions.

## Submission order (priority)

1. PyPI (gates everything else — needs to exist before Smithery /
   awesome-list / video reference work).
2. HuggingFace Space (live demo URL referenced everywhere).
3. modelcontextprotocol.io community list PR.
4. Smithery manifest + submission.
5. Awesome-list PRs (5 minutes each).
6. Demo video → X/LinkedIn/YouTube.
7. Reddit + HN posts (week after video lands so traffic compounds).

The whole list is ~4 hours of work spread over a launch week, not all
at once. Each listing pulls a different audience — PyPI gets ML
researchers; Smithery gets MCP power users; awesome-lists get the
GitHub-curious; Reddit gets the broad crowd; HN gets the technical
deep-dive crowd.
