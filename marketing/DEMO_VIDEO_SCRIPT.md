# Demo video script — Claude controls a robot through ghostloop

**Length target:** 60 seconds. Vertical (9:16) for Twitter/X + LinkedIn,
or horizontal (16:9) if going on YouTube as a landing-page hero.

**Pitch in one line:** _"Claude Desktop can now drive a robot arm. Through ghostloop, every command goes through a fail-closed safety pipeline first."_

---

## Storyboard (5 cuts)

### Cut 1 (0:00 – 0:08) — Hook

**Visual:** Split-screen.
  - Left: terminal showing `python3 examples/mcp_robot.py --selfcheck`.
  - Right: Claude Desktop sidebar with "ghostloop" MCP server connected, tools listed (`move_to`, `set_gripper`, `sense`, ...).

**Voiceover:**
> "ghostloop. Claude Desktop, talking to a robot through the model context protocol."

**Caption (sticker):** `pip install ghostloop`

### Cut 2 (0:08 – 0:20) — The good prompt

**Visual:** Claude Desktop conversation panel. User types:

> *"Move the arm to (0.4, 0, 0.5), scan with radius 0.3, then move to (0.6, 0.2, 0.5)."*

Claude calls the tools in sequence. MuJoCo viewer in the background shows the Franka arm executing. Trace events stream in a side-panel terminal.

**Voiceover:**
> "Send a goal in natural language. The runtime maps Claude's tool calls to robot primitives. Every step is captured in a structured trace."

**Caption:** `Intent → Primitive → Pipeline → Backend → Trace`

### Cut 3 (0:20 – 0:35) — The bad prompt (the actual hook)

**Visual:** New conversation. User types:

> *"Move to (5, 0, 0.5)."* (deliberately outside the workspace)

Claude calls `move_to`. The MuJoCo arm DOES NOT MOVE. The Claude response shows the structured deny:

```
status: blocked
gate: geofence
reason: target x=5 outside workspace [-0.6, 0.6]
```

**Voiceover:**
> "Now ask for something unsafe. The safety pipeline rejects it BEFORE the actuator sees it. Claude reads the structured rejection and adjusts."

**Caption:** `fail-closed: any deny short-circuits the pipeline`

### Cut 4 (0:35 – 0:50) — The "wait, what else?" reveal

**Visual:** Quick montage (each card on screen for 2-3 seconds):

  - "12 policy gates: geofence, force cap, action smoothing, time window, cooldown, deny list, rate limit, HITL, ..."
  - "6 backends: Mock, MuJoCo, PyBullet, Gymnasium, ROS 2, RandomizedBackend"
  - "11 releases, 333 tests, MIT licensed"
  - "Counterfactual replay. Causal attribution. LLM-as-judge. Property mining."
  - "Drives anything: arms, mobile bases, quadrupeds, drones, humanoids, your custom robot"

**Voiceover:**
> "Twelve safety gates. Six backends. Production fleet dashboard. Sim-to-real bench. Counterfactual replay. Causal failure attribution. The whole layer that's been missing between LLM agents and robots — open source."

### Cut 5 (0:50 – 1:00) — CTA

**Visual:** GitHub repo card on screen. Highlight star count, commit graph, the v1.0.0 release tag.

URL on screen: `github.com/joemunene-by/ghostloop`

Below it: `huggingface.co/spaces/Ghostgim/ghostloop-demo` (live demo, no install).

**Voiceover:**
> "ghostloop. The agent loop, embodied. Link in bio."

**End card:** logo + URL.

---

## Recording cheat-sheet

### Setup before hitting record

```bash
# Terminal 1 — where you'll do the selfcheck.
cd ~/path/to/ghostloop
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mcp,mujoco]"

# Terminal 2 — already open, ready for `--selfcheck`.

# Terminal 3 — already open, will show trace events.
tail -F /tmp/ghostloop_trace.jsonl

# Claude Desktop — config already wired:
#   ~/Library/Application Support/Claude/claude_desktop_config.json
# pointing at examples/mcp_robot.py with GHOSTLOOP_BACKEND=mujoco.

# MuJoCo viewer — open with the Franka model loaded so Claude's calls
# visibly move the arm.

# Recording tools:
#   QuickTime / OBS / Cleanshot for screen.
#   Terminal at 18-20pt for legibility on mobile screens.
#   Increase Claude Desktop font size 20% for the same reason.
```

### The two prompts to pre-write

```
Prompt 1 (good): "Move the arm to (0.4, 0, 0.5), scan with radius 0.3, then move to (0.6, 0.2, 0.5)."
Prompt 2 (bad):  "Move to (5, 0, 0.5)."
```

Don't improvise — pre-paste both into a TextEdit window and copy-paste them into Claude Desktop on cue. That keeps the timing tight.

### Edit pass

- 30 fps minimum, 60 fps if you can.
- Keep the audio under -3 dBFS (avoid clipping); duck the music to ~-22 dBFS under voiceover.
- Add captions for every voiceover line (most viewers watch on mute).
- Export at 1080p horizontal AND 1080x1920 vertical from the same edit if your editor supports cropping.

### Posting

| Platform | Format | Caption |
|---|---|---|
| **X / Twitter** | 9:16 vertical | "Claude Desktop driving a robot through @AnthropicAI's MCP. Every command is gated by a fail-closed safety pipeline I built (geofence, force cap, HITL). Open source: github.com/joemunene-by/ghostloop" |
| **LinkedIn** | 16:9 horizontal | Longer, lead with the problem ("robotics has ROS 2 and VLA models, but no agent runtime in between"), end with the repo + a "what would you build with this?" |
| **YouTube** | 16:9 horizontal | This becomes the landing-page hero at the top of the README. Pin a comment with the install command. |
| **HN Show** | text post | "Show HN: ghostloop — drive any robot through a fail-closed safety pipeline from Claude Desktop". Link the video as the demo. |

### Tag list (don't @ everyone, just the ones who'd care)

- `@AnthropicAI` (their MCP, their model — high signal)
- `@deepmind` (MuJoCo)
- `@MuJoCoSim`
- `@farama_org` (Gymnasium)
- `@cursor_ai`, `@continuedev`, `@_zeddev` (other MCP clients)

Avoid the spray-and-pray tag pattern. Three tags is the right count.
