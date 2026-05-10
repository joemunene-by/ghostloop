"""HuggingFace Space — interactive ghostloop demo.

Live URL: https://huggingface.co/spaces/Ghostgim/ghostloop-demo (after deploy).

Lets a visitor:
  1. Pick a robot profile (franka_arm / spot / tello / humanoid_demo).
  2. See the safety pipeline (gates active for that profile).
  3. Send an Intent and watch the runtime dispatch + record the trace.
  4. Replay the trace as JSON.
  5. Read each profile's instructions block to understand what the LLM
     "knows" about that robot.

Why a Space: GitHub repos take 30 seconds to clone + install. A Space
loads in 5 seconds, no install. That's where most reputation conversions
happen — the moment somebody clicks a link and sees Claude reasoning
about a robot in the browser.

Stack:
  gradio ≥ 4.0 (UI)
  ghostloop ≥ 1.0 (the library)

Zero hardware deps — runs against MockBackend so the Space starts
instantly on the free CPU tier.
"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from ghostloop import Intent
from ghostloop.profiles import (
    build_runtime_from_profile,
    franka_arm,
    humanoid_demo,
    spot_quadruped,
    stretch_mobile_arm,
    tello_drone,
    turtlebot_base,
)


PRESETS = {
    "franka_arm — 7-DOF arm":              franka_arm,
    "spot — Boston Dynamics quadruped":    spot_quadruped,
    "tello — quadcopter drone":            tello_drone,
    "stretch — mobile arm":                stretch_mobile_arm,
    "humanoid_demo — stationary humanoid": humanoid_demo,
    "turtlebot — wheeled mobile base":     turtlebot_base,
}


def load_profile(profile_label: str) -> tuple[str, str, str, str]:
    """Build a runtime + return its description / primitives / gates / instructions."""
    factory = PRESETS[profile_label]
    profile = factory()
    runtime = build_runtime_from_profile(profile)
    primitives_md = "\n".join(
        f"- **`{name}`** — {runtime.registry.get(name).description}"
        for name in runtime.registry.names()
    )
    gates_md = "\n".join(
        f"- {g.__class__.__name__}" for g in runtime.policy_pipeline.gates
    )
    summary = (
        f"**Profile:** `{profile.name}`  \n"
        f"**Morphology:** `{profile.morphology}`  \n"
        f"**Backend:** `{runtime.backend.name}` (mock)  \n"
        f"**Workspace:** `{profile.workspace_bounds}`  \n"
        f"**Max velocity:** `{profile.max_velocity}` m/s  \n"
        f"**HITL primitives:** `{profile.hitl_primitives}`"
    )
    return summary, primitives_md, gates_md, profile.instructions or "(no instructions block)"


def step_runtime(profile_label: str, primitive_name: str, args_json: str) -> tuple[str, str]:
    """Dispatch one Intent against a fresh runtime, return result + last trace event."""
    factory = PRESETS[profile_label]
    runtime = build_runtime_from_profile(factory())
    try:
        args = json.loads(args_json) if args_json.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("args must be a JSON object")
    except (ValueError, json.JSONDecodeError) as e:
        return f"❌ args JSON parse error: {e}", "—"
    if primitive_name not in runtime.registry.names():
        return (
            f"❌ Unknown primitive {primitive_name!r}. Available for "
            f"this profile: {runtime.registry.names()}",
            "—",
        )
    intent = Intent(name=primitive_name, args=args)
    result = runtime.step(intent)
    event = runtime.trace.events[-1]
    decision = event.decision
    status_icon = {"ok": "✅", "blocked": "🚫", "error": "❌"}.get(
        result.status.value, "⚠️"
    )
    summary = (
        f"{status_icon} **status:** `{result.status.value}`  \n"
        f"**decision:** `{decision.action.value}`"
        f" (gate: `{decision.gate_name}`)  \n"
        f"**reason:** {decision.reason}  \n"
        f"**message:** {result.message}"
    )
    trace_json = json.dumps(event.to_json(), indent=2)
    return summary, trace_json


def example_args(primitive_name: str) -> str:
    """Convenience: a sensible default args dict for common primitives."""
    examples = {
        "move_to":     '{"x": 0.4, "y": 0.0, "z": 0.5}',
        "pick":        '{"object_id": "widget-7"}',
        "place":       '{}',
        "scan":        '{"radius": 0.3}',
        "drive":       '{"linear_x": 0.2, "angular_z": 0.0}',
        "stop":        '{}',
        "goto":        '{"x": 1.5, "y": -0.5, "theta": 0.0}',
        "rotate":      '{"dtheta": 1.57}',
        "sit":         '{}',
        "stand":       '{}',
        "lie_down":    '{}',
        "walk_to":     '{"x": 2.0, "y": 1.0, "theta": 0.0}',
        "wave":        '{"hand": "right"}',
        "look_at":     '{"x": 1.0, "y": 0.0, "z": 1.5}',
        "point_at":    '{"x": 1.0, "y": 0.0, "z": 1.5}',
        "nod":         '{"direction": "yes"}',
        "takeoff":     '{"altitude": 1.0}',
        "land":        '{}',
        "fly_to":      '{"x": 1.0, "y": 0.0, "z": 1.5, "yaw": 0.0}',
        "hover":       '{"seconds": 2.0}',
        "set_joint":   '{"joint_name": "shoulder", "angle": 0.5, "duration": 1.0}',
        "set_gripper": '{"state": "open", "force": 0.0}',
        "sense":       '{"modality": "rgb"}',
        "scan_360":    '{}',
        "take_photo":  '{}',
        "read_battery": '{}',
        "wait":        '{"seconds": 1.0}',
        "emit_event":  '{"kind": "note", "message": "demo event"}',
    }
    return examples.get(primitive_name, "{}")


with gr.Blocks(
    title="ghostloop — the agent loop, embodied",
    theme=gr.themes.Soft(primary_hue="teal"),
) as demo:
    gr.Markdown(
        """
# ghostloop · live demo

The agent loop, embodied. A tool-using runtime + fail-closed safety
pipeline + sim-first execution + post-hoc analysis layer for embodied AI.

This Space runs against `MockBackend` so it starts instantly. Every
primitive call goes through the **same fail-closed safety pipeline** that
ships in the library (geofence + force cap + action smoothing + rate
limit + HITL) — try sending a `move_to` outside the workspace and watch
the geofence reject it.

[GitHub](https://github.com/joemunene-by/ghostloop) · [PyPI](https://pypi.org/project/ghostloop/) · [arXiv preprint](#) · [Sister project: GhostLM](https://github.com/joemunene-by/GhostLM)
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            profile_dd = gr.Dropdown(
                label="Robot profile",
                choices=list(PRESETS.keys()),
                value="franka_arm — 7-DOF arm",
            )
            summary_md = gr.Markdown(label="Profile summary")
            instructions_md = gr.Markdown(label="LLM instructions")

        with gr.Column(scale=1):
            primitives_md = gr.Markdown(label="Available primitives")
            gates_md = gr.Markdown(label="Active safety gates")

    profile_dd.change(
        load_profile,
        inputs=[profile_dd],
        outputs=[summary_md, primitives_md, gates_md, instructions_md],
    )

    gr.Markdown("## Dispatch a primitive")

    with gr.Row():
        primitive_in = gr.Textbox(
            label="Primitive name",
            value="move_to",
            placeholder="e.g. move_to / drive / takeoff / wave",
        )
        args_in = gr.Textbox(
            label="Args (JSON)",
            value='{"x": 0.4, "y": 0.0, "z": 0.5}',
            lines=2,
        )
        primitive_in.change(
            example_args, inputs=[primitive_in], outputs=[args_in],
        )

    run_btn = gr.Button("▶ runtime.step(intent)", variant="primary")
    result_md = gr.Markdown(label="Result")
    trace_json = gr.Code(label="Trace event (JSON)", language="json")
    run_btn.click(
        step_runtime,
        inputs=[profile_dd, primitive_in, args_in],
        outputs=[result_md, trace_json],
    )

    gr.Markdown(
        """
### Try these:

1. **Geofence violation:** keep `franka_arm` profile, send
   `move_to` with `{"x": 5.0, "y": 0.0, "z": 0.5}`. The Geofence gate
   blocks it; the trace records exactly why.
2. **HITL escalation:** switch to `tello`, send `takeoff` with
   `{"altitude": 1.0}`. The HumanInTheLoopGate is wired to `takeoff`
   for this profile — in a real terminal it'd prompt for approval; in
   the Space it returns an "approver declined" since there's no human.
3. **Cross-morphology:** switch to `spot`, send `walk_to` with
   `{"x": 2.0, "y": 1.0, "theta": 0.0}`. The same Runtime, the same
   safety pipeline pattern, completely different primitive set.

Every call records a typed `TraceEvent` you can replay, diff, query
with the trace DSL, score with the LLM judge, or attribute causally
when something fails.
"""
    )

    # Initial population.
    demo.load(
        load_profile,
        inputs=[profile_dd],
        outputs=[summary_md, primitives_md, gates_md, instructions_md],
    )


if __name__ == "__main__":
    demo.launch()
