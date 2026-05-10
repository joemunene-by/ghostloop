"""HuggingFace Space — ghostloop control panel.

Live URL: https://huggingface.co/spaces/Ghostgim/ghostloop-demo

A no-code interface to ghostloop. Visitors pick a robot profile, then
drive it via:

  - Per-primitive dispatch buttons (auto-generated from the profile's
    registry — no JSON typing required).
  - Virtual joystick (D-pad style) for mobile bases / quadrupeds /
    drones.
  - Free-form Intent dispatch (advanced — JSON args).
  - Live intervention controls: Pause, Resume, Emergency Stop.
  - Live trace pane that updates after every dispatch.
  - Live state pane showing the backend snapshot.

The runtime is held per-session so the trace accumulates across
button clicks. Switch profiles to reset.

Backend is MockBackend so the Space runs on the free CPU tier without
any sim install.
"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from ghostloop import (
    Intent,
    InterventionGate,
    InterventionState,
    LivePolicyController,
    PolicyPipeline,
    Runtime,
)
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

DEFAULT_PROFILE = "franka_arm — 7-DOF arm"


# ---------------------------------------------------------------------------
# Per-session runtime state (persisted via gr.State).
# ---------------------------------------------------------------------------


def make_session(profile_label: str) -> dict[str, Any]:
    """Construct a fresh runtime session for a profile selection."""
    factory = PRESETS[profile_label]
    profile = factory()
    runtime = build_runtime_from_profile(profile)
    controller = LivePolicyController(
        policy=lambda state: Intent("emit_event", {"kind": "noop"}),
        fallback_policy=lambda state: Intent(
            "stop" if "stop" in runtime.registry.names() else "emit_event",
            {"kind": "fallback"},
        ),
    )
    # Add the InterventionGate to the front of the pipeline so pause
    # affects future dispatches.
    runtime.policy_pipeline = PolicyPipeline(
        gates=[InterventionGate(controller=controller), *runtime.policy_pipeline.gates]
    )
    return {
        "profile_label": profile_label,
        "profile": profile,
        "runtime": runtime,
        "controller": controller,
    }


# ---------------------------------------------------------------------------
# Default-args lookup so primitive buttons launch with sensible values.
# ---------------------------------------------------------------------------


DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    # Arm primitives.
    "move_to": {"x": 0.4, "y": 0.0, "z": 0.5},
    "scan": {"radius": 0.3},
    "pick": {"object_id": "widget-7"},
    "place": {},
    "set_joint": {"joint_name": "shoulder", "angle": 0.5, "duration": 1.0},
    "set_gripper": {"state": "open", "force": 0.0},
    # Mobile base.
    "drive": {"linear_x": 0.2, "angular_z": 0.0},
    "stop": {},
    "goto": {"x": 1.0, "y": 0.0, "theta": 0.0},
    "rotate": {"dtheta": 1.57},
    # Quadruped.
    "sit": {},
    "stand": {},
    "lie_down": {},
    "walk_to": {"x": 1.5, "y": 0.0, "theta": 0.0},
    # Humanoid.
    "wave": {"hand": "right"},
    "look_at": {"x": 1.0, "y": 0.0, "z": 1.5},
    "point_at": {"x": 1.0, "y": 0.0, "z": 1.5},
    "nod": {"direction": "yes"},
    # Aerial.
    "takeoff": {"altitude": 1.0},
    "land": {},
    "fly_to": {"x": 1.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
    "hover": {"seconds": 2.0},
    # Sensing.
    "sense": {"modality": "rgb"},
    "scan_360": {},
    "take_photo": {},
    "read_battery": {},
    "wait": {"seconds": 1.0},
    "emit_event": {"kind": "note", "message": "demo event"},
}


# ---------------------------------------------------------------------------
# Render helpers — keep all formatting in one place.
# ---------------------------------------------------------------------------


def _render_profile_summary(session: dict[str, Any]) -> str:
    profile = session["profile"]
    runtime = session["runtime"]
    return (
        f"### `{profile.name}` — `{profile.morphology}`\n\n"
        f"**Backend:** `{runtime.backend.name}` (mock) · "
        f"**Workspace:** `{profile.workspace_bounds}` · "
        f"**Max velocity:** `{profile.max_velocity}` m/s · "
        f"**HITL primitives:** `{profile.hitl_primitives}`"
    )


def _render_primitives_list(session: dict[str, Any]) -> str:
    runtime = session["runtime"]
    return "\n".join(
        f"- **`{name}`** — {runtime.registry.get(name).description}"
        for name in runtime.registry.names()
    )


def _render_gates_list(session: dict[str, Any]) -> str:
    runtime = session["runtime"]
    return "\n".join(
        f"- {g.__class__.__name__}" for g in runtime.policy_pipeline.gates
    )


def _render_state(session: dict[str, Any]) -> str:
    runtime = session["runtime"]
    return f"```json\n{json.dumps(runtime.backend.snapshot(), indent=2)}\n```"


def _render_trace(session: dict[str, Any]) -> str:
    runtime = session["runtime"]
    if not runtime.trace.events:
        return "_(no events yet — dispatch a primitive to see the trace)_"
    rows = ["| step | intent | decision | gate | result | reason |",
            "|---:|---|:---:|---|:---:|---|"]
    for ev in runtime.trace.events[-12:]:
        decision_icon = {"allow": "✅", "deny": "🚫", "escalate": "⚠️"}.get(
            ev.decision.action.value, "?"
        )
        result_icon = {"ok": "✅", "blocked": "🚫", "error": "❌"}.get(
            ev.result.status.value, "?"
        )
        args_compact = json.dumps(ev.intent.args, separators=(",", ":"))
        if len(args_compact) > 40:
            args_compact = args_compact[:37] + "…"
        reason = (ev.decision.reason or ev.result.message or "")[:80]
        rows.append(
            f"| {ev.step} | `{ev.intent.name}` {args_compact} | "
            f"{decision_icon} | `{ev.decision.gate_name or ''}` | "
            f"{result_icon} | {reason} |"
        )
    n = len(runtime.trace.events)
    if n > 12:
        rows.append(f"\n_showing last 12 of {n} events_")
    return "\n".join(rows)


def _render_intervention_state(session: dict[str, Any]) -> str:
    state = session["controller"].state
    icon = {
        InterventionState.RUNNING:        "🟢",
        InterventionState.PAUSED:         "⏸️",
        InterventionState.SWAPPING:       "🔄",
        InterventionState.EMERGENCY_STOP: "🛑",
    }.get(state, "?")
    return f"### {icon} Intervention: `{state.value}`"


def _render_instructions(session: dict[str, Any]) -> str:
    return session["profile"].instructions or "(no instructions block)"


def _all_outputs(session: dict[str, Any]) -> tuple:
    return (
        _render_profile_summary(session),
        _render_primitives_list(session),
        _render_gates_list(session),
        _render_instructions(session),
        _render_state(session),
        _render_trace(session),
        _render_intervention_state(session),
    )


# ---------------------------------------------------------------------------
# Action handlers.
# ---------------------------------------------------------------------------


def select_profile(profile_label: str):
    session = make_session(profile_label)
    names = session["runtime"].registry.names()
    # Up to 12 primitive buttons; pad with empty updates for the rest.
    btn_updates = []
    for i in range(12):
        if i < len(names):
            btn_updates.append(gr.update(value=names[i], visible=True, interactive=True))
        else:
            btn_updates.append(gr.update(visible=False))
    return (session, *_all_outputs(session), *btn_updates)


def dispatch_primitive(session: dict[str, Any], primitive_name: str):
    if not primitive_name:
        return session, *_all_outputs(session)
    args = dict(DEFAULT_ARGS.get(primitive_name, {}))
    session["runtime"].step(Intent(primitive_name, args))
    return session, *_all_outputs(session)


def dispatch_custom(session: dict[str, Any], primitive_name: str, args_json: str):
    if not primitive_name.strip():
        return session, *_all_outputs(session)
    try:
        args = json.loads(args_json) if args_json.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("args must be a JSON object")
    except (ValueError, json.JSONDecodeError):
        return session, *_all_outputs(session)
    if primitive_name not in session["runtime"].registry.names():
        return session, *_all_outputs(session)
    session["runtime"].step(Intent(primitive_name, args))
    return session, *_all_outputs(session)


def dispatch_drive(session: dict[str, Any], linear_x: float, angular_z: float):
    """Joystick handler — emits drive(linear_x, angular_z) for mobile / quad."""
    runtime = session["runtime"]
    name = None
    if "drive" in runtime.registry.names():
        name = "drive"
        args = {"linear_x": float(linear_x), "angular_z": float(angular_z)}
    elif "fly_to" in runtime.registry.names():
        # Drone: interpret as relative flight.
        name = "fly_to"
        args = {"x": float(linear_x), "y": 0.0, "z": 1.0, "yaw": float(angular_z)}
    elif "walk_to" in runtime.registry.names():
        name = "walk_to"
        args = {"x": float(linear_x), "y": 0.0, "theta": float(angular_z)}
    elif "move_to" in runtime.registry.names():
        # Arm: interpret as a delta on x.
        name = "move_to"
        args = {"x": float(linear_x), "y": 0.0, "z": 0.5}
    if name is None:
        return session, *_all_outputs(session)
    runtime.step(Intent(name, args))
    return session, *_all_outputs(session)


def pause_runtime(session: dict[str, Any]):
    session["controller"].pause(operator="ui_visitor", reason="UI pause button")
    return session, *_all_outputs(session)


def resume_runtime(session: dict[str, Any]):
    session["controller"].resume(operator="ui_visitor", reason="UI resume button")
    return session, *_all_outputs(session)


def emergency_stop(session: dict[str, Any]):
    runtime = session["runtime"]
    stop_intent_name = (
        "stop" if "stop" in runtime.registry.names() else
        "land" if "land" in runtime.registry.names() else
        "lie_down" if "lie_down" in runtime.registry.names() else
        "emit_event"
    )
    session["controller"].emergency_stop(
        stop_intent=Intent(stop_intent_name, {}),
        operator="ui_visitor", reason="UI E-STOP button",
    )
    return session, *_all_outputs(session)


def clear_trace(session: dict[str, Any]):
    session["runtime"].trace.events.clear()
    return session, *_all_outputs(session)


# ---------------------------------------------------------------------------
# UI.
# ---------------------------------------------------------------------------


with gr.Blocks(
    title="ghostloop — control panel",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
    .gl-pad-button { min-height: 56px; font-weight: 600; }
    .gl-estop button { background: #DC2626 !important; color: white !important; font-weight: 700; }
    .gl-pause button { background: #F59E0B !important; color: white !important; }
    .gl-resume button { background: #10B981 !important; color: white !important; }
    """,
) as demo:
    gr.Markdown("""
# ghostloop · control panel

Pick a robot profile, then drive it through the safety pipeline. Every
dispatch goes through `Geofence + ForceCap + ActionSmoothing + RateLimit
+ HITL` — try sending a `move_to` outside the workspace and watch the
geofence reject it.

Sister to **[GhostLM](https://github.com/joemunene-by/GhostLM)** · `pip install ghostloop` · [GitHub](https://github.com/joemunene-by/ghostloop) · [PyPI](https://pypi.org/project/ghostloop/)
""")

    session_state = gr.State()

    # ---------- Profile picker ----------
    with gr.Row():
        profile_dd = gr.Dropdown(
            label="Robot profile",
            choices=list(PRESETS.keys()),
            value=DEFAULT_PROFILE,
            scale=4,
        )

    summary_md = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Available primitives")
            primitives_md = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Active safety gates")
            gates_md = gr.Markdown()

    with gr.Accordion("Robot instructions (LLM system prompt)", open=False):
        instructions_md = gr.Markdown()

    # ---------- Quick-dispatch buttons ----------
    gr.Markdown("## Dispatch a primitive (one click — uses sensible defaults)")
    primitive_buttons: list[gr.Button] = []
    with gr.Row():
        for _ in range(6):
            primitive_buttons.append(gr.Button("", variant="primary", elem_classes="gl-pad-button"))
    with gr.Row():
        for _ in range(6):
            primitive_buttons.append(gr.Button("", variant="primary", elem_classes="gl-pad-button"))

    # ---------- Joystick (D-pad) ----------
    gr.Markdown("## Virtual joystick — drive / walk / fly / move")
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            joy_forward = gr.Button("⬆ FORWARD", elem_classes="gl-pad-button")
        with gr.Column(scale=1):
            pass
    with gr.Row():
        joy_left = gr.Button("⬅ LEFT", elem_classes="gl-pad-button")
        joy_stop = gr.Button("◼ STOP", elem_classes="gl-pad-button")
        joy_right = gr.Button("RIGHT ➡", elem_classes="gl-pad-button")
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            joy_back = gr.Button("⬇ BACK", elem_classes="gl-pad-button")
        with gr.Column(scale=1):
            pass

    # ---------- Intervention controls ----------
    gr.Markdown("## Live intervention")
    intervention_md = gr.Markdown()
    with gr.Row():
        pause_btn = gr.Button("⏸ Pause", elem_classes="gl-pause")
        resume_btn = gr.Button("▶ Resume", elem_classes="gl-resume")
        estop_btn = gr.Button("🛑 EMERGENCY STOP", elem_classes="gl-estop")

    # ---------- Live trace + state ----------
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Live trace (most recent first)")
            trace_md = gr.Markdown()
            clear_trace_btn = gr.Button("Clear trace")
        with gr.Column(scale=1):
            gr.Markdown("## Current backend state")
            state_md = gr.Markdown()

    # ---------- Advanced: free-form Intent ----------
    with gr.Accordion("Advanced: free-form Intent (JSON args)", open=False):
        with gr.Row():
            adv_name = gr.Textbox(label="Primitive", value="move_to", scale=1)
            adv_args = gr.Textbox(
                label="args JSON",
                value='{"x": 0.4, "y": 0.0, "z": 0.5}',
                lines=2, scale=3,
            )
        adv_btn = gr.Button("▶ runtime.step(intent)", variant="secondary")

    # ---------- Try-this hints ----------
    gr.Markdown("""
### Try this:

1. Pick the **`franka_arm`** profile, then click **`move_to`** — it dispatches with `(0.4, 0, 0.5)` and lands inside the workspace ✅. Now expand the **Advanced** accordion and dispatch `move_to` with `{"x": 5.0, "y": 0, "z": 0}` — the GeofenceGate rejects it 🚫.
2. Switch to **`spot`**, then drive with the joystick. Each press emits `walk_to(linear_x, 0, angular_z)` through the safety pipeline.
3. Hit **🛑 EMERGENCY STOP**. Try clicking another primitive — it gets denied. Hit **▶ Resume** to recover.

Every dispatch is recorded in the trace pane below — that's the same `TraceEvent` shape the library exports for replay, diff, query, energy ledger, and LLM-judge scoring.
""")

    # ---------- Wiring ----------
    profile_outputs = [
        session_state,
        summary_md, primitives_md, gates_md, instructions_md,
        state_md, trace_md, intervention_md,
        *primitive_buttons,
    ]
    standard_outputs = [
        session_state,
        summary_md, primitives_md, gates_md, instructions_md,
        state_md, trace_md, intervention_md,
    ]

    profile_dd.change(select_profile, inputs=[profile_dd], outputs=profile_outputs)
    demo.load(select_profile, inputs=[profile_dd], outputs=profile_outputs)

    for btn in primitive_buttons:
        btn.click(
            dispatch_primitive, inputs=[session_state, btn], outputs=standard_outputs,
        )

    joy_forward.click(
        lambda s: dispatch_drive(s, 0.2, 0.0),
        inputs=[session_state], outputs=standard_outputs,
    )
    joy_back.click(
        lambda s: dispatch_drive(s, -0.2, 0.0),
        inputs=[session_state], outputs=standard_outputs,
    )
    joy_left.click(
        lambda s: dispatch_drive(s, 0.0, 0.5),
        inputs=[session_state], outputs=standard_outputs,
    )
    joy_right.click(
        lambda s: dispatch_drive(s, 0.0, -0.5),
        inputs=[session_state], outputs=standard_outputs,
    )
    joy_stop.click(
        lambda s: dispatch_drive(s, 0.0, 0.0),
        inputs=[session_state], outputs=standard_outputs,
    )

    pause_btn.click(pause_runtime, inputs=[session_state], outputs=standard_outputs)
    resume_btn.click(resume_runtime, inputs=[session_state], outputs=standard_outputs)
    estop_btn.click(emergency_stop, inputs=[session_state], outputs=standard_outputs)
    clear_trace_btn.click(clear_trace, inputs=[session_state], outputs=standard_outputs)

    adv_btn.click(
        dispatch_custom,
        inputs=[session_state, adv_name, adv_args],
        outputs=standard_outputs,
    )


if __name__ == "__main__":
    import os
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
