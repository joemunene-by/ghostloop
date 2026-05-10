"""VLAPolicy: adapter for vision-language-action models (OpenVLA, pi-0, RT-2, etc).

VLA models output continuous action vectors (joint deltas, end-effector
twists, gripper commands) at a fixed dimensionality. ghostloop's runtime
operates on Intents — ``move_to``, ``pick``, ``scan``. The adapter
bridges the two: take an action vector, decode it into an Intent the
registry knows about, and let the safety pipeline gate it like any other
agent emission.

The model interface is deliberately abstract — pass anything callable
``(observation) -> action_vector`` and a matching ``ActionDecoder``.
This works for:

  - Local checkpoints loaded via the OpenVLA HuggingFace integration
  - Remote inference endpoints (OpenVLA-on-modal, pi-0 API, RT-2-on-vLLM)
  - Diffusion-policy local servers
  - Hand-rolled scripted policies in tests

The included ``DeltaXYZDecoder`` covers the common case of "first three
action dims are end-effector position deltas" (the standard OpenVLA /
RT-2 head). More decoders ship in v0.6 alongside the camera primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from ..core import Intent


ActionVector = Iterable[float]
ObservationProvider = Callable[[], dict[str, Any]]
VLAModel = Callable[[dict[str, Any]], ActionVector]


class ActionDecoder(Protocol):
    """Map a raw action vector to a structured Intent the registry can dispatch."""

    name: str

    def decode(self, action: list[float], state: dict[str, Any]) -> Intent | None: ...


@dataclass
class DeltaXYZDecoder:
    """Decode the first 3 action dims as end-effector position deltas.

    Standard OpenVLA / RT-2 head: action[0:3] are dx/dy/dz in metres,
    action[3:6] are rotation deltas (euler), action[6] is the gripper
    open/close (>0.5 = close).

    Emits ``move_to`` (absolute position = current + delta), and on
    gripper transitions emits ``pick`` (close) / ``place`` (open).
    Returns None when the action is essentially a no-op (delta below
    the configured threshold AND no gripper change).
    """

    name: str = "delta_xyz"
    delta_scale: float = 1.0  # multiply network outputs (typical OpenVLA: ~0.05/step)
    deadband: float = 1e-3  # below this, treat as no-op
    gripper_dim: int = 6  # index in action vector
    gripper_threshold: float = 0.5
    last_gripper: bool = False  # True = closed; toggled by emitted pick/place

    def decode(self, action: list[float], state: dict[str, Any]) -> Intent | None:
        if len(action) < 3:
            return None
        dx = float(action[0]) * self.delta_scale
        dy = float(action[1]) * self.delta_scale
        dz = float(action[2]) * self.delta_scale
        gripper_close = (
            len(action) > self.gripper_dim
            and float(action[self.gripper_dim]) > self.gripper_threshold
        )

        # Gripper transitions take priority — they're discrete events, not no-ops.
        if gripper_close and not self.last_gripper:
            self.last_gripper = True
            return Intent(
                name="pick",
                args={"object_id": "vla_target"},  # downstream resolves via scan
                rationale="VLA gripper transition: open -> close",
            )
        if not gripper_close and self.last_gripper:
            self.last_gripper = False
            return Intent(
                name="place",
                args={},
                rationale="VLA gripper transition: close -> open",
            )

        # Position delta — skip if below deadband.
        if max(abs(dx), abs(dy), abs(dz)) < self.deadband:
            return None
        pos = state.get("position", [0.0, 0.0, 0.0])
        return Intent(
            name="move_to",
            args={
                "x": float(pos[0]) + dx,
                "y": float(pos[1]) + dy,
                "z": float(pos[2]) + dz,
            },
            rationale=f"VLA delta_xyz: dx={dx:.4g}, dy={dy:.4g}, dz={dz:.4g}",
        )


@dataclass
class VLAPolicy:
    """Wrap a VLA model so its action vectors flow through the runtime.

    Args:
        model: callable mapping observation dict -> action vector. The
            observation dict typically contains 'rgb', 'depth', 'state',
            'language_instruction'. Implementations vary; ghostloop
            doesn't impose a schema beyond passing whatever the
            ``observe`` callable returns.
        decoder: maps action vector + current state -> Intent. The default
            ``DeltaXYZDecoder`` covers OpenVLA / RT-2 / pi-0 style heads.
        observe: callable that produces the next observation dict given
            the runtime. Default: pulls backend.snapshot() into 'state'
            and calls ``capture_camera`` if the backend has cameras.
        max_idle_steps: terminate after this many consecutive no-op
            decode results (model emits zero-deltas). Prevents the loop
            from spinning forever when the policy thinks it's done.
    """

    model: VLAModel
    decoder: ActionDecoder = field(default_factory=DeltaXYZDecoder)
    observe: ObservationProvider | None = None
    max_idle_steps: int = 4
    _idle: int = field(default=0, init=False)

    def step(self, runtime) -> Intent | None:
        """One round-trip: observe -> infer -> decode -> Intent or None."""
        obs = self._observe(runtime)
        action = list(self.model(obs))
        intent = self.decoder.decode(action, obs.get("state") or {})
        if intent is None:
            self._idle += 1
            if self._idle >= self.max_idle_steps:
                return None  # terminate the loop
            # Synthesise a noop intent so the trace records the model emission.
            return None
        self._idle = 0
        return intent

    def _observe(self, runtime) -> dict[str, Any]:
        if self.observe is not None:
            return self.observe()
        snap = runtime.backend.snapshot()
        out: dict[str, Any] = {"state": snap}
        cameras = getattr(runtime.backend, "cameras", None)
        if isinstance(cameras, dict) and cameras:
            # Capture every attached camera and surface metadata.
            from ..sensors.camera import Camera
            frames: dict[str, dict[str, Any]] = {}
            for name, cam in cameras.items():
                if isinstance(cam, Camera) or hasattr(cam, "capture"):
                    frames[name] = cam.capture().metadata()
            out["cameras"] = frames
        return out


def vla_policy_loop(
    model: VLAModel,
    runtime,
    *,
    decoder: ActionDecoder | None = None,
    max_steps: int = 200,
) -> dict[str, Any]:
    """End-to-end driver: VLAPolicy in front of a Runtime, until done or step cap.

    Returns a summary dict (steps taken, terminated reason, last action,
    full trace path). The runtime's trace remains the canonical event log.
    """
    policy = VLAPolicy(model=model, decoder=decoder or DeltaXYZDecoder())
    terminated = "max_steps"
    steps = 0
    for _ in range(max_steps):
        intent = policy.step(runtime)
        if intent is None:
            terminated = "idle" if policy._idle >= policy.max_idle_steps else "no_action"
            break
        runtime.step(intent)
        steps += 1
    return {
        "steps": steps,
        "terminated": terminated,
        "trace_episode_id": runtime.trace.episode_id,
        "n_events": len(runtime.trace.events),
    }
