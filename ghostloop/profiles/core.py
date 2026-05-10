"""RobotProfile core types, YAML loader, and runtime builder."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core import Backend, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
from ..policies import (
    ActionSmoothingGate,
    CooldownGate,
    DenyListGate,
    ForceCapGate,
    GeofenceGate,
    HumanInTheLoopGate,
    RateLimitGate,
    cli_approver,
)
from ..primitives import composite_primitive
from ..primitives.library import primitives_for_categories
from ..primitives.motion import move_to, scan
from ..primitives.manipulation import pick, place


# ---------------------------------------------------------------------------
# Safety gate spec — declarative form for YAML.
# ---------------------------------------------------------------------------


@dataclass
class SafetyGateSpec:
    """One gate's worth of config in a portable form.

    ``kind`` picks which gate; ``args`` is the kwargs passed to its
    constructor. Unknown kinds are skipped with a warning.
    """

    kind: str
    args: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RobotProfile.
# ---------------------------------------------------------------------------


@dataclass
class RobotProfile:
    """Declarative spec for one robot.

    Fields:
        name:          robot identifier (e.g. "franka_lab_arm").
        morphology:    label like "arm" / "mobile_base" / "quadruped"
                       / "drone" / "humanoid" / "mobile_arm" / "custom".
                       Free-form; used in the trace + LLM instructions.
        categories:    primitive categories from
                       ``ghostloop.primitives.library.CATEGORIES`` to
                       include automatically.
        primitives:    extra primitives (Primitive instances) to add on
                       top of category-provided ones.
        custom_primitives_modules:
                       list of ``module:factory_name`` strings that
                       ``load_profile_yaml`` imports + calls to produce
                       primitives, so YAML users can extend without
                       writing wrapper Python.
        composites:    list of (name, sub_primitive_names, description)
                       tuples, expanded into composite primitives at
                       build time.
        instructions:  free-form prose handed to the LLM as the MCP
                       server's instructions block. Tell the model
                       what THIS robot is, what it can do, what it
                       must NOT do.
        workspace_bounds:
                       ((xmin,ymin,zmin),(xmax,ymax,zmax)) tuple. None
                       means no auto-geofence.
        max_force_n:   passes through to ForceCapGate. None = unlimited.
        max_velocity:  m/s ceiling for ActionSmoothingGate.
        max_acceleration:
                       m/s² ceiling.
        rate_limit_per_min:
                       per-primitive ceiling.
        cooldown_s:    minimum interval between identical primitive
                       calls (CooldownGate default).
        denied_primitives:
                       names to hard-deny.
        hitl_primitives:
                       names that require human approval.
        safety_gates:  extra ``SafetyGateSpec`` records to compose.
                       Built BEFORE the convenience caps above so a
                       custom gate can deny early.
        backend_factory:
                       optional callable returning a Backend. If not
                       set, the runtime uses MockBackend(name=profile.name).
                       For YAML paths, prefer ``backend_kind`` +
                       ``backend_kwargs`` so YAML stays declarative.
        backend_kind:  string matching one of: "mock" / "mujoco" /
                       "pybullet" / "gymnasium" / "ros2" /
                       "randomized<wrapped>". Used by load_profile_yaml.
        backend_kwargs:
                       kwargs forwarded to the backend constructor.
    """

    name: str
    morphology: str = "custom"
    categories: list[str] = field(default_factory=list)
    primitives: list = field(default_factory=list)
    custom_primitives_modules: list[str] = field(default_factory=list)
    composites: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    workspace_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    max_force_n: float | None = None
    max_velocity: float | None = None
    max_acceleration: float | None = None
    rate_limit_per_min: int = 60
    cooldown_s: float = 0.0
    per_primitive_cooldown: dict[str, float] = field(default_factory=dict)
    denied_primitives: list[str] = field(default_factory=list)
    hitl_primitives: list[str] = field(default_factory=list)
    safety_gates: list[SafetyGateSpec] = field(default_factory=list)
    backend_factory: Callable[[], Backend] | None = None
    backend_kind: str = "mock"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML loader.
# ---------------------------------------------------------------------------


def load_profile_yaml(path: str | Path) -> RobotProfile:
    """Parse a YAML file into a RobotProfile.

    YAML is preferred for "operator hands a robot description to ghostloop"
    flows. Schema:

        name: spot_lab
        morphology: quadruped
        categories: [quadruped, sensing, generic]
        instructions: |
          You are controlling Spot in the lab. Stay inside the demo
          arena (4m x 4m). Sit on idle.
        workspace_bounds: [[-2, -2, 0], [2, 2, 1.5]]
        max_velocity: 1.6
        max_acceleration: 5.0
        rate_limit_per_min: 30
        cooldown_s: 0.2
        denied_primitives: []
        hitl_primitives: [walk_to]
        custom_primitives:
          - module: my_robot.primitives
            factory: dance_routine
        composites:
          - name: greet
            steps: [stand, wave, sit]
            description: Greeting macro.
        safety_gates: []
        backend:
          kind: ros2
          kwargs:
            node_name: spot_node
            cmd_vel_topic: /spot/cmd_vel

    YAML is not in the stdlib; this loader prefers ``yaml`` if available
    and falls back to a tiny JSON-compatible subset reader (for
    ``.json`` profile files).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
    except ImportError:
        if p.suffix.lower() == ".json":
            import json
            data = json.loads(text)
        else:
            raise ImportError(
                f"loading {p.suffix} profiles requires PyYAML.\n"
                "  pip install PyYAML\n"
                "or convert the profile to JSON and load that instead."
            )
    if not isinstance(data, dict):
        raise ValueError(f"profile {p} must be a mapping at the top level")
    return _profile_from_dict(data)


def _profile_from_dict(data: dict[str, Any]) -> RobotProfile:
    backend_data = data.get("backend") or {}
    custom_modules: list[str] = []
    for entry in data.get("custom_primitives", []) or []:
        if isinstance(entry, str):
            custom_modules.append(entry)
        elif isinstance(entry, dict) and "module" in entry and "factory" in entry:
            custom_modules.append(f"{entry['module']}:{entry['factory']}")
    safety_specs = [
        SafetyGateSpec(kind=s["kind"], args=s.get("args", {}) or {})
        for s in (data.get("safety_gates") or [])
        if isinstance(s, dict) and "kind" in s
    ]
    bounds = data.get("workspace_bounds")
    if bounds is not None:
        bounds = (tuple(bounds[0]), tuple(bounds[1]))
    return RobotProfile(
        name=data["name"],
        morphology=data.get("morphology", "custom"),
        categories=list(data.get("categories", []) or []),
        custom_primitives_modules=custom_modules,
        composites=list(data.get("composites", []) or []),
        instructions=data.get("instructions", ""),
        workspace_bounds=bounds,
        max_force_n=data.get("max_force_n"),
        max_velocity=data.get("max_velocity"),
        max_acceleration=data.get("max_acceleration"),
        rate_limit_per_min=int(data.get("rate_limit_per_min", 60)),
        cooldown_s=float(data.get("cooldown_s", 0.0)),
        per_primitive_cooldown=dict(data.get("per_primitive_cooldown", {}) or {}),
        denied_primitives=list(data.get("denied_primitives", []) or []),
        hitl_primitives=list(data.get("hitl_primitives", []) or []),
        safety_gates=safety_specs,
        backend_kind=str(backend_data.get("kind", "mock")),
        backend_kwargs=dict(backend_data.get("kwargs", {}) or {}),
    )


# ---------------------------------------------------------------------------
# Runtime builder.
# ---------------------------------------------------------------------------


_BUILTIN_GATES = {
    "deny_list":         lambda **a: DenyListGate(denied=set(a.get("denied", []))),
    "rate_limit":        lambda **a: RateLimitGate(per_minute=int(a.get("per_minute", 120))),
    "geofence":          lambda **a: GeofenceGate(
                            min_corner=tuple(a["min_corner"]),
                            max_corner=tuple(a["max_corner"]),
                         ),
    "force_cap":         lambda **a: ForceCapGate(**a),
    "action_smoothing":  lambda **a: ActionSmoothingGate(**a),
    "cooldown":          lambda **a: CooldownGate(**a),
    "human_in_the_loop": lambda **a: HumanInTheLoopGate(
                            requires_approval=set(a.get("requires_approval", [])),
                            approver=cli_approver,
                         ),
}


def _build_backend(profile: RobotProfile) -> Backend:
    if profile.backend_factory is not None:
        return profile.backend_factory()
    kind = (profile.backend_kind or "mock").lower()
    kwargs = dict(profile.backend_kwargs or {})
    if kind == "mock":
        kwargs.setdefault("name", profile.name)
        return MockBackend(**kwargs)
    if kind == "mujoco":
        from ..backends import MuJoCoBackend
        return MuJoCoBackend(**kwargs)
    if kind == "pybullet":
        from ..backends import PyBulletBackend
        return PyBulletBackend(**kwargs)
    if kind == "gymnasium":
        from ..backends import GymnasiumBackend
        return GymnasiumBackend(**kwargs)
    if kind == "ros2":
        from ..backends import ROS2Backend
        return ROS2Backend(**kwargs)
    if kind == "randomized":
        from ..backends import RandomizedBackend, RandomizationConfig
        wrapped_kind = kwargs.pop("wraps", "mock")
        wrapped_profile = RobotProfile(
            name=profile.name + "_inner",
            backend_kind=wrapped_kind,
            backend_kwargs=kwargs.pop("wraps_kwargs", {}),
        )
        wrapped = _build_backend(wrapped_profile)
        cfg = RandomizationConfig(**kwargs.pop("config", {}))
        return RandomizedBackend(base=wrapped, config=cfg, **kwargs)
    raise ValueError(f"unknown backend kind {kind!r} in profile {profile.name!r}")


def _load_custom_primitives(modules: list[str]) -> list:
    """Resolve ``module:factory`` strings into Primitive instances."""
    out: list = []
    for spec in modules:
        try:
            module_path, _, factory_name = spec.partition(":")
            if not factory_name:
                continue
            module = importlib.import_module(module_path.strip())
            factory = getattr(module, factory_name.strip())
            out.append(factory())
        except Exception as exc:  # noqa: BLE001
            # Don't crash boot for one bad custom primitive — record it.
            import sys as _sys
            print(
                f"[ghostloop] failed to load custom primitive {spec!r}: "
                f"{type(exc).__name__}: {exc}",
                file=_sys.stderr,
            )
    return out


def _build_safety_pipeline(profile: RobotProfile, primitives: list) -> PolicyPipeline:
    gates: list = []
    if profile.denied_primitives:
        gates.append(DenyListGate(denied=set(profile.denied_primitives)))
    gates.append(RateLimitGate(per_minute=profile.rate_limit_per_min))
    if profile.cooldown_s > 0 or profile.per_primitive_cooldown:
        gates.append(CooldownGate(
            default_s=profile.cooldown_s,
            per_primitive=dict(profile.per_primitive_cooldown),
        ))
    if profile.workspace_bounds is not None:
        gates.append(GeofenceGate(
            min_corner=profile.workspace_bounds[0],
            max_corner=profile.workspace_bounds[1],
        ))
    force_kwargs: dict[str, Any] = {}
    if profile.max_force_n is not None:
        force_kwargs["force_max"] = profile.max_force_n
    if profile.max_velocity is not None:
        force_kwargs["velocity_max"] = profile.max_velocity
    if profile.max_acceleration is not None:
        force_kwargs["acceleration_max"] = profile.max_acceleration
    if force_kwargs:
        gates.append(ForceCapGate(**force_kwargs))
    if profile.max_velocity is not None or profile.max_acceleration is not None:
        gates.append(ActionSmoothingGate(
            max_velocity=profile.max_velocity or 1.0,
            max_acceleration=profile.max_acceleration or 5.0,
        ))
    # User-supplied gates last (they may inspect prior decisions).
    for spec in profile.safety_gates:
        builder = _BUILTIN_GATES.get(spec.kind.lower())
        if builder is None:
            import sys as _sys
            print(
                f"[ghostloop] unknown safety gate kind {spec.kind!r}; skipping",
                file=_sys.stderr,
            )
            continue
        try:
            gates.append(builder(**spec.args))
        except Exception as exc:  # noqa: BLE001
            import sys as _sys
            print(
                f"[ghostloop] failed to build {spec.kind!r}: "
                f"{type(exc).__name__}: {exc}",
                file=_sys.stderr,
            )
    if profile.hitl_primitives:
        gates.append(HumanInTheLoopGate(
            requires_approval=set(profile.hitl_primitives),
            approver=cli_approver,
        ))
    return PolicyPipeline(gates=gates)


def build_runtime_from_profile(
    profile: RobotProfile,
    *,
    backend: Backend | None = None,
) -> Runtime:
    """Materialise a Runtime from a RobotProfile.

    ``backend`` overrides the profile's backend if supplied — useful
    for unit tests where you want to swap in MockBackend regardless of
    what the YAML declared.
    """
    backend = backend or _build_backend(profile)
    primitives: list = list(primitives_for_categories(profile.categories))
    primitives.extend(profile.primitives)
    primitives.extend(_load_custom_primitives(profile.custom_primitives_modules))
    # Composite primitives compose existing primitives by name.
    by_name = {p.name: p for p in primitives}
    for spec in profile.composites:
        if not isinstance(spec, dict) or "name" not in spec or "steps" not in spec:
            continue
        step_prims = [by_name[s] for s in spec["steps"] if s in by_name]
        if not step_prims:
            continue
        comp = composite_primitive(
            name=spec["name"],
            steps=step_prims,
            description=spec.get("description", ""),
        )
        primitives.append(comp)
        by_name[comp.name] = comp
    # Default fallback: if no categories AND no primitives, seed with arm primitives
    # so something is callable. Keeps backward compat with arm-shaped users.
    if not primitives:
        primitives = [move_to(), scan(), pick(), place()]
    registry = PrimitiveRegistry(primitives)
    pipeline = _build_safety_pipeline(profile, primitives)
    return Runtime(backend=backend, registry=registry, policy_pipeline=pipeline)


def apply_profile_to_runtime(profile: RobotProfile, runtime: Runtime) -> None:
    """Mutate an existing Runtime to match a profile (registry + pipeline only).

    Useful when the caller already owns a Backend (e.g. a long-lived
    rclpy node) and just wants to attach a profile's primitives +
    safety gates without rebuilding the whole runtime.
    """
    primitives: list = list(primitives_for_categories(profile.categories))
    primitives.extend(profile.primitives)
    primitives.extend(_load_custom_primitives(profile.custom_primitives_modules))
    by_name = {p.name: p for p in primitives}
    for spec in profile.composites:
        if not isinstance(spec, dict) or "name" not in spec or "steps" not in spec:
            continue
        step_prims = [by_name[s] for s in spec["steps"] if s in by_name]
        if not step_prims:
            continue
        comp = composite_primitive(
            name=spec["name"],
            steps=step_prims,
            description=spec.get("description", ""),
        )
        primitives.append(comp)
        by_name[comp.name] = comp
    runtime.registry = PrimitiveRegistry(primitives)
    runtime.policy_pipeline = _build_safety_pipeline(profile, primitives)
