"""Backend registry: build backends by name, including third-party plugins.

A backend spec is ``"name"`` or ``"name:variant"`` (``"mock"``,
``"mujoco:ur5e"``, ``"omnisim:ur5e"``). Each name maps to a factory that
returns a :class:`BackendBinding`: the backend instance, the primitives bound
to it, and the embodiment it models so benches can pick a shared workspace.

Third-party packages register factories through the ``ghostloop.backends``
entry-point group, so a simulator can ship its adapter as its own package::

    [project.entry-points."ghostloop.backends"]
    omnisim = "ghostloop_omnisim:make_backend"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable

from ..core import Backend, Primitive

ENTRY_POINT_GROUP = "ghostloop.backends"


@dataclass
class BackendBinding:
    """A ready-to-run backend plus the primitives that drive it."""

    backend: Backend
    primitives: list[Primitive]
    embodiment: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


BackendFactory = Callable[..., BackendBinding]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    description: str
    source: str
    available: bool
    error: str = ""


@dataclass
class _Entry:
    factory: BackendFactory | None
    description: str
    source: str
    available: Callable[[], bool]
    error: str = ""


class BackendRegistryError(LookupError):
    pass


_REGISTRY: dict[str, _Entry] = {}
_plugins_loaded = False


def register_backend(
    name: str,
    factory: BackendFactory,
    *,
    description: str = "",
    available: Callable[[], bool] | None = None,
    replace: bool = False,
    source: str = "user",
) -> None:
    """Register a backend factory under ``name``.

    The factory is called as ``factory(variant, **kwargs)`` where ``variant``
    is the part after the colon in the spec (or ``None``).
    """
    if ":" in name or not name:
        raise ValueError(f"backend name must be non-empty and contain no ':', got {name!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(f"backend already registered: {name!r} (pass replace=True to override)")
    _REGISTRY[name] = _Entry(
        factory=factory,
        description=description or (factory.__doc__ or "").strip().split("\n")[0],
        source=source,
        available=available or (lambda: True),
    )


def unregister_backend(name: str) -> None:
    _REGISTRY.pop(name, None)


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded or os.environ.get("GHOSTLOOP_DISABLE_PLUGINS"):
        _plugins_loaded = True
        return
    _plugins_loaded = True
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in _REGISTRY:
            continue
        dist = getattr(ep.dist, "name", None) or ep.value.split(":")[0]
        try:
            factory = ep.load()
        except Exception as exc:  # noqa: BLE001
            # A broken plugin must not break `ghostloop backends` for everyone else.
            _REGISTRY[ep.name] = _Entry(
                factory=None, description="", source=f"plugin:{dist}",
                available=lambda: False, error=f"{type(exc).__name__}: {exc}",
            )
            continue
        register_backend(ep.name, factory, source=f"plugin:{dist}")


def parse_spec(spec: str) -> tuple[str, str | None]:
    name, _, variant = spec.partition(":")
    return name.strip(), (variant.strip() or None)


def create_backend(spec: str, **kwargs: Any) -> BackendBinding:
    """Build a fresh :class:`BackendBinding` from a spec like ``"mujoco:ur5e"``."""
    _load_plugins()
    name, variant = parse_spec(spec)
    entry = _REGISTRY.get(name)
    if entry is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise BackendRegistryError(f"unknown backend {name!r}; registered: {known}")
    if entry.factory is None:
        raise BackendRegistryError(f"backend {name!r} failed to load: {entry.error}")
    binding = entry.factory(variant, **kwargs)
    if not isinstance(binding, BackendBinding):
        raise TypeError(
            f"backend factory for {name!r} returned {type(binding).__name__}, expected BackendBinding"
        )
    binding.provenance.setdefault("spec", spec)
    binding.provenance.setdefault("source", entry.source)
    describe = getattr(binding.backend, "describe", None)
    if callable(describe):
        for k, v in describe().items():
            binding.provenance.setdefault(k, v)
    return binding


def list_backends() -> list[BackendInfo]:
    _load_plugins()
    out = []
    for name, entry in sorted(_REGISTRY.items()):
        try:
            ok = entry.factory is not None and bool(entry.available())
        except Exception:  # noqa: BLE001
            ok = False
        out.append(BackendInfo(name, entry.description, entry.source, ok, entry.error))
    return out


def _mock_factory(variant: str | None = None, **kwargs: Any) -> BackendBinding:
    """In-memory point robot; deterministic, unlimited reach."""
    from ..core import MockBackend
    from ..primitives import move_to, pick, place, scan

    backend = MockBackend(name=kwargs.pop("name", "mock"), **kwargs)
    return BackendBinding(
        backend=backend,
        primitives=[move_to(), scan(), pick(), place()],
        embodiment=variant,
        provenance={"engine": "mock"},
    )


def _mujoco_factory(variant: str | None = None, **kwargs: Any) -> BackendBinding:
    """MuJoCo physics; variant is a Menagerie robot (ur5e, panda, ...) or an MJCF path."""
    from .mujoco import MuJoCoBackend, move_to, scan

    if not variant:
        raise BackendRegistryError("mujoco needs a variant: mujoco:ur5e, mujoco:panda or mujoco:/path/model.xml")
    if variant.endswith((".xml", ".urdf")):
        backend = MuJoCoBackend(model_path=variant, **kwargs)
        embodiment = kwargs.get("name")
    else:
        backend = MuJoCoBackend.from_menagerie(variant, **kwargs)
        embodiment = "panda" if variant.lower() == "franka" else variant.lower()
    return BackendBinding(backend=backend, primitives=[move_to(), scan()], embodiment=embodiment)


def _pybullet_factory(variant: str | None = None, **kwargs: Any) -> BackendBinding:
    """Bullet physics; variant is a URDF path."""
    from .pybullet import PyBulletBackend, move_to, scan

    if not variant:
        raise BackendRegistryError("pybullet needs a URDF path: pybullet:/path/robot.urdf")
    return BackendBinding(
        backend=PyBulletBackend(model_path=variant, **kwargs),
        primitives=[move_to(), scan()],
        provenance={"engine": "pybullet"},
    )


def _gym_factory(variant: str | None = None, **kwargs: Any) -> BackendBinding:
    """Any Gymnasium environment; variant is the env id."""
    from .gymnasium import GymnasiumBackend, apply_action, reset_env

    if not variant:
        raise BackendRegistryError("gym needs an env id: gym:CartPole-v1")
    return BackendBinding(
        backend=GymnasiumBackend(env_id=variant, **kwargs),
        primitives=[apply_action(), reset_env()],
        provenance={"engine": "gymnasium"},
    )


def _register_builtins() -> None:
    from .gymnasium import gymnasium_available
    from .mujoco import mujoco_available
    from .pybullet import pybullet_available

    register_backend("mock", _mock_factory, source="builtin")
    register_backend("mujoco", _mujoco_factory, available=mujoco_available, source="builtin")
    register_backend("pybullet", _pybullet_factory, available=pybullet_available, source="builtin")
    register_backend("gym", _gym_factory, available=gymnasium_available, source="builtin")


_register_builtins()
