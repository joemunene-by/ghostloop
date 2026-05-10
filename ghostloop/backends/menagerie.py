"""MuJoCo Menagerie loader.

The MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie) is
the canonical curated collection of high-quality MuJoCo robot models — Franka
Emika Panda, UR5e, Stretch RE3, Allegro hand, Spot, Aloha bimanual, etc., all
under permissive licences. This loader handles two cases:

  - User has the menagerie cloned locally (set MENAGERIE_PATH env var or
    pass ``menagerie_root=...``). We resolve a model name to its xml path.
  - User has nothing. We shallow-clone the menagerie into a cache directory
    on first use (~/.cache/ghostloop/mujoco_menagerie). Subsequent calls
    reuse the clone. ``--depth=1`` so the download stays small (~80MB).

This keeps the demos plug-and-play without vendoring a 1GB dataset, and
without requiring the user to know the exact XML path inside the menagerie.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
DEFAULT_CACHE = Path.home() / ".cache" / "ghostloop" / "mujoco_menagerie"

# Hand-picked default scenes for popular models. Add more as new robots are needed.
KNOWN_MODELS = {
    "franka": "franka_emika_panda/scene.xml",
    "panda": "franka_emika_panda/scene.xml",
    "ur5e": "universal_robots_ur5e/scene.xml",
    "ur10e": "universal_robots_ur10e/scene.xml",
    "stretch": "hello_robot_stretch/scene.xml",
    "allegro": "wonik_allegro/scene_left.xml",
    "spot": "boston_dynamics_spot/scene.xml",
    "aloha": "aloha/scene.xml",
    "shadow": "shadow_hand/scene_left.xml",
    "sawyer": "rethink_robotics_sawyer/scene.xml",
}


class MenagerieError(RuntimeError):
    """Raised when the menagerie can't be located or a model name isn't resolvable."""


def _resolve_root(menagerie_root: str | None) -> Path:
    if menagerie_root:
        return Path(menagerie_root).expanduser().resolve()
    env = os.environ.get("MENAGERIE_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_CACHE


def ensure_menagerie(menagerie_root: str | None = None, *, allow_clone: bool = True) -> Path:
    """Return the menagerie root, cloning into cache on first call if needed."""
    root = _resolve_root(menagerie_root)
    if (root / "franka_emika_panda").is_dir():
        return root
    if not allow_clone:
        raise MenagerieError(
            f"menagerie not found at {root}; pass menagerie_root or set "
            "MENAGERIE_PATH, or call with allow_clone=True to fetch."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git") is None:
        raise MenagerieError("git not found on PATH; can't clone the menagerie")
    cmd = [
        "git", "clone", "--depth", "1", "--filter=blob:none",
        MENAGERIE_REPO, str(root),
    ]
    print(f"[menagerie] cloning {MENAGERIE_REPO} -> {root} (one-time, ~80MB shallow)")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise MenagerieError(
            f"clone failed (exit {res.returncode}): {res.stderr.strip()[:300]}"
        )
    return root


def resolve_model(name: str, menagerie_root: str | None = None) -> str:
    """Resolve a friendly model name (or relative XML path) to an absolute path.

    Accepted inputs:
      "franka" / "panda" / "ur5e" / ... — known model alias from KNOWN_MODELS
      "franka_emika_panda/scene.xml"   — explicit relative path inside the menagerie
      "/abs/path/to/some.xml"          — absolute path passed straight through
    """
    if os.path.isabs(name) and Path(name).is_file():
        return name
    root = ensure_menagerie(menagerie_root)
    # Friendly alias?
    rel = KNOWN_MODELS.get(name.lower())
    candidate = (root / (rel or name)).resolve()
    if candidate.is_file():
        return str(candidate)
    # Last-ditch: search for any scene.xml under a directory matching the name.
    matches = list(root.glob(f"*{name}*/scene*.xml"))
    if matches:
        return str(matches[0])
    raise MenagerieError(
        f"could not resolve model {name!r} in menagerie at {root}. "
        f"Known aliases: {sorted(KNOWN_MODELS.keys())}"
    )


def load_franka(menagerie_root: str | None = None):
    """Convenience: returns a fully-constructed MuJoCoBackend for the Franka Panda."""
    from .mujoco import MuJoCoBackend  # local import keeps top-level cheap
    path = resolve_model("franka", menagerie_root)
    return MuJoCoBackend(model_path=path, end_effector="hand", name="franka")
