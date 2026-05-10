"""Execution backends for the ghostloop runtime.

MockBackend ships in core for zero-install demos and tests. Heavier backends
are optional installs with conditional imports — they only fail at construction
time, never at package import:

  pip install ghostloop[mujoco]   # MuJoCoBackend (Apache-2.0, Google DeepMind)
  pip install ghostloop[pybullet] # PyBulletBackend (planned v0.3)
  pip install ghostloop[ros2]     # ROS2Backend (planned v0.7)
"""

from .gymnasium import GymnasiumBackend, gymnasium_available
from .menagerie import (
    KNOWN_MODELS,
    MenagerieError,
    ensure_menagerie,
    load_franka,
    resolve_model,
)
from .mujoco import MuJoCoBackend, mujoco_available
from .pybullet import PyBulletBackend, pybullet_available

__all__ = [
    "GymnasiumBackend",
    "MuJoCoBackend",
    "PyBulletBackend",
    "gymnasium_available",
    "mujoco_available",
    "pybullet_available",
    "MenagerieError",
    "KNOWN_MODELS",
    "ensure_menagerie",
    "load_franka",
    "resolve_model",
]
