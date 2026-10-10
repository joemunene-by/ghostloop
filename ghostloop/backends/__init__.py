"""Execution backends for the ghostloop runtime.

MockBackend ships in core for zero-install demos and tests. Heavier backends
are optional installs with conditional imports, they only fail at construction
time, never at package import:

  pip install ghostloop[mujoco]   # MuJoCoBackend (Apache-2.0, Google DeepMind)
  pip install ghostloop[pybullet] # PyBulletBackend
  pip install ghostloop[gym]      # GymnasiumBackend
  pip install ghostloop[ros2]     # ROS2Backend (rclpy comes from your ROS install)

Build any of them by spec with ``create_backend("mujoco:ur5e")``; third-party
simulators plug in through the ``ghostloop.backends`` entry-point group.
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
from .randomized import RandomizationConfig, RandomizedBackend
from .registry import (
    BackendBinding,
    BackendInfo,
    BackendRegistryError,
    create_backend,
    list_backends,
    register_backend,
    unregister_backend,
)
from .ros2 import ROS2Backend, ros2_available

__all__ = [
    "BackendBinding",
    "BackendInfo",
    "BackendRegistryError",
    "create_backend",
    "list_backends",
    "register_backend",
    "unregister_backend",
    "GymnasiumBackend",
    "MuJoCoBackend",
    "PyBulletBackend",
    "ROS2Backend",
    "RandomizationConfig",
    "RandomizedBackend",
    "gymnasium_available",
    "mujoco_available",
    "pybullet_available",
    "ros2_available",
    "MenagerieError",
    "KNOWN_MODELS",
    "ensure_menagerie",
    "load_franka",
    "resolve_model",
]
