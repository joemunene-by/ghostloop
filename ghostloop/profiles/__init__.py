"""Robot profiles — declarative spec of a specific robot's capabilities.

A ``RobotProfile`` carries everything ghostloop needs to bring a robot
online: its name, its morphology, the primitives it supports, its
workspace bounds + force / velocity limits, free-form instructions for
the LLM about the robot, and any custom safety gates.

Profiles can be:

  - Constructed in code (``RobotProfile(name=..., categories=[...])``).
  - Loaded from YAML via ``load_profile_yaml(path)`` so a non-developer
    operator can describe their robot without writing Python.
  - Picked from the preset library: ``franka_arm()`` /
    ``spot_quadruped()`` / ``tello_drone()`` / ``turtlebot_base()`` /
    ``humanoid_demo()`` / ``stretch_mobile_arm()``.

The MCP server reads the profile and only exposes the supported tools,
sized to the robot. Add an instructions block and the LLM gets a
robot-specific system prompt.
"""

from .core import (
    RobotProfile,
    SafetyGateSpec,
    apply_profile_to_runtime,
    build_runtime_from_profile,
    load_profile_yaml,
)
from .presets import (
    franka_arm,
    humanoid_demo,
    spot_quadruped,
    stretch_mobile_arm,
    tello_drone,
    turtlebot_base,
)

__all__ = [
    "RobotProfile",
    "SafetyGateSpec",
    "apply_profile_to_runtime",
    "build_runtime_from_profile",
    "load_profile_yaml",
    "franka_arm",
    "humanoid_demo",
    "spot_quadruped",
    "stretch_mobile_arm",
    "tello_drone",
    "turtlebot_base",
]
