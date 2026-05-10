"""Built-in Primitive factories bound to MockBackend.

These are reference implementations for the v0.1 sim-first surface. Real-robot
backends (MuJoCo, PyBullet, ROS 2) ship their own primitives bound to their
own actuator APIs; these mock versions exist so the runtime, policies, trace,
and bench harness can be developed and tested without any sim install.
"""

from .motion import move_to, scan
from .manipulation import pick, place
from .trajectory import follow_trajectory, linear_interpolate

__all__ = [
    "move_to",
    "scan",
    "pick",
    "place",
    "follow_trajectory",
    "linear_interpolate",
]
