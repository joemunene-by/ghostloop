"""Preset RobotProfiles for common robots.

Hand-curated. Edit copies of these for your specific hardware — they're
intentionally conservative on workspace + force + velocity so first
runs don't break things.
"""

from __future__ import annotations

from .core import RobotProfile


def franka_arm() -> RobotProfile:
    """Franka Panda 7-DOF arm. Covers the original arm-focused workflow."""
    return RobotProfile(
        name="franka_arm",
        morphology="arm",
        categories=["dexterous", "sensing", "generic"],
        instructions=(
            "You control a Franka Emika Panda 7-DOF arm. The end-effector is the "
            "default frame for x/y/z. Workspace is the table immediately in "
            "front of the robot. NEVER target the robot's own base. Use "
            "set_gripper before move_to when picking. On error, call stop "
            "and emit_event with kind='incident'."
        ),
        workspace_bounds=((-0.6, -0.6, 0.0), (0.6, 0.6, 1.0)),
        max_force_n=15.0,
        max_velocity=0.5,
        max_acceleration=2.0,
        rate_limit_per_min=120,
        hitl_primitives=["set_gripper"],
    )


def turtlebot_base() -> RobotProfile:
    """TurtleBot 3 / 4 — wheeled mobile base."""
    return RobotProfile(
        name="turtlebot",
        morphology="mobile_base",
        categories=["mobile_base", "sensing", "generic"],
        instructions=(
            "You control a wheeled mobile base in a flat indoor environment. "
            "Use drive(linear_x, angular_z) for velocity commands and goto(x, y, "
            "theta) for navigation. Linear velocities above 0.6 m/s are unsafe; "
            "stay under that. Always call stop before transitioning to a "
            "different mode."
        ),
        workspace_bounds=((-10.0, -10.0, 0.0), (10.0, 10.0, 0.5)),
        max_velocity=0.6,
        max_acceleration=1.0,
        rate_limit_per_min=60,
        cooldown_s=0.05,
    )


def spot_quadruped() -> RobotProfile:
    """Boston Dynamics Spot — quadruped mobile platform."""
    return RobotProfile(
        name="spot",
        morphology="quadruped",
        categories=["quadruped", "sensing", "generic"],
        instructions=(
            "You control a Boston Dynamics Spot quadruped. Prefer walk_to over "
            "drive for waypoint navigation. Always sit when idle for more than "
            "30 seconds (battery savings). Before any motion, verify "
            "stand state. Walking speed cap is 1.6 m/s on flat ground."
        ),
        workspace_bounds=((-15.0, -15.0, 0.0), (15.0, 15.0, 1.5)),
        max_velocity=1.6,
        max_acceleration=2.0,
        rate_limit_per_min=60,
        per_primitive_cooldown={"walk_to": 0.3, "stand": 1.0, "sit": 1.0},
        hitl_primitives=["walk_to"],
    )


def tello_drone() -> RobotProfile:
    """DJI Tello / similar small quadcopter."""
    return RobotProfile(
        name="tello",
        morphology="drone",
        categories=["aerial", "sensing", "generic"],
        instructions=(
            "You control a small indoor quadcopter. Always takeoff before "
            "fly_to. NEVER exceed 2m altitude indoors. On any uncertainty, "
            "land. Battery below 20% triggers automatic land — read battery "
            "with read_battery before fly_to."
        ),
        workspace_bounds=((-3.0, -3.0, 0.0), (3.0, 3.0, 2.0)),
        max_velocity=1.0,
        max_acceleration=2.0,
        rate_limit_per_min=120,
        hitl_primitives=["takeoff", "fly_to"],
    )


def stretch_mobile_arm() -> RobotProfile:
    """Hello Robot Stretch RE3 — mobile base + telescoping arm."""
    return RobotProfile(
        name="stretch",
        morphology="mobile_arm",
        categories=["mobile_base", "dexterous", "sensing", "generic"],
        instructions=(
            "You control a Hello Robot Stretch RE3. It has a wheeled base AND "
            "a telescoping arm with a gripper. drive / goto move the base; "
            "set_joint moves the arm; set_gripper opens / closes the gripper. "
            "Stop the base before extending the arm. The arm cannot reach "
            "behind the base."
        ),
        workspace_bounds=((-5.0, -5.0, 0.0), (5.0, 5.0, 1.4)),
        max_force_n=10.0,
        max_velocity=0.5,
        max_acceleration=1.0,
        rate_limit_per_min=60,
        hitl_primitives=["set_gripper"],
    )


def humanoid_demo() -> RobotProfile:
    """Generic humanoid — wave / look / point / nod, no locomotion."""
    return RobotProfile(
        name="humanoid_demo",
        morphology="humanoid",
        categories=["humanoid", "sensing", "generic"],
        instructions=(
            "You control a stationary humanoid robot used for human-robot "
            "interaction demos. The robot has a head with a camera and two "
            "arms. Use look_at to orient toward people, wave to greet, "
            "point_at to indicate objects, and nod to acknowledge. The robot "
            "DOES NOT walk."
        ),
        workspace_bounds=((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0)),
        max_velocity=1.0,
        max_acceleration=2.0,
        rate_limit_per_min=60,
    )
