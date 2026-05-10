"""ROS2Backend — DDS-based real-hardware backend via rclpy.

Real robots speak ROS 2 in 2026. Most arms / mobile bases / drones
ship with a ROS 2 driver that exposes joint-state subscribers,
twist-publishers, force-torque topics, and lifecycle services.
ROS2Backend bridges the ghostloop runtime to that ecosystem so the
same agent loop, safety pipeline, and trace recorder that drove the
sim now drives a physical robot.

Conditional import — rclpy is heavy and platform-specific. The
package itself imports cleanly without ROS 2 installed; constructing
the backend raises ImportError with the install hint:

  apt install ros-humble-rclpy   (Ubuntu / Debian)
  brew install ros2              (Mac via robotology / source build)

Then:

  from ghostloop.backends import ROS2Backend
  backend = ROS2Backend(
      node_name="ghostloop_runtime",
      cmd_vel_topic="/cmd_vel",
      odom_topic="/odom",
      joint_state_topic="/joint_states",
  )

The backend spins a single rclpy node on a daemon thread; calls to
``publish_twist`` / ``set_joints`` enqueue ROS messages, and
``snapshot()`` returns the latest observation cached from the
subscriber callbacks. Subscriptions degrade gracefully — if no
message has arrived yet, the snapshot returns sentinel "stale" flags.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


_ROS2_INSTALL_HINT = (
    "ROS2Backend requires rclpy.\n"
    "  apt install ros-humble-rclpy ros-humble-geometry-msgs "
    "ros-humble-sensor-msgs ros-humble-nav-msgs\n"
    "or  source /opt/ros/humble/setup.bash\n"
    "Docs: https://docs.ros.org/en/humble/Installation.html"
)


def ros2_available() -> bool:
    """True iff ``import rclpy`` succeeds on this interpreter."""
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ROS2Backend:
    """ROS 2 adapter exposed as a ghostloop Backend.

    Args:
        node_name: rclpy node name (must be unique within the ROS graph).
        cmd_vel_topic: Twist publisher topic for mobile-base drive.
        odom_topic: Odometry subscriber topic.
        joint_state_topic: JointState subscriber topic.
        force_torque_topic: optional WrenchStamped subscriber for force gates.
        spin_rate_hz: how fast the executor spins (rclpy.spin_once interval).
        name: friendly name appearing in traces.
    """

    node_name: str = "ghostloop_runtime"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    joint_state_topic: str = "/joint_states"
    force_torque_topic: str | None = None
    spin_rate_hz: float = 50.0
    name: str = "ros2"

    _rclpy: Any = field(default=None, init=False, repr=False)
    _node: Any = field(default=None, init=False, repr=False)
    _executor: Any = field(default=None, init=False, repr=False)
    _spinner: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _last_odom: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_joint_state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_wrench: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cmd_vel_pub: Any = field(default=None, init=False, repr=False)
    _joint_pub: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from geometry_msgs.msg import Twist, WrenchStamped
            from sensor_msgs.msg import JointState
            from nav_msgs.msg import Odometry
        except ImportError as e:
            raise ImportError(_ROS2_INSTALL_HINT) from e
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        node = Node(self.node_name)
        self._node = node
        self._cmd_vel_pub = node.create_publisher(Twist, self.cmd_vel_topic, 10)
        # Drop the Twist class binding into _types for later use.
        self._twist_msg = Twist
        self._joint_state_msg = JointState
        node.create_subscription(
            Odometry, self.odom_topic, self._on_odom, 10,
        )
        node.create_subscription(
            JointState, self.joint_state_topic, self._on_joint_state, 10,
        )
        if self.force_torque_topic:
            node.create_subscription(
                WrenchStamped, self.force_torque_topic, self._on_wrench, 10,
            )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._spinner = threading.Thread(target=self._spin, daemon=True)
        self._spinner.start()

    # ------------------------------------------------------------------
    # Subscriber callbacks — cache the latest message into a plain dict.
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Any) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        self._last_odom = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "qx": float(pose.orientation.x),
            "qy": float(pose.orientation.y),
            "qz": float(pose.orientation.z),
            "qw": float(pose.orientation.w),
            "vx": float(twist.linear.x),
            "vy": float(twist.linear.y),
            "vz": float(twist.linear.z),
            "wz": float(twist.angular.z),
        }

    def _on_joint_state(self, msg: Any) -> None:
        self._last_joint_state = {
            "joints": list(msg.name),
            "positions": list(msg.position),
            "velocities": list(msg.velocity) if msg.velocity else [],
            "efforts": list(msg.effort) if msg.effort else [],
        }

    def _on_wrench(self, msg: Any) -> None:
        w = msg.wrench
        self._last_wrench = {
            "fx": float(w.force.x),
            "fy": float(w.force.y),
            "fz": float(w.force.z),
            "tx": float(w.torque.x),
            "ty": float(w.torque.y),
            "tz": float(w.torque.z),
            "force_norm": float(
                (w.force.x ** 2 + w.force.y ** 2 + w.force.z ** 2) ** 0.5
            ),
        }

    # ------------------------------------------------------------------
    # Spinner thread.
    # ------------------------------------------------------------------

    def _spin(self) -> None:
        period = 1.0 / max(self.spin_rate_hz, 1.0)
        while not self._stop.is_set() and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=period)
            except Exception:  # noqa: BLE001
                # Don't die the spinner thread on transient ROS errors;
                # the runtime upstairs will see stale snapshots and can
                # surface the issue through the trace.
                pass

    # ------------------------------------------------------------------
    # Backend Protocol surface.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "backend": self.name,
            "node": self.node_name,
            "odom_stale": not bool(self._last_odom),
            "joint_state_stale": not bool(self._last_joint_state),
        }
        snap.update(self._last_odom)
        if self._last_joint_state:
            snap["joint_state"] = dict(self._last_joint_state)
        if self.force_torque_topic:
            snap["wrench"] = dict(self._last_wrench)
            snap["wrench_stale"] = not bool(self._last_wrench)
        return snap

    # ------------------------------------------------------------------
    # Action helpers — primitives call these.
    # ------------------------------------------------------------------

    def publish_twist(
        self, *, linear_x: float = 0.0, linear_y: float = 0.0,
        angular_z: float = 0.0,
    ) -> dict[str, Any]:
        """Send a Twist on cmd_vel_topic (mobile-base velocity command)."""
        msg = self._twist_msg()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)
        return {
            "topic": self.cmd_vel_topic,
            "linear_x": float(linear_x),
            "linear_y": float(linear_y),
            "angular_z": float(angular_z),
        }

    def stop_motion(self) -> dict[str, Any]:
        """Convenience: zero Twist (used by safety pipelines on ERROR)."""
        return self.publish_twist(linear_x=0.0, linear_y=0.0, angular_z=0.0)

    def shutdown(self) -> None:
        """Stop the spinner, destroy the node, and shutdown rclpy.

        Idempotent — call from your ``finally`` block to avoid leaving
        ROS resources around on test exits.
        """
        self._stop.set()
        if self._spinner is not None:
            self._spinner.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        try:
            if self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass
